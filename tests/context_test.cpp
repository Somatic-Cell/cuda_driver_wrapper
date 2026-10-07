#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/device.hpp>
#include <cuda_driver_wrapper/error.hpp>

#include <cstdint>
#include <exception>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>

namespace
{
namespace cdw = cuda_driver_wrapper;

void require(const bool condition, const std::string_view message)
{
    if(!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

void require_current(const CUcontext expected, const std::string_view message)
{
    // ラッパーだけでなく、生の Driver API の結果も確認する。
    CUcontext actual = nullptr;
    cdw::check_cuda(cuCtxGetCurrent(&actual), "cuCtxGetCurrent (test)");
    require(actual == expected, message);
    require(cdw::get_current_context() == actual, "get_current_context disagrees with the Driver API");
}

struct PrimaryState
{
    unsigned int flags = 0;
    int active = 0;
};

PrimaryState primary_state(const CUdevice device)
{
    PrimaryState state{};
    cdw::check_cuda(
        cuDevicePrimaryCtxGetState(device, &state.flags, &state.active),
        "cuDevicePrimaryCtxGetState (test)"
    );
    return state;
}

// 外部コードの所有する通常の context を、同じ GPU 上に用意するテスト用の型。
// 公開ライブラリの API ではない。cuCtxCreate はこの context を current にする。
class ExternalContext final
{
public:
    explicit ExternalContext(const CUdevice device)
    {
#if CUDA_VERSION >= 13000
        CUctxCreateParams parameters{};
        cdw::check_cuda(
            cuCtxCreate(&context_, &parameters, CU_CTX_SCHED_AUTO, device),
            "cuCtxCreate (external test context)"
        );
#else
        cdw::check_cuda(
            cuCtxCreate(&context_, CU_CTX_SCHED_AUTO, device),
            "cuCtxCreate (external test context)"
        );
#endif
    }

    ~ExternalContext() noexcept
    {
        static_cast<void>(cdw::report_cuda_cleanup_result(
            cuCtxDestroy(context_), "cuCtxDestroy (external test context)"
        ));
    }

    ExternalContext(const ExternalContext&) = delete;
    ExternalContext& operator=(const ExternalContext&) = delete;
    ExternalContext(ExternalContext&&) = delete;
    ExternalContext& operator=(ExternalContext&&) = delete;

    CUcontext handle() const noexcept { return context_; }

private:
    CUcontext context_ = nullptr;
};

void test_primary_lifetime(const CUdevice device)
{
    require_current(nullptr, "test must start without a current context");
    const PrimaryState before = primary_state(device);
    require(before.active == 0, "test requires an isolated process with an inactive primary context");

    {
        cdw::PrimaryContext first(device);
        require(first.handle() != nullptr, "retain must return a non-null context");
        require(first.device() == device, "device() must preserve the supplied device");
        require_current(nullptr, "retaining a primary context must not change current");
        const PrimaryState retained = primary_state(device);
        require(retained.active != 0, "retained primary context must be active");

        {
            cdw::PrimaryContext second(device);
            require(second.handle() == first.handle(), "same-device retains must identify the same primary context");
            require(primary_state(device).flags == retained.flags, "another retain must not change context flags");
        }

        require(primary_state(device).active != 0, "releasing one reference must preserve another owner's reference");
        {
            cdw::ScopedCurrentContext current(first.handle());
            require_current(first.handle(), "remaining retain must still be usable");
        }
        require_current(nullptr, "scope must pop before the last primary reference is released");
    }

    require(primary_state(device).active == 0, "the last reference must have been released");
    require_current(nullptr, "primary context destruction must not create a current context");
}

void test_same_context_nesting(const CUcontext context)
{
    {
        cdw::ScopedCurrentContext outer(context);
        require_current(context, "outer context must be current");
        {
            cdw::ScopedCurrentContext inner(context);
            require(inner.is_active(), "new scope must be active");
            inner.restore();
            require(!inner.is_active(), "restored scope must be inactive");
            require_current(context, "inner restore must preserve the outer push");
            inner.restore();
            require_current(context, "repeated restore must not pop twice");
        }
        require_current(context, "destruction after restore must not pop again");
    }
    require_current(nullptr, "outer scope must restore the initially empty stack");
}

struct IntentionalFailure {};

void test_exception_restoration(const CUcontext context)
{
    bool caught = false;
    try
    {
        cdw::ScopedCurrentContext current(context);
        require_current(context, "context must be current before throwing");
        throw IntentionalFailure{};
    }
    catch(const IntentionalFailure&)
    {
        caught = true;
    }
    require(caught, "original exception must reach its handler");
    require_current(nullptr, "stack unwinding must restore the previous current context");
}

void test_external_context(const CUdevice device, const CUcontext primary)
{
    {
        ExternalContext external(device);
        require(external.handle() != primary, "test needs two distinct contexts on one GPU");
        require_current(external.handle(), "cuCtxCreate must make the external context current");

        {
            cdw::PrimaryContext another_reference(device);
            require(another_reference.handle() == primary, "additional retain must identify the same primary context");
            require_current(external.handle(), "retain must not overwrite an external current context");
        }
        require_current(external.handle(), "release must not overwrite an external current context");

        {
            cdw::ScopedCurrentContext outer(primary);
            require_current(primary, "primary context must temporarily replace the external context");
            {
                cdw::ScopedCurrentContext inner(external.handle());
                require_current(external.handle(), "a borrowed external context must also be usable");
                const auto info = cdw::get_device_info(device);
                require(!info.name.empty(), "device query should succeed with a current context");
                require_current(external.handle(), "M0.2 queries must preserve an existing current context");

                // 異なる context が top の場合は、外側だけを先に復元できない。
                bool rejected = false;
                try { outer.restore(); }
                catch(const std::logic_error&) { rejected = true; }
                require(rejected, "out-of-order restore must be rejected before popping");
                require(outer.is_active(), "rejected restore must preserve ownership of the push");
                require_current(external.handle(), "rejected restore must leave the stack untouched");
            }
            require_current(primary, "inner scope must restore the primary context");
        }
        require_current(external.handle(), "outer scope must restore the external context");

        try
        {
            cdw::ScopedCurrentContext current(primary);
            throw IntentionalFailure{};
            }
        catch(const IntentionalFailure&) {}
        require_current(external.handle(), "exception unwinding must also restore a non-null previous context");
    }
    require_current(nullptr, "external owner must destroy its own context and restore the original stack");
}

void test_thread_local_current(const CUcontext context)
{
    cdw::ScopedCurrentContext main_scope(context);
    std::exception_ptr worker_error;

    std::jthread worker([&]
    {
        try
        {
            require_current(nullptr, "a new host thread must start with no current context");

            bool rejected = false;
            try { main_scope.restore(); }
            catch(const std::logic_error&) { rejected = true; }
            require(rejected, "restoring another thread's scope must be rejected");
            require_current(nullptr, "rejected cross-thread restore must not change the worker's stack");

            {
                cdw::ScopedCurrentContext worker_scope(context);
                require_current(context, "the same context may be made current by a separate scope on another thread");
            }
            require_current(nullptr, "worker scope must restore its own thread's stack");
        }
        catch(...)
        {
            worker_error = std::current_exception();
        }
    });

    // join するまで main_scope へアクセスしない。オブジェクトへの同時アクセスを避ける。
    worker.join();
    if(worker_error) { std::rethrow_exception(worker_error); }

    require(main_scope.is_active(), "worker must not consume the main thread's scope");
    require_current(context, "worker operations must preserve the main thread's current context");
    main_scope.restore();
    require_current(nullptr, "main thread must restore its own original stack");
}

void test_invalid_inputs()
{
    bool caught = false;
    try { cdw::ScopedCurrentContext invalid(nullptr); }
    catch(const std::invalid_argument&) { caught = true; }
    require(caught, "a null borrowed context must be rejected");
    require_current(nullptr, "rejecting nullptr must not push anything");

    caught = false;
    std::uint_least32_t expected_line = 0;
    const char* expected_file = std::source_location::current().file_name();
    try
    {
        expected_line = std::source_location::current().line() + 1;
        cdw::PrimaryContext invalid(static_cast<CUdevice>(-1));
    }
    catch(const cdw::CudaError& error)
    {
        caught = true;
        require(error.result() == CUDA_ERROR_INVALID_DEVICE, "invalid device must preserve the Driver error");
        const std::string expected = std::string(expected_file) + ": " + std::to_string(expected_line);
        const std::string_view diagnostic = error.what();
        require(diagnostic.find(expected) != std::string_view::npos, "constructor error must identify the caller's line");
    }
    require(caught, "invalid device must fail to retain a primary context");
    require_current(nullptr, "a failed constructor must not change current");
}

} // namespace

int main()
{
    static_assert(!std::is_copy_constructible_v<cdw::PrimaryContext>);
    static_assert(!std::is_move_constructible_v<cdw::PrimaryContext>);
    static_assert(std::is_nothrow_destructible_v<cdw::PrimaryContext>);
    static_assert(!std::is_copy_constructible_v<cdw::ScopedCurrentContext>);
    static_assert(!std::is_move_constructible_v<cdw::ScopedCurrentContext>);
    static_assert(std::is_nothrow_destructible_v<cdw::ScopedCurrentContext>);

    try
    {
        int count = 0;
        // 「デバイスなし」は検出段階だけで skip。それ以後の CUDA エラーは失敗。
        try
        {
            cdw::initialize_driver();
            count = cdw::get_device_count();
        }
        catch(const cdw::CudaError& error)
        {
            if(error.result() != CUDA_ERROR_NO_DEVICE) { throw; }
            std::cout << "SKIP: no CUDA device.\n";
            return 77;
        }
        if(count == 0)
        {
            std::cout << "SKIP: no visible CUDA device.\n";
            return 77;
        }
        // これはテストの選択。ライブラリ自体は device 0 に固定しない。
        const CUdevice device = cdw::get_device(0);
        std::cout << "Context tests on: " << cdw::get_device_info(device).name << '\n';

        test_invalid_inputs();
        test_primary_lifetime(device);
        {
            cdw::PrimaryContext primary(device);
            test_same_context_nesting(primary.handle());
            test_exception_restoration(primary.handle());
            test_external_context(device, primary.handle());
            test_thread_local_current(primary.handle());
        }
        require_current(nullptr, "all scopes must be balanced at test completion");
        require(primary_state(device).active == 0, "all primary references must be released at test completion");

        std::cout << "All context tests passed.\n";
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}