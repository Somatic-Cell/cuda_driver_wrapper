#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/device.hpp>
#include <cuda_driver_wrapper/error.hpp>
#include <cuda_driver_wrapper/event.hpp>
#include <cuda_driver_wrapper/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
namespace cdw = cuda_driver_wrapper;

void require(const bool condition, const std::string_view message)
{
    if(!condition) throw std::runtime_error(std::string(message));
}

template<class Exception, class Function>
void require_throws(Function&& function, const std::string_view message)
{
    try { std::forward<Function>(function)(); }
    catch(const Exception&) { return; }
    throw std::runtime_error(std::string(message));
}

template<class Function>
void require_cuda_error(Function&& function, const CUresult expected)
{
    try { std::forward<Function>(function)(); }
    catch(const cdw::CudaError& error)
    {
        require(error.result() == expected, "Unexpected CUDA error code");
        return;
    }
    throw std::runtime_error("Expected CudaError was not thrown");
}

// テスト内だけの所有型。M0.5 の DeviceBuffer を先取りして公開しない。
// この型の構築・破棄では、呼び出し側が同じ context を current に保つ。
class TestAllocation final
{
public:
    explicit TestAllocation(const std::size_t bytes)
    {
        cdw::check_cuda(cuMemAlloc(&pointer_, bytes), "cuMemAlloc (test)");
    }
    ~TestAllocation() noexcept
    {
        static_cast<void>(cdw::report_cuda_cleanup_result(
            cuMemFree(pointer_), "cuMemFree (test)"
        ));
    }
    TestAllocation(const TestAllocation&) = delete;
    TestAllocation& operator=(const TestAllocation&) = delete;
    [[nodiscard]] CUdeviceptr get() const noexcept { return pointer_; }
private:
    CUdeviceptr pointer_ = 0;
};

// 別 context との共存を、GPU 一台でも検証するためのテスト用所有型。
class ExternalContext final
{
public:
    explicit ExternalContext(const CUdevice device)
    {
#if CUDA_VERSION >= 13000
        CUctxCreateParams parameters{};
        cdw::check_cuda(cuCtxCreate(&context_, &parameters, 0, device), "cuCtxCreate (test)");
#else
        cdw::check_cuda(cuCtxCreate(&context_, 0, device), "cuCtxCreate (test)");
#endif
        try
        {
            CUcontext popped = nullptr;
            cdw::check_cuda(cuCtxPopCurrent(&popped), "cuCtxPopCurrent (test setup)");
            require(popped == context_, "Unexpected context after cuCtxCreate");
        }
        catch(...)
        {
            static_cast<void>(cdw::report_cuda_cleanup_result(
                cuCtxDestroy(context_), "cuCtxDestroy (test rollback)"
            ));
            throw;
        }
    }
    ~ExternalContext() noexcept
    {
        static_cast<void>(cdw::report_cuda_cleanup_result(
            cuCtxDestroy(context_), "cuCtxDestroy (test)"
        ));
    }
    ExternalContext(const ExternalContext&) = delete;
    ExternalContext& operator=(const ExternalContext&) = delete;
    [[nodiscard]] CUcontext handle() const noexcept { return context_; }
private:
    CUcontext context_ = nullptr;
};

void test_lifetime_and_move(const CUcontext context)
{
    static_assert(!std::is_copy_constructible_v<cdw::Stream>);
    static_assert(!std::is_copy_assignable_v<cdw::Stream>);
    static_assert(std::is_nothrow_move_constructible_v<cdw::Stream>);
    static_assert(!std::is_nothrow_move_assignable_v<cdw::Stream>);
    static_assert(std::is_nothrow_destructible_v<cdw::Stream>);
    static_assert(!std::is_copy_constructible_v<cdw::Event>);
    static_assert(!std::is_copy_assignable_v<cdw::Event>);
    static_assert(std::is_nothrow_move_constructible_v<cdw::Event>);
    static_assert(!std::is_nothrow_move_assignable_v<cdw::Event>);
    static_assert(std::is_nothrow_destructible_v<cdw::Event>);

    const CUcontext before = cdw::get_current_context();
    {
        cdw::Stream source(context);
        require(cdw::get_current_context() == before, "Stream constructor changed current");
        unsigned int flags = 0;
        {
            cdw::ScopedCurrentContext current(context);
            cdw::check_cuda(cuStreamGetFlags(source.handle(), &flags), "cuStreamGetFlags");
        }
        require(flags == CU_STREAM_NON_BLOCKING, "Incorrect default stream flags");
        const CUstream handle = source.handle();
        cdw::Stream moved(std::move(source));
        require(!source.is_valid() && source.context() == nullptr, "Stream move source not empty");
        require(moved.handle() == handle && moved.context() == context, "Stream move lost ownership");
        source.close();
        require_throws<std::logic_error>([&] { static_cast<void>(cdw::is_stream_ready(source)); },
                                        "Empty stream must not become the default stream");
        cdw::Stream destination(context, CU_STREAM_DEFAULT);
        destination = std::move(moved);
        require(!moved.is_valid() && destination.handle() == handle, "Stream move assignment failed");
        cdw::synchronize_stream(destination);
        require(cdw::is_stream_ready(destination), "Synchronized stream must be ready");
        destination.close();
        destination.close();
        require(!destination.is_valid(), "Stream close failed");
        require_throws<std::logic_error>([&] { cdw::synchronize_stream(destination); },
                                        "Closed stream accepted");

        cdw::Event event(context);
        require(cdw::is_event_ready(event), "Unrecorded event must represent empty work");
        cdw::synchronize_event(event);
        const CUevent event_handle = event.handle();
        cdw::Event moved_event(std::move(event));
        require(!event.is_valid() && event.context() == nullptr, "Event move source not empty");
        cdw::Event destination_event(context);
        destination_event = std::move(moved_event);
        require(destination_event.handle() == event_handle && !moved_event.is_valid(),
                "Event move assignment failed");
        destination_event.close();
        destination_event.close();
        require_throws<std::logic_error>([&] { static_cast<void>(cdw::is_event_ready(destination_event)); },
                                        "Closed event accepted");
        require(cdw::get_current_context() == before, "Lifetime test changed current");
        // ここで追加資源を destructor 経由でも解放する。
        cdw::Stream automatic_stream(context);
        cdw::Event automatic_event(context);
    }
    require(cdw::get_current_context() == before, "Destructor changed current");
}

void test_pipeline_and_timing(const CUcontext context)
{
    cdw::ScopedCurrentContext current(context);
    constexpr std::size_t bytes = 256 * 1024;
    constexpr unsigned char pattern = 0x5a;
    TestAllocation input(bytes);
    TestAllocation output(bytes);
    cdw::Stream producer(context);
    cdw::Stream consumer(context);
    cdw::Event ready(context);
    cdw::Event finished(context);
    cdw::Event start(context, CU_EVENT_DEFAULT);
    cdw::Event end(context, CU_EVENT_DEFAULT);

    // 未記録のイベントに対する待機は、将来の record を待たない。
    cdw::stream_wait_event(consumer, ready);
    cdw::synchronize_stream(consumer);

    // 未記録のイベントで時間計測をしてはならない。
    require_cuda_error([&] { static_cast<void>(cdw::elapsed_time_ms(start, end)); },
                       CUDA_ERROR_INVALID_HANDLE);

    cdw::record_event(start, producer);
    cdw::check_cuda(cuMemsetD8Async(input.get(), pattern, bytes, producer.handle()),
                    "cuMemsetD8Async (producer)");
    cdw::record_event(ready, producer);
    cdw::record_event(end, producer);

    // 小さな処理は既に完了している可能性もあるので、false を必須にはしない。
    static_cast<void>(cdw::is_stream_ready(producer));
    static_cast<void>(cdw::is_event_ready(ready));

    cdw::stream_wait_event(consumer, ready);
    cdw::check_cuda(cuMemcpyDtoDAsync(output.get(), input.get(), bytes, consumer.handle()),
                    "cuMemcpyDtoDAsync (consumer)");
    cdw::record_event(finished, consumer);
    cdw::synchronize_event(finished);
    cdw::synchronize_event(end);
    cdw::synchronize_stream(producer);
    cdw::synchronize_stream(consumer);

    require(cdw::is_event_ready(ready) && cdw::is_event_ready(finished), "Completed event not ready");
    require(cdw::is_stream_ready(producer) && cdw::is_stream_ready(consumer), "Completed stream not ready");

    std::vector<unsigned char> host(bytes);
    cdw::check_cuda(cuMemcpyDtoH(host.data(), output.get(), bytes), "cuMemcpyDtoH (verification)");
    require(std::all_of(host.begin(), host.end(), [](unsigned char value) { return value == pattern; }),
            "Producer/consumer result mismatch");

    const float milliseconds = cdw::elapsed_time_ms(start, end);
    require(std::isfinite(milliseconds) && milliseconds >= 0.0f, "Invalid event elapsed time");
    std::cout << "Event interval for test memset: " << milliseconds << " ms (not a benchmark)\n";

    // 同期専用のイベントで計測しようとすると、Driver のエラーを受け取る。
    require_cuda_error([&] { static_cast<void>(cdw::elapsed_time_ms(ready, finished)); },
                       CUDA_ERROR_INVALID_HANDLE);

    // 再記録と、外部からの借用ハンドルによる記録も同じ状態へ作用する。
    cdw::check_cuda(cuEventRecord(ready.handle(), producer.handle()), "cuEventRecord (borrowed handle)");
    cdw::synchronize_event(ready);
    cdw::record_event(ready, producer);
    cdw::stream_wait_event(consumer, ready);
    cdw::synchronize_stream(consumer);
    cdw::synchronize_stream(producer);
    require(cdw::get_current_context() == context, "Pipeline test changed current");
}

void test_other_context(const CUdevice device, const CUcontext primary)
{
    ExternalContext external(device);
    cdw::ScopedCurrentContext current(external.handle());
    {
        cdw::Stream source(primary);
        cdw::Event ready(primary);
        cdw::Stream target(external.handle());
        cdw::Event finished(external.handle());
        require(cdw::get_current_context() == external.handle(), "Constructors did not restore current");

        require_throws<std::invalid_argument>([&] { cdw::record_event(ready, target); },
                                              "Cross-context record must be rejected");
        require_throws<std::invalid_argument>([&] { static_cast<void>(cdw::elapsed_time_ms(ready, finished)); },
                                              "Cross-context timing must be rejected by this wrapper");
        cdw::record_event(ready, source);
        cdw::stream_wait_event(target, ready); // 別 context の event の Wait は有効。
        cdw::record_event(finished, target);
        cdw::synchronize_event(finished);
        cdw::synchronize_stream(source);
        cdw::synchronize_stream(target);
        require(cdw::get_current_context() == external.handle(), "Operations changed current");
    }
    require(cdw::get_current_context() == external.handle(), "Destructors changed other current context");
}

void test_invalid_creation(const CUcontext context)
{
    const CUcontext before = cdw::get_current_context();
    require_throws<std::invalid_argument>([] { cdw::Stream stream(nullptr); }, "Null stream context accepted");
    require_throws<std::invalid_argument>([] { cdw::Event event(nullptr); }, "Null event context accepted");
    require_cuda_error([&] { cdw::Stream stream(context, ~0u); }, CUDA_ERROR_INVALID_VALUE);
    require_cuda_error([&] { cdw::Event event(context, ~0u); }, CUDA_ERROR_INVALID_VALUE);
    require(cdw::get_current_context() == before, "Failed creation changed current");
}

} // namespace

int main()
{
    try
    {
        int count = 0;
        try
        {
            cdw::initialize_driver();
            count = cdw::get_device_count();
        }
        catch(const cdw::CudaError& error)
        {
            if(error.result() != CUDA_ERROR_NO_DEVICE) throw;
            std::cout << "Skipped: no CUDA device.\n";
            return 77;
        }
        if(count == 0)
        {
            std::cout << "Skipped: no CUDA device.\n";
            return 77;
        }

        const CUdevice device = cdw::get_device(0);
        const CUcontext before = cdw::get_current_context();
        {
            cdw::RetainPrimaryContext primary(device);
            test_lifetime_and_move(primary.handle());
            test_pipeline_and_timing(primary.handle());
            test_other_context(device, primary.handle());
            test_invalid_creation(primary.handle());
        }
        require(cdw::get_current_context() == before, "Test changed initial current context");
        std::cout << "All stream/event tests passed.\n";
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
