#include <cuda_driver_wrapper/device.hpp>
#include <cuda_driver_wrapper/error.hpp>

#include <exception>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
namespace cdw = cuda_driver_wrapper;
constexpr int skip_return_code = 77;

void require(const bool condition, const std::string_view message)
{
    if(!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

CUcontext current_context()
{
    CUcontext context = nullptr;
    cdw::check_cuda(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
    return context;
}

void test_uninitialized()
{
    // 必ず独立したプロセスで実行する．他のテストで先に初期化しない
    try
    {
        static_cast<void>(cdw::get_device_count());
    }
    catch(const cdw::CudaError& error)
    {
        require(error.result() == CUDA_ERROR_NOT_INITIALIZED,
                "an uninitialized query must preserve CUDA_ERROR_NOT_INITIALIZED");
        return;
    }
    throw std::runtime_error("get_device_count must not initialize the driver implicitly");
}

void test_invalid_ordinal(const int ordinal)
{
    // エラーコードを独自に決めず，生の Driver API と一致することを確認する
    CUdevice unused{};
    const CUresult expected = cuDeviceGet(&unused, ordinal);
    require(expected != CUDA_SUCCESS, "the test ordinal must be invalid");

    const auto location = std::source_location::current();
    try
    {
        static_cast<void>(cdw::get_device(ordinal, location));
    }
    catch(const cdw::CudaError& error)
    {
        require(error.result() == expected, "get_device must preserve the Driver error");
        
        const std::string message = error.what();
        const std::string expected_location =
                            std::string(location.file_name()) + ": " + std::to_string(location.line());
                            require(message.find(expected_location) != std::string::npos,
                            "get_device must forward the supplied source location");
        
        return;
    }
    throw std::runtime_error("get_device must reject an invalid ordinal");
}

void check_attribute(
    const CUdevice device,
    const CUdevice_attribute attribute,
    const int recorded_value
)
{
    int raw_value = 0;
    cdw::check_cuda(cuDeviceGetAttribute(&raw_value, attribute, device),
                    "cuDeviceGetAttribute (reference)");
    require(recorded_value == raw_value, "DeviceInfo attribute differs from Driver API");
    require(cdw::get_device_attribute(device, attribute) == raw_value,
            "get_device_attribute differs from Driver API");
}

int test_device_queries()
{
    int count = 0;
    try
    {
        cdw::initialize_driver();
        count = cdw::get_device_count();
    }
    catch(const cdw::CudaError& error)
    {
        // デバイスなしだけをスキップ．他の Driver 障害は外側へ伝播する
        if(error.result() == CUDA_ERROR_NO_DEVICE)
        {
            std::cout << "SKIP: no CUDA device is available.\n";
            return skip_return_code;
        }
        throw;
    }

    require(count >= 0, "device count must not be negative");
    if(count == 0)
    {
        std::cout << "SKIP: the Driver reported zero CUDA devices.\n";
        return skip_return_code;
    }

    // この実行ファイルはコンテキストを作成しない
    const CUcontext before = current_context();
    require(before == nullptr, "driver initialization must not make a context current");

    // 繰り返しの呼び出しでも，ラッパーは自身の初期化状態を持たない
    cdw::initialize_driver();
    require(current_context() == before, "initialization changed the current context");

    int raw_count = 0;
    cdw::check_cuda(cuDeviceGetCount(&raw_count), "cuDeviceGetCount (reference)");
    require(count == raw_count, "get_device_count differs from Driver API");

    for(int ordinal = 0; ordinal < count; ++ordinal)
    {
        const CUdevice device = cdw::get_device(ordinal);
        CUdevice raw_device{};
        cdw::check_cuda(cuDeviceGet(&raw_device, ordinal), "cuDeviceGet (reference)");
        require(device == raw_device, "get_device differs from Driver API");

        const auto info = cdw::get_device_info(device);
        require(!info.name.empty(), "device name must not be empty");
        require(info.total_memory_bytes > 0, "total memory must be positive");
        require(info.compute_capability_major > 0, "compute capability major must be positive");
        require(info.compute_capability_minor >= 0, "compute capability minor must not be negative");
        require(info.multiprocessor_count > 0, "multiprocessor count must be positive");
        require(info.max_threads_per_block > 0, "max threads per block must be positive");

        std::size_t raw_memory = 0;
        cdw::check_cuda(cuDeviceTotalMem(&raw_memory, device), "cuDeviceTotalMem (reference)");
        require(info.total_memory_bytes == raw_memory, "total memory differs from Driver API");

        check_attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                        info.compute_capability_major);
        check_attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                        info.compute_capability_minor);
        check_attribute(device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                        info.multiprocessor_count);
        check_attribute(device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                        info.max_threads_per_block);

        std::cout << "Device " << ordinal << ": " << info.name << '\n'
                  << "  Total memory (bytes): " << info.total_memory_bytes << '\n'
                  << "  Compute capability: " << info.compute_capability_major
                  << '.' << info.compute_capability_minor << '\n'
                  << "  Multiprocessors: " << info.multiprocessor_count << '\n'
                  << "  Max threads per block: " << info.max_threads_per_block << '\n';
    }

    test_invalid_ordinal(-1);
    test_invalid_ordinal(count);
    require(current_context() == before, "device queries changed the current context");
    std::cout << "All device tests passed.\n";
    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        if(argc == 2 && std::string_view(argv[1]) == "--uninitialized")
        {
            test_uninitialized();
            std::cout << "Uninitialized-driver test passed.\n";
            return 0;
        }
        if(argc != 1)
        {
            throw std::runtime_error("usage: device_test [--uninitialized]");
        }
        return test_device_queries();
    }
    catch(const std::exception& error)
    {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}