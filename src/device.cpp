#include <cuda_driver_wrapper/device.hpp>
#include <cuda_driver_wrapper/error.hpp>

#include <limits>
#include <stdexcept>

namespace cuda_driver_wrapper
{
namespace 
{

std::string read_device_name(
    const CUdevice device,
    const std::source_location location
)
{
    // 最初は 256 バイトを確保しておいて，名前が収まらなかったら拡張する
    std::string buffer(256, '\0');
    const std::size_t max_length = static_cast<std::size_t>((std::numeric_limits<int>::max)());

    // 名前が納まるまでバッファを拡張して取得を試みる
    for(;;)
    {
        // デバイス名を取得
        check_cuda(
            cuDeviceGetName(buffer.data(), static_cast<int>(buffer.size()), device),
            "cuDeviceGetName",
            location
        );

        // 長さを取得
        const auto length = buffer.find('\0');
        // 納まっていればバッファを返す
        if(length != std::string::npos && length < buffer.size() -1)
        {
            buffer.resize(length);
            return buffer;
        }

        // cuDeviceGetName の長さの引数は int．拡張と型変換の上限を確認
        if(buffer.size() > max_length / 2)
        {
            throw std::length_error("CUDA device name exceeds the supported buffer size");
        }
        // 拡張
        buffer.assign(buffer.size() * 2, '\0');
    }
}

} // namespace

void initialize_driver(const std::source_location location)
{
    check_cuda(cuInit(0), "cuInit(0)", location);
}

int get_device_count(const std::source_location location)
{
    int count = 0;
    check_cuda(cuDeviceGetCount(&count), "cuDeviceGetCount", location);
    return count;
}

CUdevice get_device(const int ordinal, const std::source_location location)
{
    CUdevice device{};
    check_cuda(cuDeviceGet(&device, ordinal), "cuDeviceGet", location);
    return device;
}

int get_device_attribute(
    const CUdevice device,
    const CUdevice_attribute attribute,
    const std::source_location location
)
{
    int value = 0;
    check_cuda(
        cuDeviceGetAttribute(&value, attribute, device),
        "cuDeviceGetAttribute",
        location
    );
    return value;
}

DeviceInfo get_device_info(
    const CUdevice device,
    const std::source_location location
)
{
    DeviceInfo info{};
    info.name = read_device_name(device, location);

    check_cuda(
        cuDeviceTotalMem(&info.total_memory_bytes, device),
        "cuDeviceTotalMem",
        location
    );

    
    // CUDA compute capability の取得
    info.compute_capability_major = get_device_attribute(
        device,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 
        location
    );

    info.compute_capability_minor = get_device_attribute(
        device,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        location
    );

    // アーキテクチャの情報を取得
    info.multiprocessor_count = get_device_attribute(
        device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        location
    );
    info.max_threads_per_block = get_device_attribute(
        device,
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        location
    );

    return info;
}

} // namespace cuda_driver_wrapper