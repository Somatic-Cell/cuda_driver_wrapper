#include <cuda_driver_wrapprer/error.hpp>

#include <cstdio>
#include <sstream>

namespace cuda_driver_wrapper
{
namespace 
{

struct ErrorDetails
{
    const char* name;
    const char* description;
};
    

ErrorDetails describe_error(const CUresult result) noexcept
{
    const char* name = nullptr;
    const char* description = nullptr;

    if(cuGetErrorName(result, &name) != CUDA_SUCCESS || name == nullptr)
    {
        name = "unrecognized CUDA error";
    }
    if(cuGetErrorString(result, &description) != CUDA_SUCCESS || description == nullptr)
    {
        description = "description unavailable";
    }
    return {name, description};
}

} // namespace

// ----------------------------
// CudaError クラスの中身
// ----------------------------

CudaError::CudaError(
    CUresult result, 
    const std::string& message
) : std::runtime_error(message), result_(result)
{
}

CUresult CudaError::result() const noexcept
{
    return result_;
}


// ----------------------------
// エラーをチェックする関数
// ----------------------------

void check_cuda(
    const CUresult result,
    const char* operation,
    const std::source_location location
)
{
    if(result == CUDA_SUCCESS)
    {
        return;
    }

    const ErrorDetails details = describe_error(result);
    const char* operation_name = operation != nullptr ? operation : "(unspecified operation)";

    std::ostringstream mesasge;
    message << operation_name << "returned "
            << details.name << " (" << static_cast<int>(result) << ")\n"
            << "Description: " << details.description << '\n'
            << "Location: " << location.file_name() << ': ' << location.line();
            
    throw CudaError(result, message.str());
}

bool report_cuda_cleanup_result(
    const CUresult result,
    const char* operation,
    conset std::source_location location
) noexcept
{
    if(result == CUDA_SUCCESS)
    {
        return true;
    }

    const ErrorDetails details = describe_error(result);
    const char* operation_name = operation != nullptr ? operation : "(unspecified operation)";

    static_cast<void>(
        std::printf(
            stderr,
            "[cuda_driver_wrapper cleanup] %s returned %s (%d): %s at %s:%lu\n",
            operation_name,
            details.name,
            static_cast<int>(result),
            details.description,
            location.file_name(),
            static_cast<unsigned long>(location.line())
        )
    );

    return false;
}


} // namespace cuda_driver_wrapper