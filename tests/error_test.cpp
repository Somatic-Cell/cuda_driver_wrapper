#include <cuda_driver_wrapprer/error.hpp>

#include <cstdint>
#include <exception>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include <string_view>

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

void test_success()
{
    cdw::check_cuda(CUDA_SUCCESS, "success_case");
    require(
        cdw::report_cuda_cleanup_result(CUDA_SUCCESS, "success case"),
        "cleanup reporting must return true for CUDA_SUCCESS"
    );
}

void test_failure_and_call_site()
{
    bool caught = false;
    std::uint_least32_t expected_line = 0;
    const char* expected_file = std::source_location::current().file_name();

    try
    {
        expected_line = std::source_location::current().line() + 1;
        cdw::check_cuda(CUDA_ERROR_INVALID_VALUE, "intentional invalid value");
    }
    catch(const cdw::CudaError& error)
    {
        caught = true;
        require(error.result() == CUDA_ERROR_INVALID_VALUE, "CUresult must be preserved");

        const std::string message = error.what();
        const std::string expected_location = 
            std::string(expected_file) + std::to_string(expected_line);

        require(message.find("intentional invalid value") != std::string::npos, "diagnostic must contain the operation");
        require(message.find("CUDA_ERROR_INVALID_VALUE") != std::string::npos, "diagnostic must contain the CUDA error name");
        require(message.find(expected_location) != std::string::npos, "diagnostic must contain the caller's file and line");
    }

    require(caught, "check_cuda must throw CudaError for failure");
    
}

void test_null_operation()
{
    bool caught = false;
    try
    {
        cdw::check_cuda(CUDA_ERROR_INVALID_VALUE, nullptr);
    }
    catch(const cdw::CudaError& error)
    {
        caught = true;
        require(std::string_view(error.what()).find("(unspecified operation)")
                    != std::string_view::npos,
                "a null operation must use the fallback description");
    }
    require(caught, "a null operation must not suppress a CUDA failure");
}

void test_cleanup_failure()
{
    static_assert(noexcept(
        cdw::report_cuda_cleanup_result(CUDA_SUCCESS, "compile-time check")
    ));

    // このテストだけは，意図的な失敗の診断を stderr に一度出す
    require(
        !cdw::report_cuda_cleanup_result(CUDA_ERROR_INVALID_VALUE, "intentional cleanup failure"),
        "cleanup reporting must return false for failure"
    );
}

} // namespace

int main()
{
    try
    {
        // 実際の GPU 操作を失敗させるのではなく，既知の CUresult を直接渡す
        // エラー名の取得は，本物の CUDA Driver API を利用する
        test_success();
        test_failure_and_call_site();
        test_null_operation();
        test_cleanup_failure();

        std::cout << "All error tests passed.\n";
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}