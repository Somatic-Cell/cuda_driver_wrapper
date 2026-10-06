set(TEST_NAME "cuda_driver_wrapper_error_test")
add_executable(${TEST_NAME} error_test.cpp)

target_link_libraries(${TEST_NAME}
    PRIVATE
    cuda_driver_wrapper::cuda_driver_wrapper
)

set_target_properties(${TEST_NAME}
    PROPERTIES CXX_EXTENSIONS OFF
)

if(MSVC)
    target_compile_options(${TEST_NAME}
        PRIVATE /W4 /utf-8 /permissive- /Zc:__cplusplus
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${TEST_NAME}
        PRIVATE -Wall -Wextra -Wpedantic
    )
endif()

add_test(
    NAME cuda_driver_wrapper.error
    COMMAND cuda_driver_wrapper_error_test
)
set_tests_properties(cuda_driver_wrapper.error 
    PROPERTIES 
    LABELS "host;cuda_driver"
)