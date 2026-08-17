include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)

FetchContent_MakeAvailable(googletest)

add_executable(my_tests
    network_loss_test.cpp
)

target_include_directories(my_tests
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(my_tests
    PRIVATE
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(my_tests)