cmake_minimum_required(VERSION 3.18...3.26)
cmake_policy(VERSION 3.18)

cmake_policy(SET CMP0069 NEW)
cmake_policy(SET CMP0072 NEW)
cmake_policy(SET CMP0091 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(MyLuisaComputeApp LANGUAGES C CXX VERSION 0.1)

# Compiler settings
set(CMAKE_C_COMPILER "/usr/bin/gcc-11" CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-11" CACHE STRING "C++ compiler" FORCE)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# RPATH setup (important for dynamic libs on Unix/macOS)
set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_SKIP_BUILD_RPATH OFF)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../bin;$ORIGIN/../lib")

if (NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/dist")
endif ()

# Sort packages and prefer system libraries
set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
set(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)

# Log
message(STATUS "Source dir: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

# Add LuisaCompute Coroutine (assuming it's a standalone CMake project)
add_subdirectory(ext/LuisaCompute-coroutine)

# Add your executable
add_executable(my_app main.cpp)

# Include directories
target_include_directories(my_app
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/ext/LuisaCompute-coroutine/include
)

# Link against LuisaCompute (update target names if needed)
target_link_libraries(my_app
    PRIVATE
        luisa::compute
        luisa-compute-core
        luisa-compute-include
        luisa-compute-api
)

# Install target
install(TARGETS my_app RUNTIME DESTINATION bin)

