# CMake minimum version
cmake_minimum_required(VERSION 3.17)

# Project name
project(DigitalSalon LANGUAGES CUDA CXX)
set(EXEC_NAME DigitalSalon)

# Check the operating system
if(WIN32)
    # Windows specific options
    message(STATUS "Compiling on Windows")
    include(cmake/windows.cmake)
    
elseif(UNIX AND NOT APPLE)
    # Linux specific options
    message(STATUS "Compiling on Linux")
    include(cmake/linux.cmake)

else()
    message(FATAL_ERROR "Unknown or unsupported operating system")
endif()