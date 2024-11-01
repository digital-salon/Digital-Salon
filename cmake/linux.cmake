message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")  # Refers to the root of the project
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")  # Refers to the root of the project
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")  # Refers to the src folder

# COMPILE FLAGS
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

# SOURCES
file(GLOB_RECURSE CPP_SOURCES RELATIVE "${PROJECT_SOURCE_DIR}" "src/*.cpp" "src/Rendering/*.cpp" "extern/hairsim/src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES RELATIVE "${PROJECT_SOURCE_DIR}" "extern/hairsim/src/*.cu")

# BIN DIR
set(HAIR_BIN_DIR "${PROJECT_SOURCE_DIR}/bin")

# Binary location
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${HAIR_BIN_DIR})

# Find CUDA
find_package(CUDA REQUIRED)
#include_directories(${CUDA_INCLUDE_DIRS} "/usr/local/cuda-12.6/targets/x86_64-linux/include")
include_directories(${CUDA_INCLUDE_DIRS} "/usr/local/cuda-11.6/targets/x86_64-linux/include")
list(APPEND EXTRA_LIBS ${CUDA_LIBRARIES})

# Eigen
find_package(Eigen3 REQUIRED NO_MODULE)

# nlohmann_json
find_package(nlohmann_json REQUIRED)

find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(assimp REQUIRED)

# Boost
find_package(Boost COMPONENTS program_options REQUIRED)


# Find the Python 3.x interpreter, libraries and header files
find_package(Python3 COMPONENTS Interpreter Development)

set(IMGUI_SOURCES
    ${CMAKE_SOURCE_DIR}/extern/include/imgui/imgui.cpp
    ${CMAKE_SOURCE_DIR}/extern/include/imgui/imgui_demo.cpp
    ${CMAKE_SOURCE_DIR}/extern/include/imgui/imgui_draw.cpp
    ${CMAKE_SOURCE_DIR}/extern/include/imgui/imgui_impl_glfw.cpp
    ${CMAKE_SOURCE_DIR}/extern/include/imgui/imgui_impl_glut.cpp
    ${CMAKE_SOURCE_DIR}/extern/include/imgui/imgui_impl_opengl3.cpp
    ${CMAKE_SOURCE_DIR}/extern/include/imgui/imgui_stdlib.cpp
    ${CMAKE_SOURCE_DIR}/extern/include/imgui/imgui_tables.cpp
    ${CMAKE_SOURCE_DIR}/extern/include/imgui/imgui_widgets.cpp
)
add_library(imgui ${IMGUI_SOURCES})

add_executable(${EXEC_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

# HEADERS
target_include_directories(${EXEC_NAME} PUBLIC
    "${PROJECT_SOURCE_DIR}/extern/hairsim/include"
    "${PROJECT_SOURCE_DIR}/extern/hairsim/include/cuda"
    "${PROJECT_SOURCE_DIR}/extern/hairsim/src"
    "${PROJECT_SOURCE_DIR}/src"
    ${CUDA_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    "/usr/include/nlohmann"
    "/usr/include/eigen3"
    "${PROJECT_SOURCE_DIR}/extern/"
    "${PROJECT_SOURCE_DIR}/extern/include"
    ${Python3_INCLUDE_DIRS}
)

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# CUDA specific settings
set_target_properties(${EXEC_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Set CUDA architecture
set(CUDA_ARCHITECTURES "50;60;70;75;80")

# Ensure CUDA compilation
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")

target_link_libraries(${EXEC_NAME} PUBLIC Eigen3::Eigen)
target_link_libraries(${EXEC_NAME} PUBLIC ${Python3_LIBRARIES})
target_link_libraries(${EXEC_NAME} PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(${EXEC_NAME} PUBLIC GLEW libGLEW.so glfw libglut.so libGLU.so libGL.so freeimage assimp imgui boost_system boost_filesystem)

