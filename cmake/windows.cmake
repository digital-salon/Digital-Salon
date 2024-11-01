option(BUILD_WITH_OPENVDB "Build with OpenVDB" OFF)

# COMPILE FLAGS
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

# SOURCES
file(GLOB_RECURSE SOURCES_SDF RELATIVE "${PROJECT_SOURCE_DIR}" "src/*.*" "extern/hairsim/src/*.*" "extern/include/imgui/*.*")

# Eigen
set(Eigen_DIR "C:/local/eigen-3.4.0/build")
set(nlohmann_json_DIR "C:/local/json-develop/build")
find_package (Eigen3 REQUIRED NO_MODULE)
find_package (nlohmann_json REQUIRED)
link_directories(${Eigen_INCLUDE_DIRS})

add_executable(${EXEC_NAME} ${SOURCES_SDF})

# HEADERS
target_include_directories( ${EXEC_NAME} PUBLIC
	"${PROJECT_SOURCE_DIR}/extern/include"
	"${PROJECT_SOURCE_DIR}/src"
	"${PROJECT_SOURCE_DIR}/extern/hairsim/src"
	"${PROJECT_SOURCE_DIR}/extern/hairsim/include/cuda")

# BIN DIR
SET (HAIR_BIN_DIR "${CMAKE_CURRENT_LIST_DIR}/bin")

# openvdb
if(${BUILD_WITH_OPENVDB})

	# Boost
	##set(Boost_DEBUG ON)
	set(BOOST_ROOT "C:\\local/boost_1_81_0")
	set(Boost_USE_STATIC_LIBS ON)
	find_package(Boost 1.81.0 REQUIRED COMPONENTS system filesystem thread date_time chrono)
	if(Boost_FOUND)
		target_include_directories( ${EXEC_NAME} PUBLIC ${Boost_INCLUDE_DIRS})
		LIST (APPEND EXTRA_LIBS Boost::filesystem Boost::system Boost::thread Boost::date_time Boost::chrono)
	endif()
	add_compile_definitions(WITH_OPENVDB)
	
	SET(OPENVDB_DIR "C:/local/vcpkg/packages")
	target_include_directories(${EXEC_NAME} PUBLIC "${OPENVDB_DIR}/openvdb_x64-windows/include" "${OPENVDB_DIR}/openexr_x64-windows/include" "${OPENVDB_DIR}/tbb_x64-windows/include")
	LIST(APPEND EXTRA_LIBS "${OPENVDB_DIR}/openvdb_x64-windows/lib/openvdb.lib" "${OPENVDB_DIR}/openexr_x64-windows/lib/Half-2_3_s.lib" "${OPENVDB_DIR}/tbb_x64-windows/lib/tbb.lib")
endif()


IF (MSVC)
	# MP compilation
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP ")

	# CUDA
	set_target_properties(${EXEC_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	set_target_properties(${EXEC_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

	# CREATE BINARY FOLDER
	add_custom_command(TARGET ${EXEC_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ${HAIR_BIN_DIR})

	# Binary location
	SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${HAIR_BIN_DIR}")
	SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${HAIR_BIN_DIR}")
	
	# OpenMP
	find_package(OpenMP)
	if (OPENMP_FOUND)
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	endif()

	SET(LIB_DIR "extern/lib")
	
	# GLFW
	get_filename_component(GLFW_LIBRARY "${LIB_DIR}/glfw3.lib" ABSOLUTE)

	# ASSIMP
	get_filename_component(ASSIMP_LIBRARY "${LIB_DIR}/assimp-vc143-mt.lib" ABSOLUTE)

	# GLEW
	get_filename_component(GLEW_LIBRARY "${LIB_DIR}/glew32.lib" ABSOLUTE)
	
	# GLUT
	get_filename_component(GLUT_LIBRARY "${LIB_DIR}/freeglut.lib" ABSOLUTE)
	
	# Free Image
	get_filename_component(FIMAGE_LIBRARY "${LIB_DIR}/FreeImage.lib" ABSOLUTE)
	
	LIST(APPEND EXTRA_LIBS ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${GLUT_LIBRARY} ${FIMAGE_LIBRARY} ${ASSIMP_LIBRARY})
	
	# MAKE FILTERS IN VISUAL STUDIO
	foreach(source IN LISTS SOURCES_SDF)
		get_filename_component(source_path "${source}" PATH)
		string(REPLACE "/" "\\" source_path_msvc "${source_path}")
		source_group("${source_path_msvc}" FILES "${source}")
	endforeach()
ENDIF ()

# OpenGL
find_package(OpenGL REQUIRED)
FIND_LIBRARY(OPENGL_LIBRARY OpenGL)
LIST(APPEND EXTRA_LIBS ${OPENGL_LIBRARY})

# Eigen
TARGET_LINK_LIBRARIES(${EXEC_NAME} PUBLIC Eigen3::Eigen)
TARGET_LINK_LIBRARIES(${EXEC_NAME} PUBLIC nlohmann_json::nlohmann_json)
TARGET_LINK_LIBRARIES(${EXEC_NAME} PUBLIC ${EXTRA_LIBS} "-static")