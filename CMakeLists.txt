cmake_minimum_required(VERSION 3.18...3.27)
project(py_cpp_bridge LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# Common source files
set(COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/common)
set(COMMON_SOURCES ${COMMON_DIR}/cpp_processor.cpp)
set(COMMON_HEADERS ${COMMON_DIR}/cpp_processor.hpp ${COMMON_DIR}/types.hpp)

# Include directories
include_directories(${COMMON_DIR})
include_directories(${Python_NumPy_INCLUDE_DIRS})

# Compiler flags
add_compile_definitions(NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

# =============================================================================
# Cython Extension
# =============================================================================
find_package(Cython QUIET)
if(Cython_FOUND OR EXISTS "${Python_EXECUTABLE}")
    message(STATUS "Building Cython extension")

    # Use scikit-build-core's cython support via Python
    set(CYTHON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/cython_processor)

    # Generate C++ from Cython
    set(CYTHON_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cython_processor.cpp)
    add_custom_command(
        OUTPUT ${CYTHON_OUTPUT}
        COMMAND ${Python_EXECUTABLE} -m cython
            --cplus
            -3
            ${CYTHON_DIR}/cython_processor.pyx
            -o ${CYTHON_OUTPUT}
            -I ${CYTHON_DIR}
            -I ${COMMON_DIR}
        DEPENDS ${CYTHON_DIR}/cython_processor.pyx ${CYTHON_DIR}/cython_processor.pxd
        COMMENT "Cythonizing cython_processor.pyx"
    )

    Python_add_library(cython_processor MODULE ${CYTHON_OUTPUT} ${COMMON_SOURCES})
    target_include_directories(cython_processor PRIVATE ${CYTHON_DIR} ${COMMON_DIR})
    set_target_properties(cython_processor PROPERTIES
        OUTPUT_NAME "cython_processor"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cython_processor
    )
    install(TARGETS cython_processor LIBRARY DESTINATION cython_processor)
endif()

# =============================================================================
# pybind11 Extension
# =============================================================================
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Try to find via Python
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(pybind11_DIR)
        find_package(pybind11 CONFIG REQUIRED)
    endif()
endif()

if(pybind11_FOUND)
    message(STATUS "Building pybind11 extension")
    set(PYBIND_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind_processor)

    pybind11_add_module(pybind_processor
        ${PYBIND_DIR}/pybind_processor.cpp
        ${COMMON_SOURCES}
    )
    target_include_directories(pybind_processor PRIVATE ${COMMON_DIR})
    set_target_properties(pybind_processor PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pybind_processor
    )
    install(TARGETS pybind_processor LIBRARY DESTINATION pybind_processor)
else()
    message(STATUS "pybind11 not found, skipping pybind_processor")
endif()

# =============================================================================
# nanobind Extension
# =============================================================================
find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
    # Try to find via Python
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import nanobind; print(nanobind.cmake_dir())"
        OUTPUT_VARIABLE nanobind_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(nanobind_DIR)
        find_package(nanobind CONFIG REQUIRED)
    endif()
endif()

if(nanobind_FOUND)
    message(STATUS "Building nanobind extension")
    set(NANOBIND_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/nanobind_processor)

    nanobind_add_module(nanobind_processor
        ${NANOBIND_DIR}/nanobind_processor.cpp
        ${COMMON_SOURCES}
    )
    target_include_directories(nanobind_processor PRIVATE ${COMMON_DIR})
    set_target_properties(nanobind_processor PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/nanobind_processor
    )
    install(TARGETS nanobind_processor LIBRARY DESTINATION nanobind_processor)
else()
    message(STATUS "nanobind not found, skipping nanobind_processor")
endif()

# =============================================================================
# SWIG Extension
# =============================================================================
find_package(SWIG QUIET)
if(SWIG_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/swig_processor/swig_processor.i")
    message(STATUS "Building SWIG extension")
    include(UseSWIG)

    set(SWIG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/swig_processor)
    set(CMAKE_SWIG_FLAGS "-I${COMMON_DIR}" "-I${Python_NumPy_INCLUDE_DIRS}")
    set_property(SOURCE ${SWIG_SOURCE_DIR}/swig_processor.i PROPERTY CPLUSPLUS ON)

    swig_add_library(swig_processor
        TYPE MODULE
        LANGUAGE python
        SOURCES ${SWIG_SOURCE_DIR}/swig_processor.i ${COMMON_SOURCES}
    )
    target_include_directories(swig_processor PRIVATE ${COMMON_DIR} ${Python_NumPy_INCLUDE_DIRS})
    target_link_libraries(swig_processor PRIVATE Python::Module Python::NumPy)
    set_target_properties(swig_processor PROPERTIES
        OUTPUT_NAME "_swig_processor_impl"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/swig_processor
    )
    install(TARGETS swig_processor LIBRARY DESTINATION swig_processor)
else()
    message(STATUS "SWIG not found or interface file missing, skipping swig_processor")
endif()

# =============================================================================
# ctypes Extension
# =============================================================================
message(STATUS "Building ctypes shared library")
set(CTYPES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/ctypes_processor)

# Build shared library for ctypes
add_library(ctypes_processor SHARED
    ${CTYPES_DIR}/c_wrapper.cpp
    ${COMMON_SOURCES}
)
target_include_directories(ctypes_processor PRIVATE ${COMMON_DIR})
set_target_properties(ctypes_processor PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ctypes_processor
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ctypes_processor
    PREFIX ""  # Don't add 'lib' prefix on Unix
)
install(TARGETS ctypes_processor
    LIBRARY DESTINATION ctypes_processor
    RUNTIME DESTINATION ctypes_processor
)

# =============================================================================
# cffi Extension
# =============================================================================
# cffi uses its own build system, so we just check if it's available
# and run the build script if needed
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import cffi"
    RESULT_VARIABLE CFFI_FOUND
    OUTPUT_QUIET
    ERROR_QUIET
)

if(CFFI_FOUND EQUAL 0)
    # Check if _cffi_backend is available (required C extension)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import _cffi_backend"
        RESULT_VARIABLE CFFI_BACKEND_FOUND
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(CFFI_BACKEND_FOUND EQUAL 0)
        message(STATUS "Building cffi extension")
        set(CFFI_BUILD_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/src/cffi_processor/build_cffi.py)

        # Run the cffi build script during configure time
        add_custom_target(cffi_processor ALL
            COMMAND ${Python_EXECUTABLE} ${CFFI_BUILD_SCRIPT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/cffi_processor
            COMMENT "Building cffi extension"
            DEPENDS ${CFFI_BUILD_SCRIPT}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/cffi_processor/cffi_wrapper.h
                ${CMAKE_CURRENT_SOURCE_DIR}/src/cffi_processor/cffi_wrapper.cpp
                ${COMMON_SOURCES}
        )
    else()
        message(STATUS "cffi_backend not available (Python 3.14 compatibility), skipping cffi_processor")
    endif()
else()
    message(STATUS "cffi not found, skipping cffi_processor")
endif()

# =============================================================================
# HPy Extension
# =============================================================================
# Check if HPy is available
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "from hpy.devel import HPyDevel; print(HPyDevel().include_dir)"
    OUTPUT_VARIABLE HPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE HPY_FOUND
    ERROR_QUIET
)

if(HPY_FOUND EQUAL 0)
    message(STATUS "Building HPy extension")
    set(HPY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/hpy_processor)

    # Get HPy extra sources
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "from hpy.devel import HPyDevel; print(';'.join(HPyDevel().get_extra_sources()))"
        OUTPUT_VARIABLE HPY_EXTRA_SOURCES
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # Build HPy extension module
    Python_add_library(hpy_processor MODULE
        ${HPY_DIR}/hpy_processor.c
        ${CTYPES_DIR}/c_wrapper.cpp
        ${COMMON_SOURCES}
        ${HPY_EXTRA_SOURCES}
    )
    target_include_directories(hpy_processor PRIVATE
        ${HPY_INCLUDE_DIR}
        ${COMMON_DIR}
        ${CTYPES_DIR}
    )

    # HPy requires specific compile definitions
    target_compile_definitions(hpy_processor PRIVATE
        HPY_UNIVERSAL_ABI
    )

    set_target_properties(hpy_processor PROPERTIES
        OUTPUT_NAME "hpy_processor"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/hpy_processor
    )
    install(TARGETS hpy_processor LIBRARY DESTINATION hpy_processor)
else()
    message(STATUS "HPy not found, skipping hpy_processor")
endif()

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "py-cpp-bridge build configuration:")
message(STATUS "  Python: ${Python_EXECUTABLE}")
message(STATUS "  NumPy:  ${Python_NumPy_INCLUDE_DIRS}")
message(STATUS "  pybind11: ${pybind11_FOUND}")
message(STATUS "  nanobind: ${nanobind_FOUND}")
message(STATUS "  Cython: enabled (via python -m cython)")
message(STATUS "  SWIG: ${SWIG_FOUND}")
message(STATUS "  ctypes: enabled")
if(CFFI_FOUND EQUAL 0)
    message(STATUS "  cffi: enabled")
else()
    message(STATUS "  cffi: not found")
endif()
if(HPY_FOUND EQUAL 0)
    message(STATUS "  HPy: enabled")
else()
    message(STATUS "  HPy: not found")
endif()
message(STATUS "")
