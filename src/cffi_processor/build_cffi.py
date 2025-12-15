"""
Build script for cffi bindings.

This script uses cffi's API mode to create Python bindings for the C wrapper
around the C++ ArrayProcessor class.
"""

import os
import sys
from pathlib import Path
from cffi import FFI

ffibuilder = FFI()

# Define the C interface that will be exposed to Python
ffibuilder.cdef("""
    typedef void* ArrayProcessorHandle;

    ArrayProcessorHandle array_processor_new(size_t size);
    void array_processor_delete(ArrayProcessorHandle handle);
    void array_processor_process(ArrayProcessorHandle handle, float* data, size_t size);
    float* array_processor_get_results(ArrayProcessorHandle handle);
    size_t array_processor_get_size(ArrayProcessorHandle handle);
    const char* get_numpy_type_name_wrapper(const char* type_id);
""")

# Get the directory of this script
script_dir = Path(__file__).parent
common_dir = script_dir.parent / "common"

# Source files to compile
sources = [
    str(script_dir / "cffi_wrapper.cpp"),
    str(common_dir / "cpp_processor.cpp"),
]

# Include directories
include_dirs = [
    str(script_dir),
    str(common_dir),
]

# Set the source code for the extension module
# Use "_cffi_impl" without package prefix so it's placed in the current directory
ffibuilder.set_source(
    "_cffi_impl",
    """
    #include "cffi_wrapper.h"
    """,
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args=[],
)

if __name__ == "__main__":
    # Set environment to use C++ compiler
    os.environ['CC'] = os.environ.get('CXX', 'c++')
    os.environ['LDSHARED'] = os.environ.get('CXX', 'c++') + ' -shared'

    # Change to the script directory to ensure consistent output location
    original_dir = os.getcwd()
    os.chdir(script_dir)
    try:
        ffibuilder.compile(verbose=True)
    finally:
        os.chdir(original_dir)
