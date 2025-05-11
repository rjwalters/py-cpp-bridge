import multiprocessing
import os
import sys
from typing import Optional

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as BuildExtCommand

# Enable multiprocessing support
multiprocessing.freeze_support()

# Debug flag to easily toggle development vs production settings
DEBUG = os.environ.get("DEBUG", "").lower() in ("true", "1", "t", "yes")

class CustomBuildExt(BuildExtCommand):
    """Custom build extension to handle compiler flags."""
    def build_extensions(self):
        # Remove -DNDEBUG from compiler flags to enable assertions when in debug mode
        if DEBUG and "-DNDEBUG" in self.compiler.compiler_so:
            self.compiler.compiler_so.remove("-DNDEBUG")
        super().build_extensions()

# Define source files with proper paths
SRC_DIR = "src"
CYTHON_DIR = os.path.join(SRC_DIR, "cython_processor")
COMMON_DIR = os.path.join(SRC_DIR, "common")

# Cython source file
PYX_FILE = os.path.join(CYTHON_DIR, "cython_processor.pyx")

# C++ implementation file (now in common directory)
CPP_FILE = os.path.join(COMMON_DIR, "cpp_processor.cpp")

# Define the extension module with proper package path
ext_modules = [
    Extension(
        "cython_processor",  # Full package path for module
        sources=[PYX_FILE, CPP_FILE],  # Combines Cython wrapper with C++ implementation
        include_dirs=[
            numpy.get_include(), 
            SRC_DIR,            # Include src directory
            COMMON_DIR,         # Include common directory for C++ headers
            CYTHON_DIR,         # Include cython directory
            ".",                # Include root directory
        ],
        language="c++",
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("DEBUG", "1" if DEBUG else "0"),  # Add DEBUG macro to C++ code
        ],
        extra_compile_args=["-Wno-unreachable-code"],
    )
]

# Set build directory based on debug mode
BUILD_DIR = "build_debug" if DEBUG else "build"
os.makedirs(BUILD_DIR, exist_ok=True)

# Detect number of CPU cores for parallel compilation on Linux
num_threads: Optional[int] = multiprocessing.cpu_count() if sys.platform == "linux" else None

# Configure compiler directives based on DEBUG setting
compiler_directives = {
    "language_level": "3",
    "embedsignature": True,  # Include docstrings and signatures in compiled module
    "annotation_typing": True,  # Enable type annotations
    "c_string_encoding": "utf8",
    "c_string_type": "str",
    # Always enable these for better error messages during development
    "binding": True,         # Generate Python wrapper code
    "embedsignature": True,  # Include docstrings in the C code
}

# Add debug-specific directives when DEBUG is True
if DEBUG:
    compiler_directives.update(
        {
            # Error catching settings
            "boundscheck": True,  # Check array bounds (catches index errors)
            "wraparound": True,  # Handle negative indices correctly
            "initializedcheck": True,  # Check if memoryviews are initialized
            "nonecheck": True,  # Check if arguments are None
            "overflowcheck": True,  # Check for C integer overflow
            "overflowcheck.fold": True,  # Also check operations folded by the compiler
            "cdivision": False,  # Check for division by zero (slows code down)
            # For performance debugging:
            "profile": True,  # Enable profiling
            "linetrace": True,  # Enable line tracing for coverage tools
        }
    )
else:
    # Production settings for better performance
    compiler_directives.update(
        {
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "overflowcheck": False,
            "cdivision": True,
            "profile": False,
            "linetrace": False,
        }
    )

# Setup
setup(
    name="py-cpp-bridge",  # Changed to match repo name
    version="0.1.0",
    ext_modules=cythonize(
        ext_modules,
        build_dir=BUILD_DIR,
        nthreads=num_threads,
        annotate=DEBUG,  # Generate annotated HTML files in debug mode to see Python → C translation
        compiler_directives=compiler_directives,
        include_path=[CYTHON_DIR, SRC_DIR, COMMON_DIR],  # Include directories for finding .pxd files
    ),
    package_dir={"": "src"},  # Tell setuptools packages are under src/
    packages=find_packages(where="src"),  # Find packages under src/
    include_package_data=True,
    install_requires=["numpy"],
    cmdclass={"build_ext": CustomBuildExt},  # Use our custom build_ext class
    description="Python/C++ bridging examples using various technologies (Cython, pybind11)",
    url="https://github.com/yourusername/py-cpp-bridge",
    python_requires=">=3.7",
)