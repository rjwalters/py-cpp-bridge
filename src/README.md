# Source Code

This directory contains the Python package for py-cpp-bridge, demonstrating four different methods of creating Python-to-C++ bindings.

## Directory Structure

| Directory | Technology | Description |
|-----------|-----------|-------------|
| [common/](./common/) | C++ | Shared C++ implementation used by all binding approaches |
| [cython_processor/](./cython_processor/) | Cython | Python-C++ binding using Cython |
| [pybind_processor/](./pybind_processor/) | pybind11 | Python-C++ binding using pybind11 |
| [nanobind_processor/](./nanobind_processor/) | nanobind | Python-C++ binding using nanobind |
| [swig_processor/](./swig_processor/) | SWIG | Python-C++ binding using SWIG |

## Architecture Overview

All four binding implementations wrap the same core C++ `ArrayProcessor` class from the `common/` directory. Each provides an identical Python API with three array processing methods:

- `process_preallocated()` - Reuses a pre-allocated results buffer (efficient for repeated calls)
- `process_new()` - Creates a fresh array for each call (flexible, auto-converts types)
- `process_manual()` - Manual copying with explicit casting (maximum control)

This allows direct comparison of the different binding technologies while maintaining consistent functionality.

## Quick Comparison

| Approach | Language | Compilation | Binary Size | Best For |
|----------|----------|-------------|-------------|----------|
| Cython | Python-like (.pyx) | Compiles to C/C++ | Medium | Python developers, gradual optimization |
| pybind11 | C++11 | Header-only | Larger | C++ developers, ease of use |
| nanobind | C++17 | Header-only | Smaller | Performance-critical, modern C++ |
| SWIG | Interface files (.i) | Code generation | Medium | Multi-language support |
