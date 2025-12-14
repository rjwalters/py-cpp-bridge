# Py-Cpp-Bridge

A comprehensive comparison of Python/C++ binding technologies, demonstrating multiple approaches to high-performance interoperability with a single source of truth for data types.

## Overview

This project demonstrates how to integrate high-performance C++ code with Python using multiple binding technologies. Each implementation features three different approaches to memory handling and type conversion, enabling you to compare binding approaches and choose the right solution for your specific use case.

The core functionality is a simple array processor that doubles each value in an array, but the techniques demonstrated can be applied to any C++ code you want to make available in Python.

## Binding Technologies

| Technology | Status | Description |
|------------|--------|-------------|
| **Cython** | ✅ Implemented | Python-like syntax, compiles to C |
| **pybind11** | ✅ Implemented | Header-only C++ library |
| **nanobind** | ✅ Implemented | Faster pybind11 successor (C++17) |
| **SWIG** | ✅ Implemented | Multi-language code generator |
| **ctypes** | ✅ Implemented | Built-in Python FFI ([#3](https://github.com/rjwalters/py-cpp-bridge/issues/3)) |
| **cffi** | ✅ Implemented | C FFI with PyPy support |
| **HPy** | ✅ Implemented | Universal Python API ([#5](https://github.com/rjwalters/py-cpp-bridge/issues/5)) |

See [Benchmarks](#benchmarks) below for performance comparison ([#6](https://github.com/rjwalters/py-cpp-bridge/issues/6)).

## Features

- ✅ Bidirectional data transfer between Python and C++
- ✅ Three different methods for handling memory and type conversion:
  - Pre-allocated buffer (maximum efficiency for repeated calls)
  - New contiguous array (flexibility with clean separation)
  - Manual casting (precise control over type conversion)
- ✅ Single source of truth for data types between C++ and Python
- ✅ Full type annotation support with PEP 561 stub files
- ✅ Debugging and production build configurations
- ✅ Comprehensive documentation for each method

## Requirements

- Python 3.11+
- CMake 3.18+
- NumPy 2.x
- Cython 3.0+
- pybind11 3.0+
- nanobind 2.0+
- A C++17 compiler (gcc, clang, MSVC)

## Installation

```bash
# Clone the repository
git clone https://github.com/rjwalters/py-cpp-bridge.git
cd py-cpp-bridge

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and build
make install
```

The build uses **CMake** via **scikit-build-core** for modern, cross-platform compilation.

## Usage

All implementations (Cython, pybind11, nanobind, SWIG, ctypes, cffi, HPy) provide the same API, so you can easily switch between them.

For detailed usage examples and implementation-specific information, see:

- [Cython implementation](src/cython_processor/README.md) - Python-like syntax with typed memoryviews
- [pybind11 implementation](src/pybind_processor/README.md) - Header-only C++ library
- [nanobind implementation](src/nanobind_processor/README.md) - Modern pybind11 successor
- [SWIG implementation](src/swig_processor/README.md) - Multi-language code generator
- [ctypes implementation](src/ctypes_processor/README.md) - Built-in Python FFI
- [cffi implementation](src/cffi_processor/README.md) - C FFI with excellent PyPy support
- [HPy implementation](src/hpy_processor/README.md) - Universal Python API for portability

### Type-Annotated Example

The package includes full type annotations for better IDE integration and static type checking:

```python
from typing import List
import numpy as np
import numpy.typing as npt
from cython_processor import PyArrayProcessor

def process_batches(data_batches: List[npt.NDArray], batch_size: int) -> List[npt.NDArray]:
    processor = PyArrayProcessor(batch_size)
    
    results: List[npt.NDArray] = []
    for batch in data_batches:
        # Type checkers understand this returns NDArray
        processed = processor.process_preallocated(batch)
        results.append(processed)
    
    return results
```

## Single Source of Truth for Types

This project demonstrates how to maintain a single source of truth for data types between C++ and Python. The approach works as follows:

1. Core type definitions are placed in a central C++ header file
2. Cython declarations in `.pxd` import these types
3. Type information is exposed to Python through a dedicated function
4. The same types are used consistently in both C++ and Python code

### Benefits:
- Changes to a type only need to be made in one place
- Automatic conversion between C++ and NumPy types
- Type safety between languages
- Self-documenting code with explicit type mappings

## Methods Explained

### Method 1: Pre-allocated Buffer

```python
result = processor.process_preallocated(data)
```

**Pros:**
- Efficient for repeated calls (reuses the same buffer)
- Minimizes memory allocations
- Clear type safety with explicit array typing

**Best for:** High-performance code with frequent processing calls on similarly sized data

### Method 2: New Contiguous Array

```python
result = processor.process_new(data)
```

**Pros:**
- More flexible input handling (auto-converts types)
- Clean separation between input and output
- Each result is independent

**Best for:** General-purpose use where convenience is valued over ultimate performance

### Method 3: Manual Casting

```python
result = processor.process_manual(data)
```

**Pros:**
- Maximum control over data conversion
- Can handle arrays of any type
- Can perform validation or transformation during copying

**Best for:** Cases where precise control over type conversion is needed

## Development

### Build Modes

The project supports both debug and production builds:

```bash
# Production build (optimized)
make build

# Debug build (with bounds checking, etc.)
make debug
```

### Running Tests

```bash
make test
```

### Code Formatting

```bash
make format
```

## Project Structure

```
├── src/
│   ├── common/
│   │   ├── cpp_processor.cpp    # C++ implementation
│   │   ├── cpp_processor.hpp    # C++ header
│   │   └── types.hpp            # Shared type definitions
│   ├── cython_processor/        # ✅ Implemented
│   ├── pybind_processor/        # ✅ Implemented
│   ├── nanobind_processor/      # ✅ Implemented
│   ├── swig_processor/          # ✅ Implemented
│   ├── ctypes_processor/        # ✅ Implemented
│   ├── cffi_processor/          # ✅ Implemented
│   └── hpy_processor/           # ✅ Implemented
├── tests/                       # Test scripts for each implementation
├── benchmarks/                  # Performance comparison (pytest-benchmark)
├── CMakeLists.txt               # CMake build configuration
├── pyproject.toml               # Python package configuration (scikit-build-core)
├── Makefile                     # Convenience commands
└── README.md                    # This file
```

## How It Works

1. C++ code in the `common` directory defines the core functionality
2. Type definitions in `types.hpp` are shared between C++ and all Python bindings
3. CMake builds most binding technologies (Cython, pybind11, nanobind, SWIG, ctypes, HPy) as separate modules
4. cffi uses its own build system via a Python script that compiles C wrapper code
5. scikit-build-core integrates CMake with Python packaging for `pip install` support
6. The Makefile provides convenient commands for building and testing all implementations

### Binding Technology Comparison

| Aspect | Cython | pybind11 | nanobind | SWIG | ctypes | cffi | HPy |
|--------|--------|----------|----------|------|--------|------|-----|
| **Language** | Python-like | C++ | C++ | Interface | Pure Python | C decl. | C |
| **C++ Standard** | C++11 | C++11 | C++17 | Any | Any | Any | Any |
| **Compilation** | Py→C→Binary | C++→Binary | C++→Binary | Code gen | C/C++ only | Optional | C→Binary |
| **NumPy** | Memoryviews | py::array_t | nb::ndarray | Manual | ctypes | Manual | Manual |
| **Binary size** | Medium | Large | Small | Medium | N/A | Small | Small |
| **Compile time** | Medium | Slow | Fast | Medium | None | Fast | Medium |
| **PyPy support** | Limited | None | None | Limited | Good | Excellent | Excellent |

## Debugging Features

When built in debug mode (`make debug`), the following features are enabled:

- Bounds checking for array access
- Initialization checks for memory views
- Overflow checking for arithmetic operations
- Line tracing for coverage tools
- Profiling support
- Annotated HTML output showing the Cython to C conversion

## Benchmarks

Performance comparison of all binding implementations using `pytest-benchmark`. Results show mean execution time for processing 10,000 element arrays (lower is better).

### Results Summary

| Technology | Preallocated | New Array | Manual Cast | Call Overhead |
|------------|--------------|-----------|-------------|---------------|
| **Cython** | 12.97 ms | 12.57 ms | 12.98 ms | 5.20 μs |
| **pybind11** | 12.69 ms | 12.64 ms | 12.92 ms | 5.54 μs |
| **nanobind** | 12.86 ms | 12.80 ms | 12.79 ms | 5.76 μs |
| **SWIG** | 12.88 ms | 12.70 ms | 15.05 ms | 5.81 μs |
| **ctypes** | ~13 ms | ~13 ms | ~13 ms | 9.94 μs |
| **cffi** | TBD | TBD | TBD | TBD |
| **HPy** | — | — | — | — |

*Measured on Apple M3 Pro, Python 3.14, macOS 15.3*

### Key Observations

- **Processing time**: All implementations perform similarly (~12.6-13.0 ms) since the actual work is done in C++
- **Call overhead**: Cython has the lowest call overhead (~5.2 μs), followed by pybind11 (~5.5 μs) and nanobind (~5.8 μs). ctypes shows slightly higher overhead (~10 μs) due to additional function pointer indirection
- **SWIG manual casting**: ~16% slower due to Python-layer type conversion in the wrapper
- **ctypes performance**: Despite higher call overhead, ctypes matches other implementations for array processing since computation dominates
- **Differences are minimal**: For array processing, the binding choice matters less than the algorithm

### Running Benchmarks

```bash
# Run all benchmarks
make benchmark

# Quick call overhead test
make benchmark-quick

# Compare bridges at 10K elements
make benchmark-compare

# Save results to JSON
make benchmark-save
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.