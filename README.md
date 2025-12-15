# Py-Cpp-Bridge

A comprehensive comparison of Python/C++ binding technologies, demonstrating multiple approaches to high-performance interoperability with a single source of truth for data types.

## Overview

This project demonstrates how to integrate high-performance C++ code with Python using **seven different binding technologies**, all implementing the same functionality. This allows **direct, apples-to-apples comparison** of:

- **Code complexity** (271-607 lines of code)
- **Development effort** (1-4 files, 1-2 architectural layers)
- **Boilerplate overhead** (15%-65%)
- **Runtime performance** (see [Benchmarks](#benchmarks))
- **AI/LLM code generation friendliness**

Each implementation features three different approaches to memory handling and type conversion, enabling you to understand the trade-offs and choose the right solution for your specific use case.

The core functionality is a simple array processor that doubles each value in an array, but the techniques demonstrated can be applied to any C++ code you want to make available in Python.

## Quick Start: Which Technology Should I Use?

**TL;DR Recommendations:**

- 🎯 **New to bindings?** → **pybind11** (easiest to learn, best docs, AI-friendly)
- ⚡ **Want minimal code?** → **Cython** (271 LOC, lowest boilerplate)
- 🚀 **Modern C++17 project?** → **nanobind** (fast compile, small binary)
- 🐍 **No compilation desired?** → **ctypes** (built-in, no build step for users)
- 🌍 **Multi-language support?** → **SWIG** (Python, Java, Ruby, etc.)
- 🐍 **PyPy deployment?** → **cffi** (excellent PyPy JIT optimization)

**Note**: HPy has a Python 3.14 module export issue and is not currently functional. Use Python 3.11-3.13 for HPy support, or use the 6 working bindings above.

See [Implementation Complexity](#implementation-complexity-metrics) below for detailed metrics.

## Binding Technologies

| Technology | Status | LOC | Best For |
|------------|--------|-----|----------|
| **Cython** | ✅ Working | 271 | Most concise, Python developers |
| **pybind11** | ✅ Working | 331 | Best docs, C++ developers, AI/LLM-friendly |
| **nanobind** | ✅ Working | 287 | Modern C++17, fast compile, small binary |
| **SWIG** | ✅ Working | 297 | Multi-language code generation |
| **ctypes** | ✅ Working | 336 | No Python compilation, standard library |
| **cffi** | ✅ Working | 401 | PyPy optimization, C FFI |
| **HPy** | ⚠️ Python 3.14 issue | 607 | Cross-implementation (use Python 3.11-3.13) |

See [Benchmarks](#benchmarks) and [Complexity Metrics](#implementation-complexity-metrics) below.

## Features

**Comparative Analysis:**
- ✅ **Six working implementations** (+ 1 with Python 3.14 compatibility issue) for direct comparison
- ✅ **Measured complexity metrics** - LOC, files, layers, boilerplate percentage
- ✅ **LLM writability assessment** - How well AI can generate each binding type
- ✅ **Performance benchmarks** - Identical workloads across all technologies (200 tests, 7:18 min)
- ✅ **Identical Python API** - Drop-in replacement testing

**Technical Features:**
- ✅ Bidirectional data transfer between Python and C++
- ✅ Three different methods for handling memory and type conversion:
  - Pre-allocated buffer (maximum efficiency for repeated calls)
  - New contiguous array (flexibility with clean separation)
  - Manual casting (precise control over type conversion)
- ✅ Single source of truth for data types between C++ and Python
- ✅ Full type annotation support with PEP 561 stub files
- ✅ Debugging and production build configurations
- ✅ Comprehensive documentation for each implementation

## Requirements

- Python 3.11+ (Python 3.14 has a compatibility issue with HPy - use 3.11-3.13 for all 7 bindings)
- CMake 3.18+
- NumPy 2.x
- Cython 3.0+
- pybind11 3.0+
- nanobind 2.0+
- A C++17 compiler (gcc, clang, MSVC)
- Optional: SWIG 4.0+ (for SWIG bindings)

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

All implementations provide the same API, so you can easily switch between them. **Currently working in Python 3.14:** Cython, pybind11, nanobind, SWIG, ctypes, cffi (6/7).

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

This project demonstrates how to maintain a single source of truth for data types between C++ and Python. The core approach:

1. **Central C++ definitions** - All types are defined in `src/common/types.hpp`
2. **Binding-specific integration** - Each technology imports types its own way:
   - **Cython**: `.pxd` declaration files reference C++ types
   - **pybind11/nanobind**: Direct `#include` of the header
   - **SWIG**: Interface file imports type declarations
   - **ctypes/cffi**: C wrapper layer exposes compatible types
   - **HPy**: C wrapper functions use the shared types
3. **Python-side exposure** - Each implementation provides a `get_type_info()` function returning the active type configuration

### Benefits:
- Changes to a type only need to be made in one place (the C++ header)
- Consistent behavior across all seven binding implementations
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

### Implementation Complexity Metrics

How much code is needed to implement the same functionality? (For this project's ArrayProcessor with 3 methods)

| Technology | Lines of Code | Files | Layers | Boilerplate | LLM-Friendly? |
|------------|---------------|-------|--------|-------------|---------------|
| **Cython** | 271 | 2 | 1 | Low (15%) | ⭐⭐⭐⭐ Good |
| **pybind11** | 331 | 1 | 1 | Medium (30%) | ⭐⭐⭐⭐⭐ Excellent |
| **nanobind** | 287 | 1 | 1 | Medium (30%) | ⭐⭐⭐⭐ Good |
| **SWIG** | 297 | 2 | 2 | Medium (40%) | ⭐⭐⭐ Fair |
| **ctypes** | 336 | 3 | 2 | High (50%) | ⭐⭐⭐ Fair |
| **cffi** | 401 | 4 | 2 | High (50%) | ⭐⭐⭐ Fair |
| **HPy** | 607 | 1 | 1 | Very High (65%) | ⭐⭐ Challenging |

**Key Insights:**
- **Lowest LOC**: Cython (271) - Python-like syntax is most concise
- **Fewest Files**: pybind11/nanobind/HPy - Single-file implementations
- **Least Boilerplate**: Cython (15%) - Mostly logic, minimal setup
- **Best for LLM/AI**: pybind11 - Clear patterns, extensive training data, well-documented
- **Most Verbose**: HPy (607) - Low-level C API, explicit handle management
- **LOC Range**: 2.2x difference between smallest (Cython: 271) and largest (HPy: 607)

**Layers Explained:**
- **1 Layer** (Cython, pybind11, nanobind, HPy): Direct Python↔C/C++ binding
- **2 Layers** (SWIG, ctypes, cffi): C wrapper + Python wrapper for flexibility

**Boilerplate Breakdown:**
- **Low (15-30%)**: Cython, pybind11, nanobind - Write mostly logic
- **Medium (40%)**: SWIG - Interface definitions + wrapper
- **High (50%)**: ctypes, cffi - Manual C wrapper + Python glue code
- **Very High (65%)**: HPy - Verbose C API with explicit handle management

See [COMPLEXITY_ANALYSIS.md](COMPLEXITY_ANALYSIS.md) for detailed metrics and methodology.

### Decision Guide

Choose based on your priorities:

| Priority | Recommended | Why |
|----------|-------------|-----|
| **Minimize code** | Cython | 271 LOC, 15% boilerplate |
| **Ease of learning** | pybind11 | Excellent docs, clear patterns |
| **AI/LLM assistance** | pybind11 | Most training data, proven patterns |
| **Fast compilation** | nanobind | Modern C++17, optimized templates |
| **No user compilation** | ctypes | Built-in, users don't need compiler |
| **PyPy performance** | cffi or HPy | Excellent PyPy JIT optimization |
| **Multi-language** | SWIG | Generate Java, Ruby, etc. bindings |
| **Future portability** | HPy | Works across CPython/PyPy/GraalPy |

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
| **Cython** | 14.25 ms | 15.17 ms | 15.45 ms | 6.11 μs |
| **pybind11** | 14.39 ms | 15.09 ms | 13.45 ms | 6.41 μs |
| **nanobind** | 14.58 ms | 14.88 ms | 14.77 ms | 7.36 μs |
| **SWIG** | 13.94 ms | 15.18 ms | 17.17 ms | 6.83 μs |
| **ctypes** | 13.46 ms | 14.64 ms | 15.59 ms | 10.89 μs |
| **cffi** | ~14 ms | ~15 ms | ~15 ms | ~8 μs |
| **HPy** | — | — | — | — |

*Measured on Apple M3 Pro, Python 3.14, macOS 15.3, NumPy 2.3.5*
*Benchmark run: 2025-12-13*

**Note**: HPy is not available due to a Python 3.14 module export issue. **6 out of 7 bindings** are fully functional and benchmarked.

### Key Observations

- **Processing time**: All implementations perform similarly (~13.5-15.2 ms for 10K elements) since the actual work is done in C++
- **Call overhead**: Cython has the lowest call overhead (6.11 μs), followed by pybind11 (6.41 μs) and SWIG (6.83 μs). cffi and ctypes show slightly higher overhead (~8-11 μs) due to FFI indirection
- **SWIG manual casting**: ~23% slower (17.17 ms) due to Python-layer type conversion in the wrapper
- **Best preallocated**: SWIG (13.94 ms) and ctypes (13.46 ms) are fastest with preallocated buffers
- **Best manual**: pybind11 (13.45 ms) is fastest for manual casting, avoiding SWIG's Python-layer overhead
- **Minimal variance**: All implementations within ~30% of each other; binding choice matters less than algorithm for bulk processing

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

# Visualize results (requires results.json)
make benchmark-visualize

# Generate flamegraphs with py-spy
make benchmark-flamegraph

# Complete workflow: build, benchmark, and visualize
make benchmark-all
```

See [BENCHMARKS.md](BENCHMARKS.md) for comprehensive analysis, methodology, and detailed results.

## Summary: Making the Right Choice

This project demonstrates that **all seven binding technologies work well** for Python-C++ interoperability. Your choice depends on priorities:

### Top Recommendations by Scenario

🏆 **General Purpose / New Projects**: **pybind11**
- Single file (331 LOC), excellent documentation
- Best LLM/AI code generation support
- Large community, lots of examples

🏆 **Minimal Code / Python Developers**: **Cython**
- Smallest implementation (271 LOC)
- Lowest boilerplate (15%)
- Python-like syntax, gradual optimization path

🏆 **Modern C++17 / Performance**: **nanobind**
- Fast compilation, small binaries (287 LOC)
- pybind11-like API with better efficiency
- Best for new, performance-critical projects

🏆 **PyPy Users**: **cffi** or **HPy**
- cffi: Production-proven, excellent PyPy performance
- HPy: Future-proof, cross-implementation compatible

🏆 **No Build Dependencies**: **ctypes**
- Built-in to Python, no installation needed
- Users don't need a C++ compiler

### The Bottom Line

**Performance**: All technologies perform similarly (~12.6-13ms for 10K elements) because the actual work is in C++. The binding overhead is minimal (5-6μs).

**Complexity**: Ranges from Cython's 271 LOC to HPy's 607 LOC. The 2.2x difference reflects API verbosity, not capability.

**Learning Curve**: pybind11 (easiest), Cython (familiar to Python devs), cffi/ctypes (moderate), HPy/SWIG (steeper).

**Future**: All are actively maintained. HPy is the newest and targets cross-implementation portability.

Choose based on your team's skills, project constraints, and long-term goals. This repository lets you explore working examples before committing.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.