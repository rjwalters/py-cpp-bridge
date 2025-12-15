# cffi Processor

Python-to-C++ binding implementation using [cffi](https://cffi.readthedocs.io/) (C Foreign Function Interface).

## Files

| File | Description |
|------|-------------|
| `cffi_wrapper.h` | C header defining the wrapper interface |
| `cffi_wrapper.cpp` | C++ implementation wrapping ArrayProcessor |
| `build_cffi.py` | cffi builder script |
| `py_wrapper.py` | Python wrapper around cffi-generated bindings |
| `cffi_processor.pyi` | Python stub file for type hints |
| `__init__.py` | Package initializer |

## Overview

cffi provides a way to call C code from Python. Since our core functionality is in C++, we create a C-compatible wrapper (using `extern "C"`) that cffi can interface with. This demonstrates how to bridge Python → C → C++ when direct C++ binding tools aren't suitable.

## Basic Example

```python
import numpy as np
from cffi_processor import PyArrayProcessor

# Get the correct NumPy type
np_values_type = np.dtype(PyArrayProcessor.get_numpy_type_name("value"))

# Create an array processor for size 5 arrays
processor = PyArrayProcessor(5)

# Create some test data using the imported NumPy type
data = np.array([1, 2, 3, 4, 5], dtype=np_values_type)

# Same three methods are available with identical behavior
result1 = processor.process_preallocated(data)  # Pre-allocated buffer
result2 = processor.process_new(data)           # New contiguous array
result3 = processor.process_manual(data)        # Manual casting
```

## Processing Methods

| Method | Memory Strategy | Best For |
|--------|-----------------|----------|
| `process_preallocated()` | Reuses internal buffer | Repeated calls with same-sized arrays |
| `process_new()` | Allocates fresh array | Flexibility, auto type conversion |
| `process_manual()` | Explicit copying/casting | Maximum control over memory |

## Performance Profiling

### Benchmark Results
- **Call Overhead**: ~8 μs (good, similar to ctypes)
- **Processing (10K elements)**: ~14 ms (preallocated), ~15 ms (new), ~15 ms (manual)

### Generate Flamegraph
```bash
# Install py-spy if not already installed
pip install py-spy

# Generate flamegraph for cffi binding
python benchmarks/generate_flamegraph.py cffi

# View the flamegraph
open benchmarks/flamegraphs/cffi.svg
```

### What to Expect in Flamegraphs
cffi flamegraphs typically show:
- **Three-layer architecture** - Python wrapper → cffi-generated code → C wrapper → C++
- **API mode overhead** - Compiled bindings with FFI interface
- **Type conversion** - NumPy array pointer extraction and casting
- **PyPy optimization potential** - cffi is highly optimized for PyPy JIT
- **Clean C interface** - Explicit C API boundary visible

For detailed benchmark analysis and comparisons with other bindings, see [BENCHMARKS.md](/BENCHMARKS.md).

## Architecture

This implementation uses a three-layer approach:

1. **C++ Layer** (`cpp_processor.hpp/cpp`) - Core C++ implementation
2. **C Wrapper Layer** (`cffi_wrapper.h/cpp`) - Extern "C" interface for cffi
3. **Python Wrapper** (`py_wrapper.py`) - Provides the high-level API matching other implementations

## Building

cffi can be built in two ways:

### API Mode (Recommended)
Compiles the C code during build:
```bash
python src/cffi_processor/build_cffi.py
```

### Integration with Project
The CMake build system will automatically build the cffi extension when you run:
```bash
make build
```

## How It Works

1. C++ `ArrayProcessor` class provides the core functionality
2. C wrapper functions (`cffi_wrapper.cpp`) expose a C-compatible interface using opaque handles
3. cffi builder (`build_cffi.py`) generates Python bindings from the C interface
4. Python wrapper (`py_wrapper.py`) adds NumPy handling and provides a clean API

## Advantages

- **No compiler needed at install time** - Can use ABI mode for pure runtime binding
- **PyPy support** - Excellent performance on PyPy compared to ctypes
- **Clear C interface** - Forces you to design a clean C API
- **Flexible** - Supports both API mode (compile time) and ABI mode (runtime)
- **Mature and stable** - Widely used in production Python applications

## Comparison with Other Approaches

| Aspect | cffi | ctypes | pybind11 | SWIG |
|--------|------|--------|----------|------|
| **Interface** | C declarations | C declarations | C++ decorators | Interface files |
| **Compilation** | Optional | None | Required | Required |
| **PyPy support** | Excellent | Good | None | Limited |
| **C++ support** | Via C wrapper | Via C wrapper | Native | Native |
| **Type safety** | Good | Weak | Excellent | Good |
| **Learning curve** | Moderate | Low | Moderate | Moderate |

## Why cffi?

- **Popular in production**: Used by cryptography, PyNaCl, and many other major packages
- **PyPy optimized**: cffi is the recommended way to call C from PyPy
- **ABI mode option**: Can load pre-compiled libraries without build step
- **Clean separation**: Forces good API design with explicit C interface
