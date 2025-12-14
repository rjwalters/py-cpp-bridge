# ctypes Processor

Python-to-C++ binding implementation using [ctypes](https://docs.python.org/3/library/ctypes.html), Python's built-in FFI library.

## Files

| File | Description |
|------|-------------|
| `c_wrapper.h` | C API header for ctypes |
| `c_wrapper.cpp` | C wrapper implementation |
| `ctypes_processor.py` | Python ctypes wrapper |
| `__init__.py` | Package initializer |

## Overview

ctypes is Python's built-in Foreign Function Interface library that provides C-compatible data types and allows calling functions in DLLs or shared libraries. It requires no compilation on the Python side and works directly with NumPy arrays.

## Key Features

- **No Python compilation**: Uses pure Python to interface with C libraries
- **Built-in**: Part of the Python standard library - no dependencies beyond NumPy
- **Cross-platform**: Works on Windows, macOS, and Linux
- **NumPy integration**: Direct memory access via `ctypes.data_as()`

## Basic Example

```python
import numpy as np
from ctypes_processor import PyArrayProcessor

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
- **Call Overhead**: 10.89 μs (higher but still microseconds, function pointer indirection)
- **Processing (10K elements)**: ~13.46 ms (preallocated, best!), ~14.64 ms (new), ~15.59 ms (manual)

### Generate Flamegraph
```bash
# Install py-spy if not already installed
pip install py-spy

# Generate flamegraph for ctypes binding
python benchmarks/generate_flamegraph.py ctypes

# View the flamegraph
open benchmarks/flamegraphs/ctypes.svg
```

### What to Expect in Flamegraphs
ctypes flamegraphs typically show:
- **Function pointer indirection** - CDLL dynamic loading and resolution overhead
- **Python-side wrapper** - Pure Python wrapper class visible in stack
- **Type marshalling** - `ctypes.data_as()` and pointer conversions
- **Best preallocated performance** - Efficient buffer reuse (13.46 ms, best overall!)
- **No compilation overhead** - Direct FFI calls to C functions

**Insight**: Despite higher call overhead, ctypes achieves the best preallocated performance among all bindings, making it ideal for repeated operations.

For detailed benchmark analysis and comparisons with other bindings, see [BENCHMARKS.md](/BENCHMARKS.md).

## How It Works

1. A C wrapper layer (`c_wrapper.cpp`) provides `extern "C"` functions that wrap the C++ `ArrayProcessor` class
2. The Python module (`ctypes_processor.py`) uses `ctypes.CDLL` to load the shared library
3. Function signatures are defined using `argtypes` and `restype`
4. NumPy arrays are passed using `array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))`
5. Results are converted back to NumPy arrays using `np.ctypeslib.as_array()`

## Architecture

```
┌─────────────────────────────────────┐
│   Python (ctypes_processor.py)     │
│  - PyArrayProcessor class           │
│  - NumPy array handling             │
└──────────────┬──────────────────────┘
               │ ctypes.CDLL
┌──────────────▼──────────────────────┐
│   C Wrapper (c_wrapper.cpp)         │
│  - extern "C" functions             │
│  - Opaque handle pattern            │
└──────────────┬──────────────────────┘
               │ C++ calls
┌──────────────▼──────────────────────┐
│   C++ Core (cpp_processor.cpp)      │
│  - ArrayProcessor class             │
│  - Actual array processing          │
└─────────────────────────────────────┘
```

## Advantages

- **No compilation needed**: Users don't need a C++ compiler
- **Standard library**: No additional dependencies
- **Flexible**: Can interface with any C-compatible library
- **Portable**: Works across different Python implementations (CPython, PyPy)

## Disadvantages

- **Manual wrapper**: Requires writing C wrapper layer
- **Type safety**: Less type safety than C++ binding tools
- **Performance**: Small overhead from Python-side wrapper code
- **Error handling**: Requires manual error propagation

## Comparison with Other Bindings

| Aspect | ctypes | pybind11 | Cython |
|--------|--------|----------|--------|
| Installation | Built-in | pip install | pip install |
| Compilation | C++ only | C++ only | Python + C++ |
| Type safety | Manual | Automatic | Automatic |
| NumPy support | Via ctypes | Native | Native |
| Learning curve | Moderate | Moderate | Steep |
| Use case | FFI to existing C/C++ | New bindings | Python-like C extensions |

## When to Use ctypes

ctypes is ideal when:
- You want to interface with existing C/C++ libraries without modifying them
- You don't want users to need a compiler
- You need a quick prototype without setting up a build system
- You're working with simple C-style APIs

For complex C++ APIs with templates, classes, and exceptions, consider pybind11 or nanobind instead.
