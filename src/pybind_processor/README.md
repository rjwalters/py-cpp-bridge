# pybind11 Processor

Python-to-C++ binding implementation using [pybind11](https://pybind11.readthedocs.io/).

## Files

| File | Description |
|------|-------------|
| `pybind_processor.cpp` | pybind11 implementation in C++ |
| `pybind_processor.pyi` | Python stub file for type hints |
| `__init__.py` | Package initializer |

## Overview

pybind11 is a lightweight header-only library that exposes C++ types in Python and vice versa. It uses C++11 features to create Python bindings with minimal boilerplate.

## Basic Example

```python
import numpy as np
from pybind_processor import PyArrayProcessor

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
- **Call Overhead**: 6.41 μs (excellent, 2nd lowest)
- **Processing (10K elements)**: ~14.39 ms (preallocated), ~15.09 ms (new), ~13.45 ms (manual, best!)

### Generate Flamegraph
```bash
# Install py-spy if not already installed
pip install py-spy

# Generate flamegraph for pybind11 binding
python benchmarks/generate_flamegraph.py pybind11

# View the flamegraph
open benchmarks/flamegraphs/pybind11.svg
```

### What to Expect in Flamegraphs
pybind11 flamegraphs typically show:
- **Template-based dispatch** - C++ template instantiation overhead visible
- **Type conversion layers** - Automatic Python↔C++ type conversions
- **NumPy integration** - `py::array_t<>` handling code
- **Optimized manual casting** - Best performance for element-wise type conversion
- **Computation dominance** - Most time still in `cpp_processor::process_array`

For detailed benchmark analysis and comparisons with other bindings, see [BENCHMARKS.md](/BENCHMARKS.md).

## How It Works

1. The `PYBIND11_MODULE` macro defines the Python module
2. C++ classes are exposed using `py::class_<>`
3. Methods are bound with `.def()` calls
4. NumPy arrays are handled via `py::array_t<>`

## Advantages

- Header-only: no separate library to link
- Natural C++ syntax for defining bindings
- Excellent documentation and community support
- Automatic type conversions between Python and C++
