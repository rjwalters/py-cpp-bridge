# nanobind Processor

Python-to-C++ binding implementation using [nanobind](https://nanobind.readthedocs.io/).

## Files

| File | Description |
|------|-------------|
| `nanobind_processor.cpp` | nanobind implementation in C++ |
| `nanobind_processor.pyi` | Python stub file for type hints |
| `__init__.py` | Package initializer |

## Overview

nanobind is the modern successor to pybind11, designed for C++17 and newer. It offers faster compilation, smaller binary sizes, and lower runtime overhead compared to pybind11.

## Basic Example

```python
import numpy as np
from nanobind_processor import PyArrayProcessor

# Identical API to pybind11 - nanobind is its modern successor
np_values_type = np.dtype(PyArrayProcessor.get_numpy_type_name("value"))
processor = PyArrayProcessor(5)
data = np.array([1, 2, 3, 4, 5], dtype=np_values_type)

# Same three methods, optimized for smaller binaries and faster compilation
result1 = processor.process_preallocated(data)
result2 = processor.process_new(data)
result3 = processor.process_manual(data)
```

## Processing Methods

| Method | Memory Strategy | Best For |
|--------|-----------------|----------|
| `process_preallocated()` | Reuses internal buffer | Repeated calls with same-sized arrays |
| `process_new()` | Allocates fresh array | Flexibility, auto type conversion |
| `process_manual()` | Explicit copying/casting | Maximum control over memory |

## Performance Profiling

### Benchmark Results
- **Call Overhead**: 7.36 μs (excellent, modern optimized design)
- **Processing (10K elements)**: ~14.58 ms (preallocated), ~14.88 ms (new), ~14.77 ms (manual)

### Generate Flamegraph
```bash
# Install py-spy if not already installed
pip install py-spy

# Generate flamegraph for nanobind binding
python benchmarks/generate_flamegraph.py nanobind

# View the flamegraph
open benchmarks/flamegraphs/nanobind.svg
```

### What to Expect in Flamegraphs
nanobind flamegraphs typically show:
- **Compact binding overhead** - Optimized C++17 templates, less machinery than pybind11
- **Efficient dispatch** - Modern design reduces intermediate layers
- **Native ndarray support** - `nb::ndarray<>` integration
- **Consistent performance** - All three methods show similar behavior
- **Small binary footprint** - Reflected in cleaner stack traces

For detailed benchmark analysis and comparisons with other bindings, see [BENCHMARKS.md](/BENCHMARKS.md).

## How It Works

1. The `NB_MODULE` macro defines the Python module
2. Classes are exposed using `nb::class_<>`
3. NumPy arrays are handled via `nb::ndarray<>`
4. Memory ownership is managed through capsules

## Advantages Over pybind11

- **Faster compilation** - Reduced template complexity
- **Smaller binaries** - More efficient code generation
- **Lower runtime overhead** - Optimized dispatch mechanisms
- **Modern C++17 design** - Leverages newer language features
- **Better NumPy integration** - Native ndarray support
