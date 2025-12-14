# HPy Processor

Python-to-C++ binding implementation using [HPy](https://hpyproject.org/) (Universal Python API).

## Files

| File | Description |
|------|-------------|
| `hpy_processor.c` | HPy extension implementation in C |
| `hpy_processor.pyi` | Python stub file for type hints |
| `__init__.py` | Package initializer |

## Overview

HPy is a new universal API for Python extensions designed to work across multiple Python implementations (CPython, PyPy, GraalPy). It provides a cleaner, more maintainable API than the traditional C API while enabling better optimization by alternative implementations.

This implementation uses HPy's C-based API to wrap the C++ `ArrayProcessor` class. It supports both the CPython ABI mode and a universal HPy ABI mode for maximum portability.

## Basic Example

```python
from hpy_processor import PyArrayProcessor
import numpy as np

# Get the correct NumPy dtype from C++
np_type = np.dtype(PyArrayProcessor.get_numpy_type_name("value"))
data = np.array([1, 2, 3, 4, 5], dtype=np_type)

# Create processor for arrays of size 5
processor = PyArrayProcessor(size=5)

# Method 1: Pre-allocated buffer (most efficient for repeated calls)
result1 = processor.process_preallocated(data)
print(result1)  # [2. 4. 6. 8. 10.]

# Method 2: New array each time (flexible input handling)
result2 = processor.process_new([1, 2, 3, 4, 5])
print(result2)  # [2. 4. 6. 8. 10.]

# Method 3: Manual casting (maximum control)
result3 = processor.process_manual(np.array([1, 2, 3, 4, 5], dtype=np.float64))
print(result3)  # [2. 4. 6. 8. 10.]

# Clean up
processor.close()
```

## Processing Methods

| Method | Memory Strategy | Best For |
|--------|-----------------|----------|
| `process_preallocated()` | Reuses internal buffer | Repeated calls with same-sized arrays |
| `process_new()` | Allocates fresh array | Flexibility, auto type conversion |
| `process_manual()` | Explicit copying/casting | Maximum control over memory |

## Performance Profiling

### Benchmark Results
⚠️ **Python 3.14 Compatibility Issue**: HPy has a module export function issue in Python 3.14. Benchmarks are unavailable until Python 3.11-3.13 testing is performed or the compatibility issue is resolved.

**Expected Performance** (based on design goals):
- **Call Overhead**: ~6-7 μs (competitive with pybind11/Cython)
- **Processing**: Expected to match or exceed traditional C API performance
- **PyPy Performance**: Should significantly outperform C API bindings on PyPy

### Generate Flamegraph
```bash
# Install py-spy if not already installed
pip install py-spy

# Generate flamegraph for HPy binding (requires Python 3.11-3.13)
python benchmarks/generate_flamegraph.py hpy

# View the flamegraph
open benchmarks/flamegraphs/hpy.svg
```

### What to Expect in Flamegraphs
HPy flamegraphs typically show:
- **HPy context operations** - Handle management and reference tracking
- **Universal ABI layer** - Abstraction layer for cross-implementation compatibility
- **C wrapper interface** - Bridge to C++ ArrayProcessor class
- **NumPy __array_interface__** - Zero-copy array access patterns
- **Handle lifecycle** - Creation, usage, and cleanup of HPy handles
- **Future optimization potential** - PyPy and GraalPy can optimize HPy better than C API

**Insight**: HPy's universal ABI design trades minimal overhead for massive portability gains across Python implementations.

For detailed benchmark analysis and comparisons with other bindings, see [BENCHMARKS.md](/BENCHMARKS.md).

## How It Works

1. The `HPy_MODINIT` macro defines the Python module using HPy's universal API
2. C wrapper functions interface with the C++ `ArrayProcessor` class
3. `HPyType_Spec` defines the Python class with modern type specification
4. NumPy arrays are accessed via `__array_interface__` for zero-copy operations
5. HPy handles manage Python object references automatically

## Advantages

- **Universal ABI**: Works across CPython, PyPy, and GraalPy without recompilation
- **Future-proof**: Designed for long-term compatibility as Python evolves
- **Better optimization**: Alternative implementations can optimize HPy code more effectively
- **Cleaner API**: More maintainable than the traditional C API
- **PyPy performance**: Enables better JIT optimization on PyPy and GraalPy
- **Explicit memory management**: HPy handles prevent reference counting errors

## HPy-Specific Features

This implementation showcases several HPy features:

1. **HPyField**: Proper field storage with GC integration
2. **HPy handles**: Automatic reference counting via HPy context
3. **Universal module init**: `HPy_MODINIT` macro for cross-platform support
4. **Type specification**: Modern `HPyType_Spec` API instead of legacy `PyTypeObject`
5. **Zero-copy NumPy**: Direct memory access via `__array_interface__`

## Comparison with Other Bindings

| Feature | HPy | Cython | pybind11 | SWIG | ctypes | cffi |
|---------|-----|--------|----------|------|--------|------|
| Universal ABI | ✅ | ❌ | ❌ | ❌ | N/A | N/A |
| PyPy-friendly | ✅ | ⚠️ | ❌ | ⚠️ | ✅ | ✅ |
| Compile-time | C | Cython | C++ | C/C++ | Runtime | Hybrid |
| Maturity | New | Mature | Mature | Legacy | Mature | Mature |
| Learning curve | Medium | Medium | Low | High | Low | Medium |

## Future Development

HPy is actively developed and represents a promising direction for Python extensions:

- Growing ecosystem and tooling support
- Improved debugging capabilities
- Better integration with alternative Python implementations
- Ongoing performance optimizations

## Resources

- [HPy Website](https://hpyproject.org/)
- [HPy Documentation](https://docs.hpyproject.org/)
- [HPy GitHub](https://github.com/hpyproject/hpy)
- [HPy Blog](https://hpyproject.org/blog/)
