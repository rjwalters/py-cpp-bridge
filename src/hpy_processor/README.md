# HPy Processor

HPy-based Python/C++ bridge implementation demonstrating the three memory handling patterns.

## About HPy

[HPy](https://hpyproject.org/) is a new universal API for Python extensions designed to:

- Work across multiple Python implementations (CPython, PyPy, GraalPy)
- Enable better optimization by alternative implementations
- Provide a cleaner, more maintainable API than the traditional C API
- Support both the CPython ABI and a universal HPy ABI

## Implementation Overview

This implementation uses HPy's C-based API to wrap the C++ `ArrayProcessor` class through a C wrapper layer (reusing the same C wrapper as the ctypes implementation).

### Key Features

- **Universal ABI**: Can run in either CPython ABI mode or HPy universal mode
- **Cross-implementation**: Works on CPython, PyPy, and GraalPy
- **Zero-copy operations**: Direct memory access to NumPy arrays
- **Three memory patterns**: Demonstrates pre-allocated, new array, and manual conversion approaches

### File Structure

```
src/hpy_processor/
├── hpy_processor.c          # HPy extension implementation
├── __init__.py              # Python module interface
├── hpy_processor.pyi        # Type stubs for IDE support
└── README.md                # This file
```

### Dependencies

- HPy (https://github.com/hpyproject/hpy)
- NumPy
- C compiler
- C++ compiler (for the common C++ implementation)

### Build System

The HPy extension is built using HPy's build utilities integrated with scikit-build-core.

## Usage Example

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

## Memory Handling Patterns

### 1. Pre-allocated Buffer (`process_preallocated`)

```python
result = processor.process_preallocated(data)
```

- Maintains internal results buffer
- Reuses buffer across calls
- Most efficient for repeated operations
- Buffer tied to processor lifetime

### 2. New Contiguous Array (`process_new`)

```python
result = processor.process_new(data)
```

- Converts input to correct dtype automatically
- Creates new results array each call
- Flexible input handling (arrays, lists, tuples)
- Independent results

### 3. Manual Casting (`process_manual`)

```python
result = processor.process_manual(data)
```

- Manual element-by-element conversion
- Maximum control over type conversion
- Handles any numeric input type
- Most flexible but more overhead

## Performance Notes

- **HPy Universal Mode**: May have slight overhead vs. CPython C API, but enables portability
- **PyPy/GraalPy**: HPy can enable better JIT optimizations on alternative implementations
- **Memory Access**: Uses NumPy's `__array_interface__` for direct memory access

## HPy-Specific Features

This implementation showcases several HPy features:

1. **HPyField**: Proper field storage with GC integration
2. **HPy handles**: Automatic reference counting via HPy context
3. **Universal module init**: `HPy_MODINIT` macro for cross-platform support
4. **Type specification**: Modern `HPyType_Spec` API instead of legacy `PyTypeObject`

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
