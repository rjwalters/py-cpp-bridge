# Cython Processor

Python-to-C++ binding implementation using [Cython](https://cython.org/).

## Files

| File | Description |
|------|-------------|
| `cython_processor.pyx` | Cython implementation with Python-like syntax |
| `cython_processor.pxd` | Cython declaration file (C++ interface definitions) |
| `cython_processor.pyi` | Python stub file for type hints |
| `__init__.py` | Package initializer |

## Overview

Cython is a programming language that makes writing C extensions for Python as easy as Python itself. The `.pyx` file uses Python-like syntax with C type declarations, which Cython compiles into optimized C/C++ code.

## Basic Example

```python
import numpy as np
from cython_processor import PyArrayProcessor

# Get the correct NumPy type
np_values_type = np.dtype(PyArrayProcessor.get_numpy_type_name("value"))

# Create an array processor for size 5 arrays
processor = PyArrayProcessor(5)

# Create some test data using the imported NumPy type
# This ensures type consistency with C++ expectations
data = np.array([1, 2, 3, 4, 5], dtype=np_values_type)

# Method 1: Using pre-allocated buffer (most efficient)
result1 = processor.process_preallocated(data)
print(f"Result 1: {result1}")  # Output: [2, 4, 6, 8, 10]

# Method 2: Creating new contiguous array (most flexible)
result2 = processor.process_new(data)
print(f"Result 2: {result2}")  # Output: [2, 4, 6, 8, 10]

# Method 3: Manual casting (most control)
result3 = processor.process_manual(data)
print(f"Result 3: {result3}")  # Output: [2, 4, 6, 8, 10]
```

## Processing Methods

| Method | Memory Strategy | Best For |
|--------|-----------------|----------|
| `process_preallocated()` | Reuses internal buffer | Repeated calls with same-sized arrays |
| `process_new()` | Allocates fresh array | Flexibility, auto type conversion |
| `process_manual()` | Explicit copying/casting | Maximum control over memory |

## How It Works

1. **Declaration** (`.pxd`) - Declares C++ types and functions for Cython
2. **Implementation** (`.pyx`) - Python-like code with C++ interop
3. **Compilation** - Cython generates C/C++ code, then compiles to a Python extension module

## Advantages

- Python-like syntax familiar to Python developers
- Gradual optimization: start with Python, add types incrementally
- Excellent NumPy integration via typed memoryviews
