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

## API

```python
from cython_processor import PyArrayProcessor

processor = PyArrayProcessor(size=1000)

# Three processing methods:
result = processor.process_preallocated(input_array)  # Reuses buffer
result = processor.process_new(input_array)           # Creates new array
result = processor.process_manual(input_array)        # Manual copying
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
