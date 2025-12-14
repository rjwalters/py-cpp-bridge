# SWIG Processor

Python-to-C++ binding implementation using [SWIG](http://www.swig.org/) (Simplified Wrapper and Interface Generator).

## Files

| File | Description |
|------|-------------|
| `swig_processor.i` | SWIG interface definition file |
| `py_wrapper.py` | Python wrapper around SWIG-generated bindings |
| `swig_processor.pyi` | Python stub file for type hints |
| `__init__.py` | Package initializer |

## Overview

SWIG uses a declarative interface file (`.i`) to generate wrapper code. Unlike other approaches, SWIG can generate bindings for multiple languages from a single interface definition.

## API

```python
from swig_processor import PyArrayProcessor

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

## Architecture

This implementation uses a two-layer approach:

1. **SWIG Layer** (`swig_processor.i`) - Generates low-level C++/Python bindings
2. **Python Wrapper** (`py_wrapper.py`) - Provides the high-level API matching other implementations

## How It Works

1. SWIG reads the `.i` interface file
2. Generates `_swig_processor_impl` C++ wrapper code
3. The Python wrapper adds NumPy handling and a clean API

## Advantages

- **Multi-language support** - Same interface can target Python, Java, Ruby, etc.
- **Declarative approach** - Interface file describes what to wrap
- **Mature and stable** - Long history of production use
- **Automatic exception translation** - C++ exceptions convert to Python
