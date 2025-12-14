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

## API

```python
from pybind_processor import PyArrayProcessor

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

1. The `PYBIND11_MODULE` macro defines the Python module
2. C++ classes are exposed using `py::class_<>`
3. Methods are bound with `.def()` calls
4. NumPy arrays are handled via `py::array_t<>`

## Advantages

- Header-only: no separate library to link
- Natural C++ syntax for defining bindings
- Excellent documentation and community support
- Automatic type conversions between Python and C++
