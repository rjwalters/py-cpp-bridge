# Py-Cpp-Bridge

A demonstration project showcasing efficient interoperability between Python and C++ using both **Cython** and **pybind11**, with single source of truth for data types.

## Overview

This project demonstrates how to integrate high-performance C++ code with Python using two popular binding technologies: **Cython** and **pybind11**. Both implementations feature three different approaches to memory handling and type conversion, enabling you to choose the right solution for your specific use case and compare the two binding approaches.

The core functionality is a simple array processor that doubles each value in an array, but the techniques demonstrated can be applied to any C++ code you want to make available in Python.

## Features

- ✅ Bidirectional data transfer between Python and C++
- ✅ Three different methods for handling memory and type conversion:
  - Pre-allocated buffer (maximum efficiency for repeated calls)
  - New contiguous array (flexibility with clean separation)
  - Manual casting (precise control over type conversion)
- ✅ Single source of truth for data types between C++ and Python
- ✅ Full type annotation support with PEP 561 stub files
- ✅ Debugging and production build configurations
- ✅ Comprehensive documentation for each method

## Requirements

- Python 3.11+
- NumPy 2.x
- Cython 3.0+
- pybind11 3.0+
- A C++ compiler (gcc, clang, MSVC)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/py-cpp-bridge.git
cd py-cpp-bridge

# Build and install the package
make install
```

## Usage

Both Cython and pybind11 implementations provide the same API, so you can easily switch between them.

### Basic Example (Cython)

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

### Basic Example (pybind11)

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

### Type-Annotated Example

The package includes full type annotations for better IDE integration and static type checking:

```python
from typing import List
import numpy as np
import numpy.typing as npt
from cython_processor import PyArrayProcessor

def process_batches(data_batches: List[npt.NDArray], batch_size: int) -> List[npt.NDArray]:
    processor = PyArrayProcessor(batch_size)
    
    results: List[npt.NDArray] = []
    for batch in data_batches:
        # Type checkers understand this returns NDArray
        processed = processor.process_preallocated(batch)
        results.append(processed)
    
    return results
```

## Single Source of Truth for Types

This project demonstrates how to maintain a single source of truth for data types between C++ and Python. The approach works as follows:

1. Core type definitions are placed in a central C++ header file
2. Cython declarations in `.pxd` import these types
3. Type information is exposed to Python through a dedicated function
4. The same types are used consistently in both C++ and Python code

### Benefits:
- Changes to a type only need to be made in one place
- Automatic conversion between C++ and NumPy types
- Type safety between languages
- Self-documenting code with explicit type mappings

## Methods Explained

### Method 1: Pre-allocated Buffer

```python
result = processor.process_preallocated(data)
```

**Pros:**
- Efficient for repeated calls (reuses the same buffer)
- Minimizes memory allocations
- Clear type safety with explicit array typing

**Best for:** High-performance code with frequent processing calls on similarly sized data

### Method 2: New Contiguous Array

```python
result = processor.process_new(data)
```

**Pros:**
- More flexible input handling (auto-converts types)
- Clean separation between input and output
- Each result is independent

**Best for:** General-purpose use where convenience is valued over ultimate performance

### Method 3: Manual Casting

```python
result = processor.process_manual(data)
```

**Pros:**
- Maximum control over data conversion
- Can handle arrays of any type
- Can perform validation or transformation during copying

**Best for:** Cases where precise control over type conversion is needed

## Development

### Build Modes

The project supports both debug and production builds:

```bash
# Production build (optimized)
make build

# Debug build (with bounds checking, etc.)
make debug
```

### Running Tests

```bash
make test
```

### Code Formatting

```bash
make format
```

## Project Structure

```
├── src/
│   ├── common/
│   │   ├── cpp_processor.cpp    # C++ implementation
│   │   ├── cpp_processor.hpp    # C++ header
│   │   └── types.hpp            # Shared type definitions
│   ├── cython_processor/
│   │   ├── cython_processor.pxd # Cython declarations
│   │   ├── cython_processor.pyx # Cython implementation
│   │   └── cython_processor.pyi # Type stubs for IDE support
│   └── pybind_processor/
│       ├── pybind_processor.cpp # pybind11 implementation
│       └── pybind_processor.pyi # Type stubs for IDE support
├── tests/
│   ├── test.py                  # Cython test script
│   ├── test_pybind.py           # pybind11 test script
│   ├── typed_example.py         # Cython typed example
│   └── typed_example_pybind.py  # pybind11 typed example
├── Makefile                     # Build system
├── setup.py                     # Python package configuration
└── README.md                    # This file
```

## How It Works

1. C++ code in the `common` directory defines the core functionality
2. Type definitions in `types.hpp` are shared between C++ and both Python bindings
3. Both Cython and pybind11 create Python wrappers around the C++ `ArrayProcessor` class
4. `setup.py` configures the build process for both extensions with debug/release options
5. The Makefile provides convenient commands for building and testing both implementations

### Cython vs pybind11

| Aspect | Cython | pybind11 |
|--------|--------|----------|
| Language | Python-like (.pyx) | C++ |
| Compilation | Python → C → Binary | C++ → Binary |
| NumPy integration | Via typed memoryviews | Via py::array_t |
| Type safety | Compile-time via cdef | Template-based |
| Debug output | Annotated HTML available | Standard C++ debugging |

## Debugging Features

When built in debug mode (`make debug`), the following features are enabled:

- Bounds checking for array access
- Initialization checks for memory views
- Overflow checking for arithmetic operations
- Line tracing for coverage tools
- Profiling support
- Annotated HTML output showing the Cython to C conversion

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.