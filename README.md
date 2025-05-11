# Py-Cpp-Bridge

A demonstration project showcasing efficient Cython-based interoperability between Python and C++.

## Overview

This project demonstrates how to integrate high-performance C++ code with Python using Cython. It features three different approaches to memory handling and type conversion, enabling you to choose the right solution for your specific use case.

The core functionality is a simple array processor that doubles each value in a uint8 array, but the techniques demonstrated can be applied to any C++ code you want to make available in Python.

## Features

- ✅ Bidirectional data transfer between Python and C++
- ✅ Three different methods for handling memory and type conversion:
  - Pre-allocated buffer (maximum efficiency for repeated calls)
  - New contiguous array (flexibility with clean separation)
  - Manual casting (precise control over type conversion)
- ✅ Full type annotation support with PEP 561 stub files
- ✅ Debugging and production build configurations
- ✅ Comprehensive documentation for each method

## Requirements

- Python 3.7+
- NumPy
- Cython 3.0+
- A C++ compiler (gcc, clang, MSVC)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/py-cpp-bridge.git
cd py-cpp-bridge

# Install dependencies
pip install -r requirements.txt

# Build and install the package
make
```

## Usage

### Basic Example

```python
import numpy as np
from cython_processor import PyArrayProcessor

# Create an array processor for size 5 arrays
processor = PyArrayProcessor(5)

# Create some test data
data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)

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

### Type-Annotated Example

The package includes full type annotations for better IDE integration and static type checking:

```python
from typing import List
import numpy as np
import numpy.typing as npt
from cython_processor import PyArrayProcessor

def process_batches(data_batches: List[npt.NDArray[np.uint8]], batch_size: int) -> List[npt.NDArray[np.uint8]]:
    processor = PyArrayProcessor(batch_size)
    
    results: List[npt.NDArray[np.uint8]] = []
    for batch in data_batches:
        # Type checkers understand this returns NDArray[np.uint8]
        processed = processor.process_preallocated(batch)
        results.append(processed)
    
    return results
```

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

### Type-Checked Example

```bash
make typed-example
```

### Code Formatting

```bash
make format
```

## Project Structure

```
├── cpp_processor.cpp      # C++ implementation
├── cpp_processor.hpp      # C++ header
├── cython_processor.pxd   # Cython declarations for C++ interface
├── cython_processor.pyi   # Python type stubs (PEP 561)
├── cython_processor.pyx   # Cython implementation
├── Makefile               # Build system
├── requirements.txt       # Python dependencies
├── setup.py               # Python package build configuration
├── test.py                # Simple test script
└── typed_example.py       # Example with type annotations
```

## How It Works

1. C++ code (`cpp_processor.hpp/cpp`) defines an `ArrayProcessor` class that handles uint8 arrays
2. Cython declarations (`.pxd`) expose the C++ class to Cython
3. Cython implementation (`.pyx`) creates a Python wrapper class `PyArrayProcessor`
4. Type stubs (`.pyi`) provide type hints for Python IDEs and type checkers
5. `setup.py` configures the build process

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.