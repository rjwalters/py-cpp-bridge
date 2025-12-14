# Common C++ Implementation

This directory contains the shared C++ implementation that all binding approaches (Cython, pybind11, nanobind, SWIG) wrap around.

## Files

| File | Description |
|------|-------------|
| `cpp_processor.hpp` | Header file defining the `ArrayProcessor` class |
| `cpp_processor.cpp` | Implementation of the `ArrayProcessor` class |
| `types.hpp` | Type definitions and NumPy type mapping utilities |

## ArrayProcessor Class

The core `ArrayProcessor` class provides array processing functionality:

```cpp
class ArrayProcessor {
public:
    ArrayProcessor(size_t size);
    ~ArrayProcessor();

    void process_array(const c_value_type* input, size_t size);
    const c_value_type* get_results() const;
    size_t get_size() const;
};
```

### Methods

- **Constructor** - Allocates internal buffer of specified size
- **process_array()** - Processes input array (doubles each value)
- **get_results()** - Returns pointer to processed data
- **get_size()** - Returns the processor's buffer size

## Type Definitions

`types.hpp` defines:

- `c_value_type` as `float` (32-bit floating point)
- `NUMPY_VALUE_TYPE` constant set to `"float32"`
- `get_numpy_type_name()` function for C++ to NumPy type mapping

## Usage

This code is not used directly from Python. Instead, it's compiled into the various binding modules which expose it through their respective Python APIs.
