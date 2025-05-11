import numpy as np

# Import the compiled Python module (not the Cython declarations)
from py_cpp_bridge.cython_processor import (
    PyArrayProcessor,
    get_numpy_type_name
)

# Rebuild the NumPy types using the exposed function
np_values_type = np.dtype(PyArrayProcessor.get_numpy_type_name("values"))

def main():
    # Create test data
    size = 5
    data = np.array([1, 2, 3, 4, 5], dtype=np_values_type)

    # Create processor
    processor = PyArrayProcessor(size)

    # Test Method 1: Pre-allocated buffer
    print("\n=== Testing Method 1: Pre-allocated buffer ===")
    result1 = processor.process_preallocated(data)
    print(f"Input: {data}")
    print(f"Output: {result1}")

    # Test Method 2: New contiguous array
    print("\n=== Testing Method 2: New contiguous array ===")
    data2 = np.array([10, 20, 30, 40, 50], dtype=np_values_type)
    result2 = processor.process_new(data2)
    print(f"Input: {data2}")
    print(f"Output: {result2}")

    # Test Method 3: Manual casting (closest to your code)
    print("\n=== Testing Method 3: Manual casting ===")
    data3 = np.array([5, 15, 25, 35, 45], dtype=np_values_type)
    result3 = processor.process_manual(data3)
    print(f"Input: {data3}")
    print(f"Output: {result3}")

    # Test with wrong type
    print("\n=== Testing with wrong type ===")
    data4 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    result4 = processor.process_manual(data4)
    print(f"Input (int32): {data4}")
    print(f"Output after casting: {result4}")


if __name__ == "__main__":
    main()
