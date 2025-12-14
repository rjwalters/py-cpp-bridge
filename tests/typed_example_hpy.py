#!/usr/bin/env python3
"""
Typed example demonstrating HPy processor usage with proper type hints.
"""

import numpy as np
import numpy.typing as npt
from hpy_processor import PyArrayProcessor


def demonstrate_hpy_patterns() -> None:
    """Demonstrate all three HPy memory handling patterns."""

    # Get the correct NumPy dtype from C++
    np_type = np.dtype(PyArrayProcessor.get_numpy_type_name("value"))
    print(f"Using NumPy type: {np_type}")

    # Create test data
    size = 5
    data: npt.NDArray[np.float32] = np.array([1, 2, 3, 4, 5], dtype=np_type)

    # Create processor
    processor = PyArrayProcessor(size)

    print("\n" + "=" * 60)
    print("HPy Processor - Three Memory Handling Patterns")
    print("=" * 60)

    # Pattern 1: Pre-allocated buffer (most efficient)
    print("\n1. Pre-allocated Buffer (Most Efficient)")
    print("-" * 60)
    result1: npt.NDArray[np.float32] = processor.process_preallocated(data)
    print(f"Input:  {data}")
    print(f"Output: {result1}")
    print("Note: Same buffer reused on repeated calls")

    # Verify reuse
    result1_again = processor.process_preallocated(data)
    print(f"Same buffer? {result1 is result1_again}")

    # Pattern 2: New contiguous array (flexible)
    print("\n2. New Contiguous Array (Flexible)")
    print("-" * 60)
    mixed_input = [10, 20, 30, 40, 50]
    result2: npt.NDArray[np.float32] = processor.process_new(mixed_input)
    print(f"Input:  {mixed_input} (list)")
    print(f"Output: {result2}")
    print("Note: Accepts lists, tuples, or arrays")

    # Pattern 3: Manual casting (maximum control)
    print("\n3. Manual Casting (Maximum Control)")
    print("-" * 60)
    float64_data: npt.NDArray[np.float64] = np.array([5.5, 15.5, 25.5, 35.5, 45.5], dtype=np.float64)
    result3: npt.NDArray[np.float32] = processor.process_manual(float64_data)
    print(f"Input:  {float64_data} (float64)")
    print(f"Output: {result3} (converted to float32)")
    print("Note: Manual element-by-element conversion")

    # Advanced: Test with different dtypes
    print("\n" + "=" * 60)
    print("Advanced: Type Conversion Testing")
    print("=" * 60)

    for dtype in [np.int32, np.int64, np.float32, np.float64]:
        test_data = np.array([1, 2, 3, 4, 5], dtype=dtype)
        result = processor.process_manual(test_data)
        print(f"{dtype.__name__:8} -> float32: {result}")

    # Clean up
    processor.close()
    print("\n" + "=" * 60)
    print("Demo complete - processor cleaned up")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_hpy_patterns()
