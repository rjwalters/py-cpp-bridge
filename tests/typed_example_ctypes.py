from typing import List

import numpy as np
import numpy.typing as npt

from ctypes_processor import PyArrayProcessor

# Rebuild the NumPy types using the exposed function
np_values_type = np.dtype(PyArrayProcessor.get_numpy_type_name("value"))


def process_batches(
    data_batches: List[npt.NDArray[np.float32]], batch_size: int
) -> List[npt.NDArray[np.float32]]:
    """
    Process multiple batches of data using the ctypes ArrayProcessor.

    Args:
        data_batches: List of NumPy arrays to process
        batch_size: Size of each batch

    Returns:
        List of processed NumPy arrays
    """
    processor = PyArrayProcessor(batch_size)

    results: List[npt.NDArray[np.float32]] = []
    for batch in data_batches:
        # Type checkers understand this returns NDArray[np.float32]
        processed = processor.process_preallocated(batch)
        results.append(processed)

    return results


def main():
    # Create test batches
    batch_size = 5
    batches = [
        np.array([1, 2, 3, 4, 5], dtype=np_values_type),
        np.array([10, 20, 30, 40, 50], dtype=np_values_type),
        np.array([100, 200, 300, 400, 500], dtype=np_values_type),
    ]

    # Process batches
    results = process_batches(batches, batch_size)

    # Display results
    print("Batch processing with type annotations:")
    for i, (input_batch, output_batch) in enumerate(zip(batches, results)):
        print(f"Batch {i + 1}:")
        print(f"  Input:  {input_batch}")
        print(f"  Output: {output_batch}")


if __name__ == "__main__":
    main()
