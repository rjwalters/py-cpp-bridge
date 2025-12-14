"""
Python wrapper for cffi-based ArrayProcessor bindings.

This module provides a PyArrayProcessor class that mirrors the API of the
pybind11, Cython, and SWIG implementations, offering three different approaches
to memory handling and type conversion.
"""

import numpy as np
import numpy.typing as npt

try:
    from . import _cffi_impl as _impl
except ImportError:
    # If the module is not built yet, provide a helpful error message
    raise ImportError(
        "cffi_processor extension not built. Please run 'python src/cffi_processor/build_cffi.py' first."
    )


class PyArrayProcessor:
    """
    Python wrapper for C++ ArrayProcessor using cffi.

    This class provides methods for processing arrays of numpy values
    using different memory and type handling approaches.
    """

    def __init__(self, size: int) -> None:
        """
        Initialize the ArrayProcessor with a specified size.

        Args:
            size: The size of arrays this processor will handle

        Raises:
            RuntimeError: If allocation fails
        """
        self._ffi = _impl.ffi
        self._lib = _impl.lib
        self._processor = self._lib.array_processor_new(size)
        if self._processor == self._ffi.NULL:
            raise RuntimeError("Failed to create ArrayProcessor")
        self._size = size
        # Pre-allocate the results buffer for method 1
        self._results_buffer = np.zeros(size, dtype=np.float32)

    def __del__(self):
        """Clean up the C++ ArrayProcessor when the Python object is destroyed."""
        if hasattr(self, '_processor') and self._processor != self._ffi.NULL:
            self._lib.array_processor_delete(self._processor)

    def process_preallocated(self, np_values: npt.NDArray) -> npt.NDArray[np.float32]:
        """
        Process values using a pre-allocated buffer.

        This method reuses the same results array for each call, making it
        efficient for repeated calls on values of the same size.

        Pros:
            - Efficient for repeated calls as it reuses the results buffer
            - Minimizes memory allocations
            - Clear type safety with explicit array typing

        Cons:
            - The results array is tied to the object's lifetime
            - Less flexible if result size might change
            - Could lead to issues if the view is modified externally

        Best for:
            High-performance code with frequent processing calls on
            similarly sized values

        Args:
            np_values: Input array of values. Must be 1D with length matching size.

        Returns:
            Array of processed values (doubled input values)

        Raises:
            ValueError: If values length doesn't match expected size
            TypeError: If input is not array-like
        """
        # Validate input
        arr = np.asarray(np_values)
        if arr.ndim != 1:
            raise ValueError("Expected 1-dimensional array")
        if arr.shape[0] != self._size:
            raise ValueError(
                f"Expected array of size {self._size}, got {arr.shape[0]}"
            )

        # Ensure correct dtype and contiguous
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        arr = np.ascontiguousarray(arr)

        # Get pointer to the data and process
        data_ptr = self._ffi.cast("float*", arr.ctypes.data)
        self._lib.array_processor_process(self._processor, data_ptr, self._size)

        # Get results pointer and copy to pre-allocated buffer
        results_ptr = self._lib.array_processor_get_results(self._processor)
        if results_ptr == self._ffi.NULL:
            raise RuntimeError("Failed to get results")

        # Create a NumPy array from the C pointer and copy to buffer
        results_view = np.frombuffer(
            self._ffi.buffer(results_ptr, self._size * self._ffi.sizeof("float")),
            dtype=np.float32
        )
        np.copyto(self._results_buffer, results_view)
        return self._results_buffer

    def process_new(self, np_values: npt.NDArray) -> npt.NDArray[np.float32]:
        """
        Process values by creating a new contiguous array.

        This method creates a new array for each call, providing more flexibility
        in handling different input types.

        Pros:
            - More flexible input handling (auto-converts types)
            - Clean separation between input and output
            - Each result is independent

        Cons:
            - More memory allocations
            - Slightly more overhead for type conversion

        Best for:
            General-purpose use where convenience is valued over ultimate performance

        Args:
            np_values: Input values that can be converted to a NumPy array

        Returns:
            New array of processed values (doubled input values)

        Raises:
            ValueError: If values length doesn't match expected size
        """
        # Convert to contiguous float32 array (forcecast behavior)
        arr = np.ascontiguousarray(np_values, dtype=np.float32)

        if arr.ndim != 1:
            raise ValueError("Expected 1-dimensional array")
        if arr.shape[0] != self._size:
            raise ValueError(
                f"Expected array of size {self._size}, got {arr.shape[0]}"
            )

        # Get pointer to the data and process
        data_ptr = self._ffi.cast("float*", arr.ctypes.data)
        self._lib.array_processor_process(self._processor, data_ptr, self._size)

        # Get results pointer and create a new array
        results_ptr = self._lib.array_processor_get_results(self._processor)
        if results_ptr == self._ffi.NULL:
            raise RuntimeError("Failed to get results")

        # Create a new NumPy array from the C pointer
        results_view = np.frombuffer(
            self._ffi.buffer(results_ptr, self._size * self._ffi.sizeof("float")),
            dtype=np.float32
        )
        # Return a copy to ensure independence
        return results_view.copy()

    def process_manual(self, np_values: npt.NDArray) -> npt.NDArray[np.float32]:
        """
        Process values with manual copying and casting.

        This method provides explicit control over type conversion by manually
        copying and casting each element. Accepts any numeric array type.

        Pros:
            - Maximum control over values conversion
            - Can handle arrays of any numeric type
            - Can perform validation or transformation during copying

        Cons:
            - Most verbose approach
            - Extra copying step adds overhead
            - Not needed for many simple cases

        Best for:
            Cases where precise control over type conversion is needed

        Args:
            np_values: Input array of any numeric type

        Returns:
            New array of processed values (doubled input values)
        """
        arr = np.asarray(np_values)

        if arr.ndim != 1:
            raise ValueError("Expected 1-dimensional array")

        # Determine copy size (use smaller of input size and processor size)
        input_size = arr.shape[0]
        copy_size = min(input_size, self._size)

        # Create a buffer for processing with explicit casting
        buffer = np.zeros(self._size, dtype=np.float32)

        # Manual copy with explicit casting based on input dtype
        for i in range(copy_size):
            buffer[i] = np.float32(arr[i])

        # Get pointer to the buffer and process
        data_ptr = self._ffi.cast("float*", buffer.ctypes.data)
        self._lib.array_processor_process(self._processor, data_ptr, self._size)

        # Get results pointer and create a new array
        results_ptr = self._lib.array_processor_get_results(self._processor)
        if results_ptr == self._ffi.NULL:
            raise RuntimeError("Failed to get results")

        # Create a new NumPy array from the C pointer
        results_view = np.frombuffer(
            self._ffi.buffer(results_ptr, self._size * self._ffi.sizeof("float")),
            dtype=np.float32
        )
        return results_view.copy()

    @staticmethod
    def get_numpy_type_name(type_id: str) -> str:
        """
        Get the NumPy type name for a given type identifier.

        Args:
            type_id: The type identifier (e.g., "value")

        Returns:
            The corresponding NumPy type name (e.g., "float32")
        """
        result = _impl.lib.get_numpy_type_name_wrapper(type_id.encode('utf-8'))
        return _impl.ffi.string(result).decode('utf-8')
