from typing import Union

import numpy as np
import numpy.typing as npt


class PyArrayProcessor:
    """
    Python wrapper for C++ ArrayProcessor using pybind11.

    This class provides methods for processing arrays of values
    using different memory and type handling approaches:

    1. process_preallocated: Reuses the same results array for efficiency
    2. process_new: Creates a fresh array each time for flexibility
    3. process_manual: Explicitly controls type conversion for maximum control
    """

    def __init__(self, size: int) -> None:
        """
        Initialize the ArrayProcessor with a specified size.

        Args:
            size: The size of arrays this processor will handle

        Raises:
            RuntimeError: If allocation fails
        """
        ...

    def process_preallocated(
        self, np_values: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Process values using a pre-allocated buffer.

        This method reuses the same results array for each call, making it
        efficient for repeated calls on values of the same size.

        Pros:
            * Efficient for repeated calls as it reuses the results buffer
            * Minimizes memory allocations
            * Clear type safety with explicit array typing

        Cons:
            * The results array is tied to the object's lifetime
            * Less flexible if result size might change
            * Could lead to issues if the view is modified externally

        Best for:
            High-performance code with frequent processing calls on
            similarly sized values

        Args:
            np_values: Input array of values. Must be 1D with length matching size.

        Returns:
            Array of processed values (doubled input values)

        Raises:
            RuntimeError: If values length doesn't match expected size
        """
        ...

    def process_new(
        self, np_values: Union[npt.NDArray, list, tuple]
    ) -> npt.NDArray[np.float32]:
        """
        Process values by creating a new contiguous array.

        This method creates a new array for each call, providing more flexibility
        in handling different input types.

        Pros:
            * More flexible input handling (auto-converts types)
            * Clean separation between input and output
            * Each result is independent

        Cons:
            * More memory allocations
            * Slightly more overhead for type conversion

        Best for:
            General-purpose use where convenience is valued over ultimate performance

        Args:
            np_values: Input values that can be converted to a NumPy array

        Returns:
            New array of processed values (doubled input values)

        Raises:
            RuntimeError: If values length doesn't match expected size
        """
        ...

    def process_manual(self, np_values: npt.NDArray) -> npt.NDArray[np.float32]:
        """
        Process values with manual copying and casting.

        This method provides explicit control over type conversion by manually
        copying and casting each element.

        Pros:
            * Maximum control over values conversion
            * Can handle arrays of any numeric type
            * Can perform validation or transformation during copying

        Cons:
            * Most verbose approach
            * Extra copying step adds overhead
            * Not needed for many simple cases

        Best for:
            Cases where precise control over type conversion is needed

        Args:
            np_values: Input array of any numeric type

        Returns:
            New array of processed values (doubled input values)

        Raises:
            RuntimeError: If processing fails
        """
        ...

    @staticmethod
    def get_numpy_type_name(type_id: str) -> str:
        """
        Get the NumPy type name for a given type identifier.

        Args:
            type_id: The type identifier (e.g., "value")

        Returns:
            The corresponding NumPy type name (e.g., "float32")
        """
        ...
