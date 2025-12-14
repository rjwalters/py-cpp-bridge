"""Type stubs for hpy_processor module."""

from typing import Union
import numpy as np
import numpy.typing as npt

class PyArrayProcessor:
    """
    Array processor using HPy bindings to C++.

    This class demonstrates three different memory handling patterns:
    1. process_preallocated: Pre-allocated buffer (most efficient)
    2. process_new: New array each time (flexible)
    3. process_manual: Manual element conversion (maximum control)
    """

    def __init__(self, size: int) -> None:
        """
        Initialize the array processor.

        Args:
            size: The size of arrays to process
        """
        ...

    def process_preallocated(
        self, np_values: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Process array using pre-allocated buffer (most efficient).

        The processor maintains an internal results buffer that is reused
        across calls, minimizing memory allocations.

        Args:
            np_values: Input array of float32 values

        Returns:
            Array with processed values (reused buffer)

        Raises:
            ValueError: If input size doesn't match processor size
        """
        ...

    def process_new(
        self, np_values: Union[npt.NDArray, list, tuple]
    ) -> npt.NDArray[np.float32]:
        """
        Process array creating new result array each time (flexible).

        Accepts various input types and automatically converts them to
        the correct dtype. Creates a new result array for each call.

        Args:
            np_values: Input values (array, list, or tuple)

        Returns:
            New array with processed values

        Raises:
            ValueError: If input size doesn't match processor size
        """
        ...

    def process_manual(self, np_values: npt.NDArray) -> npt.NDArray[np.float32]:
        """
        Process array with manual element conversion (maximum control).

        Manually converts each element to float32, providing maximum
        control over the conversion process.

        Args:
            np_values: Input array of any numeric type

        Returns:
            New array with processed values

        Raises:
            ValueError: If input size doesn't match processor size
        """
        ...

    def close(self) -> None:
        """Clean up resources held by the processor."""
        ...

    @staticmethod
    def get_numpy_type_name(type_id: str) -> str:
        """
        Get NumPy type name for a given type ID.

        Args:
            type_id: Type identifier (e.g., "value")

        Returns:
            NumPy dtype name (e.g., "float32")
        """
        ...
