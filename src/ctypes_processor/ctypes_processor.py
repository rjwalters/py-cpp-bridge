"""
ctypes wrapper for ArrayProcessor C++ class.

This module demonstrates using Python's built-in ctypes library to interface
with C++ code through a C wrapper layer.
"""

import ctypes
import os
import platform
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt


def _find_library() -> Path:
    """Find the ctypes_processor shared library."""
    # Start from the module directory
    module_dir = Path(__file__).parent

    # Try common library locations
    if platform.system() == "Darwin":
        lib_names = ["libctypes_processor.dylib", "ctypes_processor.dylib"]
    elif platform.system() == "Windows":
        lib_names = ["ctypes_processor.dll", "libctypes_processor.dll"]
    else:  # Linux
        lib_names = ["libctypes_processor.so", "ctypes_processor.so"]

    # Search in multiple locations
    build_dir = module_dir.parent.parent / "build"
    search_paths = [
        module_dir,  # Installed location (site-packages/ctypes_processor)
        build_dir / "ctypes_processor",  # Direct build location
        build_dir / "lib",
        build_dir,
    ]

    # Also search in platform-specific build subdirectories
    if build_dir.exists():
        for subdir in build_dir.iterdir():
            if subdir.is_dir():
                search_paths.append(subdir / "ctypes_processor")
                search_paths.append(subdir / "lib")
                search_paths.append(subdir)

    # Also check site-packages if installed
    try:
        import site
        for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
            if site_dir:
                site_path = Path(site_dir) / "ctypes_processor"
                if site_path.exists():
                    search_paths.insert(1, site_path)
    except:
        pass

    for search_path in search_paths:
        for lib_name in lib_names:
            lib_path = search_path / lib_name
            if lib_path.exists():
                return lib_path

    raise FileNotFoundError(
        f"Could not find ctypes_processor shared library. Searched in: {search_paths}"
    )


# Load the shared library
_lib_path = _find_library()
_lib = ctypes.CDLL(str(_lib_path))

# Define function signatures
_lib.array_processor_create.argtypes = [ctypes.c_size_t]
_lib.array_processor_create.restype = ctypes.c_void_p

_lib.array_processor_destroy.argtypes = [ctypes.c_void_p]
_lib.array_processor_destroy.restype = None

_lib.array_processor_process.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
_lib.array_processor_process.restype = None

_lib.array_processor_get_results.argtypes = [ctypes.c_void_p]
_lib.array_processor_get_results.restype = ctypes.POINTER(ctypes.c_float)

_lib.array_processor_get_size.argtypes = [ctypes.c_void_p]
_lib.array_processor_get_size.restype = ctypes.c_size_t

_lib.get_numpy_type_name_c.argtypes = [ctypes.c_char_p]
_lib.get_numpy_type_name_c.restype = ctypes.c_char_p


class PyArrayProcessor:
    """
    Python wrapper for the C++ ArrayProcessor class using ctypes.

    This class demonstrates three different approaches to handling memory
    and type conversion between Python and C++.
    """

    def __init__(self, size: int):
        """
        Create a new ArrayProcessor instance.

        Args:
            size: The size of arrays this processor will handle
        """
        self._handle = _lib.array_processor_create(size)
        if not self._handle:
            raise RuntimeError("Failed to create ArrayProcessor")
        self._size = size

    def __del__(self):
        """Clean up the C++ object when this Python object is destroyed."""
        if hasattr(self, "_handle") and self._handle:
            _lib.array_processor_destroy(self._handle)
            self._handle = None

    def process_preallocated(
        self, data: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Process array using pre-allocated buffer (Method 1).

        This method uses the internal buffer allocated during construction,
        making it the most efficient for repeated calls with same-sized arrays.

        Args:
            data: Input NumPy array of float32 values

        Returns:
            NumPy array containing the processed results
        """
        # Validate input
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a NumPy array")

        if len(data) != self._size:
            raise ValueError(
                f"Array size mismatch: expected {self._size}, got {len(data)}"
            )

        # Ensure the array is contiguous and float32
        data_contiguous = np.ascontiguousarray(data, dtype=np.float32)

        # Get pointer to data
        data_ptr = data_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Process the array
        _lib.array_processor_process(self._handle, data_ptr, self._size)

        # Get results pointer
        results_ptr = _lib.array_processor_get_results(self._handle)

        # Create NumPy array from results (copy data)
        result = np.ctypeslib.as_array(results_ptr, shape=(self._size,)).copy()

        return result

    def process_new(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Process array creating a new contiguous array (Method 2).

        This method creates a fresh output array for each call, providing
        clean separation between inputs and outputs.

        Args:
            data: Input NumPy array of float32 values

        Returns:
            New NumPy array containing the processed results
        """
        # This method is similar to preallocated for ctypes,
        # as we always create a new result array
        return self.process_preallocated(data)

    def process_manual(self, data: npt.NDArray) -> npt.NDArray[np.float32]:
        """
        Process array with manual type casting (Method 3).

        This method provides maximum control over type conversion,
        explicitly casting input to the correct type.

        Args:
            data: Input NumPy array (will be cast to float32)

        Returns:
            NumPy array containing the processed results
        """
        # Validate input
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a NumPy array")

        if len(data) != self._size:
            raise ValueError(
                f"Array size mismatch: expected {self._size}, got {len(data)}"
            )

        # Manual casting to float32
        data_casted = data.astype(np.float32, copy=True)

        # Ensure contiguous
        data_contiguous = np.ascontiguousarray(data_casted)

        # Get pointer to data
        data_ptr = data_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Process the array
        _lib.array_processor_process(self._handle, data_ptr, self._size)

        # Get results pointer
        results_ptr = _lib.array_processor_get_results(self._handle)

        # Create NumPy array from results (copy data)
        result = np.ctypeslib.as_array(results_ptr, shape=(self._size,)).copy()

        return result

    @staticmethod
    def get_numpy_type_name(type_id: str) -> str:
        """
        Get the NumPy type name for a given type identifier.

        This maintains the single source of truth for types between C++ and Python.

        Args:
            type_id: Type identifier (e.g., "value")

        Returns:
            NumPy type name (e.g., "float32")
        """
        result = _lib.get_numpy_type_name_c(type_id.encode("utf-8"))
        return result.decode("utf-8")
