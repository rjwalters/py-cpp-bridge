# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

"""
Python wrapper for C++ ArrayProcessor.

This module provides three different approaches for passing NumPy arrays to C++:
1. Pre-allocated buffer: Reuses the same results array for efficiency
2. New contiguous array: Creates a fresh array each time for flexibility
3. Manual casting: Explicitly controls type conversion for maximum control
"""

import numpy as np
cimport numpy as cnp
from libc.stddef cimport size_t

# Import from the .pxd file
from cython_processor cimport (
    ArrayProcessor,
    c_value_type, 
    NUMPY_VALUE_TYPE,
    get_numpy_type_name as cpp_get_numpy_type_name
)

# Define numpy value type using the C++ function
np_value_type = cpp_get_numpy_type_name(b"value")

# Initialize NumPy
cnp.import_array()

cdef class PyArrayProcessor:
    """
    Python wrapper for C++ ArrayProcessor.
    
    This class provides methods for processing arrays of numpy values
    using different memory and type handling approaches.
    """
    cdef ArrayProcessor* _cpp_processor
    cdef cnp.ndarray _results_view
    
    def __cinit__(self, size_t size):
        """
        Initialize the ArrayProcessor with a specified size.
        
        Args:
            size: The size of arrays this processor will handle
        """
        self._cpp_processor = new ArrayProcessor(<c_value_type>size)
        # Create a view of the results buffer
        self._results_view = None
    
    def __dealloc__(self):
        """
        Clean up C++ resources when the Python object is destroyed.
        """
        if self._cpp_processor != NULL:
            del self._cpp_processor
            self._cpp_processor = NULL
        # NumPy array view will be cleaned up automatically
    
    # Method 1: Using a pre-allocated buffer (reused for each call)
    def process_preallocated(self, cnp.ndarray[c_value_type, ndim=1] np_values):
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
            ValueError: If values length doesn't match expected size
        """
        # Get sizes - cast to avoid sign comparison warnings
        cdef size_t cpp_size = self._cpp_processor.get_size()
        cdef size_t np_values_size
        
        # Safely handle arrays of any size
        if np_values.shape[0] < 0:  # Should never happen, but just in case
            np_values_size = 0
        else:
            np_values_size = <size_t>np_values.shape[0]
        
        if np_values_size != cpp_size:
            raise ValueError(f"Expected array of size {cpp_size}, got {np_values_size}")
        
        # Process the array
        self._cpp_processor.process_array(<c_value_type*>np_values.data, cpp_size)
        
        # Create a NumPy view of the results if needed
        if self._results_view is None:
            self._results_view = np.zeros(cpp_size, dtype=np_value_type)
        
        # Copy results to the view
        cdef size_t i
        cdef c_value_type* results = self._cpp_processor.get_results()
        for i in range(cpp_size):
            self._results_view[i] = results[i]
        
        return self._results_view
    
    # Method 2: Creating a new contiguous array for each call
    def process_new(self, np_values):
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
            ValueError: If values length doesn't match expected size
        """
        cdef cnp.ndarray[c_value_type, ndim=1] np_values_array
        
        # Ensure np_values is a contiguous array of the right type and shape
        np_values_array = np.ascontiguousarray(np_values, dtype=np_value_type)
        
        # Get sizes - cast to avoid sign comparison warnings
        cdef size_t cpp_size = self._cpp_processor.get_size()
        cdef size_t np_values_size
        
        # Safely handle arrays of any size
        if np_values_array.shape[0] < 0:  # Should never happen, but just in case
            np_values_size = 0
        else:
            np_values_size = <size_t>np_values_array.shape[0]
            
        if np_values_size != cpp_size:
            raise ValueError(f"Expected array of size {cpp_size}, got {np_values_size}")
        
        # Process the array
        self._cpp_processor.process_array(<c_value_type*>np_values_array.data, cpp_size)
        
        # Create a NumPy view of the results
        cdef cnp.ndarray[c_value_type, ndim=1] results_array = np.zeros(cpp_size, dtype=np_value_type)
        
        # Copy results to the array
        cdef size_t i
        cdef c_value_type* results = self._cpp_processor.get_results()
        for i in range(cpp_size):
            results_array[i] = results[i]
        
        return results_array
    
    # Method 3: Taking a buffer and manual casting (closest to your step function)
    def process_manual(self, cnp.ndarray np_values):
        """
        Process values with manual copying and casting.
        
        This method provides explicit control over type conversion by manually
        copying and casting each element.
        
        Pros:
            * Maximum control over values conversion
            * Can handle arrays of any type
            * Can perform validation or transformation during copying
            
        Cons:
            * Most verbose approach
            * Extra copying step adds overhead
            * Not needed for many simple cases
            
        Best for:
            Cases where precise control over type conversion is needed
            
        Args:
            np_values: Input array of any type
            
        Returns:
            New array of processed values (doubled input values)
        """
        cdef:
            size_t cpp_size = self._cpp_processor.get_size()
            size_t i
            size_t copy_size
            cnp.ndarray[c_value_type, ndim=1] buffer = np.zeros(cpp_size, dtype=np_value_type)
        
        # Safely get the minimum size for copying
        if np_values.shape[0] < 0:  # Should never happen, but just in case
            copy_size = 0
        else:
            copy_size = <size_t>np_values.shape[0]
            
        # Use the smaller of the two sizes
        if copy_size > cpp_size:
            copy_size = cpp_size
        
        # Manual copy with explicit casting
        for i in range(copy_size):
            buffer[i] = <c_value_type>np_values[i]
        
        # Process the buffer
        self._cpp_processor.process_array(<c_value_type*>buffer.data, cpp_size)
        
        # Create results array
        cdef cnp.ndarray[c_value_type, ndim=1] results = np.zeros(cpp_size, dtype=np_value_type)
        
        # Copy results
        cdef c_value_type* results_ptr = self._cpp_processor.get_results()
        for i in range(cpp_size):
            results[i] = results_ptr[i]
        
        return results

    @classmethod
    def get_numpy_type_name(cls, str type_id):
        py_bytes = type_id.encode('utf8')
        return cpp_get_numpy_type_name(<const char*>py_bytes)
