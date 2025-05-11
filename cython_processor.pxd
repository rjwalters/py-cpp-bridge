# distutils: language = c++
# cython: language_level=3

from libc.stddef cimport size_t

import numpy as np

# Forward declarations - import directly from types.hpp
cdef extern from "types.hpp":
    # Core type definitions
    ctypedef float c_value_type;

    # Numpy type name constants
    const char* NUMPY_VALUE_TYPE

    # Type mapping function
    const char* get_numpy_type_name(const char* type_id)
    
# Import numpy types for C
np_value_type = np.dtype(get_numpy_type_name("value").decode('utf8'))

# Import C++ ArrayProcessor class
cdef extern from "cpp_processor.hpp":
    cdef cppclass ArrayProcessor:
        ArrayProcessor(c_value_type size) except +
        void process_array(c_value_type* data, size_t size) except +
        c_value_type* get_results() const
        size_t get_size() const
