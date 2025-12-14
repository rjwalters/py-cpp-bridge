/**
 * SWIG interface file for ArrayProcessor.
 *
 * This module provides Python bindings for the C++ ArrayProcessor class
 * using SWIG. NumPy array handling is done in the Python wrapper layer.
 */

%module _swig_processor_impl

%{
#define SWIG_FILE_WITH_INIT
#include "cpp_processor.hpp"
#include "types.hpp"
#include <numpy/arrayobject.h>
#include <stdexcept>
%}

// Exception handling
%exception {
    try {
        $action
    } catch (const std::runtime_error& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        SWIG_fail;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        SWIG_fail;
    }
}

// Initialize NumPy
%init %{
    import_array();
%}

// Wrap the ArrayProcessor class
class ArrayProcessor {
public:
    ArrayProcessor(size_t size);
    ~ArrayProcessor();
    size_t get_size() const;
};

// Extend ArrayProcessor with NumPy-friendly methods
%extend ArrayProcessor {
    PyObject* process_numpy_array(PyObject* input_array) {
        // Ensure input is a NumPy array
        if (!PyArray_Check(input_array)) {
            PyErr_SetString(PyExc_TypeError, "Expected a NumPy array");
            return NULL;
        }

        PyArrayObject* arr = (PyArrayObject*)input_array;

        // Check dimensions
        if (PyArray_NDIM(arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Expected 1-dimensional array");
            return NULL;
        }

        // Get size
        npy_intp input_size = PyArray_DIM(arr, 0);
        size_t expected_size = $self->get_size();

        if ((size_t)input_size != expected_size) {
            PyErr_Format(PyExc_ValueError,
                "Expected array of size %zu, got %zd", expected_size, (ssize_t)input_size);
            return NULL;
        }

        // Convert to contiguous float32 array
        PyArrayObject* contiguous = (PyArrayObject*)PyArray_GETCONTIGUOUS(arr);
        if (contiguous == NULL) {
            return NULL;
        }

        // Cast to float32 if needed
        PyArrayObject* float_arr = (PyArrayObject*)PyArray_Cast(contiguous, NPY_FLOAT);
        Py_DECREF(contiguous);
        if (float_arr == NULL) {
            return NULL;
        }

        // Get data pointer
        c_value_type* data = (c_value_type*)PyArray_DATA(float_arr);

        // Process the array
        $self->process_array(data, expected_size);

        // Get results and create output array
        c_value_type* results = $self->get_results();

        npy_intp dims[1] = {(npy_intp)expected_size};
        PyObject* output = PyArray_SimpleNew(1, dims, NPY_FLOAT);
        if (output == NULL) {
            Py_DECREF(float_arr);
            return NULL;
        }

        // Copy results to output
        c_value_type* out_data = (c_value_type*)PyArray_DATA((PyArrayObject*)output);
        memcpy(out_data, results, expected_size * sizeof(c_value_type));

        Py_DECREF(float_arr);
        return output;
    }
}

// Expose the get_numpy_type_name function
%inline %{
const char* get_numpy_type_name_wrapper(const char* type_id) {
    return get_numpy_type_name(type_id);
}
%}
