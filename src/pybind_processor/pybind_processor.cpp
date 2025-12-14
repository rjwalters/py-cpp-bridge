/**
 * Python wrapper for C++ ArrayProcessor using pybind11.
 *
 * This module provides three different approaches for passing NumPy arrays to C++:
 * 1. Pre-allocated buffer: Reuses the same results array for efficiency
 * 2. New contiguous array: Creates a fresh array each time for flexibility
 * 3. Manual casting: Explicitly controls type conversion for maximum control
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../common/cpp_processor.hpp"
#include "../common/types.hpp"

namespace py = pybind11;

/**
 * Python wrapper class for ArrayProcessor.
 *
 * Provides methods for processing arrays of numpy values using different
 * memory and type handling approaches.
 */
class PyArrayProcessor {
public:
  /**
   * Initialize the ArrayProcessor with a specified size.
   *
   * @param size The size of arrays this processor will handle
   */
  PyArrayProcessor(size_t size)
      : processor_(size), size_(size) {
    // Pre-allocate the results buffer for method 1
    results_buffer_ = py::array_t<c_value_type>(size);
  }

  /**
   * Process values using a pre-allocated buffer.
   *
   * This method reuses the same results array for each call, making it
   * efficient for repeated calls on values of the same size.
   *
   * Pros:
   *   - Efficient for repeated calls as it reuses the results buffer
   *   - Minimizes memory allocations
   *   - Clear type safety with explicit array typing
   *
   * Cons:
   *   - The results array is tied to the object's lifetime
   *   - Less flexible if result size might change
   *   - Could lead to issues if the view is modified externally
   *
   * Best for:
   *   High-performance code with frequent processing calls on
   *   similarly sized values
   *
   * @param np_values Input array of values. Must be 1D with length matching
   * size.
   * @return Array of processed values (doubled input values)
   * @throws std::runtime_error If values length doesn't match expected size
   */
  py::array_t<c_value_type>
  process_preallocated(py::array_t<c_value_type, py::array::c_style> np_values) {
    // Request buffer info with type checking
    py::buffer_info buf = np_values.request();

    if (buf.ndim != 1) {
      throw std::runtime_error("Expected 1-dimensional array");
    }

    if (static_cast<size_t>(buf.shape[0]) != size_) {
      throw std::runtime_error("Expected array of size " +
                               std::to_string(size_) + ", got " +
                               std::to_string(buf.shape[0]));
    }

    // Process the array
    processor_.process_array(static_cast<c_value_type *>(buf.ptr), size_);

    // Copy results to pre-allocated buffer
    auto result_buf = results_buffer_.request();
    c_value_type *results = processor_.get_results();
    c_value_type *result_ptr = static_cast<c_value_type *>(result_buf.ptr);

    for (size_t i = 0; i < size_; i++) {
      result_ptr[i] = results[i];
    }

    return results_buffer_;
  }

  /**
   * Process values by creating a new contiguous array.
   *
   * This method creates a new array for each call, providing more flexibility
   * in handling different input types.
   *
   * Pros:
   *   - More flexible input handling (auto-converts types)
   *   - Clean separation between input and output
   *   - Each result is independent
   *
   * Cons:
   *   - More memory allocations
   *   - Slightly more overhead for type conversion
   *
   * Best for:
   *   General-purpose use where convenience is valued over ultimate performance
   *
   * @param np_values Input values that can be converted to a NumPy array
   * @return New array of processed values (doubled input values)
   * @throws std::runtime_error If values length doesn't match expected size
   */
  py::array_t<c_value_type>
  process_new(py::array_t<c_value_type, py::array::c_style | py::array::forcecast> np_values) {
    // forcecast flag ensures input is converted to the correct type
    py::buffer_info buf = np_values.request();

    if (buf.ndim != 1) {
      throw std::runtime_error("Expected 1-dimensional array");
    }

    if (static_cast<size_t>(buf.shape[0]) != size_) {
      throw std::runtime_error("Expected array of size " +
                               std::to_string(size_) + ", got " +
                               std::to_string(buf.shape[0]));
    }

    // Process the array
    processor_.process_array(static_cast<c_value_type *>(buf.ptr), size_);

    // Create a new results array
    py::array_t<c_value_type> results_array(size_);
    auto result_buf = results_array.request();
    c_value_type *results = processor_.get_results();
    c_value_type *result_ptr = static_cast<c_value_type *>(result_buf.ptr);

    for (size_t i = 0; i < size_; i++) {
      result_ptr[i] = results[i];
    }

    return results_array;
  }

  /**
   * Process values with manual copying and casting.
   *
   * This method provides explicit control over type conversion by manually
   * copying and casting each element. Accepts any numeric array type.
   *
   * Pros:
   *   - Maximum control over values conversion
   *   - Can handle arrays of any numeric type
   *   - Can perform validation or transformation during copying
   *
   * Cons:
   *   - Most verbose approach
   *   - Extra copying step adds overhead
   *   - Not needed for many simple cases
   *
   * Best for:
   *   Cases where precise control over type conversion is needed
   *
   * @param np_values Input array of any numeric type
   * @return New array of processed values (doubled input values)
   */
  py::array_t<c_value_type> process_manual(py::array np_values) {
    // Get buffer info without type constraint
    py::buffer_info buf = np_values.request();

    if (buf.ndim != 1) {
      throw std::runtime_error("Expected 1-dimensional array");
    }

    // Determine copy size (use smaller of input size and processor size)
    size_t input_size = static_cast<size_t>(buf.shape[0]);
    size_t copy_size = std::min(input_size, size_);

    // Create a buffer for processing
    py::array_t<c_value_type> buffer(size_);
    auto buffer_info = buffer.request();
    c_value_type *buffer_ptr = static_cast<c_value_type *>(buffer_info.ptr);

    // Zero-initialize the buffer
    for (size_t i = 0; i < size_; i++) {
      buffer_ptr[i] = 0;
    }

    // Manual copy with explicit casting based on input dtype
    if (buf.format == py::format_descriptor<float>::format()) {
      float *input = static_cast<float *>(buf.ptr);
      for (size_t i = 0; i < copy_size; i++) {
        buffer_ptr[i] = static_cast<c_value_type>(input[i]);
      }
    } else if (buf.format == py::format_descriptor<double>::format()) {
      double *input = static_cast<double *>(buf.ptr);
      for (size_t i = 0; i < copy_size; i++) {
        buffer_ptr[i] = static_cast<c_value_type>(input[i]);
      }
    } else if (buf.format == py::format_descriptor<int32_t>::format()) {
      int32_t *input = static_cast<int32_t *>(buf.ptr);
      for (size_t i = 0; i < copy_size; i++) {
        buffer_ptr[i] = static_cast<c_value_type>(input[i]);
      }
    } else if (buf.format == py::format_descriptor<int64_t>::format()) {
      int64_t *input = static_cast<int64_t *>(buf.ptr);
      for (size_t i = 0; i < copy_size; i++) {
        buffer_ptr[i] = static_cast<c_value_type>(input[i]);
      }
    } else if (buf.format == py::format_descriptor<uint8_t>::format()) {
      uint8_t *input = static_cast<uint8_t *>(buf.ptr);
      for (size_t i = 0; i < copy_size; i++) {
        buffer_ptr[i] = static_cast<c_value_type>(input[i]);
      }
    } else {
      // Fallback: use Python to convert element by element
      for (size_t i = 0; i < copy_size; i++) {
        py::object item = np_values.attr("__getitem__")(i);
        buffer_ptr[i] = item.cast<c_value_type>();
      }
    }

    // Process the buffer
    processor_.process_array(buffer_ptr, size_);

    // Create results array
    py::array_t<c_value_type> results_array(size_);
    auto result_buf = results_array.request();
    c_value_type *results = processor_.get_results();
    c_value_type *result_ptr = static_cast<c_value_type *>(result_buf.ptr);

    for (size_t i = 0; i < size_; i++) {
      result_ptr[i] = results[i];
    }

    return results_array;
  }

  /**
   * Get the NumPy type name for a given type identifier.
   *
   * @param type_id The type identifier (e.g., "value")
   * @return The corresponding NumPy type name (e.g., "float32")
   */
  static std::string get_numpy_type_name(const std::string &type_id) {
    return std::string(::get_numpy_type_name(type_id.c_str()));
  }

private:
  ArrayProcessor processor_;
  size_t size_;
  py::array_t<c_value_type> results_buffer_;
};

PYBIND11_MODULE(pybind_processor, m) {
  m.doc() = R"pbdoc(
        Python wrapper for C++ ArrayProcessor using pybind11.

        This module provides three different approaches for passing NumPy arrays to C++:
        1. Pre-allocated buffer: Reuses the same results array for efficiency
        2. New contiguous array: Creates a fresh array each time for flexibility
        3. Manual casting: Explicitly controls type conversion for maximum control
    )pbdoc";

  py::class_<PyArrayProcessor>(m, "PyArrayProcessor",
                               R"pbdoc(
        Python wrapper for C++ ArrayProcessor.

        This class provides methods for processing arrays of numpy values
        using different memory and type handling approaches.
    )pbdoc")
      .def(py::init<size_t>(), py::arg("size"),
           R"pbdoc(
            Initialize the ArrayProcessor with a specified size.

            Args:
                size: The size of arrays this processor will handle
        )pbdoc")
      .def("process_preallocated", &PyArrayProcessor::process_preallocated,
           py::arg("np_values"),
           R"pbdoc(
            Process values using a pre-allocated buffer.

            This method reuses the same results array for each call, making it
            efficient for repeated calls on values of the same size.

            Args:
                np_values: Input array of values. Must be 1D with length matching size.

            Returns:
                Array of processed values (doubled input values)
        )pbdoc")
      .def("process_new", &PyArrayProcessor::process_new, py::arg("np_values"),
           R"pbdoc(
            Process values by creating a new contiguous array.

            This method creates a new array for each call, providing more flexibility
            in handling different input types.

            Args:
                np_values: Input values that can be converted to a NumPy array

            Returns:
                New array of processed values (doubled input values)
        )pbdoc")
      .def("process_manual", &PyArrayProcessor::process_manual,
           py::arg("np_values"),
           R"pbdoc(
            Process values with manual copying and casting.

            This method provides explicit control over type conversion by manually
            copying and casting each element.

            Args:
                np_values: Input array of any numeric type

            Returns:
                New array of processed values (doubled input values)
        )pbdoc")
      .def_static("get_numpy_type_name", &PyArrayProcessor::get_numpy_type_name,
                  py::arg("type_id"),
                  R"pbdoc(
            Get the NumPy type name for a given type identifier.

            Args:
                type_id: The type identifier (e.g., "value")

            Returns:
                The corresponding NumPy type name (e.g., "float32")
        )pbdoc");
}
