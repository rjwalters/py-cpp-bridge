/**
 * Python wrapper for C++ ArrayProcessor using nanobind.
 *
 * This module provides three different approaches for passing NumPy arrays to C++:
 * 1. Pre-allocated buffer: Reuses the same results array for efficiency
 * 2. New contiguous array: Creates a fresh array each time for flexibility
 * 3. Manual casting: Explicitly controls type conversion for maximum control
 *
 * nanobind is the successor to pybind11, offering:
 * - Faster compilation times
 * - Smaller binary sizes
 * - Lower runtime overhead
 * - Modern C++17 design
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "../common/cpp_processor.hpp"
#include "../common/types.hpp"

namespace nb = nanobind;

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
    results_buffer_ = new c_value_type[size];
    for (size_t i = 0; i < size; i++) {
      results_buffer_[i] = 0;
    }
  }

  ~PyArrayProcessor() {
    delete[] results_buffer_;
  }

  /**
   * Process values using a pre-allocated buffer.
   *
   * This method reuses the same results array for each call, making it
   * efficient for repeated calls on values of the same size.
   */
  nb::ndarray<nb::numpy, c_value_type, nb::ndim<1>>
  process_preallocated(nb::ndarray<c_value_type, nb::ndim<1>, nb::c_contig, nb::device::cpu> np_values) {
    // Check size
    if (np_values.shape(0) != size_) {
      throw std::runtime_error("Expected array of size " +
                               std::to_string(size_) + ", got " +
                               std::to_string(np_values.shape(0)));
    }

    // Process the array
    processor_.process_array(np_values.data(), size_);

    // Copy results to pre-allocated buffer
    c_value_type *results = processor_.get_results();
    for (size_t i = 0; i < size_; i++) {
      results_buffer_[i] = results[i];
    }

    // Return a view of the pre-allocated buffer (no ownership transfer)
    size_t shape[1] = {size_};
    return nb::ndarray<nb::numpy, c_value_type, nb::ndim<1>>(
        results_buffer_, 1, shape, nb::handle());
  }

  /**
   * Process values by creating a new contiguous array.
   *
   * This method creates a new array for each call, providing more flexibility
   * in handling different input types.
   */
  nb::ndarray<nb::numpy, c_value_type, nb::ndim<1>>
  process_new(nb::ndarray<c_value_type, nb::ndim<1>, nb::c_contig, nb::device::cpu> np_values) {
    // Check size
    if (np_values.shape(0) != size_) {
      throw std::runtime_error("Expected array of size " +
                               std::to_string(size_) + ", got " +
                               std::to_string(np_values.shape(0)));
    }

    // Process the array
    processor_.process_array(np_values.data(), size_);

    // Create a new results array with its own memory
    c_value_type *new_data = new c_value_type[size_];
    c_value_type *results = processor_.get_results();
    for (size_t i = 0; i < size_; i++) {
      new_data[i] = results[i];
    }

    // Create a capsule to own the memory
    nb::capsule owner(new_data, [](void *p) noexcept {
      delete[] static_cast<c_value_type *>(p);
    });

    size_t shape[1] = {size_};
    return nb::ndarray<nb::numpy, c_value_type, nb::ndim<1>>(
        new_data, 1, shape, owner);
  }

  /**
   * Process values with manual copying and casting.
   *
   * This method provides explicit control over type conversion by manually
   * copying and casting each element. Accepts any numeric array type.
   */
  nb::ndarray<nb::numpy, c_value_type, nb::ndim<1>>
  process_manual(nb::ndarray<nb::ndim<1>, nb::device::cpu> np_values) {
    // Determine copy size (use smaller of input size and processor size)
    size_t input_size = np_values.shape(0);
    size_t copy_size = std::min(input_size, size_);

    // Create a buffer for processing
    c_value_type *buffer = new c_value_type[size_];
    for (size_t i = 0; i < size_; i++) {
      buffer[i] = 0;
    }

    // Manual copy with explicit casting based on input dtype
    auto dtype = np_values.dtype();

    if (dtype == nb::dtype<float>()) {
      auto view = nb::ndarray<float, nb::ndim<1>, nb::device::cpu>(
          np_values.data(), 1, &input_size, nb::handle());
      for (size_t i = 0; i < copy_size; i++) {
        buffer[i] = static_cast<c_value_type>(
            static_cast<const float *>(np_values.data())[i]);
      }
    } else if (dtype == nb::dtype<double>()) {
      for (size_t i = 0; i < copy_size; i++) {
        buffer[i] = static_cast<c_value_type>(
            static_cast<const double *>(np_values.data())[i]);
      }
    } else if (dtype == nb::dtype<int32_t>()) {
      for (size_t i = 0; i < copy_size; i++) {
        buffer[i] = static_cast<c_value_type>(
            static_cast<const int32_t *>(np_values.data())[i]);
      }
    } else if (dtype == nb::dtype<int64_t>()) {
      for (size_t i = 0; i < copy_size; i++) {
        buffer[i] = static_cast<c_value_type>(
            static_cast<const int64_t *>(np_values.data())[i]);
      }
    } else if (dtype == nb::dtype<uint8_t>()) {
      for (size_t i = 0; i < copy_size; i++) {
        buffer[i] = static_cast<c_value_type>(
            static_cast<const uint8_t *>(np_values.data())[i]);
      }
    } else {
      // Fallback: try to cast through Python for unknown types
      delete[] buffer;
      throw std::runtime_error("Unsupported dtype for manual casting");
    }

    // Process the buffer
    processor_.process_array(buffer, size_);

    // Create results array
    c_value_type *results_data = new c_value_type[size_];
    c_value_type *results = processor_.get_results();
    for (size_t i = 0; i < size_; i++) {
      results_data[i] = results[i];
    }

    delete[] buffer;

    // Create a capsule to own the memory
    nb::capsule owner(results_data, [](void *p) noexcept {
      delete[] static_cast<c_value_type *>(p);
    });

    size_t shape[1] = {size_};
    return nb::ndarray<nb::numpy, c_value_type, nb::ndim<1>>(
        results_data, 1, shape, owner);
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
  c_value_type *results_buffer_;
};

NB_MODULE(nanobind_processor, m) {
  m.doc() = R"pbdoc(
        Python wrapper for C++ ArrayProcessor using nanobind.

        This module provides three different approaches for passing NumPy arrays to C++:
        1. Pre-allocated buffer: Reuses the same results array for efficiency
        2. New contiguous array: Creates a fresh array each time for flexibility
        3. Manual casting: Explicitly controls type conversion for maximum control

        nanobind is the successor to pybind11, offering faster compilation,
        smaller binaries, and lower runtime overhead.
    )pbdoc";

  nb::class_<PyArrayProcessor>(m, "PyArrayProcessor",
                               R"pbdoc(
        Python wrapper for C++ ArrayProcessor.

        This class provides methods for processing arrays of numpy values
        using different memory and type handling approaches.
    )pbdoc")
      .def(nb::init<size_t>(), nb::arg("size"),
           R"pbdoc(
            Initialize the ArrayProcessor with a specified size.

            Args:
                size: The size of arrays this processor will handle
        )pbdoc")
      .def("process_preallocated", &PyArrayProcessor::process_preallocated,
           nb::arg("np_values"),
           R"pbdoc(
            Process values using a pre-allocated buffer.

            This method reuses the same results array for each call, making it
            efficient for repeated calls on values of the same size.

            Args:
                np_values: Input array of values. Must be 1D with length matching size.

            Returns:
                Array of processed values (doubled input values)
        )pbdoc")
      .def("process_new", &PyArrayProcessor::process_new, nb::arg("np_values"),
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
           nb::arg("np_values"),
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
                  nb::arg("type_id"),
                  R"pbdoc(
            Get the NumPy type name for a given type identifier.

            Args:
                type_id: The type identifier (e.g., "value")

            Returns:
                The corresponding NumPy type name (e.g., "float32")
        )pbdoc");
}
