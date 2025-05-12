#ifndef ARRAY_PROCESSOR_HPP
#define ARRAY_PROCESSOR_HPP

#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "types.hpp"

class ArrayProcessor {
public:
  // Constructor
  ArrayProcessor(c_value_type size);

  // Destructor
  ~ArrayProcessor();

  // Process uint8_t array (similar to your step function)
  void process_array(c_value_type *data, size_t size);

  // Get results
  c_value_type *get_results() const;

  // Get size
  size_t get_size() const;

private:
  size_t _size;
  c_value_type *_results;
};

#endif