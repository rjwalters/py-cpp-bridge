
#include "cpp_processor.hpp"

#include <cstring>
#include <iostream>

#include "types.hpp"

ArrayProcessor::ArrayProcessor(c_value_type size) : _size(size)
{
  // Allocate results array
  _results = new c_value_type[size];
  std::memset(_results, 0, size * sizeof(c_value_type));

  std::cout << "ArrayProcessor created with size " << size << std::endl;
}

ArrayProcessor::~ArrayProcessor()
{
  // Clean up
  delete[] _results;
  _results = nullptr;
  std::cout << "ArrayProcessor destroyed" << std::endl;
}

void ArrayProcessor::process_array(c_value_type *data, size_t size)
{
  // Validate input
  std::cout << "Process array called with size " << size << std::endl;

  if (data == nullptr)
  {
    throw std::runtime_error("Null data pointer passed to process_array");
  }

  if (size != _size)
  {
    throw std::runtime_error("Size mismatch in process_array");
  }

  // Process the array (simple operation - double each value)
  for (size_t i = 0; i < size; i++)
  {
    std::cout << "data[" << i << "] = " << static_cast<int>(data[i]) << std::endl;
    _results[i] = data[i] * 2;
  }

  std::cout << "Array processing complete" << std::endl;
}

c_value_type *ArrayProcessor::get_results() const
{
  return _results;
}

size_t ArrayProcessor::get_size() const
{
  return _size;
}