#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstdint>
#include <cstring>
#include <map>
#include <string>

typedef float c_value_type;
#define NUMPY_VALUE_TYPE "float32" // match to c_success_type

// Function to provide NumPy type information to Python
inline const char *get_numpy_type_name(const char *type_id) {
  if (strcmp(type_id, "value") == 0)
    return NUMPY_VALUE_TYPE;
  return "unknown";
}

#endif // TYPES_HPP