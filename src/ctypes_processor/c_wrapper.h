#ifndef C_WRAPPER_H
#define C_WRAPPER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to ArrayProcessor
typedef void* ArrayProcessorHandle;

// Create a new ArrayProcessor instance
ArrayProcessorHandle array_processor_create(size_t size);

// Destroy an ArrayProcessor instance
void array_processor_destroy(ArrayProcessorHandle handle);

// Process an array
void array_processor_process(ArrayProcessorHandle handle, float* data, size_t size);

// Get results
const float* array_processor_get_results(ArrayProcessorHandle handle);

// Get size
size_t array_processor_get_size(ArrayProcessorHandle handle);

// Get NumPy type name
const char* get_numpy_type_name_c(const char* type_id);

#ifdef __cplusplus
}
#endif

#endif // C_WRAPPER_H
