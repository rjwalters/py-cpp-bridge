#ifndef CFFI_WRAPPER_H
#define CFFI_WRAPPER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to C++ ArrayProcessor
typedef void* ArrayProcessorHandle;

// Create a new ArrayProcessor
ArrayProcessorHandle array_processor_new(size_t size);

// Destroy an ArrayProcessor
void array_processor_delete(ArrayProcessorHandle handle);

// Process an array
void array_processor_process(ArrayProcessorHandle handle, float* data, size_t size);

// Get results pointer
float* array_processor_get_results(ArrayProcessorHandle handle);

// Get size
size_t array_processor_get_size(ArrayProcessorHandle handle);

// Get numpy type name
const char* get_numpy_type_name_wrapper(const char* type_id);

#ifdef __cplusplus
}
#endif

#endif // CFFI_WRAPPER_H
