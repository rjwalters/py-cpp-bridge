#include "cffi_wrapper.h"
#include "../common/cpp_processor.hpp"
#include "../common/types.hpp"
#include <stdexcept>
#include <cstring>

extern "C" {

ArrayProcessorHandle array_processor_new(size_t size) {
    try {
        ArrayProcessor* processor = new ArrayProcessor(size);
        return static_cast<ArrayProcessorHandle>(processor);
    } catch (const std::exception& e) {
        // In a production system, you'd want better error handling
        return nullptr;
    }
}

void array_processor_delete(ArrayProcessorHandle handle) {
    if (handle) {
        ArrayProcessor* processor = static_cast<ArrayProcessor*>(handle);
        delete processor;
    }
}

void array_processor_process(ArrayProcessorHandle handle, float* data, size_t size) {
    if (!handle || !data) {
        return;
    }

    try {
        ArrayProcessor* processor = static_cast<ArrayProcessor*>(handle);
        processor->process_array(data, size);
    } catch (const std::exception& e) {
        // In a production system, you'd want better error handling
    }
}

float* array_processor_get_results(ArrayProcessorHandle handle) {
    if (!handle) {
        return nullptr;
    }

    ArrayProcessor* processor = static_cast<ArrayProcessor*>(handle);
    return processor->get_results();
}

size_t array_processor_get_size(ArrayProcessorHandle handle) {
    if (!handle) {
        return 0;
    }

    ArrayProcessor* processor = static_cast<ArrayProcessor*>(handle);
    return processor->get_size();
}

const char* get_numpy_type_name_wrapper(const char* type_id) {
    return ::get_numpy_type_name(type_id);
}

} // extern "C"
