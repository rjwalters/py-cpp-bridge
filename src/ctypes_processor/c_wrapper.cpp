#include "c_wrapper.h"
#include "../common/cpp_processor.hpp"
#include "../common/types.hpp"

#include <iostream>
#include <stdexcept>

extern "C" {

ArrayProcessorHandle array_processor_create(size_t size) {
    try {
        ArrayProcessor* processor = new ArrayProcessor(size);
        return static_cast<void*>(processor);
    } catch (const std::exception& e) {
        std::cerr << "Error creating ArrayProcessor: " << e.what() << std::endl;
        return nullptr;
    }
}

void array_processor_destroy(ArrayProcessorHandle handle) {
    if (handle != nullptr) {
        ArrayProcessor* processor = static_cast<ArrayProcessor*>(handle);
        delete processor;
    }
}

void array_processor_process(ArrayProcessorHandle handle, float* data, size_t size) {
    if (handle == nullptr) {
        std::cerr << "Error: null handle in array_processor_process" << std::endl;
        return;
    }

    try {
        ArrayProcessor* processor = static_cast<ArrayProcessor*>(handle);
        processor->process_array(data, size);
    } catch (const std::exception& e) {
        std::cerr << "Error processing array: " << e.what() << std::endl;
    }
}

const float* array_processor_get_results(ArrayProcessorHandle handle) {
    if (handle == nullptr) {
        return nullptr;
    }

    ArrayProcessor* processor = static_cast<ArrayProcessor*>(handle);
    return processor->get_results();
}

size_t array_processor_get_size(ArrayProcessorHandle handle) {
    if (handle == nullptr) {
        return 0;
    }

    ArrayProcessor* processor = static_cast<ArrayProcessor*>(handle);
    return processor->get_size();
}

const char* get_numpy_type_name_c(const char* type_id) {
    return get_numpy_type_name(type_id);
}

}
