#define HPY_ABI_VERSION HPY_ABI_VERSION_RUNTIME
#include "hpy.h"
#include "../ctypes_processor/c_wrapper.h"
#include <string.h>

typedef struct {
    HPyField processor_handle;
    HPyField size_field;
    HPyField results_buffer;  // Pre-allocated buffer for process_preallocated
} PyArrayProcessorObject;

static HPyType_Spec PyArrayProcessor_spec;

// Helper to get NumPy array from HPy_Field
static HPy_ssize_t get_array_size(HPyContext *ctx, HPy array) {
    HPy shape_attr = HPyUnicode_FromString(ctx, "shape");
    HPy shape = HPy_GetAttr(ctx, array, shape_attr);
    HPy_Close(ctx, shape_attr);

    if (HPy_IsNull(shape)) {
        return -1;
    }

    HPy first_dim = HPy_GetItem_i(ctx, shape, 0);
    HPy_Close(ctx, shape);

    if (HPy_IsNull(first_dim)) {
        return -1;
    }

    HPy_ssize_t size = HPyLong_AsLong(ctx, first_dim);
    HPy_Close(ctx, first_dim);

    return size;
}

// Helper to get pointer to NumPy array data
static float* get_array_data_ptr(HPyContext *ctx, HPy array) {
    // Use __array_interface__ to get the data pointer
    HPy interface_str = HPyUnicode_FromString(ctx, "__array_interface__");
    HPy interface = HPy_GetAttr(ctx, array, interface_str);
    HPy_Close(ctx, interface_str);

    if (HPy_IsNull(interface)) {
        return NULL;
    }

    HPy data_str = HPyUnicode_FromString(ctx, "data");
    HPy data_tuple = HPy_GetAttr(ctx, interface, data_str);
    HPy_Close(ctx, data_str);
    HPy_Close(ctx, interface);

    if (HPy_IsNull(data_tuple)) {
        return NULL;
    }

    HPy addr_obj = HPy_GetItem_i(ctx, data_tuple, 0);
    HPy_Close(ctx, data_tuple);

    if (HPy_IsNull(addr_obj)) {
        return NULL;
    }

    long addr = HPyLong_AsLong(ctx, addr_obj);
    HPy_Close(ctx, addr_obj);

    return (float*)addr;
}

// __init__ method
static int
PyArrayProcessor_init(HPyContext *ctx, HPy self, const HPy *args, size_t nargs, HPy kwnames)
{
    size_t size;

    if (nargs != 1) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "PyArrayProcessor() takes exactly 1 argument");
        return -1;
    }

    size = HPyLong_AsSize_t(ctx, args[0]);
    if (HPyErr_Occurred(ctx)) {
        return -1;
    }

    // Create the ArrayProcessor instance
    ArrayProcessorHandle handle = array_processor_create(size);
    if (handle == NULL) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "Failed to create ArrayProcessor");
        return -1;
    }

    // Store the handle
    PyArrayProcessorObject *data = HPy_GETDATA(ctx, self, PyArrayProcessor_spec);
    HPy handle_long = HPyLong_FromVoidPtr(ctx, handle);
    HPyField_Store(ctx, self, &data->processor_handle, handle_long);
    HPy_Close(ctx, handle_long);

    // Store the size
    HPy size_obj = HPyLong_FromSize_t(ctx, size);
    HPyField_Store(ctx, self, &data->size_field, size_obj);
    HPy_Close(ctx, size_obj);

    // Initialize results_buffer to None (will be created on first use)
    HPyField_Store(ctx, self, &data->results_buffer, ctx->h_None);

    return 0;
}

// __del__ method
static void
PyArrayProcessor_dealloc(void *data_raw)
{
    PyArrayProcessorObject *data = (PyArrayProcessorObject *)data_raw;
    // The handle will be cleaned up when the field is freed
}

// Helper to destroy the C++ processor
static void
PyArrayProcessor_traverse(void *data_raw, HPyFunc_visitproc visit, void *arg)
{
    PyArrayProcessorObject *data = (PyArrayProcessorObject *)data_raw;
    HPyField_Visit(&data->processor_handle, visit, arg);
    HPyField_Visit(&data->size_field, visit, arg);
    HPyField_Visit(&data->results_buffer, visit, arg);
}

// Actual cleanup when object is destroyed
static HPy
PyArrayProcessor_close(HPyContext *ctx, HPy self, const HPy *args, size_t nargs)
{
    PyArrayProcessorObject *data = HPy_GETDATA(ctx, self, PyArrayProcessor_spec);
    HPy handle_obj = HPyField_Load(ctx, self, data->processor_handle);

    if (!HPy_IsNull(handle_obj)) {
        void *handle = HPyLong_AsVoidPtr(ctx, handle_obj);
        if (handle != NULL) {
            array_processor_destroy(handle);
        }
        HPy_Close(ctx, handle_obj);
        HPyField_Store(ctx, self, &data->processor_handle, ctx->h_None);
    }

    return HPy_Dup(ctx, ctx->h_None);
}

// process_preallocated - most efficient, reuses buffer
static HPy
PyArrayProcessor_process_preallocated(HPyContext *ctx, HPy self, const HPy *args, size_t nargs)
{
    if (nargs != 1) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "process_preallocated() takes exactly 1 argument");
        return HPy_NULL;
    }

    HPy np_values = args[0];
    PyArrayProcessorObject *data = HPy_GETDATA(ctx, self, PyArrayProcessor_spec);

    // Get the processor handle
    HPy handle_obj = HPyField_Load(ctx, self, data->processor_handle);
    void *handle = HPyLong_AsVoidPtr(ctx, handle_obj);
    HPy_Close(ctx, handle_obj);

    if (handle == NULL) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "ArrayProcessor not initialized");
        return HPy_NULL;
    }

    // Get size
    HPy size_obj = HPyField_Load(ctx, self, data->size_field);
    size_t cpp_size = HPyLong_AsSize_t(ctx, size_obj);
    HPy_Close(ctx, size_obj);

    // Get input array data
    float *input_data = get_array_data_ptr(ctx, np_values);
    if (input_data == NULL) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "Expected NumPy array");
        return HPy_NULL;
    }

    HPy_ssize_t input_size = get_array_size(ctx, np_values);
    if (input_size < 0 || (size_t)input_size != cpp_size) {
        HPyErr_SetString(ctx, ctx->h_ValueError, "Input array size mismatch");
        return HPy_NULL;
    }

    // Process the array
    array_processor_process(handle, input_data, cpp_size);

    // Get results from C++
    const float *results = array_processor_get_results(handle);
    if (results == NULL) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "Failed to get results");
        return HPy_NULL;
    }

    // Check if we have a pre-allocated buffer, if not create one
    HPy results_buffer = HPyField_Load(ctx, self, data->results_buffer);

    if (HPy_Is(ctx, results_buffer, ctx->h_None)) {
        // Create new buffer using numpy.zeros
        HPy numpy_module = HPyImport_ImportModule(ctx, "numpy");
        if (HPy_IsNull(numpy_module)) {
            return HPy_NULL;
        }

        HPy zeros_func = HPy_GetAttr_s(ctx, numpy_module, "zeros");
        HPy_Close(ctx, numpy_module);

        if (HPy_IsNull(zeros_func)) {
            return HPy_NULL;
        }

        HPy shape = HPyLong_FromSize_t(ctx, cpp_size);
        HPy dtype_str = HPyUnicode_FromString(ctx, "float32");

        HPy call_args[2] = {shape, dtype_str};
        results_buffer = HPy_Call(ctx, zeros_func, call_args, 2);

        HPy_Close(ctx, zeros_func);
        HPy_Close(ctx, shape);
        HPy_Close(ctx, dtype_str);

        if (HPy_IsNull(results_buffer)) {
            return HPy_NULL;
        }

        // Store the buffer for reuse
        HPyField_Store(ctx, self, &data->results_buffer, results_buffer);
    }

    // Copy results to the buffer
    float *buffer_data = get_array_data_ptr(ctx, results_buffer);
    if (buffer_data == NULL) {
        HPy_Close(ctx, results_buffer);
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "Failed to get buffer data pointer");
        return HPy_NULL;
    }

    memcpy(buffer_data, results, cpp_size * sizeof(float));

    return results_buffer;
}

// process_new - flexible, creates new array each time
static HPy
PyArrayProcessor_process_new(HPyContext *ctx, HPy self, const HPy *args, size_t nargs)
{
    if (nargs != 1) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "process_new() takes exactly 1 argument");
        return HPy_NULL;
    }

    HPy input = args[0];
    PyArrayProcessorObject *data = HPy_GETDATA(ctx, self, PyArrayProcessor_spec);

    // Get the processor handle
    HPy handle_obj = HPyField_Load(ctx, self, data->processor_handle);
    void *handle = HPyLong_AsVoidPtr(ctx, handle_obj);
    HPy_Close(ctx, handle_obj);

    if (handle == NULL) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "ArrayProcessor not initialized");
        return HPy_NULL;
    }

    // Get size
    HPy size_obj = HPyField_Load(ctx, self, data->size_field);
    size_t cpp_size = HPyLong_AsSize_t(ctx, size_obj);
    HPy_Close(ctx, size_obj);

    // Convert input to numpy array if needed (using numpy.asarray)
    HPy numpy_module = HPyImport_ImportModule(ctx, "numpy");
    if (HPy_IsNull(numpy_module)) {
        return HPy_NULL;
    }

    HPy asarray_func = HPy_GetAttr_s(ctx, numpy_module, "asarray");
    HPy dtype_str = HPyUnicode_FromString(ctx, "float32");

    HPy call_args[2] = {input, dtype_str};
    HPy np_values = HPy_Call(ctx, asarray_func, call_args, 2);

    HPy_Close(ctx, asarray_func);
    HPy_Close(ctx, dtype_str);
    HPy_Close(ctx, numpy_module);

    if (HPy_IsNull(np_values)) {
        return HPy_NULL;
    }

    // Get input array data
    float *input_data = get_array_data_ptr(ctx, np_values);
    if (input_data == NULL) {
        HPy_Close(ctx, np_values);
        HPyErr_SetString(ctx, ctx->h_TypeError, "Failed to convert to NumPy array");
        return HPy_NULL;
    }

    HPy_ssize_t input_size = get_array_size(ctx, np_values);
    if (input_size < 0) {
        HPy_Close(ctx, np_values);
        HPyErr_SetString(ctx, ctx->h_ValueError, "Failed to get array size");
        return HPy_NULL;
    }

    // Validate size
    if ((size_t)input_size != cpp_size) {
        HPy_Close(ctx, np_values);
        HPyErr_SetString(ctx, ctx->h_ValueError, "Input array size mismatch");
        return HPy_NULL;
    }

    // Process the array
    array_processor_process(handle, input_data, cpp_size);
    HPy_Close(ctx, np_values);

    // Get results from C++
    const float *results = array_processor_get_results(handle);
    if (results == NULL) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "Failed to get results");
        return HPy_NULL;
    }

    // Create NEW array for results
    numpy_module = HPyImport_ImportModule(ctx, "numpy");
    if (HPy_IsNull(numpy_module)) {
        return HPy_NULL;
    }

    HPy zeros_func = HPy_GetAttr_s(ctx, numpy_module, "zeros");
    HPy_Close(ctx, numpy_module);

    if (HPy_IsNull(zeros_func)) {
        return HPy_NULL;
    }

    HPy shape = HPyLong_FromSize_t(ctx, cpp_size);
    dtype_str = HPyUnicode_FromString(ctx, "float32");

    HPy new_call_args[2] = {shape, dtype_str};
    HPy results_array = HPy_Call(ctx, zeros_func, new_call_args, 2);

    HPy_Close(ctx, zeros_func);
    HPy_Close(ctx, shape);
    HPy_Close(ctx, dtype_str);

    if (HPy_IsNull(results_array)) {
        return HPy_NULL;
    }

    // Copy results
    float *results_data = get_array_data_ptr(ctx, results_array);
    if (results_data == NULL) {
        HPy_Close(ctx, results_array);
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "Failed to get results array data pointer");
        return HPy_NULL;
    }

    memcpy(results_data, results, cpp_size * sizeof(float));

    return results_array;
}

// process_manual - manual element-by-element conversion
static HPy
PyArrayProcessor_process_manual(HPyContext *ctx, HPy self, const HPy *args, size_t nargs)
{
    if (nargs != 1) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "process_manual() takes exactly 1 argument");
        return HPy_NULL;
    }

    HPy np_values = args[0];
    PyArrayProcessorObject *data = HPy_GETDATA(ctx, self, PyArrayProcessor_spec);

    // Get the processor handle
    HPy handle_obj = HPyField_Load(ctx, self, data->processor_handle);
    void *handle = HPyLong_AsVoidPtr(ctx, handle_obj);
    HPy_Close(ctx, handle_obj);

    if (handle == NULL) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "ArrayProcessor not initialized");
        return HPy_NULL;
    }

    // Get size
    HPy size_obj = HPyField_Load(ctx, self, data->size_field);
    size_t cpp_size = HPyLong_AsSize_t(ctx, size_obj);
    HPy_Close(ctx, size_obj);

    // Get dtype string
    HPy dtype_attr = HPy_GetAttr_s(ctx, np_values, "dtype");
    if (HPy_IsNull(dtype_attr)) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "Input must be a NumPy array");
        return HPy_NULL;
    }

    HPy dtype_name_attr = HPy_GetAttr_s(ctx, dtype_attr, "name");
    HPy_Close(ctx, dtype_attr);

    if (HPy_IsNull(dtype_name_attr)) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "Failed to get dtype name");
        return HPy_NULL;
    }

    // Get input size
    HPy_ssize_t input_size = get_array_size(ctx, np_values);
    if (input_size < 0) {
        HPy_Close(ctx, dtype_name_attr);
        HPyErr_SetString(ctx, ctx->h_ValueError, "Failed to get array size");
        return HPy_NULL;
    }

    size_t copy_size = (size_t)input_size < cpp_size ? (size_t)input_size : cpp_size;

    // Create a temporary buffer for manual conversion
    float *buffer = (float *)malloc(cpp_size * sizeof(float));
    if (buffer == NULL) {
        HPy_Close(ctx, dtype_name_attr);
        HPyErr_SetString(ctx, ctx->h_MemoryError, "Failed to allocate buffer");
        return HPy_NULL;
    }

    // Initialize buffer to zero
    memset(buffer, 0, cpp_size * sizeof(float));

    // Manual element-by-element conversion using numpy array indexing
    for (size_t i = 0; i < copy_size; i++) {
        HPy item = HPy_GetItem_i(ctx, np_values, i);
        if (HPy_IsNull(item)) {
            free(buffer);
            HPy_Close(ctx, dtype_name_attr);
            HPyErr_SetString(ctx, ctx->h_RuntimeError, "Failed to get array item");
            return HPy_NULL;
        }

        // Convert to float
        HPy float_obj = HPy_Float_FromDouble(ctx, HPyFloat_AsDouble(ctx, item));
        buffer[i] = (float)HPyFloat_AsDouble(ctx, float_obj);

        HPy_Close(ctx, item);
        HPy_Close(ctx, float_obj);
    }

    HPy_Close(ctx, dtype_name_attr);

    // Process the buffer
    array_processor_process(handle, buffer, cpp_size);
    free(buffer);

    // Get results from C++
    const float *results = array_processor_get_results(handle);
    if (results == NULL) {
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "Failed to get results");
        return HPy_NULL;
    }

    // Create results array
    HPy numpy_module = HPyImport_ImportModule(ctx, "numpy");
    if (HPy_IsNull(numpy_module)) {
        return HPy_NULL;
    }

    HPy zeros_func = HPy_GetAttr_s(ctx, numpy_module, "zeros");
    HPy_Close(ctx, numpy_module);

    if (HPy_IsNull(zeros_func)) {
        return HPy_NULL;
    }

    HPy shape = HPyLong_FromSize_t(ctx, cpp_size);
    HPy dtype_str = HPyUnicode_FromString(ctx, "float32");

    HPy call_args[2] = {shape, dtype_str};
    HPy results_array = HPy_Call(ctx, zeros_func, call_args, 2);

    HPy_Close(ctx, zeros_func);
    HPy_Close(ctx, shape);
    HPy_Close(ctx, dtype_str);

    if (HPy_IsNull(results_array)) {
        return HPy_NULL;
    }

    // Copy results
    float *results_data = get_array_data_ptr(ctx, results_array);
    if (results_data == NULL) {
        HPy_Close(ctx, results_array);
        HPyErr_SetString(ctx, ctx->h_RuntimeError, "Failed to get results array data pointer");
        return HPy_NULL;
    }

    memcpy(results_data, results, cpp_size * sizeof(float));

    return results_array;
}

// get_numpy_type_name - static method
static HPy
PyArrayProcessor_get_numpy_type_name(HPyContext *ctx, HPy self, const HPy *args, size_t nargs)
{
    if (nargs != 1) {
        HPyErr_SetString(ctx, ctx->h_TypeError, "get_numpy_type_name() takes exactly 1 argument");
        return HPy_NULL;
    }

    HPy type_id = args[0];
    const char *type_id_str = HPyUnicode_AsUTF8AndSize(ctx, type_id, NULL);
    if (type_id_str == NULL) {
        return HPy_NULL;
    }

    const char *result = get_numpy_type_name_c(type_id_str);
    return HPyUnicode_FromString(ctx, result);
}

// Method definitions
static HPyDef *PyArrayProcessor_methods[] = {
    &(HPyDef){
        .kind = HPyDef_Kind_Meth,
        .meth = {
            .name = "process_preallocated",
            .impl = (HPyCFunction)PyArrayProcessor_process_preallocated,
            .signature = HPyFunc_VARARGS,
            .doc = "Process array using pre-allocated buffer (most efficient)"
        }
    },
    &(HPyDef){
        .kind = HPyDef_Kind_Meth,
        .meth = {
            .name = "process_new",
            .impl = (HPyCFunction)PyArrayProcessor_process_new,
            .signature = HPyFunc_VARARGS,
            .doc = "Process array creating new result array each time (flexible)"
        }
    },
    &(HPyDef){
        .kind = HPyDef_Kind_Meth,
        .meth = {
            .name = "process_manual",
            .impl = (HPyCFunction)PyArrayProcessor_process_manual,
            .signature = HPyFunc_VARARGS,
            .doc = "Process array with manual element conversion (maximum control)"
        }
    },
    &(HPyDef){
        .kind = HPyDef_Kind_Meth,
        .meth = {
            .name = "close",
            .impl = (HPyCFunction)PyArrayProcessor_close,
            .signature = HPyFunc_VARARGS,
            .doc = "Clean up resources"
        }
    },
    &(HPyDef){
        .kind = HPyDef_Kind_Meth,
        .meth = {
            .name = "get_numpy_type_name",
            .impl = (HPyCFunction)PyArrayProcessor_get_numpy_type_name,
            .signature = HPyFunc_VARARGS | HPyFunc_STATIC,
            .doc = "Get NumPy type name for a given type ID"
        }
    },
    NULL
};

// Type specification
static HPyType_Spec PyArrayProcessor_spec = {
    .name = "hpy_processor.PyArrayProcessor",
    .basicsize = sizeof(PyArrayProcessorObject),
    .flags = HPy_TPFLAGS_DEFAULT | HPy_TPFLAGS_HAVE_GC,
    .defines = PyArrayProcessor_methods,
    .legacy = {
        .tp_init = PyArrayProcessor_init,
        .tp_dealloc = PyArrayProcessor_dealloc,
        .tp_traverse = (traverseproc)PyArrayProcessor_traverse,
    }
};

// Module initialization
static HPyDef *module_methods[] = {
    NULL
};

static HPyModuleDef moduledef = {
    .name = "hpy_processor",
    .doc = "HPy-based Python/C++ bridge processor",
    .size = -1,
    .defines = module_methods,
};

HPy_MODINIT(hpy_processor, moduledef)
