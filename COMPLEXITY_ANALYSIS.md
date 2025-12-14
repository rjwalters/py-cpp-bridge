# Binding Technology Complexity Analysis

## Metrics Summary

### Lines of Code (Implementation Only)
Excluding type stubs (.pyi), __init__.py, and generated files:

| Technology | LOC | Files | Structure |
|------------|-----|-------|-----------|
| **Cython** | 271 | 2 | .pyx + .pxd |
| **pybind11** | 331 | 1 | .cpp |
| **nanobind** | 287 | 1 | .cpp |
| **SWIG** | 297 | 2 | .i + py_wrapper.py |
| **ctypes** | 336 | 3 | .cpp + .h + .py |
| **cffi** | 401 | 4 | .cpp + .h + .py + build script |
| **HPy** | 607 | 1 | .c |

### Architecture Layers

| Technology | Layers | Description |
|------------|--------|-------------|
| **Cython** | 1 | Python-like → C/C++ |
| **pybind11** | 1 | Direct C++ bindings |
| **nanobind** | 1 | Direct C++ bindings |
| **SWIG** | 2 | Interface file → Generated wrapper → Python wrapper |
| **ctypes** | 2 | C wrapper → Python ctypes wrapper |
| **cffi** | 2 | C wrapper → Python cffi wrapper |
| **HPy** | 1 | Direct C API |

### Boilerplate Ratio

Estimated ratio of boilerplate/setup code vs. actual logic:

| Technology | Boilerplate | Assessment |
|------------|-------------|------------|
| **Cython** | ~15% | Mostly logic, minimal setup |
| **pybind11** | ~30% | Module macro + class setup |
| **nanobind** | ~30% | Similar to pybind11 |
| **SWIG** | ~40% | Interface definitions + wrapper layer |
| **ctypes** | ~50% | Full C wrapper + Python wrapper |
| **cffi** | ~50% | C wrapper + Python wrapper + build config |
| **HPy** | ~65% | Verbose C API, lots of handle management |

### LLM Writability Score

How easy is it for an LLM (like Claude or GPT) to generate correct, working code?

| Technology | Score | Rationale |
|------------|-------|-----------|
| **pybind11** | ⭐⭐⭐⭐⭐ | Clear patterns, extensive documentation, lots of training data |
| **Cython** | ⭐⭐⭐⭐ | Python-like syntax, but memoryviews require understanding |
| **nanobind** | ⭐⭐⭐⭐ | Similar to pybind11, less documentation available |
| **SWIG** | ⭐⭐⭐ | Interface syntax is unique, two-layer complexity |
| **ctypes** | ⭐⭐⭐ | Manual type mapping, error-prone pointer handling |
| **cffi** | ⭐⭐⭐ | C declarations straightforward, but manual memory management |
| **HPy** | ⭐⭐ | New API, verbose, less training data, easy to get wrong |

### Developer Experience Factors

| Technology | Learning Curve | Debug Ease | IDE Support | Documentation |
|------------|----------------|------------|-------------|---------------|
| **pybind11** | Gentle | Good | Excellent | Excellent |
| **Cython** | Moderate | Good | Good | Excellent |
| **nanobind** | Gentle | Good | Excellent | Good |
| **SWIG** | Steep | Moderate | Limited | Good |
| **ctypes** | Moderate | Difficult | Good | Good |
| **cffi** | Moderate | Moderate | Good | Good |
| **HPy** | Steep | Difficult | Limited | Emerging |

## Recommendations by Use Case

### Minimum Code (Simplest)
1. **pybind11** (1 file, 331 LOC, single layer)
2. **nanobind** (1 file, 287 LOC, single layer)
3. **Cython** (2 files, 271 LOC, single layer)

### Best for LLM Generation
1. **pybind11** - Most examples, clear patterns
2. **Cython** - Python-like, familiar syntax
3. **nanobind** - Similar to pybind11

### Lowest Boilerplate
1. **Cython** (~15% boilerplate)
2. **pybind11/nanobind** (~30% boilerplate)
3. **SWIG** (~40% boilerplate)

### Cross-Implementation Portability
1. **HPy** - Designed for CPython/PyPy/GraalPy
2. **cffi** - Excellent PyPy support
3. **ctypes** - Standard library, works everywhere

## Detailed Observations

### Why is HPy so verbose (607 LOC)?
- Uses C API directly (no syntactic sugar)
- Explicit handle management for every Python object
- Verbose error checking and reference counting
- Future optimizations may reduce this

### Why does cffi have 4 files (401 LOC)?
- Requires separate C wrapper layer (like ctypes)
- Separate Python wrapper for NumPy integration
- Additional build script for compilation
- Trade-off: No compilation needed at install time (ABI mode)

### Why is Cython the most concise (271 LOC)?
- Python-like syntax reduces verbosity
- Typed memoryviews handle NumPy integration elegantly
- Single-layer architecture
- Built-in type conversion

### pybind11 vs nanobind LOC
- pybind11: 331 LOC
- nanobind: 287 LOC (13% smaller)
- nanobind achieves smaller code through modern C++17 features

## Summary

For this specific use case (wrapping C++ ArrayProcessor with 3 methods):

**Simplest to implement:** Cython (271 LOC, 1 layer, Python-like)
**Best for C++ developers:** pybind11 (331 LOC, excellent docs, LLM-friendly)
**Most compact binary:** nanobind (287 LOC, modern C++17)
**Best for PyPy:** cffi (401 LOC, 2 layers, but excellent runtime)
**Future-proof:** HPy (607 LOC, but cross-implementation)
**No build tools needed:** ctypes (336 LOC, standard library)
**Multi-language:** SWIG (297 LOC, can generate Java/Ruby/etc.)
