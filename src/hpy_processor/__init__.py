"""
HPy-based Python/C++ bridge processor.

HPy (https://hpyproject.org/) is a new universal API for Python extensions
that works across CPython, PyPy, and GraalPy implementations.
"""

try:
    from .hpy_processor import PyArrayProcessor
    __all__ = ['PyArrayProcessor']
except ImportError as e:
    import warnings
    warnings.warn(f"HPy processor extension not available: {e}")
    __all__ = []
