"""
Comprehensive benchmarks comparing all binding implementations.

Benchmarks the three processing methods across all implemented bridges:
- preallocated: Reuses pre-allocated results buffer
- new: Creates new contiguous array each call
- manual: Manual element-by-element casting

Run with: pytest benchmarks/ --benchmark-only
"""

import numpy as np
import pytest

# Array sizes to benchmark
ARRAY_SIZES = [10, 100, 1_000, 10_000, 100_000, 1_000_000]

# Bridges to benchmark - will skip unavailable ones
BRIDGES = []

try:
    from cython_processor import PyArrayProcessor as CythonProcessor

    BRIDGES.append(("cython", CythonProcessor))
except ImportError:
    pass

try:
    from pybind_processor import PyArrayProcessor as PybindProcessor

    BRIDGES.append(("pybind11", PybindProcessor))
except ImportError:
    pass

try:
    from nanobind_processor import PyArrayProcessor as NanobindProcessor

    BRIDGES.append(("nanobind", NanobindProcessor))
except ImportError:
    pass

try:
    from swig_processor import PyArrayProcessor as SwigProcessor

    BRIDGES.append(("swig", SwigProcessor))
except ImportError:
    pass


def get_numpy_type():
    """Get the numpy dtype from any available bridge."""
    for _, processor_class in BRIDGES:
        type_name = processor_class.get_numpy_type_name("value")
        return np.dtype(type_name)
    return np.float32


NP_TYPE = get_numpy_type()


@pytest.fixture(params=ARRAY_SIZES, ids=lambda x: f"n{x}")
def array_size(request):
    """Parameterized fixture for array sizes."""
    return request.param


@pytest.fixture(params=BRIDGES, ids=lambda x: x[0])
def bridge(request):
    """Parameterized fixture for bridges."""
    return request.param


@pytest.fixture
def input_array(array_size):
    """Create input array of the correct type and size."""
    return np.arange(array_size, dtype=NP_TYPE)


@pytest.fixture
def processor(bridge, array_size):
    """Create a processor instance for the given bridge and size."""
    _, processor_class = bridge
    return processor_class(array_size)


# =============================================================================
# Benchmark Tests - All Methods
# =============================================================================


class TestPreallocated:
    """Benchmarks for preallocated buffer method."""

    def test_preallocated(self, benchmark, bridge, array_size, input_array, processor):
        """Benchmark preallocated buffer method."""
        bridge_name, _ = bridge

        def run():
            return processor.process_preallocated(input_array)

        result = benchmark(run)
        assert result is not None
        assert len(result) == array_size


class TestNewArray:
    """Benchmarks for new array allocation method."""

    def test_new_array(self, benchmark, bridge, array_size, input_array, processor):
        """Benchmark new array allocation method."""
        bridge_name, _ = bridge

        def run():
            return processor.process_new(input_array)

        result = benchmark(run)
        assert result is not None
        assert len(result) == array_size


class TestManualCasting:
    """Benchmarks for manual casting method."""

    def test_manual_casting(self, benchmark, bridge, array_size, input_array, processor):
        """Benchmark manual casting method."""
        bridge_name, _ = bridge

        def run():
            return processor.process_manual(input_array)

        result = benchmark(run)
        assert result is not None
        assert len(result) == array_size


# =============================================================================
# Call Overhead Benchmarks - Minimal Processing
# =============================================================================


class TestCallOverhead:
    """Benchmarks to isolate binding call overhead."""

    @pytest.fixture
    def small_array(self):
        """Tiny array to minimize processing time."""
        return np.array([1.0, 2.0], dtype=NP_TYPE)

    @pytest.fixture
    def small_processor(self, bridge):
        """Processor for tiny array."""
        _, processor_class = bridge
        return processor_class(2)

    def test_call_overhead_preallocated(
        self, benchmark, bridge, small_array, small_processor
    ):
        """Measure pure call overhead with minimal data."""
        benchmark(small_processor.process_preallocated, small_array)

    def test_call_overhead_new(self, benchmark, bridge, small_array, small_processor):
        """Measure call overhead for new array method."""
        benchmark(small_processor.process_new, small_array)


# =============================================================================
# Type Conversion Benchmarks
# =============================================================================


class TestTypeConversion:
    """Benchmarks for type conversion overhead."""

    @pytest.fixture
    def int_array(self, array_size):
        """Integer array requiring conversion."""
        return np.arange(array_size, dtype=np.int32)

    @pytest.fixture
    def float64_array(self, array_size):
        """Float64 array requiring conversion."""
        return np.arange(array_size, dtype=np.float64)

    def test_int32_conversion(self, benchmark, bridge, array_size, int_array, processor):
        """Benchmark processing int32 arrays (requires conversion)."""
        bridge_name, _ = bridge

        def run():
            return processor.process_manual(int_array)

        result = benchmark(run)
        assert result is not None

    def test_float64_conversion(
        self, benchmark, bridge, array_size, float64_array, processor
    ):
        """Benchmark processing float64 arrays (requires conversion)."""
        bridge_name, _ = bridge

        def run():
            return processor.process_manual(float64_array)

        result = benchmark(run)
        assert result is not None


# =============================================================================
# Scaling Benchmarks - Specific sizes for comparison
# =============================================================================


class TestScaling:
    """Benchmarks specifically for analyzing scaling behavior."""

    @pytest.fixture(params=[100, 1_000, 10_000, 100_000], ids=lambda x: f"n={x:,}")
    def scaling_size(self, request):
        return request.param

    @pytest.fixture
    def scaling_array(self, scaling_size):
        return np.arange(scaling_size, dtype=NP_TYPE)

    @pytest.fixture
    def scaling_processor(self, bridge, scaling_size):
        _, processor_class = bridge
        return processor_class(scaling_size)

    def test_scaling_preallocated(
        self, benchmark, bridge, scaling_size, scaling_array, scaling_processor
    ):
        """Track how preallocated scales with array size."""
        benchmark(scaling_processor.process_preallocated, scaling_array)

    def test_scaling_new(
        self, benchmark, bridge, scaling_size, scaling_array, scaling_processor
    ):
        """Track how new array allocation scales."""
        benchmark(scaling_processor.process_new, scaling_array)


