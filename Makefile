# Makefile for py-cpp-bridge project
# Uses CMake via scikit-build-core for building extensions

# Source and build directories
SRC_DIR := src
TEST_DIR := tests
BUILD_DIR := build
BENCH_DIR := benchmarks

# Python command - prefer virtual environment if available
PYTHON := $(shell if [ -f .venv/bin/python3 ]; then echo .venv/bin/python3; elif command -v python3 >/dev/null 2>&1; then echo python3; else echo python; fi)

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all              - Default target, same as 'build'"
	@echo "  build            - Build all extension modules via CMake"
	@echo "  install          - Install the package in development mode"
	@echo "  test             - Run all tests"
	@echo "  benchmark        - Run all performance benchmarks"
	@echo "  benchmark-quick  - Run quick overhead benchmarks only"
	@echo "  benchmark-save   - Run benchmarks and save results to JSON"
	@echo "  benchmark-compare- Compare bridges at 1000 elements"
	@echo "  benchmark-visualize - Visualize benchmark results (requires results.json)"
	@echo "  benchmark-flamegraph - Generate flamegraphs with py-spy"
	@echo "  benchmark-all    - Complete benchmark workflow (build, run, visualize)"
	@echo "  clean            - Remove build artifacts"
	@echo "  distclean        - Deep clean (including compiled extensions and eggs)"
	@echo "  format           - Format C++ code with clang-format"
	@echo ""
	@echo "Build System: CMake via scikit-build-core"

# Default target
.PHONY: all
all: build

# Build extensions using pip (which invokes CMake via scikit-build-core)
.PHONY: build
build:
	$(PYTHON) -m pip install --no-build-isolation -e . -v

# Install Python dependencies
.PHONY: install-deps
install-deps:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Build and install the package in development mode
.PHONY: install
install: install-deps build

# Run tests
.PHONY: test
test:
	@echo "=== Running Cython tests ==="
	-$(PYTHON) $(TEST_DIR)/test.py
	-$(PYTHON) $(TEST_DIR)/typed_example.py
	@echo ""
	@echo "=== Running pybind11 tests ==="
	-$(PYTHON) $(TEST_DIR)/test_pybind.py
	-$(PYTHON) $(TEST_DIR)/typed_example_pybind.py
	@echo ""
	@echo "=== Running nanobind tests ==="
	-$(PYTHON) $(TEST_DIR)/test_nanobind.py
	-$(PYTHON) $(TEST_DIR)/typed_example_nanobind.py
	@echo ""
	@echo "=== Running SWIG tests (if available) ==="
	-$(PYTHON) $(TEST_DIR)/test_swig.py 2>/dev/null || echo "SWIG tests skipped (not built)"
	-$(PYTHON) $(TEST_DIR)/typed_example_swig.py 2>/dev/null || echo "SWIG typed example skipped (not built)"
	@echo ""
	@echo "=== Running ctypes tests ==="
	-$(PYTHON) $(TEST_DIR)/test_ctypes.py
	-$(PYTHON) $(TEST_DIR)/typed_example_ctypes.py
	@echo ""
	@echo "=== Running cffi tests ==="
	-$(PYTHON) $(TEST_DIR)/test_cffi.py
	-$(PYTHON) $(TEST_DIR)/typed_example_cffi.py

# Run benchmarks
.PHONY: benchmark
benchmark:
	@echo "=== Running benchmarks ==="
	$(PYTHON) -m pytest $(BENCH_DIR) --benchmark-only --benchmark-group-by=func --benchmark-sort=mean

# Run quick benchmarks (smaller array sizes, fewer iterations)
.PHONY: benchmark-quick
benchmark-quick:
	@echo "=== Running quick benchmarks ==="
	$(PYTHON) -m pytest $(BENCH_DIR)/test_benchmark.py::TestCallOverhead -v --benchmark-only --benchmark-sort=mean

# Run benchmarks and save results to JSON
.PHONY: benchmark-save
benchmark-save:
	@echo "=== Running benchmarks (saving results) ==="
	$(PYTHON) -m pytest $(BENCH_DIR) --benchmark-only --benchmark-sort=mean \
		--benchmark-json=$(BENCH_DIR)/results.json
	@echo "Results saved to $(BENCH_DIR)/results.json"

# Run benchmarks comparing just the bridges (10000 element arrays)
.PHONY: benchmark-compare
benchmark-compare:
	@echo "=== Comparing bridges (10000 elements) ==="
	$(PYTHON) -m pytest $(BENCH_DIR)/test_benchmark.py -k "n10000]" -v --benchmark-only \
		--benchmark-columns=mean,stddev,rounds --benchmark-sort=mean

# Visualize benchmark results
.PHONY: benchmark-visualize
benchmark-visualize:
	@if [ ! -f $(BENCH_DIR)/results.json ]; then \
		echo "Error: $(BENCH_DIR)/results.json not found"; \
		echo "Run 'make benchmark-save' first to generate results"; \
		exit 1; \
	fi
	@echo "=== Visualizing benchmark results ==="
	$(PYTHON) $(BENCH_DIR)/visualize_benchmarks.py $(BENCH_DIR)/results.json
	@echo ""
	@echo "To generate plots, run:"
	@echo "  $(PYTHON) $(BENCH_DIR)/visualize_benchmarks.py $(BENCH_DIR)/results.json --plot"

# Generate flamegraphs with py-spy
.PHONY: benchmark-flamegraph
benchmark-flamegraph:
	@echo "=== Generating flamegraphs with py-spy ==="
	@echo "Installing py-spy if needed..."
	@$(PYTHON) -m pip install -q py-spy || true
	$(PYTHON) $(BENCH_DIR)/generate_flamegraph.py
	@echo ""
	@echo "Flamegraphs saved to $(BENCH_DIR)/flamegraphs/"
	@echo "Open with: open $(BENCH_DIR)/flamegraphs/*.svg"

# Complete benchmark workflow
.PHONY: benchmark-all
benchmark-all: clean build
	@echo "=== Complete Benchmark Workflow ==="
	@echo "1. Running benchmarks and saving results..."
	@$(MAKE) benchmark-save
	@echo ""
	@echo "2. Visualizing results..."
	@$(MAKE) benchmark-visualize
	@echo ""
	@echo "=== Benchmark workflow complete ==="
	@echo "Results: $(BENCH_DIR)/results.json"
	@echo "Run 'make benchmark-flamegraph' to generate flamegraphs (requires py-spy)"

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf $(SRC_DIR)/*.egg-info/
	find . -name "*.so" -delete
	find . -name "*.pyd" -delete
	find . -name "*.o" -delete
	find . -name "*.html" -path "*/cython_processor/*" -delete
	# Clean generated files
	find $(SRC_DIR)/cython_processor -name "*.cpp" ! -name "cpp_processor.cpp" -delete 2>/dev/null || true
	find $(SRC_DIR)/swig_processor -name "*_wrap.cxx" -delete 2>/dev/null || true
	find $(SRC_DIR)/swig_processor -name "_swig_processor_impl.py" -delete 2>/dev/null || true

# Deep clean (including all generated files)
.PHONY: distclean
distclean: clean
	rm -rf .cmake/
	rm -rf CMakeFiles/
	rm -rf _skbuild/
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "CMakeCache.txt" -delete
	find . -name "cmake_install.cmake" -delete

# Format C++ code with clang-format (if available)
.PHONY: format
format:
	@if command -v clang-format >/dev/null 2>&1; then \
		echo "Formatting C++ files with clang-format..."; \
		find $(SRC_DIR) -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | xargs clang-format -i; \
		echo "Formatting complete."; \
	else \
		echo "clang-format not found. Please install it to format C++ code."; \
	fi

# CMake configuration (for manual builds)
.PHONY: cmake-configure
cmake-configure:
	cmake -B $(BUILD_DIR) -S . \
		-DCMAKE_BUILD_TYPE=Release \
		-DPython_EXECUTABLE=$(shell which $(PYTHON))

# CMake build (for manual builds)
.PHONY: cmake-build
cmake-build: cmake-configure
	cmake --build $(BUILD_DIR) --parallel
