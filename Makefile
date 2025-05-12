# Makefile for py-cpp-bridge project
# Handles both debug and production builds

# Source and build directories
SRC_DIR := src
CYTHON_SRC_DIR := $(SRC_DIR)/cython_processor
COMMON_SRC_DIR := $(SRC_DIR)/common
PYBIND_SRC_DIR := $(SRC_DIR)/pybind_processor
TEST_DIR := tests
BUILD_DIR := build
BUILD_DEBUG_DIR := build_debug

# Python command and version
PYTHON := python
PY_VERSION := $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PLATFORM := $(shell $(PYTHON) -c "import sys; print(sys.platform)")

# Get machine architecture for build directories
ifeq ($(PLATFORM), darwin)
    ARCH := $(shell uname -m)
    LIB_SUBDIR := lib.macosx-*-$(ARCH)-cpython-*
else ifeq ($(PLATFORM), linux)
    ARCH := $(shell uname -m)
    LIB_SUBDIR := lib.linux-*-$(ARCH)-cpython-*
else ifeq ($(PLATFORM), win32)
    # For Windows, we need a different approach
    ARCH := $(shell $(PYTHON) -c "import platform; print(platform.machine().lower())")
    LIB_SUBDIR := lib.win-*
endif

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all          - Default target, same as 'build'"
	@echo "  build        - Build the Cython extension"
	@echo "  install      - Install the package in development mode"
	@echo "  debug        - Build with debug settings"
	@echo "  test         - Run the test script"
	@echo "  typed-example - Run the typed example script"
	@echo "  clean        - Remove build artifacts"
	@echo "  distclean    - Deep clean (including compiled extension and eggs)"
	@echo "  format       - Format C++ code with clang-format"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1      - Enable debug mode"

# Default target
.PHONY: all
all: build

# Debug mode can be enabled with DEBUG=1
DEBUG ?= 0

# Debug-specific flags
ifeq ($(DEBUG), 1)
	DEBUG_FLAG := DEBUG=1
	ACTIVE_BUILD_DIR := $(BUILD_DEBUG_DIR)
	INSTALL_FLAGS := --no-deps --force-reinstall
else
	DEBUG_FLAG :=
	ACTIVE_BUILD_DIR := $(BUILD_DIR)
	INSTALL_FLAGS :=
endif

# Build the extension module using setuptools
.PHONY: build
build:
	# Build using the common C++ code
	$(DEBUG_FLAG) $(PYTHON) setup.py build_ext --inplace


.PHONY: install-python
install-python:
	$(DEBUG_FLAG) $(PYTHON) -m pip install --upgrade pip
	$(DEBUG_FLAG) $(PYTHON) -m pip install -r requirements.txt

# Build and install the package in development mode
.PHONY: install
install: install-python build
	$(DEBUG_FLAG) $(PYTHON) -m pip install -e . $(INSTALL_FLAGS)

# Run tests
.PHONY: test
test: build
	$(PYTHON) $(TEST_DIR)/test.py
	$(PYTHON) $(TEST_DIR)/typed_example.py

# Clean the build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)/
	rm -rf $(BUILD_DEBUG_DIR)/
	rm -rf dist/
	find . -name "*.so" -delete
	find . -name "*.pyd" -delete
	find . -name "*.o" -delete
	find . -name "*.c" -not -path "*/\.git/*" -delete
	find $(CYTHON_SRC_DIR) -name "*.cpp" -not -name "cpp_processor.cpp" -delete
	find $(CYTHON_SRC_DIR) -name "*.html" -delete

# Deep clean (including compiled extension and eggs)
.PHONY: distclean
distclean: clean
	rm -rf *.egg-info/
	rm -rf $(SRC_DIR)/*.egg-info/
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Build with debug settings
.PHONY: debug
debug:
	DEBUG=1 $(MAKE) build

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
