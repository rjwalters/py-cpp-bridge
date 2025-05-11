# Makefile for cython_processor project
# Handles both debug and production builds

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
	@echo "  distclean    - Deep clean (including compiled extension)"
	@echo "  format       - Format C++ code with clang-format"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1      - Enable debug mode"

# Python command
PYTHON := python

# Default target
.PHONY: all
all: clean build install test

# Debug mode can be enabled with DEBUG=1
DEBUG ?= 0

# Debug-specific flags
ifeq ($(DEBUG), 1)
	DEBUG_FLAG := DEBUG=1
	BUILD_DIR := build_debug
	INSTALL_FLAGS := --no-deps --force-reinstall
else
	DEBUG_FLAG :=
	BUILD_DIR := build
	INSTALL_FLAGS :=
endif

# Build the extension module using setuptools
.PHONY: build
build:
	$(DEBUG_FLAG) $(PYTHON) setup.py build_ext

# Build and install the package in development mode
.PHONY: install
install: build
	$(DEBUG_FLAG) $(PYTHON) -m pip install -e . $(INSTALL_FLAGS)

# Run tests
.PHONY: test
test: build
	$(PYTHON) test.py

# Run typed example
.PHONY: typed-example
typed-example: build
	$(PYTHON) typed_example.py

# Clean the build artifacts
.PHONY: clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -f *.so
	rm -f *.o
	rm -f *.c
	rm -f cython_processor.*.html
	rm -f cython_processor.*.cpp

# Deep clean (including compiled extension)
.PHONY: distclean
distclean: clean
	rm -f cython_processor*.so
	rm -f cython_processor*.pyd
	rm -f cython_processor.cpp

# Build with debug settings
.PHONY: debug
debug:
	DEBUG=1 $(MAKE) build

# Format C++ code with clang-format (if available)
.PHONY: format
format:
	@which clang-format > /dev/null && clang-format -i *.cpp *.hpp || echo "clang-format not found"

