#!/usr/bin/env python3
"""
Generate CPU flamegraphs for binding implementations using py-spy.

Flamegraphs visualize where time is spent during execution, helping identify
performance bottlenecks in binding overhead vs. C++ computation.

Requirements:
    - py-spy>=0.3.14 (install with: pip install py-spy)

Usage:
    python generate_flamegraph.py              # Profile all bindings
    python generate_flamegraph.py cython       # Profile specific binding
    python generate_flamegraph.py --size 100000  # Use larger array size

Output:
    SVG flamegraphs in benchmarks/flamegraphs/
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


# Available bindings to profile
AVAILABLE_BINDINGS = [
    'cython',
    'pybind11',
    'nanobind',
    'swig',
    'ctypes',
    'cffi',
    'hpy'
]


def check_py_spy():
    """Check if py-spy is installed."""
    try:
        result = subprocess.run(
            ['py-spy', '--version'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def generate_profiling_script(binding_name: str, array_size: int, iterations: int) -> str:
    """
    Generate Python script to profile a specific binding.

    Args:
        binding_name: Name of binding (e.g., 'cython', 'pybind11')
        array_size: Size of array to process
        iterations: Number of iterations to run

    Returns:
        Python code as string
    """
    # Map binding names to module names
    module_map = {
        'cython': 'cython_processor',
        'pybind11': 'pybind_processor',
        'nanobind': 'nanobind_processor',
        'swig': 'swig_processor',
        'ctypes': 'ctypes_processor',
        'cffi': 'cffi_processor',
        'hpy': 'hpy_processor'
    }

    module_name = module_map.get(binding_name, f'{binding_name}_processor')

    script = f'''#!/usr/bin/env python3
"""Profiling script for {binding_name} binding."""

import numpy as np
import sys

try:
    from {module_name} import PyArrayProcessor
except ImportError as e:
    print(f"Cannot import {binding_name}: {{e}}", file=sys.stderr)
    sys.exit(1)

# Configuration
ARRAY_SIZE = {array_size}
ITERATIONS = {iterations}

# Get the correct numpy dtype
np_type = PyArrayProcessor.get_numpy_type_name("value")
data = np.arange(ARRAY_SIZE, dtype=np_type)

# Create processor
processor = PyArrayProcessor(ARRAY_SIZE)

# Warm up
for _ in range(10):
    result = processor.process_preallocated(data)

# Profile this loop
for i in range(ITERATIONS):
    result = processor.process_preallocated(data)

print(f"Completed {{ITERATIONS}} iterations with array size {{ARRAY_SIZE}}")
'''
    return script


def profile_binding(
    binding_name: str,
    array_size: int = 10_000,
    iterations: int = 1000,
    duration: int = 10,
    output_dir: Path = None
):
    """
    Profile a single binding using py-spy.

    Args:
        binding_name: Name of binding to profile
        array_size: Array size for profiling
        iterations: Number of iterations
        duration: Maximum profiling duration in seconds
        output_dir: Directory to save flamegraph
    """
    if output_dir is None:
        output_dir = Path('benchmarks/flamegraphs')

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{binding_name}.svg'

    print(f"\nProfiling {binding_name}...")
    print(f"  Array size: {array_size:,}")
    print(f"  Iterations: {iterations:,}")
    print(f"  Output: {output_file}")

    # Generate profiling script
    script_content = generate_profiling_script(binding_name, array_size, iterations)

    # Write script to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        # Run py-spy to profile the script
        cmd = [
            'py-spy',
            'record',
            '--rate', '100',  # Sample rate: 100 Hz
            '--output', str(output_file),
            '--format', 'flamegraph',
            '--',
            sys.executable,  # Use current Python interpreter
            script_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=duration + 10  # Add buffer to timeout
        )

        if result.returncode == 0:
            print(f"  ✓ Success! Flamegraph saved to {output_file}")
            return True
        else:
            print(f"  ✗ Failed to profile {binding_name}")
            if result.stderr:
                print(f"  Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout after {duration + 10} seconds")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    finally:
        # Clean up temporary script
        Path(script_path).unlink(missing_ok=True)


def check_binding_available(binding_name: str) -> bool:
    """Check if a binding module can be imported."""
    module_map = {
        'cython': 'cython_processor',
        'pybind11': 'pybind_processor',
        'nanobind': 'nanobind_processor',
        'swig': 'swig_processor',
        'ctypes': 'ctypes_processor',
        'cffi': 'cffi_processor',
        'hpy': 'hpy_processor'
    }

    module_name = module_map.get(binding_name, f'{binding_name}_processor')

    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate flamegraphs for Python-C++ bindings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Profile all available bindings
  python generate_flamegraph.py

  # Profile specific binding
  python generate_flamegraph.py cython

  # Use larger array and more iterations
  python generate_flamegraph.py --size 100000 --iterations 5000

  # Profile multiple specific bindings
  python generate_flamegraph.py cython pybind11 nanobind
'''
    )

    parser.add_argument(
        'bindings',
        nargs='*',
        choices=AVAILABLE_BINDINGS,
        help='Binding(s) to profile (default: all available)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=10_000,
        help='Array size for profiling (default: 10,000)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=1000,
        help='Number of iterations (default: 1,000)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Maximum profiling duration in seconds (default: 30)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('benchmarks/flamegraphs'),
        help='Output directory for flamegraphs (default: benchmarks/flamegraphs)'
    )

    args = parser.parse_args()

    # Check if py-spy is installed
    if not check_py_spy():
        print("ERROR: py-spy is not installed")
        print("\nInstall with:")
        print("  pip install py-spy")
        print("\nOr:")
        print("  make install-deps")
        sys.exit(1)

    # Determine which bindings to profile
    if args.bindings:
        bindings_to_profile = args.bindings
    else:
        # Profile all available bindings
        bindings_to_profile = [
            b for b in AVAILABLE_BINDINGS
            if check_binding_available(b)
        ]

        if not bindings_to_profile:
            print("ERROR: No bindings available to profile")
            print("\nPlease build the bindings first:")
            print("  make build")
            sys.exit(1)

    # Print configuration
    print("=" * 70)
    print("FLAMEGRAPH GENERATION")
    print("=" * 70)
    print(f"Bindings to profile: {', '.join(bindings_to_profile)}")
    print(f"Array size: {args.size:,}")
    print(f"Iterations: {args.iterations:,}")
    print(f"Output directory: {args.output}")
    print("=" * 70)

    # Profile each binding
    success_count = 0
    for binding in bindings_to_profile:
        # Check if binding is available
        if not check_binding_available(binding):
            print(f"\nSkipping {binding} (not available/built)")
            continue

        success = profile_binding(
            binding,
            array_size=args.size,
            iterations=args.iterations,
            duration=args.duration,
            output_dir=args.output
        )

        if success:
            success_count += 1

    # Print summary
    print("\n" + "=" * 70)
    print(f"Profiling complete: {success_count}/{len(bindings_to_profile)} successful")
    print(f"Flamegraphs saved to: {args.output}")
    print("=" * 70)

    # Instructions for viewing
    if success_count > 0:
        print("\nTo view flamegraphs:")
        print(f"  1. Open SVG files in a web browser:")
        print(f"     open {args.output}/*.svg")
        print(f"  2. Or use a flamegraph viewer")
        print("\nInterpreting flamegraphs:")
        print("  - Width = time spent in that function")
        print("  - Height = call stack depth")
        print("  - Colors = different code modules/files")
        print("  - Look for wide bars = performance bottlenecks")


if __name__ == '__main__':
    main()
