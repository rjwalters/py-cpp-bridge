"""
Pytest configuration for benchmarks.
"""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow benchmarks (1M+ elements)",
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (1M+ elements)")
