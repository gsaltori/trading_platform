# tests/__init__.py
"""Test suite for the trading platform."""

from .test_suite import TestTradingPlatform, IntegrationTests, run_all_tests

__all__ = [
    'TestTradingPlatform',
    'IntegrationTests',
    'run_all_tests'
]
