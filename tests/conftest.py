"""Shared pytest configuration for beyond32.

The ``slow`` marker (registered in pyproject.toml) tags tests that take more than
about a minute; run ``pytest -m "not slow"`` for the quick suite.
"""
