"""Shared pytest configuration for beyond32.

The ``slow`` marker (registered in pyproject.toml) tags the six end-to-end cross-checks
(up to about 20 s each on an idle laptop, together about half of the suite's runtime); run
``pytest -m "not slow"`` for the quick suite.
"""
