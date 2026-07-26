"""Repository-local test support package.

Keeping this package explicit prevents unrelated site-packages named ``tests``
from shadowing the shared fixtures imported by the Phase-2 contract tests.
"""
