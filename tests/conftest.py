"""Pytest configuration and fixtures for qldpc_sim tests."""

import pytest


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {"key": "value"}
