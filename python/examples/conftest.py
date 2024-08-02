import pytest
import os
import tempfile


@pytest.fixture
def device(request):
    return "cpu"