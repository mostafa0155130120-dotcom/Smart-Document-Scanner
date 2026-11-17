"""Smoke tests for the Streamlit UI using Streamlit's own AppTest runner."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_starts_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert len(at.exception) == 0


def test_sample_document_is_scanned():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio(key="source").set_value("Sample").run()
    assert len(at.exception) == 0
    assert len(at.error) == 0
