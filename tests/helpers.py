"""Shared builders for the tests (plain functions, cached, so no fixtures are needed)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from samples.generate_sample import make_document_photo, make_page


@lru_cache(maxsize=None)
def _scene(kind: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "default":
        return make_document_photo()
    if kind == "rotated":
        rel = np.float32([[0.5, 0.05], [0.92, 0.5], [0.5, 0.95], [0.08, 0.5]])
        return make_document_photo(corners_rel=rel)
    if kind == "large":
        rel = np.float32([[0.03, 0.03], [0.97, 0.04], [0.96, 0.96], [0.04, 0.95]])
        return make_document_photo(corners_rel=rel)
    raise KeyError(kind)


def scene(kind: str = "default") -> tuple[np.ndarray, np.ndarray]:
    """Return a *copy* of ``(photo, true_corners)`` so tests cannot affect each other."""
    photo, corners = _scene(kind)
    return photo.copy(), corners.copy()


def page() -> np.ndarray:
    return make_page()
