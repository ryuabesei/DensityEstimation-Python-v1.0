# GNU GPL v3
from __future__ import annotations


def sign_(a: float, b: float) -> float:
    """
    Returns |a| with the sign of b.
    Equivalent to MATLAB sign_.m
    """
    return abs(a) if b >= 0 else -abs(a)

__all__ = ["sign_"]
