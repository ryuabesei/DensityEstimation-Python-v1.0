# densityestimation/Astrofunction/spice_loader.py
# GNU GPL v3

from __future__ import annotations


def load_spice(kernelpath: str) -> None:
    """
    Python equivalent of MATLAB loadSPICE.m using spiceypy.

    Parameters
    ----------
    kernelpath : str
        Path to a meta-kernel (.tm) or a single kernel to furnsh.
    """
    try:
        import spiceypy as spice
    except Exception as e:
        raise RuntimeError(
            "spiceypy が見つかりません。`pip install spiceypy` を実行してください。"
        ) from e

    # Clear existing kernels, then load the specified one
    spice.kclear()
    spice.furnsh(kernelpath)
