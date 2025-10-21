# GNU GPL v3
from __future__ import annotations

from typing import Tuple


def timediff(UT1_UTC: float, TAI_UTC: float) -> Tuple[float, float, float, float, float]:
    """
    Compute standard time differences [seconds].

    Equivalent to MATLAB timediff.m

    Parameters
    ----------
    UT1_UTC : float
        UT1 - UTC [s]
    TAI_UTC : float
        TAI - UTC [s]

    Returns
    -------
    (UT1_TAI, UTC_GPS, UT1_GPS, TT_UTC, GPS_UTC) : tuple of float
        All in seconds.
    """
    TT_TAI = 32.184       # TT - TAI [s]
    GPS_TAI = -19.0       # GPS - TAI [s]
    TT_GPS = TT_TAI - GPS_TAI
    TAI_GPS = -GPS_TAI

    UT1_TAI = UT1_UTC - TAI_UTC
    UTC_TAI = -TAI_UTC

    UTC_GPS = UTC_TAI - GPS_TAI
    UT1_GPS = UT1_TAI - GPS_TAI
    TT_UTC = TT_TAI - UTC_TAI
    GPS_UTC = GPS_TAI - UTC_TAI

    return UT1_TAI, UTC_GPS, UT1_GPS, TT_UTC, GPS_UTC

__all__ = ["timediff"]
