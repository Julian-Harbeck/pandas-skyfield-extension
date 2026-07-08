from __future__ import annotations

from typing import TypeAlias

from skyfield import framelib

# Create type alias for all frames
Frame: TypeAlias = (
    framelib.ICRS
    | framelib.mean_equator_and_equinox_of_date
    | framelib.true_equator_and_equinox_of_date
    | type[framelib.tirs]
    | type[framelib.itrs]
    | type[framelib.ecliptic_frame]
    | type[framelib.InertialFrame]
)
"""TypeAlias for all frame classes of framelib."""
