from __future__ import annotations

from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
import pandas_units_extension as pue
from skyfield import (
    api as sf,
    positionlib,
)
from pandas.api.types import is_list_like


####################
# Helper functions #
####################


def _as_3xN_array(value: Any) -> np.ndarray:
    """Normalize a Skyfield vector to a 3×N array for concatenation."""
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    return arr


def _is_UnitsDtype_of_physical_type(dtype: Any, physical_type: str) -> bool:
    """Check if the dtype is a UnitsDtype of the specified physical type."""
    return (
        isinstance(dtype, pue.UnitsDtype) and dtype.unit.physical_type == physical_type  # type: ignore[union-attr]
    )


##################
# Concatenations #
##################


def concat_sf_positions(obj: Any) -> positionlib.ICRF:
    """Concatenate a list of Skyfield position objects into a single position object.

    Parameters
    ----------
    obj: Any
        The list of objects to concatenate. Each object should be convertible to a Skyfield position.

    Returns
    -------
    positionlib.ICRF
        The concatenated Skyfield position object.
    """
    # Convert all objects to Skyfield position objects
    try:
        positions: list[positionlib.ICRF] = [to_sf_position(x) for x in obj]
    except (TypeError, ValueError):
        raise TypeError(f"Cannot convert list of {type(obj[0])} to a position.")

    # Use first position as reference for center, target and output type
    first_pos: positionlib.ICRF = positions[0]
    cls = type(first_pos)

    # Check that all positions have the same center and target, otherwise raise an error
    if not all(
        pos.center == first_pos.center and pos.target == first_pos.target
        for pos in positions
    ):
        raise ValueError(
            "All positions must have the same center and target to be combined into a single position."
        )

    # Concatenate the positions, velocities, and times into a single position object
    position_au = np.concatenate([_as_3xN_array(x.xyz.au) for x in positions], axis=1)
    velocity_au_per_d = np.concatenate(
        [_as_3xN_array(x.velocity.au_per_d) for x in positions], axis=1
    )
    return cls(
        position_au=position_au,
        velocity_au_per_d=velocity_au_per_d,
        t=concat_sf_times([x.t for x in positions]),
        center=first_pos.center,
        target=first_pos.target,
    )


def concat_sf_times(times: list[sf.Time], ts: sf.Timescale | None = None) -> sf.Time:
    """Concatenate a list of Skyfield Time objects into a single Time object.

    Parameters
    ----------
    times: list[sf.Time]
        The list of Skyfield Time objects to concatenate.
    ts: sf.Timescale, optional
        The timescale to use for the concatenation. If None, the timescale of the first Time object will be used.

    Returns
    -------
    sf.Time
        The concatenated Skyfield Time object.
    """
    if not times:
        raise ValueError("times must not be empty")

    # Extract the timescale from the first Time object if not provided
    if ts is None:
        ts = getattr(times[0], "ts", None)
    whole_parts: list[np.ndarray] = []
    frac_parts: list[np.ndarray] = []

    # Iterate over list of Time objects and append whole and fractional Julian dates
    for t in times:
        if not isinstance(t, sf.Time):
            raise TypeError(f"Object {t} is not a skyfield Time")
        if t.ts is not ts:
            raise ValueError("All Time objects must use the same Timescale instance")

        # Convert zerodim Time objects to 1D arrays for concatenation
        whole_parts.append(np.atleast_1d(t.whole))
        frac_parts.append(np.atleast_1d(t.tt_fraction))

    # Concatenate the whole and fractional parts into single arrays
    whole = np.concatenate(whole_parts)
    frac = np.concatenate(frac_parts)

    return sf.Time(ts, whole, frac)


###############################
# High level Skyfield objects #
###############################


def to_sf_position(obj) -> positionlib.ICRF:
    """Convert the input to a Skyfield position object.

    Parameters
    ----------
    obj: Any
        The object to convert to a Skyfield position.

    Returns
    -------
    positionlib.ICRF
        The corresponding Skyfield position object.
    """
    if isinstance(obj, positionlib.ICRF):
        return obj

    # Handle SkyfieldExtensionArray objects without importing the class directly
    elif hasattr(obj, "position") and isinstance(obj.position, positionlib.ICRF):
        return obj.position

    elif isinstance(obj, list) and len(obj) == 1:
        # positionlib.ICRF objects do not have an __array__ method, during sanitize_array pandas therefore calls list(obj)
        # Therefore we need to unpack the list here
        return to_sf_position(obj[0])

    elif (
        is_list_like(obj) and len(obj) > 1 and all(type(x) is type(obj[0]) for x in obj)
    ):
        return concat_sf_positions(obj)
    else:
        raise TypeError(f"Cannot convert {obj} to a position.")


def to_sf_time(times, ts: sf.Timescale | None = None) -> sf.Time:
    """Convert the input to a Skyfield Time object.

    Parameters
    ----------
    times: datetime.datetime or sf.Time or list thereof or any object convertible by pd.to_datetime
        The times to convert.
    ts: sf.Timescale, optional
        The timescale to use for the conversion. If None, the default timescale will be used.

    Returns
    -------
    sf.Time
        The converted times as a Skyfield Time object.
    """
    if isinstance(times, sf.Time):
        return times

    # Handle list of Skyfield Time objects
    if is_list_like(times) and all(isinstance(t, sf.Time) for t in times):
        return concat_sf_times(times, ts)

    # Use default timescale if not provided
    if not ts:
        ts = sf.load.timescale()

    # Convert datetime-like objects to array of datetimes
    dts = pd.to_datetime(times).to_pydatetime()

    # If times is a single datetime, convert it to a list of one datetime
    if not is_list_like(dts):
        dts = [dts]

    # Convert to Skyfield Time object
    return ts.from_datetimes(dts)


##########################
# Skyfield Unit objects #
##########################


def to_sf_angle(obj: Any) -> sf.Angle:
    """Convert the input to a Skyfield Angle object, assumes radians if not set through astropy unit.

    Parameters
    ----------
    obj: Any
        The object to convert. Can be an Angle, a Series, or a list-like of scalars.

    Returns
    -------
    Angle
        The converted object as a Skyfield Angle.
    """
    # If the object is already an Angle, return it as is
    if isinstance(obj, sf.Angle):
        return obj

    # If the object is a Series with UnitsDtype of physical type angle, convert to Angle using the values and units otherwise just values
    elif isinstance(obj, pd.Series):
        if _is_UnitsDtype_of_physical_type(obj.dtype, u.physical.angle):
            return sf.Angle(
                radians=obj.array.to_quantity().to_value(u.rad)
            )  # TODO replace with units.to_quantity()
        return sf.Angle(radians=obj.values)

    # If the object is a DataFrame with 3 columns, assume it's x, y, z and convert to Distance
    elif isinstance(obj, pd.DataFrame):
        # Convert each Series to Angle and then combine into a single Angle object
        for col in obj.columns:
            obj[col] = to_sf_angle(obj[col]).radians

        # Convert the columns to Angle using the values
        return sf.Angle(radians=obj.T.values)

    elif isinstance(obj, u.Quantity):
        return sf.Angle(radians=obj.to_value(u.rad))

    try:
        return sf.Angle(radians=obj)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Cannot convert object of type {type(obj)} to a Skyfield Angle.\n{e}"
        )


def to_sf_distance(obj: Any) -> sf.Distance:
    """Convert the input to a Skyfield Distance object.

    Parameters
    ----------
    obj: Any
        The object to convert. Can be a Distance, a Series, DataFrame or a list-like of three scalars or arrays.

    Returns
    -------
    Distance
        The converted object as a Skyfield Distance.
    """
    # If the object is already a Distance, return it as is
    if isinstance(obj, sf.Distance):
        return obj

    # If the object is a Series with UnitsDtype of physical type length, convert to Distance using the values and units otherwise just values
    elif isinstance(obj, pd.Series):
        if _is_UnitsDtype_of_physical_type(obj.dtype, u.physical.length):
            return sf.Distance(
                obj.array.to_quantity().to_value(u.au)
            )  # TODO replace with units.to_quantity()
        return sf.Distance(obj.values)

    # If the object is a DataFrame with 3 columns, assume it's x, y, z and convert to Distance
    elif isinstance(obj, pd.DataFrame) and obj.shape[1] == 3:
        # Convert each Series to Distance and then combine into a single Distance object
        for col in obj.columns:
            obj[col] = to_sf_distance(obj[col]).au

        # Assume the columns are in the order of x, y, z and convert to Distance using the values
        return sf.Distance(obj.T.values)

    elif isinstance(obj, u.Quantity):
        return sf.Distance(obj.to_value(u.au))

    try:
        return sf.Distance(obj)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Cannot convert object of type {type(obj)} to a Skyfield Distance.\n{e}"
        )


def to_sf_velocity(obj: Any) -> sf.Velocity:
    """Convert the input to a Skyfield Velocity object.

    Parameters
    ----------
    obj: Any
        The object to convert. Can be a Velocity, a Series, DataFrame or a list-like of three scalars or arrays.

    Returns
    -------
    Velocity
        The converted object as a Skyfield Velocity.
    """
    # If the object is already a Velocity, return it as is
    if isinstance(obj, sf.Velocity):
        return obj

    # If the object is a Series with UnitsDtype of physical type velocity, convert to Velocity using the values and units otherwise just values
    elif isinstance(obj, pd.Series):
        if _is_UnitsDtype_of_physical_type(obj.dtype, u.physical.velocity):
            return sf.Velocity(
                obj.array.to_quantity().to_value(u.au / u.day)
            )  # TODO replace with units.to_quantity()
        return sf.Velocity(obj.values)

    # If the object is a DataFrame with 3 columns, assume it's vx, vy, vz and convert to Velocity
    elif isinstance(obj, pd.DataFrame) and obj.shape[1] == 3:
        # Convert each Series to Velocity and then combine into a single Velocity object
        for col in obj.columns:
            obj[col] = to_sf_velocity(obj[col]).au_per_d

        # Assume the columns are in the order of vx, vy, vz and convert to Velocity using the values
        return sf.Velocity(obj.T.values)

    elif isinstance(obj, u.Quantity):
        return sf.Velocity(obj.to_value(u.au / u.day))

    try:
        return sf.Velocity(obj)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Cannot convert object of type {type(obj)} to a Skyfield Velocity.\n{e}"
        )
