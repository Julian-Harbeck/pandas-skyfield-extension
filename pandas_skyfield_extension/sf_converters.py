from __future__ import annotations

from typing import Any

import astropy.units as u
import pandas as pd
import pandas_units_extension as pue
import skyfield.api as sf
from pandas.api.types import is_list_like


def _is_UnitsDtype_of_physical_type(dtype: Any, physical_type: str) -> bool:
    """Check if the dtype is a UnitsDtype of the specified physical type."""
    return (
        isinstance(dtype, pue.UnitsDtype) and dtype.unit.physical_type == physical_type
    )


def to_sf_time(times, ts: sf.Timescale | None = None) -> sf.Time:
    """Convert the input to a Skyfield Time object.

    Parameters
    ----------
    times: datetime.datetime or sf.Time or convertible by pd.to_datetime
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
