from __future__ import annotations

from typing import Any

import pandas as pd
import skyfield.api as sf
from skyfield import (
    positionlib,
    vectorlib,
)

from pandas_skyfield_extension.position_extension import SkyfieldPositionExtensionArray
from pandas_skyfield_extension.sf_converters import to_sf_time
from pandas_skyfield_extension.typing import Frame  # noqa: F401


def sf_position_to_series(position: positionlib.ICRF, **kwargs) -> pd.Series:
    """Convert a Skyfield position to a pandas Series.

    Parameters
    ----------
    position: positionlib.ICRF
        The position to convert.
    kwargs: dict
        Additional keyword arguments to pass to the pd.Series constructor.

    Returns
    -------
    pd.Series
        A Series containing the position.
    """
    return pd.Series(SkyfieldPositionExtensionArray(position), **kwargs)


def at(
    obj: vectorlib.VectorFunction,
    times: Any,
    set_datetime_idx: bool = True,
    ts: sf.Timescale | None = None,
    **kwargs,
) -> pd.Series:
    """Calculate the position of the object at the given times and return it as a pandas Series.

    Parameters
    ----------
    obj: skyfield.vectorlib.VectorFunction
        The object for which to calculate the position.
        Skyfield object supporting the .at() method, e.g. EarthSatellite, Topos, GeographicPosition, etc.
    times: datetime.datetime or sf.Time or convertible by pd.to_datetime()
        The times at which to calculate the position.
    set_datetime_idx: bool, default True
        If True and "index" is not in kwargs, set the index of the resulting Series to the given times.
    ts: sf.Timescale, optional
        The timescale to use for the conversion. If None, the default timescale will be used.
    kwargs: dict
        Additional keyword arguments to pass to the pd.Series constructor.

    Returns
    -------
    pd.Series
        A Series containing the positions at the given times.
    """
    # Convert times to Skyfield Time object
    times_sf = to_sf_time(times, ts)

    # Calculate positions at the given times
    pos: positionlib.ICRF = obj.at(times_sf)

    # If requested, set the index of the resulting Series to the given times
    if set_datetime_idx and "index" not in kwargs:
        # Avoid unnecessary copy of times if it's already a DatetimeIndex
        if isinstance(times, pd.DatetimeIndex):
            kwargs["index"] = times
        else:
            kwargs["index"] = pd.to_datetime(times_sf.utc_datetime())

    # If name of the resulting series is not provided, try to infer it from the object
    if "name" not in kwargs:
        name: str = (
            getattr(obj, "name", None) or getattr(obj, "target_name", None) or None
        )
        if name:
            kwargs["name"] = f"{name}"

    # Convert position to pandas Series
    return sf_position_to_series(pos, **kwargs)
