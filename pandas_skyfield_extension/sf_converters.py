from __future__ import annotations

import pandas as pd
import skyfield.api as sf
from pandas.api.types import is_list_like


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
