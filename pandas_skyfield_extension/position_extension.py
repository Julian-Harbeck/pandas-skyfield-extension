from __future__ import annotations

import datetime
import re
import sys
from typing import Any
import warnings

import astropy.units as u
import numpy as np
import pandas as pd
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    ExtensionScalarOpsMixin,
    register_extension_dtype,
    register_series_accessor,
)
from pandas.api.types import is_array_like, is_scalar
from pandas.compat import set_function_name
from pandas.core.algorithms import take
from pandas.core.dtypes.generic import ABCIndex, ABCSeries, ABCDataFrame
from pandas.core.indexers import (
    check_array_indexer,
    getitem_returns_view,
)
from pandas.util._exceptions import find_stack_level
import pandas_units_extension as pue
import skyfield.api as sf
from skyfield import positionlib, toposlib

from pandas_skyfield_extension.options import config
from pandas_skyfield_extension.typing import Frame


def as_position(obj) -> positionlib.ICRF:
    """Convert the input to a position, if possible."""
    if isinstance(obj, positionlib.ICRF):
        return obj
    elif isinstance(obj, SkyfieldPositionExtensionArray):
        return obj.position
    elif isinstance(obj, list) and len(obj) == 1:
        # positionlib.ICRF objects do not have an __array__ method, during sanitize_array pandas therefore calls list(obj)
        # Therefore we need to unpack the list here
        return as_position(obj[0])
    else:
        raise TypeError(f"Cannot convert {obj} to a position.")


def _copy_sf_time(t: sf.Time) -> sf.Time:
    """Create a copy of a Skyfield Time object."""
    # tt_fraction is subtracted from tt as constructor adds tt_fraction to tt, 
    # so it is needed to subtract it here to get the same time.
    return sf.Time(t.ts, t.tt - t.tt_fraction, t.tt_fraction)


def _copy_position(pos: positionlib.ICRF) -> positionlib.ICRF:
    """Create a copy of the position."""
    return pos.__class__(
        position_au=pos.xyz.au.copy(),
        velocity_au_per_d=pos.velocity.au_per_d.copy(),
        t=_copy_sf_time(pos.t),
        center=pos.center,
        target=pos.target
    )


def convert(pos: positionlib.ICRF, dtype: SkyfieldPositionDtype):
    raise NotImplementedError


@register_extension_dtype
class SkyfieldPositionDtype(ExtensionDtype):
    """Description of the units type.

    The name is formed as "unit[.*]" where the inside of the square
    brackets must be a unit name as understood by astropy units.
    """

    BASE_NAME = "skyfield_position"

    type = positionlib.ICRF
    frame = positionlib.ICRF
    kind = "O"

    _is_numeric = False
    _metadata = ("frame",)

    def __init__(self, frame = None):
        if frame is None:
            self.frame = positionlib.ICRF
        elif issubclass(frame, positionlib.ICRF):
            self.frame = frame
        else:
            raise ValueError(f"Invalid frame: {frame}")

    @classmethod
    def construct_from_string(cls, string: str) -> "SkyfieldPositionDtype":
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )

        # If just the base name create Dtype with base ICRF frame
        if string == cls.BASE_NAME:
            return cls()

        # Parse the string to extract the frame name
        match = re.match(f"{cls.BASE_NAME}\\[(?P<name>.*)\\]$", string)
        if not match:
            raise TypeError(f"Cannot construct a 'SkyfieldPositionDtype' from '{string}'")

        # Map from string to frame class
        str2frame_map: dict[str, type] = {
            "ICRF": positionlib.ICRF,
            "ICRS": positionlib.ICRS,
            "SSB": positionlib.SSB,
            "Geometric": positionlib.Geometric,
            "Barycentric": positionlib.Barycentric,
            "Astrometric": positionlib.Astrometric,
            "Apparent": positionlib.Apparent,
            "Geocentric": positionlib.Geocentric,
        }

        # Check if the frame name is valid and construct the Dtype
        if match["name"] not in str2frame_map:
            raise ValueError(f"Unknown frame name: {match['name']}.\nSupported frames are: {', '.join(str2frame_map.keys())}")
        return cls(str2frame_map[match["name"]])

    @classmethod
    def construct_array_type(cls) -> type:
        """Associated extension array."""
        return SkyfieldPositionExtensionArray

    @property
    def name(self) -> str:
        return f"{self.BASE_NAME}[{self.frame}]"

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.frame}")'

class SkyfieldPositionExtensionArray(ExtensionArray, ExtensionScalarOpsMixin):
    """Pandas extension array supporting skyfield positions."""

    def __init__(
        self, position: positionlib.ICRF | SkyfieldPositionExtensionArray, frame: type | None = None, copy: bool = True
    ):
        position = as_position(position)
        if position is not None and not isinstance(position, positionlib.ICRF):
            raise ValueError(f"Invalid position type: {type(position)}")
        if frame is not None and frame is not type(position):
            raise ValueError(f"Frame {frame} does not match position type {type(position)} and conversion is not yet implemented.")
        else:
            frame = type(position)
        if copy:
            position = _copy_position(position)
        self._dtype: SkyfieldPositionDtype = SkyfieldPositionDtype(frame)
        self._position: positionlib.ICRF | None = position

    @property
    def position(self) -> positionlib.ICRF | None:
        """The position itself."""
        return self._position

    @property
    def frame(self) -> positionlib.ICRF | None:
        """The frame of the position."""
        return self.dtype.frame

    @property
    def dtype(self) -> SkyfieldPositionDtype:
        return self._dtype

    def __len__(self) -> int:
        # Use length of the underlying Distance array, which is the same as the length of the position array
        return len(self.position.xyz.length().au)

    def __array__(self, dtype=object, copy=None) -> np.ndarray:
        """Implicit conversion to numpy array."""
        # Create array depending on dtype
        if dtype == object:
            if copy == False:
                raise ValueError("Cannot return object array without copy, as each element has to be its own position object.")
            arr = np.array(list(self.position), dtype=object)
            # Conversion requires a copy, so copy flag will be set to True
            copy = True
        elif dtype:
            arr = self.position.astype(dtype, copy=copy)
        else:
            arr = np.asarray(self.position, copy=copy)

        # Set writable flag depending on self._readonly and only when no copy was made
        if self._readonly and copy is not True:
            arr.setflags(write=False)

        return arr

    @property
    def nbytes(self) -> int:
        return sys.getsizeof(self.position)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False) -> "SkyfieldPositionExtensionArray":
        if dtype:
            result = cls(scalars, frame=dtype.frame, copy=copy)
        else:
            result = cls(scalars, copy=copy)
        return result

    def view(self, dtype=None) -> "SkyfieldPositionExtensionArray":
        """Create a new object with same data behind it."""
        # TODO: Useful also for 0.25???
        if dtype is not None:
            # TODO: Perhaps implement?
            raise NotImplementedError(dtype)
        result = SkyfieldPositionExtensionArray.__new__(SkyfieldPositionExtensionArray)
        result._dtype = self.dtype
        result._position = self._position
        return result

    def _formatter(self, boxed: bool = False):
        """Formatter to always include unit name in the output.

        TODO: Not sure if this is the best (differ on boxed?)
        """
        def _f(x):
            if isinstance(x, positionlib.ICRF):
                frame_name = self.dtype.frame.__name__
                formatter = {"float_kind": lambda x: f"{x:9.3f}"}
                position_km: str = np.array2string(x.xyz.km, formatter=formatter)
                velocity_km_s: str = np.array2string(x.velocity.km_per_s, formatter=formatter)
                t: str = x.t.utc_datetime().isoformat()
                return f"{frame_name} position {position_km} km and velocity {velocity_km_s} km/s at time {t} with center={x.center} target={x.target}"
            else:
                return f"{x} {self.dtype.frame}"
        return _f

    def __getitem__(self, item):
        # Return normal skyfield position object for singular item
        if is_scalar(item):
            return self._position[item]

        # Use pandas utility function to check and convert the item to a valid indexer
        item = check_array_indexer(self, item)

        # Create new UnitsExtensionArray
        result: SkyfieldPositionExtensionArray = SkyfieldPositionExtensionArray(self.position[item], frame=self.dtype.frame)

        # If the result is a view, keep read-only flag
        if getitem_returns_view(self, item):
            result._readonly = self._readonly

        return result

    def __setitem__(self, key, value):
        raise NotImplementedError

    def take(self, indices, allow_fill=False, fill_value=None) -> "SkyfieldPositionExtensionArray":
        """Integer-based selection of items."""
        if allow_fill:
            if fill_value is None or np.isnan(fill_value):
                fill_value = np.nan
            else:
                fill_value = fill_value.value
        # values = take(self.value, indices, allow_fill=allow_fill, fill_value=fill_value)
        # TODO Skipping the whole allow_fill as not sure 
        return SkyfieldPositionExtensionArray(self._position[indices], frame=self.dtype.frame)

    def isna(self):
        # TODO Can positions be NA?
        return np.isnan(self._position)

    @classmethod
    def _create_method(cls, op, coerce_to_dtype=True, result_dtype=None):
        # Overridden from the default variant
        # to by-pass conversion to numpy arrays.

        # Get info about the operator
        op_name = getattr(op, "__name__", str(op))
        is_comparison = op_name in [
            "eq", "__eq__",
            "ne", "__ne__",
            "lt", "__lt__",
            "gt", "__gt__",
            "le", "__le__",
            "ge", "__ge__",
        ]
        is_equality = op_name in ["eq", "ne", "__eq__", "__ne__"]
        is_divmod = op_name in ["divmod", "__divmod__", "rdivmod", "__rdivmod__"]
        is_not_supported = op_name in [
            "eg", "__eg__",
            "ne", "__ne__",
            "lt", "__lt__",
            "gt", "__gt__",
            "le", "__le__",
            "ge", "__ge__",
            "add", "__add__", "radd", "__radd__",
            "multiply", "__mul__", "rmul",
            "truediv", "__truediv__", "rtruediv", "__rtruediv__",
            "floor_divide", "__floordiv__", "rfloordiv",
            "divmod", "__divmod__", "rdivmod", "__rdivmod__",
        ]

        def _invalid_operator():
            if is_equality:
                return NotImplemented
            else:
                raise TypeError

        def _binop(self, other):
            if isinstance(other, (ABCIndex, ABCSeries, ABCDataFrame)):
                # rely on pandas to unbox and dispatch to us
                return NotImplemented

            elif is_scalar(other):
                if is_comparison:
                    return NotImplemented

            elif is_array_like(other):
                if not isinstance(other.dtype, SkyfieldPositionDtype):
                    if is_comparison:
                        return _invalid_operator()

            if is_not_supported:
                raise NotImplementedError(f"Operator {op_name} is not supported for SkyfieldPositionExtensionArray.")
            # Convert the thing to a skyfield position
            self_pos = as_position(self)
            other_pos = as_position(other)

            if is_comparison:
                # Try apply conversion (we need same type for comparisons)
                if is_array_like(other) and other.dtype != self.dtype:
                    try:
                        other_pos = convert(other_pos, self_pos.dtype.frame)
                    except Exception:
                        return _invalid_operator()

            result_pos = op(self_pos, other_pos)

            if coerce_to_dtype:
                return cls(result_pos)
            return result_pos

        return set_function_name(_binop, op_name, cls)

    def copy(self, deep=False) -> "SkyfieldPositionExtensionArray":
        return self.__class__(_copy_position(self.position), self.frame, copy=True)

SkyfieldPositionExtensionArray._add_arithmetic_ops()
# SkyfieldPositionExtensionArray._add_comparison_ops()


@register_series_accessor("skyfield_position")
class SkyfieldPositionSeriesAccessor:
    """Accessor adding skyfield position functionality to series."""

    def __init__(self, obj):
        # Inspired by fletcher
        if not isinstance(obj.array, SkyfieldPositionExtensionArray):
            raise AttributeError("Only SkyfieldPositionExtensionArray has skyfield_position accessor.")
        self._obj = obj

    @staticmethod
    def _result_dtype(result) -> str:
        if isinstance(result, u.Quantity):
            return "unit"
        elif isinstance(result, positionlib.ICRF):
            return "skyfield_position"
        else:
            return None

    def _wrap_series(self, result, name: str = None, dtype = None) -> pd.Series:
        """Construct a series with different data but same index and name."""
        if not name:
            name: str = self._obj.name
        if not dtype:
            dtype: str = self._result_dtype(result)
        return pd.Series(result, name=name, index=self._obj.index, dtype=dtype)
    
    def _wrap_frame(self, result_dict: dict[str, Any]) -> pd.DataFrame:
        """Construct a DataFrame from a result dict mapping the keys to the column names and the values to the column data, with same index as the original series."""
        for key, value in result_dict.items():
            result_dict[key] = self._wrap_series(value, name=key, dtype=self._result_dtype(value))
        return pd.DataFrame(result_dict)

    @property
    def position(self) -> positionlib.ICRF:
        """The position."""
        return self._obj.array.position

    @property
    def frame(self) -> positionlib.ICRF | None:
        """The frame of the position."""
        return self._obj.array.frame

    @property
    def center(self):
        """The center of the position."""
        return self.position.center

    @property
    def target(self):
        """The target of the position."""
        return self.position.target

    @property
    def subpoint(self) -> toposlib.GeographicPosition:
        """Calculate the subpoint of the position."""
        return sf.wgs84.subpoint_of(self.position)
    
    @property
    def time(self) -> datetime.datetime:
        """The time of the position."""
        return self._wrap_series(self.position.t.utc_datetime(), name="time")

    @property
    def xyz(self) -> pd.DataFrame:
        """The x, y, z components of the position vector as a DataFrame."""
        x, y, z = self.position.xyz.to(config.length_unit)
        return self._wrap_frame({"x": x, "y": y, "z": z})

    @property
    def length(self) -> pd.Series:
        """The length of the position vector."""
        return self._wrap_series(self.position.xyz.length().to(config.length_unit), name="length")

    @property
    def velocity(self) -> pd.DataFrame:
        """The x, y, z components of the velocity vector as a DataFrame."""
        vx, vy, vz = self.position.velocity.to(config.velocity_unit)
        return self._wrap_frame({"vx": vx, "vy": vy, "vz": vz})

    @property
    def speed(self) -> pd.Series:
        """The speed of the position vector."""
        return self._wrap_series(self.position.speed().to(config.velocity_unit), name="speed")

    def altaz(self) -> pd.DataFrame:
        """The (alt, az, distance) of the position relative to the observer's horizon as a DataFrame.
        
        Raises
        ------
        ValueError
            If the position of this series is not relative to an observer, i.e. if the center is an integer.
        """
        if isinstance(self.center, int):
            msg: str = f"Cannot calculate altaz for positions with center {self.center}. altaz() is only possible for positions relative to an observer."
            raise ValueError(msg)
        altitude, azimuth, distance = self.position.altaz()
        return self._wrap_frame({
            "altitude": altitude.to(config.angle_unit),
            "azimuth": azimuth.to(config.angle_unit),
            "distance": distance.to(config.length_unit),
        })

    def frame_xyz_and_velocity(self, frame: Frame) -> pd.DataFrame:
        """Converts the position to the given frames and returns the position and velocity in this frame as DataFrame.

        Parameters
        ----------
        frame: Frame
            One of the frames defined in skyfield.framelib.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the x, y and z coordinates and vx, vy and vz velocities in the new frame.
        """
        distance, velocity = self.position.frame_xyz_and_velocity(frame)
        x, y, z = distance.to(config.length_unit)
        vx, vy, vz = velocity.to(config.velocity_unit)
        return self._wrap_frame({
            "x": x, "y": y, "z": z,
            "vx": vx, "vy": vy, "vz": vz,
        })


    def __getattr__(self, name):
        """Delegate attribute access to the position object."""
        if not name.startswith("_"):
            msg: str = f"Attribute {name} not found in SkyfieldPositionSeriesAccessor, delegating to position object. Consider using the position property directly instead."
            warnings.warn(msg, stacklevel=find_stack_level())
        return getattr(self.position, name)

    def to_astropy_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with the position's time, x, y, z components, length, velocity and speed as columns.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the time of the position, x, y and z coordinates and vx, vy and vz velocities, length and speed.
        """
        return pd.concat([
                self.time,
                self.xyz,
                self.velocity,
                self.length,
                self.speed,
            ],
            axis=1
        )

    def to_geodataframe(self) -> "gpd.GeoDataFrame":
        """Create a GeoDataFrame with the position's subpoint as geometry and the time, x, y, z components, length, velocity components and speed as columns.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame containing the time of the position, x, y and z coordinates and vx, vy and vz velocities, length and speed, with the subpoint as geometry.

        Raises
        ------
        ImportError
            If the optional geopandas dependency is not installed.
        """
        # Try to load optional dependencies
        try:
            import geopandas as gpd
        except ImportError:
            msg: str = f"Optional geopandas dependency not installed, cannot convert to GeoDataFrame. Please install geopandas to use this feature."
            raise ImportError(msg)

        # Get subpoint of position in WGS 84 and create GeoPandas geometry from that
        subpoint: toposlib.GeographicPosition = self.subpoint
        geometry = gpd.points_from_xy(subpoint.longitude.to(u.deg), subpoint.latitude.to(u.deg))

        # Get position DataFrame and convert to GeoDataFrame
        df: pd.DataFrame = self.to_astropy_dataframe()
        return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
