from __future__ import annotations

import datetime

import astropy.units as u
import numpy as np
import pandas as pd
import pandas_units_extension as pue
import pytest
import skyfield.api as sf

from pandas_skyfield_extension.sf_converters import (
    _is_UnitsDtype_of_physical_type,
    to_sf_angle,
    to_sf_distance,
    to_sf_time,
    to_sf_velocity
)

@pytest.fixture(params=[u.m, u.km, u.au], ids=["m", "km", "au"])
def length_unit(request) -> u.Unit:
    return request.param

@pytest.fixture(params=[u.m/u.s, u.km/u.s, u.au/u.day], ids=["m/s", "km/s", "au/day"])
def velocity_unit(request) -> u.Unit:
    return request.param

@pytest.fixture
def ts() -> sf.Timescale:
    return sf.load.timescale()

@pytest.fixture
def time(ts) -> sf.Time:
    return ts.utc(2000, 1, 1, 0, 0, 0)

@pytest.fixture
def time_isoformat() -> str:
    return "2000-01-01T00:00:00Z"

@pytest.fixture
def three_sf_times(ts) -> sf.Time:
    return ts.utc(2000, 1, [1, 2, 3], 0, 0, 0)


class TestToSfTime:
    def test_sf_time(self, time):
        result: sf.Time = to_sf_time(time)
        assert result is time

    def test_datetime(self, time, time_isoformat):
        dt = datetime.datetime.fromisoformat(time_isoformat)
        result: sf.Time = to_sf_time(dt)
        assert result == time

    def test_timestamp(self, time, time_isoformat):
        dt = pd.Timestamp(time_isoformat)
        result: sf.Time = to_sf_time(dt)
        assert result == time

    def test_list(self, three_sf_times):
        times: list[str] = ["2000-01-01T00:00:00Z", "2000-01-02T00:00:00Z", "2000-01-03T00:00:00Z"]
        result: sf.Time = to_sf_time(times)
        assert all(result == three_sf_times)

    def test_DateTimeIndex(self, time_isoformat, three_sf_times):
        times: pd.DatetimeIndex = pd.date_range(time_isoformat, periods=3, freq="D")
        result: sf.Time = to_sf_time(times)
        assert all(result == three_sf_times)

    # Add test for a different timescale
    # def test_timescale(self, time):
    #     ts = sf.Timescale()
    #     result: sf.Time = to_sf_time(time, ts)
    #     assert result is time


class TestIsUnitsDtypeOfPhysicalType():
    def test_non_UnitsDtype(self):
        dtype = pd.Int64Dtype()
        assert not _is_UnitsDtype_of_physical_type(dtype, u.physical.length)

    def test_matching_physical_type(self):
        dtype = pue.UnitsDtype(u.m)
        assert _is_UnitsDtype_of_physical_type(dtype, u.physical.length)

    def test_non_matching_physical_type(self):
        dtype = pue.UnitsDtype(u.s)
        assert not _is_UnitsDtype_of_physical_type(dtype, u.physical.length)


class TestToSfAngle:
    def test_angle_input(self):
        angle = sf.Angle(degrees=90)
        result: sf.Angle = to_sf_angle(angle)
        assert result is angle
    
    def test_angle_1D_array(self, test_array_1d):
        result: sf.Angle = to_sf_angle(test_array_1d)
        assert np.allclose(result.radians, test_array_1d)

    def test_angle_2D_array(self, test_array_2d):
        result: sf.Angle = to_sf_angle(test_array_2d)
        assert np.allclose(result.radians, test_array_2d)

    def test_series_scalar(self, test_array_1d):
        s: pd.Series = pd.Series(test_array_1d)
        result: sf.Angle = to_sf_angle(s)
        assert np.allclose(result.radians, test_array_1d)

    def test_series_UnitsDtype(self, test_array_1d):
        s: pd.Series = pd.Series(test_array_1d, dtype=f"unit[deg]")
        result: sf.Angle = to_sf_angle(s)
        assert np.allclose(result.degrees, test_array_1d)

    def test_dataframe_scalar(self, test_array_2d):
        df: pd.DataFrame = pd.DataFrame(test_array_2d.T)
        result: sf.Angle = to_sf_angle(df)
        assert np.allclose(result.radians, test_array_2d)

    @pytest.mark.parametrize("unit", [u.deg, u.rad], ids=["deg", "rad"])
    def test_dataframe_UnitsDtype(self, test_array_2d, unit):
        df: pd.DataFrame = pd.DataFrame(
            data=test_array_2d.T * unit,
            dtype="unit"
        )
        result: sf.Angle = to_sf_angle(df)
        assert np.allclose(result.radians, test_array_2d * unit/u.rad)


class TestToSfDistance:
    def test_distance_input(self):
        dist = sf.Distance(1)
        result: sf.Distance = to_sf_distance(dist)
        assert result is dist

    def test_distance_1D_array(self, test_array_1d):
        result: sf.Distance = to_sf_distance(test_array_1d)
        assert np.allclose(result.au, test_array_1d)

    def test_distance_2D_array(self, test_array_2d):
        result: sf.Distance = to_sf_distance(test_array_2d)
        assert np.allclose(result.au, test_array_2d)

    def test_series_scalar(self, test_array_1d):
        s: pd.Series = pd.Series(test_array_1d)
        result: sf.Distance = to_sf_distance(s)
        assert np.allclose(result.au, test_array_1d)

    def test_series_UnitsDtype(self, test_array_1d, length_unit):
        s: pd.Series = pd.Series(test_array_1d, dtype=f"unit[{length_unit}]")
        result: sf.Distance = to_sf_distance(s)
        assert np.allclose(result.au, test_array_1d * length_unit/u.au)

    def test_dataframe_scalar(self, test_array_2d):
        df: pd.DataFrame = pd.DataFrame(test_array_2d.T)
        result: sf.Distance = to_sf_distance(df)
        assert np.allclose(result.au, test_array_2d)

    def test_dataframe_UnitsDtype(self, test_array_2d, length_unit):
        df: pd.DataFrame = pd.DataFrame(
            data=test_array_2d.T,
            dtype=f"unit[{length_unit}]"
        )
        result: sf.Distance = to_sf_distance(df)
        excepted = test_array_2d * length_unit/u.au
        assert np.allclose(result.au, excepted)

    def test_dataframe_mixed(self, test_array_2d):
        x, y, z = test_array_2d
        df: pd.DataFrame = pd.DataFrame({
            "x": pd.Series(x, dtype=f"unit[m]"),
            "y": pd.Series(y, dtype=f"unit[km]"),
            "z": pd.Series(z),                      # No unit, should be treated as au
        })
        result: sf.Distance = to_sf_distance(df)
        excepted = u.Quantity([x*u.m, y*u.km, z*u.au])/u.au
        assert np.allclose(result.au, excepted)


class TestToSfVelocity:
    def test_velocity_input(self):
        vel = sf.Velocity(1)
        result: sf.Velocity = to_sf_velocity(vel)
        assert result is vel

    def test_velocity_1D_array(self, test_array_1d):
        result: sf.Velocity = to_sf_velocity(test_array_1d)
        assert np.allclose(result.au_per_d, test_array_1d)

    def test_velocity_2D_array(self, test_array_2d):
        result: sf.Velocity = to_sf_velocity(test_array_2d)
        assert np.allclose(result.au_per_d, test_array_2d)

    def test_series_scalar(self, test_array_1d):
        s: pd.Series = pd.Series(test_array_1d)
        result: sf.Velocity = to_sf_velocity(s)
        assert np.allclose(result.au_per_d, test_array_1d)

    def test_series_UnitsDtype(self, test_array_1d, velocity_unit):
        s: pd.Series = pd.Series(test_array_1d, dtype=f"unit[{velocity_unit}]")
        result: sf.Velocity = to_sf_velocity(s)
        assert np.allclose(result.au_per_d, test_array_1d * velocity_unit/(u.au/u.day))

    def test_dataframe_scalar(self, test_array_2d):
        df: pd.DataFrame = pd.DataFrame(test_array_2d.T)
        result: sf.Velocity = to_sf_velocity(df)
        assert np.allclose(result.au_per_d, test_array_2d)
    
    def test_dataframe_UnitsDtype(self, test_array_2d, velocity_unit):
        df: pd.DataFrame = pd.DataFrame(
            data=test_array_2d.T,
            dtype=f"unit[{velocity_unit}]"
        )
        result: sf.Velocity = to_sf_velocity(df)
        excepted = test_array_2d * velocity_unit/(u.au/u.day)
        assert np.allclose(result.au_per_d, excepted)

    def test_dataframe_mixed(self, test_array_2d):
        vx, vy, vz = test_array_2d
        df: pd.DataFrame = pd.DataFrame({
            "vx": pd.Series(vx, dtype=f"unit[m/s]"),
            "vy": pd.Series(vy, dtype=f"unit[km/s]"),
            "vz": pd.Series(vz),                      # No unit, should be treated as au/day
        })
        result: sf.Velocity = to_sf_velocity(df)
        excepted = u.Quantity([vx*u.m/u.s, vy*u.km/u.s, vz*u.au/u.day])/(u.au/u.day)
        assert np.allclose(result.au_per_d, excepted)
