from __future__ import annotations

import datetime

import astropy.units as u
import numpy as np
import pandas as pd
import pandas_units_extension as pue
import pytest
from skyfield import (
    api as sf,
    positionlib,
)

from pandas_skyfield_extension import SkyfieldPositionExtensionArray
from pandas_skyfield_extension.sf_converters import (
    _as_3xN_array,
    _is_UnitsDtype_of_physical_type,
    concat_sf_positions,
    concat_sf_times,
    to_sf_angle,
    to_sf_distance,
    to_sf_position,
    to_sf_time,
    to_sf_velocity,
)


@pytest.fixture(params=[u.m, u.km, u.au], ids=["m", "km", "au"])
def length_unit(request) -> u.Unit:
    return request.param


@pytest.fixture(
    params=[u.m / u.s, u.km / u.s, u.au / u.day], ids=["m/s", "km/s", "au/day"]
)
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


@pytest.fixture
def scalar_position(ts) -> positionlib.ICRF:
    return positionlib.build_position(
        position_au=[1.0, 2.0, 3.0],
        velocity_au_per_d=[0.1, 0.2, 0.3],
        t=ts.utc(2000, 1, 1, 0, 0, 0),
        center=399,
        target=None,
    )


@pytest.fixture
def array_position(ts) -> positionlib.ICRF:
    return positionlib.build_position(
        position_au=[[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]],
        velocity_au_per_d=[[0.4, 0.5], [0.6, 0.7], [0.8, 0.9]],
        t=ts.utc(2000, 1, [2, 3], 0, 0, 0),
        center=399,
        target=None,
    )


@pytest.fixture
def pair_of_positions(ts, scalar_position) -> list[positionlib.ICRF]:
    pos2 = positionlib.build_position(
        position_au=[4.0, 5.0, 6.0],
        velocity_au_per_d=[0.4, 0.5, 0.6],
        t=ts.utc(2000, 1, 2, 0, 0, 0),
        center=399,
        target=None,
    )
    return [scalar_position, pos2]


####################
# Helper functions #
####################


class TestAs3xNArray:
    def test_scalar_vector_becomes_column_vector(self):
        value = np.array([1.0, 2.0, 3.0])

        result = _as_3xN_array(value)

        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result, value[:, np.newaxis])

    def test_matrix_is_unchanged(self):
        value = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = _as_3xN_array(value)

        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result, value)


class TestIsUnitsDtypeOfPhysicalType:
    def test_non_UnitsDtype(self):
        dtype = pd.Int64Dtype()
        assert not _is_UnitsDtype_of_physical_type(dtype, u.physical.length)

    def test_matching_physical_type(self):
        dtype = pue.UnitsDtype(u.m)
        assert _is_UnitsDtype_of_physical_type(dtype, u.physical.length)

    def test_non_matching_physical_type(self):
        dtype = pue.UnitsDtype(u.s)
        assert not _is_UnitsDtype_of_physical_type(dtype, u.physical.length)


##################
# Concatenations #
##################


class TestConcatSfPositions:
    def test_position_input(self, scalar_position):
        with pytest.raises(IndexError, match="list index out of range"):
            concat_sf_positions(scalar_position)

    def test_mixed_scalar_and_array(self, ts, scalar_position, array_position):
        result = concat_sf_positions([scalar_position, array_position])

        assert result.position.au.shape == (3, 3)
        assert result.velocity.au_per_d.shape == (3, 3)
        assert result.t.shape == (3,)
        assert np.all(result.t == ts.utc(2000, 1, [1, 2, 3], 0, 0, 0))

    def test_list_of_positions_same_center_target(self, ts, pair_of_positions):
        result = concat_sf_positions(pair_of_positions)

        assert result.position.au.shape == (3, 2)
        assert result.velocity.au_per_d.shape == (3, 2)
        assert result.t.shape == (2,)
        assert np.all(result.t == ts.utc(2000, 1, [1, 2], 0, 0, 0))

    def test_list_of_positions_mismatched_targets_raises(self, ts, scalar_position):
        mismatched = positionlib.build_position(
            position_au=[4.0, 5.0, 6.0],
            velocity_au_per_d=[0.4, 0.5, 0.6],
            t=ts.utc(2000, 1, 2, 0, 0, 0),
            center=0,
            target=10,
        )

        with pytest.raises(ValueError, match="same center and target"):
            concat_sf_positions([scalar_position, mismatched])


class TestConcatSfTimes:
    def test_concat_mixed_scalar_and_array(self, ts):
        scalar = ts.utc(2000, 1, 1, 0, 0, 0)
        array = ts.utc(2000, 1, [2, 3], 0, 0, 0)

        result = concat_sf_times([scalar, array])
        expected = ts.utc(2000, 1, [1, 2, 3], 0, 0, 0)

        assert result.shape == (3,)
        assert np.all(result == expected)

    def test_concat_multiple_scalars(self, ts: sf.Timescale):
        times = [ts.utc(2000, 1, 1, 0, 0, 0), ts.utc(2000, 1, 2, 0, 0, 0)]

        result = concat_sf_times(times)
        expected = ts.utc(2000, 1, [1, 2], 0, 0, 0)

        assert result.shape == (2,)
        assert np.all(result == expected)

    def test_concat_empty_raises(self):
        with pytest.raises(ValueError, match="times must not be empty"):
            concat_sf_times([])

    def test_concat_non_time_raises(self, ts: sf.Timescale):
        non_time = "2000-01-01T00:00:00Z"
        with pytest.raises(TypeError, match="Object .* is not a skyfield Time"):
            concat_sf_times([ts.utc(2000, 1, 1, 0, 0, 0), non_time])

    def test_concat_mismatched_timescales_raises(self, ts: sf.Timescale):
        other_ts = sf.load.timescale()
        time_a = ts.utc(2000, 1, 1, 0, 0, 0)
        time_b = other_ts.utc(2000, 1, 2, 0, 0, 0)

        with pytest.raises(ValueError, match="same Timescale"):
            concat_sf_times([time_a, time_b])


###############################
# High level Skyfield objects #
###############################


class TestToSfPosition:
    def test_position_input(self, scalar_position):
        result = to_sf_position(scalar_position)
        assert result is scalar_position

    def test_skyfield_position_array(self, array_position):
        result = to_sf_position(SkyfieldPositionExtensionArray(array_position))
        assert np.all(result.position.au == array_position.position.au)
        assert np.all(result.velocity.au_per_d == array_position.velocity.au_per_d)
        assert np.all(result.t == array_position.t)

    def test_list_single_item(self, scalar_position):
        result = to_sf_position([scalar_position])
        assert result is scalar_position

    def test_list_dispatches_to_concat(self, scalar_position):
        result = to_sf_position([scalar_position, scalar_position])

        assert result.position.au.shape == (3, 2)
        assert result.velocity.au_per_d.shape == (3, 2)
        assert result.t.shape == (2,)

    def test_unsupported_object_raises(self):
        with pytest.raises(TypeError, match="Cannot convert"):
            to_sf_position(object())


class TestToSfTime:
    def test_sf_time(self, time):
        result: sf.Time = to_sf_time(time)
        assert result is time

    def test_list_sf_time(self, three_sf_times):
        times = [three_sf_times[0], three_sf_times[1], three_sf_times[2]]
        result: sf.Time = to_sf_time(times)
        assert all(result == three_sf_times)

    def test_datetime(self, time, time_isoformat):
        dt = datetime.datetime.fromisoformat(time_isoformat)
        result: sf.Time = to_sf_time(dt)
        assert result == time

    def test_timestamp(self, time, time_isoformat):
        dt = pd.Timestamp(time_isoformat)
        result: sf.Time = to_sf_time(dt)
        assert result == time

    def test_list(self, three_sf_times):
        times: list[str] = [
            "2000-01-01T00:00:00Z",
            "2000-01-02T00:00:00Z",
            "2000-01-03T00:00:00Z",
        ]
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


##########################
# Skyfield Unit objects #
##########################


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
        s: pd.Series = pd.Series(test_array_1d, dtype="unit[deg]")
        result: sf.Angle = to_sf_angle(s)
        assert np.allclose(result.degrees, test_array_1d)

    def test_dataframe_scalar(self, test_array_2d):
        df: pd.DataFrame = pd.DataFrame(test_array_2d.T)
        result: sf.Angle = to_sf_angle(df)
        assert np.allclose(result.radians, test_array_2d)

    @pytest.mark.parametrize("unit", [u.deg, u.rad], ids=["deg", "rad"])
    def test_dataframe_UnitsDtype(self, test_array_2d, unit):
        df: pd.DataFrame = pd.DataFrame(data=test_array_2d.T * unit, dtype="unit")
        result: sf.Angle = to_sf_angle(df)
        assert np.allclose(result.radians, test_array_2d * unit / u.rad)


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
        assert np.allclose(result.au, test_array_1d * length_unit / u.au)

    def test_dataframe_scalar(self, test_array_2d):
        df: pd.DataFrame = pd.DataFrame(test_array_2d.T)
        result: sf.Distance = to_sf_distance(df)
        assert np.allclose(result.au, test_array_2d)

    def test_dataframe_UnitsDtype(self, test_array_2d, length_unit):
        df: pd.DataFrame = pd.DataFrame(
            data=test_array_2d.T, dtype=f"unit[{length_unit}]"
        )
        result: sf.Distance = to_sf_distance(df)
        excepted = test_array_2d * length_unit / u.au
        assert np.allclose(result.au, excepted)

    def test_dataframe_mixed(self, test_array_2d):
        x, y, z = test_array_2d
        df: pd.DataFrame = pd.DataFrame(
            {
                "x": pd.Series(x, dtype="unit[m]"),
                "y": pd.Series(y, dtype="unit[km]"),
                "z": pd.Series(z),  # No unit, should be treated as au
            }
        )
        result: sf.Distance = to_sf_distance(df)
        excepted = u.Quantity([x * u.m, y * u.km, z * u.au]) / u.au
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
        assert np.allclose(
            result.au_per_d, test_array_1d * velocity_unit / (u.au / u.day)
        )

    def test_dataframe_scalar(self, test_array_2d):
        df: pd.DataFrame = pd.DataFrame(test_array_2d.T)
        result: sf.Velocity = to_sf_velocity(df)
        assert np.allclose(result.au_per_d, test_array_2d)

    def test_dataframe_UnitsDtype(self, test_array_2d, velocity_unit):
        df: pd.DataFrame = pd.DataFrame(
            data=test_array_2d.T, dtype=f"unit[{velocity_unit}]"
        )
        result: sf.Velocity = to_sf_velocity(df)
        excepted = test_array_2d * velocity_unit / (u.au / u.day)
        assert np.allclose(result.au_per_d, excepted)

    def test_dataframe_mixed(self, test_array_2d):
        vx, vy, vz = test_array_2d
        df: pd.DataFrame = pd.DataFrame(
            {
                "vx": pd.Series(vx, dtype="unit[m/s]"),
                "vy": pd.Series(vy, dtype="unit[km/s]"),
                "vz": pd.Series(vz),  # No unit, should be treated as au/day
            }
        )
        result: sf.Velocity = to_sf_velocity(df)
        excepted = u.Quantity([vx * u.m / u.s, vy * u.km / u.s, vz * u.au / u.day]) / (
            u.au / u.day
        )
        assert np.allclose(result.au_per_d, excepted)
