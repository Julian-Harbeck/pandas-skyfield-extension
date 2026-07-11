import astropy.units as u
import pytest

import pandas_skyfield_extension as psfe
from pandas_skyfield_extension.options import _verify_unit, Config


@pytest.fixture
def config() -> Config:
    return psfe.config


@pytest.fixture(params=["length_unit", "velocity_unit", "angle_unit"])
def config_attr(request) -> str:
    return request.param


@pytest.fixture
def config_physical_type(config_attr) -> u.physical.PhysicalType:
    return {
        "length_unit": u.physical.length,
        "velocity_unit": u.physical.velocity,
        "angle_unit": u.physical.angle,
    }.get(config_attr)


@pytest.fixture
def config_default_unit(config_attr) -> u.UnitBase:
    return {
        "length_unit": u.km,
        "velocity_unit": u.km / u.s,
        "angle_unit": u.deg,
    }.get(config_attr)


class TestVerifyUnit:
    def test_matching(self):
        _verify_unit(u.m, u.physical.length)

    def test_invalid_object(self):
        msg: str = "Got instance of type <class 'str'> instead of type 'Unit'"
        with pytest.raises(ValueError, match=msg):
            _verify_unit("not a unit", u.physical.length)

    def test_invalid_physical_type(self):
        msg: str = "Got Unit with physical type time instead of length"
        with pytest.raises(ValueError, match=msg):
            _verify_unit(u.s, u.physical.length)

    def test_function_unit_unsupported(self):
        # This test is expected to fail until function units are supported
        msg: str = "Got instance of type <class 'astropy.units.function.logarithmic.DecibelUnit'> instead of type 'Unit'"
        with pytest.raises(ValueError, match=msg):
            _verify_unit(u.Unit("dB(mW)"), u.physical.power)


class TestConfig:
    def test_default_units(self, config, config_attr, config_default_unit):
        """Deprecation test warning of changes to default units."""
        assert getattr(config, config_attr) == config_default_unit

    @pytest.mark.parametrize(
        "new_unit",
        [
            pytest.param(u.m, id="length_m"),
            pytest.param(u.km, id="length_km"),
            pytest.param(u.au, id="length_au"),
            pytest.param(u.m / u.s, id="velocity_m_per_s"),
            pytest.param(u.km / u.s, id="velocity_km_per_s"),
            pytest.param(u.au / u.day, id="velocity_au_per_day"),
            pytest.param(u.deg, id="angle_deg"),
            pytest.param(u.rad, id="angle_rad"),
        ],
    )
    def test_set_new_units(self, config, config_attr, config_physical_type, new_unit):
        """Test setting a new unit for each config attribute.

        This test checks that setting a new unit of the correct physical type works,
        but setting a unit of the wrong physical type raises the expected ValueError.
        """
        if new_unit.physical_type == config_physical_type:
            setattr(config, config_attr, new_unit)
            assert getattr(config, config_attr) == new_unit
        else:
            msg: str = f"Got Unit with physical type {new_unit.physical_type} instead of {config_physical_type}"
            with pytest.raises(ValueError, match=msg):
                setattr(config, config_attr, new_unit)
