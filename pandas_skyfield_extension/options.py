from __future__ import annotations

from typing import Any
import astropy.units as u


def _verify_unit(obj: Any, expected_physical_type: u.PhysicalType | str) -> None:
    """Verify that an object is a astropy Unit of an expected physical type.

    Parameters
    ----------
    obj : Any
        Object to verify.
    expected_physical_type : u.PhysicalType | str
        Expected physical type of the unit.

    Raises
    ------
    ValueError
        If obj is either not a unit or not the expected physical type.
    """
    if not isinstance(obj, (u.UnitBase)):
        raise ValueError(f"Got instance of type {type(obj)} instead of type 'Unit'")
    if obj.physical_type != expected_physical_type:
        raise ValueError(
            f"Got Unit with physical type {obj.physical_type} instead of {expected_physical_type}"
        )


class Config:
    """Configuration class for the pandas_skyfield_extension package."""

    def __init__(self, length_unit, velocity_unit, angle_unit) -> None:
        self._length_unit: u.Unit[u.physical.length] = length_unit
        self._velocity_unit: u.Unit[u.physical.velocity] = velocity_unit
        self._angle_unit: u.Unit[u.physical.angle] = angle_unit

    @property
    def length_unit(self) -> u.Unit[u.physical.length]:
        """Get standard length unit."""
        return self._length_unit

    @length_unit.setter
    def length_unit(self, new_length_unit: u.Unit[u.physical.length]) -> None:
        """Set standard length unit.

        Parameters
        ----------
        new_length_unit : u.Unit[u.physical.length]
            New standard length unit, must be of physical type length.
        """
        _verify_unit(new_length_unit, u.physical.length)
        self._length_unit = new_length_unit

    @property
    def velocity_unit(self) -> u.Unit[u.physical.velocity]:
        """Get standard velocity unit."""
        return self._velocity_unit

    @velocity_unit.setter
    def velocity_unit(self, new_velocity_unit: u.Unit[u.physical.velocity]) -> None:
        """Set standard velocity unit.

        Parameters
        ----------
        new_velocity_unit : u.Unit[u.physical.velocity]
            New standard velocity unit, must be of physical type velocity.
        """
        _verify_unit(new_velocity_unit, u.physical.velocity)
        self._length_unit = new_velocity_unit

    @property
    def angle_unit(self) -> u.Unit[u.physical.angle]:
        """Get standard angle unit."""
        return self._angle_unit

    @angle_unit.setter
    def angle_unit(self, new_angle_unit: u.Unit[u.physical.angle]) -> None:
        """Set standard angle unit.

        Parameters
        ----------
        new_angle_unit : u.Unit[u.physical.angle]
            New standard angle unit, must be of physical type angle.
        """
        _verify_unit(new_angle_unit, u.physical.angle)
        self._length_unit = new_angle_unit


config: Config = Config(u.km, u.km / u.s, u.deg)
"""Config object for the pandas_skyfield_extension package.

Use this object to set the standard astropy units used to convert the Skyfield
Distance, Velocity and Angle objects to astropy.
"""


_LENGTH_UNIT: u.Unit = u.km
_VELOCITY_UNIT: u.Unit = u.km / u.s


def set_standard_length_unit(unit: u.Unit) -> None:
    global _LENGTH_UNIT
    _LENGTH_UNIT = unit


def get_standard_length_unit() -> u.Unit:
    return _LENGTH_UNIT


def set_standard_velocity_unit(unit: u.Unit) -> None:
    global _VELOCITY_UNIT
    _VELOCITY_UNIT = unit


def get_standard_velocity_unit() -> u.Unit:
    return _VELOCITY_UNIT
