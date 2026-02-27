# Pandas Skyfield Extension

This python package extends the popular data science library [pandas](https://github.com/pandas-dev/pandas/) with the astronomy and satellite library [Skyfield](https://rhodesmill.org/skyfield/). Warning this package is still in a very early development phase and behavior might change without further warning or deprecation.


## Installation

This library is currently not on PyPi or conda-forge available, installation therefore has to be done from the GitHub repository.

### To use the library

With pip:
```bash
pip install git+https://github.com/Julian-Harbeck/pandas-units-extension.git@dev
pip install git+https://github.com/Julian-Harbeck/pandas-skyfield-extension.git
```

With conda, will create the `pandas_skyfield_extension` conda environment:
```bash
conda env create -f environment.yml 
```

### For development

From a clone of this repository to install it in editable mode.

With pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install git+https://github.com/Julian-Harbeck/pandas-units-extension.git@dev
pip install -e .
```

Or to create the `pandas_skyfield_extension_dev` conda environment:
```bash
conda env create -f environment_dev.yml 
```

### Optional Dependencies

Optionally the library [GeoPandas](https://github.com/geopandas/geopandas) may be installed. This will allow to convert Skyfield positions to a `GeoDataFrame`. Install with:

```bash
pip install geopandas[all]
```


## Examples

```python
>>> import pandas as pd
>>> import skyfield.api as sf

>>> import pandas_skyfield_extension as psfe

>>> # Define TLE
>>> NanoFF_A_TLE_1: str = "1 58810U 23185T   26033.42930604  .00003749  00000-0  22676-3 0  9992"
>>> NanoFF_A_TLE_2: str = "2 58810  97.5437 102.7707 0011522  57.3014 302.9326 15.11204711112680"

>>> # Create EarthSatellites for NanoFF-A
>>> nanoff_a: sf.EarthSatellite = sf.EarthSatellite(NanoFF_A_TLE_1, NanoFF_A_TLE_2, name="NanoFF-A")

>>> # Create times of interest
>>> times_idx: pd.DatetimeIndex = pd.date_range(start="2026-02-02T00:00:00Z", end="2026-02-03T00:00:00Z", freq="1 min")

>>> # Create pandas Series with skyfield positions of NanoFF A
>>> nanoff_a_sr: pd.Series = psfe.at(nanoff_a, times_idx)
>>> nanoff_a_sr
2026-02-02 00:00:00+00:00    Geocentric position [ 1330.651 -6747.060  -764...
2026-02-02 00:01:00+00:00    Geocentric position [ 1258.258 -6696.250 -1210...
2026-02-02 00:02:00+00:00    Geocentric position [ 1180.413 -6616.421 -1652...
2026-02-02 00:03:00+00:00    Geocentric position [ 1097.455 -6507.936 -2086...
2026-02-02 00:04:00+00:00    Geocentric position [ 1009.747 -6371.279 -2511...
                                                   ...                        
2026-02-02 23:56:00+00:00    Geocentric position [  947.411 -6046.651 -3236...
2026-02-02 23:57:00+00:00    Geocentric position [  844.991 -5837.133 -3627...
2026-02-02 23:58:00+00:00    Geocentric position [  738.922 -5602.405 -4001...
2026-02-02 23:59:00+00:00    Geocentric position [  629.663 -5343.495 -4359...
2026-02-03 00:00:00+00:00    Geocentric position [  517.688 -5061.533 -4697...
Freq: min, Name: NanoFF-A, Length: 1441, dtype: skyfield_position[<class 'skyfield.positionlib.Geocentric'>]

>>> # Accessing commonly derived data of Skyfield positions through the skyfield_position accessor
>>> nanoff_a_sr.skyfield_position.xyz
                                     x            y            z
2026-02-02 00:00:00+00:00  1330.651175 -6747.060153  -764.348188
2026-02-02 00:01:00+00:00  1258.258441 -6696.249763 -1210.867118
2026-02-02 00:02:00+00:00  1180.413133 -6616.421443 -1652.124475
2026-02-02 00:03:00+00:00  1097.455248 -6507.935975 -2086.206542
2026-02-02 00:04:00+00:00  1009.746692 -6371.278596  -2511.23298
...                                ...          ...          ...
2026-02-02 23:56:00+00:00   947.410784 -6046.651003 -3236.760747
2026-02-02 23:57:00+00:00   844.990917  -5837.13326 -3627.198743
2026-02-02 23:58:00+00:00   738.921649 -5602.405196 -4001.928009
2026-02-02 23:59:00+00:00   629.662968 -5343.494802 -4359.335815
2026-02-03 00:00:00+00:00   517.688063 -5061.533022 -4697.886121

[1441 rows x 3 columns]
```

See [doc/skyfield_position.ipynb](doc/skyfield_position.ipynb) for a detailed introduction into the package.


## Configuration of the module

The Pandas Skyfield Extension has a configuration object `config` that can be used to set preferences for the conversion between `skyfield.units` objects like `Distance`, `Velocity` and `Angle` to `astropy` `Quantity` objects. To do so use the following code snipped:

```python
>>> import astropy.units as u
>>> import pandas_skyfield_extension as psfe

>>> # Show standard units
>>> print(f"Standard length unit:", repr(psfe.config.length_unit))
Standard length unit: Unit("km")
>>> print(f"Standard velocity unit:", repr(psfe.config.velocity_unit))
Standard velocity unit: Unit("km / s")
>>> print(f"Standard angle unit:", repr(psfe.config.angle_unit))
Standard angle unit: Unit("deg")

>>> # To change the unit just assign a new value:
>>> psfe.config.length_unit = u.m

>>> # Show new unit:
>>> print(f"Changed length unit:", repr(psfe.config.length_unit))
Changed length unit: Unit("m")
```


## History

Based on the Astropy extension array this package was developed by Julian Harbeck from Technische Universität Berlin as part of the RACCOON project and the student initiative StudOps.


## Links

- <https://rhodesmill.org/skyfield/api.html>
- <https://github.com/Julian-Harbeck/pandas-units-extension/tree/dev> - Pandas extension for Astropy Quantity objects
