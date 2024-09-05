"""pyGEEMs: Geotechnical earthquake engineering models implemented in Python."""

from importlib import version
import pathlib

import scipy.constants as SC

# Unit weight of water in kN/m³
UNIT_WT_WATER = SC.g
# Atmospheric pressure in kPa
PRESS_ATM = SC.atm / 1000
# KPA_TO_ATM = scipy.constants.kilo / scipy.constants.atm

from . import dyn_props
from . import ground_motion
from . import site_invest
from . import slope_disp

__all__ = ["dyn_props", "ground_motion", "site_invest", "slope_disp"]

__author__ = "Albert Kottke"
__copyright__ = "Copyright 2018-24 Albert Kottke"
__license__ = "MIT"
__title__ = "pygeems"
__version__ = version("pygeems")

FPATH_DATA = pathlib.Path(__file__).parent / "data"
