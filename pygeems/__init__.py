"""pyGEEMs: Geotechnical earthquake engineering models implemented in Python."""
import pathlib

import scipy.constants

FPATH_DATA = pathlib.Path(__file__).parent / "data"
KPA_TO_ATM = scipy.constants.kilo / scipy.constants.atm

__author__ = "Albert Kottke"
__copyright__ = "Copyright 2018 Albert Kottke"
__license__ = "MIT"
__title__ = "pygeems"
__version__ = "0.2.1"

from . import dyn_props
from . import ground_motion
from . import slope_disp
