"""pyGEEMs: Geotechnical earthquake engineering models implemented in Python."""
import pathlib

from pkg_resources import get_distribution

FPATH_DATA = pathlib.Path(__file__).parent / "data"

import scipy.constants

from . import ground_motion
from . import slope_disp
from . import dyn_props

__author__ = "Albert Kottke"
__copyright__ = "Copyright 2018 Albert Kottke"
__license__ = "MIT"
__title__ = "pygeems"
__version__ = get_distribution("pygeems").version

KPA_TO_ATM = scipy.constants.kilo / scipy.constants.atm
