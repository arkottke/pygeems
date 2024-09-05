#!/usr/bin/python
# -*- coding: utf-8 -*-

from pkg_resources import get_distribution

import scipy.constants as SC

# Unit weight of water in kN/m³
UNIT_WT_WATER = SC.g
# Atmospheric pressure in kPa
PRESS_ATM = SC.atm / 1000

from . import dyn_props
from . import ground_motion
from . import site_invest
from . import slope_disp

__author__ = 'Albert Kottke'
__copyright__ = 'Copyright 2018 Albert Kottke'
__license__ = 'MIT'
__title__ = 'pygeems'
__version__ = get_distribution('pygeems').version
del get_distribution
