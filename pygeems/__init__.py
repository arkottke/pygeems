#!/usr/bin/python
# -*- coding: utf-8 -*-

from pkg_resources import get_distribution

from . import ground_motion
from . import slope_disp

__author__ = 'Albert Kottke'
__copyright__ = 'Copyright 2018 Albert Kottke'
__license__ = 'MIT'
__title__ = 'pygeems'
__version__ = get_distribution('pygeems').version
