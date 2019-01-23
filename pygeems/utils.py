
import functools

import numpy as np


def check_bounds(value, lower, upper, name, format='f'):
    if lower and np.any(value < lower):
        raise ValueError(
            f'"{name}" value of {value:{format}} below lower limit of {lower:{format}}')

    if upper and np.any(value > upper):
        raise ValueError(
            f'"{name}" value of {value:{format}} above upper limit of {upper:{format}}')


def check_options(value, options, name):
    if value not in options:
        raise ValueError(
            f'"{name}" not in options: {options}'
        )


def dist_lognorm(f):
    @functools.wraps(f)
    def wrapper(*args, **kwds):
        ln_mean, ln_std = f(*args, **kwds)

        if kwds.get('stats', False):
            return ln_mean, ln_std
        else:
            return np.exp(ln_mean)

    return wrapper
