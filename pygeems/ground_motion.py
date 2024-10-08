"""Functions and classes for calculation of ground motion parameters."""

import re
from typing import Optional

import numpy as np
import numpy.typing as npt

from . import FPATH_DATA
from .utils import check_bounds
from .utils import check_options
from .utils import dist_lognorm

_CACHE_REA15 = {}


def calc_damping_scaling_rea15(
    damping: float,
    mag: float,
    dist_rup: float,
    comp: str = "rotd50",
    periods: Optional[npt.ArrayLike] = None,
) -> npt.ArrayLike:
    """Compute the damping scaling proposed by Rezaeian et al. (2014).

    Parameters
    ----------
    damping: damping ratio of the oscillator in percent (2 for 2%)
    mag: earthquake magnitude
    dist_rup: closest distance to the rupture in [km]
    comp: component, can be either: 'rotd50', 'roti50', 'vertical'
    periods: *optional* periods to provide values.

    Returns
    -------
    ret : np.rec.array
        Array with columns period, damping scaling factor, and natural logarithmic standard deviation.
    """
    check_bounds(damping, 0.5, 30, "damping")
    check_options(comp, ["rotd50", "roti50", "vertical"], "comp")

    # Only load the coefficients once
    if comp not in _CACHE_REA15:
        _CACHE_REA15[comp] = np.genfromtxt(
            FPATH_DATA / f"rezaeian_et_al_2014-{comp}.csv",
            delimiter=",",
            skip_header=1,
            names=True,
        ).view(np.recarray)
    C = _CACHE_REA15[comp]

    ln_damp = np.log(damping)
    ln_dsf = (
        C.b0
        + C.b1 * ln_damp
        + C.b2 * ln_damp**2
        + (C.b3 + C.b4 * ln_damp + C.b5 * ln_damp**2) * mag
        + (C.b6 + C.b7 * ln_damp + C.b8 * ln_damp**2) * np.log(dist_rup + 1)
    )
    ln_damp_5 = np.log(damping / 5)
    ln_std = np.abs(C.a0 * ln_damp_5 + C.a1 * ln_damp_5**2)

    if periods:
        # Interpolate over the provided periods
        ln_dsf = np.interp(np.log(periods), np.log(C.period), ln_dsf)
        ln_std = np.interp(np.log(periods), np.log(C.period), ln_std)
    else:
        periods = C.period

    ret = np.rec.fromarrays(
        [periods, np.exp(ln_dsf), ln_std], names="period,dsf,ln_std"
    )
    return ret


@dist_lognorm
def calc_period_rea05(
    kind: str,
    mag: float,
    dist_rup: float,
    site_class: str = "c",
    directivity: bool = False,
    **kwargs,
):
    """Rathje et al. (2005) period metrics."""
    # Model coefficients from Table 2
    C = {
        model: np.rec.fromrecords(values, names="c1,c2,c3,c4,c5,c6")
        for model, values in zip(
            ["period_mean", "period_avg", "period_pred"],
            [
                (-1.00, 0.18, 0.0038, 0.078, 0.27, 0.40),
                (-0.89, 0.29, 0.0030, 0.070, 0.25, 0.37),
                (-1.78, 0.30, 0.0045, 0.150, 0.33, 0.24),
            ],
        )
    }[kind]

    check_options(site_class, "bcd", "site_class")
    if kind in ["period_avg", "period_pred"]:
        check_bounds(mag, 4.7, 7.6, "mag")
    else:
        mag = np.minimum(mag, 7.25)
        check_bounds(mag, 5.0, 7.25, "mag")

    if site_class == "c":
        s_c = 1
        s_d = 0
    elif site_class == "d":
        s_c = 0
        s_d = 1
    else:
        s_c = 0
        s_d = 0

    f_d = int(directivity)

    ln_period = (
        C.c1
        + C.c2 * (mag - 6)
        + C.c3 * dist_rup
        + C.c4 * s_c
        + C.c5 * s_d
        + C.c6 * (1 - dist_rup / 20) * f_d
    )

    intra = {
        "period_mean": {"b": 0.42, "c": 0.38, "d": 0.31},
        "period_avg": {"b": 0.42, "c": 0.38, "d": 0.31},
        "period_pred": {"b": 0.42, "c": 0.38, "d": 0.31},
    }[kind][site_class]
    inter = {"period_mean": 0.17, "period_avg": 0.13, "period_pred": 0.22}[kind]
    ln_std = np.sqrt(intra**2 + inter**2)

    return ln_period, ln_std


@dist_lognorm
def calc_aris_intensity_aea16(
    mag: float,
    v_s30: float,
    pga: float,
    psa_1s: float,
    hanging_wall: bool = False,
    dist_jb: Optional[float] = None,
    ln_std_pga: Optional[float] = None,
    ln_std_psa_1s: Optional[float] = None,
    **kwargs,
):
    """Arias intensity estimate from Abrahamson, Shi, and Yang (2016)."""
    # From Table 3.1
    # Value of c8 is provided after equation 3.7
    C = np.rec.fromrecords(
        (0.47, -0.28, 0.50, 1.52, 0.21, 0.09), names="c1,c2,c3,c4,c5,c8"
    )

    ln_arias_int = (
        C.c1
        + C.c2 * np.log(v_s30)
        + C.c3 * mag
        + C.c4 * np.log(pga)
        + C.c5 * np.log(psa_1s)
    )
    if hanging_wall:
        assert dist_jb is not None
        ln_arias_int += C.c8 * np.clip(1 - (dist_jb - 5) / 5, 0, 1)

    if ln_std_pga is None and ln_std_psa_1s is None:
        # Computed from residuals
        ln_std = np.interp(
            mag, [3, 4, 5, 6, 7], [0.40, 0.39, 0.37, 0.33, 0.40], left=0.40, right=0.40
        )
    else:
        raise NotImplementedError

    return ln_arias_int, ln_std


class TimeSeries:
    def __init__(self, time_step, accels, info=""):
        self._time_step = time_step
        self._accels = accels
        self._info = info

    @property
    def info(self):
        return self._info

    @property
    def time_step(self):
        return self._time_step

    @property
    def accels(self):
        return self._accels

    @classmethod
    def read_at2(cls, filename):
        with open(filename) as fp:
            next(fp)
            info = next(fp).strip()
            next(fp)
            parts = [p for p in re.split("[ ,]", next(fp)) if p]
            count = int(parts[1])
            time_step = float(parts[3])
            accels = np.array([p for l in fp for p in l.split()]).astype(float)
        return cls(time_step, accels, info)


# FIXME: Add
# @dist_lognorm
# def calc_conditional_pgv_ab19(
#         mag,
#         dist,
#         v_s30,
#
#         **kwds
# ):
#     """Compute the PGV/PGA ratio from Abrahamson and Bhasin (2018).
#
#     """
#
#     if pga is not None:
#         c_values = ()
#     if psa_1s is not None:
#
#     else:
#         C = np.rec.fromrecords(
#             ()
#         )
#
#
#     ln_mean = (
#         3.3 +
#         0.53 * mag -
#         0.14 * np.log(dist + 3) -
#         0.32 * np.log(v_s30) - np.log(980)
#     )
#     ln_std = np.sqrt(0.45 ** 2 + (0.53 * 0.3) ** 2) * np.ones_like(mag)
#     return ln_mean, ln_std


def calc_pulse_proportion_hea12(
    dist_rup: npt.ArrayLike, epsilon: npt.ArrayLike
) -> npt.ArrayLike:
    """Calculate the pulse proportion from Hayden et al. (2012).

    Parameters
    ----------
    dist_rup : float or array_like
        Closest distance to the fault rupture [km]
    epsilon : float or array_like
        Number of stanard deviations

    Returns
    -------
    Proportion of records with pulse-like behavior

    """

    prop = np.exp(0.891 - 0.188 * dist_rup + 1.230 * epsilon) / (
        1 + np.exp(0.981 - 0.188 * dist_rup + 1.230 * epsilon)
    )

    return prop
