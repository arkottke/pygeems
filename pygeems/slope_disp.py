import collections

import numpy as np
import numba
import scipy.constants

from scipy.integrate import cumtrapz
from scipy.stats import norm

from .utils import dist_lognorm


@dist_lognorm
def calc_disp_rs08(
        yield_coef,
        pga,
        mag=None,
        pgv=None,
        **kwargs
):
    if mag is None:
        method = 'pgv'
    elif pgv is None:
        method = 'mag'
    else:
        raise NotImplementedError

    P = {
        model: np.rec.fromrecords(values, names='a1,a2,a3,a4,a5,a6,a7')
        for model, values in zip(
            ['mag', 'pgv'],
            [(4.89, -4.85, -19.64, 42.49, -29.06, 0.72, 0.89),
             (-1.56, -4.58, -20.84, 44.75, -30.50, -0.64, 1.55)])
    }[method]

    yp_ratio = yield_coef / pga
    ln_disp = (
            P.a1 + P.a2 * yp_ratio +
            P.a3 * yp_ratio ** 2 + P.a4 * yp_ratio ** 3 +
            P.a5 * yp_ratio ** 4 + P.a6 * np.log(pga)
    )
    if method == 'mag':
        ln_disp += P.a7 * (mag - 6)
    else:
        ln_disp += P.a7 * np.log(pgv)

    if method == 'mag':
        ln_std = 0.73 + 0.79 * yp_ratio - 0.54 * yp_ratio ** 2
    else:
        ln_std = 0.41 + 0.52 * yp_ratio

    return ln_disp, ln_std


@dist_lognorm
def calc_disp_ra11(
        yield_coef,
        pga,
        period_slide,
        period_mean,
        mag=None,
        pgv=None,
        **kwargs
):
    if mag is None:
        method = 'pgv'
    elif pgv is None:
        method = 'mag'
    else:
        raise NotImplementedError

    period_ratio = period_slide / period_mean
    if period_ratio < 0.1:
        k_max = pga
    else:
        ln_pr = np.log(period_ratio / 0.1)
        ln_ratio = (
                (0.459 - 0.702 * pga) * ln_pr +
                (-0.228 + 0.076 * pga) * ln_pr ** 2
        )
        k_max = pga * np.exp(ln_ratio)

    # Compute median
    if method == 'mag':
        ln_disp = \
            calc_disp_rs08(yield_coef, k_max, mag=mag, stats=True)[0]
        f_1 = np.minimum(
            3.69 * period_slide - 1.22 * period_slide ** 2,
            2.78
        )
        ln_disp += f_1
    else:
        if period_ratio < 0.2:
            k_velmax = pgv
        else:
            ln_pr = np.log(period_ratio / 0.2)
            ln_ratio = (
                    0.240 * ln_pr + (-0.091 - 0.171 * pga) * ln_pr ** 2
            )
            k_velmax = pgv * np.exp(ln_ratio)

        ln_disp = \
            calc_disp_rs08(yield_coef, k_max, pgv=k_velmax, stats=True)[0]

        f_2 = np.minimum(1.42 * period_slide, 0.71)
        ln_disp += f_2

    # Compute the standard deviation
    if method == 'mag':
        ln_std = 0.694 + 0.322 * (yield_coef / k_max)
    else:
        ln_std = 0.400 + 0.284 * (yield_coef / k_max)

    return ln_disp, ln_std


def _calc_wla06_ln_a_rms(pga):
    """Calculate $a_{rms}$ from WLA06

    Provided by Equation (2).
    """
    return -1.167 + 1.02 * np.log(pga)


def _calc_wla06_ln_dur_key(yield_coef, pga, psa_1s, mag):
    # Simplification
    ln_pga = np.log(pga)
    ln_pga_ky = np.log(pga / yield_coef)
    ln_psa_1s = np.log(psa_1s)

    return (
        -2.775 +
        0.956 * ln_pga_ky -
        1.554 / (ln_pga_ky + 0.390) -
        0.597 * ln_pga +
        0.381 * ln_psa_1s +
        0.334 * mag
    )


@dist_lognorm
def calc_disp_wla06(
        yield_coef,
        pga,
        psa_1s,
        mag,
        **kwds
):
    # Constants from Table 1.
    # Updated values were provided by Jennie
    a1 = 5.463
    b1 = 0.451
    b2 = 0.0191
    c1 = 0.591
    d1 = 0.205
    d2 = 0.0892
    e1 = 1.039
    e2 = 0.0427
    f1 = -1.409
    f2 = 0.1

    # Simplification
    ln_psa_1s = np.log(psa_1s)

    ln_dur_ky = _calc_wla06_ln_dur_key(yield_coef, pga, psa_1s, mag)
    ln_a_rms = _calc_wla06_ln_a_rms(pga)

    ln_pga_ky = np.log(pga / yield_coef)
    ln_psa_pga = np.log(psa_1s / pga)

    ln_mean = (
        a1 +
        b1 * (ln_psa_1s + 0.45) +
        b2 * (ln_psa_1s + 0.45) ** 2 +
        c1 * (ln_a_rms - 1) +
        d1 * ln_psa_pga +
        d2 * ln_psa_pga ** 2 +
        e1 * (ln_dur_ky - 0.74) +
        e2 * (ln_dur_ky - 0.74) ** 2 +
        1 / (f1 * (ln_pga_ky + f2))
    )
    # Equation not valid for PGA values less than the yield coef
    ln_mean[pga < yield_coef] = np.nan
    # From Figure 5
    ln_std = 0.53

    return ln_mean, ln_std


@dist_lognorm
def calc_disp_bt07(yield_coef, period_slide, psa_dts, **kwds):
    # Simplification
    ln_yield_coef = np.log(yield_coef)
    ln_psa_dts = np.log(psa_dts)

    # Median displacement
    # Equation (4)
    ln_mean = -0.22 if period_slide < 0.05 else -1.10
    ln_mean += (
        -2.83 * ln_yield_coef -
        0.333 * ln_yield_coef ** 2 +
        0.566 * ln_yield_coef * ln_psa_dts +
        3.04 * ln_psa_dts -
        0.244 * ln_psa_dts ** 2 +
        1.5 * period_slide
    )
    ln_std = 0.66
    return ln_mean, ln_std


def calc_prob_disp_bt07(yield_coef, period_slide, psa_dts, **kwds):
    # Probability of a non-zero displacement
    # Modified from Equation (3)
    ln_yield_coef = np.log(yield_coef)

    prob_disp = norm.cdf(
        -1.76
        - 3.22 * ln_yield_coef
        - 0.484 * period_slide * ln_yield_coef
        + 3.52 * np.log(psa_dts)
    )

    return prob_disp


@numba.jit
def _calc_block_velocity(
        time_step,
        accels,
        yield_coef
):
    """Compute the velocity of a sliding block.

    The calculation is adapted from Slammer's source code[1]_. However,
    the displacement is computed outside of the numerical integration.

    .. [1] https://github.com/mjibson/slammer/blob/master/programs/slammer/analysis/Decoupled.java#L135

    Parameters
    ----------
    time_step : float
        time series step size [seconds]
    accels : array_like
        acceleration time series [cm/sec/sec]
    yield_coef : float
        yield coefficient of the slope [cm/sec/sec]

    Returns
    -------
    vels : :class:`numpy.array`
        velocity time series of the sliding mass
    """
    vels = np.zeros_like(accels)
    sliding = False

    for i in range(1, len(accels)):
        # Compute the velocity and displacement of the sliding mass
        accel_avg = 0.5 * (accels[i] + accels[i - 1])

        if sliding:
            vels[i] = (
                vels[i - 1] + (accel_avg - yield_coef) * time_step
            )
        else:
            vels[i] = 0

        # Check if sliding
        if sliding:
            if vels[i] <= 0:
                sliding = False
                vels[i] = 0
        else:
            if accels[i] > yield_coef:
                sliding = True

    return vels


def calc_rigid_disp(
        time_step,
        accels,
        yield_coef,
        invert=False
):
    """Compute the displacement and velocity of a rigid sliding mass.

    Parameters
    ----------
    time_step : float
        time series step size [seconds]
    accels : array_like
        acceleration time series [g]
    yield_coef : float
        yield coefficient of the slope [g]
    invert : bool (optional)
        if the time series should be inverted
    Returns
    -------
    disps : :class:`numpy.array`
        displacement time series of the sliding mass
    vels : :class:`numpy.array`
        velocity time series of the sliding mass
    """
    # Convert into cm/s/s
    scale = scipy.constants.g / scipy.constants.centi

    yield_coef *= scale
    # Only invert the acceleration time series
    if invert:
        scale *= -1
    # Don't update the array inplace
    accels = accels * scale

    vels = _calc_block_velocity(time_step, accels, yield_coef)
    disps = np.r_[0, cumtrapz(vels, dx=time_step)]

    return disps, vels
