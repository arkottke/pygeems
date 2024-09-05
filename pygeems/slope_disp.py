import numpy as np
import numba
import scipy.constants

from scipy.integrate import cumtrapz
from scipy.stats import norm

import pyrotd

from typing import Optional

from .utils import dist_lognorm


@dist_lognorm
def calc_disp_sr09(
        yield_coef: float,
        pga: float,
        mag: Optional[float] = None,
        pgv: Optional[float] = None,
        **kwargs
):
    """Calculate the displacement predicted by Saygili and Rathje (2009).

    Two models are provided:
        - PGV
        - PGA and M_w
    The standard deviation model is only provided for the *PGV* based model.

    Parameters
    ----------
    yield_coef: float
        yield coefficient of the slope [g]
    pga: float
        peak ground acceleration of the ground motion [g]
    mag: float or None
        moment magnitude of the event
    pgv: float or None
        peak ground velocity of the ground motion [cm/sec]
    """

    if mag is None:
        method = 'pgv'
    elif pgv is None:
        method = 'mag'
    else:
        raise NotImplementedError

    if method == 'pgv':
        C = np.rec.fromrecords(
            [-1.56, -4.58, -20.84, 44.75, -30.50, -0.64, 1.55],
            names='a1,a2,a3,a4,a5,a6,a7'
        )
    else:
        C = np.rec.fromrecords(
            [4.89, -4.85, -19.64, 42.49, -29.06, 0.72, 0.89],
            names='a1,a2,a3,a4,a5,a6,a7'
        )

    yp_ratio = yield_coef / pga
    ln_disp = (
            C.a1 + C.a2 * yp_ratio +
            C.a3 * yp_ratio ** 2 + C.a4 * yp_ratio ** 3 +
            C.a5 * yp_ratio ** 4 + C.a6 * np.log(pga)
    )
    if method == 'mag':
        ln_disp += C.a7 * (mag - 6)
    else:
        ln_disp += C.a7 * np.log(pgv)

    if method == 'mag':
        ln_std = 0.73 + 0.79 * yp_ratio - 0.54 * yp_ratio ** 2
    else:
        ln_std = 0.41 + 0.52 * yp_ratio

    return ln_disp, ln_std


@dist_lognorm
def calc_disp_ra11(
        yield_coef: float,
        pga: float,
        period_slide: float,
        period_mean: float,
        mag: Optional[float] = None,
        pgv: Optional[float] = None,
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
            calc_disp_sr09(yield_coef, k_max, mag=mag, stats=True)[0]
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
            calc_disp_sr09(yield_coef, k_max, pgv=k_velmax, stats=True)[0]

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
        yield_coef: float,
        pga: float,
        psa_1s: float,
        mag: float,
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

    ln_mean = np.atleast_1d(
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
def calc_disp_bt07(
        yield_coef: float,
        period_slide: float,
        psa_dts: float,
        **kwds
):
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


def calc_prob_disp_bt07(
        yield_coef: float,
        period_slide: float,
        psa_dts: float,
):
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
        time_step: float,
        accels: np.ndarray,
        yield_coef: float
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
        time_step: float,
        accels: np.ndarray,
        yield_coef: float,
        invert: bool = False
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

    return {
        'disps': disps,
        'vels': vels
    }


class HaleAbrahamson19:
    """
    Parameters
    ----------
    height: float
        Height in [ft]
    shear_vel: float
        Shear-wave velocity in [ft]
    freq_nat: float
        Natural frequency in [Hz]
    nl_model: str, 'darendeli' or 'vucetic_dobry'
        Nonlinear model of the material
    """
    C = np.rec.fromrecords(
        [0.76, 0.35, -0.00049, 0.00329, -0.03558, 0.35297, -0.89726, 1.25375,
         0.00024, -1.14426, 0.00090, -1.91824, 0.00145, -0.00993, 0.86560,
         0.57934, 0.00172, 0.40392, 0.85],
        names='a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,alpha'
    )

    NL_MODELS = ('darendeli', 'vucetic_dobry')

    def __init__(self,
                 height: float,
                 shear_vel: Optional[float] = None,
                 freq_nat: Optional[float] = None,
                 nl_model: str = 'darendeli'):

        assert nl_model in self.NL_MODELS

        self._height = height
        self._nl_model = nl_model

        if shear_vel is None:
            self._freq_nat = freq_nat
            self._shear_vel = 3 * self.height * freq_nat
        elif freq_nat is None:
            self._shear_vel = shear_vel
            self._freq_nat = shear_vel / (3 * height)
        else:
            raise NotImplementedError

    @property
    def freq_nat(self):
        return self._freq_nat

    @property
    def height(self):
        return self._height

    @property
    def shear_vel(self):
        return self._shear_vel

    def __call__(self,
                 time_step: float,
                 accels: np.ndarray,
                 yield_coef: float ):
        # Compute acceleration of sliding mass
        accels = self._calc_sliding_mass_accels(time_step, accels)

        disps = [
            calc_rigid_disp(
                time_step, accels, yield_coef, invert=invert)['disps'][-1]
            for invert in [True, False]
        ]
        return disps

    def _calc_sliding_mass_accels(self,
                                  time_step: float,
                                  accels: np.ndarray ):
        # Compute the Fourier amplitude spectra
        accels = np.asarray(accels)
        fourier_amps = np.fft.rfft(accels)
        freqs = np.fft.rfftfreq(accels.size, d=time_step)
        # Compute the nonlinear period
        period_nl = self.C.a1 * np.log(1 / self.freq_nat) + self.C.a2
        ln_spec_accel = np.log(
            pyrotd.calc_oscillator_resp(
                freqs, fourier_amps, 0.05, 1 / period_nl,
                peak_resp_only=True, osc_type='psa')
        )

        trans_func = self._calc_trans_func(freqs, ln_spec_accel)
        fourier_amps *= trans_func

        return np.fft.irfft(fourier_amps)

    def _calc_trans_func(self,
                         freqs: np.ndarray,
                         ln_spec_accel: np.ndarray ):
        freqs = np.asarray(freqs)
        ln_height = np.log(self.height)

        if self._nl_model == 'darendeli':
            c3 = 1
            b5 = 0
            b6 = 0
        elif self._nl_model == 'vucetic_dobry':
            c3 = 0.32
            b5 = -2.0
            b6 = -1.0
        else:
            raise NotImplementedError

        b1 = self.C.a9 * self.shear_vel + self.C.a10
        b2 = self.C.a11 * self.shear_vel + self.C.a12
        b3 = (
            (self.C.a13 * ln_height + self.C.a14) * self.shear_vel +
            (self.C.a15 * ln_height + self.C.a16)
        )
        b4 = self.C.a17 * self.shear_vel + self.C.a18
        damping = np.exp(
            b1 + b2 / (1 + np.exp(b3 + b5 + (b4 + b6) * ln_spec_accel)))

        c1 = (
            (self.C.a3 * ln_height + self.C.a4) * self.shear_vel +
            (self.C.a5 * ln_height + self.C.a6)
        )
        c2 = self.C.a7 * c1 + self.C.a8
        period_ratio = 1 + c3 * np.exp(c1 * (c2 + ln_spec_accel))
        # Compute the effective natural frequency of the dam correcting for the
        # shortening of the frequency due to nonlinearity and damping.
        nat_freq_eff = self.freq_nat / ((1 - damping ** 2) * period_ratio)

        trans_func = self.C.alpha * (
            1 + (1j ** 2 / (freqs ** 2 - nat_freq_eff ** 2 -
                            2j * nat_freq_eff * freqs))
        )
        # Limit transfer function to be at least one below the natural
        # frequency.
        mask = (freqs < nat_freq_eff) & (trans_func < 1.)
        trans_func[mask] = 1.

        return trans_func

@dist_lognorm
def calc_disp_cr20(
        yield_coef: float,
        period_slide: float,
        pga: Optional[float] = None,
        mag: Optional[float] = None,
        pgv: Optional[float] = None):
    """Calculate the displacement predicted by Cho and Rathje (2020).

    Two models are provided:
        - PGV
        - PGA and M_w
    The standard deviation model is only provided for the *PGV* based model.

    Parameters
    ----------
    yield_coef: float
        yield coefficient of the slope [g]
    period_slide: float
        period of the sliding mass [sec]
    pga: float or None
        peak ground acceleration of the ground motion [g]
    mag: float or None
        moment magnitude of the event
    pgv: float or None
        peak ground velocity of the ground motion [cm/sec]
    """
    # Check the inputs
    if pgv is None and pga and mag:
        method = 'pga'
    elif pgv and pga is None and mag is None:
        method = 'pgv'
    else:
        raise NotImplementedError

    if method == 'pgv':
        C = np.rec.fromrecords(
            [-3.1355, -0.4253, 0.8802, -3.2971, 1.3502, 0.0313,
             0.3719, -0.1137, 0.0433, 0.1356],
            names='b0,b1,b2,b3,c0,c1,e0,e1,e2,p1'
        )
    else:
        C = np.rec.fromrecords(
            [2.6482, -0.2530, 0.8802, -3.2971, 1.3058, 0.0577, 0.6002],
            names='b0,b1,b2,b3,c0,c1,d0'
        )

    # Equation 5
    a0 = C.b0 + C.b1 * np.log(yield_coef)
    if period_slide > 0.1:
        a0 += (C.b2 + C.b3 * yield_coef) * np.log(period_slide / 0.1)

    a1 = C.c0 + C.c1 * np.log(yield_coef)

    if method == 'pgv':
        ln_disp = a0 + a1 * np.log(pgv)
    else:
        ln_disp = a0 + a1 * np.log(pga) + C.d0 * (mag - 6.5)

    if method == 'pgv':
        p0 = C.e0
        if pgv < 2:
            p0 -= C.e1 * np.log(pgv / 2)
        elif pgv > 50:
            p0 += C.e2 * np.log(pgv / 50)

        phi = p0
        if yield_coef > 0.2:
            phi += C.p1 * np.log(yield_coef / 0.2)

        tau = 0.177
        ln_std = np.sqrt(phi ** 2 + tau ** 2)
    else:
        ln_std = np.nan

    return ln_disp, ln_std

@dist_lognorm
def calc_dam_period_pk19(
        height: float,
        direction: str = 'transverse',
        **kwds
):
    if direction in ('lon', 'longitudinal'):
        a = -2.629
        b = 0.377
        ln_std = 0.464
    elif direction in ('trans', 'transverse'):
        a = -2.685
        b = 0.430
        ln_std = 0.375
    elif direction in ('vert', 'vertical'):
        a = -2.793
        b = 0.283
        ln_std = 0.456
    else:
        raise NotImplementedError

    ln_period = a + b * np.log(height)

    return ln_period, ln_std
