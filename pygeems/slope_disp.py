"""pyGEEMs: slope displacement functions and classes."""

import collections
from typing import Optional
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from scipy.stats import norm

from typing import Optional

from .utils import dist_lognorm

FloatOrArrayLike = Union[float, ArrayLike]


class Coefficients(collections.abc.Mapping):
    """Read-only container for model coefficients."""

    def __init__(self, **kwds):
        self._data = kwds

    def __getitem__(self, key):
        return self._data[key]

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


@dist_lognorm
def calc_disp_rs08(
    yield_coef: float,
    pga: FloatOrArrayLike,
    mag: Optional[float] = None,
    pgv: Optional[float] = None,
    **kwargs,
):
    """Rathje and Saygili (2008) slope displacement model.

    Two models are provided:
        - PGV
        - PGA and M_w
    The standard deviation model is only provided for the *PGV* based model.

    Parameters
    ----------
    yield_coef : float
        Slope yield coefficient
    pga : float or `array_like`
        Peak ground acceleration in [gravity]
    mag : float
        Earthquake magnitude
    pgv : float
        Peak ground velocity in [cm/s]
    stats : bool, default=False
        If a distribution should be returned

    Returns
    -------
    median : float or `numpy.ndarray`
        Median value. If `stats` is False.
    ln_mean : float or `numpy.ndarray`
        Natural log mean value. Returned if `stats` is True.
    ln_std : float or `numpy.ndarray`
        Natural log standard deviation. Returned if `stats` is True.
    """
    if mag is None:
        method = "pgv"
    elif pgv is None:
        method = "mag"
    else:
        raise NotImplementedError

    P = {
        model: np.rec.fromrecords(values, names="a1,a2,a3,a4,a5,a6,a7")
        for model, values in zip(
            ["mag", "pgv"],
            [
                (4.89, -4.85, -19.64, 42.49, -29.06, 0.72, 0.89),
                (-1.56, -4.58, -20.84, 44.75, -30.50, -0.64, 1.55),
            ],
        )
    }[method]

    yp_ratio = yield_coef / pga
    ln_disp = (
        P.a1
        + P.a2 * yp_ratio
        + P.a3 * yp_ratio**2
        + P.a4 * yp_ratio**3
        + P.a5 * yp_ratio**4
        + P.a6 * np.log(pga)
    )
    if method == "mag":
        ln_disp += P.a7 * (mag - 6)
    else:
        ln_disp += P.a7 * np.log(pgv)

    if method == "mag":
        ln_std = 0.73 + 0.79 * yp_ratio - 0.54 * yp_ratio**2
    else:
        ln_std = 0.41 + 0.52 * yp_ratio

    return ln_disp, ln_std


@dist_lognorm
def calc_disp_ra11(
    yield_coef: float,
    pga: float | npt.ArrayLike,
    period_slide: float,
    period_mean: float,
    mag: Optional[float] = None,
    pgv: Optional[float] = None,
    **kwargs,
):
    """Rathje and Antonakos (2011) slope displacement model."""
    if mag is None:
        method = "pgv"
    elif pgv is None:
        method = "mag"
    else:
        raise NotImplementedError

    period_ratio = period_slide / period_mean
    if period_ratio < 0.1:
        k_max = pga
    else:
        ln_pr = np.log(period_ratio / 0.1)
        ln_ratio = (0.459 - 0.702 * pga) * ln_pr + (-0.228 + 0.076 * pga) * ln_pr**2
        k_max = pga * np.exp(ln_ratio)

    # Compute median
    if method == "mag":
        ln_disp = calc_disp_rs08(yield_coef, k_max, mag=mag, stats=True)[0]
        f_1 = np.minimum(3.69 * period_slide - 1.22 * period_slide**2, 2.78)
        ln_disp += f_1
    else:
        if period_ratio < 0.2:
            k_velmax = pgv
        else:
            ln_pr = np.log(period_ratio / 0.2)
            ln_ratio = 0.240 * ln_pr + (-0.091 - 0.171 * pga) * ln_pr**2
            k_velmax = pgv * np.exp(ln_ratio)

        ln_disp = calc_disp_rs08(yield_coef, k_max, pgv=k_velmax, stats=True)[0]

        f_2 = np.minimum(1.42 * period_slide, 0.71)
        ln_disp += f_2

    # Compute the standard deviation
    if method == "mag":
        ln_std = 0.694 + 0.322 * (yield_coef / k_max)
    else:
        ln_std = 0.400 + 0.284 * (yield_coef / k_max)

    return ln_disp, ln_std


def _calc_wla06_ln_a_rms(pga):
    """Calculate $a_{rms}$ from WLA06.

    Provided by Equation (2).
    """
    return -1.167 + 1.02 * np.log(pga)


def _calc_wla06_ln_dur_key(yield_coef: float, pga: float, psa_1s: float, mag: float):
    """Watson-Lamprey and Abrahamson (2006) duration calculation.

    FIXME

    """
    # Simplification
    ln_pga = np.log(pga)
    ln_pga_ky = np.log(pga / yield_coef)
    ln_psa_1s = np.log(psa_1s)

    return (
        -2.775
        + 0.956 * ln_pga_ky
        - 1.554 / (ln_pga_ky + 0.390)
        - 0.597 * ln_pga
        + 0.381 * ln_psa_1s
        + 0.334 * mag
    )


@dist_lognorm
def calc_disp_wla06(yield_coef: float, pga: float, psa_1s: float, mag: float, **kwds):
    """Watson-Lamprey and Abrahamson (2006) slope displacement model."""
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

    pga = np.asarray(pga)
    ln_pga_ky = np.log(pga / yield_coef)
    ln_psa_pga = np.log(psa_1s / pga)

    ln_mean = (
        a1
        + b1 * (ln_psa_1s + 0.45)
        + b2 * (ln_psa_1s + 0.45) ** 2
        + c1 * (ln_a_rms - 1)
        + d1 * ln_psa_pga
        + d2 * ln_psa_pga**2
        + e1 * (ln_dur_ky - 0.74)
        + e2 * (ln_dur_ky - 0.74) ** 2
        + 1 / (f1 * (ln_pga_ky + f2))
    )

    # Equation not valid for PGA values less than the yield coef
    try:
        ln_mean[pga < yield_coef] = np.nan
    except TypeError:
        if ln_mean < yield_coef:
            ln_mean = np.nan

    # From Figure 5
    ln_std = 0.53

    return ln_mean, ln_std


@dist_lognorm
def calc_disp_bt07(yield_coef: float, period_slide: float, psa_dts: float, **kwds):
    """Bray and Travasarou (2007)."""
    # Simplification
    ln_yield_coef = np.log(yield_coef)
    ln_psa_dts = np.log(psa_dts)

    # Median displacement
    # Equation (4)
    ln_mean = -0.22 if period_slide < 0.05 else -1.10
    ln_mean += (
        -2.83 * ln_yield_coef
        - 0.333 * ln_yield_coef**2
        + 0.566 * ln_yield_coef * ln_psa_dts
        + 3.04 * ln_psa_dts
        - 0.244 * ln_psa_dts**2
        + 1.5 * period_slide
    )
    ln_std = 0.66
    return ln_mean, ln_std


def calc_prob_disp_bt07(yield_coef: float, period_slide: float, psa_dts: float, **kwds):
    """Bray and Travasarou (2007), probability of non-zero displacement."""
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


@dist_lognorm
def calc_disp_cr21(
    pgv: float, yield_coef: float, period_slide: float, height_ratio: float, **kwds
):
    """ANN Model 1 from Cho and Rathje (2021)."""

    # Means from Table 2, but the mean of the ln(PGV) is replaced based on Equation (2)
    ln_means = np.array([5.855, -3.873, -1.498, -2.262])
    # Bias from Table 3
    b_out = -4.687
    # Weights and coefficients from Table 3
    coefs = np.array(
        [
            [5.262, 0.66, -2.709, -0.354, 0.505, 3.024],
            [1.678, 1.054, 1.686, -4.184, -2.9, 3.303],
            [4.253, 0.527, -0.876, -0.265, -2.818, 3.044],
            [0.82, 7.245, 1.074, 6.094, -3.828, 2.14],
            [4.05, -0.154, -1.238, 0.907, -2.904, 3.449],
        ]
    )

    # Median prediction, Equation 2
    x = np.r_[
        np.log([pgv, yield_coef, period_slide, height_ratio]) / ln_means,
        # Dummy values for including the b_i value
        1,
    ]
    ln_mean = b_out + coefs[:, -1] @ (1 / (1 + np.exp(-coefs[:, :-1] @ x)))

    # Standard deviation model. Equations 6 & 7.
    phi = 0.27 - 0.137 * np.log(period_slide)
    tau = 0.106
    ln_std = np.sqrt(phi**2 + tau**2)

    # FIXME: How to return phi and tau as well?
    return ln_mean, ln_std


@dist_lognorm
def calc_disp_cr22(
    yield_coef: float,
    period_slide: float,
    height_ratio: float,
    pgv: Optional[float] = None,
    pga: Optional[float] = None,
    mag: Optional[float] = None,
    **kwds,
):
    """Displacement model from Cho and Rathje (2022)."""

    ln_yield_coef = np.log(yield_coef)
    ln_period_slide = np.log(period_slide)

    pgv_model = pga is None

    if pgv_model:
        C = Coefficients(
            b_0=-1.01,
            b_1=1.57,
            b_2=-0.25,
            c_0=0.81,
            c_1=-1.05,
            c_2=-0.60,
            b_3=-4.50,
            b_4=-1.37,
            c_3=1.51,
            c_4=0.10,
            p_0=0.26,
            p_1=-0.14,
            p_2=0.40,
            p_3=0.16,
        )
    else:
        C = Coefficients(
            b_0=3.94,
            b_1=1.28,
            b_2=-0.27,
            c_0=1.21,
            c_1=0.29,
            c_2=0.25,
            b_3=1.96,
            b_4=-0.89,
            c_3=1.47,
            c_4=0.14,
            e_0=0.56,
            e_1=0.66,
        )

    # Mean model
    if height_ratio <= 0.6:
        a_0 = C.b_0 + C.b_1 * ln_period_slide + C.b_2 * ln_yield_coef
    else:
        a_0 = C.b_3 + C.b_4 * ln_yield_coef

    if height_ratio <= 0.6:
        a_1 = C.c_0 + C.c_1 * ln_period_slide + C.c_2 * ln_period_slide**2
    else:
        a_1 = C.c_3 + C.c_4 * ln_yield_coef

    if pgv_model:
        ln_mean = a_0 + a_1 * np.log(pgv)
    else:
        d_0 = C.e_0 if height_ratio <= 0.6 else C.e_1
        ln_mean = a_0 + a_1 * np.log(pga) + d_0 * (mag - 6.5)

    # Uncertainty
    if pgv_model:
        if height_ratio <= 0.6:
            phi = C.p_0 + C.p_1 * ln_period_slide
        else:
            flag_ky = 0 if yield_coef < 0.2 else 1
            phi = C.p_2 + C.p_3 * np.log(yield_coef / 0.2) * flag_ky

        tau = 0.139
    else:
        phi = 0.72
        tau = 0.143

    ln_std = np.sqrt(phi**2 + tau**2)

    return ln_mean, ln_std


@dist_lognorm
def calc_disp_bea17(
    yield_coef: float, period_slide: float, psa_dts: float, mag: float, **kwds
):
    """Bray et al. (2017)."""
    ln_yield_coef = np.log(yield_coef)
    ln_psa_dts = np.log(psa_dts)

    ln_mean = (
        -3.535 * ln_yield_coef
        - 0.390 * ln_yield_coef**2
        + 0.538 * ln_yield_coef * ln_psa_dts
        + 3.060 * ln_psa_dts
        - 0.225 * ln_psa_dts**2
        + 0.550 * mag
    )
    if np.isclose(period_slide, 0.0):
        ln_mean -= 5.864
    else:
        if period_slide < 0.1:
            a1, a2, a3 = -5.864, -9.421, 0
        else:
            a1, a2, a3 = -6.896, 3.081, -0.803
        ln_mean += a1 + a2 * period_slide + a3 * period_slide**2

    ln_std = 0.73
    return ln_mean, ln_std


def calc_prob_disp_bea17(
    yield_coef: float, period_slide: float, psa_dts: float, **kwds
):
    """Bray et al. (2017) slope displacement model for subduction events."""
    ln_yield_coef = np.log(yield_coef)

    if period_slide <= 0.7:
        z = (
            -2.64
            - 3.20 * ln_yield_coef
            - 0.17 * ln_yield_coef**2
            - 0.49 * period_slide * ln_yield_coef
            + 2.09 * period_slide
            + 2.91 * np.log(psa_dts)
        )
    else:
        z = (
            -3.53
            - 4.78 * ln_yield_coef
            - 0.34 * ln_yield_coef**2
            - 0.30 * period_slide * ln_yield_coef
            - 0.67 * period_slide
            + 2.66 * np.log(psa_dts)
        )

    prob_disp = norm.cdf(z)
    return prob_disp


@numba.jit(nopython=True)
def _calc_block_velocity(time_step: float, accels: ArrayLike, yield_coef: float):
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
            vels[i] = vels[i - 1] + (accel_avg - yield_coef) * time_step
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
    time_step: float, accels: npt.ArrayLike, yield_coef: float, invert: bool = False
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
    disps = np.r_[0, cumulative_trapezoid(vels, dx=time_step)]

    return {"disps": disps, "vels": vels}


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
        [
            0.76,
            0.35,
            -0.00049,
            0.00329,
            -0.03558,
            0.35297,
            -0.89726,
            1.25375,
            0.00024,
            -1.14426,
            0.00090,
            -1.91824,
            0.00145,
            -0.00993,
            0.86560,
            0.57934,
            0.00172,
            0.40392,
            0.85,
        ],
        names="a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,alpha",
    )

    NL_MODELS = ("darendeli", "vucetic_dobry")

    def __init__(
        self,
        height: float,
        shear_vel: Optional[float] = None,
        freq_nat: Optional[float] = None,
        nl_model: str = "darendeli",
    ):
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

    def __call__(self, time_step: float, accels: np.ndarray, yield_coef: float):
        # Compute acceleration of sliding mass
        accels = self._calc_sliding_mass_accels(time_step, accels)

        disps = [
            calc_rigid_disp(time_step, accels, yield_coef, invert=invert)["disps"][-1]
            for invert in [True, False]
        ]
        return disps

    def _calc_sliding_mass_accels(self, time_step: float, accels: np.ndarray):
        # Compute the Fourier amplitude spectra
        accels = np.asarray(accels)
        fourier_amps = np.fft.rfft(accels)
        freqs = np.fft.rfftfreq(accels.size, d=time_step)
        # Compute the nonlinear period
        period_nl = self.C.a1 * np.log(1 / self.freq_nat) + self.C.a2
        ln_spec_accel = np.log(
            pyrotd.calc_oscillator_resp(
                freqs,
                fourier_amps,
                0.05,
                1 / period_nl,
                peak_resp_only=True,
                osc_type="psa",
            )
        )

        trans_func = self._calc_trans_func(freqs, ln_spec_accel)
        fourier_amps *= trans_func

        return np.fft.irfft(fourier_amps)

    def _calc_trans_func(self, freqs: np.ndarray, ln_spec_accel: np.ndarray):
        freqs = np.asarray(freqs)
        ln_height = np.log(self.height)

        if self._nl_model == "darendeli":
            c3 = 1
            b5 = 0
            b6 = 0
        elif self._nl_model == "vucetic_dobry":
            c3 = 0.32
            b5 = -2.0
            b6 = -1.0
        else:
            raise NotImplementedError

        b1 = self.C.a9 * self.shear_vel + self.C.a10
        b2 = self.C.a11 * self.shear_vel + self.C.a12
        b3 = (self.C.a13 * ln_height + self.C.a14) * self.shear_vel + (
            self.C.a15 * ln_height + self.C.a16
        )
        b4 = self.C.a17 * self.shear_vel + self.C.a18
        damping = np.exp(b1 + b2 / (1 + np.exp(b3 + b5 + (b4 + b6) * ln_spec_accel)))

        c1 = (self.C.a3 * ln_height + self.C.a4) * self.shear_vel + (
            self.C.a5 * ln_height + self.C.a6
        )
        c2 = self.C.a7 * c1 + self.C.a8
        period_ratio = 1 + c3 * np.exp(c1 * (c2 + ln_spec_accel))
        # Compute the effective natural frequency of the dam correcting for the
        # shortening of the frequency due to nonlinearity and damping.
        nat_freq_eff = self.freq_nat / ((1 - damping**2) * period_ratio)

        trans_func = self.C.alpha * (
            1 + (1j**2 / (freqs**2 - nat_freq_eff**2 - 2j * nat_freq_eff * freqs))
        )
        # Limit transfer function to be at least one below the natural
        # frequency.
        mask = (freqs < nat_freq_eff) & (trans_func < 1.0)
        trans_func[mask] = 1.0

        return trans_func


@dist_lognorm
def calc_disp_cr20(
    yield_coef: float,
    period_slide: float,
    pga: Optional[float] = None,
    mag: Optional[float] = None,
    pgv: Optional[float] = None,
):
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
        method = "pga"
    elif pgv and pga is None and mag is None:
        method = "pgv"
    else:
        raise NotImplementedError

    if method == "pgv":
        C = np.rec.fromrecords(
            [
                -3.1355,
                -0.4253,
                0.8802,
                -3.2971,
                1.3502,
                0.0313,
                0.3719,
                -0.1137,
                0.0433,
                0.1356,
            ],
            names="b0,b1,b2,b3,c0,c1,e0,e1,e2,p1",
        )
    else:
        C = np.rec.fromrecords(
            [2.6482, -0.2530, 0.8802, -3.2971, 1.3058, 0.0577, 0.6002],
            names="b0,b1,b2,b3,c0,c1,d0",
        )

    # Equation 5
    a0 = C.b0 + C.b1 * np.log(yield_coef)
    if period_slide > 0.1:
        a0 += (C.b2 + C.b3 * yield_coef) * np.log(period_slide / 0.1)

    a1 = C.c0 + C.c1 * np.log(yield_coef)

    if method == "pgv":
        ln_disp = a0 + a1 * np.log(pgv)
    else:
        ln_disp = a0 + a1 * np.log(pga) + C.d0 * (mag - 6.5)

    if method == "pgv":
        p0 = C.e0
        if pgv < 2:
            p0 -= C.e1 * np.log(pgv / 2)
        elif pgv > 50:
            p0 += C.e2 * np.log(pgv / 50)

        phi = p0
        if yield_coef > 0.2:
            phi += C.p1 * np.log(yield_coef / 0.2)

        tau = 0.177
        ln_std = np.sqrt(phi**2 + tau**2)
    else:
        ln_std = np.nan

    return ln_disp, ln_std


@dist_lognorm
def calc_dam_period_pk19(height: float, direction: str = "transverse", **kwds):
    if direction in ("lon", "longitudinal"):
        a = -2.629
        b = 0.377
        ln_std = 0.464
    elif direction in ("trans", "transverse"):
        a = -2.685
        b = 0.430
        ln_std = 0.375
    elif direction in ("vert", "vertical"):
        a = -2.793
        b = 0.283
        ln_std = 0.456
    else:
        raise NotImplementedError

    ln_period = a + b * np.log(height)

    return ln_period, ln_std
