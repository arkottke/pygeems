import numpy as np
import pytest
from numpy.testing import assert_allclose

import pygeems

from . import FPATH_DATA


@pytest.mark.parametrize(
    "yield_coef,pga,mag,pgv,expected",
    [
        # Examples from Rathje et. al (2014)
        (0.1, 0.54, 6.75, None, 43),
        (0.1, 0.54, None, 42, 29),
        (0.1, 0.88, 6.75, None, 113),
        (0.1, 0.88, None, 71, 81),
        # Examples from Rathje & Antonakos (2011)
        # These do not work...
        # (0.1, 0.38, 8.0, None, 63.1),
        # (0.1, 0.38, None, 80, 36.9),
    ],
)
def test_calc_disp_rs08(yield_coef, pga, mag, pgv, expected):
    disp = pygeems.slope_disp.calc_disp_rs08(yield_coef, pga, mag=mag, pgv=pgv)
    # Compare to 2 cm of displacement
    assert_allclose(disp, expected, atol=2)


@pytest.mark.parametrize(
    "yield_coef,pga,period_slide,period_mean,mag,pgv,expected",
    [
        # Examples from Rathje et. al (2014)
        # Values read from Figure 8c
        (0.05, 0.35, 0.2, 0.45, 7.0, None, 87.5),
        (0.10, 0.35, 0.2, 0.45, 7.0, None, 22.0),
        (0.05, 0.35, 0.2, 0.45, None, 30, 43.5),
        (0.10, 0.35, 0.2, 0.45, None, 30, 10.5),
        # Examples from Rathje & Antonakos (2011)
        # These do not work...
        # (0.1, 0.48, 0.2, 0.46, 8.0, None, 126),
        # (0.1, 0.48, 0.2, 0.46, None, 74, 49)
    ],
)
def test_calc_disp_ra11(yield_coef, pga, period_slide, period_mean, mag, pgv, expected):
    disp = pygeems.slope_disp.calc_disp_ra11(
        yield_coef, pga, period_slide, period_mean, mag=mag, pgv=pgv
    )
    assert_allclose(disp, expected, atol=1)


def test_calc_wla06_ln_a_rms():
    a_rms = np.exp(pygeems.slope_disp._calc_wla06_ln_a_rms(0.35))
    assert_allclose(a_rms, 0.1067, atol=0.0001)


def test_calc_wla06_ln_dur_key():
    dur_ky = np.exp(pygeems.slope_disp._calc_wla06_ln_dur_key(0.11, 0.35, 0.18, 6.5))
    assert_allclose(dur_ky, 0.5896, atol=0.0001)


def test_calc_disp_wla06():
    median = pygeems.slope_disp.calc_disp_wla06(0.11, 0.35, 0.18, 6.5)
    assert_allclose(median, 3.001, atol=0.001)


# Test values from Table 1 in BT07
@pytest.mark.parametrize(
    "yield_coef,period_slide,psa_dts,expected",
    [
        (0.35, 0.45, 0.43, 0.85),
        (0.08, 0.00, 0.24, 0.1),
        (0.14, 0.33, 0.94, 0.0),
    ],
)
def test_calc_prob_disp_bt07(yield_coef, period_slide, psa_dts, expected):
    # Convert from probability of zero displacement to probability of non-zero
    # displacement
    expected = 1 - expected

    calc = pygeems.slope_disp.calc_prob_disp_bt07(yield_coef, period_slide, psa_dts)
    assert_allclose(calc, expected, atol=0.02, rtol=0.02)


# Test values from Table 1 in BT07. Authors provided a range of values, which
# are interpreted to be log-normally distributed. The median is computed and
# tested against
@pytest.mark.parametrize(
    "yield_coef,period_slide,psa_dts,expected",
    [
        (0.08, 0.00, 0.24, np.sqrt(4 * 15)),
        (0.14, 0.33, 0.94, np.sqrt(20 * 70)),
    ],
)
def test_calc_disp_bt07(yield_coef, period_slide, psa_dts, expected):
    median = pygeems.slope_disp.calc_disp_bt07(yield_coef, period_slide, psa_dts)
    assert_allclose(median, expected, rtol=0.10)


@pytest.mark.parametrize(
    "invert,yield_coef,expected",
    [
        (False, 0.05, 69.4),
        (True, 0.05, 66.1),
        (False, 0.10, 28.5),
        (True, 0.10, 30.5),
    ],
)
def test_calc_block_displacement(invert, yield_coef, expected):
    time_step = 0.01
    # Acceleration data in g
    accels = np.loadtxt(str(FPATH_DATA / "kobe-nis-000.dat"))
    yield_coef = yield_coef

    disps, vels = pygeems.slope_disp.calc_rigid_disp(
        time_step, accels, yield_coef, invert
    )
    disp_max = disps[-1]
    # Large rtol because of slightly different implementations
    assert_allclose(disp_max, expected, rtol=0.04)


# Test values from Table 2 of Bray et al. (2017). Coastline slope and Nishigo dam
@pytest.mark.parametrize(
    "yield_coef,period_slide,psa_dts,mag,expected",
    [
        (0.10, 0.6, 0.25, 8.0, np.sqrt(3 * 12)),
        (0.26, 0.15, 1.51, 9.0, np.sqrt(14 * 58)),
    ],
)
def test_calc_disp_bea17(yield_coef, period_slide, psa_dts, mag, expected):
    ret = pygeems.slope_disp.calc_disp_bea17(
        yield_coef, period_slide, psa_dts, mag, stats=False
    )
    # Limited number of digits in table make testing difficult
    assert_allclose(ret, expected, rtol=0.50)


# Test values from Table 2 of Bray et al. (2017). Esperanza and Tutuven dams.
@pytest.mark.parametrize(
    "yield_coef,period_slide,psa_dts,expected",
    [
        (0.24, 0.40, 0.43, 0.50),
        (0.39, 0.15, 0.75, 0.60),
    ],
)
def test_calc_prob_disp_bea17(yield_coef, period_slide, psa_dts, expected):
    ret = pygeems.slope_disp.calc_prob_disp_bea17(yield_coef, period_slide, psa_dts)
    # Reported values are probability of no displacement
    # Limited number of digits in table make testing difficult
    assert_allclose(ret, 1 - expected, rtol=0.25)


# Test values from Figure 13 of Cho and Rathje (20222)
@pytest.mark.parametrize(
    "yield_coef,period_slide,height_ratio,pgv,expected",
    [
        (0.20, 0.30, 0.7, 20, 5.7),
        (0.20, 0.30, 0.5, 20, 3.1),
        (0.20, 0.45, 0.5, 20, 6.8),
        (0.20, 0.60, 0.5, 20, 8.3),
        (0.20, 0.30, 0.7, 100, 50),
        (0.20, 0.30, 0.5, 100, 22),
        (0.20, 0.45, 0.5, 100, 53),
        (0.20, 0.60, 0.5, 100, 58),
    ],
)
def test_calc_disp_cr22_pgv(yield_coef, period_slide, height_ratio, pgv, expected):
    ret = pygeems.slope_disp.calc_disp_cr22(
        yield_coef, period_slide, height_ratio, pgv=pgv
    )
    # Reported values are probability of no displacement
    # Limited number of digits in table make testing difficult
    assert_allclose(ret, expected, rtol=0.05)


# Test values from Figure 14 of Cho and Rathje (2022)
@pytest.mark.parametrize(
    "yield_coef,period_slide,height_ratio,mag,pga,expected",
    [
        (0.10, 0.30, 0.7, 7.5, 0.7, 71),
        (0.20, 0.30, 0.7, 7.5, 0.7, 36),
        (0.40, 0.30, 0.7, 7.5, 0.7, 19),
    ],
)
def test_calc_disp_cr22_mag_pga(
    yield_coef, period_slide, height_ratio, mag, pga, expected
):
    ret = pygeems.slope_disp.calc_disp_cr22(
        yield_coef, period_slide, height_ratio, mag=mag, pga=pga
    )
    # Reported values are probability of no displacement
    # Limited number of digits in table make testing difficult
    assert_allclose(ret, expected, rtol=0.05)


# Tests from Figure 11 in the Cho and Rathje (2021)
@pytest.mark.parametrize(
    "yield_coef,period_slide,height_ratio,pgv,expected",
    [
        (0.1, 0.4, 0.2, 20, 6.0),
        (0.2, 0.4, 0.8, 20, 5.6),
        (0.4, 0.4, 1.6, 20, 3.1),
        (0.1, 0.3, 0.2, 20, 3.9),
        (0.2, 0.4, 0.2, 20, 5.3),
        (0.4, 0.5, 0.2, 20, 6.3),
    ],
)
def test_calc_disp_cr21(yield_coef, period_slide, height_ratio, pgv, expected):
    ret = pygeems.slope_disp.calc_disp_cr21(pgv, yield_coef, period_slide, height_ratio)
    # Reported values are probability of no displacement
    # Limited number of digits in table make testing difficult
    assert_allclose(ret, expected, rtol=0.05)


def test_ha19_trans_func():
    ha19 = pygeems.slope_disp.HaleAbrahamson19(100, freq_nat=5.0)
    np.testing.assert_allclose(ha19.shear_vel, 1500)

    trans_func = ha19._calc_trans_func([0.1], 0.01)
    np.testing.assert_allclose(trans_func, [1])


# FIXME: Add test value
def test_ha19_displacement():
    # Acceleration data in g
    ts = pygeems.ground_motion.TimeSeries.read_at2(
        FPATH_DATA / "RSN4863_CHUETSU_65036EW.AT2"
    )

    ha19 = pygeems.slope_disp.HaleAbrahamson19(50, shear_vel=800)
    disp = ha19(ts.time_step, ts.accels, 0.1)


@pytest.mark.parametrize(
    "pgv,pga,mag,expected",
    # Test values selected from Figure 3.13
    [
        (10, None, None, 5.9),
        (None, 0.1, 7, 6.1),
    ],
)
def test_calc_disp_cr20(pgv, pga, mag, expected):
    disp = pygeems.slope_disp.calc_disp_cr20(0.1, 0.6, pga, mag, pgv)
    np.testing.assert_allclose(disp, expected, rtol=0.01)
