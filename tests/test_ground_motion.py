import json

import pytest
from numpy.testing import assert_allclose

import pygeems
from . import FPATH_DATA


@pytest.mark.parametrize(
    "kind,mag,dist_rup,site_class,directivity,expected",
    [
        # Example from Rathje et. al (2014)
        ("period_mean", 7, 5, "b", False, 0.45),
        # Example from Rathje and Antonakos (2011)
        ("period_mean", 8, 2, "b", False, 0.46),
    ],
)
def test_calc_period_rea05(kind, mag, dist_rup, site_class, directivity, expected):
    actual = pygeems.ground_motion.calc_period_rea05(
        kind, mag, dist_rup, site_class, directivity
    )
    assert_allclose(actual, expected, rtol=0.01)


@pytest.mark.parametrize(
    "case", json.load(open(FPATH_DATA / "rezaeian_et_al-2014.json"))
)
def test_calc_damping_scaling_rea15(case):
    ret = pygeems.ground_motion.calc_damping_scaling_rea15(
        case["damping"], case["mag"], case["dist_rup"], comp=case["comp"]
    )

    assert_allclose(ret.dsf, case["dsf"], rtol=0.01)
    assert_allclose(ret.ln_std, case["ln_std"], rtol=0.01)


@pytest.fixture()
def timeseries():
    ts = pygeems.ground_motion.TimeSeries.read_at2(
        FPATH_DATA / "RSN4863_CHUETSU_65036EW.AT2"
    )
    return ts


def test_at2_time_step(timeseries):
    assert_allclose(timeseries.time_step, 0.01)


def test_at2_accels(timeseries):
    assert_allclose(timeseries.accels.size, 6000)
    assert_allclose(timeseries.accels[[0, -1]], [-0.2674464e-03, -0.4684600e-04])
