import pytest

import pygeems

from numpy.testing import assert_allclose


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
