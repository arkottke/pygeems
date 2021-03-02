import numpy as np

from . import FPATH_DATA
from .utils import check_bounds, check_options, dist_lognorm


def calc_damping_scaling_rea15(damping, mag, dist_rup, comp="rotd50", periods=None):
    """Compute the damping scaling proposed by Rezaeian et al. (2014)."""
    assert 0.5 <= damping <= 30
    assert comp in ["rotd50", "roti50"]

    C = np.genfromtxt(
        FPATH_DATA / f"rezaeian_et_al_2014-{comp}.csv",
        delimiter=",",
        skip_header=1,
        names=True,
    ).view(np.recarray)

    ln_damp = np.log(damping)
    ln_dsf = (
        C.b0
        + C.b1 * ln_damp
        + C.b2 * ln_damp ** 2
        + (C.b3 + C.b4 * ln_damp + C.b5 * ln_damp ** 2) * mag
        + (C.b6 + C.b7 * ln_damp + C.b8 * ln_damp ** 2) * np.log(dist_rup + 1)
    )
    ln_damp_5 = np.log(damping / 5)
    ln_std = np.abs(C.a0 * ln_damp_5 + C.a1 * ln_damp_5 ** 2)

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
def calc_period_rea05(kind, mag, dist_rup, site_class="c", directivity=False, **kwargs):
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
    ln_std = np.sqrt(intra ** 2 + inter ** 2)

    return ln_period, ln_std


@dist_lognorm
def calc_aris_intensity_aea16(
    mag,
    v_s30,
    pga,
    psa_1s,
    hanging_wall=False,
    dist_jb=None,
    ln_std_pga=None,
    ln_std_psa_1s=None,
    **kwargs,
):
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
        ln_arias_int += C.c8 * np.clip(1 - (dist_jb - 5) / 5, 0, 1)

    if ln_std_pga is None and ln_std_psa_1s is None:
        # Computed from residuals
        ln_std = np.interp(
            mag, [3, 4, 5, 6, 7], [0.40, 0.39, 0.37, 0.33, 0.40], left=0.40, right=0.40
        )
    else:
        raise NotImplementedError

    return ln_arias_int, ln_std
