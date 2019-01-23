
import numpy as np

from .utils import check_bounds, check_options, dist_lognorm


@dist_lognorm
def calc_period_rea05(
        kind,
        mag,
        dist_rup,
        site_class='c',
        directivity=False,
        **kwargs
):
    # Model coefficients from Table 2
    C = {
        model: np.rec.fromrecords(values, names='c1,c2,c3,c4,c5,c6')
        for model, values in zip(
            ['period_mean', 'period_avg', 'period_pred'],
            [(-1.00, 0.18, 0.0038, 0.078, 0.27, 0.40),
             (-0.89, 0.29, 0.0030, 0.070, 0.25, 0.37),
             (-1.78, 0.30, 0.0045, 0.150, 0.33, 0.24)])
    }[kind]

    check_options(site_class, 'bcd', 'site_class')
    if kind in ['period_avg', 'period_pred']:
        check_bounds(mag, 4.7, 7.6, 'mag')
    else:
        mag = np.minimum(mag, 7.25)
        check_bounds(mag, 5.0, 7.25, 'mag')

    if site_class == 'c':
        s_c = 1
        s_d = 0
    elif site_class == 'd':
        s_c = 0
        s_d = 1
    else:
        s_c = 0
        s_d = 0

    f_d = int(directivity)

    ln_period = (
            C.c1 + C.c2 * (mag - 6) + C.c3 * dist_rup +
            C.c4 * s_c + C.c5 * s_d + C.c6 * (1 - dist_rup / 20) * f_d
    )

    intra = {
        'period_mean': {'b': 0.42, 'c': 0.38, 'd': 0.31},
        'period_avg': {'b': 0.42, 'c': 0.38, 'd': 0.31},
        'period_pred': {'b': 0.42, 'c': 0.38, 'd': 0.31}
    }[kind][site_class]
    inter = {
        'period_mean': 0.17, 'period_avg': 0.13, 'period_pred': 0.22
    }[kind]
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
        **kwargs
):
    # From Table 3.1
    # Value of c8 is provided after equation 3.7
    C = np.rec.fromrecords(
        (0.47, -0.28, 0.50, 1.52, 0.21, 0.09),
        names='c1,c2,c3,c4,c5,c8'
    )

    ln_arias_int = (
            C.c1 + C.c2 * np.log(v_s30) + C.c3 * mag +
            C.c4 * np.log(pga) + C.c5 * np.log(psa_1s)
    )
    if hanging_wall:
        ln_arias_int += C.c8 * np.clip(1 - (dist_jb - 5) / 5, 0, 1)

    if ln_std_pga is None and ln_std_psa_1s is None:
        # Computed from residuals
        ln_std = np.interp(
            mag,
            [3, 4, 5, 6, 7],
            [0.40, 0.39, 0.37, 0.33, 0.40],
            left=0.40, right=0.40
        )
    else:
        raise NotImplementedError

    return ln_arias_int, ln_std
