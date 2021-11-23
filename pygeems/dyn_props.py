import numpy as np

from . import PRESS_ATM


def calc_mod_shear_ws18(
    stress_mean_eff,
    fines,
    plas_index=None,
    over_consol_ratio=1.0,
    unif_coef=None,
    diam_mean=None,
    water_content=None,
    void_ratio=None,
):
    """Compute the shear modulus based on Wang & Stokoe model."""

    if fines <= 0.12:
        # Clean sands and gravels
        c_g = 63.9e3
        f_g = unif_coef ** -0.21 * void_ratio ** (-1.12 - (0.09 * diam_mean) ** 0.54)
        stress_exp = 0.48 * unif_coef ** 0.08 - 1.03 * fines
    elif fines > 0.12 and plas_index is None:
        c_g = 84.8e3
        f_g = np.exp(-0.53) * (1 - 1.32 * water_content)
        stress_exp = 0.52
    elif fines > 0.12 and plas_index > 0:
        c_g = 232.9e3
        f_g = (
            (1 + 0.96 * void_ratio) ** -2.42
            * (1.92 + over_consol_ratio) ** (0.27 + 0.46 * plas_index)
            * (1 - 0.44 * fines)
        )
        stress_exp = 0.49
    else:
        raise NotImplementedError

    return c_g * f_g * stress_mean_eff ** stress_exp


def calc_vel_shear_spt_wds12(blows, stress_vert_eff, soil_type=all, age=None):
    """Estimate the shear-wave velocity based on blow count.

    Equations from Section 4.5 of Wair, Dejong, and Shantz (2012).

    Parameters
    ----------
    blows: `array_like`
        Blow counts, N_60
    stress_vert_eff: `array_like`
        Vertical effective stress (atm)
    soil_type: str, optional
        Soil type. Possible options include: 'fine_grained', 'sand', 'gravels',
        or 'all' (default).
    age: str, optional
        Age of soil. Possible options inlude: 'holocene' or 'pleistocene'. If
        *None*, then no scaling applied.
    Returns
    -------
    vel_shear: :class:`numpy.ndarray`
        Shear-wave velocity (m/s)
    """
    C = {
        "fine_grained": {
            "coeffs": (26, 0.17, 0.32),
            "asf": {
                "holocene": 0.88,
                "pleistocene": 1.12,
            },
        },
        "sand": {
            "coeffs": (30, 0.23, 0.23),
            "asf": {
                "holocene": 0.90,
                "pleistocene": 1.17,
            },
        },
        "gravel": {
            "coeffs": (78, 0.19, 0.18),
            "asf": {
                "holocene": 53 / 78,
                "pleistocene": 115 / 78,
            },
        },
        "all": {
            "coeffs": (30, 0.215, 0.275),
            "asf": {
                "holocene": 0.87,
                "pleistocene": 1.13,
            },
        },
    }

    # Convert to kPa
    stress_vert_eff = stress_vert_eff * PRESS_ATM

    coeff, pow_blows, pow_stress = C[soil_type]["coeffs"]
    vel_shear = coeff * blows ** pow_blows * stress_vert_eff ** pow_stress

    if age is not None:
        asf = C[soil_type]["asf"][age]
        vel_shear *= asf

    return vel_shear


def calc_density(vel_shear):
    """Density model from Boore et al. (2016)

    Parameters
    ----------
    vel_shear: `array_like`
        shear-wave velocity (m/s)

    Returns
    -------
    density: :class:`numpy.ndarray` or float
        density (gm / cm³)
    """
    # Convert to km/sec. Copy, rather than modify inplace.
    vel_shear = np.asarray(vel_shear) / 1000.0
    vel_comp = calc_vel_comp(vel_shear)

    density = np.select(
        [vel_shear < 0.30, (0.30 <= vel_shear) & (vel_shear < 3.55), 3.55 <= vel_shear],
        [
            1 + 1.53 * vel_shear ** 0.85 / (0.35 + 1.889 * vel_shear ** 1.7),
            1.74 * vel_comp ** 0.25,
            1.6612 * vel_comp
            - 0.4721 * vel_comp ** 2
            + 0.0671 * vel_comp ** 3
            - 0.0043 * vel_comp ** 4
            + 0.000106 * vel_comp ** 5,
        ],
    )
    return density


def calc_vel_comp(vel_shear):
    """Compression-wave velocity (Vp) from Boore et al. (2016).

    Parameters
    ----------
    vel_shear : `array_like`
        shear-wave velocity (m/s)

    Returns
    -------
    vel_comp : np.ndarray or float
        compression-wave velocity (m/s)
    """
    vel_shear = np.asarray(vel_shear) / 1000
    vel_comp = 1000 * (
        0.9409
        + 2.0947 * vel_shear
        - 0.8206 * vel_shear ** 2
        + 0.2683 * vel_shear ** 3
        - 0.0251 * vel_shear ** 4
    )
    return vel_comp
