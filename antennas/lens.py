"""
Contains functions useful for lens design
"""

import numpy as np
import antenna_toolbox as ant
from scipy.special import jv
from scipy.special import struve


def ll_eps_r(r: float):
    """
    Returns the permittivity of an ideal Luneburg lens, with focal point at its circumference, at some radius from its
    center, using Eq. (1) from [1].

    Sources:
    [1] B. Fuchs, L. Le Coq, O. Lafond, S. Rondineau, and M. Himdi, “Design optimization of multishell Luneburg lenses,”
     IEEE Trans. Antennas Propag., vol. 55, no. 2, pp. 283–289, 2007.

    :param r: normalized radius / unitless
    :return: relative permittivity at the normalized radius / unitless
    """
    return 2 - r**2


def grin_lens_compute_ep_r(f: float, r: float, n: int, ep_r_range: []):
    """
    Computes permittivity of each ring of a planar GRIN lens using the method described by:

    Hernandez, C. A. M., Elmansouri, M., & Filipovic, D. S. (2019). High-Directivity Beam-Steerable Lens Antenna for
    Simultaneous Transmit and Receive. IEEE International Symposium on Phased Array Systems and Technology,
    2019-Octob. https://doi.org/10.1109/PAST43306.2019.9020904

    :param f: focal length (m; float)
    :param r: radius (m; float)
    :param n: number of rings (unitless; int)
    :param ep_r_range: array with two elements, the lower and upper permittivity value for the lens, in that order...
    ... (unitless; array)
    :return: tuple, with array that contains permittivity values of each ring AND thickness of the lens
    """

    # compute path lengths
    l_n = np.zeros(n)
    for ii in range(1, n + 1):
        l_n[ii - 1] = np.sqrt(f ** 2 + (r ** 2) * ((1 + 2 * (ii - 1)) / (2 * n)) ** 2)

    # compute lens thickness
    t = (l_n[-1] - l_n[0]) / (np.sqrt(ep_r_range[-1]) - np.sqrt(ep_r_range[0]))

    # compute permittivities for each ring
    ep_r = np.zeros(n)
    ep_r[0] = ep_r_range[-1]
    ep_r[-1] = ep_r_range[0]
    for ii in range(2, n):
        ep_r[ii - 1] = ((l_n[0] - l_n[ii - 1]) / t + np.sqrt(ep_r[0])) ** 2

    # return permittivity array and lens thickness
    return ep_r, t


def ll_2d_normalized_e_field_azimuth(az: float, r: float):
    """
    Computes the normalized E-field pattern of a 2D Luneburg lens, from the source, in the azimuthal plane (assume the
    radius of the LL is in the azimuthal plane... Eq. (8) in the source)

    G. D. M. Peeler and D. H. Archer, “A Two-Dimensional Microwave Luneherg Lens,” IRE Trans. Antennas Propag., vol. 1,
    no. 1, pp. 12–23, 1952.

    :param az: azimuth angles [deg, array-like]
    :param r: radius of lens... goes to r_0 / lambda in the source [lambda]
    :return: normalized complex field (magnitude at boresight is 1) from the 2d LL at each azimuth angle
     [normalized units, array-like]
    """

    # function throws error at az == 0, so replace that with a number very close by
    az = np.where(az == 0, 0.001, az)

    # compute intermediate variable to make actual equation less disgusting
    B = 2 * 2 * np.pi * r * ant.math_funcs.sind(az / 2)

    # compute terms... Eq. (8) = t_1 - t_2 * t_3
    t_1 = ant.math_funcs.cosd(az / 2) ** 2 * (jv(0, B) + jv(2, B))
    t_2 = 1j * 2 * ant.math_funcs.sind(az) / (np.pi * B ** 2)
    t_3 = np.sin(B * ant.math_funcs.cosd(az / 2)) - B * ant.math_funcs.cosd(az / 2) * \
          np.cos(B * ant.math_funcs.cosd(az / 2))

    return t_1 - t_2 * t_3


def ll_2d_normalized_e_field_elevation(el: float, r: float, t: float):
    """
    Computes the normalized E-field pattern of a 2D Luneburg lens, from the source, in the elevation plane (assume the
    radius of the LL is in the azimuthal plane... Eq. (9) in the source)... for some reason, the code is off by a factor
    of two so the final result is multiplied by 2... This seems to work...

    G. D. M. Peeler and D. H. Archer, “A Two-Dimensional Microwave Luneherg Lens,” IRE Trans. Antennas Propag., vol. 1,
    no. 1, pp. 12–23, 1952.

    :param az: azimuth angles [deg, array-like]
    :param r: radius of lens... goes to r_0 / lambda in the source [lambda]
    :param t: thickness of lens... aperture height in source [lambda]
    :return: normalized complex field (magnitude at boresight is 1) from the 2d LL at each azimuth angle [normalized units, array-like]
    """

    # function throws error at el == 0, so replace that with a number very close by
    el = np.where(el == 0, 0.001, el)

    # compute intermediate variable to make actual equation less disgusting
    D = 2 * 2 * np.pi * r * ant.math_funcs.sind(el / 2) ** 2

    # compute terms... Eq. (9) = (t_1 / t_2) * t_3
    t_1 = (1 + ant.math_funcs.cosd(el)) / 2 * np.cos(2 * np.pi * t * ant.math_funcs.sind(el) / 2)
    t_2 = 1 - (2 * np.pi * t * ant.math_funcs.sind(el) / np.pi) ** 2
    t_3 = (jv(0, D) - jv(1, D) / D) - 1j * (struve(0, D) - struve(1, D) / D)

    # factor of 2 added to correct the code
    return 2 * (t_1 / t_2) * t_3

