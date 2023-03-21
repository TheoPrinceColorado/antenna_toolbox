"""
Contains functions related to electromagnetic principles/quantities
"""

import numpy as np
from scipy import constants


def free_space_eps_mu():
    """

    :return: permittivity and permeability of free space, respectively
    :rtype: tuple of floats
    """
    return constants.epsilon_0, constants.mu_0


def free_space_impedance():
    """
    
    :return: impedance of free space
    :rtype: float
    """
    return 1 / (constants.epsilon_0 * constants.speed_of_light)


def phase_constant(frequency, ep_r_prime, sigma, mu_r_prime):
    """
    Computes phase constant in a lossy material using the formula from Table 4-1 in Balanis

    Source: Balanis Advanced Engineering Electromagnetics, 1989, Table 4-1 pg 150

    :param frequency: frequency [Hz]
    :param ep_r_prime: real component of relative complex permittivity [unitless]
    :param sigma: electric conductivity [S/m]
    :param mu_r_prime: real component of relative complex permeability [unitless]
    :return: phase constant (beta) [rad/m]
    """

    # grab constants
    ep_0, mu_0 = free_space_eps_mu()

    # compute omega [rad/s]
    w = 2 * np.pi * frequency

    # compute phase constant
    return w * np.sqrt(mu_r_prime * mu_0 * ep_r_prime * ep_0) * \
           np.sqrt(0.5 * (np.sqrt(1 + (sigma / (w * ep_r_prime * ep_0)) ** 2) + 1))


def attenuation_constant(frequency, ep_r_prime, sigma, mu_r_prime):
    """
    Computes attenuation constant in a lossy material using the formula from Table 4-1 in Balanis

    Source: Balanis Advanced Engineering Electromagnetics, 1989, Table 4-1 pg 150

    :param frequency: frequency [Hz]
    :param ep_r_prime: real component of relative complex permittivity [unitless]
    :param sigma: electric conductivity [S/m]
    :param mu_r_prime: real component of relative complex permeability [unitless]
    :return: phase constant (alpha) [Np/m]
    """

    # grab constants
    ep_0, mu_0 = free_space_eps_mu()

    # compute omega [rad/m]
    w = 2 * np.pi * frequency

    # compute phase constant
    return w * np.sqrt(mu_r_prime * mu_0 * ep_r_prime * ep_0) * \
           np.sqrt(0.5 * (np.sqrt(1 + (sigma / (w * ep_r_prime * ep_0)) ** 2) - 1))


def wave_impedance(frequency, ep_r_prime, sigma=0, mu_r_prime=1):
    """
    Computes attenuation constant in a lossy material using the formula from Table 4-1 in Balanis

    Source: Balanis Advanced Engineering Electromagnetics, 1989, Table 4-1 pg 150

    :param frequency: frequency [Hz]
    :param ep_r_prime: real component of relative complex permittivity [unitless]
    :param sigma: electric conductivity (default 0) [S/m]
    :param mu_r_prime: real component of relative complex permeability (default 1) [unitless]
    :return: wave impedance (Z_w) [Ohm]; only the real component if sigma == 0
    """

    # grab constants
    ep_0, mu_0 = free_space_eps_mu()

    # compute omega [rad/s]
    w = 2 * np.pi * frequency

    # compute wave impedance
    z_w = np.sqrt(1j * w * mu_r_prime * mu_0 / (sigma + 1j * w * ep_r_prime * ep_0))

    # if conductivity = 0 (sigma), return only the real component of z_w (imaginary component = 0)
    if sigma == 0:
        return z_w.real
    else:
        return z_w


def wavelength(frequency, ep_r_prime=1, ep_r_2prime=0, mu_r_prime=1):
    """
    Computes wavelength in a lossy material using the formula from Table 4-1 in Balanis

    Source: Balanis Advanced Engineering Electromagnetics, 1989, Table 4-1 pg 150

    :param frequency: frequency [Hz]
    :param ep_r_prime: real component of relative complex permittivity [unitless]
    :param ep_r_2prime: imaginary component of relative complex permittivity [unitless]
    :param mu_r_prime: real component of relative complex permeability [unitless]
    :return: wavelength in the medium [m]
    """

    # grab constants
    ep_0, mu_0 = free_space_eps_mu()

    # compute omega [rad/s]
    w = 2 * np.pi * frequency

    # compute electric conductivity
    sigma = w * ep_r_2prime * ep_0

    # compute phase constant
    beta = phase_constant(frequency, ep_r_prime, sigma, mu_r_prime)

    # return wavelength
    return 2 * np.pi / beta


def wave_velocity(frequency, ep_r_prime, ep_r_2prime=0, mu_r_prime=1):
    """
    Computes wave propagation velocity in a lossy material using the formula from Table 4-1 in Balanis... Balanis also
    defines this as phase velocity on pg 135.

    Source: Balanis Advanced Engineering Electromagnetics, 1989, Table 4-1 pg 150

    :param frequency: frequency [Hz]
    :param ep_r_prime: real component of relative complex permittivity [unitless]
    :param ep_r_2prime: imaginary component of relative complex permittivity [unitless]
    :param mu_r_prime: real component of relative complex permeability [unitless]
    :return: wave velocity in the medium [m/s]
    """

    # grab constants
    ep_0, mu_0 = free_space_eps_mu()

    # compute omega [rad/s]
    w = 2 * np.pi * frequency

    # compute electric conductivity
    sigma = w * ep_r_2prime * ep_0

    # compute phase constant
    beta = phase_constant(frequency, ep_r_prime, sigma, mu_r_prime)

    # return wavelength
    return w / beta


def skin_depth(frequency, ep_r_prime, sigma, mu_r_prime):
    """
    Computes skin depth in a lossy material using the formula from Table 4-1 in Balanis

    Source: Balanis Advanced Engineering Electromagnetics, 1989, Table 4-1 pg 150

    :param frequency: frequency [Hz]
    :param ep_r_prime: real component of relative complex permittivity [unitless]
    :param sigma: electric conductivity [S/m]
    :param mu_r_prime: real component of relative complex permeability [unitless]
    :return: skin depth [m]
    """

    # compute skin depth
    return 1 / attenuation_constant(frequency, ep_r_prime, sigma, mu_r_prime)


def refractive_index(frequency, ep_r_prime, ep_r_2prime=0, mu_r_prime=1):
    """
    Computes index of refraction as a function of material parameters as it appears in Eq 8.29 of Ulaby

    Source: Ulaby Fundamentals of Applied Electromagnetics, 1997, pg 284

    :param frequency: frequency / Hz
    :param ep_r_prime: real component of relative complex permittivity / unitless
    :param ep_r_2prime: imaginary component of relative complex permittivity / unitless
    :param mu_r_prime: real component of relative complex permeability / unitless
    :return: index of refraction / unitless
    """

    # compute index of refraction
    return constants.speed_of_light / wave_velocity(frequency, ep_r_prime, ep_r_2prime, mu_r_prime)
