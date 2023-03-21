"""
Contains functions related to aperture antennas
"""

import numpy as np
import math_funcs
import electromagnetics


def rectangular_aperture_fields(f, a, b, theta, phi, **kwargs):
    """
    Computes far-zone electric fields from a rectangular aperture, given by [1]. Electric fields on aperture run
    parallel to the y-axis.

    Sources:
    [1] C. A. Balanis, Antenna Theory: Analysis and Design. Hoboken, NJ, USA: Wiley, 2016. pp. 658-659.

    :param f: frequency / Hz; float
    :param a: length of aperture along the x-axis / m; float
    :param b: length of aperture along the y-axis / m; float
    :param theta: theta angles to compute FF at / deg; array
    :param phi: phi angles to compute FF at / deg; array
    :param **distribution: 'Uniform' or 'TE10'... If TE10, assume a ground plane is present. Default 'Uniform' / string
    :param **ground: True if a ground plane is present, False if not. Default False / bool
    :param **E_0: electric field magnitude. Default 1 V/m / float
    :param **r: radius to compute FF powers at. Default 1 m / float
    :return: theta-pol E-field, phi-pol E-field, theta-pol H-field, phi-pol H-field
    """

    # parse kwargs, set defaults
    distribution = 'Uniform'
    ground = False
    E_0 = 1
    r = 1
    for key in kwargs.keys():
        if key == 'distribution':
            distribution = kwargs[key]
        elif key == 'ground':
            ground = kwargs[key]
        elif key == 'E_0':
            E_0 = kwargs[key]
        elif key == 'r':
            r = kwargs[key]

    # compute abbreviated quantities
    k = 2 * np.pi / electromagnetics.wavelength(f)    # wavenumber
    X = k * a / 2 * math_funcs.sind(theta) * math_funcs.cosd(phi)
    Y = k * b / 2 * math_funcs.sind(theta) * math_funcs.sind(phi)
    C = 1j * (a*b*k*E_0*np.exp(-1j*k*r)) / (2*np.pi*r)
    Z_0 = electromagnetics.wave_impedance(f, 1)

    # compute fields
    e_theta = None
    e_phi = None
    if (distribution == 'Uniform') and (ground == True):  # Uniform distribution aperture on ground plane
        e_theta = C * math_funcs.sind(phi) * math_funcs.sincr(X) * math_funcs.sincr(Y)
        e_phi = C * math_funcs.cosd(theta) * math_funcs.cosd(phi) * math_funcs.sincr(X) * math_funcs.sincr(Y)

    elif (distribution == 'Uniform') and (ground == False):  # Uniform distribution aperture in free-space
        e_theta = C / 2 * math_funcs.sind(phi) * (1 + math_funcs.cosd(theta)) * math_funcs.sincr(X) * math_funcs.sincr(Y)
        e_phi = C / 2 * math_funcs.cosd(phi) * (1 + math_funcs.cosd(theta)) * math_funcs.sincr(X) * math_funcs.sincr(Y)

    elif (distribution == 'TE10'):  # TE10-mode distribution aperture on ground plane
        e_theta = -np.pi / 2 * C * math_funcs.sind(phi) * np.cos(X) / (X ** 2 - (np.pi / 2) ** 2) * math_funcs.sincr(Y)
        e_phi = -np.pi / 2 * C * math_funcs.cosd(theta) * math_funcs.cosd(phi) * np.cos(X) / (X ** 2 - (np.pi / 2) ** 2) * math_funcs.sincr(Y)

    return e_theta, e_phi, -1*e_phi/Z_0, e_theta/Z_0