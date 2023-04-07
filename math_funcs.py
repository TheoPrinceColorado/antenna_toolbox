"""
Contains math functions useful for antenna engineering
"""

import numpy as np


def power_2_db(x):
    """
    Converts a power to dB.

    :param x: complex number
    :return: dB
    """
    return 10 * np.log10(np.abs(x))

def voltage_2_db(x):
    """
    Converts voltage to dB.

    :param x: complex number in linear
    :return: dB
    """
    return power_2_db(x) * 2

def amps_2_db(x):
    """
    Converts a amperage to dB

    :param x: complex number
    :return: dB
    """
    return power_2_db(x) * 2

def db_2_power(x):
    """
    Converts a magnitude in dB to magnitude in power

    :param x: magnitude in dB
    :return: magnitude in power
    """
    return 10**(x/10)

def db_2_voltage(x):
    """
    Converts a magnitude in dB to magnitude in voltage

    :param x: magnitude in dB
    :return: magnitude in voltage
    """
    return 10**(x/20)

def db_2_amperage(x):
    """
    Converts a magnitude in dB to magnitude in amperage

    :param x: magnitude in dB
    :return: magnitude in amperage
    """
    return 10**(x/10)

def sind(theta):
    """
    Sine in degrees... since apparently its not in any python package????

    :param theta: input angle to sine
    :return: sine(theta)
    """
    return np.sin(np.deg2rad(theta))


def cosd(theta):
    """
    Cosine in degrees... since apparently its not in any python package????

    :param theta: input angle to cosine
    :return: cos(theta)
    """
    return np.cos(np.deg2rad(theta))


def cosnd_trunc(theta, alpha, n=1.0, replace=0.0):
    """
    Returns cosine where the negative half of the period is replaced with some number. Useful for modelling ideal beams
    with some HPBW, provided n in cos^n is known.

    :param theta: angles for cosine / deg
    :param alpha: location of peak of cosine / deg
    :param n: power of cosine
    :param replace: value to replace the negative half of the period with, default 0 / unitless
    :return: cosine with the negative half of its period replaced with number replace
    """
    base_cos = cosd(theta - alpha)
    return np.where(base_cos >= 0, base_cos**n, replace)


def tand(theta):
    """
    Tangent in degrees... since apparently its not in any python package????

    :param theta: input angle to tangent
    :return: tan(theta)
    """
    return np.tan(np.deg2rad(theta))


def sincr(theta):
    """
    Computes the UNNORMALIZED sinc function sinc(theta) = sin(theta)/theta with angles in radians.

    :param theta: angle [rad]
    :return: sinc function of said angles [unitless]
    """
    return np.sinc(theta / np.pi)


def sincd(theta):
    """
    Computes the UNNORMALIZED sinc function sinc(theta) = sin(theta)/theta with angles in degrees.

    :param theta: angle array [deg]
    :return: sinc function of said angles [unitless]
    """
    return np.sinc(np.deg2rad(theta) / np.pi)


def thetaphi_2_l3(phi, etheta, ephi):
    """
    Computes L3X and L3Y fields based on Ludwig's 3rd definition [1]

    Sources:
    [1] A. C. Ludwig, “The Definition of Cross Polarization,” IEEE Trans. Antennas Propag., vol. 21, no. 1, pp. 116–119,
     1973, doi: 10.1109/TAP.1973.1140406.

    :param theta: array of theta angles (deg)
    :param phi: array of phi angles (deg)
    :param etheta: array of theta-pol electric fields (complex; V/m)
    :param ephi: array of phi-pol electric fields (complex; V/m)
    :return: L3X and L3Y electric fields, respectively (complex; V/m)
    """
    L3Y = etheta * sind(phi) + ephi * cosd(phi)
    L3X = etheta * cosd(phi) - ephi * sind(phi)
    return L3X, L3Y


def normalize_2_max(array, db=False):
    """
    Normalizes an array to the maximum value (0->1)... or (something->0) in dB
    :param array: array to normalize
    :param db: is the data in dB?
    :return: normalized version of that array
    """
    if db:
        return array - np.max(array)
    else:
        return array/np.max(array)


def v_pearson_r(X, y):
    """
    Returns correlation coefficients of vector y (1 x k) with every row of matrix X (N x k)...
    Vectorized so Python go brrrrrrrrrrrr...

    Source (2/1/2021): https://waterprogramming.wordpress.com/2014/06/13/numpy-vectorized-correlation-coefficient/

    :param X: calibration data (N x k)
    :param y: measured data (1 x k)
    :return: vector of pearson correlation coefficients (N x 1)
    """
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    ym = np.mean(y)
    r_num = np.sum((X - Xm) * (y - ym), axis=1)
    r_den = np.sqrt(np.sum((X - Xm) ** 2, axis=1) * np.sum((y - ym) ** 2))
    r = r_num / r_den
    return r


def rms(x):
    """
    Computes root mean square of 1D data.

    :param x: 1D data
    :return: root mean square of that data
    """
    return np.sqrt(np.sum(x**2)/len(x))


def linear_fit(x1, y1, x2, y2):
    """
    Solves for slope and y-intercept of a line between two points.

    :param x1: x coordinate of point 1
    :param y1: y coordinate of point 1
    :param x2: x coordinate of point 2
    :param y2: y coordinate of point 2
    :return: slope (m) and y-intercept (b) of the line
    """
    m = (y2 - y1)/(x2 - x1)     # compute slope
    b = y1 - m*x1               # compute y-intercept
    return m, b
