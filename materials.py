"""
Contains materials characterization functions
"""

import numpy as np
import skrf
import pandas as pd
import electromagnetics
import scipy.optimize


def nrw_method_complex_eps_mu(network_data, sample_length, lambda_cutoff=np.inf, **kwargs):
    """
    Implements the NRW method for dielectric characterization using a transmission line with s-parameter measurements.

    W. B. Weir, "Automatic measurement of complex dielectric constant and permeability at microwave frequencies," in
    Proceedings of the IEEE, vol. 62, no. 1, pp. 33-36, Jan. 1974

    :param network_data: scikit-rf network object with 2-ports
    :param sample_length: thickness of the sample
    :param lambda_cutoff: cutoff wavelength of the transmission line (infinite for TEM lines; default np.inf)
    :param **renorm_impedance: renormalize the impedance of the s-parameter file to this value not None (default None)
    :param **symmetric: S-parameters symmetric? If yes, average return and through parameters to increase SNR of
    measurement (default True)
    :return: DataFrame with columns 'Frequency', 'Eps_r_prime', 'Eps_r_2prime', 'Tan_d_e', 'Mu_r_prime', 'Mu_r_2prime',
    'Tan_d_m', representing the components of complex permittivity and permeability vs frequency. If more than 2 ports,
    the names of the columns are appended with the port notation in the same manner as s-parameters
    (Ex: 'Eps_r_prime_2_1').
    """

    # error handling codes
    if type(network_data) != skrf.Network:
        raise TypeError('Argument <network_data> must be of type skrf.Network.')
    else:
        if network_data.number_of_ports < 2:
            raise TypeError('Argument <network_data> must have >= 2 ports.')

    # parse kwargs, set defaults
    renorm_impedance = None
    symmetric = True
    for key in kwargs.keys():
        if key == 'renorm_impedance':
            renorm_impedance = kwargs[key]
        elif key == 'symmetric':
            symmetric = kwargs[key]

    # renormalize impedance if renorm_impedance is nonzero
    if renorm_impedance is not None:
        network_data.renormalize(renorm_impedance)

    # grab frequency data
    f = network_data.f

    # function that implements NRW chain of equations
    def nrw_eqs(s_ref, s_thru):

        # eq 6
        chi = (s_ref ** 2 - s_thru ** 2 + 1) / (2 * s_ref)

        # eq 5
        gamma = chi + np.sqrt(chi ** 2 - 1)

        # some magical error handling code that Carlos wrote in Python 2, not sure if it's relevant here in Python 3 lol
        try:
            if any(np.absolute(gamma) > 1):
                gamma = chi - np.sqrt(chi ** 2 - 1)
        except TypeError:
            if np.abs(np.absolute(gamma)) > 1:
                gamma = chi - np.sqrt(chi ** 2 - 1)

        # eq 7
        P = (s_ref + s_thru - gamma) / (1 - (s_ref + s_thru) * gamma)

        # compute free space wavelength
        lambda_free = electromagnetics.wavelength(f)

        # intermediate step for the computation (eps*mu = ...)
        eps_mu = -(lambda_free / (2 * np.pi * sample_length) * np.log(P)) ** 2 + (lambda_free / lambda_cutoff) ** 2

        # intermediate step to compute permeability
        mu_2 = (((1 + gamma) / (1 - gamma)) ** 2) * \
               (eps_mu - (lambda_free / lambda_cutoff) ** 2) / (1 - (lambda_free / lambda_cutoff) ** 2)

        # compute complex permeability, permittivity
        mu = np.sqrt(mu_2)
        eps = eps_mu / mu
        return eps, mu


    # symmetric material processing
    if symmetric:

        if network_data.number_of_ports == 2:

            # grab s-parameters, then average through and reflect parameters to increase SNR
            s11 = network_data.s[:, 0, 0]
            s21 = network_data.s[:, 1, 0]
            s12 = network_data.s[:, 0, 1]
            s22 = network_data.s[:, 1, 1]
            s_ref = (s11 + s22) / 2
            s_thru = (s21 + s12) / 2

            # compute epsilon/mu then make into dictionary to convert into DataFrame to return
            eps, mu = nrw_eqs(s_ref, s_thru)
            data_dict = {
                'Frequency': f,
                'Eps_r_prime': eps.real,
                'Eps_r_2prime': eps.imag,
                'Tan_d_e': np.abs(eps.imag / eps.real),
                'Mu_r_prime': mu.real,
                'Mu_r_2prime': mu.imag,
                'Tan_d_m': np.abs(mu.imag / mu.real)
            }
            return pd.DataFrame(data_dict)

        # TODO implement symmetric case with > two ports
        elif network_data.number_of_ports > 2:
            print('Implement the symmetric case with > 2 ports')

    # TODO implement asymmetric case
    elif symmetric == False:
        print('Code the asymmetric case please :)')


# TODO implement np.vectorize to handle list of eps_i, eps_h, p, etc? Might be cool
def solve_abg(eps_i, eps_h, p, eps_range):
    """
    Solves the A-BG formula for effective materials from [1] using a scalar optimizer.

    :param eps_i: inclusion material relative permittivity / unitless
    :param eps_h: host material relative permittivity / unitless
    :param p: fill factor of inclusion material with respect to unit cell / unitless
    :param eps_range: list containing the low and high rel. permittivities for the effective material, respectively
    :return: effective relative permittivity of the material with some host material, inclusion material, and fill factor

    Sources:
    [1] W. M. Merrill, R. E. Diaz, M. M. LoRe, M. C. Squires, and N. G. Alexopoulos, “Effective medium theories for
    artificial materials composed of multiple sizes of spherical inclusions in a host continuum,” IEEE Trans. Antennas
    Propag., vol. 47, no. 1, pp. 142–148, Jan. 1999.
    """

    # goal function, which is the A-BG formula, for solving eps_eff
    def goal_function(eps_eff):
        """
        A-BG formula from [1]
        :param eps_eff: effective permittivity
        :return: LHS - RHS of A-BG formula... minimize the goal function to solve
        """
        return np.abs((eps_i - eps_eff) / (eps_i - eps_h) - (1 - p) * (eps_eff/eps_h)**(1/3))

    # solve equation and return effective permittivity
    result = scipy.optimize.minimize_scalar(goal_function,
                                            bounds=eps_range,
                                            method='Bounded',
                                            options={'maxiter': 1000})
    return result.x