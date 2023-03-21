import numpy as np
import scipy as sp
import scipy.constants
import scipy.integrate
import scipy.special
import pandas as pd


def circular_patch_antenna(
    design_frequency,
    offset_ratio,
    epsilon_r=4.4,
    mu_r=1,
    h=1.57e-3,  # m, based on standard thickness of FR4 PCBs,
    loss_tangent=0.01,
    # S/m from https://en.wikipedia.org/wiki/Electrical_resistivity_and_conductivity)
    conductivity=5.96e7,
):
    """_summary_

    Args:
        design_frequency (_type_): _description_
        offset_ratio (_type_): _description_
        basedonstandardthicknessofFR4PCBs (_type_): _description_
        epsilon_r (float, optional): _description_. Defaults to 4.4.
        mu_r (int, optional): _description_. Defaults to 1.
        h (_type_, optional): _description_. Defaults to 1.57e-3.
        loss_tangent (float, optional): _description_. Defaults to 0.01.
        conductivity (_type_, optional): _description_. Defaults to 5.96e7.

    Returns:
        _type_: _description_
    """

    #########--------------- Only Calculations below this point ---------------#########
    # Calculate the radius based on design equations from Balanis, Antenna Theory and Design, pp 815-832, not accounting for fringing fields
    a_no_fringing = 1.8412 / (2 * np.pi * design_frequency
                              * np.sqrt(scipy.constants.mu_0 * mu_r
                                        * scipy.constants.epsilon_0 * epsilon_r))

    # Calculate the radius based on a first order approximation of the fringing field affects Balanis, Antenna Theory and Design, pp 815-832
    F = 8.7941e9 / (design_frequency * np.sqrt(epsilon_r))
    a_fringing = F / (1 + (2 * h * 100 / (np.pi * epsilon_r * F)
                           * (np.log(np.pi * F / (2 * h * 100)) + 1.7726)))**(1/2) / 100

    # The input impedance of the patch is estimated, from the radius of the patch including fringing: a_fringing
    k_0 = 2 * np.pi * design_frequency * \
        np.sqrt(scipy.constants.epsilon_0 * scipy.constants.mu_0)
    k_FR4 = k_0 * np.sqrt(epsilon_r * mu_r)

    def radiation_conductance_J_02_prime(theta): return scipy.special.jv(0, k_0 * a_fringing * np.sin(theta)) \
        + scipy.special.jv(2, k_0 * a_fringing * np.sin(theta))

    def radiation_conductance_J_02(theta): return scipy.special.jv(0, k_0 * a_fringing * np.sin(theta)) \
        - scipy.special.jv(2, k_0 * a_fringing * np.sin(theta))

    def radiation_conductance_integrand(theta): return (radiation_conductance_J_02_prime(theta)
                                                        + (np.cos(theta)**2)
                                                        * radiation_conductance_J_02(theta)) \
        * np.sin(theta)
    radiation_conductance = ((k_0 * a_fringing)**2 / (480)) \
        * scipy.integrate.quad(radiation_conductance_integrand, 0, np.pi / 2)[0]

    conductor_loss_conductance = np.pi * ((np.pi * scipy.constants.mu_0 * design_frequency) ** (-3/2)) \
        * (((k_FR4 * a_fringing)**2) - 1) \
        / (2 * (h**2) * np.sqrt(conductivity))

    dielectric_conductance = loss_tangent * (((k_FR4 * a_fringing)**2) - 1)\
        / (2 * scipy.constants.mu_0 * h * design_frequency)

    # Estimate the total conductance, and the radiation efficiency
    total_conductance = radiation_conductance + \
        conductor_loss_conductance + dielectric_conductance

    radiation_efficiency = (1 / radiation_conductance) / \
        ((1 / conductor_loss_conductance) + (1 / dielectric_conductance))

    # Estimate the bandwidth using a very questionable design equation from Balanis (what is l?)
    # I think  this definition of bandwidth is gain bandwidth which is not the limiter in Patch Antennas Typically
    bandwidth = h * total_conductance / (2
                                         * (a_fringing * 2)
                                         * epsilon_r * scipy.constants.epsilon_0 * (a_fringing / 2))

    # Calculate the impedance at different center offsets of the feeding probes
    # Define center offset
    r_offset = a_fringing * offset_ratio

    #########--------------- Only Calculations below this point ---------------#########
    real_input_impedance = (1 / total_conductance) * (scipy.special.jv(
        1, k_FR4 * r_offset)**2) / (scipy.special.jv(1, k_FR4 * a_fringing)**2)

    return {
        'a_no_fringing': a_no_fringing,
        'a_fringing': a_fringing,
        'total_conductance': total_conductance,
        'radiation_conductance': radiation_conductance,
        'conductor_loss_conductance': conductor_loss_conductance,
        'dielectric_conductance': dielectric_conductance,
        'radiation_efficiency': radiation_efficiency,
        'bandwidth': bandwidth,
        'r_offset': r_offset,
        'real_input_impedance': real_input_impedance
    }

def circular_patch_antenna_resonant_frequency(
    a,
    h,
    offset_ratio,
    epsilon_r_substrate,
    loss_tangent_substrate,
    mu_r_substrate,
    # S/m from https://en.wikipedia.org/wiki/Electrical_resistivity_and_conductivity)
    conductivity=5.96e7,
):
    """_summary_

    Args:
        a (_type_): _description_
        h (_type_): _description_
        offset_ratio (_type_): _description_
        epsilon_r_substrate (_type_): _description_
        loss_tangent_substrate (_type_): _description_
        mu_r_substrate (_type_): _description_
        conductivity (_type_, optional): _description_. Defaults to 5.96e7.

    Returns:
        _type_: _description_
    """
    
    # The resonant frequency is found 
    epsilon_r_modified = epsilon_r_substrate / 1
    a_e = a * np.sqrt(1 + (np.log(np.pi * a / (2 * h)) + 1.7726)\
         * 2 * h / (np.pi * a * epsilon_r_modified))
    resonant_frequency = ( 1 / a_e ) * 1.8412 / (np.pi * 2 * \
        np.sqrt(scipy.constants.mu_0 * mu_r_substrate
            * scipy.constants.epsilon_0 * epsilon_r_substrate))

    # The input impedance of the patch is estimated, from the radius of the patch including fringing: a
    k_0 = 2 * np.pi * resonant_frequency * \
        np.sqrt(scipy.constants.epsilon_0 * scipy.constants.mu_0)
    k_FR4 = k_0 * np.sqrt(epsilon_r_substrate * mu_r_substrate)

    def radiation_conductance_J_02_prime(theta): return scipy.special.jv(0, k_0 * a * np.sin(theta)) \
        + scipy.special.jv(2, k_0 * a * np.sin(theta))

    def radiation_conductance_J_02(theta): return scipy.special.jv(0, k_0 * a * np.sin(theta)) \
        - scipy.special.jv(2, k_0 * a * np.sin(theta))

    def radiation_conductance_integrand(theta): return (radiation_conductance_J_02_prime(theta)
                                                        + (np.cos(theta)**2)
                                                        * radiation_conductance_J_02(theta)) \
        * np.sin(theta)
    radiation_conductance = ((k_0 * a)**2 / (480)) \
        * scipy.integrate.quad(radiation_conductance_integrand, 0, np.pi / 2)[0]

    conductor_loss_conductance = np.pi * ((np.pi * scipy.constants.mu_0 * resonant_frequency) ** (-3/2)) \
        * (((k_FR4 * a)**2) - 1) \
        / (2 * (h**2) * np.sqrt(conductivity))

    dielectric_conductance = loss_tangent_substrate * (((k_FR4 * a)**2) - 1)\
        / (2 * scipy.constants.mu_0 * h * resonant_frequency)

    # Estimate the total conductance, and the radiation efficiency
    total_conductance = radiation_conductance + \
        conductor_loss_conductance + dielectric_conductance

    radiation_efficiency = (1 / radiation_conductance) / \
        ((1 / conductor_loss_conductance) + (1 / dielectric_conductance))

    # Estimate the bandwidth using a very questionable design equation from Balanis (what is l?)
    # I think  this definition of bandwidth is gain bandwidth which is not the limiter in Patch Antennas Typically
    bandwidth = h * total_conductance / (2 * (a * 2) * \
        epsilon_r_substrate * scipy.constants.epsilon_0 * (a / 2))

    # Calculate the impedance at different center offsets of the feeding probes
    # Define center offset
    r_offset = a * offset_ratio

    #########--------------- Only Calculations below this point ---------------#########
    real_input_impedance = (1 / total_conductance) * (scipy.special.jv(
        1, k_FR4 * r_offset)**2) / (scipy.special.jv(1, k_FR4 * a)**2)

    return {
        'epsilon_r_substrate': epsilon_r_substrate,
        'resonant_frequency': resonant_frequency,
        'total_conductance': total_conductance,
        'radiation_conductance': radiation_conductance,
        'conductor_loss_conductance': conductor_loss_conductance,
        'dielectric_conductance': dielectric_conductance,
        'radiation_efficiency': radiation_efficiency,
        'bandwidth': bandwidth,
        'r_offset': r_offset,
        'real_input_impedance': real_input_impedance
    }