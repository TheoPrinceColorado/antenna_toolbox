import numpy as np
import scipy.integrate
import pandas as pd
import warnings
import xarray as xr
from antenna_toolbox import math_funcs
from antenna_toolbox import electromagnetics
from antenna_toolbox import core


def _read_in_file(file):
    """
    For loop that reads in a text file line-by-line.

    :param file: file with extension
    :return: list containing lines
    """
    open_file = open(file, 'r')
    lines = open_file.readlines()
    open_file.close()
    return lines


def _replace_newline_chars(lines):
    """
    Replaces newline characters in a list of strings

    :param lines: list of strings
    :return: list of strings without newline characters
    """
    return [line.replace('\n', '') for line in lines]


def _pattern_integration(re_etheta, im_etheta, re_ephi, im_ephi, theta_res, phi_res, theta_meshgrid):
    """
    Perform integration of fields, over some spherical coordinate system, to get total power from those fields

    :param re_etheta: real component of theta polarized field... 2D nparray that is len(theta) X len(phi) / V/m
    :param im_etheta: imaginary component of theta polarized field... 2D nparray that is len(theta) X len(phi) / V/m
    :param re_ephi: real component of phi polarized field... 2D nparray that is len(theta) X len(phi) / V/m
    :param im_ephi: imaginary component of phi polarized field... 2D nparray that is len(theta) X len(phi) / V/m
    :param theta_res: theta resolution / deg
    :param phi_res: phi resolution / deg
    :param theta_meshgrid: meshgrid of theta points... phi_meshgrid, theta_meshgrid = np.meshgrid(phi, theta)
    :return: total power contained within said fields / W
    """
    Z_0 = electromagnetics.free_space_impedance()
    U_tot = 1 / (2 * Z_0) * (np.abs((re_etheta + 1j*im_etheta))**2 + np.abs((re_ephi + 1j*im_ephi))**2) # W/m^2
    I1 = scipy.integrate.simpson(U_tot * np.sin(np.deg2rad(np.abs(theta_meshgrid))), dx=np.deg2rad(theta_res), axis=1)
    return scipy.integrate.simpson(I1, dx=np.deg2rad(phi_res))      # power / W


def _compute_pattern(re_etheta: float, im_etheta: float, re_ephi: float, im_ephi: float, p: float, phi: float):
    """
    Computes pattern given field quantities and a power p. Uses equation 4*pi*U / p to compute pattern [1]. Pass
    power according to the farfield type you would like (ex: total radiated power for directivity, accepted power
    for IEEE gain, stimulated power for realized gain).

    Sources:
    [1] C. A. Balanis, Antenna Theory: Analysis and Design. Hoboken, NJ, USA: Wiley, 2016. pp 41, 69.

    :param re_etheta: Real component of theta-pol electric field / V/m
    :param im_etheta: Imaginary component of theta-pol electric field /  V/m
    :param re_ephi: Real component of phi-pol electric field /  V/m
    :param im_ephi: Imaginary component of theta-pol electric field /  V/m
    :param p: power / W
    :param phi: phi angle / deg
    :return: tuple containing theta pattern, phi pattern, L3X pattern, L3Y pattern, LHCP pattern, RHCP pattern,
    total pattern, x-pol ratio (L3Y/L3X), x-pol ratio (L3X/L3Y), x-pol ratio (LHCP/RHCP),
    x-pol ratio (RHCP/LHCP), axial ratio (major/minor), and polarization angle (deg from theta vector)...
    (dBi, dBic, dB, or deg)
    """

    # compute impedance of free space
    Z_0 = electromagnetics.free_space_impedance()

    # compute radiated power densities
    e_theta = re_etheta + 1j * im_etheta
    e_phi = re_ephi + 1j * im_ephi
    const = 1 / (2 * Z_0)
    U_theta: float = const * (re_etheta ** 2 + im_etheta ** 2)
    U_phi: float = const * (re_ephi ** 2 + im_ephi ** 2)
    E_L3X, E_L3Y = math_funcs.thetaphi_2_l3(phi, e_theta, e_phi)
    U_L3X: float = const * np.absolute(E_L3X) ** 2
    U_L3Y: float = const * np.absolute(E_L3Y) ** 2
    E_LHCP: float = 1/np.sqrt(2) * (e_phi + 1j*e_theta)     # from FEKO documentation
    E_RHCP: float = 1/np.sqrt(2) * (e_phi - 1j*e_theta)     # from FEKO documentation
    U_LHCP: float = const * np.absolute(E_LHCP) ** 2
    U_RHCP: float = const * np.absolute(E_RHCP) ** 2

    # compute patterns (linear)
    pat_theta: float = 4 * np.pi * U_theta / p
    pat_phi: float = 4 * np.pi * U_phi / p
    pat_L3X: float = 4 * np.pi * U_L3X / p
    pat_L3Y: float = 4 * np.pi * U_L3Y / p
    pat_LHCP: float = 4 * np.pi * U_LHCP / p
    pat_RHCP: float = 4 * np.pi * U_RHCP / p
    pat_total: float = pat_theta + pat_phi

    # compute xpol ratios
    Xpol_YoverX = pat_L3Y / pat_L3X
    Xpol_XoverY = pat_L3X / pat_L3Y
    Xpol_LHoverRH = pat_LHCP / pat_RHCP
    Xpol_RHoverLH = pat_RHCP / pat_LHCP

    # compute axial ratio from Balanis [1]
    # TODO: check correctness of polarization angle
    delta_phi = np.angle(E_L3Y) - np.angle(E_L3X)
    exo = np.abs(E_L3X)
    eyo = np.abs(E_L3Y)
    term1 = exo**2 + eyo**2
    term2 = np.sqrt(exo**4 + eyo**4 + 2*(exo**2)*(eyo**2)*np.cos(2*delta_phi))
    oa = np.sqrt(0.5 * (term1 + term2))
    ob = np.sqrt(0.5 * (term1 - term2))
    AR = oa / ob
    polarization_angle = np.rad2deg(np.pi/2 - 0.5*np.arctan2(2*exo*eyo * np.cos(delta_phi), (exo**2 - eyo**2) * np.cos(delta_phi)))

    # convert to dBi, return
    return (math_funcs.power_2_db(pat_theta),
            math_funcs.power_2_db(pat_phi),
            math_funcs.power_2_db(pat_L3X),
            math_funcs.power_2_db(pat_L3Y),
            math_funcs.power_2_db(pat_LHCP),
            math_funcs.power_2_db(pat_RHCP),
            math_funcs.power_2_db(pat_total),
            math_funcs.power_2_db(Xpol_YoverX),
            math_funcs.power_2_db(Xpol_XoverY),
            math_funcs.power_2_db(Xpol_LHoverRH),
            math_funcs.power_2_db(Xpol_RHoverLH),
            math_funcs.voltage_2_db(AR),
            polarization_angle)


def _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, data):
    """
    Returns pivoted numpy table based on the below StackOverflow solution:
    https://stackoverflow.com/questions/17028329/python-create-a-pivot-table

    :param theta_col_pos: the 2nd output of the line "np.unique(<all thetas>, return_inverse=True)"
    :param phi_col_pos: the 2nd output of the line "np.unique(<all phis>, return_inverse=True)"
    :param theta_samples: number of unique theta points in the dataset
    :param phi_samples: number of unique phi points in the dataset
    :param data: list of data that corresponds to the theta/phis
    :return: 2D numpy array containing the data at each theta/phi... theta = rows, phi = columns
    """
    data_pivot = np.zeros((theta_samples, phi_samples))
    data_pivot[theta_row_pos, phi_col_pos] = data[:]
    return data_pivot


def from_ffe(file):
    """
    Parses a .ffe file (Version 8) from FEKO and populates a pattern object with its data.

    :param file: .ffe file to be parsed WITH path / string
    :return: pattern object with said data
    """
    # Error handling for file type
    file_name_split = file.split('.')
    if file_name_split[-1] != 'ffe':
        raise TypeError('File ' + file + ' is not a .ffs file and cannot be parsed with from_ffe().')

    # load file into RAM
    lines = []
    for line in open(file, 'r'):
        lines.append(line)

    # remove newline characters from lines
    lines = [line.replace('\n', '') for line in lines]

    # grab metadata
    version_number = float(lines[1].split(':')[-1])
    source = lines[2].split(': ')[-1]
    date_time = lines[3].split(': ')[-1]
    simulation_software = 'FEKO'
    simulation_software_version_number = lines[4].split('Version ')[-1]
    configuration_name = lines[7].split(': ')[-1]
    request_name = lines[8].split(': ')[-1]
    theta_samples = int(lines[11].split(': ')[-1])
    phi_samples = int(lines[12].split(': ')[-1])

    # check version number to see if that version is supported, if no, warn the user
    supported_version_numbers = [8.0]
    if version_number not in supported_version_numbers:
        warnings.warn('Version ' + str(version_number) + ' of .ffs files is not currently supported by the parser.' +
                      ' Output may be erroneous.', UserWarning)

    # determine number of frequencies in file based on knowledge of header lines and theta/phi points
    num_samples = theta_samples * phi_samples
    num_header_lines = 10
    num_whitespace_lines = 1  # one whitespace line after data
    number_frequencies = int((len(lines) - 6) / (num_header_lines + num_samples + num_whitespace_lines))

    # initialize frequency, radiation efficiency, power data, and theta/phi
    freqs = np.zeros(number_frequencies)  # frequency / Hz
    rad_efficiency = np.zeros(freqs.shape)  # radiation efficiency / %
    p_r = np.zeros(freqs.shape)  # power radiated / W
    p_a = np.zeros(freqs.shape)  # power accepted / W
    theta = np.zeros(theta_samples)  # for easy initialization of pattern object
    phi = np.zeros(phi_samples)  # ^^^

    # initialize pattern object
    export_pattern = core.pattern(Re_Etheta=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Im_Etheta=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Re_Ephi=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Im_Ephi=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Directivity_Theta=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Directivity_Phi=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Directivity_Total=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Directivity_L3X=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Directivity_L3Y=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Directivity_LHCP=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Directivity_RHCP=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Gain_Theta=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Gain_Phi=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Gain_Total=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Gain_L3X=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Gain_L3Y=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Gain_LHCP=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Gain_RHCP=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Realized_Gain_Theta=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Realized_Gain_Phi=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Realized_Gain_Total=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Realized_Gain_L3X=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Realized_Gain_L3Y=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Realized_Gain_LHCP=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Realized_Gain_RHCP=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Xpol_Ratio_Y_to_X=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Xpol_Ratio_X_to_Y=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Xpol_Ratio_LH_to_RH=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Xpol_Ratio_RH_to_LH=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Axial_Ratio=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Polarization_Angle=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 frequency=freqs,
                                 theta=theta,
                                 phi=phi
                                 )

    # grab data
    first_config_line_idx = 7
    lines_after_config_in_block = num_header_lines + num_samples + num_whitespace_lines
    for ii in range(0, len(freqs)):

        # grab frequency, radiation efficiency
        freqs[ii] = float(lines[first_config_line_idx + 2 + ii * lines_after_config_in_block].split(':')[-1])
        rad_efficiency[ii] = float(lines[first_config_line_idx + 7 + ii * lines_after_config_in_block].split(':')[-1])

        # grab field lines
        # columns of array: 0: theta, 1: phi, 2: Re(E_Theta), 3: Im(E_Theta), 4: Re(E_Phi), 5: Im(E_Phi)
        begin_field_line = first_config_line_idx + num_header_lines + ii * lines_after_config_in_block
        end_field_line = begin_field_line + num_samples - 1
        field_lines = lines[begin_field_line:end_field_line + 1]
        field_lines = [line.split() for line in field_lines]  # split lines by spaces
        fields = np.array(field_lines).astype(float)

        # get get angle stuff, get electric fields, perform pattern integration
        theta_row, theta_row_pos = np.unique(fields[:, 0], return_inverse=True)
        phi_col, phi_col_pos = np.unique(fields[:, 1], return_inverse=True)
        re_etheta = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, fields[:, 2])
        im_etheta = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, fields[:, 3])
        re_ephi = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, fields[:, 4])
        im_ephi = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, fields[:, 5])
        pp, tt = np.meshgrid(phi_col, theta_row)
        theta_res = np.abs(theta_row[1] - theta_row[0])
        phi_res = np.abs(phi_col[1] - phi_col[0])
        p_r[ii] = _pattern_integration(re_etheta, im_etheta, re_ephi, im_ephi, theta_res, phi_res, tt)
        p_a[ii] = p_r[ii] / rad_efficiency[ii]

        # save theta/phi points for first frequency
        if ii == 0:
            export_pattern.data_array.coords['theta'] = theta_row
            export_pattern.data_array.coords['phi'] = phi_col

        # compute directivities and other FF quantities
        d_tuple = _compute_pattern(fields[:, 2], fields[:, 3], fields[:, 4], fields[:, 5], p_r[ii],
                                   fields[:, 0])
        e_theta_re = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, fields[:, 2])
        e_theta_im = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, fields[:, 3])
        e_phi_re = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, fields[:, 4])
        e_phi_im = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, fields[:, 5])
        d_theta = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[0])
        d_phi = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[1])
        d_L3X = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[2])
        d_L3Y = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[3])
        d_LHCP = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[4])
        d_RHCP = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[5])
        d_total = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[6])
        xpol_YoverX = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[7])
        xpol_XoverY = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[8])
        xpol_LHoverRH = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[9])
        xpol_RHoverLH = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[10])
        AR = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[11])
        polarization_angle = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[12])

        # compute IEEE gain from radiation efficiency
        rad_eff_db = math_funcs.power_2_db(rad_efficiency[ii])
        g_theta = d_theta + rad_eff_db
        g_phi = d_phi + rad_eff_db
        g_L3X = d_L3X + rad_eff_db
        g_L3Y = d_L3Y + rad_eff_db
        g_LHCP = d_LHCP + rad_eff_db
        g_RHCP = d_RHCP + rad_eff_db
        g_total = d_total + rad_eff_db

        # save pattern quantities in pattern object
        export_pattern.data_array.loc['Re_Etheta'][ii, :, :] = e_theta_re
        export_pattern.data_array.loc['Im_Etheta'][ii, :, :] = e_theta_im
        export_pattern.data_array.loc['Re_Ephi'][ii, :, :] = e_phi_re
        export_pattern.data_array.loc['Im_Ephi'][ii, :, :] = e_phi_im
        export_pattern.data_array.loc['Directivity_Theta'][ii, :, :] = d_theta
        export_pattern.data_array.loc['Directivity_Theta'][ii, :, :] = d_theta
        export_pattern.data_array.loc['Directivity_Phi'][ii, :, :] = d_phi
        export_pattern.data_array.loc['Directivity_L3X'][ii, :, :] = d_L3X
        export_pattern.data_array.loc['Directivity_L3Y'][ii, :, :] = d_L3Y
        export_pattern.data_array.loc['Directivity_LHCP'][ii, :, :] = d_LHCP
        export_pattern.data_array.loc['Directivity_RHCP'][ii, :, :] = d_RHCP
        export_pattern.data_array.loc['Directivity_Total'][ii, :, :] = d_total
        export_pattern.data_array.loc['Xpol_Ratio_Y_to_X'][ii, :, :] = xpol_YoverX
        export_pattern.data_array.loc['Xpol_Ratio_X_to_Y'][ii, :, :] = xpol_XoverY
        export_pattern.data_array.loc['Xpol_Ratio_LH_to_RH'][ii, :, :] = xpol_LHoverRH
        export_pattern.data_array.loc['Xpol_Ratio_RH_to_LH'][ii, :, :] = xpol_RHoverLH
        export_pattern.data_array.loc['Axial_Ratio'][ii, :, :] = AR
        export_pattern.data_array.loc['Polarization_Angle'][ii, :, :] = polarization_angle
        export_pattern.data_array.loc['Gain_Theta'][ii, :, :] = g_theta
        export_pattern.data_array.loc['Gain_Phi'][ii, :, :] = g_phi
        export_pattern.data_array.loc['Gain_L3X'][ii, :, :] = g_L3X
        export_pattern.data_array.loc['Gain_L3Y'][ii, :, :] = g_L3Y
        export_pattern.data_array.loc['Gain_LHCP'][ii, :, :] = g_LHCP
        export_pattern.data_array.loc['Gain_RHCP'][ii, :, :] = g_RHCP
        export_pattern.data_array.loc['Gain_Total'][ii, :, :] = g_total

    # save metadata to pattern object
    export_pattern.data_array.attrs['file'] = file
    export_pattern.data_array.attrs['file_version_number'] = version_number
    export_pattern.data_array.attrs['simulation_software'] = simulation_software
    export_pattern.data_array.attrs['simulation_software_version_number'] = simulation_software_version_number
    export_pattern.data_array.attrs['date_time'] = date_time
    export_pattern.data_array.attrs['source_name'] = source
    export_pattern.data_array.attrs['configuration_name'] = configuration_name
    export_pattern.data_array.attrs['request_name'] = request_name
    export_pattern.data_array.attrs['power_radiated'] = p_r
    export_pattern.data_array.attrs['power_accepted'] = p_a
    export_pattern.data_array.attrs['radiation_efficiency'] = rad_efficiency * 100  # percent

    return export_pattern


def from_ffs(file):
    """
    Parses a .ffs file from CST Microwave Studio high frequency simulation. Populates a pattern object with its data.

    :param file: .ffs file to be parsed WITH path / string
    :return: pattern object with said data
    """

    # Error handling for file type
    file_name_split = file.split('.')
    if file_name_split[-1] != 'ffs':
        raise TypeError('File ' + file + ' is not a .ffs file and cannot be parsed with from_ffs().')

    # load file into RAM
    lines = []
    for line in open(file, 'r'):
        lines.append(line)

    # remove newline characters from line
    lines = [line.replace('\n', '') for line in lines]

    # grab metadata
    version_number = float(lines[3])  # file version number
    data_type = lines[6]  # type of data
    number_frequencies = int(lines[9])  # number of frequencies present
    position_line = lines[12].split(' ')
    position = (float(position_line[0]), float(position_line[1]), float(position_line[2]))  # FF reference position
    z_axis_line = lines[15].split(' ')
    z_axis = (float(z_axis_line[0]), float(z_axis_line[1]), float(z_axis_line[2]))  # coordinate system z-axis
    x_axis_line = lines[18].split(' ')
    x_axis = (float(x_axis_line[0]), float(x_axis_line[1]), float(x_axis_line[2]))  # coordinate system z-axis

    # check version number to see if that version is supported, if no, warn the user
    supported_version_numbers = [3.0]
    if version_number not in supported_version_numbers:
        warnings.warn('Version ' + str(version_number) + ' of .ffs files is not currently supported by the parser.' +
                      ' Output may be erroneous.', UserWarning)

    # parse frequency/power data
    p_r = np.empty(number_frequencies)  # power radiated / W
    p_a = np.empty(number_frequencies)  # power accepted / W
    p_s = np.empty(number_frequencies)  # power stimulated / W
    freqs = np.empty(number_frequencies)  # frequency / Hz
    power_freq_header_line_number = 20  # line number of '// Radiated/Accepted/Stimulated Power , Frequency' (0 indexed)
    for ff in range(0, number_frequencies):
        beginning_of_block = power_freq_header_line_number + 1 + ff * 5
        p_r[ff] = float(lines[beginning_of_block])
        p_a[ff] = float(lines[beginning_of_block + 1])
        p_s[ff] = float(lines[beginning_of_block + 2])
        freqs[ff] = float(lines[beginning_of_block + 3])

    # grab number of theta points, number of phi points
    theta_phi_line = lines[power_freq_header_line_number + 5 * number_frequencies + 3].split(' ')
    phi_samples = int(theta_phi_line[0])
    theta_samples = int(theta_phi_line[1])
    phi = np.linspace(0, 360, phi_samples)
    theta = np.linspace(0, 180, theta_samples)
    number_angles = theta_samples * phi_samples  # total number of simulated angles in the file

    # compute efficiencies if power stimulated is NOT zero...
    # ...meaning the CST simulation uses a port instead of a field source
    # ... also initialize pattern object with correct fields (with gains or without gains) if gains can be computed
    rad_efficiency = None
    total_efficiency = None
    export_pattern = None
    if np.max(p_s) > 0:

        # compute efficiencies
        total_efficiency = p_r / p_s
        rad_efficiency = p_r / p_a

        # initialize pattern object with GAINZZZZZZZ BRUUUUUUUHHHHHHHH (GET BIIIIIIIIIIIG)
        export_pattern = core.pattern(Re_Etheta=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Im_Etheta=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Re_Ephi=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Im_Ephi=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Directivity_Theta=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Directivity_Phi=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Directivity_Total=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Directivity_L3X=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Directivity_L3Y=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Directivity_LHCP=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Directivity_RHCP=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Gain_Theta=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Gain_Phi=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Gain_Total=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Gain_L3X=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Gain_L3Y=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Gain_LHCP=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Gain_RHCP=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Realized_Gain_Theta=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Realized_Gain_Phi=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Realized_Gain_Total=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Realized_Gain_L3X=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Realized_Gain_L3Y=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Realized_Gain_LHCP=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Realized_Gain_RHCP=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Xpol_Ratio_Y_to_X=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Xpol_Ratio_X_to_Y=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Xpol_Ratio_LH_to_RH=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Xpol_Ratio_RH_to_LH=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Axial_Ratio=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Polarization_Angle=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     frequency=freqs,
                                     theta=theta,
                                     phi=phi
                                     )


    else:
        # initialize pattern object
        export_pattern = core.pattern(Re_Etheta=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Im_Etheta=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Re_Ephi=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Im_Ephi=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Directivity_Theta=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Directivity_Phi=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Directivity_Total=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Directivity_L3X=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Directivity_L3Y=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Directivity_LHCP=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Directivity_RHCP=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Xpol_Ratio_Y_to_X=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Xpol_Ratio_X_to_Y=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Xpol_Ratio_LH_to_RH=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Xpol_Ratio_RH_to_LH=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Axial_Ratio=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     Polarization_Angle=np.empty((number_frequencies, theta_samples, phi_samples)),
                                     frequency=freqs,
                                     theta=theta,
                                     phi=phi
                                     )

    # parse field data
    begin_field_line = None
    end_field_line = None
    for ii in range(0, number_frequencies):

        # compute beginning and end frequency line
        if ii == 0:
            begin_field_line = power_freq_header_line_number + 1 * number_frequencies * 5 + 6
        end_field_line = begin_field_line + number_angles - 1

        # grab fields, convert to number array...
        # Columns of array... 0: phi, 1: theta, 2: Re(E_Theta), 3: Im(E_Theta), 4: Re(E_Phi), 5: Im(E_Phi)
        field_lines = lines[begin_field_line:end_field_line + 1]  # grab lines with data... n_points x 1 array
        field_lines = [line.split() for line in field_lines]  # split lines by spaces... n_points x 6 array
        fields = np.array(field_lines).astype(float)

        # compute fields, directivities, xpol ratios, axial ratio, and polarization angle
        # (2D arrays where theta is on rows, phi on columns, datapoints are patterns)
        theta_row, theta_row_pos = np.unique(fields[:, 1], return_inverse=True)
        phi_col, phi_col_pos = np.unique(fields[:, 0], return_inverse=True)
        d_tuple = _compute_pattern(fields[:, 2], fields[:, 3], fields[:, 4], fields[:, 5], p_r[ii], fields[:, 0])
        e_theta_re = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, fields[:, 2])
        e_theta_im = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, fields[:, 3])
        e_phi_re = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, fields[:, 4])
        e_phi_im = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, fields[:, 5])
        d_theta = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[0])
        d_phi = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[1])
        d_L3X = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[2])
        d_L3Y = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[3])
        d_LHCP = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[4])
        d_RHCP = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[5])
        d_total = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[6])
        xpol_YoverX = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[7])
        xpol_XoverY = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[8])
        xpol_LHoverRH = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[9])
        xpol_RHoverLH = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[10])
        AR = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[11])
        polarization_angle = _pivot_np(theta_row_pos, phi_col_pos, theta_samples, phi_samples, d_tuple[12])

        # save directivities and other quantities in field object
        export_pattern.data_array.loc['Re_Etheta'][ii, :, :] = e_theta_re
        export_pattern.data_array.loc['Im_Etheta'][ii, :, :] = e_theta_im
        export_pattern.data_array.loc['Re_Ephi'][ii, :, :] = e_phi_re
        export_pattern.data_array.loc['Im_Ephi'][ii, :, :] = e_phi_im
        export_pattern.data_array.loc['Directivity_Theta'][ii, :, :] = d_theta
        export_pattern.data_array.loc['Directivity_Theta'][ii, :, :] = d_theta
        export_pattern.data_array.loc['Directivity_Phi'][ii, :, :] = d_phi
        export_pattern.data_array.loc['Directivity_L3X'][ii, :, :] = d_L3X
        export_pattern.data_array.loc['Directivity_L3Y'][ii, :, :] = d_L3Y
        export_pattern.data_array.loc['Directivity_LHCP'][ii, :, :] = d_LHCP
        export_pattern.data_array.loc['Directivity_RHCP'][ii, :, :] = d_RHCP
        export_pattern.data_array.loc['Directivity_Total'][ii, :, :] = d_total
        export_pattern.data_array.loc['Xpol_Ratio_Y_to_X'][ii, :, :] = xpol_YoverX
        export_pattern.data_array.loc['Xpol_Ratio_X_to_Y'][ii, :, :] = xpol_XoverY
        export_pattern.data_array.loc['Xpol_Ratio_LH_to_RH'][ii, :, :] = xpol_LHoverRH
        export_pattern.data_array.loc['Xpol_Ratio_RH_to_LH'][ii, :, :] = xpol_RHoverLH
        export_pattern.data_array.loc['Axial_Ratio'][ii, :, :] = AR
        export_pattern.data_array.loc['Polarization_Angle'][ii, :, :] = polarization_angle

        # compute gains from efficiencies if efficiencies are present (ie CST simulation uses ports, not a field source)
        if np.max(p_s) > 0:
            # compute gains
            rad_eff_db = math_funcs.power_2_db(rad_efficiency[ii])  # convert efficiencies to dB
            total_eff_db = math_funcs.power_2_db(total_efficiency[ii])
            g_theta = d_theta + rad_eff_db  # IEEE Gain
            g_phi = d_phi + rad_eff_db
            g_L3X = d_L3X + rad_eff_db
            g_L3Y = d_L3Y + rad_eff_db
            g_LHCP = d_LHCP + rad_eff_db
            g_RHCP = d_RHCP + rad_eff_db
            g_total = d_total + rad_eff_db
            gr_theta = d_theta + total_eff_db  # Realized Gain
            gr_phi = d_phi + total_eff_db
            gr_L3X = d_L3X + total_eff_db
            gr_L3Y = d_L3Y + total_eff_db
            gr_LHCP = d_LHCP + total_eff_db
            gr_RHCP = d_RHCP + total_eff_db
            gr_total = d_total + total_eff_db

            # save in pattern object
            export_pattern.data_array.loc['Gain_Theta'][ii, :, :] = g_theta
            export_pattern.data_array.loc['Gain_Phi'][ii, :, :] = g_phi
            export_pattern.data_array.loc['Gain_L3X'][ii, :, :] = g_L3X
            export_pattern.data_array.loc['Gain_L3Y'][ii, :, :] = g_L3Y
            export_pattern.data_array.loc['Gain_LHCP'][ii, :, :] = g_LHCP
            export_pattern.data_array.loc['Gain_RHCP'][ii, :, :] = g_RHCP
            export_pattern.data_array.loc['Gain_Total'][ii, :, :] = g_total
            export_pattern.data_array.loc['Realized_Gain_Theta'][ii, :, :] = gr_theta
            export_pattern.data_array.loc['Realized_Gain_Phi'][ii, :, :] = gr_phi
            export_pattern.data_array.loc['Realized_Gain_L3X'][ii, :, :] = gr_L3X
            export_pattern.data_array.loc['Realized_Gain_L3Y'][ii, :, :] = gr_L3Y
            export_pattern.data_array.loc['Realized_Gain_LHCP'][ii, :, :] = gr_LHCP
            export_pattern.data_array.loc['Realized_Gain_RHCP'][ii, :, :] = gr_RHCP
            export_pattern.data_array.loc['Realized_Gain_Total'][ii, :, :] = gr_total

        # update beginning of field line for next block
        begin_field_line = end_field_line + 6

    # save metadata to pattern object
    export_pattern.data_array.attrs['file'] = file
    export_pattern.data_array.attrs['file_version_number'] = version_number
    export_pattern.data_array.attrs['simulation_software'] = 'CST'
    export_pattern.data_array.attrs['reference_position'] = position
    export_pattern.data_array.attrs['z_axis'] = z_axis
    export_pattern.data_array.attrs['x-axis'] = x_axis
    export_pattern.data_array.attrs['power_radiated'] = p_r
    if np.max(p_a) > 0:                 # store accepted power of simulation domain if it is present
        export_pattern.data_array.attrs['power_accepted'] = p_a            # Ex: PW with no port excitation
    if np.max(p_s) > 0:                 # store efficiencies and stimulated power if present
        export_pattern.data_array.attrs['power_stimulated'] = p_s
        export_pattern.data_array.attrs['radiation_efficiency'] = rad_efficiency * 100         # percent
        export_pattern.data_array.attrs['total_efficiency'] = total_efficiency * 100           # percent

    return export_pattern


# TODO update docstring with file format 
def from_txt(file):
    """
    Parse a .txt file.

    :param file: file to be parsed
    :return: pattern object with the data of that file
    """

    # Error handling for file type
    file_name_split = file.split('.')
    if file_name_split[-1] != 'txt':
        raise TypeError('File ' + file + ' is not a .txt file and cannot be parsed with from_argnsi().')

    # bring file in as a DataFrame
    raw_data = pd.read_csv(file, delimiter='\s+')

    # grab unique frequency/theta/phi samples, correct for NSI putting a tiny number (ex: 1.98952e-13) in for phi=0
    freqs = pd.unique(raw_data['#'])
    number_frequencies = len(freqs)
    theta = pd.unique(raw_data['Theta'])
    theta_res = np.abs(theta[1] - theta[0])
    theta_samples = len(theta)
    phi = pd.unique(raw_data['Phi'])
    phi = np.where(phi > 1e-3, phi, 0)
    phi_res = np.abs(phi[1] - phi[0])
    phi_samples = len(phi)

    # create meshgrids of theta/phi for pattern integration later
    pp, tt = np.meshgrid(phi, theta)

    # generate columns with fields
    temp_theta_fields = 10 ** (raw_data['dB(Etheta)'] / 20) * np.exp(1j * np.deg2rad(raw_data['Phase(Etheta)']))
    temp_phi_fields = 10 ** (raw_data['dB(Ephi)'] / 20) * np.exp(1j * np.deg2rad(raw_data['Phase(Ephi)']))
    raw_data['Re_Etheta'] = np.real(temp_theta_fields)
    raw_data['Im_Etheta'] = np.imag(temp_theta_fields)
    raw_data['Re_Ephi'] = np.real(temp_phi_fields)
    raw_data['Im_Ephi'] = np.imag(temp_phi_fields)

    # initialize pattern object with directivities, since this is what we will get from measurement... Frequency in Hz
    export_pattern = core.pattern(Re_Etheta=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Im_Etheta=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Re_Ephi=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Im_Ephi=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Directivity_Theta=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Directivity_Phi=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Directivity_Total=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Directivity_L3X=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Directivity_L3Y=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Directivity_LHCP=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Directivity_RHCP=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Xpol_Ratio_Y_to_X=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Xpol_Ratio_X_to_Y=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Xpol_Ratio_LH_to_RH=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Xpol_Ratio_RH_to_LH=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Axial_Ratio=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 Polarization_Angle=np.empty((number_frequencies, theta_samples, phi_samples)),
                                 frequency=freqs * 1e9,
                                 theta=theta,
                                 phi=phi
                                 )

    # initialize power radiated
    p_r = np.empty(number_frequencies)

    # frequency loop
    theta_loop = None
    phi_loop = None
    for ii in range(0, number_frequencies):

        # get data at specific frequency, get fields
        temp_df = raw_data[raw_data['#'] == freqs[ii]]

        # grab theta/phi on the first frequency loop, since all frequencies should have the same theta/phi
        if ii == 0:
            theta_loop = temp_df['Theta'].to_numpy()
            phi_loop = temp_df['Phi'].to_numpy()
            phi_loop = np.where(phi_loop > 1e-3, phi_loop, 0)  # fix dumb NSI 2000 bug where phi=0 -> small number
        else:
            pass

        # make table of theta/phi/fields based on inner/outer loop
        theta_row, theta_row_pos = np.unique(theta_loop, return_inverse=True)
        phi_row, phi_row_pos = np.unique(phi_loop, return_inverse=True)
        re_etheta = _pivot_np(theta_row_pos, phi_row_pos, theta_samples, phi_samples,
                                        temp_df['Re_Etheta'].to_numpy())
        im_etheta = _pivot_np(theta_row_pos, phi_row_pos, theta_samples, phi_samples,
                                        temp_df['Im_Etheta'].to_numpy())
        re_ephi = _pivot_np(theta_row_pos, phi_row_pos, theta_samples, phi_samples,
                                      temp_df['Re_Ephi'].to_numpy())
        im_ephi = _pivot_np(theta_row_pos, phi_row_pos, theta_samples, phi_samples,
                                      temp_df['Im_Ephi'].to_numpy())

        # pattern integrate fields to get power radiated, compute directivity/other params
        p_r[ii] = _pattern_integration(re_etheta, im_etheta, re_ephi, im_ephi, theta_res, phi_res, tt)
        d_tuple = _compute_pattern(re_etheta, im_etheta, re_ephi, im_ephi, p_r[ii], pp)
        d_theta = d_tuple[0]
        d_phi = d_tuple[1]
        d_L3X = d_tuple[2]
        d_L3Y = d_tuple[3]
        d_LHCP = d_tuple[4]
        d_RHCP = d_tuple[5]
        d_total = d_tuple[6]
        xpol_YoverX = d_tuple[7]
        xpol_XoverY = d_tuple[8]
        xpol_LHoverRH = d_tuple[9]
        xpol_RHoverLH = d_tuple[10]
        AR = d_tuple[11]
        polarization_angle = d_tuple[12]

        # save data
        export_pattern.data_array.loc['Re_Etheta'][ii, :, :] = re_etheta
        export_pattern.data_array.loc['Im_Etheta'][ii, :, :] = im_etheta
        export_pattern.data_array.loc['Re_Ephi'][ii, :, :] = re_ephi
        export_pattern.data_array.loc['Im_Ephi'][ii, :, :] = im_ephi
        export_pattern.data_array.loc['Directivity_Theta'][ii, :, :] = d_theta
        export_pattern.data_array.loc['Directivity_Theta'][ii, :, :] = d_theta
        export_pattern.data_array.loc['Directivity_Phi'][ii, :, :] = d_phi
        export_pattern.data_array.loc['Directivity_L3X'][ii, :, :] = d_L3X
        export_pattern.data_array.loc['Directivity_L3Y'][ii, :, :] = d_L3Y
        export_pattern.data_array.loc['Directivity_LHCP'][ii, :, :] = d_LHCP
        export_pattern.data_array.loc['Directivity_RHCP'][ii, :, :] = d_RHCP
        export_pattern.data_array.loc['Directivity_Total'][ii, :, :] = d_RHCP
        export_pattern.data_array.loc['Xpol_Ratio_Y_to_X'][ii, :, :] = xpol_YoverX
        export_pattern.data_array.loc['Xpol_Ratio_X_to_Y'][ii, :, :] = xpol_XoverY
        export_pattern.data_array.loc['Xpol_Ratio_LH_to_RH'][ii, :, :] = xpol_LHoverRH
        export_pattern.data_array.loc['Xpol_Ratio_RH_to_LH'][ii, :, :] = xpol_RHoverLH
        export_pattern.data_array.loc['Axial_Ratio'][ii, :, :] = AR
        export_pattern.data_array.loc['Polarization_Angle'][ii, :, :] = polarization_angle

    # save metadata to pattern object
    export_pattern.data_array.attrs['file'] = file
    export_pattern.data_array.attrs['measurement_software'] = 'NSI2000 - Aman script'
    export_pattern.data_array.attrs['power_radiated'] = p_r

    return export_pattern

    
def from_netcdf(file):
    """
    Imports a pattern object from a netcdf4 file (.nc)
    
    :param file: file to read in
    :type file: string
    
    :return: pattern object with data contained in nc file
    
    """
    export_pattern = core.pattern()
    dr = xr.open_dataarray(file)
    export_pattern.data_array = dr
    dr.close()
    return export_pattern
    
def read_csv(file_name, data_dict, coord_dict=['field', 'frequency', 'theta', 'phi']):
    """
    Reads in a CSV file and returns a pattern object. The implementation uses the pandas read_csv. 
    One needs to rename the columns in the csv file to match either a default coordinate name or a field name.
    The renaming can be done at runtime by using the data_dict and coord_dict dictionaries to rename the columns. ::

    #TODO Implement coord dict so that if and pattern.DEFAULT_DIMS is not specified, an empty column is added ::

    :param file_name: Name of the file to load
    :type file_name: str

    :param data_dict: Dictionary mapping column names in the file (keys) to field names (values)
    :type data_dict: dict

    :param coord_dict: Dictionary mapping column names in the file (keys) to coordinate names, defaults to pattern.DEFAULT_DIMS
    :type coord_dict: dict, optional

    :return: Pattern object
    :rtype: pattern

    Usage:
        import pandas as pd::
        df = pd.read_csv(file_name)::

        df::
        	d [mm]	Freq [GHz]	Phi [deg]	Theta [deg]	dB(DirLHCP) []	dB(DirRHCP) []::
        0	0.508	0.425	-180	0	-29.980465	7.353579::
        1	0.508	0.425	-180	1	-29.381891	7.353245::
        2	0.508	0.425	-180	2	-28.727721	7.347768::
        3	0.508	0.425	-180	3	-28.037381	7.337168::
        4	0.508	0.425	-180	4	-27.326933	7.321464::
        ...	...	...	...	...	...	...::
        356	0.508	0.425	-180	356	-31.402653	7.303198::
        357	0.508	0.425	-180	357	-31.228061	7.323577::
        358	0.508	0.425	-180	358	-30.923778	7.338759::
        359	0.508	0.425	-180	359	-30.501703	7.348756::
        360	0.508	0.425	-180	360	-29.980465	7.353579::

        coord_dict = {'Freq [GHz]':'frequency', 'Phi [deg]': 'phi', 'Theta [deg]':'theta',}::
        data_dict = {'dB(DirLHCP) []':'Directivity_LHCP', 'dB(DirRHCP) []':'Directivity_RHCP'}::

        antenna_toolbox.from_csv(file_name, data_dict, coord_dict)

        <xarray.Dataset>::
        Dimensions:    (field: 2, frequency: 1, phi: 1, theta: 361)::
        Coordinates::
        * field      (field) object 'Directivity_LHCP' 'Directivity_RHCP'::
        * frequency  (frequency) float64 0.425::
        * phi        (phi) int64 -180::
        * theta      (theta) int64 0 1 2 3 4 5 6 7 ... 353 354 355 356 357 358 359 360::
        Data variables:::
            value      (field, frequency, phi, theta) float64 -29.98 -29.38 ... 7.354::
    """
    df = pd.read_csv(file_name)

    df = df.rename(coord_dict, axis='columns')

    df = df.rename(data_dict, axis='columns')
    
    field_names = data_dict.keys()

    field_names = list(data_dict.values())
    other_coord_names = list(coord_dict.values())

    df = df[field_names + other_coord_names].copy()
    df = df.melt(id_vars=other_coord_names, value_vars=field_names, var_name='field')

    df = df.set_index(['field'] + other_coord_names)

    data_array = df.to_xarray()

    return core.pattern(data_array=data_array)