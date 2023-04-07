import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cycler import cycler

def enable_ieee_conference_formatting():
    """
    Enables the ieee conference formatting. 
    The plots generated will have Times New Roman fonts set, tex output enabled, a font size of 8, figure size of 3.5 in x 2.5 in and a dpi of 600. 
    A color cycler and linestyle cycler will be setup. 
    Beware this will modify your matplotlib rcParams.
    """
    default_cycler = (cycler(color=['k', 'r', 'b', 'g']) +
                    cycler(linestyle=['-', '--', ':', '-.']))

    plt.rc('lines', linewidth=1)
    plt.rc('axes', prop_cycle=default_cycler)

    plt.rcParams["font.family"] = "Times"
    plt.rcParams["font.size"] = 8

    plt.rcParams['figure.figsize'] = (3.5, 2.5)
    plt.rcParams['figure.dpi'] = 600

    plt.rcParams['text.usetex'] = True

def polar_phi_cut(pattern_object, field_names, frequency, phi, field_labels=None, legend_location_in_deg=80):
    """
    Plot a phi cut of a pattern object on a polar plot

    :param pattern_object: _description_
    :type pattern_object: _type_
    :param field_names: _description_
    :type field_names: _type_
    :param frequency: _description_
    :type frequency: _type_
    :param phi: _description_
    :type phi: _type_
    :param field_labels: _description_, defaults to None
    :type field_labels: _type_, optional
    :param legend_location_in_deg: _description_, defaults to 80
    :type legend_location_in_deg: int, optional
    """
    
    data_array = pattern_object.data_array

    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1) # increasing theta clockwise

    for i, field in enumerate(field_names):
        data_array_cut = data_array.sel(field=field, frequency=frequency, phi=phi)
        theta = data_array_cut.coords['theta'].values*np.pi/180.0
        data =  data_array_cut.value

        if field in pattern_object.FIELDS_WITH_UNITS_DB:
            data = np.real(data)

        if field_labels is None:
            plt.polar(theta, data, label=field)
        else:
            plt.polar(theta, data, label=field_labels[i])

    angle = np.deg2rad(legend_location_in_deg)
    ax.legend(loc="lower left", frameon=False,
            bbox_to_anchor=(.5 + np.cos(angle)/2, .5 + np.sin(angle)/2))

    plt.tight_layout()

def polar_theta_cut(pattern_object, field_names, frequency, theta, field_labels=None, legend_location_in_deg=80):
    """
    Plot a theta cut of a pattern object

    :param pattern_object: _description_
    :type pattern_object: _type_
    :param field_names: _description_
    :type field_names: _type_
    :param frequency: _description_
    :type frequency: _type_
    :param theta: _description_
    :type theta: _type_
    :param field_labels: _description_, defaults to None
    :type field_labels: _type_, optional
    :param legend_location_in_deg: _description_, defaults to 80
    :type legend_location_in_deg: int, optional
    """
    
    data_array = pattern_object.data_array

    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1) # increasing theta clockwise

    for i, field in enumerate(field_names):
        data_array_cut = data_array.sel(field=field, frequency=frequency, theta=theta)
        phi = data_array_cut.coords['phi'].values*np.pi/180.0
        data =  data_array_cut.value

        if field in pattern_object.FIELDS_WITH_UNITS_DB:
            data = np.real(data)

        if field_labels is None:
            plt.polar(phi, data, label=field)
        else:
            plt.polar(phi, data, label=field_labels[i])

    angle = np.deg2rad(legend_location_in_deg)
    ax.legend(loc="lower left", frameon=False,
            bbox_to_anchor=(.5 + np.cos(angle)/2, .5 + np.sin(angle)/2))

    plt.tight_layout()

def rect_phi_cut(pattern_object, field_names, frequency, phi, field_labels=None):
    """Plot rectangular phi cut

    :param pattern_object: _description_
    :type pattern_object: _type_
    :param field_names: _description_
    :type field_names: _type_
    :param frequency: _description_
    :type frequency: _type_
    :param phi: _description_
    :type phi: _type_
    :param field_labels: _description_, defaults to None
    :type field_labels: _type_, optional
    """
    data_array = pattern_object.data_array

    ax = plt.subplot(111)

    units = set()
    for i, field in enumerate(field_names):
        data_array_cut = data_array.sel(field=field, frequency=frequency, phi=phi)
        theta = data_array_cut.coords['theta'].values
        data =  data_array_cut.value

        if field in pattern_object.FIELDS_WITH_UNITS_DB:
            data = np.real(data)

        if field_labels is None:
            plt.plot(theta, data, label=field)
        else:
            plt.plot(theta, data, label=field_labels[i])
        
        units.add(pattern_object.DEFAULT_UNITS[field])

    if len(units) == 1:
        plt.ylabel(list(units)[0])
    plt.xlabel(pattern_object.DEFAULT_UNITS['Theta'])
    plt.legend()
    plt.tight_layout()


def rect_theta_cut(pattern_object, field_names, frequency, theta, field_labels=None):
    """Plot rectangular theta cut

    :param pattern_object: _description_
    :type pattern_object: _type_
    :param field_names: _description_
    :type field_names: _type_
    :param frequency: _description_
    :type frequency: _type_
    :param theta: _description_
    :type theta: _type_
    :param field_labels: _description_, defaults to None
    :type field_labels: _type_, optional
    """
    data_array = pattern_object.data_array

    ax = plt.subplot(111)

    units = set()
    for i, field in enumerate(field_names):
        data_array_cut = data_array.sel(field=field, frequency=frequency, theta=theta)
        phi = data_array_cut.coords['phi'].values*np.pi/180.0
        data =  data_array_cut.value

        if field in pattern_object.FIELDS_WITH_UNITS_DB:
            data = np.real(data)

        if field_labels is None:
            plt.plot(phi, data, label=field)
        else:
            plt.plot(phi, data, label=field_labels[i])
        
        units.add(pattern_object.DEFAULT_UNITS[field])

    if len(units) == 1:
        plt.ylabel(list(units)[0])
    plt.xlabel(pattern_object.DEFAULT_UNITS['Theta'])
    plt.legend()
    plt.tight_layout()