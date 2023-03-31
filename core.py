import xarray as xr
import pandas as pd
import numpy as np
import warnings
import copy
from os.path import splitext
from antenna_toolbox import electromagnetics
from antenna_toolbox import math_funcs
from antenna_toolbox import parse
from antenna_toolbox import constants

def _get_keys_whose_values_contain_string(dictionary, search_string):
    found_keys = []
    for k in dictionary.keys():
        found_keys.append(k) if search_string in dictionary[k] else None
    return found_keys

def _remove_all_list_elements_in_l2_from_l1(l1, l2):
    return [x for x in l1 if x not in l2]

class pattern():
    VALID_FIELD_NAMES = [
        'Etheta',
        'Ephi',
        'ERHCP',
        'ELHCP',
        'Utheta',
        'Uphi',
        'URHCP',
        'ULHCP',
        'Directivity_Theta',
        'Directivity_Phi',
        'Directivity_Total',
        'Directivity_L3X',
        'Directivity_L3Y',
        'Directivity_LHCP',
        'Directivity_RHCP',
        'Gain_Theta',
        'Gain_Phi',
        'Gain_Total',
        'Gain_L3X',
        'Gain_L3Y',
        'Gain_LHCP',
        'Gain_RHCP',
        'Realized_Gain_Theta',
        'Realized_Gain_Phi',
        'Realized_Gain_Total',
        'Realized_Gain_L3X',
        'Realized_Gain_L3Y',
        'Realized_Gain_LHCP',
        'Realized_Gain_RHCP',
        'Xpol_Ratio_Y_to_X',
        'Xpol_Ratio_X_to_Y',
        'Xpol_Ratio_LH_to_RH',
        'Xpol_Ratio_RH_to_LH',
        'Axial_Ratio',
        'Polarization_Angle'
    ]

    DEFAULT_UNITS = {
        'Frequency': 'Hz',
        'Theta': 'deg',
        'Phi': 'deg',
        'Elevation': 'deg',
        'Azimuth': 'deg',
        'Re_Etheta': 'V/m',
        'Im_Etheta': 'V/m',
        'Re_Ephi': 'V/m',
        'Im_Ephi': 'V/m',
        'Re_Htheta': 'A/m',
        'Im_Htheta': 'A/m',
        'Re_Hphi': 'A/m',
        'Im_Hphi': 'A/m',
        'Directivity_Theta': 'dBi',
        'Directivity_Phi': 'dBi',
        'Directivity_Total': 'dBi',
        'Directivity_L3X': 'dBi',
        'Directivity_L3Y': 'dBi',
        'Directivity_LHCP': 'dBic',
        'Directivity_RHCP': 'dBic',
        'Gain_Theta': 'dBi',
        'Gain_Phi': 'dBi',
        'Gain_Total': 'dBi',
        'Gain_L3X': 'dBi',
        'Gain_L3Y': 'dBi',
        'Gain_LHCP': 'dBic',
        'Gain_RHCP': 'dBic',
        'Realized_Gain_Theta': 'dBi',
        'Realized_Gain_Phi': 'dBi',
        'Realized_Gain_Total': 'dBi',
        'Realized_Gain_L3X': 'dBi',
        'Realized_Gain_L3Y': 'dBi',
        'Realized_Gain_LHCP': 'dBic',
        'Realized_Gain_RHCP': 'dBic',
        'Xpol_Ratio_Y_to_X': 'dB',
        'Xpol_Ratio_X_to_Y': 'dB',
        'Xpol_Ratio_LH_to_RH': 'dB',
        'Xpol_Ratio_RH_to_LH': 'dB',
        'Axial_Ratio': 'dB',
        'Polarization_Angle': 'deg'
    }

    FIELDS_WITH_UNITS_DB =  _get_keys_whose_values_contain_string(DEFAULT_UNITS, 'dB')

    REAL_UNITS = ['V/m', 'A/m']
    FIELD_WITH_REAL_UNITS = _get_keys_whose_values_contain_string(DEFAULT_UNITS, 'V/m') \
        + _get_keys_whose_values_contain_string(DEFAULT_UNITS, 'A/m')

    FIELDS_SAFE_FOR_ADD_SUB_MUL_DIV = FIELD_WITH_REAL_UNITS
    FIELDS_UNSAFE_FOR_ADD_SUB_MUL_DIV = _remove_all_list_elements_in_l2_from_l1(DEFAULT_UNITS, FIELDS_SAFE_FOR_ADD_SUB_MUL_DIV)

    DEFAULT_DIMS = ['field', 'frequency', 'theta', 'phi']

    SUPPORTED_FILE_TYPES = ['.ffs', '.ffe', '.nc', '.csv']

    def _get_coords_as_list(self, coord):
        return list(self.data_array.coords[coord].values)

    def _are_fields_safe_to_add_sub_mul_div(self):
        if any(x in self._get_coords_as_list('field') for x in self.FIELDS_UNSAFE_FOR_ADD_SUB_MUL_DIV):
            raise KeyError('Fields unsafe to apply add, subtract, multiply or divide. Try slicing fields to include only ' \
                + str(self.FIELDS_SAFE_FOR_ADD_SUB_MUL_DIV))

    def __init__(
            self,
            data=None, coords=None,
            data_array=None,
            **kwargs
    ):
        """
        Initializer for a pattern object
        
        :param data: field data, Analogous the argument to xarray.DataArray, defaults to None
        :type data: numpy.Array, required
        
        :param coords: Analogous the argument to xarray.DataArray, defaults to None
        :type coords: dictionary of lists, optional
        
        Usage:
            pattern()::
            <xarray.DataArray (field: 0, frequency: 0, theta: 0, phi: 0)>
            array([], shape=(0, 0, 0, 0), dtype=float64)
            Dimensions without coordinates: field, frequency, theta, phi
            
            
            data = np.asarray(::
                [[[::
                    [0, 1, 2, 3],::
                    [4, 5, 6, 7]::
                ]]]::
            )::
            coords = {::
                'field': ['Re_Ephi'],::
                'frequency' : [1e9],::
                'theta' : [0, 90],::
                'phi' : [0, 90, 180, 270]::
            }::

            pattern(data, coords)::
            <xarray.DataArray (field: 1, frequency: 1, theta: 2, phi: 4)>
            array([[[[0, 1, 2, 3],
                    [4, 5, 6, 7]]]])
            Coordinates:
            * field      (field) <U7 'Re_Ephi'
            * frequency  (frequency) float64 1e+09
            * theta      (theta) int32 0 90
            * phi        (phi) int32 0 90 180 270
            
            
            re_ephi_data = np.asarray(::
                [[::
                    [0, 1, 2, 3],::
                    [4, 5, 6, 7]::
                ]]::
            )::
            im_ephi_data = np.asarray(::
                [[::
                    [4, 5, 6, 7],::
                    [0, 1, 2, 3],::
                ]]::
            )::

            pattern(::
                Re_Ephi=re_ephi_data,::
                Im_Ephi=im_ephi_data,::
                frequency=[1e9],::
                theta=[0, 90],::
                phi=[0, 90, 180, 270]::
            )::
            <xarray.DataArray (field: 2, frequency: 1, theta: 2, phi: 4)>
            array([[[[0., 1., 2., 3.],
                    [4., 5., 6., 7.]]],


                [[[0., 1., 2., 3.],
                    [4., 5., 6., 7.]]]])
            Coordinates:
            * frequency  (frequency) float64 1e+09
            * theta      (theta) int32 0 90
            * phi        (phi) int32 0 90 180 270
            * field      (field) <U7 'Re_Ephi' 'Im_Ephi'
        """

        kwarg_coords = dict()
        kwarg_fields = dict()

        for key, value in kwargs.items():
            if _check_field_in_valid_fields(key):
                kwarg_fields[key] = value
            elif key in pattern.DEFAULT_DIMS:
                kwarg_coords[key] = value
            else:
                raise ValueError("Passed kwarg is not a support field or a recognized argument. " + str(key))

        # if kwargs passed with field names, combine all of the field data into one
        if len(kwarg_fields) > 0:
            kwarg_fields_coord = list()
            assumed_data_shape = list(kwarg_fields[list(kwarg_fields.keys())[0]].shape)
            data = np.empty(
                shape=(
                    len(kwarg_fields),
                    *assumed_data_shape
                ))
            i = 0
            for key, value in kwarg_fields.items():
                kwarg_fields_coord.append(key)
                if list(value.shape) != assumed_data_shape:
                    raise ValueError('Could not append field ' + key + \
                                     ' because shape of passed field data does not match shape of other data passed')
                data[i] = value
                i += 1
            kwarg_coords[pattern.DEFAULT_DIMS[0]] = kwarg_fields_coord
            if len(kwarg_coords) > 0 and coords:
                if pattern.DEFAULT_DIMS[0] in coords:
                    raise ValueError('Fields passed as kwargs and fields in coords dimension')
            coords = kwarg_coords

        if not (data is None) and not (coords is None):
            if not type(coords) is dict:
                raise ValueError("Pass coords as dict type. Type of coords was " + type(coords))

            for key, value in coords.items():
                if not key in pattern.DEFAULT_DIMS:
                    raise ValueError("Passed keys in coords but be in the pattern.DEFAULT_DIMS. Passed " \
                                     + str(key) + " but key must be in " + ' '.join(pattern.DEFAULT_DIMS))
                elif key == pattern.DEFAULT_DIMS[0]:
                    # check that the field dimension is a valid field name
                    tf = list(map(
                        _check_field_in_valid_fields,
                        value
                    ))

                    if not all(tf):
                        raise ValueError('Fields in field dimension are not in pattern.VALID_FIELD_NAMES. ' + \
                                         'Invalid values marked False: ' + str(dict(zip(value, tf))))

            self.data_array = xr.DataArray(
                data=data,
                coords=coords,
                dims=pattern.DEFAULT_DIMS
            )
        elif not (data_array is None):
            self.data_array = data_array

            if not set(data_array.coords.keys()) == set(pattern.DEFAULT_DIMS):
                raise ValueError("Passed coordinates names must be in and can only be in the pattern.DEFAULT_DIMS. Passed [" \
                    + ','.join(data_array.coords.keys()) + "] but pattern.DEFAULT_DIMS is [" + ','.join(pattern.DEFAULT_DIMS) + ']')

            tf = list(map(
                _check_field_in_valid_fields,
                data_array.coords[pattern.DEFAULT_DIMS[0]]
            ))

            if not all(tf):
                raise ValueError('Fields in field dimension are not in pattern.VALID_FIELD_NAMES. ' + \
                                    'Invalid values marked False: ' + str(dict(zip(data_array.coords[pattern.DEFAULT_DIMS[0]], tf))))

        elif not (data is None) or not (coords is None):
            raise ValueError("Both data and coords must be passed")
        else:
            self.data_array = xr.DataArray(
                data=np.empty(shape=(0, 0, 0, 0), dtype=np.float64),
                dims=pattern.DEFAULT_DIMS
            )

    ## Magic methods section

    def __str__(self):
        return self.data_array.__str__()

    def __repr__(self):
        return self.data_array.__repr__()

    def concat(self, other, dim):
        """
        Concatenates other to the pattern object along the dimension specified 
        and returns a new pattern object

        :param other: pattern object to concatenate
        :type other: pattern
        :param dim: valid field name to concatenate
        :type dim: str

        :return: 
        :rtype: _type_
        """
        return pattern(data_array=xr.concat([self.data_array, other.data_array], dim=dim))

    def _do_pattern_objects_contain_the_same_fields(self, other):
        if not all(item in self._get_coords_as_list('field') for item in other._get_coords_as_list('field')):
            raise KeyError('Fields of objects must be aligned. Try slicing along fields and then retry the operation.')

    def _are_patterns_safe_to_add_sub_mul_div(self, other):
        self._are_fields_safe_to_add_sub_mul_div()
        other._are_fields_safe_to_add_sub_mul_div()
        # self._do_pattern_objects_contain_the_same_fields(other)

    def is_empty(self):
        return self.data_array.size == 0

    def __add__(self, other):
        """
        Implements addition
        """
        if np.isscalar(other):
            return pattern(data_array=self.data_array + other)
        elif isinstance(other, pattern):
            self._are_patterns_safe_to_add_sub_mul_div(other)
            p = pattern(data_array=self.data_array + other.data_array)
            if p.is_empty():
                raise Warning('Resulting pattern is empty, did you mean to use pattern.concat(other_pattern)?')
            return p
        else:
            raise ValueError('Can only add pattern objects and scalars')

    def __sub__(self, other):
        """
        Implements subtract 
        """
        if np.isscalar(other):
            return pattern(data_array=self.data_array - other)
        elif isinstance(other, pattern):
            self._are_patterns_safe_to_add_sub_mul_div(other)
            p = pattern(data_array=self.data_array - other.data_array)
            if p.is_empty():
                raise Warning('Resulting pattern is empty')
            return p
        else:
            raise ValueError('Can only add pattern objects and scalars')

    def __mul__(self, other):
        """
        Implements multiply
        """
        if np.isscalar(other):
            return pattern(data_array=self.data_array * other)
        elif isinstance(other, pattern):
            self._are_patterns_safe_to_add_sub_mul_div(other)
            p = pattern(data_array=self.data_array * other.data_array)
            if p.is_empty():
                raise Warning('Resulting pattern is empty')
            return p
        else:
            raise ValueError('Can only add pattern objects and scalars')

    def __truediv__(self, other):
        """
        Implements division
        """
        if np.isscalar(other):
            return pattern(data_array=self.data_array / other)
        elif isinstance(other, pattern):
            self._are_patterns_safe_to_add_sub_mul_div(other)
            p = pattern(data_array=self.data_array / other.data_array)
            if p.is_empty():
                raise Warning('Resulting pattern is empty')
            return p
        else:
            raise ValueError('Can only add pattern objects and scalars')

    def _append_field(self, field, field_name):
        """
        Appends a field to full data object after a compute is performed

        :param field: a data array with a single unnamed field
        :type field: xr.data_array
        :param field_name: name of the field
        :type field_name: str
        """
        field.coords['field'] = [field_name]
        self.data_array = xr.concat( [self.data_array, field], dim='field')

    def __getitem__(self, key):
        """
        Implement indexing
        """
        if isinstance(key, str):
            field = [key]
            return pattern(data_array=self.data_array.loc[dict(field=field)])
        elif isinstance(key, slice):
            field = key
            return pattern(data_array=self.data_array.loc[dict(field=field)])
        elif not isinstance(key, list) and not isinstance(key, tuple):
            raise KeyError('Passed indexer is not a field name string, list, tuple or slice')
        elif len(key) == 1:
            field = key
            return pattern(data_array=self.data_array.loc[dict(field=field)])
        elif len(key) == 2:
            field, frequency = key
            if isinstance(field, str):
                field = [field]
            elif not isinstance(field, list) and not isinstance(field, slice):
                raise KeyError('Passed field is not a string, slice or list')
            
            if not isinstance(frequency, list) and not isinstance(frequency, slice):
                frequency = [frequency]

            return pattern(data_array=self.data_array.loc[dict(field=field, frequency=frequency)])
        elif len(key) == 3:
            field, frequency, theta = key

            if isinstance(field, str):
                field = [field]
            elif not isinstance(field, list) and not isinstance(field, slice):
                raise KeyError('Passed field is not a string, slice or list')
            
            if not isinstance(frequency, list) and not isinstance(frequency, slice):
                frequency = [frequency]

            if not isinstance(theta, list) and not isinstance(theta, slice):
                theta = [theta]

            return pattern(data_array=self.data_array.loc[dict(field=field, frequency=frequency, theta=theta)])
        elif len(key) == 4:
            field, frequency, theta, phi = key
            if isinstance(field, str):
                field = [field]
            elif not isinstance(field, list) and not isinstance(field, slice):
                raise KeyError('Passed field is not a string, slice or list')
            
            if not isinstance(frequency, list) and not isinstance(frequency, slice):
                frequency = [frequency]

            if not isinstance(theta, list) and not isinstance(theta, slice):
                theta = [theta]

            if not isinstance(phi, list) and not isinstance(phi, slice):
                phi = [phi]
            return pattern(data_array=self.data_array.loc[dict(field=field, frequency=frequency, theta=theta, phi=phi)])
        else:
            raise KeyError('Invalid indexing')
        
    def __setitem__(self, key, value):
        """
        Implement assignment indexing
        """
        if isinstance(key, str):
            field = [key]
            self.data_array.loc[dict(field=field)] = value.data_array
        elif isinstance(key, slice):
            field = key
            self.data_array.loc[dict(field=field)] = value.data_array
        elif not isinstance(key, list) and not isinstance(key, tuple):
            raise KeyError('Passed indexer is not a field name string, list, tuple or slice')
        elif len(key) == 1:
            field = key
            self.data_array.loc[dict(field=field)] = value.data_array
        elif len(key) == 2:
            field, frequency = key
            if isinstance(field, str):
                field = [field]
            elif not isinstance(field, list) and not isinstance(field, slice):
                raise KeyError('Passed field is not a string, slice or list')
            
            if not isinstance(frequency, list) and not isinstance(frequency, slice):
                frequency = [frequency]

            self.data_array.loc[dict(field=field, frequency=frequency)] = value.data_array
        elif len(key) == 3:
            field, frequency, theta = key

            if isinstance(field, str):
                field = [field]
            elif not isinstance(field, list) and not isinstance(field, slice):
                raise KeyError('Passed field is not a string, slice or list')
            
            if not isinstance(frequency, list) and not isinstance(frequency, slice):
                frequency = [frequency]

            if not isinstance(theta, list) and not isinstance(theta, slice):
                theta = [theta]

            self.data_array.loc[dict(field=field, frequency=frequency, theta=theta)] = value.data_array
        elif len(key) == 4:
            field, frequency, theta, phi = key
            if isinstance(field, str):
                field = [field]
            elif not isinstance(field, list) and not isinstance(field, slice):
                raise KeyError('Passed field is not a string, slice or list')
            
            if not isinstance(frequency, list) and not isinstance(frequency, slice):
                frequency = [frequency]

            if not isinstance(theta, list) and not isinstance(theta, slice):
                theta = [theta]

            if not isinstance(phi, list) and not isinstance(phi, slice):
                phi = [phi]
            self.data_array.loc[dict(field=field, frequency=frequency, theta=theta, phi=phi)] = value.data_array
        else:
            raise KeyError('Invalid indexing')
            
    # Implement of interlibrary interface functions
    def to_numpy(self):
        """Returns numpy array from internal data format

        :return: all field data across all dimensions as one numpy array
        :rtype: numpy array
        """
        return self.data_array.values

    def to_dataframe(self):
        """Returns the pattern object as a multindex dataframe

        :rtype: pandas DataFrame
        """
        return self._to_multindex_dataframe()

    def _to_flat_dataframe(self):
        return self._to_multindex_dataframe().reset_index()

    def _to_multindex_dataframe(self):
        return self.data_array.to_dataframe('value')

    # Implement pattern calculation functions
    def request_field(self, field_str_array):
        """
        Attempts to compute all of the fields listed in the field_str_array in place

        :param field_str_array: valid field names that the user would like to compute
        :type field_str_array: array of strings
        """

        self.compute_ERHCP() if 'ERHCP' in field_str_array else None
        self.compute_ELHCP() if 'ELHCP' in field_str_array else None
        self.compute_Ephi() if 'Ephi' in field_str_array else None
        self.compute_Etheta() if 'Etheta' in field_str_array else None
        
        self.compute_EL3X_from_linear_E() if 'E3X' in field_str_array else None
        self.compute_EL3Y_from_linear_E() if 'E3Y' in field_str_array else None

        # I'm not 100% sure it makes sense to just use thesese functions automatically 
        # since the success of these functions is predicated on other fields existing
        # , but I can't think of a better way to do it
        self.compute_Utheta_from_Etheta() if 'Utheta' in field_str_array else None
        self.compute_Uphi_from_Ephi() if 'Uphi' in field_str_array else None
        self.compute_URHCP_from_ERHCP() if 'URHCP' in field_str_array else None
        self.compute_ULHCP_from_ELHCP() if 'ULHCP' in field_str_array else None
        self.compute_UL3X_from_EL3X() if 'UL3X' in field_str_array else None
        self.compute_UL3Y_from_EL3Y() if 'UL3Y' in field_str_array else None

    def _append_field(self, field, field_name):
        """
        Appends a field to full data object after a compute is performed

        :param field: a data array with a single unnamed field
        :type field: xr.data_array
        :param field_name: name of the field
        :type field_name: str
        """
        field.coords['field'] = [field_name]
        self.data_array = xr.concat( [self.data_array, field], dim='field')

    def compute_ERHCP(self):
        """
        Computes the ERHCP in place from Ephi and Etheta
        """
        temp =  (1/np.sqrt(2)) * (self.data_array.loc[dict(field='Ephi')] \
            - 1j * self.data_array.loc[dict(field='Etheta')])
        self._append_field(temp, 'ERHCP')

    def compute_ELHCP(self):
        """
        Computes the ELHCP in place from Ephi and Etheta
        """
        temp =  (1/np.sqrt(2)) * (self.data_array.loc[dict(field='Ephi')] \
            + 1j * self.data_array.loc[dict(field='Etheta')])
        self._append_field(temp, 'ELHCP')

    def compute_Etheta(self):
        """
        Computes Etheta in place from ERHCP and ELHCP
        """
        temp =  (1j * np.sqrt(2) / 2) * (self.data_array.loc[dict(field='ERHCP')] \
            - self.data_array.loc[dict(field='ELHCP')])
        self._append_field(temp, 'Etheta')

    def compute_Ephi(self):
        """
        Computes Ephi in place from ERHCP and ELHCP
        """
        temp =  (np.sqrt(2) / 2) * (self.data_array.loc[dict(field='ERHCP')] \
            + self.data_array.loc[dict(field='ELHCP')])
        self._append_field(temp, 'Ephi')

    def compute_EL3X_from_linear_E(self):
        """
        Computes L3X fields in place based on Ludwig's 3rd definition [1]

        Sources:
        [1] A. C. Ludwig, “The Definition of Cross Polarization,” IEEE Trans. Antennas Propag., vol. 21, no. 1, pp. 116–119,
        1973, doi: 10.1109/TAP.1973.1140406.
        """
        temp =  self.data_array.loc[dict(field='Etheta')] * np.cos(np.deg2rad(self.data_array.coords['phi'])) \
            - self.data_array.loc[dict(field='Ephi')] * np.sin(np.deg2rad(self.data_array.coords['phi']))
        self._append_field(temp, 'EL3X')

    def compute_EL3Y_from_linear_E(self):
        """
        Computes L3Y fields in place based on Ludwig's 3rd definition [1]

        Sources:
        [1] A. C. Ludwig, “The Definition of Cross Polarization,” IEEE Trans. Antennas Propag., vol. 21, no. 1, pp. 116–119,
        1973, doi: 10.1109/TAP.1973.1140406.
        """
        temp =  self.data_array.loc[dict(field='Etheta')] * np.sin(np.deg2rad(self.data_array.coords['phi'])) \
            + self.data_array.loc[dict(field='Ephi')] * np.cos(np.deg2rad(self.data_array.coords['phi']))
        self._append_field(temp, 'EL3Y')

    def _compute_U_from_E(self, u_field_name, e_field_name):
        """
        Computes U (power) from an E field measurement

        :param u_field_name: _description_
        :type u_field_name: xr.data_array
        :param e_field_name: _description_
        :type e_field_name: xr.data_array
        """
        temp =  (1 / (2 * constants.Z_0)) \
            * (np.abs(self.data_array.loc[dict(field=e_field_name)])**2)
        self._append_field(temp, u_field_name)

    def compute_Utheta_from_Etheta(self):
        """
        Computes Utheta in place from Etheta
        """
        self._compute_U_from_E('Utheta', 'Etheta')

    def compute_Uphi_from_Ephi(self):
        """
        Computes Uphi in place from Ephi
        """
        self._compute_U_from_E('Uphi', 'Ephi')

    def compute_URHCP_from_ERHCP(self):
        """
        Computes URHCP in place from ERHCP
        """
        self._compute_U_from_E('URHCP', 'ERHCP')

    def compute_ULHCP_from_ELHCP(self):
        """
        Computes ULHCP in place from ELHCP
        """
        self._compute_U_from_E('ULHCP', 'ELHCP')

    def compute_UL3X_from_EL3X(self):
        """
        Computes UL3X in place from EL3X
        """
        self._compute_U_from_E('UL3X', 'EL3X')

    def compute_UL3Y_from_EL3Y(self):
        """
        Computes UL3X in place from EL3X
        """
        self._compute_U_from_E('UL3Y', 'EL3Y')

    #TODO compute polarization_angle
    #TODO compute axial ratio
    #TODO compute Xpol ratios
    #TODO directivity from pattern integration

    def find_global_extrema(self, field, coord, extrema_type, **kwargs):
        """
        Finds the maximum of a data field for every coordinate. Saves data as a DataArray that is an attribute of
        pattern.data_array with the following naming convention: <extrema_type>_<field>_vs_<coord>. Ex: Find maximum
        Directivity_L3Y vs frequency, then the data is stored as pattern.data_array.max_Directivity_L3Y_vs_frequency.
        
        Coordinates for the maximum data are saved as <extrema_type>_<field>_vs_<coord>_<remaining_coord_0> and 
        <extrema_type>_<field>_vs_<coord>_<remaining_coord_1>.
        
        All data is saved as a numpy array, which goes along the 'coord' argument.

        Essentially a wrapper around xarray.DataArray.max
        (http://xarray.pydata.org/en/stable/generated/xarray.DataArray.max.html)

        :param field: field in data_array to search for abs max or min in
        :type field: string

        :param coord: coordinate along which abs max or min is searched for (ex: frequency)
        :type coord: string

        :param extrema_type: 'max' or 'min' depending on if max or minimum data is desired
        :type extrema_type: string

        :param **save_name: string to save the name of the calculation as something other than
        '<extrema_type>_<field>_vs_<coord>'
        :type **save_name: string
        """

        # generate save name
        sep = '_'  # I simply did not feel like typing a bunch of underscores and quotes
        vs = 'vs'  # more lazy codeeeeeee
        save_name = extrema_type + sep + field + sep + vs + sep + coord

        # handle kwargs
        for key in kwargs.keys():
            if key == 'save_name':
                save_name = kwargs[key]

        # error handling
        if not _check_field_in_valid_fields(field):
            raise ValueError('field is not in pattern.VALID_FIELD_NAMES.')
        if coord not in self.DEFAULT_DIMS:
            raise ValueError('coord is not in pattern.DEFAULT_DIMS.')
        else:
            if coord == 'field':
                raise ValueError("coord must be 'theta', 'phi', or 'frequency'.")
        if extrema_type != 'max' or extrema_type != 'min':
            raise ValueError("extrema_type is not in 'max' or 'min'.")

        # get coordinates that are NOT the coordinate to search for max/min along
        remaining_coords = ['frequency', 'theta', 'phi']
        remaining_coords.remove(coord)

        # find maximum or minimum vs coord
        arg = None
        if extrema_type == 'max':
            self.data_array.attrs[save_name] = self.data_array.loc[field].max(remaining_coords).to_numpy()
            arg = self.data_array.loc[field].argmax(remaining_coords)
        elif extrema_type == 'min':
            self.data_array.attrs[save_name] = self.data_array.loc[field].min(remaining_coords).to_numpy()
            arg = self.data_array.loc[field].argmin(remaining_coords)
        self.data_array.attrs[save_name + '_' + remaining_coords[0]] = self.data_array.coords[remaining_coords[0]][arg[remaining_coords[0]].to_numpy()].to_numpy()
        self.data_array.attrs[save_name + '_' + remaining_coords[1]] = self.data_array.coords[remaining_coords[1]][arg[remaining_coords[1]].to_numpy()].to_numpy()

    def calc_aperture_efficiency(self, pattern_type, area, **kwargs):
        """
        Computes aperture efficiency for the antenna given an aperture area. Sets
        data_array.attrs.'aperture_efficiency_<pattern_type>' to a numpy array containing aperture efficiency vs
        data_array.frequency (%) and  _data_array.attrs.'aperture_area' to float from argument area (m)

        :param pattern_type: type of pattern, from VALID_FIELD_NAMES, to compute aperture efficiency from... basically
        specifies polarization
        :type pattern_type: str

        :param area: aperture area / m
        :type area: float or int

        :param **beam_peak: location of beam peak in (theta, phi) to use for aperture efficiency calculation
        :type **beam_peak: tuple of floats

        :param **peak_finding: boolean value, when True, searches for beam peak. When False, assume peak of beam is at
        (theta, phi) = (0, 0) and the aperture is in the x-y plane (aperture normal is [0, 0, 1]). Also runs 
        pattern.find_global_extrema(...) and creates stored data for that... also saves projected aperture area as
        _data_array.attrs.'aperture_projected_area'... overrides other kwargs
        :type **peak_finding: bool

        :param **aperture_normal: normal vector to the apertue for reference when computing aperture efficiency of a
        beam that is NOT scanned to boresight/zenith (with respect to the array). Default [0, 0, 1]... np.array([x, y, z])
        / m
        :type **aperture_normal: numpy array of length 3

        :param **save_name: string to save the name of the calculation as something other than 'aperture_efficiency_
        <pattern_type>'
        :type **save_name: string
        
        :param **area_arg_projected: if True, argument area is used to compute aperture efficiency regardless of where beam peak is (default False)
        :type **area_arg_projected: bool
        """

        # handle kwargs
        beam_peak = (0, 0)
        peak_finding = False
        aperture_normal = np.array([0, 0, 1])
        save_name = 'aperture_efficiency_' + pattern_type
        area_arg_projected = False
        for key in kwargs.keys():
            if key == 'beam_peak':
                beam_peak = kwargs[key]
            elif key == 'peak_finding':
                peak_finding = kwargs[key]
            elif key == 'aperture_normal':
                aperture_normal = kwargs[key]
            elif key == 'save_name':
                save_name = kwargs[key]
            elif key == 'area_arg_projected':
                area_arg_projected = kwargs[key]

        # error handling to see if field is an actual antenna pattern
        if _check_field_in_valid_fields(pattern_type):
            if not ('Directivity' in pattern_type or 'Gain' in pattern_type):
                raise ValueError('pattern_type is not a radiation pattern (gain or directivity).')
        else:
            raise ValueError('pattern_type is not in pattern.VALID_FIELD_NAMES.')

        # aperture efficiency function
        def ap_eff(pat_value, area_arg):
            """
            Computes aperture efficiency using internal data from the class
            :param pat_value: value of pattern vs frequency / some sort of dB
            :type pat_value: 1D numpy array

            :param area_arg: projected area of the aperture at beam peak / m^2
            :type projected_area: float or int

            :return: aperture efficiency / %
            :rtype: 1D numpy array
            """
            return math_funcs.db_2_mag(pat_value) / \
                   (4 * np.pi * area_arg / (electromagnetics.wavelength(self.data_array.frequency.values) ** 2)) * 100

        # compute aperture efficiency using peak finding or NOT using peak finding
        # TODO implement angle between aperture and beam peak code
        peak_angle_theta = None
        peak_pattern = None
        if not peak_finding:
            peak_angle_theta = np.ones(len(self.data_array.frequency.to_numpy())) * beam_peak[0]    # set beam peak theta location
            peak_angle_phi = np.ones(len(self.data_array.frequency.to_numpy())) * beam_peak[1]      # set beam peak phi location
            if area_arg_projected:
                projected_area = area
            else:
                projected_area = area * np.cos(np.deg2rad(beam_peak[0]))
        elif peak_finding: 
            self.find_global_extrema(pattern_type, 'frequency', 'max')                              # find beam peak location
            peak_theta_angle_name = 'max_' + pattern_type + '_vs_frequency_theta'                   # name of peak theta angle data
            peak_phi_angle_name = 'max_' + pattern_type + '_vs_frequency_phi'                       # name of peak phi angle data
            peak_angle_theta = self.data_array.attrs[peak_theta_angle_name]                         # get beam peak location in theta
            peak_angle_phi = self.data_array.attrs[peak_phi_angle_name]                             # get beam peak location in phi
            if area_arg_projected:
                projected_area = area
            else:
                projected_area = area * np.cos(np.deg2rad(peak_angle_theta))
        peak_pattern = self.data_array.loc[pattern_type, :, peak_angle_theta, peak_angle_phi].values
        result = ap_eff(peak_pattern, projected_area).flatten()                                     # compute aperture efficiency
        self.data_array.attrs[save_name] = result                                                   # store aperture efficiency
        self.data_array.attrs['aperture_area'] = area                                               # store aperture area
        self.data_array.attrs['aperture_projected_area'] = projected_area                           # store projected area

    # TODO implement calc_phase_center
    def calc_phase_center(self):
        pass
        
    def find_beamwidth(self, pattern_type, bw_setting, plane, **kwargs):
        """
        Computes some amplitude beamwidth (ex: -3 dB BW) of an antenna pattern in a specific plane. Computed at each frequency
        Result is stored as an attribute in self.data_array with the following naming convention:
        'beamwidth_<bw_setting>dB_<pattern_type>_<param_plane[0]>_<param_plane[1]>deg'
        
        :param pattern_type: pattern type to find beamwidth of... from VALID_FIELD_NAMES
        :type pattern_type: string
        
        :param bw_setting: beamwidth, down from beam peak, in dB (ex: -3 dB)
        :type bw_setting: float
        
        :param plane: tuple containing angular coordinate (theta or phi), and its value (deg) for cut that beamwidth is in
        :type plane: tuple (string, float)
        
        :param **save_name: string to save the name of the calculation as something other than 'beamwidth_<bw_setting>_<pattern_type>_<param_plane[0]>_<param_plane[1]>deg'
        :type **save_name: string

        """
        
        # handle kwargs
        save_name = 'beamwidth_' + str(bw_setting).replace('.', 'p') + '_' + pattern_type + '_' + plane[0] + '_' + str(plane[1]).replace('.', 'p')
        for key in kwargs:
            if key == 'save_name':
                save_name = kwargs[key]
        
        # error handling to see if field is an actual antenna pattern
        if _check_field_in_valid_fields(pattern_type):
            if not ('Directivity' in pattern_type or 'Gain' in pattern_type):
                raise ValueError('pattern_type is not a radiation pattern (gain or directivity).')
        else:
            raise ValueError('pattern_type is not in pattern.VALID_FIELD_NAMES.')
        
        # get data in the specified cut, error handling
        data_xr = None
        opposite_coord = None           # used for slicing data later
        if plane[0] == 'theta':
            data_xr = self.data_array.loc[pattern_type, :, plane[1], :]
            opposite_coord = 'phi'
        elif plane[0] == 'phi':
            data_xr = self.data_array.loc[pattern_type, :, :, plane[1]]
            opposite_coord = 'theta'
        else: 
            raise ValueError("First element of argument 'plane' is not either 'theta' or 'phi'.")
        
        # finding beamwidth
        bw_result = []
        for freq in self.data_array.frequency.to_numpy():
            
            # grab data for specified frequency
            pat_coord = data_xr.coords[opposite_coord].values
            pat_amp = data_xr.sel(frequency=freq).values.flatten()
            
            # get maximum of pattern
            max_idx = np.argmax(pat_amp)
            max_amp = pat_amp[max_idx]
            bw_amp = max_amp + bw_setting
            angle_max = pat_coord[max_idx]
            
            # walk to the left of the main beam and find nearest point to BW
            incomplete = True           # haven't found points yet?
            walk_idx = max_idx          # index of stepping nearest to main beam
            left_point_above_idx = np.nan
            left_point_above_mag = np.nan
            left_point_below_idx = np.nan
            left_point_below_mag = np.nan
            while incomplete:

                # grab point at index and the one below it on the pattern
                idx_above = walk_idx
                idx_below = walk_idx - 1
                point_above = pat_amp[idx_above]
                point_below = pat_amp[idx_below]

                # exit while loop if the BW setting point is between the two points, if not, keep going
                if (bw_amp >= pat_amp[idx_below]) and (bw_amp <= pat_amp[idx_above]):
                    incomplete = False
                    left_point_above_idx = idx_above
                    left_point_above_mag = point_above
                    left_point_below_idx = idx_below
                    left_point_below_mag = point_below
                elif walk_idx - 1 == 0:      # exit loop if the last index of pattern is approached
                    warnings.warn('No BW point found on left side of beam; returning NaN.', category=UserWarning)
                    break
                else:               # keep searching... update index...
                    walk_idx -= 1
            
            # walk to the right of the main beam and find nearest point to BW
            incomplete = True  # haven't found points yet?
            walk_idx = max_idx  # index of stepping nearest to main beam
            right_point_above_idx = np.nan
            right_point_above_mag = np.nan
            right_point_below_idx = np.nan
            right_point_below_mag = np.nan
            while incomplete:

                # grab point at index and the one below it on the pattern
                idx_above = walk_idx
                idx_below = walk_idx + 1
                point_above = pat_amp[idx_above]
                point_below = pat_amp[idx_below]

                # exit while loop if the BW setting point is between the two points, if not, keep going
                if (bw_amp >= pat_amp[idx_below]) and (bw_amp <= pat_amp[idx_above]):
                    incomplete = False
                    right_point_above_idx = idx_above
                    right_point_above_mag = point_above
                    right_point_below_idx = idx_below
                    right_point_below_mag = point_below
                elif walk_idx + 1 == (len(pat_amp) - 1):  # exit loop if the last index of pattern is approached
                    warnings.warn('No BW point found on right side of beam; returning NaN.', category=UserWarning)
                    break
                else:  # keep searching... update index...
                    walk_idx += 1
             
            # linear fit function
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
            
            # return case handling if there is a NaN present in the HPBW points
            if np.isnan(left_point_above_idx) or np.isnan(right_point_above_idx) or np.isnan(left_point_below_idx) or np.isnan(right_point_below_idx):
                if (np.isnan(left_point_above_idx) or np.isnan(left_point_below_idx)) and not (np.isnan(right_point_above_idx) and np.isnan(right_point_below_idx)):
                    m_right, b_right = linear_fit(pat_coord[right_point_above_idx], right_point_above_mag,
                                                  pat_coord[right_point_below_idx], right_point_below_mag)
                    angle_right = (bw_amp - b_right) / m_right
                    warnings.warn('Computing HPBW based on right side point.', category=UserWarning)
                    bw_result.append(2*np.abs(angle_max - angle_right))
                elif (np.isnan(right_point_above_idx) or np.isnan(right_point_below_idx)) and not ((np.isnan(left_point_above_idx) and np.isnan(left_point_below_idx))):
                    m_left, b_left = linear_fit(pat_coord[left_point_below_idx], left_point_below_mag,
                                                pat_coord[left_point_above_idx], left_point_above_mag)
                    angle_left = (bw_amp - b_left) / m_left
                    warnings.warn('Computing HPBW based on left side point.', category=UserWarning)
                    bw_result.append(2*np.abs(angle_max - angle_left))
                else:
                    bw_result.append(np.nan)
            
            # compute beamwidth otherwise
            else:
                m_left, b_left = linear_fit(pat_coord[left_point_below_idx], left_point_below_mag,
                                            pat_coord[left_point_above_idx], left_point_above_mag)
                m_right, b_right = linear_fit(pat_coord[right_point_above_idx], right_point_above_mag,
                                              pat_coord[right_point_below_idx], right_point_below_mag)
                angle_left = (bw_amp - b_left) / m_left
                angle_right = (bw_amp - b_right) / m_right
                bw_result.append(np.abs(angle_right - angle_left))
        
        # store beamwidth results
        self.data_array.attrs[save_name] = np.array(bw_result)
              
    def sph_2_array_coordinates(self):
        """
        Covnerts a pattern object with coordinates (0 <= theta <= 180, 0 <= phi <= 360) to
        (-180 <= theta <= 180, 0 <= phi <= 180).

        General algorithm for converting a single (theta, phi) point:
        if phi >= 180:
            phi = phi - 180
            theta = theta * -1

        """

        # TODO debug warning message
        # # warn user if data might already be in array coordinates
        # if self.data_array.phi.values.max <= 180:
        #     warnings.warn('Maximum phi value <= 180 deg, data might already be in array coordinates.', UserWarning)

        # do conversion on coordinates with 180 <= phi <= 360, flip for concatenation, remove -0=theta point
        upper_data = copy.copy(self.data_array.loc[:, :, :, 180:360])           # I'm not sure if this copy is needed
        new_coords = {'theta': upper_data.theta.values * -1,
                      'phi': upper_data.phi.values - 180}
        upper_data = upper_data.assign_coords(new_coords)                       # do conversion
        upper_data = upper_data.sortby('theta')                                 # sort by theta
        upper_data = upper_data.where(upper_data.theta != 0, drop=True)         # remove theta=-0 point

        # concatenate xarrays of 0<=phi<=180 and formally 180<=phi<=360
        self.data_array = xr.concat([upper_data, self.data_array.loc[:, :, :, 0:180]], dim='theta')
    
    def sph_2_az_el(self):
        """
        Converts spherical coordinates to azimuth/elevation instead. Changes 
        theta/phi coords to elevation/azimuth.

        elevation = 90 - theta
        azimuth = phi

        """
        self.data_array['theta'] = 90 - self.data_array['theta']
        self.data_array = self.data_array.rename(
            {'theta': 'elevation', 
            'phi': 'azimuth'})
             

def _check_field_in_valid_fields(field_name):
    return field_name in pattern.VALID_FIELD_NAMES


def supported_file_types():
    return pattern.SUPPORTED_FILE_TYPES


def from_file(file_name, save=False):
    """
    Creates a pattern object from data found in a file. File must be in the format
    of pattern.SUPPORTED_FILE_TYPES.

    :param file_name: name of file to load
    :type file_name: str

    :param save: save pattern.data_array as netcdf (.nc) if True (default False)
    :type save: bool

    :return: pattern object, constructed from data in file
    """
    # grab extension
    root, ext = splitext(file_name)

    # warnings
    if ext not in supported_file_types():
        warnings.warn('File type not supported for parsing. Returning None.', UserWarning)
    
    # parse
    pat = None
    if ext == '.ffs':
        pat = parse.from_ffs(file_name)
    elif ext == '.ffe':
        pat = parse.from_ffe(file_name)
    elif ext == '.nc':
        pat = parse.from_netcdf(file_name)

    # save if requested
    if save == True:
        pat.data_array.to_netcdf(root + '.nc')

    return pat
    