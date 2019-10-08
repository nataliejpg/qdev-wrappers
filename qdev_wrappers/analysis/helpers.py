import numpy as np
import json
from qcodes.dataset.data_export import load_by_id
from qcodes.dataset.data_set import DataSet


def make_json_metadata(datasets, analysers, mappings, setpoints):
    analysis_metadata = []
    dataset_metadata = []
    for a in analysers:
        analysis_metadata.append(a.metadata)
    for d in datasets:
        dataset_metadata.append({'run_id': dataset.run_id,
                                 'exp_id': dataset.exp_id,
                                 'exp_name': dataset.exp_name,
                                 'sample_name': dataset.sample_name})
    metadata = {'analysers': analyser_metadata,
                'inferred_from': dataset_metadata,
                'parameter_mapping': mappings,
                'setpoints': setpoints}
    return json.dumps(metadata)


def load_json_metadata(dataset):
    try:
        return json.loads(dataset.metadata['analysis_metadata'])
    except KeyError:
        raise RuntimeError(
            "'analysis_metadata' not found in dataset metadata, are you sure "
            "this is a fitted dataset?")


def get_experiment_data(dataset, parm_names, mappings):
    data = {}
    d = dataset.get_parameter_data()
    for name in parm_names:
        mapped_name = mappings.get(name, name)
        data.update(d[mapped_name])
    return data


def refine_data_dict(data, setpoints, mappings):
    setpoint_vals = {}
    for s_name, s_val in setpoints.items():
        mapped_name = mappings.get(s_name, s_name)
        arr = data[mapped_name]
        idx = []
        nearest_vals = []
        for v in np.array([s_val]).flatten():
            idx = np.argmin(np.abs(arr - v))
            nearest_vals.append(arr[idx])
            setpoint_vals[mapped_name] = nearest_vals
    res = {}
    for p_name, p_arr in data.items():
        if p_name not in setpoint_vals:
            indices = []
            for s_name, s_val in setpoint_vals.items():
                s_arr = data[s_name]
                indices.append(set(np.argwhere(s_arr == s_val).flatten()))
            if len(indices) > 0:
                intersection = list(set.intersection(*indices))
            else:
                intersection = None
            if len(np.array([p_arr]).flatten()) > 1:
                res[p_name] = p_arr[intersection].flatten()
            else:
                res[p_name] = np.array([p_arr]).flatten()
    res.update(setpoint_vals)
    return res


def scrape_data_dict(data, parm_names=None, cutoff=None):
    res = {}
    for k, v in data.items():
        if cutoff is not None:
            v = np.array([v]).flatten()[:cutoff]
        if (parm_names is None) or (k in parm_names):
            res[k] = np.array([v]).flatten()
    return res


def map_data_dict(data, mappings):
    inverse_mappings = {v: k for k, v in mappings.items()}
    res = {}
    for k, v in data.items():
        n = inverse_mappings.get(k, k)
        res[n] = np.array([v]).flatten()
    return res


def organize_experiment_data(dataset, analyser, setpoints, mappings,
                             apply_map=True, scrape=True):
    meas_names = list(analyser.metadata['measurement_parameters'].keys())
    exp_names = list(analyser.metadata['experiment_parameters'].keys())
    data = get_experiment_data(dataset, meas_names, mappings)
    data = refine_data_dict(data, setpoints, mappings)
    if apply_map:
        data = map_data_dict(data, mappings)
    if scrape:
        data = scrape_data_dict(data, meas_names + exp_names)
    return data


# def get_all_data(data):
#     parameters = data.parameters.split(',')
#     all_data = {}
#     for p in parameters:
#         pspec = data.paramspecs[p]
#         all_data['p'] = {'label': pspec.label,
#                          'unit': pspec.unit,
#                          'data': np.array(data.get_data(p)).flatten()}
#     return all_data


def organize_fit_data(data, **setpoint_values):
    # extract metadata and parameters present
    metadata = load_json_metadata(data)
    parameters = data.parameters.split(',')

    # check fit parameters and setpoint parameters are present
    if 'success' not in parameters:
        raise RuntimeError(
            f"'success' parameter found "
            "in dataset. Parameters present are {parameters}")
    if not all(v in parameters for v in metadata['fitter']['fit_parameters']):
        raise RuntimeError(
            f"{metadata['fitter']['fit_parameters']} not all found "
            "in dataset. Parameters present are {parameters}")
    if not all(v in parameters for v in setpoint_values.keys()):
        raise RuntimeError(
            f"{list(setpoint_values.keys())} not all found "
            "in dataset. Parameters present are {parameters}")

    # find indices for specified setpoint_values
    indices = []
    point_values = []
    for setpoint, value in setpoint_values.items():
        d = np.array(data.get_data(setpoint)).flatten()
        nearest_val = value + np.amin(d - value)
        indices.append(set(np.argwhere(d == nearest_val).flatten()))
        point_values.append(nearest_val)
    if len(indices) > 0:
        u = list(set.intersection(*indices))
    else:
        u = None

    # populate dictionaries
    setpoints = {}
    for p in parameters:
        depends_on = data.paramspecs[p].depends_on.split(', ')
        for d in depends_on:
            non_vals = list(setpoint_values.keys()) + ['']
            if d not in non_vals:
                setpoints[d] = None
    fit = {}
    variance = {}
    initial_values = {}
    for name in parameters:
        if name == 'success':
            success = {
                'name': name,
                'label': data.paramspecs[name].label,
                'unit': data.paramspecs[name].unit,
                'data': np.array(data.get_data(name)).flatten()[u].flatten()}
        elif name in metadata['fitter']['fit_parameters']:
            fit[name] = {
                'name': name,
                'label': data.paramspecs[name].label,
                'unit': data.paramspecs[name].unit,
                'data': np.array(data.get_data(name)).flatten()[u].flatten()}
        elif name in metadata['fitter'].get('variance_parameters', []):
            variance[name] = {
                'name': name,
                'label': data.paramspecs[name].label,
                'unit': data.paramspecs[name].unit,
                'data': np.array(data.get_data(name)).flatten()[u].flatten()}
        elif name in metadata['fitter'].get('initial_value_parameters', []):
            initial_values[name] = {
                'name': name,
                'label': data.paramspecs[name].label,
                'unit': data.paramspecs[name].unit,
                'data': np.array(data.get_data(name)).flatten()[u].flatten()}
        elif name in setpoints:
            setpoints[name] = {
                'name': name,
                'label': data.paramspecs[name].label,
                'unit': data.paramspecs[name].unit,
                'data': np.array(data.get_data(name)).flatten()[u].flatten()}

    return success, fit, variance, initial_values, setpoints, point_values


# def organize_exp_data(data, dependent_parameter_name,
#                       *independent_parameter_names, **setpoint_values):
#     """
#     Given a dataset and the names of the parameters tp be used in the fit
#     organizes the data into dictionaries. Filters the data based on
#     specified setpoint_values.

#     Args:
#         data (qcodes dataset)
#         dependent_parameter_name (str): name of the parameter in the dataset
#             to be fit to (think y axis)
#         independent_parameter_names (list of strings): names of the
#             parameters in the dataset at which you measure the dependent
#             parameter and are to be used in the fit (think x axis)
#         setpoint_values (dict) (optional): keys should be names of parameters
#             in the dataset (not dependent or independent parameters). values
#             are the values of these parameters for which you want the data
#             extracted.

#     Returns:
#         dependent (dict): data dictionary of the dependent variable with keys
#             'name', 'label', 'unit', 'data'.
#         independent (dict): dictionary of independent variables data with keys
#             being the names of the independent variables and values being the
#             corresponding data dictionaries.
#         setpoints (dict): dictionariy of setpoint variables where present
#             (ie all parameters in the dataset outside of the independent and
#             dependent parameters). This excludes any specified in
#             setpoint_values. Same format as independent dict so keys are
#             parameter names and values are data dictionaries.
#     """
#     parameters = data.parameters.split(',')
#     if not all(v in parameters for v in setpoint_values.keys()):
#         raise RuntimeError(f'{list(setpoint_values.keys())} not all found '
#                            f'in dataset. Parameters present are {parameters}')

#     indices = []
#     for setpoint, value in setpoint_values.items():
#         d = np.array(data.get_data(setpoint)).flatten()
#         nearest_val = value + np.amin(d - value)
#         indices.append(set(np.argwhere(d == nearest_val).flatten()))
#     if len(indices) > 0:
#         u = list(set.intersection(*indices))
#     else:
#         u = None

#     setpoints = {}
#     for p in parameters:
#         depends_on = data.paramspecs[p].depends_on.split(', ')
#         for d in depends_on:
#             non_vals = list(setpoint_values.keys()) + \
#                 [''] + list(independent_parameter_names)
#             if d not in non_vals:
#                 setpoints[d] = None
#     dependent = None
#     independent = {}
#     for name in parameters:
#         if name == dependent_parameter_name:
#             dependent = {
#                 'name': name,
#                 'label': data.paramspecs[name].label,
#                 'unit': data.paramspecs[name].unit,
#                 'data': np.array(data.get_data(name)).flatten()[u].flatten()}
#         elif name in independent_parameter_names:
#             independent[name] = {
#                 'name': name,
#                 'label': data.paramspecs[name].label,
#                 'unit': data.paramspecs[name].unit,
#                 'data': np.array(data.get_data(name)).flatten()[u].flatten()}
#         elif name in setpoints:
#             setpoints[name] = {
#                 'name': name,
#                 'label': data.paramspecs[name].label,
#                 'unit': data.paramspecs[name].unit,
#                 'data': np.array(data.get_data(name)).flatten()[u].flatten()}
#     if dependent is None:
#         raise RuntimeError(f'{dependent_parameter_name} not found in dataset. '
#                            f'Parameters present are {parameters}')
#     if len(independent) != len(independent_parameter_names):
#         raise RuntimeError(f'{independent_parameter_names} not all found '
#                            f'in dataset. Parameters present are {parameters}')
#     return dependent, independent, setpoints


def organize_fit_data(data, **setpoint_values):
    """
    Given a fit dataset works out the categories from the metadata and sorts
    the data into corresponding dictionaries. Filters the data based on
    specified setpoint_values.

    Args:
        data (qcodes dataset)
        setpoint_values (dict) (optional): keys should be names of parameters
            in the dataset. values are the values of these parameters for
            which you want the data extracted.

    Returns:
        success (dict): data dictionary of the success variable of the fitter
            with keys 'name', 'label', 'unit', 'data'
        fit (dict): dictionary of fit_parameter data with keys
            being the names of the fit_parameters and values being the
            corresponding data dictionaries
        variance (dict): dictionary of variance_parameter data where present.
            Same format as fit. Otherwise empty dict.
        initial_values (dict): dictionary of initial_value_parameter data
            where present. Same format as fit. Otherwise empty dict.
        setpoints (dict): dictionariy of setpoint variables where present
            (ie all parameters in the dataset outside of the independent and
            dependent parameters). This excludes any specified in
            setpoint_values. Same format as independent dict so keys are
            parameter names and values are data dictionaries.
        point_values (list): actual value for each specified setpoint_value
             in the same order where the data returned in other dictionaries
             will be for these setpoint values as they were the nearest in the
             dataset to those specified.
    """

    # extract metadata and parameters present
    metadata = load_json_metadata(data)
    parameters = data.parameters.split(',')

    # check fit parameters and setpoint parameters are present
    if 'success' not in parameters:
        raise RuntimeError(
            f"'success' parameter found "
            "in dataset. Parameters present are {parameters}")
    if not all(v in parameters for v in metadata['fitter']['fit_parameters']):
        raise RuntimeError(
            f"{metadata['fitter']['fit_parameters']} not all found "
            "in dataset. Parameters present are {parameters}")
    if not all(v in parameters for v in setpoint_values.keys()):
        raise RuntimeError(
            f"{list(setpoint_values.keys())} not all found "
            "in dataset. Parameters present are {parameters}")

    # find indices for specified setpoint_values
    indices = []
    point_values = []
    for setpoint, value in setpoint_values.items():
        d = np.array(data.get_data(setpoint)).flatten()
        nearest_val = value + np.amin(d - value)
        indices.append(set(np.argwhere(d == nearest_val).flatten()))
        point_values.append(nearest_val)
    if len(indices) > 0:
        u = list(set.intersection(*indices))
    else:
        u = None

    # populate dictionaries
    setpoints = {}
    for p in parameters:
        depends_on = data.paramspecs[p].depends_on.split(', ')
        for d in depends_on:
            non_vals = list(setpoint_values.keys()) + ['']
            if d not in non_vals:
                setpoints[d] = None
    fit = {}
    variance = {}
    initial_values = {}
    for name in parameters:
        if name == 'success':
            success = {
                'name': name,
                'label': data.paramspecs[name].label,
                'unit': data.paramspecs[name].unit,
                'data': np.array(data.get_data(name)).flatten()[u].flatten()}
        elif name in metadata['fitter']['fit_parameters']:
            fit[name] = {
                'name': name,
                'label': data.paramspecs[name].label,
                'unit': data.paramspecs[name].unit,
                'data': np.array(data.get_data(name)).flatten()[u].flatten()}
        elif name in metadata['fitter'].get('variance_parameters', []):
            variance[name] = {
                'name': name,
                'label': data.paramspecs[name].label,
                'unit': data.paramspecs[name].unit,
                'data': np.array(data.get_data(name)).flatten()[u].flatten()}
        elif name in metadata['fitter'].get('initial_value_parameters', []):
            initial_values[name] = {
                'name': name,
                'label': data.paramspecs[name].label,
                'unit': data.paramspecs[name].unit,
                'data': np.array(data.get_data(name)).flatten()[u].flatten()}
        elif name in setpoints:
            setpoints[name] = {
                'name': name,
                'label': data.paramspecs[name].label,
                'unit': data.paramspecs[name].unit,
                'data': np.array(data.get_data(name)).flatten()[u].flatten()}

    return success, fit, variance, initial_values, setpoints, point_values
