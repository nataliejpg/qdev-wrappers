import numpy as np
import json
from qcodes.dataset.sqlite.database import get_DB_location, connect
from qcodes.dataset.data_export import load_by_id


def load_xarrays(run_id, *names, conn=None):
    conn = conn or connect(get_DB_location())
    data = load_by_id(run_id, conn=conn)
    dfs = data.get_data_as_pandas_dataframe()
    xarrays = []
    for name in names:
        xarrays.append(dfs[name].to_xarray()[name])
    return xarrays


def strip_none(arr, u):
    return np.array(arr).flatten()[u].flatten()
    # return arr[arr != np.array(None)]


def make_json_metadata(dataset, fitter, dependent_parameter_name,
                       *independent_parameter_names, average_names=None):
    average_names = average_names or []
    exp_metadata = {'run_id': dataset.run_id,
                    'exp_id': dataset.exp_id,
                    'exp_name': dataset.exp_name,
                    'sample_name': dataset.sample_name}
    metadata = {'fitter': fitter.metadata,
                'inferred_from': {'dept_var': dependent_parameter_name,
                                  'indept_vars': independent_parameter_names,
                                  'average_vars': average_names,
                                  **exp_metadata}}
    metadata_key = 'fitting_metadata'
    return metadata_key, json.dumps(metadata)


def load_json_metadata(dataset):
    try:
        return json.loads(dataset.metadata['fitting_metadata'])
    except KeyError:
        raise RuntimeError(
            "'fitting_metadata' not found in dataset metadata, are you sure "
            "this is a fitted dataset?")


def organize_exp_data(run_id, dept_name, *indept_names, average_names=None,
                      conn=None, **setpoint_values):
    data = load_by_id(run_id, conn=conn)
    parameters = data.paramspecs.keys()
    parm_names = [dept_name] + list(indept_names) + list(setpoint_values.keys())
    parm_names += average_names if average_names is not None else []
    for n in parm_names:
        if n not in parameters:
            raise RuntimeError('{} not found in dataset {}'.format(n, run_id))
    dept_xarr = load_xarrays(run_id, dept_name, conn=conn)[0]
    for s, v in setpoint_values.items():
        nearest_val = dept_xarr[s].values[np.argmin(abs(dept_xarr[s] - v))]
        setpoint_values[s] = v
    dept_xarr = dept_xarr.sel(**setpoint_values)
    if average_names:
        dept_xarr = dept_xarr.mean(*average_names)
    indept_xarrs = []
    for i_n in indept_names:
        indept_xarrs.append(dept_xarr[i_n])
    return dept_xarr, indept_xarrs


def organize_fit_data(run_id, conn=None):
    data = load_by_id(run_id, conn=conn)
    metadata = load_json_metadata(data)
    parameters = data.paramspecs.keys()
    fit_parm_names = metadata['fitter']['fit_parameters']
    p_names = ['success'] + fit_parm_names
    for n in p_names:
        if n not in parameters:
            raise RuntimeError('{} not found in {}'.format(n, run_id))
    fit_xarrs = load_xarrays(run_id, *fit_parm_names, conn=conn)
    success_xarr = load_xarrays(run_id, 'success', conn=conn)[0]
    var_pnames = metadata['fitter'].get('variance_parameters', [])
    initial_val_pnames = metadata['fitter'].get('initial_value_parameters', [])
    if var_pnames:
        var_xarrs = load_xarrays(run_id, *var_pnames, conn=conn)
    else:
        var_xarrs = None
    if initial_val_pnames:
        initial_val_xarrs = load_xarrays(run_id, *initial_val_pnames, conn=conn)
    else:
        initial_val_xarrs = None
    return success_xarr, fit_xarrs, var_xarrs, initial_val_xarrs


# def organize_exp_data(run_id, dept_parm_name,
#                       *indept_parm_names, conn=None,
#                       **setpoint_values):
#     """
#     Returns:
#         dependent (dict): data dictionary of the dependent variable with keys
#             'name', 'label', 'unit', 'data'.
#         independent (list): list of dicts of independent variables data
#             dictionaries
#         setpoints (list): list of dicts of setpoint data dictionaries with
#             nearest setpoint value
#     """
#     data = load_by_id(run_id, conn=conn)
#     parameters = data.paramspecs.keys()
#     for n in [dept_parm_name] + list(indept_parm_names) + list(setpoint_values.keys()):
#         if n not in parameters:
#             raise RuntimeError('{} not found in dataset {}'.format(n, run_id))
#     dept_xarr = load_xarrays(run_id, dept_parm_name, conn=conn)[0]

#     other_data_dicts = []
#     for n in parameters:
#         if n not in [dept_parm_name] + list(indept_parm_names) + list(setpoint_values.keys()):
#             if n in dept_xarr.coords:
#                 other_data_dicts.append({'name': n,
#                                          'label': data.paramspecs[n].label,
#                                          'unit': data.paramspecs[n].unit,
#                                          'data': dept_xarr[n]})

#     setpoint_dicts = []
#     for k, v in setpoint_values.items():
#         nearest_val = dept_xarr[k].values[np.argmin(abs(dept_xarr[k] - v))]
#         cond = {k: nearest_val}
#         dept_xarr = dept_xarr.sel(**cond)
#         for d in other_data_dicts:
#             d['data'] = d['data'].sel(**cond)
#         setpoint_dicts.append({'name': k,
#                                'label': data.paramspecs[k].label,
#                                'unit': data.paramspecs[k].unit,
#                                'data': nearest_val})

#     for d in other_data_dicts:
#         d['data'] = d['data'].sortby(d['data'])
#         dept_xarr = dept_xarr.sortby(d['name'])
#     setpoint_dicts = other_data_dicts + setpoint_dicts

#     dept_dict = {'name': dept_parm_name,
#                  'label': data.paramspecs[dept_parm_name].label,
#                  'unit': data.paramspecs[dept_parm_name].unit,
#                  'data': dept_xarr}

#     indept_dicts = []
#     for k, v in dept_xarr.coords.items():
#         if k in indept_parm_names:
#             indept_dicts.append({'name': k,
#                                  'label': data.paramspecs[k].label,
#                                  'unit': data.paramspecs[k].unit,
#                                  'data': v})

#     return dept_dict, indept_dicts, setpoint_dicts


# def organize_fit_data(run_id, conn=None, **setpoint_values):
#     """
#     Returns:
#         success (dict): data dictionary of the success variable of the fitter
#             with keys 'name', 'label', 'unit', 'data'
#         fit (list): list of data dictionaries dictionary of fit_parameter data
#         variance (list): list of data dictionaries dictionary of fit_parameter
#             variance data if present
#         initial_values (list): list of data dictionaries dictionary of
#             fit_parameter initial value data if present
#         setpoints (list): list of dicts of setpoint data dictionaries with
#                           nearest setpoint value
#     """
#     data = load_by_id(run_id, conn=conn)
#     # extract metadata and parameters present
#     metadata = load_json_metadata(data)
#     parameters = data.paramspecs.keys()

#     # check fit parameters and setpoint parameters are present
#     if 'success' not in parameters:
#         raise RuntimeError(
#             f"'success' parameter found "
#             "in dataset. Parameters present are {parameters}")
#     p_names = ['success'] + metadata['fitter']['fit_parameters'] + list(setpoint_values.keys())
#     for n in p_names:
#         if n not in parameters:
#             raise RuntimeError(f'{n} not found in {run_id}')
#     fit_xarrs = load_xarrays(run_id, *metadata['fitter']['fit_parameters'], conn=conn)
#     success_xarr = load_xarrays(run_id, 'success', conn=conn)[0]
#     var_pnames = metadata['fitter'].get('variance_parameters', [])
#     initial_val_pnames = metadata['fitter'].get('initial_value_parameters', [])
#     if var_pnames:
#         var_xarrs = load_xarrays(run_id, *var_pnames, conn=conn)
#     else:
#         var_xarrs = []
#     if initial_val_pnames:
#         initial_val_xarrs = load_xarrays(run_id, *initial_val_pnames, conn=conn)
#     else:
#         initial_val_xarrs = []

#     other_data_dicts = []
#     for n in parameters:
#         if n not in p_names + var_pnames + initial_val_pnames:
#             other_data_dicts.append({'name': n,
#                                      'label': data.paramspecs[n].label,
#                                      'unit': data.paramspecs[n].unit,
#                                      'data': fit_xarrs[0][n]})

#     setpoint_dicts = []
#     for k, v in setpoint_values.items():
#         nearest_val = fit_xarrs[0][k].values[np.argmin(abs(fit_xarrs[0][k] - v))]
#         cond = {k: nearest_val}
#         for i in range(len(metadata['fitter']['fit_parameters'])):
#             fit_xarrs[i] = fit_xarrs[i].sel(**cond)
#             var_xarrs[i] = var_xarrs[i].sel(**cond)
#             initial_val_xarrs[i] = initial_val_xarrs[i].sel(**cond)
#         success_xarr = success_xarr.sel(**cond)
#         for d in other_data_dicts:
#             d['data'] = d['data'].sel(**cond)
#         setpoint_dicts.append({'name': k,
#                                'label': data.paramspecs[k].label,
#                                'unit': data.paramspecs[k].unit,
#                                'data': nearest_val})

#     for d in other_data_dicts:
#         d['data'] = d['data'].sortby(d['data'])
#         for i in range(len(metadata['fitter']['fit_parameters'])):
#             fit_xarrs[i] = fit_xarrs[i].sortby(d['name'])
#             var_xarrs[i] = var_xarrs[i].sortby(d['name'])
#             initial_val_xarrs[i] = initial_val_xarrs[i].sortby(d['name'])
#         success_xarr = success_xarr.sortby([d['name']])
#     setpoint_dicts = other_data_dicts + setpoint_dicts

#     success_dict = {'name': 'success',
#                     'label': data.paramspecs['success'].label,
#                     'unit': data.paramspecs['success'].unit,
#                     'data': success_xarr}

#     fit_dicts = []
#     for i, n in enumerate(metadata['fitter']['fit_parameters']):
#         fit_dicts.append({'name': n,
#                           'label': data.paramspecs[n].label,
#                           'unit': data.paramspecs[n].unit,
#                           'data': fit_xarrs[i]})

#     var_dicts = []
#     for i, n in enumerate(var_pnames):
#         var_dicts.append({'name': n,
#                           'label': data.paramspecs[n].label,
#                           'unit': data.paramspecs[n].unit,
#                           'data': var_xarrs[i]})

#     initial_val_dicts = []
#     for i, n in enumerate(initial_val_pnames):
#         initial_val_dicts.append({'name': n,
#                                   'label': data.paramspecs[n].label,
#                                   'unit': data.paramspecs[n].unit,
#                                   'data': initial_val_xarrs[i]})

#     return success_dict, fit_dicts, var_dicts, initial_val_dicts, setpoint_dicts
