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
        if len(np.atleast_1d(v)) == 1:
            ind = np.argmin(abs(dept_xarr[s] - v))
            nearest_val = dept_xarr[s].values[ind]
            setpoint_values[s] = nearest_val
        else:
            for i in range(len(np.atleast_1d(v))):
                ind = np.argmin(abs(dept_xarr[s] - v[i]))
                nearest_val = dept_xarr[s].values[ind]
                setpoint_values[s][i] = nearest_val
    dept_xarr = dept_xarr.sel(**setpoint_values)
    if average_names:
        dept_xarr = dept_xarr.mean(*average_names)
    indept_xarrs = []
    for i_n in indept_names:
        indept_xarrs.append(dept_xarr[i_n])
    return dept_xarr, indept_xarrs


def organize_fit_data(run_id, conn=None, **setpoint_values):
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
    for s, v in setpoint_values.items():
        if len(np.atleast_1d(v)) == 1:
            ind = np.argmin(abs(success_xarr[s] - v))
            nearest_val = success_xarr[s].values[ind]
            setpoint_values[s] = nearest_val
        else:
            for i in range(len(np.atleast_1d(v))):
                ind = np.argmin(abs(success_xarr[s] - v[i]))
                nearest_val = success_xarr[s].values[ind]
                setpoint_values[s][i] = nearest_val
    success_xarr = success_xarr.sel(**setpoint_values)
    for f in fit_xarrs:
        f = f.sel(**setpoint_values)
    if var_pnames:
        for v in var_xarrs:
            v = v.sel(**setpoint_values)
    if initial_val_pnames:
        for i in initial_val_xarrs:
            i = i.sel(**setpoint_values)
    return success_xarr, fit_xarrs, var_xarrs, initial_val_xarrs
