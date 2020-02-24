import numpy as np
import json
from itertools import product
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.data_export import load_by_id
from qcodes.dataset.experiment_container import load_last_experiment
from qcodes.instrument.parameter import Parameter
from qdev_wrappers.fitting.fitters import LeastSquaresFitter
from qdev_wrappers.fitting.plotting import plot_fit_by_id
from qdev_wrappers.fitting.helpers import organize_exp_data, make_json_metadata


def fit_by_id(data_run_id, fitter,
              dept_name,
              *indept_names,
              average_names=None,
              plot=True,
              save_plots=True,
              show_variance=True,
              show_initial_values=False,
              source_conn=None,
              target_conn=None,
              **kwargs):

    # load and organise data
    exp_data = load_by_id(data_run_id, conn=source_conn)
    p_names = exp_data.paramspecs.keys()
    setpoint_values = {}
    for p in p_names:
        if p in kwargs:
            setpoint_values[p] = kwargs.pop(p)
    dependent, independent = organize_exp_data(
        data_run_id, dept_name, *indept_names,
        average_names=average_names, conn=source_conn, **setpoint_values)
    for s in setpoint_values.keys():
        setpoint_values[s] = dependent[s].values
    for coord in dependent.coords:
        if coord not in list(setpoint_values.keys()) + list(indept_names) + ['index']:
            setpoint_values[coord] = dependent[coord].values

    # set up measurment
    if target_conn is not None:
        target_exp = load_last_experiment(target_conn)
    else:
        target_exp = None
    meas = Measurement(exp=target_exp)

    # register setpoints
    setpoint_names = list(setpoint_values.keys())
    for setpoint_name in setpoint_names:
        paramspec = exp_data.paramspecs[setpoint_name]
        meas.register_custom_parameter(name=paramspec['name'],
                                       label=paramspec['label'],
                                       unit=paramspec['unit'])

    # register fit parameters
    for param in fitter.all_parameters:
        meas.register_parameter(param,
                                setpoints=setpoint_names or None)

    # generate metadata
    metadata = make_json_metadata(
        exp_data, fitter, dept_name, *indept_names,
        average_names=average_names)

    # fit and save result
    setpoint_comb_values = product(*[v for v in setpoint_values.values()])
    with meas.run() as datasaver:
        datasaver._dataset.add_metadata(*metadata)
        fit_run_id = datasaver.run_id
        # if len(setpoint_names) == 0:
        #     dept_data = dependent.values
        #     indept_data = [dependent[ind].values for ind in indept_names]
        #     fitter.fit(dept_data, *indept_data, **kwargs)
        #     result = [(f.name, f()) for f in fitter.all_parameters]
        #     datasaver.add_result(*result)
        for vals in setpoint_comb_values:
            comb_dict = dict(zip(setpoint_names, vals))
            dept_data = dependent.sel(**comb_dict)
            indept_data = [dept_data[ind].values for ind in indept_names]
            dept_data = dept_data.values
            fitter.fit(dept_data, *indept_data, **kwargs)
            result = [(k, v) for k, v in comb_dict.items()]
            for f in fitter.all_parameters:
                result.append((f.name, f()))
            datasaver.add_result(*result)

    # plot
    if plot:
        axes, colorbar = plot_fit_by_id(
            fit_run_id,
            show_variance=show_variance,
            show_initial_values=show_initial_values,
            source_conn=source_conn,
            target_conn=target_conn,
            save_plots=save_plots,
            **setpoint_values)
    else:
        axes, colorbar = [], None

    return fit_run_id, axes, colorbar




    # # run fit for data
    # with meas.run() as datasaver:
    #     datasaver._dataset.add_metadata(*metadata)
    #     fit_run_id = datasaver.run_id
    #     if len(setpoints) > 0:
    #         # find all possible combinations of setpoint values
    #         setpoint_combinations = product(
    #             *[set(v['data']) for v in setpoints.values()])
    #         for setpoint_combination in setpoint_combinations:
    #             # find indices where where setpoint combination is satisfied
    #             indices = []
    #             for i, setpoint in enumerate(setpoints.values()):
    #                 indices.append(
    #                     set(np.argwhere(setpoint['data'] ==
    #                                     setpoint_combination[i]).flatten()))
    #             u = None
    #             if len(indices) > 0:
    #                 u = list(set.intersection(*indices))
    #             dependent_data = dependent['data'][u].flatten()
    #             independent_data = [d['data'][u].flatten()
    #                                 for d in independent.values()]
    #             fitter.fit(dependent_data, *independent_data, **kwargs)
    #             result = list(zip(setpoint_paramnames, setpoint_combination))
    #             for fit_param in fitter.all_parameters:
    #                 result.append((fit_param.name, fit_param()))
    #             datasaver.add_result(*result)
    #     else:
    #         dependent_data = dependent['data']
    #         independent_data = [d['data'] for d in independent.values()]
    #         fitter.fit(dependent_data, *independent_data, **kwargs)
    #         result = [(p.name, p()) for p in fitter.all_parameters]
    #         datasaver.add_result(*result)
    # plot
    # if plot:
    #     print(':(', setpoint_values)
    #     axes, colorbar = plot_fit_by_id(fit_run_id,
    #                                     show_variance=show_variance,
    #                                     show_initial_values=show_initial_values,
    #                                     source_conn=source_conn,
    #                                     target_conn=target_conn,
    #                                     save_plots=save_plots,
    #                                     **setpoint_values)
    # else:
    #     axes, colorbar = [], None

    # return fit_run_id, axes, colorbar
