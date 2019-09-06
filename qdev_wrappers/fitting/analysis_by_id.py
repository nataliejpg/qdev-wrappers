import numpy as np
import json
from itertools import product
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.data_export import load_by_id
from qcodes.instrument.parameter import Parameter
from qdev_wrappers.bayesian_analysis.base import BayesianAnalyser
from qdev_wrappers.bayesian_analysis.plotting import plot_analysis_by_id
from qdev_wrappers.analysis.helpers import organize_exp_data, make_json_metadata
from qdev_wrappers.bayesian_analysis.readout_fidelity import fidelity_info_from_run_id


def analysis_by_id(data_run_id: int,
                   analysers,
                   converters=None,
                   plot=True,
                   save_plots=True,
                   num_pts=100,
                   randomised=False,
                   fidelity_run_id=None,
                   **parameter_mappings):


    if converters is not None:
        if type(converters) is not list:
            converters = [converters]
        for c in converters:
            organized_data = []

    # find readout fidelity info
    if fidelity_run_id is not None:
        fidelity_info = fidelity_info_from_run_id(fidelity_run_id)
        angle = fidelity_info.angle * np.pi / 180
        decision_val = fidelity_info.tmp_bins[fidelity_info.max_separation_index]

        converter = Projector()
        meas_parm_names = converter.metadata['input_data']
    else:
        meas_parm_names = list(
            bayesian_analysers[0].metadata['measurement_parameters'].keys())
    mapped_meas_names = [parameter_mappings.get(n, n) for n in meas_parm_names]

    # load and organize data
    exp_data = load_by_id(data_run_id)
    measurement, experiment, data_indices = organize_exp_data(
        exp_data, *mapped_meas_names)

    # project data
    if fidelity_run_id is not None:
        converter_input = [measurement[p]['data'] for p in mapped_meas_names]
        projected_measurement = converter.convert(*converter_input, angle,
                                                  decision_value=decision_val,
                                                  real_imag=True)
        state_data = projected_measurement['qubit_state']['data']

    # set up measurement and register data
    num_updates = bayesian_analysers[0].num_updates
    data_index = Parameter(name='data_index',
                           label=data_indices['label'])
    model_params = []
    distr_params = []
    meas = Measurement()
    meas.register_parameter(num_updates)
    meas.register_parameter(data_index, setpoints(num_updates,))
    for b in bayesian_analysers:
        for n in b.metadata['model_parameters'].keys():
            model_param = b.model_parameters[n]
            var_param = b.variance_parameters[n + '_variance']
            meas.register_parameter(model_param, setpoints=(num_updates,))
            meas.register_parameter(var_param, setpoints=(num_updates,))
            model_params.extend([model_param, var_param])
            posterior_param = b.distr_parameters[
                n + 'posterior_marginal']
            posterior_param_setpoints = b.distr_parameters[
                n + 'posterior_marginal_setpoints']
            meas.register_parameter(prior_param, setpoints=(num_updates,))
            distr_params.extend([posterior_param, posterior_param_setpoints])

    exp_parm_names = [experiment_parameter_names[k] for k in expected_exp_parm_names.keys()]
    meas_parm_names = ['qubit_state']
    metadata = make_json_metadata(
        exp_data, meas_parm_names, exp_parm_names, *bayesian_analysers,
        converter=converter)

    # run analysis
    with meas.run() as datasaver:
        datasaver._dataset.add_metadata(*metadata)
        analysis_run_id = datasaver.run_id
        if randomised:
            analysis_indices = random.sample(np.arange(len(state_data)),
                                             num_pts)
        else:
            analysis_indices = np.arange(num_pts)

        for b in bayesian_analysers:
            b.reset_updater()

        res = [('num_updates', num_updates())]
        for p in distr_params:
            res.append((p.name. p()))

        datasaver.add_result(*res)
        for i in range(num_pts):
            j = analysis_indices[i]
            data_index = data_indices['data'][j]
            zero_state = int(not state_data[j])
            exp_param_vals = {k: experiment[v]['data'][j]
                              for k, v in experiment_parameter_names.items()}
            for b in bayesian_analysers:
                b.update(zero_state, **exp_param_vals)
            res = [('data_index', data_index),
                   ('num_updates', num_updates())]
            res.extend([(p.name, p()) for p in model_params])
            datasaver.add_result(*res)

        res = [('num_updates', num_updates())]
        for p in distr_params:
            res.append((p.name. p()))

    # plot
    if plot:
        axes, colorbar = plot_analysis_by_id(
            analysis_run_id,
            save_plots=save_plots)
    else:
        axes, colorbar = [], None

    return analysis_run_id, axes, colorbar