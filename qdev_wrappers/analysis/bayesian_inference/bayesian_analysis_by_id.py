from qcodes.dataset.data_export import load_by_id
from qdev_wrappers.analysis.base import AnalyserBase
from qcodes.dataset.measurements import Measurement
from qdev_wrappers.analysis.helpers import make_json_metadata
from qcodes.dataset.experiment_container import load_experiment
import random
from qdev_wrappers.analysis.basic.readout_fidelity import ReadoutFidelityAnalyser, plot_readout_fidelity
from qdev_wrappers.analysis.basic.projector import RealImagProjector
from qdev_wrappers.analysis.basic.decision_maker import DecisionMaker
from qdev_wrappers.analysis.bayesian_inference.base import plot_bayesian_analysis
import numpy as np
from qdev_wrappers.analysis.helpers import scrape_data_dict, organize_experiment_data, refine_data_dict


def bayesian_analysis_by_id(run_id, fidelity_run_id, analyser,
                            num=None, randomised=False, setpoints=None,
                            mappings=None, save=False):
    if save:
        raise RuntimeError('Not implemented with saving yet')

    plots = []
    if setpoints is None:
        setpoints = {}
    if mappings is None:
        mappings = {}

    print(f'Loading data from run_id {fidelity_run_id} for readout fidelity')
    # readout fidelity analysis
    d = load_by_id(fidelity_run_id)
    fidelity_experiment = load_experiment(d.exp_id)
    fidelity_analyser = ReadoutFidelityAnalyser()
    fidelity_mappings = {**mappings,
                         'real': mappings.get('fidelity_real',
                                              'cavity_real_response'),
                         'imaginary': mappings.get('fidelity_imaginary',
                                                   'cavity_imaginary_response')}
    fidelity_data = organize_experiment_data(d, fidelity_analyser, {}, fidelity_mappings)
    print(f'Running readout fidelity analysis')
    fidelity_analyser.analyse(**fidelity_data)
    fidelity_result = {}
    for m, p in fidelity_analyser.model_parameters.parameters.items():
        fidelity_result[m] = p()
    fidelity_experiment = load_experiment(fidelity_run_id)
    rf_title = 'Readout Fidelity ({} on {} (run ID {})'.format(
        fidelity_experiment.name, fidelity_experiment.sample_name, fidelity_run_id)
    plots.extend(plot_readout_fidelity(
        {**fidelity_data, **fidelity_result}, title=rf_title))

    print(f'Loading data from run_id {run_id} for bayesian analysis')
    # load experiment data
    d = load_by_id(run_id)
    projector = RealImagProjector()
    data = organize_experiment_data(d, projector, setpoints, mappings,
                                    scrape=False)
    data.update({'angle': fidelity_result['angle'],
                 'decision_value': fidelity_result['decision_value']})

    print(f'Projecting data')
    p_names = list(projector.metadata['measurement_parameters'].keys())
    p_names.extend(list(projector.metadata['experiment_parameters'].keys()))
    scraped_data = scrape_data_dict(data, p_names)
    projector.analyse(**scraped_data)
    data['projected_data'] = projector.model_parameters.projected_data()
    data.pop('real_data')
    data.pop('imaginary_data')

    # average projected data
    print(f'Averaging data')
    rep_num_name = mappings.get('rep_num', 'rep_num')
    if rep_num_name in data:
        rep_num = data[rep_num_name]
        averaged_data = data['projected_data'].reshape(
            (np.max(rep_num + 1), -1))
        averaged_projected_data = np.mean(averaged_data, axis=0)
    else:
        averaged_projected_data = None

    # decision make data
    print(f'Making decisions')
    decision_maker = DecisionMaker()
    p_names = list(decision_maker.metadata['measurement_parameters'].keys())
    p_names.extend(list(decision_maker.metadata['experiment_parameters'].keys()))
    scraped_data = scrape_data_dict(data, p_names)
    decision_maker.analyse(**scraped_data)
    data['qubit_state'] = decision_maker.model_parameters.qubit_state()

    # indices generation
    print(f'Making indices')
    if num is None:
        d_len = len(data['qubit_state'])
        num = d_len
    elif rep_num_name in data:
        num_per_rep = np.count_nonzero(data[rep_num_name] == 0)
        reps_needed = np.ceil(num / num_per_rep)
        data = refine_data_dict(data, {rep_num_name: np.arange(reps_needed)}, {})
        data.pop(rep_num_name)
        s = len(data['qubit_state'])
    if randomised:
        data_indices = random.sample(np.arange(d_len), num)
        for k, v in data.items():
            if len(v) > 1:
                v = v[data_indices]
    data = scrape_data_dict(data, cutoff=num)

    # TODO: register parameters

    # run analysis
    print(f'Reorganising data')
    model_names = list(analyser.metadata['model_parameters'].keys())
    res = {n: [] for n in model_names}
    res.update({n + '_variance': [] for n in model_names})
    res.update({n + '_posterior_marginal': [] for n in model_names})
    res.update({n + '_posterior_marginal_setpoints': [] for n in model_names})
    res['averaged_projected_data'] = averaged_projected_data[:num]
    analyser.reset_updater()
    exp_names = list(analyser.metadata['experiment_parameters'].keys())
    meas_names = list(analyser.metadata['measurement_parameters'].keys())
    p_names = meas_names + exp_names
    scraped_data = scrape_data_dict(data, p_names)
    success = True
    i = 0
    print(f'Running analysis')
    while i < (num - 1) and success:
        if i == 0:
            for n in model_names:
                d_p = analyser.distr_parameters.parameters[n + '_posterior_marginal']
                d_p_s = analyser.distr_parameters.parameters[n + '_posterior_marginal_setpoints']
                res[n + '_posterior_marginal'].append(d_p())
                res[n + '_posterior_marginal_setpoints'].append(d_p_s())
        ind_data = {k: v[i] for k, v in scraped_data.items()}
        analyser.analyse(**ind_data)
        success = analyser.success()
        for n in model_names:
            res[n].append(analyser.model_parameters.parameters[n]())
            res[n + '_variance'].append(analyser.variance_parameters.parameters[n + '_variance']())
        i += 1
    if success:
        for n in model_names:
            d_p = analyser.distr_parameters.parameters[n + '_posterior_marginal']
            d_p_s = analyser.distr_parameters.parameters[n + '_posterior_marginal_setpoints']
            res[n + '_posterior_marginal'].append(d_p())
            res[n + '_posterior_marginal_setpoints'].append(d_p_s())

    scraped_data = scrape_data_dict(data, exp_names)
    data['estimate'] = analyser.evaluate(**scraped_data)
    data.update(res)
    experiment = load_experiment(d.exp_id)
    ba_title = 'Bayesian analysis ({} on {} (run ID {})'.format(
        experiment.name, experiment.sample_name, run_id)
    plots.extend(plot_bayesian_analysis(data, analyser.metadata, title=ba_title))
    return plots, data