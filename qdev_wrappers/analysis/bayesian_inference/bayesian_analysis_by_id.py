from qcodes.dataset.data_export import load_by_id
from qdev_wrappers.analysis.base import AnalyserBase
from qcodes.dataset.measurements import Measurement
from qdev_wrappers.analysis.helpers import make_json_metadata
from qcodes.dataset.experiment_container import load_experiment
import random
from qdev_wrappers.analysis.basic.readout_fidelity import plot_readout_fidelity
from qdev_wrappers.analysis.bayesian_inference.base import plot_bayesian_analysis
import numpy as np
from qdev_wrappers.analysis.helpers import scrape_data_dict, \
    organize_experiment_data, refine_data_dict, get_labeling_dict, save_analysis_figs


def bayesian_analysis_by_id(run_id, fidelity_run_id, analyser,
                            fidelity_analyser, projector, decision_maker,
                            num=None, randomised=False, setpoints=None,
                            mappings=None, save=False, plot=True,
                            plot_fidelity=False,
                            save_plots=True):
    if save:
        raise RuntimeError('Not implemented with saving yet')

    if setpoints is None:
        setpoints = {}
    if mappings is None:
        mappings = {}

    print(f'Loading data from run_id {fidelity_run_id} for readout fidelity')
    # readout fidelity analysis
    d = load_by_id(fidelity_run_id)
    fidelity_experiment = load_experiment(d.exp_id)
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
    rf_title = 'Readout Fidelity ({} on {} (run ID {})'.format(
        fidelity_experiment.name, fidelity_experiment.sample_name, fidelity_run_id)
    if plot_fidelity:
        fidelity_plots = plot_readout_fidelity(
            {**fidelity_data, **fidelity_result}, title=rf_title)
    else:
        fidelity_plots = []

    print(f'Loading data from run_id {run_id} for bayesian analysis')
    # load experiment data
    d = load_by_id(run_id)
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

    # decision make data
    print(f'Making decisions')
    p_names = list(decision_maker.metadata['measurement_parameters'].keys())
    p_names.extend(list(decision_maker.metadata['experiment_parameters'].keys()))
    scraped_data = scrape_data_dict(data, p_names)
    decision_maker.analyse(**scraped_data)
    data['qubit_state'] = decision_maker.model_parameters.qubit_state()

    # average projected data
    print(f'Averaging data')
    # rep_num_name = mappings.get('rep_num', 'rep_num')
    if 'rep_num' in data:
        rep_num = data['rep_num']
        averaged_data = (data['qubit_state'].reshape(
            (np.max(rep_num + 1), -1)) - 1) * -1
        averaged_state_data = np.mean(averaged_data, axis=0)
    else:
        averaged_state_data = None

    # indices generation
    print(f'Making indices')
    d_len = len(data['qubit_state'])
    if num is None:
        num = d_len
    if 'rep_num' in data:
        num_per_rep = np.count_nonzero(data['rep_num'] == 0)
        reps_needed = np.ceil(num / num_per_rep)
        data = refine_data_dict(data, {'rep_num': np.arange(reps_needed)}, {})
        data.pop('rep_num')
        d_len = len(data['qubit_state'])
    if randomised:
        data_indices = np.array(random.sample(list(np.arange(d_len)), k=num))
        for k, v in data.items():
            if len(np.array([v]).flatten()) > 1:
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
    try:
        res['averaged_state_data'] = averaged_state_data[:num]
    except TypeError:
        pass
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
    ba_labels = get_labeling_dict(analyser)
    if plot:
        ba_plots = plot_bayesian_analysis(data, analyser.metadata,
                                          labels=ba_labels, title=ba_title)
    else:
        ba_plots = []
    if save_plots:
        save_analysis_figs(fidelity_plots, fidelity_experiment.name,
                           fidelity_experiment.sample_name, fidelity_run_id)
        save_analysis_figs(ba_plots, experiment.name, experiment.sample_name,
                           run_id)
    plots = list(fidelity_plots) + list(ba_plots)
    return plots, data
