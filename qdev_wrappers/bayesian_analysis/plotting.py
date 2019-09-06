import matplotlib.pyplot as plt
import numpy as np
from qdev_wrappers.baysian_analysis.helpers import load_json_metadata, organize_exp_data, organize_analysis_data
from qcodes.dataset.data_export import load_by_id
from qdev_wrappers.dataset.doNd import save_image


def plot_analysis_by_id(analysis_run_id: int, save_plots: bool=True):
    # load data and metadata and sort into dictionaries based on metadata and
    # setpoint_values
    analysis = load_by_id(analysis_run_id)
    metadata = load_json_metadata(fit_data)
    real_parameter_name = metadata['inferred_from']['real_parameter_name']
    imaginary_parameter_name = metadata['inferred_from']['imaginary_parameter_name']
    experiment_parameter_names = metadata['inferred_from']['experiment_parameter_names']
    angle = metadata['inferred_from']['angle']
    decision_val = metadata['inferred_from']['decision_val']
    npts = metadata['inferred_from']['npts']
    exp_run_id = metadata['inferred_from']['run_id']
    exp_data = load_by_id(exp_run_id)

    projected, qubit_state, experiment_params = organize_exp_data(
        exp_data, real_parameter_name, imaginary_parameter_name,
        angle, decision_val, real_imag=True, **experiment_parameter_names)

    bayesian_analysis_metadata, inferred_arr = organize_analysis_data(analysis)

    axes = []
    for ba_name, meta in bayesian_analysis_metadata.items():
        for mp_name, info in meta['model_parameters'].items():
            f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.plot(np.arange(len(npts)),
                     info['data'], label='Mean', color='C0')
            ax1.legend()
            ax1.fill_between(np.arange(num),
                             np.array(info['data']) - np.sqrt(np.array(info['data_var'])),
                             np.array(rabi_freq) + np.sqrt(np.array(rabi_freq_var)),
                             alpha=0.3, facecolor='g')
            ax1.set_ylabel('{} ({})'.format(info['label'], info['units']))
            ax2.plot(np.arange(num), info['data_var'],
                     label='Variance', color='g')
            ax2.set_ylabel(info['label'] + ' Variance')
            ax2.set_xlabel('Number of Updates')
            ax2.legend()
            plt.suptitle('{} Analyser {} Estimate'.format(ba_name, mp_name))
            plt.tight_layout()
            axes.extend([ax1, ax2])
            f = plt.figure()
            plt.plot(info['prior_setpoints'], info['prior'], label='prior')
            plt.plot(info['posterior_setpoints'],
                     info['posterior'], label='posterior')
            plt.legend()
            plt.title('{} Analyser {} Distribution'.format(ba_name, mp_name))
            axes.extend(f.axes)

        if len(experiment_params) == 1:
            exp_param = list(experiment_params.values())[0]
            inferred_arr.sort()
            f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.plot(exp_param['data'][inferred_arr],
                     projected['data'][inferred_arr],
                     label='projected cavity response')
            ax1.plot(exp_param['data'][inferred_arr],
                     np.ones(len(exp_param['data'][inferred_arr])) *
                     metadata['inferred_from']['decision_val'],
                     linestyle='--')
            ax1.set_ylabel('Projected Cavity Response (V)')
            ax1.legend(loc='upper right')
            kwargs = {k: v['data'][-1] for k, v in meta['model_parameters']}
            kwargs.update({exp_param['name']: exp_param['data'][inferred_arr],
                           'np': np})
            fitted_arr = eval(meta['likelihood']['np'], kwargs)
            ax2.plot(exp_param['data'][inferred_arr], fitted_arr,
                     label='estimated ground state probability')
            ax1.set_ylabel('P(0)')
            ax2.legend(loc='upper right')
            ax2.set_xlabel('{} ({})'.format(exp_param['label'], exp_param['unit']))
            plt.suptitle('Data vs {} Analyser Ground State Probability Estimate'.format(ba_name))
            axes.extend([ax1, ax2])
        # 2D PLOTTING: nope
        else:
            warnings.warn(
                "Plotting for more than one experiment not yet implemented")

    # saving
    if save_plots:
        kwargs = {'run_id': analysis.run_id,
                  'exp_id': analysis.exp_id,
                  'exp_name': analysis.exp_name,
                  'sample_name': analysis.sample_name}
        save_image(axes, name_extension='analysis', **kwargs)
    return axes, colorbar