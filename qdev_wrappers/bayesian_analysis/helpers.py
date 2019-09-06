import numpy as np
from qcodes.dataset.data_export import load_by_id


def project_data(data_run_id, real_mag_name, imag_phase_name,
                 angle, decision_value, real_imag=True):
    data = load_by_id(data_run_id)
    if not real_imag:
        real = np.array(data.get_data(real_parameter_name)).flatten()
        imaginary = np.array(data.get_data(imaginary_parameter_name)).flatten()
    else:
        mag = np.array(data.get_data(real_mag_name)).flatten()
        phase = np.array(data.get_data(imag_phase_name)).flatten()
        real = np.real(mag * np.exp(1j * phase))
        imaginary = np.imag(mag * np.exp(1j * phase))
    projected_data = real * np.cos(angle) + imaginary * np.sin(angle) - decision_value
    state_data = (projected_data > 0).astype(int)
    return projected_data, state_data


def organize_exp_data(data, real_mag_name, imag_phase_name,
                      angle, decision_val, real_imag=True,
                      **experiment_parameter_names):
    experiment_param_data = {}
    for k, v in experiment_parameter_names.items():
        experiment_param_data[k] = np.array(data.get_data(v)).flatten()
    projected_data, state_data = project_data(
        data.run_id, real_mag_name, imag_phase_name, angle, decision_val,
        real_imag=True,)
    projected = {'name': 'projected_cavity_response',
                 'label': 'Projected Cavity Response',
                 'unit': 'a.u',
                 'data': projected_data}
    qubit_state = {'name': 'qubit_state',
                   'label': 'Qubit State',
                   'unit': 'a.u',
                   'data': state_data}
    experiment_params = {k: {'name': k,
                             'label': data.paramspecs[experiment_parameter_names[k]].label,
                             'unit': data.paramspecs[experiment_parameter_names[k]].unit,
                             'data': v} for k, v in
                         experiment_param_data.items()}
    return projected, qubit_state, experiment_params


def organize_analysis_data(data):
    metadata = load_json_metadata(data)
    inferred_arr = np.array(data.get_data('data_index')).flatten()
    parameter_data = metadata['bayesian_analysers']
    for ba_name, meta in parameter_data.items():
        for mp_name, info in meta['model_parameters'].keys():
            info['data'] = np.array(
                data.get_data('{}_{}'.format(ba_name, mp_name))).flatten()
            info['var_data'] = np.array(
                data.get_data(
                    '{}_{}_variance'.format(ba_name, mp_name))).flatten()
            num_updates = np.array(
                data.get_data(
                    'num_updates'.format(ba_name, mp_name))).flatten()
            posterior_marginal = np.array(
                data.get_data(
                    '{}_{}_posterior_marginal'.format(
                        ba_name, mp_name))).flatten()
            posterior_marginal_setpoints = np.array(
                data.get_data(
                    '{}_{}_posterior_marginal_setpoints'.format(
                        ba_name, mp_name))).flatten()
            info['prior'] = posterior_marginal[(num_updates == 0)]
            info['prior_setpoints'] = posterior_marginal_setpoints[(
                num_updates == 0)]
            info['posterior'] = posterior_marginal[(
                num_updates == num_updates[-1])]
            info['posterior_setpoints'] = posterior_marginal_setpoints[(
                num_updates == num_updates[-1])]
    return parameter_data, inferred_arr


def make_json_metadata(dataset, num_pts, real_parameter_name,
                       imaginary_parameter_name, bayesian_analysers,
                       **experiment_parameter_names):
    exp_metadata = {'run_id': dataset.run_id,
                    'exp_id': dataset.exp_id,
                    'exp_name': dataset.exp_name,
                    'sample_name': dataset.sample_name}
    metadata = {'bayesian_analysers': {
        ba.name: ba.metadata for ba in bayesian_analysers},
        'inferred_from': {
        'npts': num_pts,
        'angle': angle,
        'decision_val': decision_val,
        'real_parameter_name': real_parameter_name,
        'imaginary_parameter_name': imaginary_parameter_name,
        'experiment_parameter_names': experiment_parameter_names,
        **exp_metadata}}
    metadata_key = 'bayesian_analysis_metadata'
    return metadata_key, json.dumps(metadata)


def load_json_metadata(dataset):
    try:
        return json.loads(dataset.metadata['bayesian_analysis_metadata'])
    except KeyError:
        raise RuntimeError(
            "'bayesian_analysis_metadata' not found in dataset metadata, "
            "are you sure this is an analysis dataset?")
