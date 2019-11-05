from qdev_wrappers.analysis.bayesian_inference.base import BayesianAnalyserBase


def build_rabi_analyser(name, timescale=1e-6, bad_initialisation=False,
                        decaying=False, detuned=False):
    model_parameters = {r'rabi_freq': {'label': 'Rabi Frequency',
                                       'unit': 'Hz',
                                       'scaling_value': 1 / timescale,
                                       'prior': [5, 10]}
                        }
    experiment_parameters = {r'pulse_dur': {'label': 'Pulse Duration',
                                            'unit': 's',
                                            'scaling_value': timescale}
                             }
    if bad_initialisation:
        model_parameters[r'excited_pop'] = {
            'label': 'Excited State Initial Probability',
            'prior': [0, 0.5]}
    if decaying:
        model_parameters[r'T1_inv'] = {
            'label': '1 / T1',
            'scaling_value': 1 / timescale,
            'prior': [0, 1]}
    if detuned:
        model_parameters[r'qubit_freq'] = {
            'label': 'Qubit Frequency',
            'unit': 'Hz',
            'scaling_value': 1 / timescale,
            'prior': [1e3, 10e3]}
        experiment_parameters[r'drive_freq'] = {
            'label': 'Drive Frequency',
            'unit': 'Hz',
            'scaling_value': 1 / timescale}

    if not (bad_initialisation or decaying or detuned):
        model = {
            'str': r'$f(t)=\cos(\omega_r t)^2$',
            'np': 'np.cos(rabi_freq * 2 * np.pi * pulse_dur)**2'}
    elif bad_initialisation and not (decaying or detuned):
        model = {
            'str': r'$f(t)=\cos(\omega_r t + \phi)^2$',
            'np': 'np.cos(rabi_freq * 2 * np.pi * '
                  'pulse_dur + np.arccos(np.sqrt((1 - excited_pop))))**2'}
    elif decaying and not (bad_initialisation or detuned):
        model = {
            'str': r'$f(t)=\cos(\omega_r t)^2 + (1 - \e^t/T1) / 2$',
            'np': '0.5 * (1 + np.exp(-pulse_dur * T1_inv) * '
                  'np.cos(rabi_freq * 4 * np.pi * pulse_dur))'}
    elif detuned and not (bad_initialisation or decaying):
        model = {
            'str': r'$f(t)=\cos(\sqrt(\omega_r^2 + \omega_d^2) t)^2$',
            'np': 'np.cos(np.sqrt(rabi_freq**2 + '
                  '(qubit_freq - drive_freq)**2) * 2 * np.pi * '
                  'pulse_dur)**2'}
    elif (bad_initialisation and decaying) and not detuned:
        model = {
            'str': r'$f(t)=\cos(\omega_r t + \phi)^2 + (1 - \e^t/T1) / 2$',
            'np': '0.5 * (1 + np.exp(-pulse_dur * T1_inv) * ('
                  'np.cos(rabi_freq * 4 * np.pi * pulse_dur + '
                  'np.arccos(np.sqrt((1 - excited_pop))))))'}
    elif (bad_initialisation and detuned) and not decaying:
        model = {
            'str': r'$f(t)=\cos(\sqrt(\omega_r^2 + \omega_d^2) t + '
                   r'\phi)^2$',
            'np': 'np.cos(np.sqrt(rabi_freq**2 + (qubit_freq - '
                  'drive_freq)**2) * 2 * np.pi * pulse_dur + '
                  'np.arccos(np.sqrt((1 - excited_pop))))**2'}
    elif (detuned and decaying) and not bad_initialisation:
        model = {
            'str': r'$f(t)=\cos(\sqrt(\omega_r^2 + \omega_d^2) t)^2 + '
                   r'(1 - \e^t/T1) / 2$',
            'np': '0.5 * (1 + np.exp(-pulse_dur * T1_inv) *'
                  'np.cos(np.sqrt(rabi_freq**2 + (qubit_freq - '
                  'drive_freq)) * 4 * np.pi * pulse_dur))'}
    else:
        model = {
            'str': r'$f(t)=\cos(\sqrt(\omega_r^2 + \omega_d^2) t + \phi)^2'
                   r' + (1 - \e^t/T1) / 2$',
            'np': '0.5 * (1 + np.exp(-pulse_dur * T1_inv) * ('
                  'np.cos(np.sqrt(rabi_freq**2 + (qubit_freq '
                  '- drive_freq)**2) * 4 * np.pi * pulse_dur + '
                  'np.arccos(np.sqrt((1 - excited_pop))))))'}

    class RabiBayesianAnalyser(BayesianAnalyserBase):
        MODEL_PARAMETERS = model_parameters
        EXPERIMENT_PARAMETERS = experiment_parameters
        MODEL = model

    return RabiBayesianAnalyser(name=name)
