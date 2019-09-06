from qdev_wrappers.bayesian_analysis.base import BayesianAnalyserBase


class RabiAnalyser(BayesianAnalyserBase):
    MODEL_PARAMETERS = {r'rabi_freq': {'label': 'Rabi Frequency',
                                       'unit': 'Hz',
                                       'prior': [0.1, 10]}
                        }
    EXPERIMENT_PARAMETERS = {r'pulse_dur': {'label': 'Pulse Duration',
                                            'unit': 's'}
                             }

    def __init__(self, timescale=1e-6, bad_initialisation=False,
                 decaying=False, detuned=False):
        self.MODEL_PARAMETERS['scaling_value'] = 1 / timescale
        self.EXPERIMENT_PARAMETERS['scaling_value'] = 1 / timescale
        if bad_initialisation:
            self.MODEL_PARAMETERS[r'excited_pop'] = {
                'label': 'Excited State Initial Probability',
                'prior': [0, 0.5]}
        if decaying:
            self.MODEL_PARAMETERS[r'T1_inv'] = {
                'label': '1 / T1',
                'scaling_value': 1 / timescale,
                'prior': [0.0005, 10]}
        if detuned:
            self.MODEL_PARAMETERS[r'qubit_freq'] = {
                'label': 'Qubit Frequency',
                'unit': 'Hz',
                'scaling_value': 1 / timescale,
                'prior': [1e3, 10e3]}
            self.EXPERIMENT_PARAMETERS[r'drive_freq'] = {
                'label': 'Drive Frequency',
                'unit': 'Hz',
                'scaling_value': 1 / timescale}

        if not (bad_initialisation or decaying or detuned):
            self.MODEL = {
                'str': r'$f(t)=\cos(\omega_r t)^2$',
                'np': 'np.cos(rabi_freq * 2 * np.pi * pulse_dur)**2'}
        elif bad_initialisation and not (decaying or detuned):
            self.MODEL = {
                'str': r'$f(t)=\cos(\omega_r t + \phi)^2$',
                'np': 'np.cos(rabi_freq * 2 * np.pi * '
                      'pulse_dur + np.arcos(np.sqrt(excited_pop)))**2'}
        elif decaying and not (bad_initialisation or detuned):
            self.MODEL = {
                'str': r'$f(t)=\cos(\omega_r t)^2 + (1 - \e^t/T1) / 2$',
                'np': 'np.cos(rabi_freq * 2 * np.pi * pulse_dur)**2 + '
                      '(1 - np.exp(-pulse_dur * T1_inv)) / 2'}
        elif detuned and not (bad_initialisation or decaying):
            self.MODEL = {
                'str': r'$f(t)=\cos(\sqrt(\omega_r^2 + \omega_d^2) t)^2$',
                'np': 'np.cos(np.sqrt(rabi_freq**2 + '
                      '(qubit_freq - drive_freq)**2) * 2 * np.pi * '
                      'pulse_dur)**2'}
        elif (bad_initialisation and decaying) and not detuned:
            self.MODEL = {
                'str': r'$f(t)=\cos(\omega_r t + \phi)^2 + (1 - \e^t/T1) / 2$',
                'np': 'np.cos(rabi_freq * 2 * np.pi * pulse_dur + '
                      'np.arcos(np.sqrt(excited_pop)))**2 + (1 - '
                      'np.exp(-pulse_dur * T1_inv)) / 2'}
        elif (bad_initialisation and detuned) and not decaying:
            self.MODEL = {
                'str': r'$f(t)=\cos(\sqrt(\omega_r^2 + \omega_d^2) t + '
                       r'\phi)^2$',
                'np': 'np.cos(np.sqrt(rabi_freq**2 + (qubit_freq - '
                      'drive_freq)**2) * 2 * np.pi * pulse_dur + '
                      'np.arcos(np.sqrt(excited_pop)))**2'}
        elif (detuned and decaying) and not bad_initialisation:
            self.MODEL = {
                'str': r'$f(t)=\cos(\sqrt(\omega_r^2 + \omega_d^2) t)^2 + '
                       r'(1 - \e^t/T1) / 2$',
                'np': 'np.cos(np.sqrt(rabi_freq**2 + (qubit_freq_diff - '
                      'drive_freq_diff)**2) * 2 * np.pi * pulse_dur)**2 + '
                      '(1 - np.exp(-pulse_dur * T1_inv)) / 2'}
        else:
            self.MODEL = {
                'str': r'$f(t)=\cos(\sqrt(\omega_r^2 + \omega_d^2) t + \phi)^2'
                       r' + (1 - \e^t/T1) / 2$',
                'np': 'np.cos(np.sqrt(rabi_freq**2 + (qubit_freq_diff '
                      '- drive_freq_diff)**2) * 2 * np.pi * pulse_dur + '
                      'np.arcos(np.sqrt(excited_pop)))**2 + (1 - '
                      'np.exp(-pulse_dur * T1_inv)) / 2'}
        super().__init__()
