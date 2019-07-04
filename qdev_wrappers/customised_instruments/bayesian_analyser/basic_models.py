import qinfer as qi
import numpy as np


class BasicModel(qi.FiniteOutcomeModel):
    def __init__(self, model_parameters, experiment_parameters,
                 likelihood_function, n_outcomes=2):
        self._model_parameters = model_parameters
        self._experiment_parameters = experiment_parameters
        self._likelihood_function = likelihood_function
        self._n_outcomes = n_outcomes
        super().__init__()

    @property
    def n_modelparams(self):
        return len(self._model_parameters)

    @property
    def modelparam_names(self):
        return list(self._model_parameters.keys())

    @property
    def expparams_dtype(self):
        return [(k, 'float') for k in self._experiment_parameters.keys()]

    def n_outcomes(self, modelparams):
        return self._n_outcomes

    def are_models_valid(self, modelparams):
        return np.all(modelparams >= 0, axis=1)

    def likelihood(self, outcomes, modelparams, expparams):
        modelparam_vals = modelparams.T[:, :, None]
        kwargs = dict.zip(self.modelparam_names, modelparam_vals)
        kwargs.update({k: expparams[k]
                       for k in self._experiment_parameters.keys()})
        pr0 = np.empty((m.shape[0] for m in modelparams))
        pr0[:, :] = eval(self._likelihood_function['np'], kwargs)
        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)


class BasicRabiModel(BasicModel):
    def __init__(self, timescale=1e-6):
        model_parameters = {
            r'rabi_freq': {'label': 'Rabi Frequency',
                           'unit': 'Hz',
                           'scaling_value': 1 / timescale,
                           'prior': [0.1, 10]},
            r'excited_pop': {'label': 'Excited State Initial Probability',
                             'prior': [0, 0.5]}
        }
        experiment_parameters = {
            r'pulse_dur': {'label': 'Pulse Duration',
                           'unit': 's',
                           'scaling_value': timescale,}
        }
        likelihood_metadata = {
            'str': r'$f(t)=\cos(\omega_r t + \phi)^2$',
            'np': 'np.cos(rabi_freq * 2 * np.pi * pulse_dur + excited_pop * np.pi / 2)**2'}

        super().__init__(model_parameters, experiment_parameters,
                         likelihood_metadata)


class DecayingRabiModel(BasicModel):
    def __init__(self, timescale=1e-6):
        model_parameters = {
            r'rabi_freq': {'label': 'Rabi Frequency',
                           'unit': 'Hz',
                           'scaling_value': 1 / timescale,
                           'prior': [0.1, 10]},
            r'excited_pop': {'label': 'Excited State Initial Probability',
                             'prior': [0, 0.5]},
            r'T1_inv': {'label': '1 / T1',
                        'scaling_value': 1 / timescale,
                        'prior': [0.0005, 10]}
        }
        experiment_parameters = {
            r'pulse_dur': {'label': 'Pulse Duration',
                           'unit': 's',
                           'scaling_value': timescale},
        }
        likelihood_metadata = {
            'str': r'$f(t)=\cos(\omega_r t + \phi)^2 + (1 - \e^t/T1) / 2$',
            'np': 'np.cos(rabi_freq * 2 * np.pi * pulse_dur + excited_pop * np.pi / 2)**2 + (1 - np.exp(-pulse_dur * T1_inv)) / 2'}

        super().__init__(model_parameters, experiment_parameters,
                         likelihood_metadata)


class DetunedRabiModel(BasicModel):
    def __init__(self, timescale=1e-6):
        model_parameters = {
            r'rabi_freq': {'label': 'Rabi Frequency',
                           'unit': 'Hz',
                           'scaling_value': 1 / timescale,
                           'prior': [0.1, 10]},
            r'excited_pop': {'label': 'Excited State Initial Probability',
                             'prior': [0, 0.5]},
            r'qubit_freq': {'label': 'Qubit Frequency',
                            'unit': 'Hz',
                            'scaling_value': 1 / timescale,
                            'prior': [1e3, 10e3]}
        }
        experiment_parameters = {
            r'pulse_dur': {'label': 'Pulse Duration',
                           'unit': 's',
                           'scaling_value': timescale},
            r'drive_freq': {'label': 'Drive Frequency',
                            'unit': 'Hz',
                            'scaling_value': 1 / timescale}
        }
        likelihood_metadata = {
            'str': r'$f(t)=\cos(\sqrt(\omega_r^2 + \omega_d^2) t + \phi)^2$',
            'np': 'np.cos(np.sqrt(rabi_freq**2 + (qubit_freq - drive_freq)**2) * 2 * np.pi * pulse_dur + excited_pop * np.pi / 2)**2'}

        super().__init__(model_parameters, experiment_parameters,
                         likelihood_metadata)


class DetunedRabiExtModel(BasicModel):
    def __init__(self, timescale=1e-6):
        model_parameters = {
            r'rabi_freq': {'label': 'Rabi Frequency',
                           'unit': 'Hz',
                           'scaling_value': 1 / timescale,
                           'prior': [0.1, 10]},
            r'excited_pop': {'label': 'Excited State Initial Probability',
                             'prior': [0, 0.5]},
            r'qubit_freq_diff': {
                'label': 'Qubit Frequency Difference From Offset',
                'unit': 'Hz',
                'scaling_value': 1 / timescale,
                'prior': [0, 10]}
        }
        experiment_parameters = {
            r'pulse_dur': {'label': 'Pulse Duration',
                           'unit': 's',
                           'scaling_value': timescale},
            r'drive_freq_diff': {
                'label': 'Drive Frequency Difference From Offset',
                'unit': 'Hz',
                'scaling_value': 1 / timescale}
        }
        likelihood_metadata = {
            'str': r'$f(t)=\cos(\sqrt(\omega_r^2 + \omega_d^2) t + \phi)^2$',
            'np': 'np.cos(np.sqrt(rabi_freq**2 + (qubit_freq_diff - drive_freq_diff)**2) * 2 * np.pi * pulse_dur + excited_pop * np.pi / 2)**2'}

        super().__init__(model_parameters, experiment_parameters,
                         likelihood_metadata)


class DecayingDetunedRabiModel(BasicModel):
    def __init__(self, timescale=1e-6):
        model_parameters = {
            r'rabi_freq': {'label': 'Rabi Frequency',
                           'unit': 'Hz',
                           'scaling_value': 1 / timescale,
                           'prior': [0.1, 10]},
            r'excited_pop': {'label': 'Excited State Initial Probability',
                             'prior': [0, 0.5]},
            r'qubit_freq_diff': {
                'label': 'Qubit Frequency Difference From Offset',
                'unit': 'Hz',
                'scaling_value': 1 / timescale,
                'prior': [0, 10]},
            r'T1_inv': {'label': '1 / T1',
                        'scaling_value': 1 / timescale,
                        'prior': [0.0005, 10]}
        }
        experiment_parameters = {
            r'pulse_dur': {'label': 'Pulse Duration',
                           'unit': 's',
                           'scaling_value': timescale},
            r'drive_freq_diff': {
                'label': 'Drive Frequency Difference From Offset',
                'unit': 'Hz',
                'scaling_value': 1 / timescale}
        }
        likelihood_metadata = {
            'str': r'$f(t)=\cos(\sqrt(\omega_r^2 + \omega_d^2) t + \phi)^2 + (1 - \e^t/T1) / 2$',
            'np': 'np.cos(np.sqrt(rabi_freq**2 + (qubit_freq_diff - drive_freq_diff)**2) * 2 * np.pi * pulse_dur + excited_pop * np.pi / 2)**2 + (1 - np.exp(-pulse_dur * T1_inv)) / 2'}

        super().__init__(model_parameters, experiment_parameters,
                         likelihood_metadata)
