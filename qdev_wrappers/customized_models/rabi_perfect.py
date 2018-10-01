import qinfer as qi
import numpy as np


class PerfectRabiModel(qi.FiniteOutcomeModel):
    # TODO: docstring.

    @property
    def n_modelparams(self):
        return len(self.modelparam_names)

    @property
    def modelparam_names(self):
        return [
            r'rabi_frequency', r'qubit_frequency', r'initial_0_probability'
        ]

    @property
    def expparams_dtype(self):
        return [('pulse_duration', 'float'), ('drive_frequency', 'float')]

    def n_outcomes(self, modelparams):
        return 2

    def are_models_valid(self, modelparams):
        return np.all(modelparams >= 0, axis=1)

    def likelihood(self, outcomes, modelparams, expparams):
        rabi_frequency, qubit_frequency, initial_0_probability = modelparams.T[:, :, None]
        omega_qubit = 2 * np.pi * qubit_frequency
        omega_rabi = 2 * np.pi * rabi_frequency
        detuning = 2 * np.pi * (qubit_frequency - expparams['drive_frequency'])
        initial_1_probability = 1 - initial_0_probability
        initial_0_amp = np.sqrt(initial_0_probability)
        initial_1_amp = np.sqrt(initial_1_probability)
        omega = np.sqrt(qubit_frequency**2 + detuning ** 2)
        t = expparams['pulse_duration']
        pr0 = np.empty((qubit_frequency.shape[0], t.shape[0]))
        pr0[:, :] = (np.cos(omega * t / 2)**2 + (detuning * np.sin(omega * t / 2) / omega)**2) * initial_0_amp + \
                    (omega_rabi * np.sin(omega * t / 2) / omega)**2 * initial_1_amp

        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)
