import qinfer as qi
import numpy as np


class DecayingRabiModel(qi.FiniteOutcomeModel):

    @property
    def n_modelparams(self):
        return len(self.modelparam_names)

    @property
    def modelparam_names(self):
        return [
            r'rabi_frequency', r'qubit_frequency', r'phi', r'T1_inv'
        ]

    @property
    def expparams_dtype(self):
        return [('pulse_duration', 'float'), ('drive_frequency', 'float')]

    def n_outcomes(self, modelparams):
        return 2

    def are_models_valid(self, modelparams):
        return np.all(modelparams >= 0, axis=1)

    def likelihood(self, outcomes, modelparams, expparams):
        rabi_frequency, qubit_frequency, phi, T1_inv = modelparams.T[:, :, None]
        omega_rabi = np.pi * rabi_frequency
        omega_detuning = np.pi * (qubit_frequency - expparams['drive_frequency'])
        t = expparams['pulse_duration']
        # omega = np.sqrt(omega_detuning ** 2 + omega_rabi ** 2)
        omega = omega_rabi
        visibility = np.exp(-t * T1_inv)
        pr0 = np.empty((rabi_frequency.shape[0], t.shape[0]))
        pr0[:, :] = visibility * np.cos(omega * t + phi) ** 2 + (1 - visibility) / 2

        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)
