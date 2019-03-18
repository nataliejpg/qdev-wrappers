import qinfer as qi
import numpy as np


class BasicRabiModel(qi.FiniteOutcomeModel):

    @property
    def n_modelparams(self):
        return len(self.modelparam_names)

    @property
    def modelparam_names(self):
        return [
            r'rabi_frequency', r'phi',
        ]

    @property
    def expparams_dtype(self):
        return [('pulse_duration', 'float')]

    def n_outcomes(self, modelparams):
        return 2

    def are_models_valid(self, modelparams):
        return np.all(modelparams >= 0, axis=1)

    def likelihood(self, outcomes, modelparams, expparams):
        rabi_frequency, phi = modelparams.T[:, :, None]
        omega_rabi = np.pi * rabi_frequency
        t = expparams['pulse_duration']
        pr0 = np.empty((rabi_frequency.shape[0], t.shape[0]))
        pr0[:, :] = np.cos(omega_rabi * t + phi) ** 2

        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)
