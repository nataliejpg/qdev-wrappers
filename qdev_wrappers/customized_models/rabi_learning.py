import qinfer as qi
import numpy as np

class RabiModel(qi.FiniteOutcomeModel):
    # TODO: docstring.

    @property
    def n_modelparams(self): return 2

    @property
    def modelparam_names(self): return [r'\omega_{Rabi}', r'\phi']

    @property
    def expparams_dtype(self):
        return [('pulse_duration', 'float')]

    def n_outcomes(self, modelparams):
        return 2

    def are_models_valid(self, modelparams):
        return np.all(modelparams >= 0, axis=1)

    def likelihood(self, outcomes, modelparams, expparams):
        w, phi = modelparams.T[:, :, None]
        t = expparams['pulse_duration']

        visibility = 1 # np.exp(-t * T2_inv)

        pr0 = np.empty((w.shape[0], t.shape[0]))
        pr0[:, :] = visibility * np.cos(w * t / 2 + phi) ** 2 + (1 - visibility) / 2

        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)
