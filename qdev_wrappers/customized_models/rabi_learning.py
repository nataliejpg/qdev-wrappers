import qinfer as qi
import numpy as np

class PhenomonlogicalRabiModel(qi.FiniteOutcomeModel):
    # TODO: docstring.

    @property
    def n_modelparams(self):
        return len(self.modelparam_names)

    @property
    def modelparam_names(self):
        return [
            r'omega_rabi', r'phi', 'T1_inv'
        ]

    @property
    def expparams_dtype(self):
        return [('pulse_duration', 'float')]

    def n_outcomes(self, modelparams):
        return 2

    def are_models_valid(self, modelparams):
        return np.all(modelparams >= 0, axis=1)

    def likelihood(self, outcomes, modelparams, expparams):
        # Input shapes:
        #     outcomes:        (n_outcomes,)
        #     modelparams:     (n_models, n_modelparams)
        #     expparams:       (n_experiments, )
        #    
        # Output shapes:    
        #     likelihood:      (n_outcomes, n_models, n_experiments)
        #
        # Intermediate shapes:
        #     w, phi, T1_inv:  (n_models,             1)
        #     t:               (          n_experiments)
        #     visibility:      (n_models, n_experiments)
        w, phi, T1_inv = modelparams.T[:, :, None]
        t = expparams['pulse_duration']

        visibility = np.exp(-t * T1_inv)

        pr0 = np.empty((w.shape[0], t.shape[0]))
        pr0[:, :] = visibility * np.cos(w * t / 2 + phi) ** 2

        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)
