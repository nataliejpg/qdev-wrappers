import qinfer as qi
import numpy as np

## FUNCTIONS ##


def lindblad(H):
    return np.kron(I, H.T) - np.kron(H, I)


def vectorized_rabi_lindblad(ωR, δω):
    # Expand ωR and δω to have indices which broadcast against LX and LZ.
    ωR = ωR[..., None, None]
    δω = δω[..., None, None]

    return ωR * LX + δω * LZ


def vectorized_dissipator(p, Γ):
    p = p[..., None, None]
    Γ = Γ[..., None, None]

    return Γ * (p * diss_p + (1 - p) * diss_m)


def vectorized_generator(ωR, δω, p, Γ):
    return -1j * vectorized_rabi_lindblad(ωR, δω) + vectorized_dissipator(p, Γ)


def vectorized_expm(As):
    """
    Assumes shape As == (..., n).
    """
    # Diagonalize each A.
    #     eigenvalues.shape:    [...,    n]
    #     eigenvectors.shape:   [..., n, n]
    eigenvalues, eigenvectors = np.linalg.eig(As)
    return np.einsum(
        '...ij,...j,...kj->...ik',
        eigenvectors,
        np.exp(eigenvalues),
        eigenvectors.conj()
    )


def sweep_expparams(ts, ωts, model=None):
    if model is None:
        model = RabiGADModel()
    expparams = np.empty((ts.shape[0], ωts.shape[0]), dtype=model.expparams_dtype)
    expparams['t'] = ts[:, None]
    expparams['ωt'] = ωts
    return expparams


def sweep_heuristic(updater, ts, ωts):
    expparams = sweep_expparams(ts, ωts, model=updater.model).flatten()

    def heuristic():
        return expparams[len(updater.data_record) % len(expparams), None]

    return heuristic

## CONSTANTS ##


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1 + 0j, -1 + 0j])

σp = (X + 1j * Y) / 2
σm = (X - 1j * Y) / 2

Ep = np.array([[1, 0], [0, 0]], dtype=complex)
Em = np.array([[0, 0], [0, 1]], dtype=complex)

LX = lindblad(X)
LZ = lindblad(Z)

diss_p = np.kron(σp, σp) - (np.kron(Em, I) + np.kron(I, Em)) / 2
diss_m = np.kron(σm, σm) - (np.kron(Ep, I) + np.kron(I, Ep)) / 2

## MODELS ##


class RabiGADModel(qi.FiniteOutcomeModel):
    @property
    def modelparam_names(self):
        return ['ωR', 'ω0', 'p', 'Γ']

    @property
    def n_modelparams(self):
        return len(self.modelparam_names)

    @property
    def expparams_dtype(self):
        return [('t', 'float'), ('ωt', 'float')]

    def n_outcomes(self, modelparams):
        return 2

    def are_models_valid(self, modelparams):
        return np.ones((modelparams.shape[0], ), dtype=bool)

    def likelihood(self, outcomes, modelparams, expparams):
        super(RabiGADModel, self).likelihood(outcomes, modelparams, expparams)
        ωR, ω0, p, Γ = modelparams.T[:, :, None]
        t = expparams['t']
        ωt = expparams['ωt']

        δω = ωt - ω0

        scaled_generators = t[..., None, None] * vectorized_generator(ωR, δω, p, Γ)
        superoperators = vectorized_expm(scaled_generators)
        pr0 = superoperators[..., 0, 0]

        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)
