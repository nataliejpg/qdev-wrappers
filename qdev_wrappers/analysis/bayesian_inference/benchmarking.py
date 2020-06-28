from qdev_wrappers.analysis.bayesian_inference.base import BayesianAnalyserBase


class BenchmarkingBayesianAnalyser(BayesianAnalyserBase):
    MODEL_PARAMETERS = {'A': {'prior': [0, 1]},
                        'p': {'label': 'Clifford Gate Fidelity',
                              'prior': [0, 1]},
                        'B': {'prior': [0, 1]}}
    EXPERIMENT_PARAMETERS = {'x': {'label': 'Clifford Gate Count'}}
    MODEL = {'str': r'$f(x) = A p^x + B$',
             'np': 'A * p**x + B'}
