import numpy as np
from qdev_wrappers.analysis.base import AnalyserBase


class DecisionMaker(AnalyserBase):
    METHOD = 'DecisionMaking'
    MODEL_PARAMETERS = {'qubit_state': {'label': 'Qubit State',
                                        'parameter_class': 'arr'}}
    MEASUREMENT_PARAMETERS = {'projected_data': {}}
    EXPERIMENT_PARAMETERS = {'decision_value': {},
                             'decision_direction': {}}

    def analyse(self, projected_data,
                decision_value=0, decision_direction='positive'):
        if decision_direction == 'positive':
            res = (np.array(projected_data) > decision_value).astype(int)
        elif decision_direction == 'negative':
            res = (np.array(projected_data) > decision_value).astype(int)
        self.model_parameters.qubit_state.shape = res.shape
        self.model_parameters.qubit_state._val = res
