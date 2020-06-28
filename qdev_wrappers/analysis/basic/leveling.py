import numpy as np
from qdev_wrappers.analysis.base import AnalyserBase

class Leveler(AnalyserBase):
    METHOD = 'Leveling'
    MODEL_PARAMETERS = {'leveled_data': {'label': 'Leveled Data',
                                         'parameter_class': 'arr'}}
    EXPERIMENT_PARAMETERS = {'leveling_axis': {'label': 'Leveling Axis'}}
    MEASUREMENT_PARAMETERS = {'data': {}}

    def analyse(self, **kwargs):
        kwargs = {'leveling_axis': 0, **kwargs}
        leveling_arr = np.mean(np.array(kwargs['leveled_data']),
                               axis=kwargs['leveling_axis'])
        kwargs['leveled_data'] - leveling_arr
        res = np.array(kwargs['real_data']) * np.cos(kwargs['angle']) + \
            np.array(kwargs['imaginary_data']) * np.sin(kwargs['angle'])
        self.model_parameters.projected_data.shape = res.shape
        self.model_parameters.projected_data._val = res

