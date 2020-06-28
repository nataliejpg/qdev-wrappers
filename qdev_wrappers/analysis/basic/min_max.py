import numpy as np
from qdev_wrappers.analysis.base import AnalyserBase


class MinMaxFitterNd(AnalyserBase):
    METHOD = 'ExhaustiveSearch'
    MODEL_PARAMETERS = {'value': {'parameter_class': 'arr'},
                        'location': {'parameter_class': 'arr'}}
    EXPERIMENT_PARAMETERS = {'x': {}}
    MEASUREMENT_PARAMETERS = {'y': {}}
    """
    Fitter instrument which finds the minimum of a 1d array and
    returns the location and value of this point
    """

    def __init__(self, minimum=True, dimension=1):
        self._min = minimum
        self._dim = dimension
        if minimum:
            text = 'Minimum'
        else:
            text = 'Maximum'
        self.MODEL_PARAMETERS['location']['label'] = 'Location of {}'.format(
            text)
        self.MODEL_PARAMETERS['value']['label'] = 'Value at {}'.format(text)
        self.MODEL['str'] = 'Find {} in {}d'.format(text, dimension)
        if dimension == 1:
            self.MODEL_PARAMETERS['location'] = {'label': 'Minimum Location'}
            self.EXPERIMENT_PARAMETERS['x']['label'] = 'Search Axis'
        if dimension == 2:
            self.MODEL_PARAMETERS['location_1'] = {
                'label': 'Minimum Location in Axis_1'}
            self.MODEL_PARAMETERS['location_2'] = {
                'label': 'Minimum Location in Axis_2'}
            self.EXPERIMENT_PARAMETERS['x']['label'] = 'Search Axis_1'
            self.EXPERIMENT_PARAMETERS['xx'] = {'label': 'Search Axis_2'}
        else:
            raise RuntimeError('More than 2 dimensions not supported')
        super().__init__(name='{}Fitter'.format(text))

    def analyse(self, measured, **experiment):
        """
        Find minimum value and location, update fit_parameters and
        save succcess as yay.
        """
        self._check_experiment_parameters(**experiment)
        shape = np.array(list(experiment.values()))[0].shape
        self._check_measurement_parametrs(y=measured, shape=shape)
        if self._min:
            index = np.argmin(measured)
        else:
            index = np.argmax(measured)
        self.model_parameters.value.shape = measured[index].shape
        self.model_parameters.location.shape = measured[index].shape
        self.model_parameters.value._val = measured[index]
        if self._dim == 1:
            loc_vals = np.array(list(experiment.values())[0])
            self.model_parameters.location._val = (loc_vals[index])
        else:
            loc_vals = np.array(list(experiment.values()))
            self.model_parameters.location_1._val = (loc_vals[0][index[0]])
            self.model_parameters.location_2._val = (loc_vals[1][index[1]])
        self.success._save_val(1)
