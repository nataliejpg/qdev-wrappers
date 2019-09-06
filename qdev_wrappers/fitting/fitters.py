from qdev_wrappers.analysis.base import Analyser
from qdev_wrappers.analysis.least_squares_base import LeastSquaresBase
from qdev_wrappers.fitting import guess
import numpy as np

# TODO: add fitter for resonators (nataliejpg)


class MinMaxFitterNd(Analyser):
    METHOD = 'ExhaustiveSearch'
    MODEL = {'str': None}
    MODEL_PARAMETERS = {'value': {}}
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
        self.fit_parameters.value._save_val(measured[index])
        if self._dim == 1:
            exp_vals = np.array(list(experiment.values())[0])
            self.fit_parameters.location._save_val(exp_vals[index])
        else:
            exp_vals = np.array(list(experiment.values()))
            self.fit_parameters.location_1._save_val(exp_vals[0][index[0]])
            self.fit_parameters.location_2._save_val(exp_vals[1][index[1]])
        self.success._save_val(1)


class ExpDecayFitter(LeastSquaresBase):
    MODELPARAMETERS = {'a': {'label': '$a$'},
                       'T': {'label': '$T$', 'unit': 's'},
                       'c': {'label': '$c$'}}
    MODEL = {'str': r'$f(x) = a \exp(-x/T) + c$',
             'np': 'a * np.exp(-x / T) + c'}
    GUESS = guess.exp_decay
    """
    Least Squares Fitter which fits to an exponential decay using the equation
        a * np.exp(-x / T) + c
    given the measured results and the values of x. Useful for T1 fitting.
    """


class ExpDecayBaseFitter(LeastSquaresBase):
    MODELPARAMETERS = {'a': {'label': '$a$', 'unit': 'V'},
                       'p': {'label': '$p$'},
                       'b': {'label': '$b$', 'unit': 'V'}}
    MODEL = {'str': r'$f(x) = A p^x + B$',
             'np': 'a * p**x + b'}
    GUESS = guess.power_decay
    """
    Least Squares Fitter which fits to an exponential using the equation
        a * p**x + b
    and given the measured results and the values of x. Useful for fitting
    benchmarking results.
    """


class CosFitter(LeastSquaresBase):
    MODELPARAMETERS = {'a': {'label': '$a$'},
                       'w': {'label': r'$\omega$', 'unit': 'Hz'},
                       'p': {'label': r'$\phi$'},
                       'c': {'label': '$c$', 'unit': ''}}
    MODEL = {'str': r'$f(x) = a\cos(\omega x + \phi)+c$',
             'np': 'a * np.cos(w * x + p) + c'}
    GUESS = guess.cosine
    """
    Least Squares Fitter which fits to a cosine using the equation
        a * np.cos(w * x + p) + c
    and given the measured results and the values of x. Useful for fitting
    Rabi oscillations.
    """


class ExpDecaySinFitter(LeastSquaresBase):
    MODELPARAMETERS = {'a': {'label': '$a$', 'unit': ''},
                       'T': {'label': '$T$', 'unit': 's'},
                       'w': {'label': r'$\omega$', 'unit': 'Hz'},
                       'p': {'label': r'$\phi$'},
                       'c': {'label': '$c$'}}
    MODEL = {
        'str': r'$f(x) = a \exp(-x / T) \sin(\omega x + \phi) + c$',
        'np': 'a * np.exp(-x / T) * np.sin(w * x + p) + c'}
    GUESS = guess.exp_decay_sin
    """
    Least Squares Fitter which fits to an exponentially decaying sine using the
    equation
        a * np.exp(-x / T) * np.sin(w * x + p) + c
    and given the measured results and the values of x. Useful for fitting
    Ramsey oscillations to find T2*.
    """
