import numpy as np
from qdev_wrappers.analysis.base import AnalyserBase


class RealImagProjector(AnalyserBase):
    METHOD = 'Projection'
    MODEL_PARAMETERS = {'projected_data': {'label': 'Projected Data',
                                           'unit': 'V',
                                           'parameter_class': 'arr'}}
    EXPERIMENT_PARAMETERS = {'angle': {'label': 'Rotation Angle',
                                       'unit': 'radians'}}
    MEASUREMENT_PARAMETERS = {'real_data': {},
                              'imaginary_data': {}}

    def analyse(self, **kwargs):
        kwargs = {'angle': 0, **kwargs}
        res = np.array(kwargs['real_data']) * np.cos(kwargs['angle']) + \
            np.array(kwargs['imaginary_data']) * np.sin(kwargs['angle'])
        self.model_parameters.projected_data.shape = res.shape
        self.model_parameters.projected_data._val = res


class MagPhaseProjector(AnalyserBase):
    METHOD = 'Projection'
    MODEL_PARAMETERS = {'projected_data': {'label': 'Projected Data',
                                           'unit': 'V',
                                           'parameter_class': 'arr'}}
    EXPERIMENT_PARAMETERS = {'angle': {'label': 'Rotation Angle',
                                       'unit': 'radians'}}
    MEASUREMENT_PARAMETERS = {'magnitude_data': {},
                              'phase_data': {}}

    def analyse(self, **kwargs):
        kwargs = {'angle': 0, **kwargs}
        real = np.real(np.array(kwargs['magnitude_data']) * np.exp(1j * np.array(kwargs['phase_data'])))
        imag = np.imag(np.array(kwargs['magnitude_data']) * np.exp(1j * np.array(kwargs['phase_data'])))
        res = real * np.cos(kwargs['angle']) + imag * np.sin(kwargs['angle'])
        self.model_parameters.projected_data.shape = res.shape
        self.model_parameters.projected_data._val = res


class ComplexProjector(AnalyserBase):
    METHOD = 'Projection'
    MODEL_PARAMETERS = {'projected_data': {'label': 'Projected Data',
                                           'unit': 'V',
                                           'parameter_class': 'arr'}}
    EXPERIMENT_PARAMETERS = {'angle': {'label': 'Rotation Angle',
                                       'unit': 'radians'}}
    MEASUREMENT_PARAMETERS = {'complex_data': {}}

    def analyse(self, **kwargs):
        kwargs = {'angle': 0, **kwargs}
        real = np.real(np.array(kwargs['complex_data']))
        imag = np.imag(np.array(kwargs['complex_data']))
        res = real * np.cos(kwargs['angle']) + imag * np.sin(kwargs['angle'])
        self.model_parameters.projected_data.shape = res.shape
        self.model_parameters.projected_data._val = res
