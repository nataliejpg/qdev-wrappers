import numpy as np
from qcodes.utils import validators as vals
from qcodes.instrument.base import Instrument
from qdev_wrappers.fitting.base import strip
from functools import partial


class Converter(Instrument):
    INPUT = None
    OUTPUT = None
    PROPERTIES = {}

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        super().__init__(name)
        if type(self.INPUT) is not list:
            raise RuntimeError('INPUT must be a list')
        if any([type(v) is not dict for v in [self.OUTPUT, self.PROPERTIES]]):
            raise RuntimeError('INPUT and PROPERTIES must be dictionaries')
        for k, v in self.OUTPUT.items():
            self.add_parameter(name=k, **v)
        for k, v in self.PROPERTIES.items():
            self.add_parameter(name=k, set_cmd=partial(self._set_property, k),
                               **v)
            if 'value' in v:
                self.parameters[k](v['value'])

        self.metadata = {'name': name,
                         'input': self.INPUT,
                         'output': strip(self.OUTPUT),
                         'properties': strip(self.PROPERTIES)}

    def _set_property(self, p_name, val):
        self.metadata['properties'][p_name]['value'] = val

    def convert(self, *args, **kwargs):
        raise NotImplementedError(
            'convert function must be implemented in Children')


class Projector(Converter):
    INPUT = ['real_mag', 'imag_phase']
    OUTPUT = {'projected_data': {'label': 'Projected Data',
                                 'unit': 'V'}}
    PROPERTIES = {'angle': {'label': 'Rotation Angle',
                            'unit': 'rad'},
                  'data_type': {'vals': vals.Enum('real_imag', 'mag_phase'),
                                'value': 'real_imag'}}

    def convert(self, real_mag, imag_phase, **kwargs):
        for k, v in kwargs.items():
            self.parameters[k](v)
        if self.data_type() == 'real_imag':
            real = real_mag
            imag = imag_phase
        else:
            real = np.real(real_mag * np.exp(1j * imag_phase))
            imag = np.imag(real_mag * np.exp(1j * imag_phase))
        return real * np.cos(self.angle()) + imag * np.sin(self.angle())


class DecisionMaker(Converter):
    INPUT = ['projected_data']
    OUTPUT = {'qubit_state': {'label': 'Qubit State'}}
    PROPERTIES = {'decision_value': {'label': 'Decision Value',
                                     'value': 0},
                  'decision_direction': {'vals': vals.Enum('normal',
                                                           'inverse'),
                                         'value': 'normal'}}

    def convert(self, projected_data, **kwargs):
        for k, v in kwargs.items():
            self.parameters[k](v)
        if self.decision_direction() is 'normal':
            return (projected_data > self.decision_value).astype(int)
        else:
            return (projected_data > self.decision_value).astype(int)
