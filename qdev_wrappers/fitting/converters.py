import numpy as np
from qcodes.utils import validators as vals
from qcodes.instrument.base import Instrument


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
        if any([type(v) is not dict for v in [self.INPUT, self.PROPERTIES]]):
            raise RuntimeError('INPUT and PROPERTIES must be dictionaries')
        for k, v in self.INPUT.items():
            self.add_parameter(name=k, **v)
        for k, v in self.OUTPUT.items():
            self.add_parameter(name=k, **v)
        for k, v in self.PROPERTIES.items():
            self.add_parameter(name=k, **v)

    @property
    def metadata(self):
        m = {'name': self.name,
             'input': self.INPUT,
             'output': self.OUTPUT,
             'properties': self.PROPERTIES}
        for p in self.PROPERTIES.keys():
            m['properties']['p']['value'] = self.properties[p]()
        return m

    def convert(self, *args, **kwargs):
        raise RuntimeError('Implemented in children')


class Projector(Converter):
    INPUT = ['real_mag', 'imag_phase']
    OUTPUT = {'projected_data': {'label': 'Projected Data',
                                 'unit': 'V'}}
    PROPERTIES = {'angle': {'label': 'Rotation Angle',
                            'unit': 'rad'},
                  'data_type': {'vals': vals.Enum('real_imag', 'mag_phase'),
                                'initial_value': 'real_imag'}}

    def convert(self, real_mag, imag_phase, angle=0, data_type=None):
        if data_type is not None:
            self.data_type(data_type)
        if angle is not None:
            self.angle(angle)
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
                                     'initial_value': 0},
                  'decision_direction': {'vals': vals.Enum('normal',
                                                           'inverse'),
                                         'initial_value': 'normal'}}

    def convert(self, projected_data, decision_value=None,
                decision_direction=None):
        if decision_value is not None:
            self.decision_value(decision_value)
        if decision_direction is not None:
            self.decision_direction(decision_direction)
        if self.decision_direction() is 'normal':
            return (projected_data > self.decision_value).astype(int)
        else:
            return (projected_data > self.decision_value).astype(int)
