import numpy as np
from copy import deepcopy
import warnings
from datetime import datetime
from scipy.optimize import curve_fit
from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel
from qcodes.instrument.parameter import ArrayParameter
from qcodes import validators as vals
from qcodes.instrument.parameter import Parameter

# TODO: remove r2 limit or make it a saveable parameter (nataliejpg)
# TODO: add fitter for >1 independent and dependent variable (nataliejpg)


def strip(d):
    new_d = deepcopy(d)
    for v in new_d.values():
        v.pop('label', None)
        v.pop('unit', None)
        v.pop('vals', None),
        v.pop('initial_value', None)
        v.pop('parameter_class', None)
        v.pop('shape', None)
    return new_d


class AnalysisParameter(Parameter):
    def __str__(self):
        return self.root_instrument.name + '_' + self.name


class ArrayAnalysisParameter(ArrayParameter):
    def _save_array(self, arr):
        # self._latest = {'value': arr, 'ts': datetime.now(),
        #                 'raw_value': arr}
        self.shape = arr.shape
        self._val = arr

    def get_raw(self):
        return self._val

    def __str__(self):
        return self.root_instrument.name + '_' + self.name


class AnalyserBase(Instrument):
    METHOD = None
    MODEL_PARAMETERS = None
    EXPERIMENT_PARAMETERS = None
    MEASUREMENT_PARAMETERS = None
    """
    Instrument which can perform fits on data and has parameters to store the
    fit parameter values. Can be used with the fit_by_id and saved datasets
    generated can be plotted using plot_fit_by_id helpers.
    """

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        super().__init__(name)
        if type(self.METHOD) is not str:
            raise RuntimeError('String description of method must be provided'
                               'in child class of Analyser')
        if any([type(v) is not dict for v in [self.MODEL_PARAMETERS,
                                              self.EXPERIMENT_PARAMETERS,
                                              self.MEASUREMENT_PARAMETERS]]):
            raise RuntimeError('MODEL_PARAMETERS, EXPERIMENT_PARAMETERS and'
                               'MEASUREMENT_PARAMETERS dictionaries must be '
                               'spcified in Analyser subclass')
        self.add_parameter(name='success',
                           set_cmd=False,
                           parameter_class=AnalysisParameter,
                           vals=vals.Enum(0, 1))
        self.success._save_val(0)
        model_parameters = InstrumentChannel(self, 'model_parameters')
        self.add_submodule('model_parameters', model_parameters)
        for paramname, paraminfo in self.MODEL_PARAMETERS.items():
            if paraminfo.get('parameter_class', False):
                paraminfo['parameter_class'] = ArrayAnalysisParameter
                paraminfo['shape'] = (1,)
            self.model_parameters.add_parameter(name=paramname,
                                                **paraminfo)
        self.metadata = {'name': name,
                         'method': self.METHOD,
                         'model_parameters': strip(self.MODEL_PARAMETERS),
                         'experiment_parameters': strip(self.EXPERIMENT_PARAMETERS),
                         'measurement_parameters': strip(self.MEASUREMENT_PARAMETERS)}

    @property
    def all_parameters(self):
        """
        Gathers all parameters on instrument and on it's submodules
        and returns them in a list.
        """
        params = []
        params += [p for n, p in self.parameters.items() if n != 'IDN']
        for s in self.submodules.values():
            params += list(s.parameters.values())
        return params

    def _check_experiment_parameters(self, **experiment_parm_values):
        if set(experiment_parm_values.keys()) != set(self.EXPERIMENT_PARAMETERS.keys()):
            raise RuntimeError(
                'All experiment parameters must be specified.'
                ' Expected: {} got {}'.format(self.EXPERIMENT_PARAMETERS.keys(),
                                              experiment_parm_values.keys()))
        vals = list(experiment_parm_values.values())
        for a in vals[1:]:
            if a.shape != vals[0].shape:
                raise RuntimeError(
                    f"experiment_parameter data does not have the same shape "
                    f"as measured data.\nMeasured data shape: {vals[0].shape},"
                    f"experiment_parameter data shape: {a.shape}")

    def _check_measurement_parameters(self, shape=None, **measured_parm_vals):
        if set(measured_parm_vals.keys()) != set(self.MEASUREMENT_PARAMETERS.keys()):
            raise RuntimeError(
                'All measurement parameters must be specified.'
                ' Expected: {} got {}'.format(self.MEASUREMENT_PARAMETERS.keys(),
                                              measured_parm_vals.keys()))
        if shape is None:
            shape = np.array(list(measured_parm_vals.values())[0]).shape
        for a in measured_parm_vals.values():
            if a.shape != shape:
                raise RuntimeError(
                    f"measurement_parameter data does not expected shape. "
                    f"\nMeasured data shape: Expected: {shape},"
                    f"got {a.shape}")

    def analyse(self, *args, **kwargs):
        """
        Function which is expected to take data and any other information,
        perform the fit or analysis and (minimally) update the model_parameters
        and success parameter.
        """
        raise NotImplementedError(
            'Analyse function must be implemented in Children')
