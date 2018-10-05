import numpy as np
from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel
from qinfer import Model, SMCUpdater
from qinfer.distributions import Distribution
from qcodes.utils import validators as vals
from typing import Union, Dict
from itertools import product
from typing import Optional


dtype_val_mapping = {'float': vals.Numbers, 'int': vals.Ints}

class BayesianAnalyser(Instrument):
    _updater : Optional[SMCUpdater] = None
    def __init__(self, name, model: Model, prior: Distribution,
                 n_particles: int=4000,
                 scaling_values=Optional[Dict[str, float]]):
        super().__init__(name)
        if scaling_values is None:
            scaling_values = {}
        self.model = model # TODO: should be parameter
        self.prior = prior # TODO: should be parameter
        self.n_particles = n_particles # TODO: should be parameter
        model_parameters = InstrumentChannel(self, 'model_parameters')
        self.add_submodule('model_parameters', model_parameters)
        experiment_parameters = InstrumentChannel(self, 'experiment_parameters')
        self.add_submodule('experiment_parameters', experiment_parameters)
        for param in model.modelparam_names:
            self.model_parameters.add_parameter(param,
                                                get_cmd=None,
                                                vals=vals.Numbers(),
                                                scale=scaling_values.get(param, 1))
            self.model_parameters.add_parameter(param + '_variance',
                                                get_cmd=None,
                                                vals=vals.Numbers(),
                                                scale=scaling_values.get(param, 1) ** 2,)
        for param, dtype in model.expparams_dtype:
            self.experiment_parameters.add_parameter(param,
                                                     set_cmd=None,
                                                     get_cmd=None,
                                                     vals=dtype_val_mapping[dtype](),
                                                     scale=scaling_values.get(param, 1))
        self.reset_updater()

    def update(self, meas: Union[float, int, np.ndarray]):
        setpoints = {param_name: param for param_name, param in self.experiment_parameters.parameters.items()}
        expparams = np.empty((len(setpoints),),
                             dtype=self.model.expparams_dtype)
        for setpoint_param_name, setpoint_param in setpoints.items():
            expparams[setpoint_param_name] = setpoint_param._latest['raw_value']
        self._updater.update(meas, expparams)
        model_param_estimates = self._updater.est_mean()
        covariance_matrix = self._updater.est_covariance_mtx()
        for i, model_param_name in enumerate(self.model.modelparam_names):
            model_param = self.model_parameters.parameters[model_param_name]
            model_param_variance = self.model_parameters.parameters[model_param_name + '_variance']
            model_param._latest['raw_value'] = model_param_estimates[i]
            model_param_variance._latest['raw_value'] = covariance_matrix[i, i]


    def reset_updater(self):
        self._updater = SMCUpdater(self.model, self.n_particles, self.prior)

    def simulate_experiment(self, **setpoints):
        if len(setpoints) != len(self.model.expparams_dtype):
            raise RuntimeError(
                'Must specify a setpoint value for all expparams of the model. '
                'Expected: {} got {}'.format(len(self.model.expparams_dtype),
                                             len(setpoints)))
        n_experiments = 1
        for setpoint_name, setpoint_value in setpoints.items():
            setpoint_value_arr = np.array([setpoint_value]).flatten() 
            n_experiments *= len(setpoint_value_arr)
            setpoints[setpoint_name] = setpoint_value_arr
        expparams = np.empty((n_experiments),
                             dtype=self.model.expparams_dtype)
        modelparams = np.empty((1, len(self.model.modelparam_names)))
        setpoint_combinations = product(*[v for v in setpoints.values()])
        setpoint_names =  setpoints.keys()
        for i, setpoint_combination in enumerate(setpoint_combinations):
            for j, setpoint_name in enumerate(setpoint_names):
                setpoint_param = self.experiment_parameters.parameters.get(setpoint_name)
                setpoint_param(setpoint_combination[j])
                expparams[i][setpoint_name] = setpoint_param._latest['raw_value']
        for i, model_param_name in enumerate(self.model.modelparam_names):
            modelparams[0, i] = self.model_parameters.parameters.get(model_param_name)._latest['raw_value']
        return self.model.simulate_experiment(modelparams, expparams, repeat=1)
