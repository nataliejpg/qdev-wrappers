import numpy as np
from qcodes.instrument.base import Instrument
from qinfer import Model, SMCUpdater
from qinfer.distributions import Distribution
from qcodes.utils import validators as vals
from typing import Union, Dict

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
        self.model = model
        self.prior = prior
        self.n_particles = n_particles
        for param in model.modelparam_names:
            self.add_parameter(param,
                               set_cmd=None,
                               get_cmd=None,
                               vals=vals.Numbers(),
                               scale=scaling_values.get(param, None))
        for param, dtype in model.expparams_dtype:
            self.add_parameter(param,
                               set_cmd=None,
                               get_cmd=None,
                               vals=dtype_val_mapping[dtype](),
                               scale=scaling_values.get(param, None))
        self.reset_updater(self)

    def update(self, meas: Union[float, int, np.ndarray], **setpoints):
        if len(setpoints) != len(self.model.expparams_dtype):
            raise RuntimeError(
                'Must specify setpoint values for all expparams of the model. '
                'Expected: {} got {}'.format(len(self.model.expparams_dtype),
                                             len(setpoints)))
        expparams = np.empty((len(setpoints),),
                             dtype=self.model.expparams_dtype)
        for setpoint_name, setpoint_value in setpoints.items():
            setpoint_param = getattr(self, setpoint_name)
            setpoint_param(setpoint_value)
            expparams[setpoint_name] = setpoint_param.raw_value
        self._updater.update(meas, expparams)
        model_param_estimates = self._updater.est_mean()
        for i, param_name in enumerate(self.model.modelparam_names):
            model_param = getattr(self, param_name)
            model_param(model_param_estimates[i])

    def reset_updater(self):
        self._updater = SMCUpdater(self.model, self.n_particles, self.prior)
