import numpy as np
from qcodes.instrument.base import Instrument
from qinfer import Model, SMCUpdater
from qinfer.distributions import Distribution
from qcodes.utils import validators as vals
from typing import Union

from typing import Optional


dtype_val_mapping = {'float': vals.Numbers, 'int': vals.Ints}


class BayesianAnalyser(Instrument):
    _updater : Optional[SMCUpdater] = None
    def __init__(self, name, model: Model, prior: Distribution,
                 n_particles: int=4000):
        super().__init__(name)
        self.model = model
        self.prior = prior
        self.n_particles = n_particles
        for param in model.modelparam_names:
            self.add_parameter(param,
                               set_cmd=False,
                               get_cmd=None,
                               vals=vals.Numbers())
        for param, dtype in model.expparams_dtype:
            self.add_parameter(param,
                               set_cmd=False,
                               get_cmd=None,
                               vals=dtype_val_mapping[dtype]())

    def start_data_run(self):
        self._updater = SMCUpdater(self.model, self.n_particles, self.prior)

    def update(self, meas: Union[float, int, np.ndarray], **setpoints):
        if self._updater is None:
            self.start_data_run()
        if len(setpoints) != len(self.model.expparams_dtype):
            raise RuntimeError(
                'Must specify setpoint values for all expparams of the model. '
                'Expected: {} got {}'.format(len(self.model.expparams_dtype),
                                             len(setpoints)))
        expparams = np.empty((len(setpoints),),
                             dtype=self.model.expparams_dtype)
        for setpoint_name, setpoint_value in setpoints.items():
            setpoint_param = getattr(self, setpoint_name)
            setpoint_param._save_val(setpoint_value)
            expparams[setpoint_name] = setpoint_value
        self._updater.update(meas, expparams)
        model_param_estimates = self._updater.est_mean()
        for i, param_name in enumerate(self.model.modelparam_names):
            model_param = getattr(self, param_name)
            model_param._save_val(model_param_estimates[i])


