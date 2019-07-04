import numpy as np
from typing import Optional
from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel
from qinfer import SMCUpdater, UniformDistribution, ProductDistribution, PostselectedDistribution
from qdev_wrappers.customised_instruments.bayesian_analyser.basic_models import BasicModel
from qcodes.utils import validators as vals
from typing import Union, Dict, Optional
from itertools import product
from copy import deepcopy

dtype_val_mapping = {'float': vals.Numbers, 'int': vals.Ints}


class BayesianAnalyser(Instrument):
    def __init__(self, name, model: BasicModel, prior_limits: Optional[dict]=None):
        super().__init__(name)
        self.add_parameter(name='num_particles',
                           initial_value=10000,
                           set_cmd=None,
                           vals=vals.Ints())
        self.add_parameter(name='num_updates',
                           set_cmd=False,
                           initial_value=0)
        self._model = model
        model_parameters = InstrumentChannel(self, 'model_parameters')
        self.add_submodule('model_parameters', model_parameters)
        for paramname, paraminfo in model._model_parameters.items():
            self.model_parameters.add_parameter(
                name=paramname,
                vals=vals.Numbers(),
                label=paraminfo.get('label',
                                    paramname.replace('_', ' ').title()),
                unit=paraminfo.get('unit', None))
            self.model_parameters.add_parameter(
                param + '_variance',
                vals=vals.Numbers(),
                label=paraminfo.get('label',
                                    paramname.replace('_', ' ').title()), + ' Variance',
                unit=paraminfo['unit'] + '^2' if 'unit' in paraminfo else None)
        prior_dict = {k: v.get('prior', [0, 1]) for k, v in
                      model._model_parameters.items}
        if prior_limits is not None:
            prior_dict.update(prior_limits)
        distrs = [UniformDistribution(val) for val in prior_dict.values()]
        self._prior = PostselectedDistribution(ProductDistribution(*distrs))
        metadata = {'method': 'BayesianInferrence',
                    'name': name,
                    'model_parameters': {},
                    'experiment_parameters': {},
                    'likelihood_function': model._likelihood_function}
        for m, v in model._model_parameters.items():
            metadata['model_parameters'][m] = {
                'scaling_value': v.get('scaling_value', 1),
                'prior': prior_dict[m]}
        for e, v in model._experiment_parameters.items():
            metadata['experiment_parameters'][e] = {
                'scaling_value': v.get('scaling_value', 1)}
        self.metadata = metadata
        self.reset_updater()

    @property
    def scaling_values(self):
        scaling_dict = {}
        scaling_dict.update({m: v['scaling_value'] for m in self.metadata['model_parameters']})
        scaling_dict.update({e: v['scaling_value'] for e in self.metadata['experiment_parameters']})
        return scaling_dict

    def update(self, measured_value, **experiment_values):
        self._check_exp_values(**experiment_values)
        expparams = np.empty((len(self._model.expparams_dtype),),
                             dtype=self._model.expparams_dtype)
        for exp_param_name, exp_param_value in experiment_values.items():
            scaled_param_val = exp_param_value / \
                self.scaling_values[exp_param_name]
            expparams[exp_param_name] = scaled_param_val
        self._updater.update(measured_value, expparams)
        model_param_estimates = self._updater.est_mean()
        covariance_matrix = self._updater.est_covariance_mtx()
        for i, model_param_name in enumerate(self.model.modelparam_names):
            model_param = self.model_parameters.parameters[model_param_name]
            model_param_variance = self.model_parameters.parameters[model_param_name + '_variance']
            scaled_estimate = model_param_estimates[i] * \
                self.scaling_values[model_param_name]
            scaled_var_estimate = covariance_matrix[i, i] * self.scaling_values.[model_param_name]**2
            model_param._save_val(scaled_estimate)
            model_param_variance._save_val(scaled_var_estimate)
        self.num_updates._save_val(self.num_updates() + 1)

    def reset_updater(self):
        del self._updater
        self._updater = SMCUpdater(
            self.model, self.num_particles(), deepcopy(self._prior))
        self.num_updates.save_val(0)

    def _check_exp_values(self, **experiment_values):
        if len(experiment_values) != len(self.model.expparams_dtype):
            raise RuntimeError(
                'Must specify a setpoint value for all expparams of the model. '
                'Expected: {} got {}'.format(len(self.model.expparams_dtype),
                                             len(experiment_values)))

    def simulate_experiment(self, **experiment_values):
        self._check_exp_values(**experiment_values)
        n_experiments = 1
        for exp_param_name, exp_param_value in experiment_values.items():
            exp_value_arr = np.array([exp_param_value]).flatten()
            n_experiments *= len(setpoint_value_arr)
            experiment_values[exp_param_name] = setpoint_value_arr
        expparams = np.empty((n_experiments),
                             dtype=self.model.expparams_dtype)
        modelparams = np.empty((1, len(self.model.modelparam_names)))
        exp_value_combinations = product(
            *[v for v in experiment_values.values()])
        exp_param_names = experiment_values.keys()
        for i, combination in enumerate(exp_value_combinations):
            for j, exp_param_name in enumerate(exp_param_names):
                scaled_param_val = combination[j] / \
                    self.scaling_values[exp_param_name]
                expparams[i][exp_param_name] = scaled_param_val
        for i, model_param_name in enumerate(self.model.modelparam_names):
            scaled_param_val = self.model_parameters.parameters[model_param_name] / \
                self.scaling_values[model_param_name]
            modelparams[0, i] = scaled_param_val
        return self.model.simulate_experiment(modelparams, expparams, repeat=1)
