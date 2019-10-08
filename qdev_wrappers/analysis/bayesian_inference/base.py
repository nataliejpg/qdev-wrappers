import numpy as np
from typing import Optional
from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel
from qinfer import SMCUpdater, UniformDistribution, ProductDistribution, PostselectedDistribution
from qcodes.utils import validators as vals
from typing import Union, Dict, Optional
from itertools import product
from copy import deepcopy
from functools import partial
from qcodes.instrument.parameter import ParameterWithSetpoints, Parameter
import qinfer as qi
from qdev_wrappers.analysis.base import AnalyserBase, AnalysisParameter
import matplotlib.pyplot as plt
from matplotlib import gridspec
# from qcodes.dataset.plotting import _rescale_ticks_and_units, plot_on_a_plain_grid

# TODO: allow other measurement_parameters


class BAModel(qi.FiniteOutcomeModel):
    def __init__(self, modelparam_names,
                 experimentparam_names, likelihood_fn):
        self._n_modelparams = len(modelparam_names)
        self._modelparam_names = modelparam_names
        self._experimentparam_names = experimentparam_names
        self._likelihood_fn = likelihood_fn

    @property
    def n_modelparams(self):
        return self._n_modelparams

    @property
    def is_n_outcomes_constant(self):
        return True

    @property
    def modelparam_names(self):
        return self._modelparam_names

    @property
    def expparams_dtype(self):
        return [(k, 'float') for k in self._experimentparam_names]

    def n_outcomes(self, modelparams):
        return 2

    def are_models_valid(self, modelparams):
        return np.all(modelparams >= 0, axis=1)

    def likelihood(self, outcomes, modelparams, expparams):
        # modelparams shape (n_models, n_modelparameters)
        # expparams shape (n_experiments, n_expparameters)
        # (n_modelparameters, n_models, 1)
        modelparam_vals = modelparams.T[:, :, np.newaxis]
        expparam_vals = expparams.T  # (n_expparameters, n_experiments)
        # (n_models, 1)
        kwargs = dict(zip(self.modelparam_names, modelparam_vals))
        kwargs.update({k: expparams[k]
                       for k in self._experimentparam_names})  # (n_experiments)
        kwargs['np'] = np
        # (n_models, n_experiments)
        pr0 = np.empty((modelparams.shape[0], expparams.shape[0]))
        pr0[:, :] = eval(self._likelihood_fn, kwargs)
        return self.pr0_to_likelihood_array(outcomes, pr0)


class PosteriorMarginalSetpointsParameter(Parameter):
    def __init__(self, m_idx, scaling_val, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index = m_idx
        self._scaling_val = scaling_val

    def __str__(self):
        return self.root_instrument.name + '_' + self.name

    def get_raw(self):
        res = self.root_instrument._updater.posterior_marginal(idx_param=self._index)
        return np.array(res[0]) * self._scaling_val


class PosteriorMarginalParameter(ParameterWithSetpoints):
    def __init__(self, m_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index = m_idx

    def __str__(self):
        return self.root_instrument.name + '_' + self.name

    def get_raw(self):
        res = self.root_instrument._updater.posterior_marginal(idx_param=self._index)
        return np.array(res[1])


class BayesianAnalyserBase(AnalyserBase):
    METHOD = 'BayesianInferrence'
    MEASUREMENT_PARAMETERS = {'qubit_state': {'label': 'Qubit State'}}
    MODEL = None

    def __init__(self, name: str=None):
        if not isinstance(self.MODEL, dict):
            raise RuntimeError('MODEL must be dict')
        super().__init__(name=name)
        self.metadata['model'] = self.MODEL
        self._model = BAModel(modelparam_names=self.MODEL_PARAMETERS.keys(),
                              experimentparam_names=self.EXPERIMENT_PARAMETERS.keys(),
                              likelihood_fn=self.MODEL['np'])
        for k, v in self.MODEL_PARAMETERS.items():
            v['prior'] = v.get('prior', [0, 1])
            v['scaling_value'] = v.get('scaling_value', 1)
            v['label'] = v.get('label', k.replace('_', ' ').title())
            v['unit'] = v.get('label', None)
        for k, v in self.EXPERIMENT_PARAMETERS.items():
            v['scaling_value'] = v.get('scaling_value', 1)
            v['label'] = v.get('label', k.replace('_', ' ').title())
            v['unit'] = v.get('label', None)

        self.add_parameter(name='num_particles',
                           initial_value=10000,
                           set_cmd=None,
                           vals=vals.Ints(),
                           parameter_class=AnalysisParameter)
        self.add_parameter(name='num_updates',
                           set_cmd=False,
                           parameter_class=AnalysisParameter)
        self.num_updates._save_val(0)
        variance_parameters = InstrumentChannel(self, 'variance_parameters')
        self.add_submodule('variance_parameters', variance_parameters)
        distr_parameters = InstrumentChannel(self, 'distr_parameters')
        self.add_submodule('distr_parameters', distr_parameters)
        for i, (paramname, paraminfo) in enumerate(self.MODEL_PARAMETERS.items()):
            self.variance_parameters.add_parameter(
                name=paramname + '_variance',
                vals=vals.Numbers(),
                label=paraminfo['label'] + ' Variance',
                unit=(paraminfo['unit'] + '^2') if paraminfo['unit'] is not None else None,
                parameter_class=AnalysisParameter)
            self.distr_parameters.add_parameter(
                name=paramname + '_prior_start',
                vals=vals.Numbers(),
                set_cmd=partial(self._set_prior_start, paramname),
                initial_value=paraminfo['prior'][0],
                parameter_class=AnalysisParameter)
            self.distr_parameters.add_parameter(
                name=paramname + '_prior_stop',
                vals=vals.Numbers(),
                set_cmd=partial(self._set_prior_stop, paramname),
                initial_value=paraminfo['prior'][1],
                parameter_class=AnalysisParameter)
            self.distr_parameters.add_parameter(
                name=paramname + '_posterior_marginal_setpoints',
                unit=paraminfo['unit'],
                m_idx=i,
                scaling_val=paraminfo['scaling_value'],
                vals=vals.Arrays(shape=(100,)),
                parameter_class=PosteriorMarginalSetpointsParameter)
            setpoints = self.distr_parameters.parameters[
                paramname + '_posterior_marginal_setpoints']
            self.distr_parameters.add_parameter(
                name=paramname + '_posterior_marginal',
                m_idx=i,
                setpoints=(setpoints,),
                label=paraminfo['label'] + 'Posterior Marginal',
                vals=vals.Arrays(shape=(100,)),
                parameter_class=PosteriorMarginalParameter)
        self.reset_updater()

    @property
    def scaling_values(self):
        scaling_dict = {}
        scaling_dict.update(
            {m: v['scaling_value'] for m, v in self.MODEL_PARAMETERS.items()})
        scaling_dict.update(
            {e: v['scaling_value'] for e, v in self.EXPERIMENT_PARAMETERS.items()})
        return scaling_dict

    def _set_prior_start(self, paramname, val):
        self.metadata['model_parameters'][paramname]['prior'][0] = val

    def _set_prior_stop(self, paramname, val):
        self.metadata['model_parameters'][paramname]['prior'][1] = val

    def reset_prior(self):
        try:
            del self._prior
        except AttributeError:
            pass
        prior_vals = []
        for m in self.MODEL_PARAMETERS.keys():
            start = self.distr_parameters.parameters[m + '_prior_start']()
            stop = self.distr_parameters.parameters[m + '_prior_stop']()
            prior_vals.append([start, stop])
        distrs = [UniformDistribution(val) for val in prior_vals]
        self._prior = PostselectedDistribution(ProductDistribution(
            *distrs), self._model)

    def reset_updater(self):
        self.reset_prior()
        try:
            del self._updater
        except AttributeError:
            pass
        self._updater = SMCUpdater(
            self._model, self.num_particles(), self._prior)
        self.num_updates._save_val(0)

    def evaluate(self, **experiment_values):
        kwargs = {k: v() for k, v in self.model_parameters.parameters.items()}
        kwargs['np'] = np
        kwargs.update(experiment_values)
        return eval(self.MODEL['np'], kwargs)

    def analyse(self, **kwargs):
        # self._check_experiment_parameters(**experiment_values)
        # shape = np.array(list(experiment_values.values())[0]).shape
        # self._check_measurement_parameters(qubit_state=measured, shape=shape)
        experiment_values = {k: v for k, v in kwargs.items() if k in self.metadata['experiment_parameters']}
        qubit_state = kwargs['qubit_state']
        expparams = np.empty((len(self._model.expparams_dtype),),
                             dtype=self._model.expparams_dtype)
        for exp_param_name, exp_param_value in experiment_values.items():
            scaled_param_val = exp_param_value / \
                self.scaling_values[exp_param_name]
            expparams[exp_param_name] = scaled_param_val
        try:
            self._updater.update(qubit_state, expparams)
            model_param_estimates = self._updater.est_mean()
            covariance_matrix = self._updater.est_covariance_mtx()
            success = 1
        except RuntimeError:
            success = 0
        self.success._save_val(success)
        for i, model_param_name in enumerate(self._model.modelparam_names):
            model_param = self.model_parameters.parameters[model_param_name]
            model_param_variance = self.variance_parameters.parameters[
                model_param_name + '_variance']
            if success:
                scaled_estimate = model_param_estimates[i] * \
                    self.scaling_values[model_param_name]
                scaled_var_estimate = covariance_matrix[i][i] * \
                    self.scaling_values[model_param_name]**2
            else:
                scaled_estimate = scaled_var_estimate = float('nan')
            model_param._save_val(scaled_estimate)
            model_param_variance._save_val(scaled_var_estimate)
        self.num_updates._save_val(self.num_updates() + 1)


def plot_model_param(mean, var, y_label=None, title=None):
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=2, ncols=1, hspace=0,
                           height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    num = np.arange(len(mean))
    ax1.plot(num, mean, label='Mean', color='C0')
    ax1.legend()
    ax1.fill_between(num, mean - np.sqrt(var),
                     mean + np.sqrt(var), alpha=0.3, facecolor='g')
    if y_label is not None:
        ax1.set_ylabel(y_label)

    ax2.plot(num, var, label='Variance', color='g')
    ax2.set_ylabel('Variance')
    ax2.set_xlabel('Shot Number')
    if title is not None:
        plt.suptitle(title)
    ax2.legend()
    return fig


def plot_estimate(estimate=None, averaged_projected_data=None,
                  projected_data=None, title=None, **setpoints):
    plottables = {'estimate': estimate,
                  'averaged_projected_data': averaged_projected_data,
                  'projected_data': projected_data}
    for k, v in plottables.items():
        if v is None:
            del plottables[k]
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=len(plottables),
                           ncols=1, hspace=0)
    axes = []
    for i in range(len(plottables)):
        if i == 0:
            axes.append(fig.add_subplot(gs[0]))
        else:
            axes.append(fig.add_subplot(gs[i], sharex=axes[0]))

    setpoint_arrs = {k: v for k, v in setpoints.items() if len(v) > 1}
    setpoint_points = {k: v for k, v in setpoints.items() if len(v) == 1}

    if len(setpoint_arrs) > 1:
        raise RuntimeError('Sorry only 1 dim plots at the mo')
    else:
        setpoint_name = list(setpoint_arrs.keys())[0]
        setpoint_arr = setpoint_arrs[setpoint_name]

    lines = []
    for i, (p, vals) in enumerate(plottables.items()):
        if p == 'estimate':
            axes[i].set_ylabel('P(0)')
            label = 'Estimated ground state probability'
            c = 'b'
        elif p == 'averaged_projected_data':
            axes[i].set_ylabel('a.u')
            label = 'Averaged projected cavity response'
            c = 'g'
        else:
            axes[i].set_ylabel('a.u')
            label = 'Projected cavity response'
            c = 'm'
        if i == len(plottables) - 1:
            axes[i].set_xlabel(setpoint_name.title())
        lines.extend(axes[i].plot(setpoint_arr, vals, c, label=label))
    if title is not None:
        plt.suptitle(title)
    fig.legend(handles=lines,
               bbox_to_anchor=(0.95, 0.5), loc='upper left')
    return fig


def plot_prior_posterior(prior, prior_setpoints,
                         posterior, posterior_setpoints,
                         x_label=None, title=None):
    f = plt.figure()
    plt.plot(prior_setpoints, prior, label='prior')
    plt.plot(posterior_setpoints, posterior, label='posterior')
    plt.legend()
    if x_label is not None:
        plt.xlabel(x_label)
    if title is not None:
        plt.suptitle(title)


def plot_bayesian_analysis(data, metadata, title=None):
    figs = []
    for name in metadata['model_parameters'].keys():
        p_data = data.pop(name)
        v_data = data.pop(name + '_variance')
        figs.append(plot_model_param(p_data, v_data, y_label=name, title=title))
        prior, posterior = data.pop(name + '_posterior_marginal')
        prior_setpoints, posterior_setpoints = data.pop(name + '_posterior_marginal_setpoints')
        figs.append(plot_prior_posterior(prior, prior_setpoints,
                                         posterior, posterior_setpoints,
                                         x_label=name, title=title))

    est = data.pop('estimate', None)
    ave = data.pop('averaged_projected_data', None)
    proj = data.pop('projected_data', None)
    qubit_state = data.pop('qubit_state', None)
    if any([i is not None for i in [est, ave, proj]]):
        figs.append(plot_estimate(est, ave, proj, title=title,
                                  **data))
    return figs



    # def simulate_experiment(self, **experiment_values):
    #     self._check_experiment_parameters(**experiment_values)
    #     n_experiments = 1
    #     for exp_param_name, exp_param_value in experiment_values.items():
    #         exp_value_arr = np.array([exp_param_value]).flatten()
    #         n_experiments *= len(setpoint_value_arr)
    #         experiment_values[exp_param_name] = setpoint_value_arr
    #     expparams = np.empty((n_experiments),
    #                          dtype=self._model.expparams_dtype)
    #     modelparams = np.empty((1, len(self._model.modelparam_names)))
    #     exp_value_combinations = product(
    #         *[v for v in experiment_values.values()])
    #     exp_param_names = experiment_values.keys()
    #     for i, combination in enumerate(exp_value_combinations):
    #         for j, exp_param_name in enumerate(exp_param_names):
    #             scaled_param_val = combination[j] / \
    #                 self.scaling_values[exp_param_name]
    #             expparams[i][exp_param_name] = scaled_param_val
    #     for i, model_param_name in enumerate(self._model.modelparam_names):
    #         scaled_param_val = self.model_parameters.parameters[
    #             model_param_name] / self.scaling_values[model_param_name]
    #         modelparams[0, i] = scaled_param_val
    #     return self._model.simulate_experiment(modelparams, expparams,
    #                                            repeat=1)
