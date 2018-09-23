import qcodes as qc
import numpy as np
from scipy.optimize import curve_fit
from itertools import product
from qdev_wrappers.fitting.least_squares_fit import LeastSquaresFit
from typing import Dict

import matplotlib.pyplot as plt
from qcodes.dataset.data_set import load_by_id
from qcodes.dataset.data_export import get_data_by_id, flatten_1D_data_for_plot
from qcodes.dataset.data_export import (datatype_from_setpoints_1d,
                          datatype_from_setpoints_2d, reshape_2D_data)
from qcodes.dataset.plotting import _rescale_ticks_and_units, plot_on_a_plain_grid, plot_2d_scatterplot

import warnings
warnings.simplefilter("always")

#alldata = get_data_by_id(run_id)
#data_info = load_by_id(run_id)
#data_info = load_by_id(run_id)
#experiment_name = data_info.exp_name
#sample_name = data_info.sample_name
# ToDo: use this to get data instead of making data dictionary


class Fitter:
    """
    Class which performs fit for data based on data, fitclass and names of

    Args:
        data (dict)
        fitclass (currently limited to LeastSquaresFit)
        indept_var (str): name of parameter in data dict to be used as
            independent variable in fit procedure. eg 'pulse_readout_delay'
        dept_var (str): name of parameter in data dict which represents
            dependent variable for fitting. eg 'cavity_magnitude_result'
    """

    def __init__(self, data: Dict, fitclass: LeastSquaresFit,
                  indept_var: str, dept_var: str, r2_limit = 0.6):
        self.fitclass = fitclass
        self.experiment_info = {'exp_id': data['exp_id'],
                                'run_id': data['run_id'],
                                'sample_name': data['sample_name']}
        self.indept_var = data[indept_var]
        self.dept_var = data[dept_var]
        self.setpoints = {k: data[k] for k in data.keys() if
                          (k not in [indept_var, dept_var]) and (k in data['variables'])}  # Todo: this will mark phase as a setpoint if magnitude is chosen as a variable
        self.fit_parameters = {
            fitclass.param_names[i]:
            {'name': fitclass.param_names[i],
             'label': fitclass.param_labels[i],
             'unit': fitclass.param_units[i]} for
            i in range(len(fitclass.param_names))}
        self.estimator = {'method': 'LSF',
                          'type': fitclass.name,
                          'fit_function_str': fitclass.fun_str,
                          'r2_limit': r2_limit}  # TODO: add dill here
        self.fit_results = self._do_fitting_procedure(data)

    def _do_fitting_procedure(self, data):
        """
        Populates fit_results list with one dictionary per combination
        of setpoints.

        eg
        for self.setpoints = {'frequency':
                                {'label': '',
                                 'unit': 'Hz',
                                 'data': [1e9, 2e9, 3e9]},
                              'power':
                                {'label': '',
                                 'unit': 'dBm',
                                 'data': [-10, -30]}}

        will make 6 dictionaries of the form
        {
            'setpoint_names': ['frequency', 'power'],
            'setpoint_labels': {'frequency': '' , 'power': ''},
            'setpoint_units': {'frequency': 'Hz' , 'power': 'dBm'},
            'setpoint_values': {'frequency': 1e9 , 'power': -10},
            'param_names': ['a', 'b'],
            'param_labels': {'a': 'T1', 'b': 'b'},
            'param_units': {'a': 's', 'b': ''},
            'param_start_values': {'a': 3, 'b': 10},
            'param_values': {'a': 2.5, 'b': 11},
            'param_variance': {'a': 0.1, 'b': 2},
            'indept_var_name': 'pulse_readout_delay',
            'indept_var_label': 'Pulse Readout Delay',
            'indept_var_unit': 's',
            'indept_var_values': [0.1, 0.2, 0.3, 0.4],
            'dept_var_name': 'cavity_magnitude_response',
            'dept_var_label': 'Cavity Response',
            'dept_var_unit': '',
            'dept_var_values': [1.15, 1.26, 1.34, 1.23],
            'estimate_values' [1.1, 1.2, 1.3, 1.4]
        }
        """
        fit_results = []
        # if setpoints then perform fit for all
        if len(self.setpoints) > 0:
#            raise RuntimeError('many setpoints is currently broken, apologies')
            # find all possible combinations of setpoint values
            setpoint_combinations = product(
                *[set(v['data']) for v in self.setpoints.values()])
            setpoint_names = list(self.setpoints.keys())
            setpoint_labels = [v['label'] for v in self.setpoints.values()]
            setpoint_units = [v['unit'] for v in self.setpoints.values()]
            for setpoint_combination in setpoint_combinations:
                # find indices where where setpoint combination is satisfied
                indices = set(np.argwhere(
                    data[setpoint_names[0]]['data'] == setpoint_combination[0]).flatten())
                for i in range(1, len(setpoint_names)):
                    new_indices = np.argwhere(
                            data[setpoint_names[0]]['data'] == setpoint_combination[0]).flatten()
                    indices = indices.intesection(new_indices)
                indices = list(indices)
                input_data_array = data[self.indept_var['name']
                                        ]['data'][indices]
                output_data_array = data[self.dept_var['name']
                                         ]['data'][indices]
                param_dict = self._perform_fit(input_data_array,
                                               output_data_array)
                result = param_dict.copy()
                result.update(
                    {'setpoint_names': setpoint_names,
                     'setpoint_labels': dict(zip(setpoint_names,
                                                 setpoint_labels)),
                     'setpoint_units': dict(zip(setpoint_names,
                                                setpoint_units)),
                     'setpoint_values': dict(zip(setpoint_names,
                                                 setpoint_combination)),
                     'indept_var_values': input_data_array,
                     'dept_var_values': output_data_array,
                     'indept_var_name': self.indept_var['name'],
                     'dept_var_name': self.dept_var['name'],
                     'indept_var_label': self.indept_var['label'],
                     'dept_var_label': self.dept_var['label'],
                     'indept_var_unit': self.indept_var['unit'],
                     'dept_var_unit': self.dept_var['unit'],
                     'estimate_values':
                        self._find_estimate(input_data_array,
                                           param_dict['param_values'])})
                fit_results.append(result)
#                except Exception:  # TODO: what kind of exception
#                    print('no data for setpoint combination ', dict(
#                        zip(self.setpoints.keys(), setpoint_combination)))
        else:
            input_data_array = data[self.indept_var['name']]['data']
            output_data_array = data[self.dept_var['name']]['data']
            param_dict = self._perform_fit(input_data_array,
                                       output_data_array)
            result = param_dict.copy()
            result.update(
                {'setpoint_names': [],
                 'indept_var_values': input_data_array,
                 'dept_var_values': output_data_array,
                 'indept_var_name': self.indept_var['name'],
                 'dept_var_name': self.dept_var['name'],
                 'indept_var_label': data[self.indept_var['name']]['label'],
                 'dept_var_label': data[self.dept_var['name']]['label'],
                 'indept_var_unit': data[self.indept_var['name']]['unit'],
                 'dept_var_unit': data[self.dept_var['name']]['unit'],
                 'estimate_values':
                    self._find_estimate(input_data_array,
                                               param_dict['param_values'])})
            fit_results.append(result)
        return fit_results

    def get_result(**setpoint_values):
        """
        Args:
            kwargs for each setpoints
                eg 'frequency=1e9, power=-10'
        Returns:
            dict for fit where these conditions are satisfiedxw
        """
        if len(setpoint_values) != len(self.setpoints):
            raise RuntimeError('Must specify a value for each setpoint')
        elif len(self.setpoints) == 0:
            return self.fit_results[0]
        else:
            return next(res for res in self.fit_results if
                        all(res[setpoint] == value for
                            setpoint, value in setpoint_values.items()))

    def get_r2(self, estimate, data):
        """
        Finds residual and total sum of squares, calculates the R^2 value
        Args:
            A np.array of data, and a corresponding np.array of data estimate based off a fit to the data.
        """
        ss_res = np.sum((data - estimate) ** 2)
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def _perform_fit(self, input_data_array, output_data_array):

        # make fit params dictionary
        params_dict = {'param_names': [], 'param_labels': {},
                       'param_units': {}, 'param_start_values': {},
                       'param_variance': {}, 'param_values': {}}
        for i, param_name in enumerate(self.fitclass.param_names):
            params_dict['param_names'].append(param_name)
            params_dict['param_labels'][param_name] = self.fitclass.param_labels[i]
            params_dict['param_units'][param_name] = self.fitclass.param_units[i]

        # find start parameters, run curve_fit function to perform fit
        p_guess = self.fitclass.guess(input_data_array, output_data_array)
        popt, pcov = curve_fit(self.fitclass.fun,
                               input_data_array,
                               output_data_array,
                               p0=p_guess)

        # add guess and fit results to dict
        for i, param_name in enumerate(self.fitclass.param_names):
            params_dict['param_start_values'][param_name] = p_guess[i]
            params_dict['param_values'][param_name] = popt[i]
            params_dict['param_variance'][param_name] = pcov[i, i]

        # r2 test to approve fit
        est_data = self._find_estimate(input_data_array, params_dict['param_values'])
        r2 = self.get_r2(est_data, output_data_array)
        if r2 < self.estimator['r2_limit']:
            params_dict['param_values'] = None
            params_dict['param_variance'] = None
            warnings.warn('Unsuccessful fit')

        return params_dict

    def _find_estimate(self, input_data_array, params_values_dict):
        if params_values_dict is None:
            return [None] * len(input_data_array)
        else:
            return self.fitclass.fun(input_data_array, **params_values_dict)

    def _plot_1d(self, ax, xdata, ydata, parameter_values, parameter_variance, rescale_axes):

        ax.plot(xdata, ydata, marker='.', markersize=5, linestyle='', color='C0')
        if parameter_values is None:
            pass
        else:
            x = np.linspace(xdata.min(), xdata.max(), len(xdata) * 10)
            ax.plot(x, self.fitclass.fun(x, **parameter_values), color='C1')

        # axes labels and title
        ax.set_xlabel(f"{self.indept_var['label']} ({self.indept_var['unit']})")
        ax.set_ylabel(f"{self.dept_var['label']} ({self.dept_var['unit']})")

        if rescale_axes:
            data_lst = [self.indept_var, self.dept_var]
            _rescale_ticks_and_units(ax, data_lst)

        # fit result box
        if parameter_values is None:
            textstr = '{} \n Unsuccessful fit'.format(self.fitclass.fun_str)
        else:
            p_label_list = [self.fitclass.fun_str]
            for parameter in parameter_values:
                value = parameter_values[parameter]
                unit = self.fit_results[0]['param_units'][parameter]
                standard_dev = np.sqrt(parameter_variance[parameter])
                p_label_list.append('{} = {:.3g} $\pm$ {:.3g} {}'.format(parameter, value, standard_dev, unit))
            textstr = '\n'.join(p_label_list)

        ax.text(1.05, 0.7, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox={'ec': 'k', 'fc': 'w'})
        # Todo: scaling for fit result numbers and units, units based on input

    def plot(self, setpoint=None, rescale_axes: bool=True):
        # setpoint should be None or a list of the form {setpoint_name: value, setpoint_name2: value, ...}
        #Todo: should this really be part of the fitter? What about if you want to plot the fit from the fit_id later?
        #Todo: wouldn't it be better to save the fit first, and then plot from the experiment database?

        # make figure, axes
        fig, ax = plt.subplots(1, 1)
        colorbar = None

        # make title
        run_id = self.experiment_info['run_id']
        experiment_name = self.experiment_info['exp_id']
        sample_name = self.experiment_info['sample_name']
        title = f"Run #{run_id} fitted, Experiment {experiment_name} ({sample_name})"
        ax.set_title(title)

        if len(self.setpoints) == 0:  # 1D PLOTTING

            parameter_values = self.fit_results[0]['param_values']
            parameter_variance = self.fit_results[0]['param_variance']
            xpoints = self.indept_var['data']
            ypoints = self.dept_var['data']

            self._plot_1d(ax, xpoints, ypoints, parameter_values, parameter_variance, rescale_axes)


        elif len(self.setpoints) >= 1 : # 2D PLOTTING

            # make copy of fit_results
            plot_results = self.fit_results.copy()

            # if setpoints are specified, remove all results from plot_results that are not at the specified setpoints
            if setpoint is not None:
                setpoint_names = [key for key in setpoint.keys()]
                for name in setpoint_names:
                    for result in self.fit_results:
                        if result['setpoint_values'][name] != setpoint[name]:
                            plot_results.remove(result)


            # if, after removing all results not at specified setpoint, only 1 fit is left, plot the 1D fit of this cut
            if len(plot_results) == 1:

                title += '\n' + str(setpoint)
                ax.set_title(title)

                parameter_values = plot_results[0]['param_values']
                parameter_variance = plot_results[0]['param_variance']
                xpoints = plot_results[0]['indept_var_values']
                ypoints = plot_results[0]['dept_var_values']

                self._plot_1d(ax, xpoints, ypoints, parameter_values, parameter_variance, rescale_axes)


            # if, after removing all results, there are many fits left in plot_results, attempt 2D plot + params plots
            elif len(plot_results) > 1:

                # confirm that there is only 1 setpoint type besides the independent variable i.e. that data is 2D
                if setpoint is None:
                    setpoint_list = [name for name in self.setpoints]
                else:
                    setpoint_list = [name for name in self.setpoints if name not in setpoint]
                if len(setpoint_list) > 1:
                    raise NotImplementedError('Function has too many ({}) unspecified setpoints.'.format(len(setpoint_list)))


                # get setpoint info for plot labels
                setpoint_name = setpoint_list[0]
                setpoint_unit = self.setpoints[setpoint_name]['unit']
                setpoint_label = self.setpoints[setpoint_name]['label']

                # retrieve independent variable and setpoint data, output estimate, parameter values
                x = []
                y = []
                z = []
                params = {'param_setpoints' : []}
                for param_name in self.fit_parameters:
                    params[param_name] = []
                    params[param_name + '_variance'] = []


                for result in plot_results:
                    setpoint_value = result['setpoint_values'][setpoint_name]
                    for xvalue in result['indept_var_values']:
                        x.append(xvalue)
                        y.append(setpoint_value)
                    for est_value in result['estimate_values']:
                        z.append(est_value)
                    if result['param_values'] is not None:
                        params['param_setpoints'].append(setpoint_value)
                        for parameter in result['param_values']:
                            params[parameter].append( result['param_values'][parameter] )
                            params[parameter + '_variance'].append( result['param_variance'][parameter] )




                xdata = np.array(x)
                ydata = np.array(y)
                zdata = np.array(z)
                for item in params: params[item] = np.array(params[item])



                # 2D heatmap plot
                # how to plot - is this necessary? Don't we always want to plot on a grid?
                how_to_plot = {'grid': plot_on_a_plain_grid,
                               'equidistant': plot_on_a_plain_grid,
                               'point': plot_2d_scatterplot,
                               'unknown': plot_2d_scatterplot}

                plottype = datatype_from_setpoints_2d([xdata ,ydata])

                xpoints = flatten_1D_data_for_plot(xdata)
                ypoints = flatten_1D_data_for_plot(ydata)
                zpoints = flatten_1D_data_for_plot(zdata)
                plot_func = how_to_plot[plottype]

                ax, colorbar = plot_func(xpoints, ypoints, zpoints, ax, colorbar)

                ax.set_xlabel(f"{self.indept_var['label']} ({self.indept_var['unit']})")
                ax.set_ylabel(f"{setpoint_label} ({setpoint_unit})")
                colorbar.set_label(f"{self.dept_var['label']} ({self.dept_var['unit']})")

                if rescale_axes:
                    # create list of dictionaries for x, y and z containing units and labels
                    data_lst = [self.indept_var, self.setpoints[setpoint_name], self.dept_var]
                    #run rescale function to create new, scaled axes
                    _rescale_ticks_and_units(ax, data_lst, colorbar)




                # Parameter vs setpoint plots
                order = params['param_setpoints'].argsort()
                setpoint_data_dict = self.setpoints[setpoint_name]
                setpoint_data_dict['data'] = params['param_setpoints'][order]

                for param in self.fit_parameters:
                        # Todo: this should be like in plot_by_id, containing n subplots where ax holds all of them
                    fig, ax = plt.subplots(1, 1)

                    param_data_dict = self.fit_parameters[param]
                    param_data_dict['data'] = params[param][order]
                    param_variance = params[param + '_variance'][order]
                    param_standard_dev = np.array( [np.sqrt(variance) for variance in param_variance] )

                    ax.errorbar(setpoint_data_dict['data'], param_data_dict['data'], param_standard_dev, None, 'g.-')

                    ax.set_title(title)
                    ax.set_xlabel(f"{setpoint_data_dict['label']} ({setpoint_data_dict['unit']})")
                    ax.set_ylabel(f"{param_data_dict['label']} ({param_data_dict['unit']})")
                    if rescale_axes:
                        # create list of dictionaries for x, y and z containing units and labels
                        data_lst = [setpoint_data_dict, param_data_dict]
                        # run rescale function to create new, scaled axes
                        _rescale_ticks_and_units(ax, data_lst)

        return ax, colorbar



