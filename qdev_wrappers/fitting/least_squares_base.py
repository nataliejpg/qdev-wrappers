import numpy as np
from qdev_wrappers.analysis.base import AnalyserBase, AnalysisParameter
from qcodes.instrument.channel import InstrumentChannel

# TODO: add guess to metadata as string


class LeastSquaresBase(AnalyserBase):
    METHOD = 'LeastSquares'
    EXPERIMENT_PARAMETERS = {'x': {}}
    MEASUREMENT_PARAMETERS = {'y': {}}
    GUESS = None
    """
    An extension of the Fitter which uses the least squares method to fit
    an array of data to a function and learn the most likely parameters of
    this function and the variance on this knowledge. Also stored the initial
    guesses used.

    Args:
        name (str)
        fit_parameters (dict): dictionary describing the parameters in the
            function which is being fit to, keys are taken as parameter names
            as they appear in the function, values are dictionary with keys
            'unit' and 'label' to be used in plotting.
        function_metadata (dict): description of the function to be fit to
            including the exact code to be evaluated (key 'np') and the
            label to be printed (key 'str')
    """

    def __init__(self):
        super().__init__()
        variance_parameters = InstrumentChannel(self, 'variance_parameters')
        self.add_submodule('variance_parameters', variance_parameters)
        initial_val_params = InstrumentChannel(self, 'initial_value_parameters')
        self.add_submodule('initial_value_parameters', initial_val_params)
        for paramname, paraminfo in self.MODEL_PARAMETERS.items():
            self.variance_parameters.add_parameter(
                name=paramname + '_variance',
                label=paraminfo.get('label', paramname) + ' Variance',
                unit=paraminfo['unit'] + '^2' if 'unit' in paraminfo else None,
                set_cmd=False,
                parameter_class=AnalysisParameter)
            self.initial_value_parameters.add_parameter(
                name=paramname + '_initial_value',
                label=paraminfo.get('label', paramname) + ' Initial Value',
                unit=paraminfo.get('unit', None),
                set_cmd=False,
                parameter_class=AnalysisParameter)

    def _get_r2(self, estimate, measured):
        """
        Finds residual and total sum of squares, calculates the R^2 value
        """
        meas = np.array(measured_values)
        est = np.array(estimate)
        ss_res = np.sum((meas - est) ** 2)
        ss_tot = np.sum((meas - np.mean(meas)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def guess(self, measured_values, experiment_values):
        if self.GUESS is not None:
            return self.GUESS(measured_values, experiment_values)
        else:
            return [1. for _ in self.initial_value_parameters.parameters]

    def evaluate(self, experiment_values, *fit_values):
        """
        Evaluates the function to be fit to (stored in the metadata) for
        the values of the independent variable (experiment value) and the
        fit parameters provided.
        """
        fit_params = list(self.fit_parameters.parameters.keys())
        if len(fit_values) == 0:
            fit_values = [v() for v in self.fit_parameters.parameters.values()]
        kwargs = {'x': experiment_values,
                  'np': np,
                  **dict(zip(fit_params, fit_values))}
        return eval(self.metadata['function']['np'], kwargs)

    def analyse(self, measured, experiment, initial_values=None,
                r2_limit=None, variance_limited=False,):
        """
        Performs a fit based on a measurement and using the evaluate function.
        Updates the instrument parameters to record the success and results
        of the fit.
        """
        self._check_experiment_parameters(x=experiment)
        self._check_measurement_parameters(
            y=measured, shape=np.array(experiment).shape)
        if initial_values is None:
            initial_values = self.guess(measured, experiment_values)
        for i, initial_param in enumerate(self.initial_value_parameters.parameters.values()):
            initial_param._save_val(initial_values[i])
        try:
            popt, pcov = curve_fit(
                self.evaluate, experiment, measured,
                p0=initial_values)
            variance = np.diag(pcov)
            estimate = self.evaluate(experiment, *popt)
            r2 = self._get_r2(estimate, measured)
            if r2_limit is not None and r2 < r2_limit:
                success = 0
                message = 'r2 {:.2} exceeds limit {:.2}'.format(r2, r2_limit)
            elif variance_limited and any(variance == np.inf):
                success = 0
                message = 'infinite variance'
            else:
                success = 1
        except (RuntimeError, ValueError) as e:
            success = 0
            message = str(e)
        self.success._save_val(success)
        fit_params = list(self.fit_parameters.parameters.values())
        variance_params = list(
            self.variance_parameters.parameters.values())
        initial_params = list(
            self.initial_value_parameters.parameters.values())
        if success:
            for i, val in enumerate(popt):
                fit_params[i]._save_val(val)
                if variance[i] == np.inf:
                    variance_params[i]._save_val(float('nan'))
                else:
                    variance_params[i]._save_val(variance[i])
        else:
            for i, param in enumerate(fit_params):
                param._save_val(float('nan'))
                variance_params[i]._save_val(float('nan'))
            warnings.warn('Fit failed due to: ' + message)
