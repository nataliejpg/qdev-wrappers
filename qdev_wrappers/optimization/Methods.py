from numpy.random import randint
from itertools import product


class ReadoutFidelityOptimization:

    def __init__(self, *params, max_attempts=250):
        # Todo: what happens if it is initialized without params? I would like it to save an empty list
        self.params = [param for param in params]
        self.measured_params = {'readout_fidelity': {'label': 'Readout Fidelity', 'unit': ''}}
        self.max_attempts = max_attempts

    def check_next(self, optimization):
        """Takes current location in parameter space, decides next coordinates to measure
            Returns a list of dictionaries, one for each location, with parameters (both variable
            and measured) and their values"""
        pass
        # return next_results

    def select_next_location(self, next_results, optimization):
        """Takes next coordinates and measurements from check_next, and the current
         best value found from the optimization, uses them to decide where to go next.
         Returns parameters and a measurement for next location"""
        pass
        # return next_location

    def measurement_function(self):
        """defines what to call to measure"""
        results = pwa.alazar_channels.data()
        no_pi_real = results[0][:, 0]
        no_pi_im = results[1][:, 0]
        pi_real = results[0][:, 1]
        pi_im = results[1][:, 1]

        fidelity_info = calculate_fidelity(no_pi_real, no_pi_im, pi_real, pi_im)

        return [fidelity_info.max_separation_value]

    def stopping_condition(self, optimization):
        if optimization.num_attempts > self.max_attempts:
            return True
        else:
            return False

    def cost_val(self, measurement):
        if isinstance(measurement, dict):
            val = measurement['readout_fidelity']
        elif isinstance(measurement, float):
            val = measurement
        else:
            raise RuntimeError(f"I don't know what to do with a measurement in this format: {measurement}")
        return 1-val


class BestNeighbor(ReadoutFidelityOptimization):

    def check_next(self, optimization):
        """Takes current location in parameter space, decides next coordinates to measure
            Returns a list of dictionaries, one for each location, with parameters (both variable
            and measured) and their values"""
        next_param_vals = {}
        next_locations = []

        for i, param in enumerate(self.params):
            step = randint(0, param['max_step']) + 1
            # ToDo: how to decide step size??
            next_param_vals[param.full_name] = [optimization.current[param.full_name] - step,
                                                optimization.current[param.full_name] + step]

        param_combinations = product(*[set(v) for v in next_param_vals.values()])

        for combination in param_combinations:
            d = {}
            for i, param in enumerate(self.params):
                param(combination[i])
                d[param.full_name] = combination[i]
            measurement = self.measurement_function()
            for i, measured_param in enumerate(self.measured_params):
                d[measured_param] = measurement[i]
            next_locations.append(d)

        return next_locations

    def select_next_location(self, next_locations, optimization):
        # If any of the new locations are better than the current, return the new current location.
        # Otherwise, return the current location.
        current_location = optimization.current
        for candidate in next_locations:
            if self.cost_val(candidate) < self.cost_val(current_location):
                current_location = candidate
        return current_location


class WeightedMovement(ReadoutFidelityOptimization):

    def check_next(self, optimization):
        pass

        next_locations = []

        for i, param in enumerate(self.params):
            step = randint(0, param['max_step']) + 1
            # ToDo: how to decide step size??
            start_val = param()
            param_vals = [start_val - step,
                          start_val + step]

            for val in param_vals:
                d = {}
                param(val)
                for parameter in self.params:
                    d[parameter.full_name] = parameter()
                measurement = self.measurement_function()
                for index, measured_param in enumerate(self.measured_params):
                    d[measured_param] = measurement[index]
                next_locations.append(d)

            param(start_val)

        return next_locations

    def select_next_location(self, next_locations, optimization):
        pass
        # find new location based on weighting cost functions of the measured locations in next_locations
        # consider also changing step size based on result of weighting


class PendulumSearch:
    pass
