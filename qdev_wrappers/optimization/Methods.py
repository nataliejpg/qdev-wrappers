import qcodes as qc
from numpy.random import randint
from itertools import product


class ReadoutFidelityOptimization:

    def __init__(self, *params, get_data, max_attempts=250):
        # get_data: pwa.alazar_channels.data
        # ToDo: figure out better solution for step size?
        # ToDo: I think params just go in the optimize function, instead of in the model, so model just needs get_data
        self.params = [item for item in params if isinstance(item, qc.Parameter)]
        self.step_size = [item for item in params if isinstance(item, float)]
        self.measured_params = {'readout_fidelity': {'label': 'Readout Fidelity', 'unit': ''}}
        self.max_attempts = max_attempts
        self.get_data = get_data

    def check_next(self, current_location):
        """Takes current location in parameter space, decides next coordinates to measure
            Returns a list of dictionaries, one for each location, with parameters (both variable
            and measured) and their values"""
        pass
        # return next_results

    def select_next_location(self, next_results, current_location):
        """Takes next coordinates and measurements from check_next, and the current
         best value found from the optimization, uses them to decide where to go next.
         Returns parameters and a measurement for next location"""
        pass
        # return next_location

    def measurement_function(self):
        """defines what to call to measure"""
        results = self.get_data()
        no_pi_real = results[0][:, 0]
        no_pi_im = results[1][:, 0]
        pi_real = results[0][:, 1]
        pi_im = results[1][:, 1]

        fidelity_info = calculate_fidelity(no_pi_real, no_pi_im, pi_real, pi_im)

        return [fidelity_info.max_separation_value]

    def stopping_condition(self, num_attempts):
        if num_attempts > self.max_attempts:
            return True
        else:
            return False

    def cost_val(self, measurement):
        if isinstance(measurement, dict):
            val = measurement['readout_fidelity']
        elif isinstance(measurement, float):
            val = measurement
        elif isinstance(measurement, list):
            if len(measurement) > 1:
                raise RuntimeError(f"Too many values for measurement - expected 1 and got {len(measurement)}")
            val = measurement[0]
        else:
            raise RuntimeError(f"I don't know what to do with a measurement in this format: {measurement}")
        return 1-val


class BestNeighbour(ReadoutFidelityOptimization):

    def check_next(self, current_location):
        """Takes current location in parameter space, decides next coordinates to measure
            Returns a list of dictionaries, one for each location, with parameters (both variable
            and measured) and their values"""
        next_param_vals = {}
        next_locations = []

        for i, param in enumerate(self.params):
            # select randomly to go anywhere from 1-10 steps
            step = randint(1, 11) * self.step_size[i]
            next_param_vals[param.full_name] = [current_location[param.full_name] - step,
                                                current_location[param.full_name] + step]

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

    def select_next_location(self, next_locations, current_location):
        # If any of the new locations are better than the current, return the new current location.
        # Otherwise, return the current location.
        for candidate in next_locations:
            if self.cost_val(candidate) < self.cost_val(current_location):
                current_location = candidate
        return current_location


class WeightedMovement(ReadoutFidelityOptimization):

    def check_next(self, optimization):
        """Takes current location in parameter space, decides next coordinates to measure
            Returns a list of dictionaries, one for each location, with parameters (both variable
            and measured) and their values"""
        next_param_vals = {}
        next_locations = []

        step_multiplier = 10

        for i, param in enumerate(self.params):
            step = step_multiplier * self.step_size[i]
            step_multiplier -= 1
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
        # go to new location based on weighted
        current_location = optimization.current
        delta_params = {param: 0 for param in self.params}

        for candidate in next_locations:
            cv = self.cost_val(candidate)
            for param in candidate:
                d = candidate[param] - current_location[param]
                delta_params[param] += d/cv

        for param in current_location:
            current_location[param] += delta_params[param]

        return current_location


class PendulumSearch:
    pass
