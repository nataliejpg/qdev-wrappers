from qcodes.dataset.measurements import Measurement
import numpy as np
from optimization_from_completed_data import get_measurement_from_data
from Methods import ReadoutFidelityOptimization


class Optimization:

    #ToDo: separate optimize function from Optimization, so that it creates and returns an optimization instead
    #ToDo: params and step size in optimization instead of in method
    #ToDo: make measurement optional when optimizing

    def __init__(self, method: ReadoutFidelityOptimization):
        self.method = method
        self.params = method.params
        self.num_attempts = 0
        self.start = {param.full_name: param() for param in self.params}

        self.current = self.start.copy()
        initial_measurement = method.measurement_function()
        for i, param in enumerate(method.measured_params):
            self.current[param] = initial_measurement[i]

        self.best_cost_val = method.cost_val(method.measurement_function())
        self.best = self.start

    def optimize(self, *params):

        self.method.params = [param for param in params]
        #ToDo: this is not great - it needs to update step size too, and these things shouldn't be stored in the model.
        #But maybe a list of possible parameters and their default step sizes are stored in the model?

        meas = Measurement()

        setpoints = []
        for parameter in self.method.params:
            setpoints.append(parameter.full_name)
            meas.register_parameter(parameter)
        for parameter, param_info in self.method.measured_params.items():
            meas.register_custom_parameter(name=parameter,
                                           label=param_info['label'],
                                           unit=param_info['unit'],
                                           setpoints=tuple(setpoints))

        with meas.run() as datasaver:
            run_id = datasaver.run_id

            while not self.method.stopping_condition(self.num_attempts):
                # check new parameter settings, select one to move to
                next_results = self.method.check_next(self.current)
                self.num_attempts += 1

                for result in next_results:
                    res = []
                    for param, val in result.items():
                        res.append((param, val))
                    datasaver.add_result(*res)

                next_location = self.method.select_next_location(next_results, self.current)
                
                if next_location not in next_results:
                    res = []
                    for param, val in next_location.items():
                        res.append((param, val))
                    datasaver.add_result(*res)

                self.current = next_location
                if self.method.cost_val(next_location) < self.best_cost_val:
                    self.best = self.current
                    self.best_cost_val = self.method.cost_val(next_location)

        print(f"Best: {self.best}")
        return run_id

    def try_many(self, *params, num_repetitions=100, success_condition=None):
        # ToDo: change this to be for optimization while measuring
        # ToDo: make into measurement that records number of iterations, time for optimization, optimal parameters
        # success_condition: a function that takes the best_value and returns True for a success, otherwise False
        if success_condition is not None:
            successes = 0
        measurements_done = 0
        iterations = 0
        best_value = []
        best_coordinates = []
        starts = []

        for attempt in range(0, num_repetitions):
            start, checked, best = self.optimize(*params)

            iterations += len(checked)
            measurements_done += len(np.unique(checked))
            starts.append(start)
            best_coordinates.append(best)
            best_value.append(get_measurement_from_data(params, self.method.measured_params, best))

            if success_condition is not None:
                success = success_condition(best_value)
                if success:
                    successes += 1

        avg_measurements = measurements_done / num_repetitions
        avg_num_iterations = iterations / num_repetitions

        if num_repetitions % 10 == 0:
            print(num_repetitions)
        if success_condition is not None:
            return best_coordinates, best_value, starts, avg_measurements, avg_num_iterations, successes/num_repetitions
        else:
            return best_coordinates, best_value, starts, avg_measurements, avg_num_iterations
