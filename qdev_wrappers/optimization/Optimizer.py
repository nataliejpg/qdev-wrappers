from Optimization import Optimization
from qcodes.dataset.measurements import Measurement
import numpy as np
from optimization_from_completed_data import get_measurement_from_data


class Optimizer:

    def __init__(self, method):
        self.method = method
        self.current = []

    def optimize(self, *params):

        self.method.params = [param for param in params]
        optimization = Optimization(self.method)

        meas = Measurement()

        setpoints = []
        for parameter in self.method.params:
            setpoints.append(parameter.full_name)
            meas.register_parameter(parameter)
        for parameter, param_info in self.method.measured_param.items():
            meas.register_custom_parameter(name=parameter,
                                           label=param_info['label'],
                                           unit=param_info['unit'],
                                           setpoints=tuple(setpoints))

        with meas.run() as datasaver:
            optimization.run_id = datasaver.run_id

            while not self.method.stopping_condition(optimization):
                # check new parameter settings, select one to move to
                next_results = self.method.check_next(optimization)
                optimization.num_attempts += 1

                for result in next_results:
                    res = []
                    for param, val in result.items():
                        res.append((param, val))
                    datasaver.add_result(*res)

                next_location = self.method.select_next_location(next_results, optimization.current)
                
                if next_location not in next_results:
                    res = []
                    for param, val in next_location.items():
                        res.append((param, val))
                    datasaver.add_result(*res)

                optimization.current = next_location
                if self.method.cost_val(next_location) < optimization.best_cost_val:
                    optimization.best = optimization.current
                    optimization.best_cost_val = self.method.cost_val(next_location)

        print(f"Best: {optimization.best}")
        return optimization

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
