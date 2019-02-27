import qcodes as qc
import numpy as np
from numpy.random import randint
from select_next_coordinates import get_neighbours
from itertools import product

# this goes in
#from optimization_from_completed_data import get_measurement_from_data
#measure = get_measurement_from_data
# ToDo: make this a class? with measure, stopping condition, cost_val and requirements as class methods?
# Then optimize would only take params (dictionary of parameters to vary and their values)
# or maybe it also stores the params info, and just takes spans?
# also maybe it doesn't need spans
# then it should also contain information about the parameters to be measured


class Optimization():
    def __init__(self, params, measure, stopping_condition, cost_val, select_next, reqs, from_completed_dataset):
        self.checked = []
        self.params = params
        self.num_attempts = 0
        if from_completed_dataset:
            self.start = [randint(0, len(param['values'])) for param in params]
        else:
            self.start = [param.get() for param in params]
        self.best = self.start
        self.current = self.start

        self.reqs = reqs
        self.measure = measure
        self.stop = stopping_condition
        self.cost_val = cost_val
        self.select_next = select_next

    def requirements(self, location):
        if self.reqs is None:
            return True
        else:
            return reqs(location)




    def try_many(self, num_attempts, success_condition=None):
        # success_condition: a function that takes the best_value and returns True for a success, otherwise False
        if success_condition is not None:
            successes = 0
        measurements_done = 0
        iterations = 0
        best_value = []
        best_coordinates = []
        starts = []

        for attempt in range(0, num_attempts):
            start, checked, best = optimize()

            iterations += len(checked)
            measurements_done += len(np.unique(checked))
            starts.append(start)
            best_coordinates.append(best)
            best_value.append(get_measurement_from_data(best))

            if success_condition is not None:
                success = success_condition(best_value)
                if success:
                    successes += 1

        avg_measurements = measurements_done / num_attempts
        avg_num_iterations = iterations / num_attempts

        if num_attempts % 10 == 0:
            print(num_attempts)
        if success_condition is not None:
            return best_coordinates, best_value, starts, avg_measurements, avg_num_iterations, successes / num_attempts
        else:
            return best_coordinates, best_value, starts, avg_measurements, avg_num_iterations

    def move_to_best_neighbour(self, current_location, params, cost_val, requirements):
        """returns a list of tuples, each with indicies for neighboring points"""
        next_param_coordinates = {}
        checked = []

        for i, param in enumerate(params):
            datalength = len(params[param]['values'])
            step = randint(0, params[param]['max_step']) + 1
            next_param_coordinates[param] = [(current_location[i] - step) % datalength,
                                             (current_location[i] + step) % datalength]

        param_combinations = product(*[set(v) for v in next_param_coordinates.values()])
        neighbours = [combination for combination in param_combinations]
        for candidate in neighbours:
            checked.append(candidate)
            if self.requirements(candidate):
                if (cost_val(candidate) < cost_val(current_location)):
                    new_location = candidate
                    # this currently takes fewer measurements, since it takes the first good thing it finds
                    # but it also gives it a tendency to move in a set direction...
                    # compare to testing all "neighbors" before selecting

        return new_location, checked

    def move_to_weighted_location(self, current_location, params):
        """returns a list of tuples, each with indicies for neighboring points"""
        next_param_coordinates = {}

        for i, param in enumerate(params):
            datalength = len(params[param]['values'])
            step = randint(0, params[param]['max_step']) + 1
            next_param_coordinates[param] = [(current_location[i] - step) % datalength,
                                             (current_location[i] + step) % datalength]

        param_combinations = product(*[set(v) for v in next_param_coordinates.values()])
        neighbours = [combination for combination in param_combinations]
        for candidate in neighbours:
            if candidate not in self.checked:
                freq = params['frequency']['values'][candidate[0]]
                pulse_dur = params['pulse_duration']['values'][candidate[1]]
                cav_response = measure(params, candidate)
                datasaver.add_result(('frequency', freq),
                                     ('pulse_duration', pulse_dur),
                                     ('cavity_response', cav_response))
            checked.append(candidate)
            if requirements(candidate):
                if (cost_val(candidate) < cost_val(current_location)):
                    new_coordinates = candidate

        return new_location