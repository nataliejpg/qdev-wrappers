import qcodes as qc
import numpy as np
from numpy.random import randint
from qcodes.dataset.measurements import Measurement
from select_next_coordinates import get_neighbours

# this goes in
from optimization_from_completed_data import get_measurement_from_data
measure = get_measurement_from_data
# ToDo: make this a class? with measure, stopping condition, cost_val and requirements as class methods?





def optimize(variable_params, measured_params, stopping_condition, cost_val, requirements=None):
    #requirements, if given, should take a 'candidate' (parameter coordinates) and
    # return True or False depending on whether the settings or resulting measurement meet a set of requirements
    num_attempts = 0
    start_coordinates = [randint(0, len(value['values'])) for value in params.values()]
    best = start_coordinates
    current_coordinates = start_coordinates
    checked = []

    if requirements is None:
        def requirements(candidate):
            return True

    meas = Measurement()

    meas.register_custom_parameter(name='frequency', label='Frequency', unit='Hz')
    meas.register_custom_parameter(name='pulse_duration', label='Pulse Duration', unit='s')
    meas.register_custom_parameter(name='cavity_response', label='Cavity Response', unit='V',
                                   setpoints=('pulse_duration', 'frequency'))

    with meas.run() as datasaver:
        run_id = datasaver.run_id

        # while not(stopping_condition(num_attempts)):
        while not (stopping_condition(params, best, num_attempts)):
            neighborhood = get_neighbours(current_coordinates, params)
            for candidate in neighborhood:
                if candidate not in checked:
                    freq = params['frequency']['values'][candidate[0]]
                    pulse_dur = params['pulse_duration']['values'][candidate[1]]
                    cav_response = measure(params, candidate)
                    datasaver.add_result(('frequency', freq),
                                         ('pulse_duration', pulse_dur),
                                         ('cavity_response', cav_response))
                checked.append(candidate)
                if requirements(candidate):
                    if (cost_val(candidate) < cost_val(current_coordinates)):
                        current_coordinates = candidate
            if cost_val(current_coordinates) < cost_val(best):
                best = current_coordinates

            num_attempts += 1
            if num_attempts % 10 == 0:
                print(num_attempts)
        print(f"Best value: {get_measurement_from_data(params, best)}")

        return start_coordinates, checked, best


def try_many(num_attempts, success_condition=None):
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
        return best_coordinates, best_value, starts, avg_measurements, avg_num_iterations, successes/num_attempts
    else:
        return best_coordinates, best_value, starts, avg_measurements, avg_num_iterations
