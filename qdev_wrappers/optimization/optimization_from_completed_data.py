import numpy as np
from qcodes.dataset.data_export import get_data_by_id
from Optimization import Optimization
from qcodes.dataset.measurements import Measurement


def get_measurement_from_data(variable_params, measured_params, coordinates):
    # currently ignores all but first independent variable
    measurement_values = measured_params[0]['values']
    all_indicies = []
    param_values = []

    # find parameter values based on current 'coordinates' in parameter space
    for i, param in enumerate(variable_params):
        param_index = coordinates[i]
        param_value = param['values'][param_index]
        param_values.append(param_value)

    # find index that corresponds to the combination of values in param_values
    for param, value in zip(variable_params, param_values):
        indicies = set(np.argwhere(param['data'] == value).flatten())
        all_indicies.append(indicies)
    index = all_indicies[0]
    for indicies_list in all_indicies:
        index = index.intersection(indicies_list)
    if len(index) != 1:
        raise RuntimeError(f"Found {len(indicies)} measurement points matching the setpoints")
    else:
        index = list(index)[0]

    return measurement_values[index]


def get_params_dict(run_id):
    all_data = get_data_by_id(run_id)
    variable_params = []
    measured_params = []

    for data in all_data:
        for param in data[0:-1]:
            param_info = param.copy()
            param_info['values'] = np.unique(param['data'])
            variable_params.append(param_info)
        param_info = data[-1]
        param_info['values'] = np.unique(data[-1]['data'])
        measured_params.append(param_info)

    return variable_params, measured_params


def get_results(variable_params, measured_params, coordinates):
    results = []
    for co, param in zip(coordinates, variable_params):
        results.append((param['name'], param['values'][co]))
    for param in measured_params:
        data = get_measurement_from_data(variable_params, measured_params, coordinates)
        results.append((param['name'], data))
    return results


def optimize_from_runid(run_id,
                        variable_params,
                        measured_params,
                        get_new_coordinates,
                        cost_val,
                        stopping_condition,
                        max_num_attempts=250):
    # requirements, if given, should take a 'candidate' or 'measurement' (parameter coordinates) and
    # return True or False depending on whether the settings or resulting measurement meet a set of requirements

    optimization = Optimization(variable_params,
                                measured_params,
                                stopping_condition,
                                cost_val,
                                get_new_coordinates,
                                reqs=None,
                                from_completed_dataset=True)


    optimization.run_id = run_id

    while not stopping_condition(variable_params, optimization.best) \
            and not optimization.num_attempts == max_num_attempts:
        # check new parameter settings, select one to move to
        new_coordinates = get_new_coordinates(optimization.current, variable_params)
        optimization.num_attempts += 1

        for coordinates in new_coordinates:
            # add location to list of locations checked
            optimization.checked.append(coordinates)
            # ToDo: this is not general and needs to be separated out from the rest of it
            if cost_val(variable_params, measured_params, coordinates) < cost_val(variable_params, measured_params, optimization.current):
                optimization.current = coordinates

        if cost_val(variable_params, measured_params, optimization.current) < cost_val(variable_params, measured_params, optimization.best):
            optimization.best = optimization.current

    print(f"Best value: {get_measurement_from_data(variable_params, measured_params, optimization.best)}")
    return optimization


def try_many(num_attempts,
             run_id,
             variable_params,
             measured_params,
             get_new_coordinates,
             cost_val,
             stopping_condition,
             success_condition=None):
    # success_condition: a function that takes the best_value and returns True for a success, otherwise False
    if success_condition is not None:
        successes = 0
    measurements_done = 0
    iterations = 0
    best_value = []
    best_coordinates = []
    starts = []

    for attempt in range(0, num_attempts):
        optimization = optimize_from_runid(run_id,
                                           variable_params,
                                           measured_params,
                                           get_new_coordinates,
                                           cost_val,
                                           stopping_condition,
                                           max_num_attempts=150)

        iterations += len(optimization.checked)
        measurements_done += len(np.unique(optimization.checked))
        starts.append(optimization.start)
        best_coordinates.append(optimization.best)

        value = get_measurement_from_data(variable_params, measured_params, optimization.best)
        best_value.append(value)
        if success_condition is not None:
            if success_condition(value):
                successes += 1

    avg_measurements = measurements_done / num_attempts
    avg_num_iterations = iterations / num_attempts

    # if num_attempts % 10 == 0:
    #     print(num_attempts)

    if success_condition is not None:
        return best_coordinates, best_value, starts, avg_measurements, avg_num_iterations, successes/num_attempts
    else:
        return best_coordinates, best_value, starts, avg_measurements, avg_num_iterations
