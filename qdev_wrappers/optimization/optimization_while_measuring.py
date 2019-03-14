import qcodes as qc
import numpy as np
from numpy.random import randint
from qcodes.dataset.data_export import get_data_by_id
from itertools import product
from qcodes.dataset.measurements import Measurement
from Optimization import Optimization


def measure_candidate(params, indicies_tuple, measurement):
    for i, param in enumerate(params):
        index = indicies_tuple[i]
        value = params[param]['values'][index]
        param.set(value)  # fix this, so it works!!
    measurement()


def make_params_dict(*params):
    # params should be in the format (parameter, step_size)
    for param in params:
        param_dict[param.full_name] = {'param': param[0],
                                       'min_step_size': param[1]}


def optimize_while_measuring(
                        variable_params,
                        measured_params,
                        get_new_coordinates,
                        cost_val,
                        stopping_condition,
                        max_num_attempts=150):
    # requirements, if given, should take a 'candidate' or 'measurement' (parameter coordinates) and
    # return True or False depending on whether the settings or resulting measurement meet a set of requirements

    optimization = Optimization(variable_params,
                                measured_params,
                                stopping_condition,
                                cost_val,
                                get_new_coordinates,
                                reqs=None,
                                from_completed_dataset=True)



    meas = Measurement()

    setpoints = []
    for parameter in variable_params:
        setpoints.append(parameter['name'])
        meas.register_custom_parameter(name=parameter['name'],
                                       label=parameter['label'],
                                       unit=parameter['unit'])
    for parameter in measured_params:
        meas.register_custom_parameter(name=parameter['name'],
                                       label=parameter['label'],
                                       unit=parameter['unit'],
                                       setpoints=tuple(setpoints))

    with meas.run() as datasaver:
        optimization.run_id = datasaver.run_id

        while not stopping_condition(variable_params, optimization.best) \
                and not optimization.num_attempts == max_num_attempts:
            # check new parameter settings, select one to move to
            new_coordinates = get_new_coordinates(optimization.current, variable_params)
            optimization.num_attempts += 1

            for coordinates in new_coordinates:
                measurements = []
                # add location to list of locations checked
                optimization.checked.append(coordinates)
                # save 'measurements' at each point checked
                results = get_results(variable_params, measured_params, coordinates)
                measurements.append(results[-1])
                datasaver.add_result(*results)

                # ToDo: this is not general and needs to be separated out from the rest of it
                if cost_val(variable_params, measured_params, coordinates) < cost_val(variable_params, measured_params, optimization.current):
                    optimization.current = coordinates

            if cost_val(variable_params, measured_params, optimization.current) < cost_val(variable_params, measured_params, optimization.best):
                 optimization.best = optimization.current

        print(f"Best value: {get_measurement_from_data(variable_params, measured_params, optimization.best)}")

        return optimization