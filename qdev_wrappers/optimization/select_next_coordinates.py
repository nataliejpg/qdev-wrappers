import qcodes as qc
import numpy as np
from numpy.random import randint
from itertools import product


def get_neighbours(current_coordinates, variable_params):
    """returns a list of tuples, each with indices for neighboring points"""
    next_param_coordinates = {}

    for i, param in enumerate(variable_params):
        datalength = len(param['values'])
        step = randint(0, param['max_step']) + 1
        next_param_coordinates[param['name']] = [(current_coordinates[i] - step) % datalength,
                                                 (current_coordinates[i] + step) % datalength]

    param_combinations = product(*[set(v) for v in next_param_coordinates.values()])
    neighbours = [combination for combination in param_combinations]
    return neighbours


def weighted_movement(current_coordinates, variable_params):
    pass


def pedulum_search(current_coordinates, variable_params):
    pass
