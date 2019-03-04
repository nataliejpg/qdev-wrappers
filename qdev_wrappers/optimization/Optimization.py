from numpy.random import randint
from optimization_from_completed_data import get_measurement_from_data
from Methods import ReadoutFidelityOptimization


class Optimization:
    def __init__(self, method: ReadoutFidelityOptimization):
        # self.checked = []
        self.params = method.params
        self.num_attempts = 0
        self.start = [param() for param in self.params]

        self.current = self.start
        self.best_cost_val = method.measurement_function()
        self.best = self.start


class OptimizationFromDataSet:
    def __init__(self, params):
        # self.checked = []
        self.params = params
        self.num_attempts = 0
        self.start = [randint(0, len(param['values'])) for param in params]
        self.best_cost_val = get_measurement_from_data(self.start, params, params)
        # ToDo: won't work until get_measurement_from_data can differentiate between measured and variable params
        self.best = self.start
        self.current = self.start


