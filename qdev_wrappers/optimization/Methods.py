from numpy.random import randint


class BestNeighbor:

    def get_neighbours(self, current_coordinates, variable_params):
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

    def select_new_location(self, neighbours):
        for candidate in neighbours:
            if self.requirements(candidate):
                if (cost_val(candidate) < cost_val(current_coordinates)):
                    current_coordinates = candidate

    def move_to_best_neighbor(self, current_coordinates, new_coordinates, measurements):
        for meas in measurements:
        if (self.cost_val(meas) < cost_val(current_coordinates)):
                current_coordinates = candidate

    def cost_val(self):
        pass

    def stopping_condition(self):
        pass


class WeightedMovement:
    pass

class PendulumSearch:
    pass