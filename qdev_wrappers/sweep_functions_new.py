import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from qcodes.dataset.measurements import Measurement
from qcodes.instrument.parameter import _BaseParameter
from qcodes.dataset.plotting import plot_by_id
from qcodes.dataset.data_set import load_by_id
from qcodes.dataset import experiment_container
import time
import progressbar

Task = namedtuple('Task', 'type callable')

def do0d(*tasks):
    '''Measure param_meas.'''    
    meas = Measurement()
    meas.write_period = 1.
    task_tuples = []
    for task in tasks:
        if isinstance(task, _BaseParameter):
            meas.register_parameter(task, setpoints=())
            task_tuples.append(Task('param', task))
        else:
            task_tuples.append(Task('function', task))

    with meas.run() as datasaver:
        run_id = datasaver.run_id
        for task_tuple in task_tuples:
            if task_tuple.type == 'param':
                datasaver.add_result((task_tuple.callable, task_tuple.callable.get()))
            else:
                task_tuple.callable()
    return run_id

def do1d(param_set, start, stop, num_points, delay, *tasks):
    '''Scan 1D of param_set and measure param_meas.'''
    meas = Measurement()
    refresh_time = 1. # in s
    meas.write_period = refresh_time
    meas.register_parameter(param_set)
    param_set.post_delay = delay
    task_tuples = []
    for task in tasks:
        if isinstance(task, _BaseParameter):
            meas.register_parameter(task, setpoints=(param_set,))
            task_tuples.append(Task('param', task))
        else:
            task_tuples.append(Task('function', task))
    progress_bar = progressbar.ProgressBar(max_value=num_points)
    points_taken = 0
    time.sleep(0.1)
    
    with meas.run() as datasaver:
        run_id = datasaver.run_id
        last_time = time.time()
        for set_point in np.linspace(start, stop, num_points):
            param_set.set(set_point)
            for task_tuple in task_tuples:
                if task_tuple.type == 'param':
                    datasaver.add_result((param_set, set_point),
                                         (task_tuple.callable, task_tuple.callable.get()))
                else:
                    task_tuple.callable()
            points_taken += 1
            current_time = time.time()
            if current_time - last_time >= refresh_time:
                last_time = current_time
                progress_bar.update(points_taken)
        progress_bar.update(points_taken)
    return run_id


def do2d(param_set1, start1, stop1, num_points1, delay1,
         param_set2, start2, stop2, num_points2, delay2,
         *tasks):
    '''Scan 2D of param_set and measure param_meas.'''
    meas = Measurement()
    refresh_time = 1. # in s
    meas.write_period = refresh_time
    meas.register_parameter(param_set1)
    param_set1.post_delay = delay1
    meas.register_parameter(param_set2)
    param_set2.post_delay = delay2
    task_tuples = []
    for task in tasks:
        if isinstance(task, _BaseParameter):
            meas.register_parameter(task, setpoints=(param_set1, param_set2))
            task_tuples.append(Task('param', task))
        else:
            task_tuples.append(Task('function', task))
    progress_bar = progressbar.ProgressBar(max_value=num_points1 * num_points2)
    points_taken = 0
    time.sleep(0.1)
    
    with meas.run() as datasaver:
        run_id = datasaver.run_id
        last_time = time.time()
        for set_point1 in np.linspace(start1, stop1, num_points1):
            param_set1.set(set_point1)
            for set_point2 in np.linspace(start2, stop2, num_points2):
                param_set2.set(set_point2)
                for task_tuple in task_tuples:
                    if task_tuple.type == 'param':
                        datasaver.add_result((param_set1, set_point1),
                                             (param_set2, set_point2),
                                             (task_tuple.callable, task_tuple.callable.get()))
                    else:
                        task_tuple.callable()
                points_taken += 1
                current_time = time.time()
                if current_time - last_time >= refresh_time:
                    last_time = current_time
                    progress_bar.update(points_taken)
        progress_bar.update(points_taken)
    return run_id

