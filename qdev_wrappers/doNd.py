import numpy as np
from collections import namedtuple
import os
import time
import progressbar
from qcodes.dataset.experiment_container import load_experiment
from qcodes.config.config import Config
from qcodes.dataset.measurements import Measurement
from qcodes.instrument.parameter import _BaseParameter
from qcodes.dataset.plotting import plot_by_id
from qcodes.dataset.data_set import load_by_id

Task = namedtuple('Task', 'type callable')


def make_filename(run_id, index=None, analysis=False, extension=None, conn=None):
    dataset = load_by_id(run_id, conn=conn)
    experiment = load_experiment(dataset.exp_id, conn=conn)
    db_path = Config()['core']['db_location']
    db_folder = os.path.dirname(db_path)
    plot_folder_name = '{}_{}'.format(experiment.sample_name, experiment.name)
    plot_folder = os.path.join(db_folder, plot_folder_name)
    if analysis:
        plot_folder = os.path.join(plot_folder, 'analysis')
    os.makedirs(plot_folder, exist_ok=True)
    filename = '{}'.format(run_id)
    if extension is not None:
        filename += '_{}'.format(extension)
    if index is not None:
        filename += '_{}'.format(index)
    filename += '.png'
    plot_path = os.path.join(plot_folder, filename)
    return plot_path


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
            results = []
            if task_tuple.type == 'param':
                results.append((task_tuple.callable, task_tuple.callable.get()))
            else:
                task_tuple.callable()
        datasaver.add_result(*results)
    axes, clb = plot_by_id(run_id)
    for i, ax in enumerate(axes):
        ax.figure.savefig(make_filename(run_id, i))
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
            results = []
            for task_tuple in task_tuples:
                if task_tuple.type == 'param':
                    results.append((task_tuple.callable, task_tuple.callable.get()))
                else:
                    task_tuple.callable()
            datasaver.add_result((param_set, set_point),
                     *results)
            points_taken += 1
            current_time = time.time()
            if current_time - last_time >= refresh_time:
                last_time = current_time
                progress_bar.update(points_taken)
        progress_bar.update(points_taken)
    axes, clb = plot_by_id(run_id)
    for i, ax in enumerate(axes):
        ax.figure.savefig(make_filename(run_id, i))
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
                results = []
                for task_tuple in task_tuples:
                    if task_tuple.type == 'param':
                        results.append((task_tuple.callable, task_tuple.callable.get()))
                    else:
                        task_tuple.callable()
                datasaver.add_result((param_set1, set_point1),
                                     (param_set2, set_point2),
                                     *results)
                points_taken += 1
                current_time = time.time()
                if current_time - last_time >= refresh_time:
                    last_time = current_time
                    progress_bar.update(points_taken)
        progress_bar.update(points_taken)
    axes, clb = plot_by_id(run_id)
    for i, ax in enumerate(axes):
        ax.figure.savefig(make_filename(run_id, i))
    return run_id

