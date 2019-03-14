import os
from qcodes.dataset.data_set import load_by_id
from qcodes.dataset.experiment_container import load_experiment
from qcodes.config.config import Config
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

fidelity_result = namedtuple(
    'FidelityResult', [
        'no_pi_mean_coords', 'pi_mean_coords',
        'angle', 'tmp_bins',
        'no_pi_projected', 'pi_projected',
        'no_pi_cumsum', 'pi_cumsum',
        'max_separation', 'max_separation_index'])


def calculate_fidelity(no_pi_real, no_pi_imaginary,
                       pi_real, pi_imaginary):
    num_reps = float(len(no_pi_real))

    # find centre coordiantes
    no_pi_real_mean = np.mean(no_pi_real)
    no_pi_imaginary_mean = np.mean(no_pi_imaginary)
    pi_real_mean = np.mean(pi_real)
    pi_imaginary_mean = np.mean(pi_imaginary)

    # find imaginary and real distance between centres
    real_difference = no_pi_real_mean - pi_real_mean
    imaginary_difference = no_pi_imaginary_mean - pi_imaginary_mean

    # find angle between line and real axis
    angle = np.angle(real_difference + 1j * imaginary_difference)
    deg_angle = 180 * angle / np.pi

    # rotate data clockwise by angle and project onto 'real' / 'x' axis
    no_pi_projected = no_pi_real * \
        np.cos(angle) + no_pi_imaginary * np.sin(angle)
    pi_projected = pi_real * np.cos(angle) + pi_imaginary * np.sin(angle)

    # find extremes of projected data and create bins for histogram
    max_projected = max(np.append(no_pi_projected, pi_projected))
    min_projected = min(np.append(no_pi_projected, pi_projected))
    tmp_bins = np.linspace(min_projected, max_projected, 100)

    # histogram data from projected data
    no_pi_histogram, binedges = np.histogram(no_pi_projected, tmp_bins)
    pi_histogram, binedges = np.histogram(pi_projected, tmp_bins)

    # find the cumulative sum data from the histogram data
    no_pi_cumsum = np.cumsum(no_pi_histogram)
    pi_cumsum = np.cumsum(pi_histogram)

    # find the maximum separation of the cumulative sums and the index where
    # it occurs
    max_separation = np.amax(abs(no_pi_cumsum - pi_cumsum)) / num_reps
    max_separation_index = np.argmax(abs(no_pi_cumsum - pi_cumsum))

    return fidelity_result(
        no_pi_mean_coords=(no_pi_real_mean, no_pi_imaginary_mean),
        pi_mean_coords=(pi_real_mean, pi_imaginary_mean),
        no_pi_projected=no_pi_projected,
        pi_projected=pi_projected,
        angle=deg_angle,
        tmp_bins=tmp_bins,
        no_pi_cumsum=no_pi_cumsum,
        pi_cumsum=pi_cumsum,
        max_separation=max_separation,
        max_separation_index=max_separation_index)


def make_analysis_filename(run_id, append_string=''):
    dataset = load_by_id(run_id)
    experiment = load_experiment(dataset.exp_id)
    db_path = Config()['core']['db_location']
    db_folder = os.path.dirname(db_path)
    plot_folder_name = '{}_{}'.format(experiment.sample_name, experiment.name)
    plot_folder = os.path.join(db_folder, plot_folder_name)
    os.makedirs(plot_folder, exist_ok=True)
    filename = 'analysis_{}_{}.png'.format(run_id, append_string)
    return os.path.join(plot_folder, filename)


def plot_no_pi_data(axes, no_pi_real, no_pi_imaginary, no_pi_mean_coords):
    axes.plot(no_pi_real, no_pi_imaginary, 'm.', label='no pi')
    axes.plot(*no_pi_mean_coords, 'ko',
              label='({:.1e}, {:.1e})'.format(*no_pi_mean_coords))
    axes.set_xlabel('Real')
    axes.set_ylabel('Imaginary')
    axes.legend(loc=1, numpoints=1)


def plot_pi_data(axes, pi_real, pi_imaginary, pi_mean_coords):
    axes.plot(pi_real, pi_imaginary, 'b.', label='pi')
    axes.plot(*pi_mean_coords, 'ko',
              label='({:.1e}, {:.1e})'.format(*pi_mean_coords))
    axes.set_xlabel('Real')
    axes.set_ylabel('Imaginary')
    axes.legend(loc=1, numpoints=1)


def plot_pi_and_no_pi_data(axes, no_pi_real, no_pi_imaginary,
                           pi_real, pi_imaginary,
                           no_pi_mean_coords, pi_mean_coords,
                           angle):
    axes.plot(no_pi_real, no_pi_imaginary, 'm.', label='no pi')
    axes.plot(pi_real, pi_imaginary, 'b.', label='pi')
    axes.plot([no_pi_mean_coords[0], pi_mean_coords[0]],
              [no_pi_mean_coords[1], pi_mean_coords[1]],
              label='rotation_angle = {:.1}'.format(angle))
    axes.set_xlabel('Real')
    axes.set_ylabel('Imaginary')
    axes.legend(loc=1, numpoints=1)


def plot_histogram(axes, tmp_bins, no_pi_projected, pi_projected):
    axes.hist(no_pi_projected, bins=tmp_bins, label='no pi')
    axes.hist(pi_projected, bins=tmp_bins, label='pi')
    axes.legend(loc=1, numpoints=1)


def plot_cum_sum(axes, no_pi_cumsum, pi_cumsum,
                 max_separation_index, max_separation):
    x = np.arange(99)
    axes.plot(x, no_pi_cumsum, label='no pi')
    axes.plot(x, pi_cumsum, label='pi')
    axes.axvline(x=max_separation_index, color='k', linestyle='dashed',
                 linewidth=1, label='{:.2} seperation'.format(max_separation))
    axes.legend(loc=2)


def make_scatter_fig(run_id):
    data = load_by_id(run_id)
    experiment = load_experiment(data.exp_id)
    (no_pi_real, no_pi_imaginary,
        pi_real, pi_imaginary) = extract_pi_no_pi_data(data)
    fidelity_info = calculate_fidelity(no_pi_real, no_pi_imaginary,
                                       pi_real, pi_imaginary)
    fig, axes = plt.subplots(3, squeeze=False, sharex=True, sharey=True)
    plot_no_pi_data(axes[0, 0], no_pi_real, no_pi_imaginary,
                    fidelity_info.no_pi_mean_coords)
    plot_pi_data(axes[1, 0], pi_real, pi_imaginary,
                 fidelity_info.no_pi_mean_coords)
    plot_pi_and_no_pi_data(axes[2, 0],
                           no_pi_real, no_pi_imaginary,
                           pi_real, pi_imaginary,
                           fidelity_info.no_pi_mean_coords,
                           fidelity_info.pi_mean_coords,
                           fidelity_info.angle)
    fig.suptitle('{} on {} (run ID {})'.format(
        experiment.name, experiment.sample_name, run_id))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = make_analysis_filename(
        run_id, append_string="readout_fidelity_scatter_plots")
    fig.set_size_inches(5, 7)
    fig.savefig(filename)


def make_cum_sum_histogram_fig(run_id):
    data = load_by_id(run_id)
    experiment = load_experiment(data.exp_id)
    fidelity_info = calculate_fidelity(*extract_pi_no_pi_data(data))
    fig, axes = plt.subplots(2, squeeze=False)
    plot_histogram(axes[0, 0],
                   fidelity_info.tmp_bins,
                   fidelity_info.no_pi_projected, fidelity_info.pi_projected)
    
    
    plot_cum_sum(axes[1, 0],
                 fidelity_info.no_pi_cumsum, fidelity_info.pi_cumsum,
                 fidelity_info.max_separation_index,
                 fidelity_info.max_separation)
    fig.suptitle('{} on {} (run ID {})'.format(
        experiment.name, experiment.sample_name, run_id))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = make_analysis_filename(
        run_id, append_string="readout_fidelity_cum_sum_hist")
    fig.set_size_inches(5, 7)
    fig.savefig(filename)


def extract_pi_no_pi_data(dataset):
    no_pi_real = np.array(dataset.get_data(
        'cavity_real_response',
        select_condition='pi_pulse_present = 0')).flatten()

    no_pi_imaginary = np.array(dataset.get_data(
        'cavity_imaginary_response',
        select_condition='pi_pulse_present = 0')).flatten()

    pi_real = np.array(dataset.get_data(
        'cavity_real_response',
        select_condition='pi_pulse_present = 1')).flatten()

    pi_imaginary = np.array(dataset.get_data(
        'cavity_imaginary_response',
        select_condition='pi_pulse_present = 1')).flatten()

    return no_pi_real, no_pi_imaginary, pi_real, pi_imaginary


def fidelity_info_from_run_id(run_id):
    data = load_by_id(run_id)
    return calculate_fidelity(*extract_pi_no_pi_data(data))


def make_readout_fidelity_plots(run_id):
    make_scatter_fig(run_id)
    make_cum_sum_histogram_fig(run_id)
