import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
from qdev_wrappers.analysis.base import AnalyserBase


class ReadoutFidelityAnalyser(AnalyserBase):
    METHOD = 'ReadoutFidelity'
    MODEL_PARAMETERS = {'readout_fidelity': {},
                        'angle': {},
                        'decision_value': {},
                        'np_projected': {'parameter_class': 'arr'},
                        'p_projected': {'parameter_class': 'arr'},
                        'bin_edges': {'parameter_class': 'arr'},
                        'np_cumsum': {'parameter_class': 'arr'},
                        'p_cumsum': {'parameter_class': 'arr'}
                        }
    MEASUREMENT_PARAMETERS = {'real': {},
                              'imaginary': {}}
    EXPERIMENT_PARAMETERS = {'pi_pulse_present': {}}

    def analyse(self, real, imaginary, pi_pulse_present):
        np_indices = np.argwhere(np.array(pi_pulse_present) == 0)
        p_indices = np.argwhere(np.array(pi_pulse_present) == 1)
        np_real = real[np_indices]
        np_imaginary = imaginary[np_indices]
        p_real = real[p_indices]
        p_imaginary = imaginary[p_indices]

        # find centre coordiantes
        np_real_mean = np.mean(np_real)
        np_imaginary_mean = np.mean(np_imaginary)
        p_real_mean = np.mean(p_real)
        p_imaginary_mean = np.mean(p_imaginary)

        # find imaginary and real distance between centres
        real_difference = p_real_mean - np_real_mean
        imaginary_difference = p_imaginary_mean - np_imaginary_mean

        # find angle between line and real axis
        angle = np.angle(real_difference + 1j * imaginary_difference)
        self.model_parameters.angle._save_val(angle)

        # rotate data clockwise by angle and project onto 'real' / 'x' axis
        np_projected = np_real * \
            np.cos(angle) + np_imaginary * np.sin(angle)
        p_projected = p_real * \
            np.cos(angle) + p_imaginary * np.sin(angle)
        self.model_parameters.np_projected._save_array(np_projected)
        self.model_parameters.p_projected._save_array(p_projected)

        # find extremes of projected data and create bins for histogram
        max_projected = max(np.append(np_projected, p_projected))
        min_projected = min(np.append(np_projected, p_projected))
        bins = np.linspace(min_projected, max_projected, 100)
        self.model_parameters.bin_edges._save_array(bins)

        # histogram data from projected data
        np_histogram, binedges = np.histogram(np_projected, bins)
        p_histogram, binedges = np.histogram(p_projected, bins)

        # find the cumulative sum data from the histogram data
        num_reps = float(len(np_real))
        np_cumsum = np.cumsum(np_histogram) / num_reps
        p_cumsum = np.cumsum(p_histogram) / num_reps
        self.model_parameters.np_cumsum._save_array(np_cumsum)
        self.model_parameters.p_cumsum._save_array(p_cumsum)

        # find the maximum separation of the cumulative sums and the index
        # where it occurs
        max_separation = np.amax(abs(np_cumsum - p_cumsum))
        max_separation_index = np.argmax(abs(np_cumsum - p_cumsum))

        self.model_parameters.readout_fidelity._save_val(max_separation)
        self.model_parameters.decision_value._save_val(
            bins[max_separation_index])


def _plot_pnp_data(axes, real, imaginary, p=True):
    color = 'b.' if p else 'm.'
    axes.plot(real, imaginary, color)
    axes.set_ylabel('Imaginary')


def plot_scatter(real, imaginary, pi_pulse_present, title=None):
    np_indices = np.argwhere(np.array(pi_pulse_present) == 0)
    p_indices = np.argwhere(np.array(pi_pulse_present) == 1)
    np_real = real[np_indices]
    np_imaginary = imaginary[np_indices]
    p_real = real[p_indices]
    p_imaginary = imaginary[p_indices]

    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=3, ncols=1, hspace=0,
                           height_ratios=[1, 1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    _plot_pnp_data(ax1, np_real, np_imaginary, p=False)
    _plot_pnp_data(ax2, p_real, p_imaginary, p=True)
    _plot_pnp_data(ax3, np_real, np_imaginary, p=False)
    _plot_pnp_data(ax3, p_real, p_imaginary, p=True)
    ax3.set_xlabel('Real')
    np_patch = mpatches.Patch(
        color='m', label='Ground State\n({:.1e}, {:.1e})'.format(
            np.mean(np_real), np.mean(np_imaginary)))
    p_patch = mpatches.Patch(
        color='b', label='Excited State\n({:.1e}, {:.1e})'.format(
            np.mean(p_real), np.mean(p_imaginary)))

    legend = fig.legend(handles=[np_patch, p_patch],
               bbox_to_anchor=(0.95, 0.5), loc='upper left')
    if title is not None:
        fig.suptitle(title)
    fig.set_size_inches(5, 7)
    return fig, legend


def _plot_histogram(axes, np_projected, p_projected, bin_edges,
                    decision_value):
    axes.hist(np_projected, bins=bin_edges,
              label='Ground State', color='m')
    axes.hist(p_projected, bins=bin_edges,
              label='Excited State', color='b')
    axes.axvline(x=decision_value, color='k',
                 linestyle='dashed', linewidth=1)
    axes.set_ylabel('Counts')


def _plot_cum_sum(axes, np_cumsum, p_cumsum, bin_edges,
                  decision_value, readout_fidelity):
    axes.plot(bin_edges[:-1], np_cumsum, color='m')
    axes.plot(bin_edges[:-1], p_cumsum, color='b')
    line = axes.axvline(x=decision_value, color='k',
                        linestyle='dashed', linewidth=1,
                        label='{:.2} seperation'.format(readout_fidelity))
    axes.set_xlabel('Projected Values')
    axes.set_ylabel('Fraction of Total Counts')
    return line


def plot_hist_cumsum(np_projected, p_projected, bin_edges, decision_value,
                     np_cumsum, p_cumsum, readout_fidelity, title=None):
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=2, ncols=1, hspace=0,
                           height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _plot_histogram(ax1, np_projected, p_projected, bin_edges,
                    decision_value)
    line = _plot_cum_sum(ax2, np_cumsum, p_cumsum, bin_edges,
                         decision_value, readout_fidelity)
    np_patch = mpatches.Patch(color='m', label='Ground State')
    p_patch = mpatches.Patch(color='b', label='Excited State')
    if title is not None:
        fig.suptitle(title)
    legend = fig.legend(handles=[np_patch, p_patch, line],
               bbox_to_anchor=(0.95, 0.5), loc='upper left')
    fig.set_size_inches(5, 7)
    return fig, legend


def plot_readout_fidelity(data, title=None):
    scatter_fig = plot_scatter(data['real'], data['imaginary'],
                               data['pi_pulse_present'],
                               title=title)
    hist_cumsum_fig = plot_hist_cumsum(
        data['np_projected'], data['p_projected'], data['bin_edges'],
        data['decision_value'], data['np_cumsum'], data['p_cumsum'],
        data['readout_fidelity'], title=title)
    return scatter_fig, hist_cumsum_fig
