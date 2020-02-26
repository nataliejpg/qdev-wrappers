import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
from qdev_wrappers.fitting.helpers import load_xarrays
from qdev_wrappers.doNd import make_filename


def readout_fidelity(run_id, conn=None,
                     re_name='cavity_real_response',
                     im_name='cavity_imaginary_response',
                     qubit=1,
                     plot=True):
    re_array, im_array = load_xarrays(run_id, re_name, im_name, conn=conn)
    subfolder = 'readout_fidelity'
    title_text = subfolder.title() + ' Q{} (#{})'.format(qubit, run_id)

    # unclassified
    zero_x = re_array.sel(pi_pulse_present=0).values
    one_x = re_array.sel(pi_pulse_present=1).values
    zero_y = im_array.sel(pi_pulse_present=0).values
    one_y = im_array.sel(pi_pulse_present=1).values
    classified_fdict = learn_classified(zero_x, zero_y, one_x, one_y)
    if plot:
        c_figs, c_legs = _plot_classified(zero_x, zero_y, one_x, one_y,
                                          classified_fdict)

    # unclassified
    all_x = re_array.values.flatten()
    all_y = im_array.values.flatten()
    unclassified_fdict = learn_unclassified(
        all_x, all_y, centers=classified_fdict['centers'])
    if plot:
        u_figs, u_legs = _plot_unclassified(all_x, all_y,
                                            unclassified_fdict)

    # save plots
    if plot:
        figs = c_figs + u_figs
        legs = c_legs + u_legs
        for i, fig in enumerate(figs):
            title = fig.suptitle(title_text)
            filename = make_filename(
                run_id, index=i, conn=conn,
                analysis=True, extension=f'q{qubit}_{subfolder}')
            fig.savefig(filename, dpi=500, bbox_extra_artists=(legs[i], title),
                        bbox_inches='tight')
            plt.close()

    return {'classified': classified_fdict, 'unclassified': unclassified_fdict}


def state_learning(run_id, conn=None,
                   re_name='cavity_real_response',
                   im_name='cavity_imaginary_response',
                   qubit=1,
                   subfolder=None,
                   centers=None,
                   plot=True,
                   index_start=0):
    re_array, im_array = load_xarrays(run_id, re_name, im_name, conn=conn)
    all_x = re_array.values.flatten()
    all_y = im_array.values.flatten()
    unclassified_fdict = learn_unclassified(
        all_x, all_y, centers=centers)
    if plot:
        figs, legs = _plot_unclassified(all_x, all_y,
                                        unclassified_fdict)
        title_text = 'Single Shot Q{} (#{})'.format(qubit, run_id)
        for i, fig in enumerate(figs):
            title = fig.suptitle(title_text)
            filename = make_filename(
                run_id, index=index_start + i, analysis=False,
                extension=f'q{qubit}_readout', conn=conn)
            fig.savefig(filename, dpi=500, bbox_extra_artists=(legs[i], title),
                        bbox_inches='tight')
            plt.close()
    return unclassified_fdict


def learn_classified(zero_x_data, zero_y_data,
                     one_x_data, one_y_data):
    angle = angle_from_centers(np.array([[zero_x_data, zero_y_data],
                                         [one_x_data, one_y_data]]))
    zero_z_data = project(zero_x_data, zero_y_data, angle)
    one_z_data = project(one_x_data, one_y_data, angle)
    min_z = min(np.append(zero_z_data, one_z_data))
    max_z = max(np.append(zero_z_data, one_z_data))
    bins = np.linspace(min_z, max_z, 100)
    zero_hist, bin_edges = np.histogram(zero_z_data, bins)
    one_hist, bin_edges = np.histogram(one_z_data, bins)
    num_reps = float(len(zero_x_data))
    zero_cumsum = np.cumsum(zero_hist) / num_reps
    one_cumsum = np.cumsum(one_hist) / num_reps
    max_separation = np.amax(abs(zero_cumsum - one_cumsum))
    max_separation_index = np.argmax(abs(zero_cumsum - one_cumsum))
    decision_value = bins[max_separation_index]
    centers = np.array([[np.mean(zero_x_data), np.mean(zero_y_data)],
                        [np.mean(one_x_data), np.mean(one_y_data)]])
    return {'angle': angle,
            'decision_value': decision_value,
            'centers': centers,
            'max_separation': max_separation,
            'zero_z': zero_z_data,
            'one_z': one_z_data}


def learn_unclassified(x_data, y_data, centers=None):
    all_data = np.array(list(zip(x_data, y_data)))
    if centers is None:
        n_init = 10
        init = 'k-means++'
    else:
        n_init = 1
        init = centers
    kmeans = KMeans(n_clusters=2, init=init, n_init=n_init).fit(all_data)
    one_x, one_y = kmeans.cluster_centers_[1]
    angle = angle_from_centers(kmeans.cluster_centers_)
    zero_z = project(*kmeans.cluster_centers_[0], angle)
    one_z = project(*kmeans.cluster_centers_[1], angle)
    decision_value = one_z + (zero_z - one_z) / 2
    zero_indices = np.where(kmeans.labels_ == 0)
    one_indices = np.where(kmeans.labels_ == 1)
    zero_z_data = project(x_data, y_data, angle)[tuple(zero_indices)]
    one_z_data = project(x_data, y_data, angle)[tuple(one_indices)]
    return {'angle': angle,
            'decision_value': decision_value,
            'centers': kmeans.cluster_centers_,
            'zero_z': zero_z_data,
            'one_z': one_z_data,
            'kmeans': kmeans}


def angle_from_centers(centres):
    x_diff = np.mean(centres[0][0]) - np.mean(centres[1][0])
    y_diff = np.mean(centres[0][1]) - np.mean(centres[1][1])
    angle = np.angle(x_diff + 1j * y_diff)
    return angle


def project(x_data, y_data, angle):
    return x_data * np.cos(angle) + y_data * np.sin(angle)


def decide_state(x_data, y_data, angle, decision_value):
    z_data = x_data * np.cos(angle) + y_data * np.sin(angle)
    return z_data < decision_value


# def decide_state_groovy(x_data, y_data, kmeans):
#     all_data = np.array(list(zip(x_data, y_data)))
#     return kmeans.predict(all_data)

def _plot_classified(zero_x, zero_y, one_x, one_y,
                     classified_fdict):
    figs = []
    legs = []
    fig, leg = plot_scatter(zero_x, zero_y, one_x, one_y,
                            **classified_fdict)
    figs.append(fig)
    legs.append(leg)
    fig, leg = plot_hist_cumsum(**classified_fdict)
    figs.append(fig)
    legs.append(leg)
    return figs, legs


def _plot_unclassified(x_data, y_data,
                       unclassified_fdict):
    figs = []
    legs = []
    all_data = np.array(list(zip(x_data, y_data)))
    labels = unclassified_fdict['kmeans'].predict(all_data)
    zero_x = x_data[tuple(np.where(labels == 0))]
    zero_y = y_data[tuple(np.where(labels == 0))]
    one_x = x_data[tuple(np.where(labels == 1))]
    one_y = y_data[tuple(np.where(labels == 1))]
    fig, leg = plot_scatter(zero_x, zero_y, one_x, one_y,
                            **unclassified_fdict)
    figs.append(fig)
    legs.append(leg)
    fig, leg = plot_hist_cumsum(**unclassified_fdict)
    figs.append(fig)
    legs.append(leg)
    return figs, legs


def plot_scatter(zero_x_data, zero_y_data, one_x_data, one_y_data,
                 centers=None, decision_value=None, max_separation=None,
                 **kwargs):
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=3, ncols=1, hspace=0,
                           height_ratios=[1, 1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax1.plot(zero_x_data, zero_y_data, 'm.')
    ax2.plot(one_x_data, one_y_data, 'b.')
    ax3.plot(zero_x_data, zero_y_data, 'm.', alpha=0.3)
    ax3.plot(one_x_data, one_y_data, 'b.', alpha=0.3)
    zero_patch = mpatches.Patch(
        color='m', label='Ground State')
    one_patch = mpatches.Patch(
        color='b', label='Excited State')
    handles = [zero_patch, one_patch]
    if centers is not None:
        zero_center, = ax1.plot(
            centers[0][0], centers[0][1], 'wv',
            label='({:.3f}, {:.3f})'.format(centers[0][0], centers[0][1]))
        one_center, = ax2.plot(
            centers[1][0], centers[1][1], 'w^',
            label='({:.3f}, {:.3f})'.format(centers[1][0], centers[1][1]))
        handles.insert(1, zero_center)
        handles.insert(3, one_center)
        ax3.plot(centers[0][0], centers[0][1], 'wv')
        ax3.plot(centers[1][0], centers[1][1], 'w^')
        angle = angle_from_centers(centers)
        if decision_value is not None:
            min_x = min(np.append(zero_x_data, one_x_data))
            max_x = max(np.append(zero_x_data, one_x_data))
            min_y = min(np.append(zero_y_data, one_y_data))
            max_y = max(np.append(zero_y_data, one_y_data))
            bisector_x = decision_value * np.cos(angle)
            bisector_y = decision_value * np.sin(angle)
            x_diff = centers[0][0] - centers[1][0]
            y_diff = centers[0][1] - centers[1][1]
            m = - x_diff / y_diff
            c = bisector_y - m * bisector_x
            x = np.array([min_x, max_x])
            y = m * x + c
            if max_separation is not None:
                sep_line, = ax3.plot(
                    x, y, 'k--',
                    label='separation {:.2}'.format(max_separation))
                handles.append(sep_line)
            else:
                ax3.plot(x, y, 'k--')
            ax3.set_ylim([min_y, max_y])
            ax3.set_xlim([min_x, max_x])
    ax3.set_xlabel('Real')
    ax1.set_ylabel('Imag')
    ax2.set_ylabel('Imag')
    ax3.set_ylabel('Imag')
    min_y = min(np.append(zero_y_data, one_y_data))
    max_y = max(np.append(zero_y_data, one_y_data))
    ax1.set_ylim([min_y, max_y])
    ax2.set_ylim([min_y, max_y])
    legend = fig.legend(handles=handles,
                        bbox_to_anchor=(0.95, 0.6), loc='upper left')
    fig.set_size_inches(5, 7)
    return fig, legend


def plot_hist_cumsum(zero_z=None, one_z=None, decision_value=None,
                     max_separation=None, title=None, **kwargs):
    min_z = min(np.append(zero_z, one_z))
    max_z = max(np.append(zero_z, one_z))
    bins = np.linspace(min_z, max_z, 100)
    zero_hist, bin_edges = np.histogram(zero_z, bins)
    one_hist, bin_edges = np.histogram(one_z, bins)
    num_reps = float(len(zero_z) + len(one_z)) / 2
    zero_cumsum = np.cumsum(zero_hist) / num_reps
    one_cumsum = np.cumsum(one_hist) / num_reps
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=2, ncols=1, hspace=0,
                           height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _plot_histogram(ax1, zero_z, one_z, bin_edges,
                    decision_value)
    zero_patch = mpatches.Patch(color='m', label='Ground State')
    one_patch = mpatches.Patch(color='b', label='Excited State')
    line = _plot_cum_sum(ax2, zero_cumsum, one_cumsum, bin_edges,
                         decision_value, max_separation)
    legend = fig.legend(handles=[zero_patch, one_patch, line],
                        bbox_to_anchor=(0.95, 0.5), loc='upper left')
    fig.set_size_inches(5, 7)
    return fig, legend


def _plot_histogram(axes, zero_z_data, one_z_data, bin_edges,
                    decision_value):
    axes.hist(zero_z_data, bins=bin_edges,
              label='Ground State', color='m', alpha=0.5)
    axes.hist(one_z_data, bins=bin_edges,
              label='Excited State', color='b', alpha=0.5)
    axes.axvline(decision_value, c='k', ls='--',)
    axes.set_ylabel('Counts')


def _plot_cum_sum(axes, zero_cumsum, one_cumsum, bin_edges,
                  decision_value, max_separation=None):
    axes.plot(bin_edges[:-1], zero_cumsum, color='m')
    axes.plot(bin_edges[:-1], one_cumsum, color='b')
    axes.set_xlabel('Projected Values')
    axes.set_ylabel('Fraction of Total Counts')
    label = 'separation'
    if max_separation is not None:
        label += ' {:.2}'.format(max_separation)
    line = axes.axvline(decision_value, c='k', ls='--',
                        label=label)
    return line
