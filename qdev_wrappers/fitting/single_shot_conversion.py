import os
import copy
import qcodes as qc
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
from qcodes.dataset.data_export import load_by_id
from qcodes.dataset.experiment_container import load_experiment
from qcodes.dataset.plotting import _rescale_ticks_and_units
from qdev_wrappers.fitting.plotting import plot_on_a_plain_grid
from qdev_wrappers.fitting.helpers import load_xarrays
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.plotting import plot_by_id
from qdev_wrappers.doNd import make_filename
from qcodes.dataset.experiment_container import load_last_experiment


coord_label_lookup = {'alazar_controller_pulse_duration': ['Pulse Duration', 's'],
                      'alazar_controller_pulse_pulse_delay': ['Delay', 's'],
                      'alazar_controller_operation_index': ['Sequence Index', ''],
                      'alazar_controller_clifford_gate_count': ['# of Cliffords', '']}


# def make_filename(run_id, qubit=None, index=0, conn=None,
#                   subfolder=None, name_extension=None):
#     if conn is not None:
#         dataset = load_by_id(run_id, conn=conn)
#         exp = load_experiment(dataset.exp_id, conn=conn)
#         s_name = f'{exp.sample_name}_{exp.name}'
#     else:
#         s_name = 'unspecified_sample'
#     name_list = [f'{run_id}', f'{index}']
#     if name_extension is not None:
#         name_list.insert(1, name_extension)
#     if qubit is not None:
#         name_list.insert(1, f'q{qubit}')
#     name = '_'.join(name_list)
#     name += '.png'
#     join_list = ['..', 'figs', s_name]
#     if subfolder is not None:
#         join_list.append(subfolder)
#     plot_folder = os.path.join(*join_list)
#     os.makedirs(plot_folder, exist_ok=True)
#     fullname = os.path.join(plot_folder, name)
#     return fullname


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


def one_qubit_exp(run_id, fdict,
                  re_name='alazar_controller_ch_0_r_records_buffers_data',
                  im_name='alazar_controller_ch_0_i_records_buffers_data',
                  reps_name='alazar_controller_repetitions',
                  qubit=1,
                  subfolder=None,
                  plot=True,
                  index_start=0,
                  reps_name2=None,
                  save=False,
                  source_conn=None,
                  target_conn=None,
                  **setpoints):
    re_array, im_array = load_xarrays(run_id, re_name, im_name, conn=source_conn)
    for k, v in setpoints.items():
        re_array = re_array.sel(k=v)
        im_array = im_array.sel(k=v)
    all_x = re_array.values.flatten()
    all_y = im_array.values.flatten()
    coord_names = list(re_array.coords.keys())
    avg_names = [n for n in [reps_name, reps_name2] if n is not None]
    avg_inds = []
    for r_n in avg_names:
        avg_i = coord_names.index(r_n)
        avg_inds.append(avg_i)
        del coord_names[avg_i]
    projected = project(re_array, im_array, fdict['angle']).mean(
        dim=avg_names).values
    if 'kmeans' in fdict:
        all_data = np.array(list(zip(all_x, all_y)))
        probability = fdict['kmeans'].predict(all_data).reshape(
            re_array.values.shape).mean(axis=tuple(avg_inds))
    else:
        all_data = project(all_x, all_y, fdict['angle'])
        probability = (all_data < fdict['decision_value']).reshape(
            re_array.values.shape).mean(axis=tuple(avg_inds))

    projected_info = {'data': projected,
                      'label': f'Cavity Response Q{qubit}',
                      'unit': 'V'}
    probability_info = {'data': probability,
                        'label': f'Excited Population Q{qubit}',
                        'unit': ''}

    coord_dict = {}
    for coord_name in coord_names:
        coord_label, coord_unit = coord_label_lookup.get(
            coord_name, [coord_name, ''])
        coord_dict[coord_name] = {'data': re_array[coord_name].values,
                                  'label': coord_label,
                                  'unit': coord_unit}

    res_dict = {'cavity_response': projected_info,
                'excited_population': probability_info,
                **coord_dict}

    if save:
        if target_conn is not None:
            if target_conn is not None:
                target_exp = load_last_experiment(target_conn)
            else:
                target_exp = None
        meas = Measurement(exp=target_exp)
        for coord_name, coord_info in coord_dict.items():
            meas.register_custom_parameter(coord_name,
                                           label=coord_info['label'],
                                           unit=coord_info['unit'])
        meas.register_custom_parameter('cavity_response',
                                       label=projected_info['label'],
                                       unit=projected_info['unit'],
                                       setpoints=tuple(coord_names))
        meas.register_custom_parameter('excited_population',
                                       label=probability_info['label'],
                                       setpoints=tuple(coord_names))
        with meas.run() as datasaver:
            analysis_run_id = datasaver.run_id
            result = {}
            inds = []
            for coord_name, coord_info in coord_dict.items():
                    inds.append(len(coord_info['data']))
            for i, l in enumerate(inds):
                for j in range(l):
                    result[coord_names[i]] = coord_dict[coord_names[i]]['data'][j]
            for i in range(len(projected)):
                result = [('cavity_response', projected[i]),
                          ('excited_population', probability[i])]
                for coord_name, coord_info in coord_dict.items():
                    result.append((coord_name, coord_info['data'][i]))
                datasaver.add_result(*result)
        if plot:
            axes, cbar = plot_by_id(analysis_run_id, conn=target_conn)
            for i, ax in enumerate(axes):
                if i == 1:
                    ax.set_ylim(0, 1)
                filename = make_filename(
                    analysis_run_id, index=i, analysis=False,
                    extension=f'q{qubit}_{subfolder}', conn=target_conn)
                ax.figure.savefig(filename, dpi=500, bbox_inches='tight')
    elif plot:
        analysis_run_id = None
        plot_result(res_dict, 'cavity_response', 'excited_population',
                    run_id=run_id, qubit=qubit, subfolder=subfolder, conn=source_conn)
    else:
        analysis_run_id = None
    return res_dict, run_id


def two_qubit_exp(run_id, fdict1, fdict2,
                  re_name1='alazar_controller_ch_0_r_records_buffers_data',
                  im_name1='alazar_controller_ch_0_i_records_buffers_data',
                  re_name2='alazar_controller_ch_1_r_records_buffers_data',
                  im_name2='alazar_controller_ch_1_i_records_buffers_data',
                  reps_name='alazar_controller_repetitions',
                  correct_errors=False,
                  subfolder=None,
                  plot=True,
                  index_start=0,
                  reps_name2=None,
                  save=False,
                  source_conn=None,
                  target_conn=None,
                  **setpoints):
    re_array1, im_array1, re_array2, im_array2 = load_xarrays(
        run_id, re_name1, im_name1, re_name2, im_name2, conn=source_conn)
    for k, v in setpoints.items():
        re_array1 = re_array1.sel(k=v)
        im_array1 = im_array1.sel(k=v)
        re_array2 = re_array2.sel(k=v)
        im_array2 = im_array2.sel(k=v)
    all_x1 = re_array1.values.flatten()
    all_y1 = im_array1.values.flatten()
    all_x2 = re_array2.values.flatten()
    all_y2 = im_array2.values.flatten()
    all_data1 = np.array(list(zip(all_x1, all_y1)))
    all_data2 = np.array(list(zip(all_x2, all_y2)))
    coord_names = list(re_array1.coords.keys())
    avg_names = [n for n in [reps_name, reps_name2] if n is not None]
    avg_inds = []
    for r_n in avg_names:
        avg_i = coord_names.index(r_n)
        avg_inds.append(avg_i)
        del coord_names[avg_i]
    projected1 = project(re_array1, im_array1, fdict1['angle']).values.mean(axis=tuple(avg_inds))
    projected2 = project(re_array1, im_array1, fdict2['angle']).values.mean(axis=tuple(avg_inds))
    probability1 = fdict1['kmeans'].predict(all_data1)
    probability2 = fdict2['kmeans'].predict(all_data2)
    if correct_errors:
        for i, q1val in enumerate(projected1):
            probability1 = probability1.astype('float')
            probability2 = probability2.astype('float')
            q2val = projected2[i]
            if q2val + q1val != 1:
                probability1[i] = np.nan
                probability2[i] = np.nan
    probability1 = np.nanmean(probability1.reshape(re_array1.values.shape), axis=tuple(avg_inds))
    probability2 = np.nanmean(probability2.reshape(re_array1.values.shape), axis=tuple(avg_inds))

    projected_info1 = {'data': projected1,
                       'label': 'Cavity Response Q1',
                       'unit': 'V'}
    projected_info2 = {'data': projected2,
                       'label': 'Cavity Response Q2',
                       'unit': 'V'}
    probability_info1 = {'data': probability1,
                         'label': 'Excited Population Q1',
                         'unit': ''}
    probability_info2 = {'data': probability2,
                         'label': 'Excited Population Q2',
                         'unit': ''}

    coord_dict = {}
    for coord_name in coord_names:
        coord_label, coord_unit = coord_label_lookup.get(
            coord_name, [coord_name, ''])
        coord_dict[coord_name] = {'data': re_array1[coord_name].values,
                                  'label': coord_label,
                                  'unit': coord_unit}

    res_dict_proj = {'cavity_response1': projected_info1,
                     'cavity_response2': projected_info2,
                     **coord_dict}

    res_dict_prob = {'excited_population1': probability_info1,
                     'excited_population2': probability_info2,
                     **coord_dict}

    if save:
        if target_conn is not None:
            if target_conn is not None:
                target_exp = load_last_experiment(target_conn)
            else:
                target_exp = None
        meas = Measurement(exp=target_exp)
        for coord_name, coord_info in enumerate(coord_dict):
            meas.register_custom_parameter(coord_name,
                                           label=coord_info['label'],
                                           unit=coord_info['unit'])
        meas.register_custom_parameter('cavity_response1',
                                       label=projected_info1['label'],
                                       unit=projected_info1['unit'],
                                       setpoints=tuple(coord_names))
        meas.register_custom_parameter('excited_population1',
                                       label=probability_info1['label'],
                                       setpoints=tuple(coord_names))
        meas.register_custom_parameter('cavity_response2',
                                       label=projected_info2['label'],
                                       unit=projected_info2['unit'],
                                       setpoints=tuple(coord_names))
        meas.register_custom_parameter('excited_population2',
                                       label=probability_info2['label'],
                                       setpoints=tuple(coord_names))

        with meas.run() as datasaver:
            dim = len(projected1.shape)
            analysis_run_id = datasaver.run_id
            if dim == 1:
                for i in range(len(projected)):
                    result = [('cavity_response1', projected1[i]),
                              ('excited_population1', probability1[i]),
                              ('cavity_response2', projected2[i]),
                              ('excited_population2', probability2[i])]
                    for coord_name, coord_info in enumerate(coord_dict):
                        coord_data = re_array1.mean(reps_name)[i].coords
                        result.append((coord_name, coord_info['data'][i]))
                        datasaver.add_result(*result)
        if plot:
            axes, cbar = plot_by_id(analysis_run_id, conn=target_conn)
            for i, ax in enumerate(axes):
                filename = make_filename(
                    analysis_run_id, index=i, analysis=False,
                    extension=f'{subfolder}', conn=target_conn)
                ax.figure.savefig(filename, dpi=500, bbox_inches='tight')
    elif plot:
        plot_result(res_dict_proj, 'cavity_response1', 'cavity_response2',
                    index=0, run_id=run_id, subfolder=subfolder)
        plot_result(res_dict_prob, 'excited_population1', 'excited_population2',
                    index=1, run_id=run_id, subfolder=subfolder)

    return {**res_dict_proj, **res_dict_prob}


def plot_result(res_dict, *plottable_names, index=0, qubit=None,
                run_id=0, subfolder=None, conn=None):
    res_dict = copy.copy(res_dict)
    dim = len(res_dict) - len(plottable_names)
    plottables = {}
    for plottable_name in plottable_names:
        plottables[plottable_name] = res_dict.pop(plottable_name)
    if dim == 1:
        coord_name = list(res_dict.keys())[0]
        coord_info = res_dict[coord_name]
        fig = plt.figure()
        gs = gridspec.GridSpec(
            nrows=len(plottable_names), ncols=1, hspace=0,
            height_ratios=[1 for _ in plottable_names])
        # axes = []
        for i, plottable_name in enumerate(plottable_names):
            plottable_info = plottables[plottable_name]
            if i == 0:
                ax = fig.add_subplot(gs[0])
            else:
                ax = fig.add_subplot(gs[i], sharex=fig.axes[0])
            ax.plot(coord_info['data'], plottable_info['data'])
            if plottable_name == 'excited_population':
                ax.set_ylim([0, 1])
            ax.set_ylabel(plottable_info['label'])
            ax.set_xlabel(coord_info['label'])
            data_lst = [coord_info, plottable_info]
            _rescale_ticks_and_units(ax, data_lst)
    elif dim == 2:
        coord_names = list(res_dict.keys())
        coord_info_list = [res_dict[n] for n in coord_names]
        coord_data1 = np.multiply.outer(
            coord_info_list[0]['data'],
            np.ones((len(coord_info_list[1]['data'])))).flatten()
        coord_data2 = np.multiply.outer(
            np.ones((len(coord_info_list[0]['data']))),
            coord_info_list[1]['data']).flatten()
        coord_info_list[0]['data'] = coord_data1
        coord_info_list[1]['data'] = coord_data2

        fig = plt.figure()
        gs = gridspec.GridSpec(
            nrows=len(plottable_names), ncols=1, hspace=0.01,
            height_ratios=[1 for _ in plottable_names],
            width_ratios=[0.95, 0.05])
        for i, plottable_name in enumerate(plottable_names):
            plottable_info = plottables[plottable_name]
            if i == 0:
                ax = fig.add_subplot(gs[0, 0])
                cax = fig.add_subplot(gs[0, 1])
            else:
                ax = fig.add_subplot(gs[i, 0], sharex=fig.axes[0])
                cax = fig.add_subplot(gs[i, 1])
            ax, cb = plot_on_a_plain_grid(
                coord_data1, coord_data2, plottable_info['data'].flatten(),
                ax, cax=cax, cmap=qc.config.plotting.default_color_map)
            cb.set_label(plottable_info['label'])
            if plottable_name == 'excited_population':
                cb.set_clim(0, 1)
            ax.set_xlabel(coord_info_list[0]['label'])
            ax.set_ylabel(coord_info_list[1]['label'])
            data_lst = [*coord_info_list, plottable_info]
            _rescale_ticks_and_units(ax, data_lst)
    fig.set_size_inches(5, 7)
    if qubit is not None:
        title_text = 'Q{} (#{})'.format(qubit, run_id)
    else:
        title_text = 'Analysis of #{}'.format(run_id)
    if subfolder is not None:
        title_text = subfolder.title() + ' ' + title_text
    title = fig.suptitle(title_text)
    filename = make_filename(
        run_id, index=index, analysis=True, extension=subfolder,
        conn=conn)
    fig.savefig(filename, dpi=500, bbox_extra_artists=(title,),
                bbox_inches='tight')
    plt.close()


def angle_from_centers(centres):
    x_diff = np.mean(centres[0][0]) - np.mean(centres[1][0])
    y_diff = np.mean(centres[0][1]) - np.mean(centres[1][1])
    angle = np.angle(x_diff + 1j * y_diff)
    return angle


def project(x_data, y_data, angle):
    return x_data * np.cos(angle) + y_data * np.sin(angle)


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


def decide_state(x_data, y_data, angle, decision_value):
    z_data = x_data * np.cos(angle) + y_data * np.sin(angle)
    return z_data < decision_value


# def decide_state_groovy(x_data, y_data, kmeans):
#     all_data = np.array(list(zip(x_data, y_data)))
#     return kmeans.predict(all_data)


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
