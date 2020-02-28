import os
import copy
import json
import qcodes as qc
import numpy as np
from itertools import product
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
from qdev_wrappers.fitting.readout_fidelity import project
from qcodes.dataset.experiment_container import load_last_experiment


# coord_label_lookup = {'alazar_controller_pulse_duration': ['Pulse Duration', 's'],
#                       'alazar_controller_pulse_pulse_delay': ['Delay', 's'],
#                       'alazar_controller_operation_index': ['Sequence Index', ''],
#                       'alazar_controller_clifford_gate_count': ['# of Cliffords', '']}


def make_fdict_json_metadata(dataset, fdict, average_names):
    average_names = average_names or []
    exp_metadata = {'run_id': dataset.run_id,
                    'exp_id': dataset.exp_id,
                    'exp_name': dataset.exp_name,
                    'sample_name': dataset.sample_name}
    stripped_fdict = {'angle': fdict['angle'],
                      'decision_value': fdict['decision_value'],
                      'centers': [list(c) for c in fdict['centers']]}
    metadata = {'projection_info': stripped_fdict,
                'inferred_from': {'average_vars': average_names,
                                  **exp_metadata}}
    metadata_key = 'projection_metadata'
    return metadata_key, json.dumps(metadata)


def one_qubit_exp(run_id, fdict,
                  re_name='alazar_controller_ch_0_r_records_buffers_data',
                  im_name='alazar_controller_ch_0_i_records_buffers_data',
                  average_names=None,
                  qubit=1,
                  subfolder=None,
                  plot=True,
                  index_start=0,
                  save=False,
                  source_conn=None,
                  target_conn=None,
                  **setpoint_values):
    # load and organise data
    exp_data = load_by_id(run_id, conn=source_conn)
    re_array, im_array = load_xarrays(run_id, re_name, im_name, conn=source_conn)
    for k, v in setpoint_values.items():
        nearest_val = re_array[k].values[np.argmin(abs(re_array[k] - v))]
        setpoint_values[k] = nearest_val
        re_array = re_array.sel(k=nearest_val)
        im_array = im_array.sel(k=nearest_val)
    if average_names is None:
        average_names = []
    elif isinstance(average_names, str):
        average_names = [average_names]

    # do projection and decision_making and update setpoints
    projected = project(re_array, im_array, fdict['angle'])
    probability = (projected < fdict['decision_value']).mean(dim=average_names)
    projected = projected.mean(dim=average_names)
    for coord in projected.coords:
        if coord not in list(setpoint_values.keys()) + ['index']:
            setpoint_values[coord] = projected[coord].values

    # generate metadata
    projected_info = {'label': f'Cavity Response Q{qubit}',
                      'unit': 'V',
                      'data': projected}
    probability_info = {'label': f'Excited Population Q{qubit}',
                        'unit': '',
                        'data': probability}
    setpoint_dict = {}
    for k, v in setpoint_values.items():
        setpoint_parm = exp_data.paramspecs[k]
        setpoint_dict[k] = {'label': setpoint_parm.label,
                            'unit': setpoint_parm.unit,
                            'data': projected[k].values}
    res_dict = {'cavity_response': projected_info,
                'excited_population': probability_info,
                **setpoint_dict}
    metadata = make_fdict_json_metadata(exp_data, fdict, average_names)

    # save and or plot
    if save:
        if target_conn is not None:
            if target_conn is not None:
                target_exp = load_last_experiment(target_conn)
            else:
                target_exp = None
        meas = Measurement(exp=target_exp)
        setpoint_names = list(setpoint_values.keys())
        for setpoint_name in setpoint_names:
            paramspec = exp_data.paramspecs[setpoint_name]
            meas.register_custom_parameter(name=paramspec.name,
                                           label=paramspec.label,
                                           unit=paramspec.unit)

        meas.register_custom_parameter('cavity_response',
                                       label=projected_info['label'],
                                       unit=projected_info['unit'],
                                       setpoints=tuple(setpoint_names))
        meas.register_custom_parameter('excited_population',
                                       label=probability_info['label'],
                                       setpoints=tuple(setpoint_names))
        setpoint_comb_values = product(*[v for v in setpoint_values.values()])
        with meas.run() as datasaver:
            datasaver._dataset.add_metadata(*metadata)
            analysis_run_id = datasaver.run_id
            result = {}
            inds = []
            for vals in setpoint_comb_values:
                comb_dict = dict(zip(setpoint_names, vals))
                projected_data = projected.sel(**comb_dict).values
                probablility_data = probability.sel(**comb_dict).values
                result = [(k, v) for k, v in comb_dict.items()]
                result.extend([('cavity_response', projected_data),
                               ('excited_population', probablility_data)])
                datasaver.add_result(*result)
        if plot:
            axes, cbars = plot_by_id(analysis_run_id, conn=target_conn)
            for i, ax in enumerate(axes):
                if i == 1 and cbars[i] is None:
                    ax.set_ylim(0, 1)
                filename = make_filename(
                    analysis_run_id, index=i, analysis=False,
                    extension=f'q{qubit}_{subfolder}', conn=target_conn)
                ax.figure.savefig(filename, dpi=500, bbox_inches='tight')
                plt.close()
    else:
        analysis_run_id = None
        if plot:
            plot_result(res_dict, 'cavity_response', 'excited_population',
                        run_id=run_id, qubit=qubit, subfolder=subfolder,
                        conn=source_conn)
    return res_dict, analysis_run_id


# def two_qubit_exp(run_id, fdict1, fdict2,
#                   re_name1='alazar_controller_ch_0_r_records_buffers_data',
#                   im_name1='alazar_controller_ch_0_i_records_buffers_data',
#                   re_name2='alazar_controller_ch_1_r_records_buffers_data',
#                   im_name2='alazar_controller_ch_1_i_records_buffers_data',
#                   reps_name='alazar_controller_repetitions',
#                   correct_errors=False,
#                   subfolder=None,
#                   plot=True,
#                   index_start=0,
#                   reps_name2=None,
#                   save=False,
#                   source_conn=None,
#                   target_conn=None,
#                   **setpoints):
#     re_array1, im_array1, re_array2, im_array2 = load_xarrays(
#         run_id, re_name1, im_name1, re_name2, im_name2, conn=source_conn)
#     for k, v in setpoints.items():
#         re_array1 = re_array1.sel(k=v)
#         im_array1 = im_array1.sel(k=v)
#         re_array2 = re_array2.sel(k=v)
#         im_array2 = im_array2.sel(k=v)
#     all_x1 = re_array1.values.flatten()
#     all_y1 = im_array1.values.flatten()
#     all_x2 = re_array2.values.flatten()
#     all_y2 = im_array2.values.flatten()
#     all_data1 = np.array(list(zip(all_x1, all_y1)))
#     all_data2 = np.array(list(zip(all_x2, all_y2)))
#     coord_names = list(re_array1.coords.keys())
#     avg_names = [n for n in [reps_name, reps_name2] if n is not None]
#     avg_inds = []
#     for r_n in avg_names:
#         avg_i = coord_names.index(r_n)
#         avg_inds.append(avg_i)
#         del coord_names[avg_i]
#     projected1 = project(re_array1, im_array1, fdict1['angle']).mean(dim=avg_names)
#     projected2 = project(re_array1, im_array1, fdict2['angle']).mean(dim=avg_names)
#     if 'kmeans' in fdict1:
#         probability1 = fdict1['kmeans'].predict(all_data1)
#     else:
#         all_data = project(all_x1, all_y1, fdict1['angle'])
#         probability1 = (all_data < fdict1['decision_value']).reshape(
#             re_array1.values.shape).mean(axis=tuple(avg_inds))
#     if 'kmeans' in fdict1:
#         probability2 = fdict2['kmeans'].predict(all_data2)
#     else:
#         all_data = project(all_x2, all_y2, fdict2['angle'])
#         probability2 = (all_data < fdict2['decision_value']).reshape(
#             re_array12values.shape).mean(axis=tuple(avg_inds))
#     if correct_errors:
#         for i, q1val in enumerate(projected1):
#             probability1 = probability1.astype('float')
#             probability2 = probability2.astype('float')
#             q2val = projected2[i]
#             if q2val + q1val != 1:
#                 probability1[i] = np.nan
#                 probability2[i] = np.nan
#     probability1 = np.nanmean(probability1.reshape(re_array1.values.shape), axis=tuple(avg_inds))
#     probability2 = np.nanmean(probability2.reshape(re_array1.values.shape), axis=tuple(avg_inds))

#     projected_info1 = {'data': projected1,
#                        'label': 'Cavity Response Q1',
#                        'unit': 'V'}
#     projected_info2 = {'data': projected2,
#                        'label': 'Cavity Response Q2',
#                        'unit': 'V'}
#     probability_info1 = {'data': probability1,
#                          'label': 'Excited Population Q1',
#                          'unit': ''}
#     probability_info2 = {'data': probability2,
#                          'label': 'Excited Population Q2',
#                          'unit': ''}

#     coord_dict = {}
#     for coord_name in coord_names:
#         coord_label, coord_unit = coord_label_lookup.get(
#             coord_name, [coord_name, ''])
#         coord_dict[coord_name] = {'data': re_array1[coord_name].values,
#                                   'label': coord_label,
#                                   'unit': coord_unit}

#     res_dict_proj = {'cavity_response1': projected_info1,
#                      'cavity_response2': projected_info2,
#                      **coord_dict}

#     res_dict_prob = {'excited_population1': probability_info1,
#                      'excited_population2': probability_info2,
#                      **coord_dict}

#     if save:
#         if target_conn is not None:
#             if target_conn is not None:
#                 target_exp = load_last_experiment(target_conn)
#             else:
#                 target_exp = None
#         meas = Measurement(exp=target_exp)
#         for coord_name, coord_info in enumerate(coord_dict):
#             meas.register_custom_parameter(coord_name,
#                                            label=coord_info['label'],
#                                            unit=coord_info['unit'])
#         meas.register_custom_parameter('cavity_response1',
#                                        label=projected_info1['label'],
#                                        unit=projected_info1['unit'],
#                                        setpoints=tuple(coord_names))
#         meas.register_custom_parameter('excited_population1',
#                                        label=probability_info1['label'],
#                                        setpoints=tuple(coord_names))
#         meas.register_custom_parameter('cavity_response2',
#                                        label=projected_info2['label'],
#                                        unit=projected_info2['unit'],
#                                        setpoints=tuple(coord_names))
#         meas.register_custom_parameter('excited_population2',
#                                        label=probability_info2['label'],
#                                        setpoints=tuple(coord_names))

#         with meas.run() as datasaver:
#             dim = len(projected1.shape)
#             analysis_run_id = datasaver.run_id
#             if dim == 1:
#                 for i in range(len(projected)):
#                     result = [('cavity_response1', projected1[i]),
#                               ('excited_population1', probability1[i]),
#                               ('cavity_response2', projected2[i]),
#                               ('excited_population2', probability2[i])]
#                     for coord_name, coord_info in enumerate(coord_dict):
#                         coord_data = re_array1.mean(reps_name)[i].coords
#                         result.append((coord_name, coord_info['data'][i]))
#                         datasaver.add_result(*result)
#             elif dim == 2:
#                 print('hio')
#         if plot:
#             axes, cbar = plot_by_id(analysis_run_id, conn=target_conn)
#             for i, ax in enumerate(axes):
#                 filename = make_filename(
#                     analysis_run_id, index=i, analysis=False,
#                     extension=f'{subfolder}', conn=target_conn)
#                 ax.figure.savefig(filename, dpi=500, bbox_inches='tight')
#                 plt.close()
#     elif plot:
#         plot_result(res_dict_proj, 'cavity_response1', 'cavity_response2',
#                     index=0, run_id=run_id, subfolder=subfolder)
#         plot_result(res_dict_prob, 'excited_population1', 'excited_population2',
#                     index=1, run_id=run_id, subfolder=subfolder)

#     return {**res_dict_proj, **res_dict_prob}


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
