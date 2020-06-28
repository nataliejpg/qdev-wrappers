import qcodes as qc
from qcodes.dataset.data_export import load_by_id
from qdev_wrappers.fitting.helpers import organize_exp_data, organize_fit_data, load_json_metadata, load_xarrays
from qcodes.dataset.plotting import _rescale_ticks_and_units
from qcodes.dataset.data_export import reshape_2D_data
from qcodes.dataset.plotting import plot_by_id
from qdev_wrappers.doNd import make_filename
import json
import warnings
import matplotlib.pyplot as plt
import numpy as np
# from qdev_wrappers.dataset.doNd import save_image


def plot_least_squares_1d(indept, dept, metadata, title,
                          fit=None, variance=None,
                          initial_values=None,
                          text=None):
    """
    Plots a 1d plot of the data against the fit result (if provided) with the
    initial fit values result also plotted (optionally) and text about the
    fit values, fit function and (optionally) the variance.

    Args:
        indept (dict): data dictionary with keys 'data', 'label', 'unit',
            'name' for x axis
        dept (dict): data dictionary with keys 'data', 'label', 'unit', 'name'
            for y axis
        metadata (dict): fit metadata dictionary
        title (str)
        fit (dict) (optional): dictionary with a key for each fit_parameter
            and with data dictionaries as values
        variance (dict) (optional): dictionary with a key for each
            variance_parameter which if fit then also prints standard deviation
            on fit parameters
        initial_values (dict) (optional): dictionary with a key for each
            initial_value_parameter which if fit then also plots the fit result
            using the initial values used in fitting
        text (str) (optional): string to be prepended to the fit parameter
            values information text box

    Returns:
        matplotlib ax
    """
    plt.figure(figsize=(10, 4))
    ax = plt.subplot(111)

    # plot data
    ax.plot(indept['data'], dept['data'],
            marker='.', markersize=5, linestyle='', color='C0',
            label='data')

    # plot fit
    if fit is not None:
        fit_paramnames = metadata['fitter']['fit_parameters']
        variance_paramnames = metadata['fitter'].get('variance_parameters')
        initial_value_paramnames = metadata['fitter'].get(
            'initial_value_parameters')
        x = np.linspace(indept['data'].min(), indept['data'].max(),
                        num=len(indept['data']) * 10)
        finalkwargs = {'x': x,
                       'np': np,
                       **{f['name']: np.atleast_1d(f['data'].values)[0] for f in fit}}
        finaly = eval(metadata['fitter']['function']['np'],
                      finalkwargs)
        ax.plot(x, finaly, color='C1', label='fit')
        if initial_values is not None:
            initialkwargs = {}
            for i, name in enumerate(initial_value_paramnames):
                initialkwargs[fit_paramnames[i]
                              ] = initial_values[i]['data']
            initialkwargs.update(
                {'x': x,
                 'np': np})
            intialy = eval(metadata['fitter']['function']['np'],
                           initialkwargs)
            ax.plot(x, intialy, color='grey', label='initial fit values')
        plt.legend()
        p_label_list = [text] if text else []
        p_label_list.append(metadata['fitter']['function']['str'])
        for i, f in enumerate(fit):
            fit_val = np.atleast_1d(f['data'].values)[0]
            if variance is not None:
                variance_val = np.atleast_1d(variance[i]['data'].values)[0]
                standard_dev = np.sqrt(variance_val)
                p_label_list.append('{} = {:.3g} ± {:.3g} {}'.format(
                    f['label'], fit_val,
                    standard_dev, f['unit']))
            else:
                p_label_list.append('{} = {:.3g} {}'.format(
                    f['label'], fit_val, f['unit']))
        textstr = '\n'.join(p_label_list)
    else:
        textstr = '{} \n Unsuccessful fit'.format(
            fitter.metadata['fitter']['function']['str'])

    # add text, axes, labels, title and rescale
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.text(1.05, 0.7, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox={'ec': 'k', 'fc': 'w'})
    ax.set_xlabel(f"{indept['label']} ({indept['unit']})")
    ax.set_ylabel(f"{dept['label']} ({dept['unit']})")
    ax.set_title(title)
    data_lst = [indept, dept]
    _rescale_ticks_and_units(ax, data_lst)

    return ax


def plot_heat_map(x, y, z, title, text=None):
    """
    Plots a 2d heatmap of the x, y and z data provided

    Args:
        x (dict): data dictionary with keys 'data', 'label', 'unit', 'name'
            for x axis (data should be 1d - length n)
        y (dict): data dictionary with keys 'data', 'label', 'unit', 'name'
            for y axis (data should be 1d - length m)
        z (dict): data dictionary with keys 'data', 'label', 'unit', 'name'
            for z axis (data should be 2d - shape n * m)
        title (str)

    Returns:
        matplotlib ax
        matplotlib colorbar
    """
    fig, ax = plt.subplots(1, 1)
    ax, colorbar = plot_on_a_plain_grid(
        x['data'], y['data'], z['data'], ax,
        cmap=qc.config.plotting.default_color_map)
    ax.set_xlabel(f"{x['label']} ({x['unit']})")
    ax.set_ylabel(f"{y['label']} ({y['unit']})")
    colorbar.set_label(f"{z['label']} ({z['unit']})")
    ax.set_title(title)
    data_lst = [x, y, z]
    _rescale_ticks_and_units(ax, data_lst)
    if text is not None:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        ax.text(1.05, 0.7, text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox={'ec': 'k', 'fc': 'w'})
    return ax, colorbar


def plot_on_a_plain_grid(x, y, z, ax, cax=None, **kwargs):
    xrow, yrow, z_to_plot = reshape_2D_data(x, y, z)
    dxs = np.diff(xrow) / 2
    dys = np.diff(yrow) / 2
    x_edges = np.concatenate((np.array([xrow[0] - dxs[0]]),
                              xrow[:-1] + dxs,
                              np.array([xrow[-1] + dxs[-1]])))
    y_edges = np.concatenate((np.array([yrow[0] - dys[0]]),
                              yrow[:-1] + dys,
                              np.array([yrow[-1] + dys[-1]])))
    if 'rasterized' in kwargs.keys():
        rasterized = kwargs.pop('rasterized')
    else:
        rasterized = len(x_edges) * len(y_edges) \
            > qc.config.plotting.rasterize_threshold

    cmap = kwargs.pop('cmap') if 'cmap' in kwargs else None

    colormesh = ax.pcolormesh(x_edges, y_edges,
                              np.ma.masked_invalid(z_to_plot),
                              rasterized=rasterized,
                              cmap=cmap,
                              **kwargs)
    if cax is not None:
        colorbar = ax.figure.colorbar(colormesh, ax=ax, cax=cax)
    else:
        colorbar = ax.figure.colorbar(colormesh, ax=ax)

    return ax, colorbar


def plot_fit_param_1ds(setpoint, fit, metadata, title,
                       variance=None, initial_values=None):
    """
    Plots a 1d plot for each fit parameter against the setpoint.

    Args:
        setpoint (dict): data dictionary with keys 'data', 'label', 'unit',
            'name' for x axes
        fit (dict): dictionary with a key for each fit_parameter and with data
            dictionaries as values
        metadata (dict): fit metadata dictionary
        title (str)
        variance (dict) (optional): dictionary with a key for each
            variance_parameter which if fit then also adds error bars
        initial_values (dict) (optional): dictionary with a key for each
            initial_value_parameter which if fit then also plots these against
            the setpoint

    Returns:
        list of matplotlib axes
    """
    axes = []
    order = setpoint['data'].argsort()
    xpoints = setpoint['data'][order]
    # fit_paramnames = metadata['fitter']['fit_parameters']
    variance_paramnames = metadata['fitter'].get('variance_parameters')
    intial_value_paramnames = metadata['fitter'].get(
        'initial_value_parameters')
    for i, f in enumerate(fit):
        fig, ax = plt.subplots(1, 1)
        ypoints = f['data'].values[order]
        if variance is not None:
            var = variance[i]['data'].values[order]
            standard_dev = np.sqrt(var)
            ax.errorbar(xpoints, ypoints,
                        standard_dev, None, 'g.-', label='fitted value')
        else:
            ax.plot(xpoints, ypoints, 'g.-', label='fitted value')
        # ax.text(0.05, 0.95, metadata['fitter']['function']['str'],
        #         transform=ax.transAxes, fontsize=12,
        #         verticalalignment='top', bbox={'ec': 'k', 'fc': 'w'})
        if initial_values is not None:
            ini = initial_values[i]['data'].values[order]
            ax.plot(xpoints, ini, '.-', color='grey', label='initial guess')
            plt.legend()

        # add axes, labels, title and rescale
        xlabel = setpoint['label'] or setpoint['name']
        xlabel += f" ({setpoint['unit']})"
        ax.set_xlabel(xlabel)
        ylabel = f['label'] or f['name']
        ylabel += f" ({f['unit']})"
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        data_lst = [setpoint, f]
        _rescale_ticks_and_units(ax, data_lst)
        box = ax.get_position()
        text = metadata['fitter']['function']['str']
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        ax.text(1.05, 0.7, text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox={'ec': 'k', 'fc': 'w'})
        axes.append(ax)
    return axes


def plot_fit_by_id(fit_run_id,
                   show_variance=True,
                   show_initial_values=False,
                   save_plots=True,
                   source_conn=None,
                   target_conn=None,
                   variance_limited=False,
                   **setpoint_values):
    """
    Plots the result of a fit (against the data where possible) and
    optionallt saves the plots

    For a 1d plots the fit result against the data if the fitter method
    is 'LeastSquares'. For a 2d plots a 1d for each fit parameter against the
    setpoint variable unless a specific value of the setpoint variable
    is provided (in which case it plots a 1d of the fit result against the
    data at this point if the fitter method is 'LeastSquares'). In the case
    of a 'LeastSquares' fitter also plots the 2d fit result as a heatmap
    (if setpoint_values not provided).

    Args:
        fit_run_id (int): run id of fit dataset
        show_variance (bool) (default True)
        show_initial_values (bool) (default False)
        save_plots (bool) (default True)
        **setpoint_values: key, value pairs of names of the setpoints and the
            value at which the cut should be taken. This is only relevant if
            the fit has been performed at multiple values of some 'setpoint'
            variable

    Returns
        list of matplotlib axes
        matplotlib colorbar of the 2d heatmap (None if not generated)
    """

    # load and organize data
    fit_data = load_by_id(fit_run_id, conn=target_conn)
    metadata = load_json_metadata(fit_data)
    fit_names = metadata['fitter']['fit_parameters']
    var_names = metadata['fitter'].get('variance_parameters', None)
    initial_val_names = metadata['fitter'].get(
        'initial_value_parameters', None)
    dependent_name = metadata['inferred_from']['dept_var']
    independent_names = metadata['inferred_from']['indept_vars']
    average_names = metadata['inferred_from']['average_vars']
    exp_run_id = metadata['inferred_from']['run_id']
    exp_data = load_by_id(exp_run_id, source_conn)
    success, fit, var, initial_vals = organize_fit_data(
        fit_run_id, conn=target_conn, **setpoint_values)
    # var = var if show_variance else None
    # initial_vals = initial_vals if show_initial_values else None
    for s in setpoint_values.keys():
        setpoint_values[s] = success[s].values
    for coord in success.coords:
        if coord not in list(setpoint_values.keys()) + ['index']:
            setpoint_values[coord] = success[coord].values
    dependent, independent = organize_exp_data(
        exp_run_id, dependent_name, *independent_names,
        average_names=average_names, conn=source_conn, **setpoint_values)
    for s, v in setpoint_values.items():
        if any(np.atleast_1d(dependent[s].values) != np.atleast_1d(v)):
            raise RuntimeError(
                'Fit setpoint {} {} does not match exp setpoint val {}'.format(
                    s, dependent[s], v))
    if len(independent) > 1:
        raise NotImplementedError("Plotting only currently works for "
                                  "fitters with 1 independent variable")
    independent = independent[0]

    # generate additional label text and filename text
    text_list = []
    extra_save_text_list = []
    to_r = []
    for i, (s, v) in enumerate(setpoint_values.items()):
        if v.shape == ():
            s_parm = exp_data.paramspecs[s]
            text_list.append(
                '{} = {:.3g} {}'.format(s_parm.label, v, s_parm.unit))
            extra_save_text_list.append(
                '{}{:.3g}'.format(s_parm.name, v).replace('.', 'p'))
            to_r.append(s)
    for s in to_r:
        setpoint_values.pop(s)
    text = '\n'.join(text_list)

    # organise data into dictionaries
    dependent_parm = exp_data.paramspecs[dependent_name]
    dependent_dict = {'name': dependent_parm.name,
                      'label': dependent_parm.label,
                      'unit': dependent_parm.unit,
                      'data': dependent.values}
    independent_parm = exp_data.paramspecs[independent_names[0]]
    independent_dict = {'name': independent_parm.name,
                        'label': independent_parm.label,
                        'unit': independent_parm.unit,
                        'data': independent.values}
    fit_dicts = []
    initial_val_dicts = []
    var_dicts = []
    for i, f in enumerate(fit_names):
        fit_parm = fit_data.paramspecs[f]
        if variance_limited and show_variance and var is not None:
            var_parm = fit_data.paramspecs[var_names[i]]
            max_var = 0.5 * np.mean(fit[i].values)
            inds = np.argwhere(var[i] < max_var)
            fit_d = fit[i].sel(*inds)
            initial_vals_d = initial_vals[i].sel(*inds)
            var_d = var[i].sel(*inds)
        else:
            fit_d = fit[i]
            if show_variance and var is not None:
                var_d = var[i]
            else:
                var_dicts = None
            if show_initial_values and initial_vals is not None:
                initial_vals_d = initial_vals[i]
            else:
                initial_val_dicts = None
        fit_dicts.append({'name': fit_parm.name,
                          'label': fit_parm.label,
                          'unit': fit_parm.unit,
                          'data': fit_d})
        if initial_val_dicts is not None:
            initial_val_parm = fit_data.paramspecs[initial_val_names[i]]
            initial_val_dicts.append({'name': initial_val_parm.name,
                                      'label': initial_val_parm.label,
                                      'unit': initial_val_parm.unit,
                                      'data': initial_vals_d})
        if var_dicts is not None:
            var_parm = fit_data.paramspecs[var_names[i]]
            var_dicts.append({'name': var_parm.name,
                              'label': var_parm.label,
                              'unit': var_parm.unit,
                              'data': var_d})
    setpoint_dicts = []
    for s, v in setpoint_values.items():
        setpoint_parm = exp_data.paramspecs[s]
        setpoint_dicts.append({'name': setpoint_parm.name,
                               'label': setpoint_parm.label,
                               'unit': setpoint_parm.unit,
                               'data': v})

    # generate title
    axes = []
    colorbar = None
    title = "Run #{} (#{} fitted), Experiment {} ({})".format(
        fit_run_id,
        metadata['inferred_from']['run_id'],
        metadata['inferred_from']['exp_id'],
        metadata['inferred_from']['sample_name'])

    # calculate data dimension
    dimension = len(setpoint_dicts)
    if dimension > 2:
        raise RuntimeError('Cannot make 3D plot. Try specifying a cut')

    # make the plots
    if dimension == 0:  # 1D PLOTTING - data + fit 1d plot
        if metadata['fitter']['method'] == 'LeastSquares':
            if not success.values:
                fit_dicts = None
                var_dicts = None
                initial_val_dicts = None
            ax = plot_least_squares_1d(
                independent_dict, dependent_dict, metadata, title,
                fit=fit_dicts, variance=var_dicts,
                initial_values=initial_val_dicts, text=text)
            axes.append(ax)
        else:
            warnings.warn(
                'Attempt to plot a 1d plot of a non LeastSquares fit')
    elif dimension == 1:  # 2D PLOTTING: 2d fit heat map plot + fit param plots
        setpoint_dict = setpoint_dicts.pop()
        xpoints, ypoints, zpoints = [], [], []
        if metadata['fitter']['method'] == 'LeastSquares':
            for i, val in enumerate(setpoint_dict['data']):
                xpoints.append(np.ones(len(independent_dict['data'])) * val)
                ypoints.append(independent_dict['data'])
                success_point = success[i]
                if success_point:
                    fit_vals = {f['name']: f['data'].values[i]
                                for f in fit_dicts}
                    kwargs = {
                        'np': np,
                        'x': independent_dict['data'],
                        **fit_vals}
                    zpoints.append(eval(metadata['fitter']['function']['np'],
                                        kwargs))
                else:
                    zpoints.append([None] * len(independent_dict['data']))
            xpoints = np.array(xpoints).flatten()
            ypoints = np.array(ypoints).flatten()
            zpoints = np.array(zpoints).flatten()
            # 2D simulated heatmap plot
            x = {**setpoint_dict, 'data': xpoints}
            y = {**independent_dict, 'data': ypoints}
            z = {**dependent_dict, 'data': zpoints,
                 'label': 'Simulated ' + dependent_dict['label']}
            ax, colorbar = plot_heat_map(x, y, z, title,
                                         metadata['fitter']['function']['str'])
            axes.append(ax)

        # 1D fit parameter vs setpoint plots
        axes += plot_fit_param_1ds(setpoint_dict, fit_dicts, metadata, title,
                                   variance=var_dicts,
                                   initial_values=initial_val_dicts)
    # 3D PLOTTING: nope
    else:
        axes, colorbar = plot_by_id(fit_run_id, conn=target_conn)

    # saving
    if save_plots:
        name_extension = '_'.join(['fit'] + extra_save_text_list)
        for i, ax in enumerate(axes):
            filename = make_filename(fit_data.run_id, index=i, analysis=True,
                                     extension=name_extension, conn=target_conn)
            ax.figure.savefig(filename)
            plt.close()
    return axes, colorbar


def plot_cut(run_id, conn=None, **setpoint_values):
    data = load_by_id(run_id, conn=conn)
    dependent_parm_names = [p.name for p in data.dependent_parameters]
    xarrays = load_xarrays(run_id, *dependent_parm_names, conn=conn)
    experiment_name = data.exp_name
    sample_name = data.sample_name
    title = f"Run #{data.captured_run_id}, " \
        f"Experiment {experiment_name} ({sample_name})"
    fig_num = 0
    axes = []
    clbs = []
    for i, d_n in enumerate(dependent_parm_names):
        d_xarr = xarrays[i]
        d_p = data.paramspecs[d_n]
        independent_parm_names = d_p.depends_on.split(', ')
        d_setpoints = {k: v for k, v in setpoint_values.items() if k in independent_parm_names}
        d_xarr = d_xarr.sel(**d_setpoints, method='nearest')
        d_setpoints.update({k: d_xarr[k].values for k in d_setpoints.keys()})
        label_text_list = []
        save_text_list = []
        for k, v in d_setpoints.items():
            parm = data.paramspecs[k]
            label_text_list.append('{} = {:.3g} {}'.format(
                parm.label, v, parm.unit))
            save_text_list.append('{}{:.3g}'.format(k, v).replace('.', 'p'))
        label_text_string = '\n'.join(label_text_list)
        save_text_string = '_'.join(save_text_list)
        if len(d_xarr.shape) == 1:
            d_dict = {'label': d_p.label, 'unit': d_p.unit,
                      'data': d_xarr.values}
            ind_n = [c for c in d_xarr.coords if c not in d_setpoints][0]
            ind_p = data.paramspecs[ind_n]
            ind_dict = {'label': ind_p.label, 'unit': ind_p.unit,
                        'data': d_xarr[ind_n].values}
            ax, clb = _plot_cut_1d(d_dict, ind_dict, title, label_text_string)
            filename = make_filename(run_id, index=fig_num,
                                     extension=save_text_string, conn=conn)
            ax.figure.savefig(filename)
            plt.close()
            fig_num += 1
        elif len(d_xarr.shape) == 2:
            print('no 2d cuts rn sry not sry')
            ax, clb = None, None
            # ind_ns = [c for c in d_xarr.coords if c not in d_setpoints]
            # ind_dicts = []
            # for ind_n in ind_ns:
            #     ind_p = data.paramspecs[ind_n]
            #     ind_dicts.append({'label': ind_p.label, 'unit': ind_p.unit,
            #                       'data': d_xarr[ind_n].values})
            # d_dict = {'label': d_p.label, 'unit': d_p.unit,
            #           'data': d_xarr.values}
            # ax, clb = _plot_cut_2d(d_dict, *ind_dicts, title, label_text_string)
            # axes.append(ax)
            # filename = make_filename(run_id, index=fig_num,
            #                          extension=save_text_string, conn=conn)
            # ax.figure.savefig(filename)
            # plt.close()
            # fig_num += 1
        else:
            print('no 3d cuts now or ever!')
            ax, clb = None, None
        axes.append(ax)
        clbs.append(clb)
    return axes, clbs


def _plot_cut_1d(dept, indept, title, text):
    plt.figure(figsize=(10, 4))
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    ax.plot(indept['data'], dept['data'])

    ax.text(1.05, 0.7, text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox={'ec': 'k', 'fc': 'w'})
    ax.set_xlabel(f"{indept['label']} ({indept['unit']})")
    ax.set_ylabel(f"{dept['label']} ({dept['unit']})")
    ax.set_title(title)
    data_lst = [indept, dept]
    _rescale_ticks_and_units(ax, data_lst)

    return ax, None


def _plot_cut_2d(dept, indept1, indept2, title, text):
    plt.figure(figsize=(10, 4))
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax, colorbar = plot_on_a_plain_grid(
        indept1['data'], indept2['data'], dept['data'], ax,
        cmap=qc.config.plotting.default_color_map)
    ax.text(1.05, 0.7, text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox={'ec': 'k', 'fc': 'w'})
    ax.set_xlabel(f"{indept1['label']} ({indept1['unit']})")
    ax.set_ylabel(f"{indept2['label']} ({indept2['unit']})")
    colorbar.set_label(f"{dept['label']} ({dept['unit']})")
    ax.set_title(title)
    data_lst = [indept1, indept2, dept]
    _rescale_ticks_and_units(ax, data_lst)
    return ax, colorbar