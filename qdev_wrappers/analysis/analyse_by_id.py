from qcodes.dataset.data_export import load_by_id
from qdev_wrappers.analysis.base import AnalyserBase
from qcodes.dataset.measurements import Measurement
from qdev_wrappers.analysis.helpers import make_json_metadata
from qcodes.dataset.experiment_container import load_experiment
import random
from qdev_wrappers.analysis.basic.readout_fidelity import ReadoutFidelityAnalyser, plot_readout_fidelity
from qdev_wrappers.analysis.basic.projector import RealImagProjector
from qdev_wrappers.analysis.basic.decision_maker import DecisionMaker
from qdev_wrappers.analysis.bayesian_inference.base import plot_bayesian_analysis
import numpy as np
from qdev_wrappers.analysis.helpers import scrape_data_dict, organize_experiment_data, refine_data_dict


def analyse_by_id(*args, **kwargs):
    analysers = []
    exp_datasets = []
    for a in args:
        if isinstance(a, AnalyserBase):
            analysers.append(a)
        elif isinstance(a, int):
            exp_datasets.append(load_by_id(a))
        else:
            raise TypeError('Args must be AnalyserBase or run id')
    mappings = {}
    setpoints = {}
    for k, v in kwargs.items():
        if isinstance(v, float):
            setpoints[k] = v
        elif isinstance(v, str):
            mappings[k] = v

    metadata = make_json_metadata(exp_datasets, analysers,
                                  mappings, setpoints)

    meas = Measurement()
    for a in analysers:
        for p in a.all_parameters:
            meas.register_parameter(p)
    current_data = {}
    for a in analysers:
        for d in exp_datasets:
            current_data.update(organise_experiment_data(d, a, mappings, setpoints))
        relevant_data = list(a.metadata['experiment_parameters'].keys())
        relevant_data.extend(list(a.metadata['measurement_parameters'].keys()))
        analysis_kwargs = {}
        for r in relevant_data:
            mapped_name = mappings.get()
            analysis_kwargs[r] = current_data.get(r, r)
        a.analyse(**analysis_kwargs)
        for m, p in a.model_parameters.parameters.items():
            current_data[m] = p()

