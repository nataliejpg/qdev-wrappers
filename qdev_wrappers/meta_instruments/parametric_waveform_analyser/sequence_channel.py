from qcodes.utils import validators as vals
from qdev_wrappers.customised_instruments.composite_instruments.parametric_waveform_analyser.setpoints_channel import SetpointsChannel
from qdev_wrappers.customised_instruments.composite_instruments.sidebander.sidebander import SequenceManager
from qcodes.instrument.channel import InstrumentChannel
from qdev_wrappers.customised_instruments.parameters.delegate_parameters import DelegateParameter
import importlib
import qcodes as qc
import os
import sys
from warnings import warn
import yaml
from lomentum.loader import read_element, register_user_module
from qdev_wrappers.pulse_building import atoms_ext 

register_user_module(atoms_ext)


class SequenceChannel(InstrumentChannel, SequenceManager):
    def __init__(self, parent, name, sequencer, **kwargs):
        super().__init__(parent, name, **kwargs)
        self.sequencer = sequencer
        self.template_element_dict = None
        self.first_element = None
        self.sequencer._do_upload = False
        self._sequencer_up_to_date = False
        inner = SetpointsChannel(self, 'inner')
        outer = SetpointsChannel(self, 'outer')
        self.add_submodule('inner', inner)
        self.add_submodule('outer', outer)
        self.add_parameter(name='sequence_mode',
                           source=sequencer.sequence_mode,
                           set_fn=self._set_seq_mode,
                           parameter_class=DelegateParameter)
        self.add_parameter(name='repetition_mode',
                           source=sequencer.repetition_mode,
                           set_fn=self._set_rep_mode,
                           parameter_class=DelegateParameter)
        self.add_parameter(name='template_element',
                           set_cmd=self.set_sequencer_not_up_to_date)

    def _set_seq_mode(self, val):
        if val == 'sequence':
            self.parent.alazar.seq_mode(True)
        elif val == 'element':
            self.parent.alazar.seq_mode(False)
        self.sequence_mode._save_val(val)
        self.parent.readout.update_all_alazar()

    def _set_rep_mode(self, val):
        if self.parent.readout.num() > 1:
            warn('Repetition mode set to single but readout num > 1'
                           ': necessary that awg is triggered so that '
                           'sequence/element plays num times')
        self.repetition_mode._save_val(val)
        self.parent.readout.update_all_alazar()

    def run(self):
        self.sequencer.run()

    def stop(self):
        self.sequencer.stop()

    def update(self):
        if not self._sequencer_up_to_date:
            self.sequencer._do_upload = True
            self.sequencer.change_sequence(
                self.template_element_dict[self.template_element()],
                initial_element=self.first_element,
                inner_setpoints=self.inner.setpoints,
                outer_setpoints=self.outer.setpoints,
                **self.generate_context())
            self.sequencer._do_upload = False
            self.sync_repeat_parameters()
            self._sequencer_up_to_date = True

    def reload_template_element_dict(self, pulsebuildingfolder=None):
        if pulsebuildingfolder is None:
            pulsebuildingfolder = qc.config["user"]["pulsebuildingfolder"]
        if not os.path.isdir(pulsebuildingfolder):
            raise ValueError(
                f'Pulse building folder {pulsebuildingfolder} cannot be found')
        sys.path.append(pulsebuildingfolder)
        elem_dict = {}
        for file in os.listdir(pulsebuildingfolder):
            element_name = os.path.splitext(file)[0]
            if file.endswith(".yaml"):
                filepath = os.path.join(pulsebuildingfolder, file)
                with open(filepath) as f:
                    yf = yaml.safe_load(f)
                elem_dict[element_name] = read_element(yf)
            elif file.endswith(".py"):
                try:
                    elem_dict[element_name] = importlib.import_module(
                        element_name).create_template_element()
                except Exception as e:
                    warn('Error importing {}: {}'.format(element_name, str(e)))
                    continue
        self.first_element = elem_dict.pop('first_element', None)
        self.template_element_dict = elem_dict
        self.template_element.vals = vals.Enum(*elem_dict.keys())
        self._sequencer_up_to_date = False

    @property
    def pulse_building_parameters(self):
        return self.parent.pulse_building_parameters

    def set_sequencer_not_up_to_date(self, *args):
        self._sequencer_up_to_date = False
