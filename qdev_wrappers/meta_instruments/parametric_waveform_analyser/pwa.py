from warnings import warn
import numpy as np
from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel, ChannelList
import math
from qdev_wrappers.customised_instruments.composite_instruments.parametric_waveform_analyser.readout import ReadoutChannel
from qdev_wrappers.customised_instruments.composite_instruments.parametric_waveform_analyser.drive import DriveChannel
from qdev_wrappers.customised_instruments.composite_instruments.parametric_waveform_analyser.sequence_channel import SequenceChannel

# TODO: make instruments private

class QubitChannel(InstrumentChannel):
    def __init__(self, parent, name, readout_sidebander, drive_sidebander):
        super().__init__(parent, name)
        self.add_submodule('readout', readout_sidebander)
        self.add_submodule('drive', drive_sidebander)
        self.data = self.readout.data

class ParametricWaveformAnalyser(Instrument):
    """
    The PWA represents a composite instrument. It comprises a parametric
    sequencer, an Alazar card, an alazar controller instrument, a heterodyne
    source and a qubit source. The idea is that the parametric sequencer is
    used to sideband the heterodyne and qubit sources to create a drive and
    readout tone per qubit and also optionally to vary some parameter in a
    sequence. The setpoints of this sequence and the demodulation frequencies
    calculated from the heterodyne source demodulation frequency and the
    parametric sequencer sideband frequencies are communicated to the Alazar
    controller.
    """

    def __init__(self,
                 name: str,
                 sequencer_name,
                 alazar_name,
                 alazar_controller_name,
                 heterodyne_source_name,
                 drive_source_if_name) -> None:
        super().__init__(name)
        self.sequencer = Instrument.find_instrument(sequencer_name)
        self.alazar = Instrument.find_instrument(alazar_name)
        self.alazar_controller = Instrument.find_instrument(alazar_controller_name)
        self.heterodyne_source = Instrument.find_instrument(heterodyne_source_name)
        self.drive_source = Instrument.find_instrument(drive_source_if_name)
        sequence_channel = SequenceChannel(self, 'sequence', self.sequencer)
        self.add_submodule('sequence', sequence_channel)
        readout_channel = ReadoutChannel(self, 'readout', self.sequencer,
                                         self.heterodyne_source, self.alazar_controller)
        self.add_submodule('readout', readout_channel)
        drive_channel = DriveChannel(self, 'drive', self.sequencer, self.drive_source)
        self.add_submodule('drive', drive_channel)
        qubits = ChannelList(self, 'qubits', QubitChannel)
        self.add_submodule('qubits', qubits)
        self._sequencer_up_to_date = False
        self.sequence.reload_template_element_dict()

    def update(self):
        self.sequence.update()
        self.readout.update_all_alazar()

    def add_qubit(self, readout_frequency=None, drive_frequency=None):
        """
        Adds a SidebandingChannel to each of the
        ReadoutChannel and DriveChannels.
        """
        self._set_sequencer_up_to_date_flag(False)
        qubit_num = len(self.readout.sidebanders)
        readout = self.readout.add_sidebander(name=f'Q{qubit_num}_R', symbol_prepend=f'Q{qubit_num}_readout')
        drive = self.drive.add_sidebander(name=f'Q{qubit_num}_D', symbol_prepend=f'Q{qubit_num}_drive')
        if readout_frequency is not None:
            readout.frequency(readout_frequency)
        if drive_frequency is not None:
            drive.frequency(drive_frequency)
        qubit = QubitChannel(self, f'Q{qubit_num}', readout, drive)
        self.qubits.append(qubit)
        self.add_submodule(f'Q{qubit_num}', qubit)

    def clear_qubits(self):
        """
        Clears the alazar channels and all SidebandingChannels
        (both from the ReadoutChannel and the DriveChannel)
        """
        self._set_sequencer_up_to_date_flag(False)
        self.alazar_controller.channels.clear()
        self.readout.sidebanders.clear()
        self.drive.sidebanders.clear()
        self.qubits.clear()

    def _set_sequencer_up_to_date_flag(self, value):
        self.sequence._sequencer_up_to_date = value

    _sequencer_up_to_date = property(None, _set_sequencer_up_to_date_flag)

    @property
    def pulse_building_parameters(self):
        return {**self.readout.pulse_building_parameters,
                **self.drive.pulse_building_parameters}

    def get_alazar_ch_settings(self):
        """
        Based on the current instrument settings calculates the settings
        configuration for an alazar channel as a dictionary with keys:

        buffers: int
        average_buffers: bool
        records: int
        average_records: bool

        and optionally:

        buffer_setpoints: array
        buffer_setpoint_name: str, None
        buffer_setpoint_label: str, None
        buffer_setpoint_unit: str, None
        records_setpoints: array, None
        records_setpoint_name: str, None
        records_setpoint_label: str, None
        records_setpoint_unit: str, None
        """
        if not self.sequence._sequencer_up_to_date:
            return None
        if self.sequence.sequence_mode() == 'element':
            seq_mode = False
        else:
            seq_mode = True
        num = self.readout.num() or 1
        single_shot = self.readout.single_shot()
        integrate_time = self.readout.average_time()
        settings = {'integrate_time': integrate_time}
        if not single_shot:
            settings['average_buffers'] = True
            if (seq_mode and self.sequence.inner.symbol() is not None):
                if self.sequence.outer.setpoints is not None:
                    warn('Averaging channel will average over '
                                'outer setpoints of sequencer sequence')
                record_symbol = self.sequence.inner.setpoints[0]
                record_parameter = self.pulse_building_parameters.get(record_symbol, None)
                if record_parameter is not None:
                    record_label = record_parameter.label or record_symbol.title()
                    record_unit = record_parameter.unit or ''
                else:
                    record_label = record_symbol.title()
                    record_unit = ''
                record_setpoints = self.sequence.inner.setpoints[1]
                settings['records'] = len(record_setpoints)
                settings['buffers'] = num
                settings['average_records'] = False
                settings['record_setpoints'] = record_setpoints
                settings['record_setpoint_name'] = record_symbol
                settings['record_setpoint_label'] = record_label
                settings['record_setpoint_unit'] =  record_unit
            else:
                settings['average_records'] = True
                max_samples = self.alazar_controller.board_info['max_samples']
                samples_per_rec = self.alazar_controller.samples_per_record()
                max_records_per_buffer = math.floor(
                    max_samples / samples_per_rec)
                tot_samples = num * samples_per_rec
                if tot_samples > max_samples:
                    settings['records'] = max_records_per_buffer
                    settings['buffers'] = math.ceil(
                        num / max_records_per_buffer)
                else:
                    settings['records'] = num
                    settings['buffers'] = 1
        else:
            settings['average_buffers'] = False
            settings['average_records'] = False
            if seq_mode and self.sequence.inner.symbol() is not None:
                if self.sequence.outer is not None and num > 1:
                    raise RuntimeError(
                        'Cannot have outer setpoints and multiple nreps')
                record_symbol = self.sequence.inner.setpoints[0]
                record_parameter = self.pulse_building_parameters.get(record_symbol, None)
                if record_parameter is not None:
                    record_label = record_parameter.label or record_symbol.title()
                    record_unit = record_parameter.unit or ''
                else:
                    record_label = record_symbol.title()
                    record_unit = ''
                record_setpoints = self.sequence.inner.setpoints[1]
                settings['records'] = len(record_setpoints)
                settings['record_setpoints'] = record_setpoints
                settings['record_setpoint_name'] = record_symbol
                settings['record_setpoint_label'] = record_label
                settings['record_setpoint_unit'] = record_unit
                if self.sequence.outer.symbol() is not None:
                    buffer_symbol = self.sequence.outer.setpoints[0]
                    buffer_parameter = self.pulse_building_parameters.get(buffer_symbol, None)
                    if buffer_parameter is not None:
                        buffer_label = buffer_parameter.label or buffer_symbol.title()
                        buffer_unit = record_parameter.unit or ''
                    else:
                        buffer_label = buffer_symbol.title()
                        buffer_unit = ''
                    buffer_setpoints = self.sequence.outer.setpoints[1]
                    settings['buffers'] = len(buffer_setpoints)
                    settings['buffer_setpoints'] = buffer_setpoints
                    settings['buffer_setpoint_name'] = buffer_symbol
                    settings['buffer_setpoint_label'] = buffer_label
                    settings['buffer_setpoint_unit'] = buffer_unit
                else:
                    settings['buffers'] = num
                    settings['buffer_setpoints'] = np.arange(num)
                    settings['buffer_setpoint_name'] = 'repetitions'
                    settings['buffer_setpoint_label'] = 'Repetitions'
                    settings['buffer_setpoint_unit'] = None
            else:
                max_samples = self.alazar_controller.board_info['max_samples']
                samples_per_rec = self.alazar_controller.samples_per_record()
                tot_samples = num * samples_per_rec
                if tot_samples > max_samples:
                    records = math.floor(max_samples / samples_per_rec)
                    buffers = math.ceil(max_samples / records)
                else:
                    records = num
                    buffers = 1
                settings['records'] = records
                settings['buffers'] = buffers
                settings['record_setpoints'] = np.arange(records)
                settings['record_setpoint_name'] = 'record_repetitions'
                settings['record_setpoint_label'] = 'Record Repetitions'
                settings['buffer_setpoints'] = np.arange(buffers)
                settings['buffer_setpoint_name'] = 'buffer_repetitions'
                settings['buffer_setpoint_label'] = 'Buffer Repetitions'
        settings.update(self._drive_setpoints_from_sideband_setpoints(settings))
        return settings

    def _drive_setpoints_from_sideband_setpoints(self, settings):
        for setpoint_type in ['record_setpoint', 'buffer_setpoint']:
            try:
                name = settings[setpoint_type + '_name']
            except KeyError:
                return {}
            if name.endswith('sideband_frequency'):
                new_settings = {}
                if 'drive' in name:
                    carrier_freq = self.drive.carrier_frequency()
                elif 'readout' in name:
                    carrier_freq = self.readout.carrier_frequency()
                new_name = name.replace('sideband_frequency', 'frequency')
                new_label = new_name.replace('_', ' ').title()
                new_settings[setpoint_type + '_name'] = new_name
                new_settings[setpoint_type + '_label'] = new_label
                new_settings[setpoint_type + 's'] = settings[setpoint_type + 's'] + carrier_freq
                print(new_settings)
        return new_settings