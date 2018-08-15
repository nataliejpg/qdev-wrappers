from typing import Dict, List, Tuple, Sequence
from broadbean.element import Element
from broadbean.types import ContextDict, Symbol
import logging
import numpy as np
from contextlib import contextmanager
from qcodes import Station, Instrument, ChannelList
from qcodes.instrument.channel import InstrumentChannel
from qdev_wrappers.parameters import DelegateParameter
from qdev_wrappers.alazar_controllers.ATSChannelController import (
    ATSChannelController, AlazarMultiChannelParameter)
from qdev_wrappers.alazar_controllers.acquisition_parameters import NonSettableDerivedParameter
from qdev_wrappers.alazar_controllers.alazar_channel import AlazarChannel
from qcodes.utils import validators as vals


logger = logging.getLogger(__name__)


class AlazarChannel_ext(AlazarChannel):
    def __init__(self, parent, pwa, name: str,
                 demod: bool=False,
                 demod_ch=None,
                 alazar_channel: str='A',
                 average_buffers: bool=True,
                 average_records: bool=True,
                 integrate_samples: bool=True):
        super().__init__(parent, name, demod=demod,
                         alazar_channel=alazar_channel,
                         average_buffers=average_buffers,
                         average_records=average_records,
                         integrate_samples=integrate_samples)
        del self.parameters['num_averages']
        self.add_parameter(name='single_shot',
                           set_cmd=False,
                           get_cmd=self._get_single_shot_status)
        if self.single_shot():
            self.add_parameter(name='num_reps',
                               set_cmd=self._set_num,
                               get_cmd=lambda: self._num)
        else:
            self.add_parameter(name='num_averages',
                               set_cmd=self._set_num,
                               get_cmd=lambda: self._num)
        self._demod_ch = demod_ch
        self._pwa = pwa
        self._num = 1

    def _get_single_shot_status(self):
        if not self._average_records and not self._average_buffers:
            return True
        return False

    def _set_num(self, val):
        settings = self._pwa.get_alazar_ch_settings(val, self.single_shot())
        self.update(settings)
        self._num = val

    def update(self, settings):
        fail = False
        if settings['average_records'] != self._average_records:
            fail = True
        elif settings['average_buffers'] != self._average_buffers:
            fail = True
        if fail:
            raise RuntimeError(
                'alazar channel cannot be updated to change averaging '
                'settings, run clear_channels before changing settings')  # TODO
        self.records_per_buffer._save_val(settings['records'])
        self.buffers_per_acquisition._save_val(settings['buffers'])
        if self.dimensions == 1 and self._integrate_samples:
            self.prepare_channel(
                setpoints=settings['record_setpoints'],
                setpoint_name=settings['record_setpoint_name'],
                setpoint_label=settings['record_setpoint_label'],
                setpoint_unit=settings['record_setpoint_unit'])
        elif self.dimensions == 1:
            self.prepare_channel()
        else:
            self.prepare_channel(
                record_setpoints=settings['record_setpoints'],
                buffer_setpoints=settings['buffer_setpoints'],
                record_setpoint_name=settings['record_setpoint_name'],
                buffer_setpoint_name=settings['buffer_setpoint_name'],
            )
        self._num = settings['num']
        if self.single_shot():
            self.num_reps._save_val(settings['num'])
        else:
            self.num_averages._save_val(settings['num'])


class DemodulationChannel(InstrumentChannel):
    def __init__(self, parent, name: str, index: int, drive_frequency=None) -> None:
        """
        This is a channel which does the logic assuming a carrier
        microwave source and a local os separated by 'base_demod'
        which the parent of this channel knows about. The channel then
        takes care of working out what frequency will actually be output
        ('drive') if we sideband the carrier byt a 'sideband' and what the
        total demodulation frequency is in this case 'demodulation'.

        Note that no actual mocrowave sources are updated and we expect any
        heterodyne source used with this to do the legwork and update the
        parent carrier and base_demod

        Note that no awg is actaully connected so all it does is mark a flag
        as False

        Note that it DOES however have access to alazar channels so it can
        update their demod frequency and there is also a channel list so you
        could get the measurmenemts for everything demodulated at this one
        frequency which should correspond to measuring one qubit :)

        # TODO: what about if we have new fancy version with only one microwave
        source or if we don't sideband?
        """
        super().__init__(parent, name)
        self.index = index
        self.add_parameter(
            name='sideband_frequency',
            alternative='drive_frequency, heterodyne_source.frequency',
            parameter_class=NonSettableDerivedParameter)
        self.add_parameter(
            name='demodulation_frequency',
            alternative='drive_frequency, '
            'heterodyne_source.demodulation_frequency',
            parameter_class=NonSettableDerivedParameter)
        self.add_parameter(
            name='drive_frequency',
            set_cmd=self._set_drive_freq)
        alazar_channels = ChannelList(
            self, "Channels", AlazarChannel,
            multichan_paramclass=AlazarMultiChannelParameter)
        self.add_submodule("alazar_channels", alazar_channels)
        if drive_frequency is not None:
            self.drive_frequency(drive_frequency)

    def _set_drive_freq(self, drive_frequency):
        """
        changes the sideband frequencies in order to get the required drive,
        marks the awg as not up to date and updates the demodulation
        frequencies on the relevant alazar channels
        """
        sideband = self._parent._carrier_freq - drive_frequency
        demod = self._parent._base_demod_freq + sideband
        sequencer_sideband = getattr(
            self._parent.sequencer.sequence,
            'readout_sideband_{}'.format(self.index))
        sequencer_sideband(sideband)
        self.sideband_frequency._save_val(sideband)
        self.demodulation_frequency._save_val(demod)
        for ch in self.alazar_channels:
            ch.demod_freq(demod)

    def update(self, sideband=None, drive=None):
        """
        updates everything based on the carrier and base demod of
        the parent, either using existing settings or if a new drive
        is specified will update the sideband
        or a new sideband specified will cause the drive to be updated
        """
        base_demod = self._parent._base_demod_freq
        carrier = self._parent._carrier_freq
        old_sideband = self.sideband_frequency()
        old_demod = self.demodulation_frequency()
        if sideband is not None and drive is not None:
            raise RuntimeError('Cannot set drive and sideband simultaneously')
        elif drive is not None:
            sideband = carrier - drive
        elif sideband is not None:
            drive = carrier - sideband
        else:
            sideband = self.sideband_frequency()
            drive = carrier - sideband
        demod = base_demod + sideband
        if old_sideband != sideband:
            sequencer_sideband = getattr(
                self._parent.sequencer.sequence,
                'readout_sideband_{}'.format(self.index))
            sequencer_sideband(sideband)
            self.sideband_frequency._save_val(sideband)
        self.drive_frequency._save_val(drive)
        if demod != old_demod:
            self.demodulation_frequency._save_val(demod)
            for ch in self.alazar_channels:
                ch.demod_freq(demod)


class ParametricWaveformAnalyser(Instrument):
    """
    The PWA represents a composite instrument. It is similar to a
    spectrum analyzer, but instead of a sine wave it probes using
    waveforms described through a set of parameters.
    For that functionality it compises an AWG and a Alazar as a high speed ADC.
    """
    # TODO: write code for single microwave source
    # TODO: go through and use right types of parameters

    def __init__(self,
                 name: str,
                 sequencer,
                 alazar,
                 heterodyne_source,
                 initial_sequence_settings: Dict=None,
                 station: Station=None) -> None:
        super().__init__(name)
        self.add_parameter('')
        self.station, self.sequencer = station, sequencer
        self.alazar, self.heterodyne_source = alazar, heterodyne_source
        self.alazar_controller = ATSChannelController(
            'alazar_controller', alazar.name)
        self.alazar_channels = self.alazar_controller.channels
        demod_channels = ChannelList(self, "Channels", DemodulationChannel)
        self.add_submodule("demod_channels", demod_channels)
        self._base_demod_freq = heterodyne_source.demodulation_frequency()
        self._carrier_freq = heterodyne_source.frequency()
        self.add_parameter(name='int_time',
                           set_cmd=self._set_int_time,
                           initial_value=1e-6)
        self.add_parameter(name='int_delay',
                           set_cmd=self._set_int_delay,
                           initial_value=0)
        self.add_parameter(name='seq_mode',
                           set_cmd=self._set_seq_mode,
                           get_cmd=self._get_seq_mode)
        self.add_parameter(name='carrier_frequency',
                           set_cmd=self._set_carrier_frequency,
                           initial_value=self._carrier_freq)
        self.add_parameter(name='base_demodulation_frequency',
                           set_cmd=self._set_base_demod_frequency,
                           initial_value=self._base_demod_freq)
        self._sequence_settings = initial_sequence_settings
        awg_seq_mode = True if self.sequencer.repeat_mode() == 'sequence' else False
        self.seq_mode(awg_seq_mode)

    def _set_int_time(self, int_time):
        self.alazar_controller.int_time(int_time)
        for ch in self.alazar_channels:
            if not ch._integrate_samples:
                ch.prepare_channel()

    def _set_int_delay(self, int_delay):
        self.alazar_controller.int_delay(int_delay)
        for ch in self.alazar_channels:
            if not ch._integrate_samples:
                ch.prepare_channel()

    def _set_seq_mode(self, mode):
        """
        updated the sequencing mode on the alazar and the awg and reset all
        the alazar channels so that they average over everything which is a
        sensible default for if we are just playing one element on loop
        # TODO: what if we also want statistics or a time trace?
        # TODO: num_averages
        """
        if str(mode).upper() in ['TRUE', '1', 'ON']:
            self.alazar.seq_mode('on')
            self.sequencer.repeat_mode('sequence')
        else:
            self.alazar.seq_mode('off')
            self.sequencer.repeat_mode('element')
        settings_list = []
        for ch in self.alazar_channels:
            settings = {'demod_channel_index': ch._demod_ch.index,
                        'demod_type': ch.demod_type()[0],
                        'integrate_time': ch._integrate_samples,
                        'single_shot': ch.single_shot(),
                        'num_reps': ch._num,
                        'num_averages': ch._num}
            settings_list.append(settings)
        self.clear_alazar_channels()
        for settings in settings_list:
            self.add_alazar_channel(**settings)


    def _get_seq_mode(self):
        if self.alazar.seq_mode() == 'on' and self.sequencer.repeat_mode() == 'sequence':
            return True
        elif self.alazar.seq_mode() == 'off' and self.sequencer.repeat_mode() == 'element':
            return False
        elif (self.alazar.seq_mode() == 'off') and (len(self.sequencer.get_inner_setpoints()) == 1 and self.sequencer.get_outer_setpoints() is None):
            return False
        else:
            raise RuntimeError(
                'seq modes on sequencer and alazar do not match')

    def _set_base_demod_frequency(self, demod_freq):
        """
        update the demodulation frequency locally and also runs update
        on the demodulation channels to propagate this to the demodulation
        frequencies after sidebanding which should end up on the alazar
        channels
        """
        self._base_demod_freq = demod_freq
        self.heterodyne_source.demodulation_frequency(demod_freq)
        for demod_ch in self.demod_channels:
            demod_ch.update()

    @contextmanager
    def sideband_update(self):
        old_drives = [demod_ch.drive_frequency()
                      for demod_ch in self.demod_channels]
        yield
        for i, demod_ch in enumerate(self.demod_channels):
            demod_ch.update(drive=old_drives[i])

    def _set_carrier_frequency(self, carrier_freq):
        """
        update the carrier frequency locally and also runs update
        on the demodulation channels to propagate this to the demodulation
        frequencies after sidebanding which should end up on the alazar
        channels there is option to change the sidebands to keep the resultant
        drive frequencies the same or to leave them as is and then the drive
        changes
        """
        self._carrier_freq = carrier_freq
        self.heterodyne_source.frequency(carrier_freq)
        for demod_ch in self.demod_channels:
            demod_ch.update()

    def add_demodulation_channel(self, drive_frequency):
        demod_ch_num = len(self.demod_channels)
        demod_ch = DemodulationChannel(
            self, 'ch_{}'.format(demod_ch_num), demod_ch_num,
            drive_frequency=drive_frequency)
        self.demod_channels.append(demod_ch)

    def clear_demodulation_channels(self):
        for ch in list(self.demod_channels):
            self.demod_channels.remove(ch)
        for ch in list(self.alazar_channels):
            self.alazar_channels.remove(ch)

    def add_alazar_channel(
            self, demod_ch_index: int, demod_type: str, single_shot: bool=False,
            num_averages: int=1, num_reps: int=1, integrate_time: bool=True):
        if single_shot:
            settings = self.get_alazar_ch_settings(num_reps, True)
        else:
            settings = self.get_alazar_ch_settings(num_averages, False)
        demod_ch = self.demod_channels[demod_ch_index]
        name = 'ch_{}_{}'.format(demod_ch_index, demod_type)
        averaging_settings = {
            'integrate_time': integrate_time,
            **{k: settings[k] for k in ('average_records', 'average_buffers')}, }
        appending_string = '_'.join(
            [k.split('_')[1] for k, v in averaging_settings.items() if not v])
        if appending_string:
            name += '_' + appending_string
        chan = AlazarChannel_ext(self.alazar_controller,
                                 self,
                                 name=name,
                                 demod=True,
                                 demod_ch=demod_ch,
                                 average_records=settings['average_records'],
                                 average_buffers=settings['average_buffers'],
                                 integrate_samples=integrate_time)
        chan.demod_freq(demod_ch.demodulation_frequency())
        if demod_type in 'm':
            chan.demod_type('magnitude')
            chan.data.label = 'Cavity Magnitude Response'
        elif demod_type == 'p':
            chan.demod_type('phase')
            chan.data.label = 'Cavity Phase Response'
        elif demod_type == 'i':
            chan.demod_type('imag')
            chan.data.label = 'Cavity Imaginary Response'
        elif demod_type == 'r':
            chan.demod_type('real')
            chan.data.label = 'Cavity Real Response'
        else:
            raise NotImplementedError(
                'only magnitude and phase, imaginary and real currently implemented')
        self.alazar_controller.channels.append(chan)
        chan.update(settings)
        demod_ch.alazar_channels.append(chan)

    def set_sequencer_template(
            self,
            template_element: Element,
            inner_setpoints: Tuple[Symbol, Sequence],
            context: ContextDict={},
            units: Dict[Symbol, str]={},
            labels: Dict[Symbol, str]={},
            first_sequence_element: Element=None,
            initial_element: Element=None):
        self._sequence_settings['context'].update(context)
        self._sequence_settings['units'].update(units)
        self._sequence_settings['labels'].update(labels)
        self.sequencer.set_template(template_element,
                                    inner_setpoints=inner_setpoints,
                                    context=self._sequence_settings['context'],
                                    units=self._sequence_settings['units'],
                                    labels=self._sequence_settings['labels'],
                                    first_sequence_element=first_sequence_element,
                                    initial_element=initial_element)
        for ch in list(self.alazar_channels):
            settings = self.get_alazar_ch_settings(
                ch._num, single_shot=ch.single_shot())
            ch.update(settings)

    def clear_saved_sequence_settings(self):
        self._sequence_settings = {'context': {}, 'units': {}, 'labels': {}}

    def clear_alazar_channels(self):
        if self.alazar_channels is not None:
            for ch in list(self.alazar_channels):
                self.alazar_channels.remove(ch)
                del ch
            for demod_ch in list(self.demod_channels):
                for alazar_ch in demod_ch.alazar_channels:
                    demod_ch.alazar_channels.remove(alazar_ch)

    def make_all_alazar_channels_play_nice(self):
        raise NotImplementedError

    def get_alazar_ch_settings(self, num: int, single_shot: bool):
        '''
        If single shot then num is number of reps, else it is the
        number of averages
        '''
        seq_mode = self.seq_mode()
        settings = {'num': num}
        if not single_shot:
            settings['average_buffers'] = True
            settings['buffer_setpoints'] = None
            settings['buffer_setpoint_name'] = None
            settings['buffer_setpoint_label'] = None
            settings['buffer_setpoint_unit'] = None
            if seq_mode and len(self.sequencer.get_inner_setpoints().values) > 1:
                if self.sequencer.get_outer_setpoints() is not None:
                    logger.warn('Averaging channel will average over '
                                'outer setpoints of AWG sequence')
                record_symbol = self.sequencer.get_inner_setpoints().symbol
                record_setpoints = self.sequencer.get_inner_setpoints().values
                record_param = getattr(self.sequencer.repeat, record_symbol)
                settings['records'] = len(record_setpoints)
                settings['buffers'] = num
                settings['average_records'] = False
                settings['record_setpoints'] = record_setpoints
                settings['record_setpoint_name'] = record_symbol
                settings['record_setpoint_label'] = record_param.label
                settings['record_setpoint_unit'] = record_param.unit

            else:
                settings['average_records'] = True
                max_samples = self.alazar_controller.board_info['max_samples']
                samples_per_rec = self.alazar_controller.samples_per_record()
                tot_samples = num * samples_per_rec
                if tot_samples > max_samples:
                    settings['records'] = math.floor(
                        max_samples / samples_per_rec)
                    settings['buffers'] = math.ceil(max_samples / records)
                else:
                    settings['records'] = num
                    settings['buffers'] = 1
                settings['record_setpoints'] = None
                settings['record_setpoint_name'] = None
                settings['record_setpoint_label'] = None
                settings['record_setpoint_unit'] = None
        else:
            settings['average_buffers'] = False
            settings['average_records'] = False
            if seq_mode and len(self.sequencer.get_inner_setpoints().values) > 1:
                if self.sequencer.get_outer_setpoints() is not None and num > 1:
                    raise RuntimeError(
                        'Cannot have outer setpoints and multiple nreps')
                record_symbol = self.sequencer.get_inner_setpoints().symbol
                record_setpoints = self.sequencer.get_inner_setpoints().values
                records_param = getattr(self.sequencer.repeat, record_symbol)
                settings['records'] = len(record_setpoints)
                settings['record_setpoints'] = record_setpoints
                settings['record_setpoint_name'] = record_symbol
                settings['record_setpoint_label'] = records_param.label
                settings['record_setpoint_unit'] = records_param.unit
                if self.sequencer.get_outer_setpoints() is not None:
                    buffers_symbol = self.sequencer.get_outer_setpoints().symbol
                    buffers_setpoints = self.sequencer.get_outer_setpoints().values
                    buffers_param = getattr(
                        self.sequencer.repeat, buffers_symbol)
                    settings['buffer_setpoints'] = buffers_setpoints
                    settings['buffer_setpoint_name'] = buffers_symbol
                    settings['buffer_setpoint_label'] = buffers_param.label
                    settings['buffer_setpoint_unit'] = buffers_param.unit
                    settings['buffers'] = len(buffer_setpoints)
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
                settings['record_setpoint_unit'] = None
                settings['buffer_setpoints'] = np.arange(buffers)
                settings['buffer_setpoint_name'] = 'buffer_repetitions'
                settings['buffer_setpoint_label'] = 'Buffer Repetitions'
                settings['buffer_setpoint_unit'] = None
        return settings
