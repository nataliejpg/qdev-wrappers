from lomentum.types import ForgedSequenceType
from lomentum.plotting import plotter
import numpy as np


class AWGInterface:

    def upload(self, forged_sequence: ForgedSequenceType):
        raise NotImplementedError()

    def set_infinit_loop(self, element_index: int,
                         true_on_false_off: bool):
        raise NotImplementedError()

    def set_repeated_element(self, index):
        raise NotImplementedError()

    def set_repeated_element_series(self, start_index, stop_index):
        raise NotImplementedError()

    def repeat_full_sequence(self):
        raise NotImplementedError()

    def get_SR(self):
        raise NotImplementedError()


class SimulatedAWGInterface(AWGInterface):
    def __init__(self):
        self.forged_sequence = None

    def upload(self, forged_sequence: ForgedSequenceType):
        print(f'uploading')
        SR = self.get_SR()
        self.forged_sequence = forged_sequence
        plotter(forged_sequence, SR=SR)

    def set_repeated_element(self, index):
        print(f'setting repeated element to {index}')
        # AWG is not zero indexed but one, convert to zero index
        index -= 1
        if self.forged_sequence is None:
            print(f'but there was not sequence uploaded')
            return
        plotter(self.forged_sequence[index], SR=self.get_SR())

    def set_repeated_element_series(self, start_index, stop_index):
        print(f'setting repeated element series from {start_index} to '
              f'{stop_index}')
        # AWG is not zero indexed but one, convert to zero index
        start_index -= start_index
        stop_index -= stop_index
        if self.forged_sequence is None:
            print(f'but there was not sequence uploaded')
            return
        plotter(self.forged_sequence[start_index:stop_index], SR=self.get_SR())

    def repeat_full_sequence(self):
        print(f'repeating full series')
        plotter(self.forged_sequence, SR=self.get_SR())

    def get_SR(self):
        # fake this for now
        return 1e9


class AWG5014Interface(AWGInterface):
    def __init__(self, awg):
        self.awg = awg
        self.last_repeated_element = None
        self.forged_sequence = None
        self.last_repeated_element_series = (None, None)

    def upload(self, forged_sequence: ForgedSequenceType):
        self.awg.make_send_and_load_awg_file_from_forged_sequence(
            forged_sequence)
        self.forged_sequence = forged_sequence
        # uploading a sequence results in reverting the information on the
        # elements
        self.last_repeated_element = None
        self.last_repeated_element_series = (None, None)
        self.awg.all_channels_on()
        self.awg.run()

    def set_repeated_element(self, index):
        print(f'stop repeating {self.last_repeated_element} start {index}')
        self.awg.set_sqel_loopcnt_to_inf(index, state=1)
        self.awg.sequence_pos(index)
        if (self.last_repeated_element is not None and
                self.last_repeated_element != index):
            self.awg.set_sqel_loopcnt_to_inf(self.last_repeated_element,
                                             state=0)
        self.last_repeated_element = index

    def set_repeated_element_series(self, start_index, stop_index):
        self._restore_sequence_state()
        self.awg.set_sqel_goto_target_index(stop_index, start_index)
        self.awg.sequence_pos(start_index)

    def repeat_full_sequence(self):
        self._restore_sequence_state()
        self.awg.sequence_pos(1)

    def get_SR(self):
        return self.awg.clock_freq()

    def _restore_sequence_state(self):
        if self.last_repeated_element is not None:
            self.awg.set_sqel_loopcnt_to_inf(self.last_repeated_element,
                                             state=0)
        lres = self.last_repeated_element_series
        if lres[0] is not None or lres[1] is not None:
            assert (lres[0] is not None and
                    lres[1] is not None and
                    self.forged_sequence is not None)
            if lres[0] == len(self.forged_sequence):
                goto_element = 1
            else:
                goto_element = 0
            self.awg.set_sqel_goto_target_index(lres[0], goto_element)


class Abaco4DSPInterface(AWGInterface):
    def __init__(self, awg):
        self.awg = awg
        self.forged_sequence = None

    def upload(self, forged_sequence: ForgedSequenceType):
        self.awg.make_and_send_awg_file(forged_sequence)
        self.forged_sequence = forged_sequence
        self.awg.load_waveform_from_file(self.awg.FILENAME)
        self.awg.run()

    def get_SR(self):
#        return self.awg.clock_freq()
        return 1e9


class AbacoTektronixInterface(AWGInterface):
    """For uploading to the Abaco4DSP system and also uploading to the marker channels on the Tektronix5014C, 
    to trigger the 4DSP and MIDAS. Adds a delay every 2048 sequence elements to allow MIDAS buffer to flush"""
    
    def __init__(self, abaco, tektronix):
        self.deadtime = 10e-3
        self.burst_size = 2048
        self.pulse_spacing = None   # desired space between output triggers in microseconds, f.x. 20e-6
        self.abaco = abaco
        self.tek = tektronix
        self.forged_sequence = None
        # for tektronix to be in sequence mode (not used by 4dsp):
        self.last_repeated_element = None
        self.last_repeated_element_series = (None, None)
        self._last_go_to = 1
        self.upload_to_4dsp = True
        self.upload_to_tek = True

    def upload(self, forged_sequence: ForgedSequenceType):

        self.tek.stop()

        # make sure that the bust size is divisible by the number of pulses in the sequence
        # ToDo: this is not necesary if a 10ms break in measurement in the middle of the sequence is not a problem - it should possible by optional
        if self.burst_size % len(forged_sequence) != 0:
            raise RuntimeError(f"{len(forged_sequence)} pulses doesn't divide evenly into burst size {self.burst_size}")
        reps_per_sequence = int(self.burst_size / len(forged_sequence))

        if self.upload_to_4dsp:
            print("Getting 4DSP forged sequence")
            # allocate numerical channel data to the abaco forged sequence
            all_channels = []
            abaco_fs = []

            for element in forged_sequence:
                abaco_fs.append({'data': {}, 
                                 'sequencing': element['sequencing']}) 
                for ch in element['data']:
                    all_channels.append(ch)
                    if isinstance(ch, int):
                        abaco_fs[-1]['data'][ch] = element['data'][ch]

            print("Organizing and saving 4DSP upload file")
            # upload the numbered channels to the Abaco4DSP
            self.abaco.make_and_send_awg_file(abaco_fs)
            self.abaco.load_waveform_from_file(self.abaco.FILENAME)

            self.abaco_sequence = abaco_fs

        if self.upload_to_tek:
            
            print("Making Tektronix forged sequence")

            print("Setting tektronix element duration")
            max_freq = self.abaco.max_trigger_freq() - 5e3
            # confirm that the pulse spacing is large enough to allow the 4DSP time to output each pulse element
            if self.pulse_spacing is not None:
                if self.pulse_spacing < 1/(max_freq):
                    raise RuntimeError(f"Specified pulse spacing for the Tektronix is {self.pulse_spacing}, but given Abaco pulse durations, the minimum pulse spacing for triggers is {1/(max_freq)}. Please decrease duration of Abaco pulses or increase trigger spacing.")
                tektronix_element_duration = self.pulse_spacing
            # OR find total duration per element for tektronix upload based on Abaco maximum trigger frequency
            else:
                tektronix_element_duration = 1/(max_freq)
            
            samples_per_tektronix_element = int(tektronix_element_duration * 1e9)
            samples_per_abaco_element = int(self.abaco.waveform_size()/(2*self.abaco.num_blocks())) # factor 2 is because waveform size comes back in bytes, not samples

            print("Creating deadtime data")
            padding = samples_per_tektronix_element - samples_per_abaco_element
            elements_of_deadtime = int(self.deadtime / tektronix_element_duration)
            self.trigger_spacing = tektronix_element_duration
            self.trigger_frequency = max_freq

            print("Putting together pulse burst and deadtime")
            # assume trigger elements are all identical (faster upload), make bursts of self.burst_size pulses
            tek_fs = [{'data': {ch : np.pad(forged_sequence[0]['data'][ch], (0, padding), 'constant', constant_values=(0, 0))
                                    for ch in forged_sequence[0]['data'] if not isinstance(ch, int)},
                       'sequencing': {'nrep': self.burst_size}}]

            # add inter-burst delay (default 10ms) for the tektronix sequence
            channels = [ch for ch in tek_fs[0]['data']]

            deadtime_data = {ch: np.zeros(samples_per_tektronix_element) for ch in channels}
            tek_fs.append({'data': deadtime_data, 
                           'sequencing' : {'nrep' : elements_of_deadtime-1}})
            tek_fs.append({'data': deadtime_data, 
                           'sequencing' : {'nrep' : 1, 'goto_state': self._last_go_to}})

            print("Uploading to the tektronix")
            # upload the marker channels to the Tektronix5014C (and set repeated_element info back to None)
            self.tek.make_send_and_load_awg_file_from_forged_sequence(tek_fs)
            self.last_repeated_element = None
            self.last_repeated_element_series = (None, None)
            self.tek.all_channels_on()

            self.tektronix_sequence = tek_fs

        self.forged_sequence = forged_sequence

        # start AWGs, first enabling Abaco to output, then starting the triggers
        self.abaco.run()
        self.tek.run()

    def repeat_full_sequence(self):
        self._restore_sequence_state()
        self.tek.sequence_pos(1)

    def change_pulse_spacing(self, pulse_spacing):
        self.upload_to_4dsp = False 
        self.pulse_spacing = pulse_spacing

        self.upload(self.forged_sequence)

        self.upload_to_4dsp = True
    def get_SR(self):
        # not sure what this should do, since there are two different awgs. Get both, I guess, and then throw an error if they are different? But maybe its okay if they are different... depends what this is used for.
#        return self.awg.clock_freq()
        return 1e9

    def _restore_sequence_state(self):
        if self.last_repeated_element is not None:
            self.tek.set_sqel_loopcnt_to_inf(self.last_repeated_element,
                                             state=0)
        lres = self.last_repeated_element_series
        if lres[0] is not None or lres[1] is not None:
            assert (lres[0] is not None and
                    lres[1] is not None and
                    self.forged_sequence is not None)
            if lres[0] == len(self.forged_sequence):
                goto_element = 1
            else:
                goto_element = 0
            self.tek.set_sqel_goto_target_index(lres[0], goto_element)