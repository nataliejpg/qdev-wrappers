import numpy as np
from pulse_building_legacy.rabi_template_element import create_rabi_template_element
from pulse_building.readout_template_element import create_readout_template_element
from pulse_building_legacy.spectroscopy_ssb_template_element import create_spectroscopy_ssb_template_element
from pulse_building_legacy.initial_element import create_initial_element


def cavity_sweep_upload(pwa):
    pwa.clear_alazar_channels()
    pwa.seq_mode(0)

    template_element = create_readout_template_element()
    pwa.set_sequencer_template(template_element, inner_setpoints=('dummy_time', [1]))
    pwa.add_alazar_channel(0, 'm', num=1000)
    

def rabi_upload(pwa, num_avgs, max_duration=800e-9, gaussian=False, DRAG=False):
    num_pulses = int(max_duration*1e9+1)
    if gaussian:
        pulse_durations = np.linspace(0e-9, max_duration, num=num_pulses) + 2e-9
    else:
        pulse_durations = np.linspace(0e-9, max_duration, num=num_pulses)
    pwa.clear_alazar_channels()
    pwa.seq_mode(1)
    template_element = create_rabi_template_element(gaussian=gaussian, DRAG=DRAG)
    pwa.set_sequencer_template(
            template_element,
            inner_setpoints=('pulse_duration', pulse_durations),
            initial_element=create_initial_element(),
            context={'qubit_marker_duration': 1e-6})
    pwa.add_alazar_channel(0, 'r', num=num_avgs)
    pwa.add_alazar_channel(0, 'i', num=num_avgs)


def readout_fidelity_upload(pi_pulse_duration, pwa, gaussian=False, DRAG=False):
    num_reps = 3000

    pwa.clear_alazar_channels()
    pwa.seq_mode(1)
    template_element = create_rabi_template_element(gaussian=gaussian, DRAG=DRAG)
    pwa.set_sequencer_template(
        template_element,
        inner_setpoints=('pulse_amplitude', [0, 1]),
        context={'pulse_duration': pi_pulse_duration,
                 'qubit_marker_duration': 1e-6},
        initial_element=create_initial_element())

    pwa.add_alazar_channel(0, 'r', single_shot=True, num=num_reps)
    pwa.add_alazar_channel(0, 'i', single_shot=True, num=num_reps)


def spec_upload(pwa, frequency_span=300e6, frequency_step=1e6):
    pwa.clear_alazar_channels()
    pwa.seq_mode(1)

    frequency_npts = int(frequency_span / frequency_step + 1)
    upper_sideband_frequencies = np.linspace(frequency_span/2, -frequency_span/2, num=frequency_npts)

    template_element = create_spectroscopy_ssb_template_element()
    pwa.set_sequencer_template(
        template_element,
        inner_setpoints=('drive_sideband_frequency', upper_sideband_frequencies),
        context={'drive_sideband_frequency': 0,
                 'pulse_duration': 1e-6,
                 'qubit_marker_duration': 5e-6},
        initial_element=create_initial_element())
