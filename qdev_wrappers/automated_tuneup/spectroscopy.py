import numpy as np
from qcodes.dataset.data_export import get_data_by_id
from qcodes.dataset.plotting import plot_by_id
from qdev_wrappers.doNd import do0d
from qdev_wrappers.automated_tuneup.pulse_uploads import rabi_upload, spec_upload
from qdev_wrappers.fitting.fitter import Fitter
from qdev_wrappers.fitting.least_squares_models import CosineModel


def set_frequency_center(pwa, qubit, frequency_center, frequency_span=300e6, frequency_step=1e6):
    pwa.clear_alazar_channels()
    pwa.add_alazar_channel(0, 'm', num=1000)
    pwa.add_alazar_channel(0, 'p', num=1000)

    frequency_npts = int(frequency_span / frequency_step + 1)
    upper_sideband_frequencies = np.linspace(frequency_span/2, -frequency_span/2, num=frequency_npts)

    pwa.alazar_channels.ch_0_m_records.data.setpoints = (frequency_center + upper_sideband_frequencies,)
    pwa.alazar_channels.ch_0_m_records.data.setpoint_names = ('qubit_drive_frequency',)
    pwa.alazar_channels.ch_0_m_records.data.setpoint_labels = ('Qubit Drive Frequency',)
    pwa.alazar_channels.ch_0_m_records.data.setpoint_units = ('Hz',)
    pwa.alazar_channels.ch_0_p_records.data.setpoints = (frequency_center + upper_sideband_frequencies,)
    pwa.alazar_channels.ch_0_p_records.data.setpoint_names = ('qubit_drive_frequency',)
    pwa.alazar_channels.ch_0_p_records.data.setpoint_labels = ('Qubit Drive Frequency',)
    pwa.alazar_channels.ch_0_p_records.data.setpoint_units = ('Hz',)

    qubit.frequency(frequency_center)


def find_peaks(pwa=None, runid=None, plot=False, from_saved_data=False):

    if from_saved_data:
        data = get_data_by_id(runid)
        mag = data[0][1]['data']
        phase = data[1][1]['data']
        freq = data[0][0]['data']
    else:
        data = pwa.alazar_channels.data()
        mag = data[0]
        phase = data[1]
        freq = pwa.alazar_channels.data.setpoints[0][0]
    
    # noise floor data is either the bottom or top 2/3rd in terms of magnitude
    if np.abs(np.mean(mag)-max(mag)) > np.abs(np.mean(mag)-min(mag)):
        noise_floor_data = np.sort(mag)[:int(len(mag)*0.66)]
    else:
        noise_floor_data = np.sort(mag)[int(len(mag)*0.33):]
    std_m = np.std(noise_floor_data)
    noise_floor_m = np.mean(noise_floor_data)

    if np.abs(np.mean(phase)-max(phase)) > np.abs(np.mean(phase)-min(phase)):
        noise_floor_data = np.sort(phase)[0:int(len(phase)*0.66)]
    else:
        noise_floor_data = np.sort(phase)[int(len(phase)*0.33):]
    std_p = np.std(noise_floor_data)
    noise_floor_p = np.mean(noise_floor_data)
    
    peaks = []
    new_peak = None
    
    for f, m, p in zip(freq, mag, phase):
        if np.abs(m - noise_floor_m) > 3*std_m and np.abs(p - noise_floor_p) > 3*std_p:
            if new_peak is None:
                new_peak = [(f, m, p)]
            else:
                new_peak.append((f, m, p))
        else:
            if new_peak is not None:
                if peaks:
                    # if the last frequency in the most recent peak is less than
                    # 2 MHz away from the first frequency in this peak, take
                    # them to be the same feature, and just add this 'new' peak
                    # on to the most recent peak in peaks
                    if np.abs(peaks[-1][-1][0]-new_peak[0][0]) <= 2e6:
                        peaks[-1] += new_peak
                    else:
                        peaks.append(new_peak)
                else:
                    peaks.append(new_peak)
                new_peak = None
    if new_peak is not None:
        peaks.append(new_peak)
                
    locations = []
    peak_mags = []
    peak_widths = []
    
    for peak in peaks:
        frequencies = [f for (f, m, p) in peak]
        magnitudes = [np.abs(m-noise_floor_m) for (f, m, p) in peak]
        width = peak[0][0]-peak[-1][0]

        locations.append(np.mean(frequencies))
        peak_mags.append(np.max(magnitudes))
        peak_widths.append(width)
    
    if plot:
        if runid is None:
            runid = (do0d(pwa.alazar_channels.data))
        axes, clb = plot_by_id(runid)
        axes[0].plot(locations, [noise_floor_m]*len(locations), marker='o', linestyle='')
        axes[0].plot([freq[0], freq[-1]], [noise_floor_m + 3 * std_m]*2, marker='', linestyle='-', color='C1')
        axes[0].plot([freq[0], freq[-1]], [noise_floor_m - 3 * std_m]*2, marker='', linestyle='-', color='C1')

        axes[1].plot(locations, [noise_floor_p]*len(locations), marker='o', linestyle='')
        axes[1].plot([freq[0], freq[-1]], [noise_floor_p + 3 * std_p]*2, marker='', linestyle='-', color='C1')
        axes[1].plot([freq[0], freq[-1]], [noise_floor_p - 3 * std_p]*2, marker='', linestyle='-', color='C1')

        # ToDo: add plot saving?

    return locations, peak_mags, peak_widths


def find_and_verify_peaks(pwa, plot=False):
    # find initial peak locations
    (locations, mags, widths) = find_peaks(pwa, plot)

    # if no peaks, return the empty lists
    if len(locations) == 0:
        return locations, mags, widths

    # if peaks are found, repeat measurement, and confirm that it is replicable
    else:
        (locations2, mags2, widths2) = find_peaks(pwa, plot)
        # if there are no peaks in the new measurement, return the empty lists
        if len(locations2) == 0:
            return locations2, mags2, widths2
        # keep only the peaks at indices where the 2 measurements found peaks within 1MHz of each other
        else:
            verified_peaks = ([], [], [])
            for i, freq in enumerate(locations):
                diff = [freq2-freq for freq2 in locations2]
                if min(np.abs(diff)) < 1e6:
                    verified_peaks[0].append(locations[i])
                    verified_peaks[1].append(mags[i])
                    verified_peaks[2].append(widths[i])
            return verified_peaks


def get_new_center(center_frequency, move, start=5e9):
    # Todo: should max be some distance below the lowest cavity instead of 6.35? Is 3.5 a resonable minimum?
    if start <= center_frequency <= 6.35e9:
        center_frequency += move
    elif 3.5e9 <= center_frequency < start:
        center_frequency -= move
    elif center_frequency > 6.35e9:
        center_frequency = start - move
    elif center_frequency < 3.5e9:
        center_frequency = start

    return center_frequency


def look_for_qubit(qubit, pwa, start_freq=5e9, qubit_power=-5, measure=False):
    qubit.power(qubit_power)
    qubit.status(1)
    freq_max_reached = False
    freq_min_reached = False

    transition_freqs = None
    f01 = None
    anharmonicity = None

    spec_upload(pwa)

    center_frequency = start_freq
    while not (freq_max_reached and freq_min_reached):
        set_frequency_center(pwa, qubit, center_frequency)
        if measure:
            do0d(pwa.alazar_channels.data)

        peaks = find_and_verify_peaks(pwa)
        num_peaks = len(peaks[0])

        # if no peaks, move 300MHz
        if num_peaks == 0:
            print(f"No peaks at frequency center {center_frequency}")
            if center_frequency > 6.35e9:
                freq_max_reached = True
            elif center_frequency < 3.5e9:
                freq_min_reached = True
            center_frequency = get_new_center(center_frequency, move=300e6, start=start_freq)
        elif num_peaks > 4:
            qubit_power -= 3
            qubit.power(qubit_power)
            print(f"Found {num_peaks} peaks at frequency center {center_frequency}.")
            print(f"Decreasing qubit drive power to {qubit_power}.")
        # otherwise, center on the tallest peak and repeat
        else:
            # candidates should be a list of possible qubit frequencies from lowest to highest
            candidates = np.sort(peaks[0])
            print(f"Found {num_peaks} peaks at frequency center {center_frequency}.")
            # verify_qubit uploads rabi sequence
            qubit_found, transition_freqs = verify_qubit(candidates, pwa, qubit)
            if qubit_found:
                break
            # if qubit not found yet, re-upload spectroscopy sequence, move to next location
            spec_upload(pwa)
            if center_frequency > 6.35e9:
                freq_max_reached = True
            elif center_frequency < 3.5e9:
                freq_min_reached = True
            center_frequency = get_new_center(center_frequency, move=300e6, start=start_freq)

    if transition_freqs is None:
        print("Searched full frequency range unsuccessfully")
    else:
        f01 = transition_freqs[-1]
        if len(transition_freqs) == 2:
            f02 = transition_freqs[0]
            anharmonicity = 2 * (f01 - f02)
        else:
            print(f"Found {len(transition_freqs)-1} candidates for 02 transition, could not determine anharmonicity")

    return f01, anharmonicity, qubit_power


def verify_qubit(candidates, pwa, qubit):

    is_qubit = False
    freqs = []
    fitter = Fitter(CosineModel())

    rabi_upload(pwa, 500)

    for candidate in candidates:
        qubit.frequency(candidate)
        pulse_dur = pwa.alazar_channels.data.setpoints[0][0]
        data = pwa.alazar_channels.data()[0]

        fit = fitter.fit(data, x=pulse_dur)

        if fit[0] is not None:
            omega = fit[0]['w']
            pi_pulse_duration = np.pi / omega
            if pi_pulse_duration <= 300e-9:
                freqs.append(candidate)

    if len(freqs) > 0:
        is_qubit = True

    return is_qubit, freqs
