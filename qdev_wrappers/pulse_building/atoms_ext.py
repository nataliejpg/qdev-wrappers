import numpy as np
from lomentum.atoms import atom


@atom
def sine_multi(time, frequencies=None, amplitudes=1, phases=0):
    if time.size == 1:
        return 0
    if frequencies is None:
        frequencies = [0]
    if type(amplitudes) in [int, float]:
        amplitudes = np.ones(len(frequencies)) * amplitudes
    if type(phases) in [int, float]:
        phases = np.ones(len(frequencies)) * phases
    if not (len(frequencies) == len(amplitudes) and
            len(frequencies) == len(phases)):
        raise Exception(
            '{} frequencies, {} amplitudes and {} phases provided'.format(
                len(frequencies), len(amplitudes), len(phases)))
    output = np.zeros(time.shape)
    for i, frequency in enumerate(frequencies):
        output += amplitudes[i] * \
            np.sin(frequency * 2 * np.pi * time + phases[i])
    return output / len(frequencies)


@atom
def sine_modulated(time, mod_frequency, frequency, mod_phase=0, amplitude=1, phase=0):
    if time.size == 1:
        return 0
    return amplitude * np.sin(mod_frequency * 2 * np.pi * time + mod_phase) *  np.sin(frequency * 2 * np.pi * time + phase)

@atom
def gaussianDRAG(time, sigma_cutoff=4, amplitude=1, DRAG=False, offset=0):
    if time.size == 1:
        return 0
    sigma = time[-1] / (2 * sigma_cutoff)
    t = time - time[-1] / 2
    if DRAG:
        return amplitude * t / sigma * np.exp(-(t / (2. * sigma))**2) + offset
    else:
        return amplitude * np.exp(-(t / (2. * sigma))**2) + offset

@atom
def sine_gaussianDRAG(time, frequency, phase=0, sigma_cutoff=4, amplitude=1, DRAG=False,
                      offset=0):
    if time.size == 1:
        return 0
    sigma = time[-1] / (2 * sigma_cutoff)
    t = time - time[-1] / 2
    sideband =  np.sin(frequency * 2 * np.pi * time + phase)
    if DRAG:
        return sideband * t / sigma * np.exp(-(t / (2. * sigma))**2) + offset
    else:
        return sideband * np.exp(-(t / (2. * sigma))**2) + offset


@atom
def gaussian_ramp(time, start=0, stop=1, sigma_cutoff=4):
    if time.size == 1:
        return 0
    if start < stop: 
        t = (time - time[0])[::-1]
    else:
        t = (time - time[0])
    dur = time[-1] - time[0]
    sigma = dur / sigma_cutoff
    res = np.exp(-(t / (2. * sigma))**2)
    amplitude = abs(start - stop)
    offset = min(start, stop)
    res *= amplitude / (max(res) - min(res))
    res += -min(res) + offset
    return res


@atom
def cosine_ramp(time, start=0, stop=1):
    if time.size == 1:
        return 0
    if start < stop: 
        t = (time - time[0])[::-1]
    else:
        t = (time - time[0])
    dur = time[-1] - time[0]
    freq = 1 / (2 * dur)
    res = np.cos(t * 2 * np.pi * freq)
    amplitude = abs(start - stop)
    offset = min(start, stop)
    res = (res + 1) * amplitude / 2 + offset
    return res

