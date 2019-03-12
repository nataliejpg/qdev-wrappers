import numpy as np
from qdev_wrappers.fitting.fitter import Fitter
from qdev_wrappers.fitting.models import SimpleMinimum


def get_unpushed_cavity_frequency(cavity, guess, pwa, span=50e6, step=1e6, measurement=False):
    cavity.power(-10)
    fitter = Fitter(SimpleMinimum())
    if measurement:
        runid = do1d(cavity_drive_frequency, guess-span/2, guess+span/2, span/step+1, 0, pwa.alazar_channels.ch_0_m.data)
        fit = fitter.fit_by_id(runid, 'alazar_controller_ch_0_m_data', x='cavity_drive_frequency', save_fit=False)
        cav_freq = fit.get_result()['param_values']['location']
        
    else:
        setpoints = np.linspace(guess-span/2, guess+span/2, int(span/step+1))
        results = []
        for freq in setpoints:
            cavity_drive_frequency.set(freq)
            results.append(pwa.alazar_channels.ch_0_m.data.get())
        cav_freq = fitter.fit(np.array(results), x=setpoints)[0]['location']

    print(f"Unpushed cavity frequency: {cav_freq}")

    return cav_freq


def get_pushed_cavity_settings(cavity,
                               unpushed_freq,
                               pwa,
                               span=10e6, 
                               step=0.5e6, 
                               power_span=50, 
                               power_step=5):

    points = [(unpushed_freq, -10)]
    slope = [1000]
    power_start = 0-power_step

    freq_setpoints = np.linspace(unpushed_freq-span/2, unpushed_freq+span/2, int(span/step+1))
    power_setpoints = np.linspace(power_start, power_start-power_span, int(power_span/power_step+1))
    
    fitter = Fitter(SimpleMinimum())

    for power in power_setpoints:
        cavity.power(power)
        results = []
        for freq in freq_setpoints:
            cavity_drive_frequency.set(freq)
            results.append(pwa.alazar_channels.ch_0_m.data.get())
        cav_freq = fitter.fit(np.array(results), x=freq_setpoints)[0]['location']
            
        points.append((cav_freq, power))
            
        d_freq = points[-1][0] - points[-2][0]
        d_pow = power_step
        m = d_freq/d_pow
        slope.append(m)
        
        if np.abs(cav_freq-unpushed_freq) > 1e6 and m == 0:
            # ToDo: m==0 is a pretty strict criteria. Some slope should still be okay.
            break

    print(f"Pushed cavity frequency: {points[-2]}")
            
    return points[-2]

