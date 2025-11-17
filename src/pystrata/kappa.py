import pystrata
import pykooh
import numpy as np

from scipy.stats import linregress


def calculate_kappa(motion, start_freq, end_freq):
    if isinstance(motion,pystrata.motion.TimeSeriesMotion):
        raise ValueError("Input motion is not a TimeSeriesMotion class.")
    
    fourier_amps = abs(motion.fourier_amps)
    freqs = motion.freqs

    smoothed_amps = pykooh.smooth(freqs,freqs,fourier_amps,30)

    freq_min = start_freq
    freq_max = end_freq

    mask = (freqs >= freq_min) & (freqs <= freq_max)

    if np.any(mask):
        freqs = freqs[mask]
        log_amps = np.log(smoothed_amps[mask])

        slope, intercept, r_value, p_value, std_err = linregress(freqs, log_amps)

        kappa = -slope / np.pi

        return kappa, freqs[mask], np.exp(slope*freqs[mask]+intercept)