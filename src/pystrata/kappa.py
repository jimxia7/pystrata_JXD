import pystrata
import pykooh
import numpy as np
import pandas as pd
from scipy.stats import linregress


def calculate_kappa(data, start_freq, end_freq):
    if isinstance(data,pystrata.motion.TimeSeriesMotion):

        
        fourier_amps = abs(data.fourier_amps)
        freqs = data.freqs

        smoothed_amps = pykooh.smooth(freqs,freqs,fourier_amps,30)

        freq_min = start_freq
        freq_max = end_freq

        mask = (freqs >= freq_min) & (freqs <= freq_max)

        if np.any(mask):
            freqs_masked = freqs[mask]
            log_amps = np.log(smoothed_amps[mask])

            slope, intercept, r_value, p_value, std_err = linregress(freqs_masked, log_amps)

            kappa = -slope / np.pi

            return kappa, freqs_masked, np.exp(slope*freqs_masked+intercept)
        
    elif isinstance(data,pystrata.output.FourierAmplitudeSpectrumOutput):

        df = data.to_dataframe()
        freqs = df.index.to_numpy()
        freq_min = start_freq
        freq_max = end_freq
        mask = (freqs >= freq_min) & (freqs <= freq_max)
        
        kappa_df = pd.DataFrame()
        if np.any(mask):
            freqs_masked = freqs[mask]
            fitted_lines_df = pd.DataFrame(index=freqs[mask])

            for col in df.columns:
                fourier_amps = df[col].to_numpy()

                smoothed_amps = pykooh.smooth(freqs,freqs,fourier_amps,30)

                log_amps = np.log(smoothed_amps[mask])

                slope, intercept, r_value, p_value, std_err = linregress(freqs_masked, log_amps)

                kappa = -slope / np.pi

                fitted_line = np.exp(slope*freqs_masked+intercept)
                fitted_lines_df[col] = fitted_line
                kappa_df.loc[col,'kappa'] = kappa

                return kappa_df, fitted_lines_df