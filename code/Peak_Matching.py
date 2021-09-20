import numpy as np
import pandas as pd
from bsqi import bsqi

from scipy import signal

def nan_helper(y):
    """
    Finds all np.nan in a numpy array
    :param y: A numpy array to search through
    :return: A list of booleans where Nan is True, A function that allows for later interpolation
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def calculate_ptt(ppg_peaks, ecg_peaks, fs=256, max_ptt=0.54, min_ptt=0.20, smoothing_length=300):
    """
    Calculates the Pulse Transition Time (PTT) from the ECG R-Peaks and PPG Systolic Peaks.
    :param ppg_peaks: A numpy array of PPG systolic peaks position when sampled at fs [Hz]
    :param ecg_peaks: A numpy array of ECG R-Peaks when sampled at fs [Hz]
    :param fs: Sample rate [Hz] of PPG and ECG peaks
    :param max_ptt: The maximum time [Seconds] between the R-Peak and it's PPG-Peak
    :param min_ptt: The minimum time [Seconds] between the R-Peak and it's PPG-Peak
    :param smoothing_length: Number of DataPoints to smooth signal.
    :return: A PTT duration at same fs [Hz] for each ECG R-Peak
    """
    midpoint = (max_ptt * fs + min_ptt * fs) / 2
    ptt = np.zeros_like(ecg_peaks).astype(np.float32)
    ptt[:] = midpoint

    for i in range(len(ecg_peaks) - 1):
        try:
            start = ecg_peaks[i]
            end = ecg_peaks[i + 1]
            times = ppg_peaks[(ppg_peaks > start) & (ppg_peaks < end)]
            if len(times) > 0:
                ptt[i] = times[0] - start
            else:
                ptt[i] = np.nan
        except:
            print("failed")
            if i > 0:
                ptt[i] = np.nan

    nans, x = nan_helper(ptt)
    ptt[nans] = np.interp(x(nans), x(~nans), ptt[~nans])

    ptt[(ptt > max_ptt * fs) | (ptt < min_ptt * fs)] = np.mean(ptt[(ptt < max_ptt * fs) & (ptt > min_ptt * fs)])
    ptt = signal.filtfilt(np.ones(smoothing_length) / smoothing_length, 1, ptt, padlen=smoothing_length)
    ptt = np.array(ptt[0:len(ecg_peaks)]).astype(int)

    return ptt


def calculate_delayed_ecg(ppg_peaks, ecg_peaks, fs=256):
    """
    Forecasts the expected position of PPG fiduciaries from the ECG-R-Peaks taking PTT into consideration
    :param ppg_peaks: A numpy array of PPG fiduciaries positions when sampled at fs [Hz]
    :param ecg_peaks: A numpy array of ECG R-Peaks when sampled at fs [Hz]
    :param fs:
    :return: Forecast of PPG fiduciaries positions.
    """
    return ecg_peaks + calculate_ptt(ppg_peaks, ecg_peaks, fs=fs)

def calculate_windowed_delayed_ppg_ecg_bsqi(ppg_peaks, ecg_peaks, len_ppg=None, fs=256, window=30, agw=0.15):
    """
    For each window of length window [Seconds]
    :param ppg_peaks: A numpy array of PPG fiduciaries positions when sampled at 'fs'
    :param ecg_peaks: A numpy array of ECG R-Peaks when sampled at 'fs'
    :param fs: Sample rate [Hz] of PPG and ECG peaks
    :param window: Window size [Seconds] in which to calculate results
    :param agw: Maximum time [Seconds] between expected and forecast PPG figuciary,
    :return: A Pandas Dataframe with peak matching F1 score for each window.
    """

    # Limit the peaks to the ECG reference.
    ppg_peaks = ppg_peaks[ppg_peaks < ecg_peaks[-1]]
    ppg_peaks = ppg_peaks[ppg_peaks > ecg_peaks[0]]

    # Delay the PPG signal using the PTT
    delayed_ecg_peaks = calculate_delayed_ecg(ppg_peaks, ecg_peaks)

    # Window the results
    window_fs = fs * window
    windows = np.arange(0, len_ppg, window_fs)
    window_stats = pd.DataFrame()

    for i in (range(windows.shape[0] - 1)):
        window_ppg_peaks = ppg_peaks[(ppg_peaks >= window_fs*i)*(ppg_peaks < window_fs*(i+1))]
        window_delayed_ecg_peaks = delayed_ecg_peaks[(delayed_ecg_peaks >= window_fs*i)*(delayed_ecg_peaks < window_fs*(i+1))]
        window_stats = window_stats.append({'Epoch': i, **bsqi(window_delayed_ecg_peaks, window_ppg_peaks, fs=fs, agw=agw, return_dict=True)}, ignore_index=True)

    return window_stats
