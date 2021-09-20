import numpy as np
import pandas as pd
from scipy import signal, interpolate

def calculate_itervals_forwards(points):
    """
    Similar to numpy.gradient. Acts in forward direction. Adds a Nan at the end to maintain shape.
    :param points: A numpy array of sorted fiduciary positions
    :return: The beat to beat interval
    """
    return np.append((points[1:] - points[0:-1]), np.nan)

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def moving_average_filter(ibi, win_samples, percent):
    """
    Outlier detection and removal outliers. Adapted from Physiozoo filtrr moving average filter.
    https://github.com/physiozoo/mhrv/blob/2f67075e3db11120b92dd29c869a3ef4a527a2c2/%2Bmhrv/%2Brri/filtrr.m
    :param ibi: A numpy array of Inter Beat Intervals
    :param win_samples: The number consecutive IBIs to include in the moving average filter
    :param percent: The percentage above/below the average to use for filtering
    :return: Filtered ibi
    """
    b_fir = 1 / (2 * win_samples) * np.append(np.ones(win_samples), np.append(0, np.ones(win_samples)))
    points_moving_average = signal.filtfilt(b_fir, 1, ibi)
    points_filtered = ibi.copy()
    points_filtered[
        ~((ibi < (1 + percent / 100) * points_moving_average) & (ibi > (1 - percent / 100) * points_moving_average))] = np.nan
    nans, x = nan_helper(points_filtered)
    points_filtered[nans] = np.interp(x(nans), x(~nans), points_filtered[~nans])
    return points_filtered

def find_closest_smaller_value(find_value, list_of_values):
    """
    Returns the closest value from a list of values that is smaller than find_value
    :param find_value: The value we are searching for
    :param list_of_values: The list of values
    :return: The index of the closes value in the list. Returns -1 if not found.
    """
    for i in reversed(range(len(list_of_values) - 1)):
        if (list_of_values[i] < find_value):
            return i
    return -1

def find_closest_bigger_value(value, list_of_values):
    """
        Returns the closest value from a list of values that is bigger than find_value
        :param find_value: The value we are searching for
        :param list_of_values: The list of values
        :return: The index of the closes value in the list. Returns -1 if not found.
        """
    for i in range(len(list_of_values) - 1):
        if (list_of_values[i] > value):
            return i
    return -1

def calculate_windowed_IHR_IPR_agreement(ppg_peaks, ecg_peaks, fs=256, window=30, ptt=0.45, max_HR_detla=5):
    """
    :param ppg_peaks: A numpy array of PPG fiduciaries positions when sampled at 'fs'
    :param ecg_peaks: A numpy array of ECG R-Peaks when sampled at 'fs'
    :param fs: Sample rate [Hz] of PPG and ECG peaks
    :return: A Pandas Dataframe with peak matching F1 score for each window.
    :param window: Window size [Seconds]
    :param ptt: Approximate Pulse Transition Time [Seconds]
    :param max_HR_detla: Size of the IHR window [BPM]
    :return:
    """

    # 1) Shift the ECG peaks by approximate ppt
    ecg_peaks = ecg_peaks + int(ptt * fs)

    # 2) Limit the signal ranges to one another.
    start_arg, end_arg = 0, ppg_peaks[-1]
    if ppg_peaks[-1] > ecg_peaks[-1]:
        end_arg = find_closest_smaller_value(ecg_peaks[-1], ppg_peaks) + 1
    if ppg_peaks[0] < ecg_peaks[0]:
        start_arg = find_closest_bigger_value(ecg_peaks[0], ppg_peaks)

    ppg_peaks = ppg_peaks[start_arg:end_arg]

    # 3) Calculate the RR interval and filter out really bad points. Convert to HR estimate
    RR = calculate_itervals_forwards(ecg_peaks) / fs
    RR_filt = moving_average_filter(RR, win_samples=10,
                                    percent=50)  # Moving average window of 10 beats. #Filter @ 50% from moving average
    HR_RR = 60 / (RR_filt)

    # 3) Calculate the PP intervals for the patient. Convert to HR estimate
    PP = calculate_itervals_forwards(ppg_peaks) / fs
    HR_PP = 60 / PP

    # 4) Build the HR band and continuous IHR and IPR functions
    HR_RR_continous = interpolate.interp1d(ecg_peaks, HR_RR)
    HR_PP_continous = interpolate.interp1d(ppg_peaks, HR_PP)

    # 5) Resample all the HR's to 2Hz
    resample_2Hz = np.arange(ppg_peaks[0], ppg_peaks[-1], fs / 2)
    HR_RR = HR_RR_continous(resample_2Hz)
    HR_PP = HR_PP_continous(resample_2Hz)

    # 6) Calculate the agreement inside windows
    fs_2hz = 2
    window_2hz = window*fs_2hz
    len_ppg_in_s = ppg_peaks[-1]/fs
    len_ppg_at_2hz = len_ppg_in_s*fs_2hz
    windows = np.arange(0, len_ppg_at_2hz, window_2hz)
    window_stats = pd.DataFrame()

    for i in (range(windows.shape[0] - 1)):
        window_HR_RR = HR_RR[i*window_2hz:(i+1)*window_2hz]
        window_HR_PP = HR_PP[i*window_2hz:(i+1)*window_2hz]
        agreement_1 = np.sum(((window_HR_PP < window_HR_RR+1) & (window_HR_PP >= window_HR_RR-1)) | np.isnan(window_HR_RR)) / len(window_HR_RR)
        agreement_2 = np.sum(((window_HR_PP < window_HR_RR+2) & (window_HR_PP >= window_HR_RR-2)) | np.isnan(window_HR_RR)) / len(window_HR_RR)
        agreement_3 = np.sum(((window_HR_PP < window_HR_RR+3) & (window_HR_PP >= window_HR_RR-3)) | np.isnan(window_HR_RR)) / len(window_HR_RR)
        agreement_4 = np.sum(((window_HR_PP < window_HR_RR+4) & (window_HR_PP >= window_HR_RR-4)) | np.isnan(window_HR_RR)) / len(window_HR_RR)
        agreement_5 = np.sum(((window_HR_PP < window_HR_RR+5) & (window_HR_PP >= window_HR_RR-5)) | np.isnan(window_HR_RR)) / len(window_HR_RR)

        window_stats = window_stats.append({'Epoch': i,
                             'Agreement 1BPM': agreement_1,
                             'Agreement 2BPM': agreement_2,
                             'Agreement 3BPM': agreement_3,
                             'Agreement 4BPM': agreement_4,
                             'Agreement 5BPM': agreement_5}, ignore_index=True)
    return window_stats
