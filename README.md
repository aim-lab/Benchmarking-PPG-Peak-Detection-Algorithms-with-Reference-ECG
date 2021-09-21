# Benchmarking PPG Peak Detection Algorithms using ECG as a Reference

- See /paper for the publication
- See /code for the Beat Matching and IHR_IPR Accuracy Algorithms
- Requirements: Python3, Numpy, Pandas, Scipy

## Abstract
**Introduction:** Photoplethysmography (PPG) is fast becoming the signal of choice for the widespread monitoring of sleep metrics obtained by wearable devices. Robust peak detection is critical for the extraction of meaningful features from the PPG  waveform. There is however no consensus on what PPG peak detection algorithms perform best on nocturnal continuous PPG recordings. We introduce two methods to benchmark the performance of PPG peak detectors. 

**Methods:** We make use of data where nocturnal PPG and electrocardiogram (ECG) are measured synchronously. Within this setting, the ECG, a signal for which there are established R-peak detectors, is used as reference. The first method for benchmarking, denoted "Peak Matching", consists of forecasting the expected position of the PPG peaks using the ECG R-peaks as reference. The second technique, denoted "IHR-IPR Accuracy", compares the instantaneous pulse rate (IPR) extracted from the PPG with the instantaneous heart rate (IHR) extracted from the ECG. For benchmarking, we used the MESA dataset consisting of 2,055 overnight polysomnography recordings with a combined length of over 16,300 hours.Four open PPG peak detectors were benchmarked. 

**Results:** The "Pulses" detector performed best with a Peak Matching F1-score of 0.94 and an IHR-IPR Accuracy of 89.6\%. 

**Discussion and conclusion:** We introduced two new methods for benchmarking PPG peak detectors. Among the four detectors evaluated, "Pulses" performed best. Benchmarking of further PPG detectors and on other data source (e.g. daytime recordings, recordings from patients with arrhythmia) is needed. 
