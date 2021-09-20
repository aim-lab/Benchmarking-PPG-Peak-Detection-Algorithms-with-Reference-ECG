from scipy.spatial import cKDTree
import numpy as np

def bsqi(refqrs, testqrs, agw=0.05, fs=200, return_dict=False):

    """
    This function is based on the following paper:
        Li, Qiao, Roger G. Mark, and Gari D. Clifford.
        "Robust heart rate estimation from multiple asynchronous noisy sources
        using signal quality indices and a Kalman filter."
        Physiological measurement 29.1 (2007): 15.

    The implementation itself is based on:
        Behar, J., Oster, J., Li, Q., & Clifford, G. D. (2013).
        ECG signal quality during arrhythmia and its application to false alarm reduction.
        IEEE transactions on biomedical engineering, 60(6), 1660-1666.

    :param refqrs:                  Annotation of the reference peak detector (Indices of the peaks).
    :param testqrs:                 Annotation of the test peak detector (Indices of the peaks).
    :param agw:                     Agreement window size (in seconds)
    :param fs:                      Sampling frquency [Hz]
    :param return_type:             If dict, returns a dictionary of the the metrics. Else returns F1
    :returns F1 or metrics-dict:    The 'bsqi' score, between 0 and 1.
    """

    agw *= fs
    if len(refqrs) > 0 and len(testqrs) > 0:
        NB_REF = len(refqrs)
        NB_TEST = len(testqrs)

        tree = cKDTree(refqrs.reshape(-1, 1))
        Dist, IndMatch = tree.query(testqrs.reshape(-1, 1))
        IndMatchInWindow = IndMatch[Dist < agw]
        NB_MATCH_UNIQUE = len(np.unique(IndMatchInWindow))
        TP = NB_MATCH_UNIQUE
        FN = NB_REF-TP
        FP = NB_TEST-TP
        Se  = TP / (TP+FN)
        PPV = TP / (FP+TP)
        if (Se+PPV) > 0:
            F1 = 2 * Se * PPV / (Se+PPV)
            _, ind_plop = np.unique(IndMatchInWindow, return_index=True)
            Dist_thres = np.where(Dist < agw)[0]
            meanDist = np.mean(Dist[Dist_thres[ind_plop]]) / fs
        else:
            if return_dict:
                return {'TP': TP, 'FN': FN, 'FP':FP, 'Se': 0, 'PPV': 0, 'F1':0}
            else:
                return 0
    else:
        F1 = 0
        if return_dict:
            return {'TP': 0, 'FN': 0, 'FP': 0, 'Se': 0, 'PPV': 0, 'F1': 0}
        else:
            return 0

    if return_dict:
        return {'TP': TP, 'FN': FN, 'FP':FP, 'Se': Se, 'PPV': PPV, 'F1':F1}
    else:
        return F1
