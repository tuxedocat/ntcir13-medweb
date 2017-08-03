"""Apply post-process based on outputs from the ML model"""

import numpy as np
import pandas as pd
import functools

COL_FLU = 0
COL_FEVER = 5


def apply_pp(labels: np.array) -> np.array:
    _labels = labels.copy()
    funcs = [flu]
    for f in funcs:
        _labels = f(_labels)
    return _labels


def flu(labels: np.array) -> np.array:
    """If label 'Influenza' is positive, mark 'Fever' positive too"""
    labels[(labels[:, COL_FLU] == 1) & (labels[:, COL_FEVER] == 0), COL_FEVER] = 1.
    return labels
