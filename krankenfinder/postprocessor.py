"""Apply post-process based on outputs from the ML model"""

import numpy as np
import pandas as pd
import functools


def apply_ja(labels: np.array) -> np.array:
    return np.zeros_like(labels)
