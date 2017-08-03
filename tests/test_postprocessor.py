import os
import sys
import pytest
import numpy as np
import pandas as pd
from typing import *

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

import krankenfinder
from krankenfinder import postprocessor
from krankenfinder.task import LABELCOLS


@pytest.fixture
def dummy_results() -> np.array:
    """Dummy classification results"""
    arr = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],  # target
                    [0., 0., 0., 1., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 1.],
                    [0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0.]])  # kep this alone
    return arr


def test_force_fever_label(dummy_results):
    expected = np.array([[1., 0., 0., 0., 0., 1., 0., 0.],  # target
                         [0., 0., 0., 1., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 1., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 1.],
                         [0., 0., 0., 0., 1., 0., 0., 0.],
                         [0., 1., 0., 0., 0., 0., 0., 0.],
                         [0., 1., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 1., 0., 0.]])  # kep this alone
    result = postprocessor.flu(dummy_results)
    np.testing.assert_array_equal(result, expected)


def test_pp(dummy_results):
    expected = np.array([[1., 0., 0., 0., 0., 1., 0., 0.],  # target
                         [0., 0., 0., 1., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 1., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 1.],
                         [0., 0., 0., 0., 1., 0., 0., 0.],
                         [0., 1., 0., 0., 0., 0., 0., 0.],
                         [0., 1., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 1., 0., 0.]])  # kep this alone
    result = postprocessor.apply_pp(dummy_results)
    np.testing.assert_array_equal(result, expected)
