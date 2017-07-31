import os
import sys
import pytest

from features.ja_semantics import PASUtils, SemanticFeatures

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

import krankenfinder
from krankenfinder.features.rule_based import *


@pytest.fixture
def ja_dummy():
    parsed = {}
    return parsed


@pytest.fixture
def en_dummy():
    parsed = {}
    return parsed


def test_ja_1(ja_dummy):
    assert True


def test_en_1(en_dummy):
    assert True
