import os
import sys
import pytest

from features.ja_semantics import PASUtils, SemanticFeatures

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

import krankenfinder
from krankenfinder.features.en_semantics import *


def test_paslist():
    pass


@pytest.fixture
def predicate_dummy():
    predicate_fdic = {}
    return predicate_fdic


@pytest.fixture
def argument_dummy():
    argument_fdic = {}
    return argument_fdic


def test_get_tense(predicate_dummy):
    result = ''  # get_tense(predicate_dummy)
    assert result == 'past'


def test_get_ne(argument_dummy):
    result = ''  # get_ne(argument_dummy)
    assert result == 'person'


def test_get_featuredict():
    pass
