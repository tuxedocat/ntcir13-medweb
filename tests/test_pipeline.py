import os
import sys

import pathlib
import pytest
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

ROOT = os.path.dirname(os.path.abspath(__file__)) + '/../'
sys.path.insert(0, ROOT)

import krankenfinder
from krankenfinder import task


def test_merge_feature_cols():
    df1 = pd.DataFrame({'f_sem': [['A0'], ['A1'], ['A2'], ['A3']],
                        'f_suf': [['B0'], ['B1'], ['B2'], ['B3']],
                        'f_pos': [['C0'], ['C1'], ['C2'], ['C3']],
                        'f_ner': [['D0'], ['D1'], ['D2'], ['D3']]})
    df1['features'] = np.empty((len(df1.f_sem), 0)).tolist()
    df = task._merge_feature_columns(df1)
    print(df1)
    print(df)


@pytest.fixture
def load_dataframe():
    f = pathlib.Path(ROOT) / pathlib.Path('data/ja_train_mini.xlsx')
    df = task.load_dataset(f)
    return df


def test_feature_extractor(load_dataframe):
    df = load_dataframe
    df = task.preprocess_df(df)
    X = task.feature_extraction(df)
    assert X is not None
    print(X)


def test_evaluation(load_dataframe):
    df = load_dataframe
    df = task.preprocess_df(df)

    train_df, test_df = task.train_test_split(df, random_seed=12345)
    Xtr = task.feature_extraction(train_df)
    Xtr = np.array(list(map(dict, Xtr)))
    ytr = task.get_labels(train_df)
    vectrizor = DictVectorizer()
    Xtr = vectrizor.fit_transform(Xtr)
    rfcv_model = task.define_model()
    rfcv_model.fit(Xtr, ytr)

    Xts = task.feature_extraction(test_df)
    Xts = np.array(list(map(dict, Xts)))
    yts = task.get_labels(test_df)
    Xts = vectrizor.transform(Xts)
    report, predictions = task.evaluate_on_testset(rfcv_model, Xts, yts)
    print(report)

    report_df = task.error_analysis(test_df, predictions, rfcv_model)
    print(report_df)
