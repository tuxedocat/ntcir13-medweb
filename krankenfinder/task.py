"""NTCIR13-MedWeb"""

from typing import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import os
import sys
import pathlib
import functools
from collections import Counter
from copy import deepcopy
import pickle
import click

import sklearn
from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import sklearn.utils

import pyknp
from natto import MeCab, MeCabNode
import spacy

from krankenfinder.features.ja_semantics import SemanticFeatures
from krankenfinder.utils.normalize import normalize_neologd

import logging
import logging.config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.captureWarnings(True)

try:
    NEOLOGD = os.environ['NEOLOGD']
except KeyError:
    NEOLOGD = '/usr/lib/mecab/dic/mecab-ipadic-neologd/'

LABELCOLS = ['Influenza', 'Diarrhea', 'Hayfever', 'Cough', 'Headache', 'Fever', 'Runnynose', 'Cold']
Model = Union[RandomForestClassifier, RandomizedSearchCV]


class ModelDefinition:
    def __init__(self, model: Model,
                 dv: DictVectorizer, comment: str = None) -> None:
        self.model = model
        self.dv = dv
        self.comments = comment if comment else ''


def load_dataset(corpus_path: pathlib.Path) -> Optional[pd.DataFrame]:
    """Load dataset from given xlsx or csv files, as dataframe

    :param corpus_path:
    :type corpus_path:
    :return:
    :rtype:
    """
    if str(corpus_path.suffix == '.xlsx'):
        if corpus_path.name.startswith('ja'):
            raw = pd.read_excel(str(corpus_path), sheetname='ja_train')
        elif corpus_path.name.startswith('en'):
            raw = pd.read_excel(str(corpus_path), sheetname='en_train')
        else:
            raise ValueError('Invalid format corpus file: {}'.format(str(corpus_path.absolute())))
    else:
        raise NotImplementedError('Only xlsx corpus is allowed.')

    logger.debug('Data loaded: {}'.format(str(corpus_path.name)))
    return raw


def _parser_func_mecab_detailed(parser: MeCab) -> Callable[[str], List[Tuple[str, str]]]:
    def parse_to_morphs(s: str) -> List[Tuple[str, str]]:
        return [tuple(l.split('\t')) for l in parser.parse(normalize_neologd(s)).split('\n')]

    return parse_to_morphs


def _parser_func_mecab(parser: MeCab) -> Callable[[str], List[str]]:
    def parse_to_surf(s: str) -> List[str]:
        return [node.surface for node in parser.parse(normalize_neologd(s), as_nodes=True) if node.surface]

    return parse_to_surf


def _tokenizer_func(spacy_model: spacy.en.English) -> Callable[[str], List[str]]:
    def tokenizer_f(s: str) -> List[str]:
        return [w.lemma_ for w in spacy_model(s)]

    return tokenizer_f


def _binarize_pn(df: pd.DataFrame) -> pd.DataFrame:
    for c in LABELCOLS:
        # cast to float64 for later use with sklearn
        df[c] = df[c].apply(lambda s: 1 if s == 'p' else 0).astype(np.float64)
    return df


def _pp_ja(df: pd.DataFrame) -> pd.DataFrame:
    mecab = MeCab('-d {}'.format(NEOLOGD))
    parser = _parser_func_mecab(mecab)
    df['words'] = df['Tweet'].apply(parser)
    # df['raw'] = df['Tweet'].apply(normalize_neologd) # KNP fails to parse with some hankaku characters
    df['raw'] = df['Tweet'].copy()  # TODO: add some preprocess for KNP
    return df


def _pp_en(df: pd.DataFrame) -> pd.DataFrame:
    sp = spacy.load('en')
    tokenizer_ = _tokenizer_func(sp)
    df['words'] = df['Tweet'].apply(tokenizer_)
    # TODO: Implement normalization for English tweets if needed.
    df['raw'] = df['Tweet'].copy()
    return df


def _check_lang(df: pd.DataFrame) -> str:
    return df['ID'].iloc[0][-2:]


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Perform preprocessing for given dataframe,
    including, binarizing p/n labels to 1/0, normalization (JP), adding column of tokenized sentence.

    :param df:
    :type df:
    :return:
    :rtype:
    """
    df = _binarize_pn(df)

    if _check_lang(df) == 'ja':
        df = _pp_ja(df)
    elif _check_lang(df) == 'en':
        df = _pp_en(df)
    else:
        raise NotImplementedError('Not implemented for this language')

    logger.debug('Finished preprocessing')
    return df


def train_test_split(df: pd.DataFrame, ratio: float = 0.8, random_seed: Optional[int] = None) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform train/test split

    :param random_seed:
    :type random_seed:
    :param ratio:
    :type ratio:
    :param df:
    :type df:
    :return:
    :rtype:
    """
    split_id = int(len(df) * ratio)
    _df = sklearn.utils.shuffle(df.copy(), random_state=random_seed)
    train = _df.iloc[:split_id]
    test = _df.iloc[split_id:]
    return train, test


def add_surface_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add surface-bag-of-words column to given dataframe"""

    def get_counts_of_words(l: List[str]) -> List[Tuple[str, int]]:
        return list(Counter(l).items())

    df['f_surface'] = df['words'].apply(get_counts_of_words)
    logger.debug('Extracted word-surface features')
    return df


def add_semantic_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add semantic feature column to given dataframe

    .. note:: currently implemented only for Japanese tweets
    """
    fe = SemanticFeatures()

    logger.debug('Started extracting semantic features')

    def get_semantic_featuredict(s: str) -> List[Tuple[str, int]]:
        _ = fe.pas_features(s)  # type: Dict[str, int]
        return list(_.items())

    df['f_semantic'] = df['raw'].apply(get_semantic_featuredict)
    logger.debug('Extracted semantic features')
    return df


def _merge_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Prerequisite: df must have 'features' column

    WARNING: this function will modify dataframe inplace, better to copy df beforehand."""
    feature_columns = sorted([cname for cname in df.columns if cname.startswith('f_')], reverse=True)
    if not feature_columns:
        return df
    elif len(feature_columns) == 1:
        cn = feature_columns.pop()
        df['features'] = pd.Series(df['features'] + df[cn])
        df.drop(cn, 1, inplace=True)
        return df
    else:
        cn1 = feature_columns[0]
        cn2 = feature_columns[1]
        df['features'] = pd.Series(df['features'] + df[cn2] + df[cn1])
        df.drop([cn1, cn2], 1, inplace=True)
        _merge_feature_columns(df)


def feature_extraction(df: pd.DataFrame, surface=True, semantic=True) -> np.array:
    # use deepcopy instead of df.copy() because it doesn't recursively copy objects even when deep=True.
    df = deepcopy(df)
    if surface:
        df = add_surface_feature(df)

    if semantic:
        lang = _check_lang(df)
        if lang == 'ja':
            df = add_semantic_feature(df)

    df['features'] = np.empty((len(df['ID']), 0)).tolist()
    _merge_feature_columns(df)
    logger.debug('Extracted features')

    return df['features'].values


def get_labels(df: pd.DataFrame) -> np.array:
    return df[LABELCOLS].values


def define_model() -> Model:
    # TODO: needs refinements
    rf = RandomForestClassifier(random_state=None)

    search_space = dict(
        n_estimators=[10, 16, 20, 24, 28, 32, 36, 40],
        criterion=['gini', 'entropy'],
        max_features=['auto', 'log2', None],
        max_depth=list(range(8, 25, 2))
    )

    rfcv = model_selection.RandomizedSearchCV(estimator=rf,
                                              param_distributions=search_space,
                                              n_iter=100,
                                              n_jobs=8,
                                              cv=5
                                              )
    return rfcv


def evaluate_on_testset(model: Model, X_test, y_test) -> Tuple[str, np.array]:
    predictions = model.predict(X_test)
    report = metrics.classification_report(y_test, predictions, target_names=LABELCOLS)
    return report, predictions


# def _annotate_errors(df_info: pd.DataFrame, prefix_gold: str = 'gold_', prefix_pred: str = 'pred_') -> pd.DataFrame:
#     """Annotate result types (TP/FP/FN/TN)"""
#     for c in LABELCOLS:
#         df_info[c] =

def _get_types(v: int) -> str:
    if v == 0:
        return 'TN'
    elif v == -1:
        return 'FP'
    elif v == 1:
        return 'FN'
    elif v == 2:
        return 'TP'
    else:
        return 'NA'


def error_analysis(df_test: pd.DataFrame, predictions: np.array, model: Model) -> pd.DataFrame:
    """Get detailed information for analysing error cases."""
    # Prefix for columns
    P_G = 'gold_'
    P_P = 'pred_'
    P_C = 'code_'

    _columns = ['ID', 'Tweet'] + LABELCOLS
    rename_dic = {org: P_G + org for org in LABELCOLS}

    df = df_test.loc[:, _columns].copy()
    df = df.rename(columns=rename_dic)

    # Add prediction columns to df
    for c in range(predictions.shape[1]):
        col = pd.Series(predictions[:, c], index=df_test['ID'], dtype=np.float64)
        name = P_P + LABELCOLS[c]
        df[name] = col.values

    # Add metadata
    for c in LABELCOLS:
        g = P_G + c
        p = P_P + c
        code_col = P_C + c
        # Encode TN/FP/FN/TP -> 0, -1, 1, 2
        df[code_col] = (df[g] * (df[g] + df[p]) + df[p] * (df[g] - df[p])).astype(np.int64)
        df[c] = df[code_col].apply(_get_types)

    return df[_columns]


def end2end(task: str = 'ja', use_cache: bool = True, use_model_cache: bool = True):
    """Main API"""
    project_root = os.path.dirname(os.path.abspath(__file__)) + '/../'
    if task == 'ja':
        corpus = pathlib.Path(project_root) / pathlib.Path('data/ja_train_20170501.xlsx')
    elif task == 'en':
        corpus = pathlib.Path(project_root) / pathlib.Path('data/en_train_20170501.xlsx')
    elif task == 'ja-dev':
        corpus = pathlib.Path(project_root) / pathlib.Path('data/ja_train_mini.xlsx')
    elif task == 'en-dev':
        corpus = pathlib.Path(project_root) / pathlib.Path('data/en_train_mini.xlsx')
    else:
        raise ValueError

    if use_cache and os.path.exists('_cache'):
        train_df = pd.read_pickle('_cache/_{}_train_df_cache.pkl.gz'.format(task))
        test_df = pd.read_pickle('_cache/_{}_test_df_cache.pkl.gz'.format(task))
        Xtr = np.load('_cache/_{}_train_X_cache.npy'.format(task))
        ytr = np.load('_cache/_{}_train_y_cache.npy'.format(task))
    else:
        if not os.path.exists('cache'):
            os.mkdir('cache')

        df = load_dataset(corpus)
        df = preprocess_df(df)
        train_df, test_df = train_test_split(df, random_seed=12345)
        pd.to_pickle(train_df, '_cache/_{}_train_df_cache.pkl.gz'.format(task))
        pd.to_pickle(test_df, '_cache/_{}_test_df_cache.pkl.gz'.format(task))
        Xtr = feature_extraction(train_df)
        Xtr = np.array(list(map(dict, Xtr)))
        ytr = get_labels(train_df)
        np.save('_cache/_{}_train_X_cache.npy'.format(task), Xtr)
        np.save('_cache/_{}_train_y_cache.npy'.format(task), ytr)

    vectrizor = DictVectorizer()

    # Transform to BoW representation
    Xtr = vectrizor.fit_transform(Xtr)

    if use_model_cache:
        with open('_cache/rf_model.pkl', 'rb') as f:
            rfcv_model = pickle.load(f)
    else:
        # Train model using randomized CV
        rfcv_model = define_model()
        logger.debug('Training model...')
        rfcv_model.fit(Xtr, ytr)
        logger.debug('Training model... Done.')
        with open('_cache/rf_model.pkl', 'wb') as f:
            pickle.dump(rfcv_model, f)

    # Evaluation on test split
    logger.debug('Started evaluation.')

    if use_cache:
        Xts = np.load('_cache/_{}_test_X_cache.npy'.format(task))
        yts = np.load('_cache/_{}_test_y_cache.npy'.format(task))
    else:
        Xts = feature_extraction(test_df)
        Xts = np.array(list(map(dict, Xts)))
        yts = get_labels(test_df)
        np.save('_cache/_{}_test_X_cache.npy'.format(task), Xts)
        np.save('_cache/_{}_test_y_cache.npy'.format(task), yts)

    Xts = vectrizor.transform(Xts)
    report, predictions = evaluate_on_testset(rfcv_model, Xts, yts)
    print(report)

    report_df = error_analysis(test_df, predictions, rfcv_model)
    report_df.to_csv('analysis.csv', index=False)
    report_df.to_excel('analysis.xlsx', sheet_name='result', index=False)

    logger.debug('All done.')


@click.group()
def cli():
    pass


@cli.command(help='ja: train and eval')
@click.option('--cache', is_flag=True, default=False)
@click.option('--model-cache', is_flag=True, default=False)
def ja(cache, model_cache):
    end2end('ja', cache, model_cache)


@cli.command(help='en: train and eval')
@click.option('--cache', is_flag=True, default=False)
@click.option('--model-cache', is_flag=True, default=False)
def en(cache, model_cache):
    end2end('en', cache, model_cache)


if __name__ == '__main__':
    cli()