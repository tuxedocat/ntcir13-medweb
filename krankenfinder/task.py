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
logging.basicConfig(level=logging.INFO)
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

    logger.info('Data loaded: {}'.format(str(corpus_path.name)))
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

    logger.info('Finished preprocessing')
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
    logger.info('Extracted word-surface features')
    return df


def add_semantic_feature(df: pd.DataFrame, verbose=False) -> pd.DataFrame:
    """Add semantic feature column to given dataframe

    .. note:: currently implemented only for Japanese tweets
    """
    fe = SemanticFeatures(verbose=verbose)

    logger.info('Started extracting semantic features')

    def get_semantic_featuredict(s: str) -> List[Tuple[str, int]]:
        _ = fe.pas_features(s)  # type: Dict[str, int]
        return list(_.items())

    df['f_semantic'] = df['raw'].apply(get_semantic_featuredict)
    logger.info('Extracted semantic features')
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


def feature_extraction(df: pd.DataFrame, surface=True, semantic=True, verbose=False) -> np.array:
    # use deepcopy instead of df.copy() because it doesn't recursively copy objects even when deep=True.
    df = deepcopy(df)
    if surface:
        df = add_surface_feature(df)

    if semantic:
        lang = _check_lang(df)
        if lang == 'ja':
            df = add_semantic_feature(df, verbose=verbose)

    df['features'] = np.empty((len(df['ID']), 0)).tolist()
    _merge_feature_columns(df)
    logger.info('Extracted features')

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


def _get_types(v: int) -> str:
    """Get string representations of error-types from integer-encoding"""
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
        # Encode TN/FP/FN/TP -> 0, -1, 1, 2 -> get error-name as string
        df[code_col] = (df[g] * (df[g] + df[p]) + df[p] * (df[g] - df[p])).astype(np.int64)
        df[c] = df[code_col].apply(_get_types)

    return df[_columns]


def get_interpretation_of_model(model: Model, transformer: DictVectorizer) -> pd.DataFrame:
    try:
        f_importances = model.feature_importances_
    except AttributeError:
        f_importances = model.best_estimator_.feature_importances_

    d = [(fname, weight) for fname, weight in zip(transformer.feature_names_, f_importances)]
    return pd.DataFrame(sorted(d, key=lambda x: x[1], reverse=True), columns=["feature", "importance"])


def end2end(task: str = 'ja',
            use_cache: bool = True,
            use_model_cache: bool = True,
            cache_dir: str = None,
            report_dir: str = None,
            features: List[str] = None,
            random_seed: int = None,
            verbose: bool = False):
    """Main API"""
    project_root = pathlib.Path(os.path.dirname(os.path.abspath(__file__)) + '/../')
    if task == 'ja':
        corpus = project_root / pathlib.Path('data/ja_train_20170501.xlsx')
    elif task == 'en':
        corpus = project_root / pathlib.Path('data/en_train_20170501.xlsx')
    elif task == 'ja-dev':
        corpus = project_root / pathlib.Path('data/ja_train_mini.xlsx')
    elif task == 'en-dev':
        corpus = project_root / pathlib.Path('data/en_train_mini.xlsx')
    else:
        raise ValueError

    _f_surface = True if 'suf-bow' in features else False
    _f_semantic = True if 'pas' in features else False

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
        train_df, test_df = train_test_split(df, random_seed=random_seed)
        pd.to_pickle(train_df, '_cache/_{}_train_df_cache.pkl.gz'.format(task))
        pd.to_pickle(test_df, '_cache/_{}_test_df_cache.pkl.gz'.format(task))
        Xtr = feature_extraction(train_df, surface=_f_surface, semantic=_f_semantic, verbose=verbose)
        Xtr = np.array(list(map(dict, Xtr)))
        ytr = get_labels(train_df)
        np.save('_cache/_{}_train_X_cache.npy'.format(task), Xtr)
        np.save('_cache/_{}_train_y_cache.npy'.format(task), ytr)

    vectorizer = DictVectorizer()

    # Transform to BoW representation
    Xtr = vectorizer.fit_transform(Xtr)

    if use_model_cache:
        with open('_cache/rf_model.pkl', 'rb') as f:
            rfcv_model = pickle.load(f)
    else:
        # Train model using randomized CV
        rfcv_model = define_model()
        logger.info('Training model...')
        rfcv_model.fit(Xtr, ytr)
        logger.info('Training model... Done.')
        with open('_cache/rf_model.pkl', 'wb') as f:
            pickle.dump(rfcv_model, f)

    # Evaluation on test split
    logger.info('Started evaluation.')

    if use_cache:
        Xts = np.load('_cache/_{}_test_X_cache.npy'.format(task))
        yts = np.load('_cache/_{}_test_y_cache.npy'.format(task))
    else:
        Xts = feature_extraction(test_df, surface=_f_surface, semantic=_f_semantic, verbose=verbose)
        Xts = np.array(list(map(dict, Xts)))
        yts = get_labels(test_df)
        np.save('_cache/_{}_test_X_cache.npy'.format(task), Xts)
        np.save('_cache/_{}_test_y_cache.npy'.format(task), yts)

    Xts = vectorizer.transform(Xts)
    report, predictions = evaluate_on_testset(rfcv_model, Xts, yts)
    print(report)

    if report_dir:
        _report_dir = pathlib.Path(report_dir)
    else:
        _report_dir = project_root / pathlib.Path('reports')

    if not _report_dir.exists():
        _report_dir.mkdir(exist_ok=True)

    report_df = error_analysis(test_df, predictions, rfcv_model)
    result_fn = _report_dir / pathlib.Path('result.log')
    analysis_fn = _report_dir / pathlib.Path('analysis')
    modelreport_fn = _report_dir / pathlib.Path('feature_importance')

    with result_fn.open('w') as log:
        log.write(report)
        log.write('\n\n')
        log.write('seed={}\n'.format(random_seed))
        log.write('features={}\n'.format(features))
        try:
            log.write('model={}\n'.format(rfcv_model.best_estimator_))
        except AttributeError:
            log.write('model={}\n'.format(rfcv_model))

    # Pandas doesn't work properly with Path obj., conversion to string is workaround.
    report_df.to_csv(str(analysis_fn.with_suffix('.csv')), index=False)
    report_df.to_excel(str(analysis_fn.with_suffix('.xlsx')), sheet_name='result', index=False)

    model_interpretation = get_interpretation_of_model(rfcv_model, vectorizer)
    model_interpretation.to_csv(str(modelreport_fn.with_suffix('.csv')))
    model_interpretation.to_excel(str(modelreport_fn.with_suffix('.xlsx')), sheet_name='features')

    logger.info('All done.')


@click.command()
@click.argument('task', type=str)
@click.option('--cache', is_flag=True, default=False)
@click.option('--model-cache', is_flag=True, default=False)
@click.option('--cache-dir', type=str, default='.')
@click.option('--report-dir', type=str, default='../reports')
@click.option('--feature', '-f', type=str, multiple=True, help='eg. -f pas -f suf-bow')
@click.option('--verbose', '-v', is_flag=True, default=False)
@click.option('--seed', type=int, help='random seed which is used for train/test split')
def cli(task, cache, model_cache, cache_dir, report_dir, feature, seed, verbose):
    print(feature)
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    end2end(task=task, use_cache=cache, use_model_cache=model_cache, report_dir=report_dir, features=feature,
            random_seed=seed, verbose=verbose)


if __name__ == '__main__':
    cli()
