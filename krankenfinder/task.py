"""NTCIR13-MedWeb"""

from typing import *
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn import metrics
from sklearn.externals import joblib
import sklearn.utils

import pyknp
from natto import MeCab, MeCabNode
import spacy

from features.basics import ngram_features
from krankenfinder.features.ja_semantics import SemanticFeatures
from krankenfinder.features import bow_juman
from krankenfinder.utils.normalize import normalize_neologd
from krankenfinder import postprocessor

import logging
import logging.config

# This logger will be overridden by logger defined in __main__
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
logger.propagate = False

try:
    MECAB_OPTS = "-F %m,%f[0],%f[1],%f[6] -d {}".format(os.environ['NEOLOGD'])
except KeyError:
    MECAB_OPTS = "-F %m,%f[0],%f[1],%f[6] -d /usr/lib/mecab/dic/mecab-ipadic-neologd/"

LABELCOLS = ['Influenza', 'Diarrhea', 'Hayfever', 'Cough', 'Headache', 'Fever', 'Runnynose', 'Cold']
Model = Union[RandomForestClassifier, RandomizedSearchCV]


def load_dataset(corpus_path: Path) -> Optional[pd.DataFrame]:
    """Load dataset from given xlsx or csv files, as dataframe

    Parameters
    ----------
    corpus_path : pathlib.Path
        dataset file's path

    Returns
    -------
    loaded dataframe
    """

    def get_sheetname(p: Path) -> str:
        if p.name.startswith('ja_train'):
            return 'ja_train'
        elif p.name.startswith('ja_test'):
            return 'ja_test'
        elif p.name.startswith('en_train'):
            return 'en_train'
        elif p.name.startswith('en_test'):
            return 'en_test'
        else:
            return ''

    if str(corpus_path.suffix == '.xlsx'):
        sheetname = get_sheetname(corpus_path)
        raw = pd.read_excel(str(corpus_path), sheetname=sheetname)
    else:
        raise NotImplementedError('Only xlsx corpus is allowed.')

    logger.info('Data loaded: {}'.format(str(corpus_path.name)))
    return raw


def _parser_func_mecab_detailed(parser: MeCab) -> Callable[[str], List[Tuple[str, str]]]:
    def parse_to_morphs(s: str) -> List[Tuple[str, str]]:
        return [tuple(l.split('\t')) for l in parser.parse(normalize_neologd(s)).split('\n')]

    return parse_to_morphs


def _get_lemma(node: MeCabNode) -> str:
    """Assuming format "<surface>,<pos>,<posd>,<lemma>" """
    try:
        return node.feature.split(',')[3]
    except IndexError:
        logger.error(node.feature)
        return ''


def _pos_included(node: MeCabNode) -> bool:
    exclude_pos = {'記号'}
    exclude_posd = {'格助詞', '接続助詞'}
    suf, pos, posd, lemma = node.feature.split(',')
    if pos in exclude_pos:
        return False
    if posd in exclude_posd:
        return False
    return True


def _parser_func_mecab(parser: MeCab) -> Callable[[str], List[str]]:
    def parse_to_surf(s: str) -> List[str]:
        return [_get_lemma(node) for node in parser.parse(normalize_neologd(s), as_nodes=True)
                if node.is_nor() and _pos_included(node)]

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


def _pp_ja(df: pd.DataFrame, userdict: str = None, jumanpp: bool = False) -> pd.DataFrame:
    if not jumanpp:  # use mecab-neologd by default
        if userdict:
            mecab = MeCab('{} -u {}'.format(MECAB_OPTS, userdict))
            logger.info('Using customdict {}'.format(userdict))
        else:
            mecab = MeCab('{}'.format(MECAB_OPTS))
        parser = _parser_func_mecab(mecab)
        # df['raw'] = df['Tweet'].apply(normalize_neologd) # KNP fails to parse with some hankaku characters
    else:  # use jumanpp
        parser = bow_juman.parser_func_jumanpp(lemmatize=True)
        logger.info('Using Juman++ instead of MeCab')

    df['words'] = df['Tweet'].apply(parser)
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


def preprocess_df(df: pd.DataFrame, userdict: str = None, jumanpp: bool = False) -> pd.DataFrame:
    """Perform preprocessing for given dataframe,
    including, binarizing p/n labels to 1/0, normalization (JP), adding column of tokenized sentence.
    """
    df = _binarize_pn(df)

    if _check_lang(df) == 'ja':
        df = _pp_ja(df, userdict=userdict, jumanpp=jumanpp)
    elif _check_lang(df) == 'en':
        df = _pp_en(df)
    else:
        raise NotImplementedError('Not implemented for this language')

    logger.info('Finished preprocessing')
    return df


def train_test_split(df: pd.DataFrame, ratio: float = 0.8, random_seed: Optional[int] = None) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform train/test split """
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


def add_ngram_feature(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    ngram_func = functools.partial(ngram_features, n=n, padding=True)
    df['f_{}gram'.format(n)] = df['words'].apply(ngram_func)
    logger.info('Extracted {}gram features'.format(n))
    return df


def add_semantic_feature(df: pd.DataFrame, verbose=False) -> pd.DataFrame:
    """Add semantic feature column to given dataframe

    .. note:: currently implemented only for Japanese tweets
    """
    fe = SemanticFeatures(verbose=verbose)
    logger = logging.getLogger(__name__)
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

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


def feature_extraction(df: pd.DataFrame,
                       surface=True,
                       ngram_n: Union[Tuple[int], Tuple[int, int], None] = (3,),
                       semantic=True,
                       verbose=False) -> np.array:
    # use deepcopy instead of df.copy() because it doesn't recursively copy objects even when deep=True.
    df = deepcopy(df)
    if surface:
        df = add_surface_feature(df)
        if ngram_n:
            for n in ngram_n:
                df = add_ngram_feature(df, n=n)

    if semantic:
        lang = _check_lang(df)
        if lang == 'ja':
            df = add_semantic_feature(df, verbose=verbose)

    df['features'] = np.empty((len(df['ID']), 0)).tolist()
    _merge_feature_columns(df)

    return df['features'].values


def get_labels(df: pd.DataFrame) -> np.array:
    return df[LABELCOLS].values


def define_model(n_random_search: int = 100) -> Model:
    # TODO: needs refinements
    rf = RandomForestClassifier(random_state=None)
    # Or Extremely Randomized Trees, but currently no big difference in terms of performance.
    # rf = ExtraTreesClassifier(random_state=None)
    _n_estimators = list(range(8, 64, 4))
    _n_estimators += [100, 250, 500, 1000, 2000]
    _max_depth = list(range(8, 24, 2))
    _max_depth.append(None)

    search_space = dict(
        n_estimators=_n_estimators,
        criterion=['gini', 'entropy'],
        max_features=['auto', 'log2', None],
        max_depth=_max_depth
    )

    ncores = joblib.cpu_count() // 2  # ... be nice.
    rfcv = model_selection.RandomizedSearchCV(estimator=rf,
                                              param_distributions=search_space,
                                              n_iter=n_random_search,
                                              n_jobs=ncores,
                                              cv=5,
                                              verbose=1
                                              )
    return rfcv


def get_preds_and_probs(model: Model, X_test, use_postprocess: bool = False):
    predictions = model.predict(X_test)
    if use_postprocess:
        predictions = postprocessor.apply_pp(predictions)

    probabilities = model.predict_proba(X_test)
    return predictions, probabilities


def evaluate_on_testset(model: Model, X_test, y_test, use_postprocess: bool = False) -> Tuple[str, np.array, np.array]:
    predictions, probabilities = get_preds_and_probs(model, X_test, use_postprocess)
    report = metrics.classification_report(y_test, predictions, target_names=LABELCOLS)
    return report, predictions, probabilities


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


def to_submission_df(df_test: pd.DataFrame, predictions: np.array) -> pd.DataFrame:
    """Format outputs for submission"""
    _columns = ['ID', 'Tweet'] + LABELCOLS
    df = df_test.loc[:, _columns].copy()
    for i, colname in enumerate(LABELCOLS):
        preds = pd.Series(predictions[:, i], index=df['ID'], dtype=bool)
        df[colname] = preds.values
        df[colname] = df[colname].apply(lambda s: 'p' if s is True else 'n')
    return df[_columns]


def to_prob_df(df_test: pd.DataFrame, probabilities: np.array) -> pd.DataFrame:
    """TODO: rethink design"""
    _columns = ['ID', 'Tweet'] + LABELCOLS
    df = df_test.loc[:, _columns].copy()
    for i, colname in enumerate(LABELCOLS):
        probs = pd.Series(probabilities[i][:, 1], index=df['ID'], dtype=np.double)
        df[colname] = probs.values
    return df[_columns]


def get_interpretation_of_model(model: Model, transformer: DictVectorizer) -> pd.DataFrame:
    try:
        f_importances = model.feature_importances_
    except AttributeError:
        f_importances = model.best_estimator_.feature_importances_

    d = [(fname, weight) for fname, weight in zip(transformer.feature_names_, f_importances)]
    return pd.DataFrame(sorted(d, key=lambda x: x[1], reverse=True), columns=["feature", "importance"])


def get_fn(dirpath: Path, name: str, ext: str) -> str:
    p = str(dirpath / Path('{}'.format(name)).with_suffix(ext))
    return p


def end2end(task: str = 'ja',
            use_cache: bool = True,
            use_model_cache: bool = True,
            report_dir: str = None,
            features: List[str] = None,
            random_seed: int = None,
            verbose: bool = False,
            formal_run: bool = False,
            n_random_search: int = 100,
            use_jumanpp: bool = False,
            postprocess: bool = False,
            selftest: bool = False):
    """Main API"""

    # Setup logger
    global logger
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    logger.propagate = False

    if verbose:
        handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)

    logger.addHandler(handler)

    # Input dirs and files
    project_root = Path(os.path.dirname(os.path.abspath(__file__)) + '/../')
    if task == 'ja':
        corpus = project_root / Path('data/ja_train_20170705.xlsx')
    elif task == 'en':
        corpus = project_root / Path('data/en_train_20170705.xlsx')
    elif task == 'ja-dev':
        corpus = project_root / Path('data/ja_train_mini.xlsx')
    elif task == 'en-dev':
        corpus = project_root / Path('data/en_train_mini.xlsx')
    else:
        raise ValueError

    if formal_run and task == 'ja':
        test_corpus = project_root / Path('data/ja_test_20170724.xlsx')
    elif formal_run and task == 'en':
        test_corpus = project_root / Path('data/en_test_20170724.xlsx')
    else:
        test_corpus = None

    # Feature flags
    _f_surface = True if 'suf' in features else False
    _f_semantic = True if 'pas' in features else False
    _ns = []
    if 'bigram' in features:
        _ns.append(2)
    if 'trigram' in features:
        _ns.append(3)
    _ns = tuple(_ns)

    if 'userdict' in features:
        _userdict = str(project_root / Path('mecab-dict/cat.dic'))
    else:
        _userdict = None

    # Output dirs
    if report_dir:
        _report_dir = Path(report_dir)
    else:
        _report_dir = project_root / Path('reports')

    if not _report_dir.exists():
        _report_dir.mkdir(exist_ok=True)

    cache_dir = _report_dir / Path('_cache')
    if not cache_dir.exists():
        cache_dir.mkdir(exist_ok=True)

    # Preprocessing
    if use_cache and cache_dir.exists():
        train_df = pd.read_pickle(get_fn(cache_dir, task + '_train', '.pkl.gz'))
        test_df = pd.read_pickle(get_fn(cache_dir, task + '_test', '.pkl.gz'))
        Xtr = np.load(get_fn(cache_dir, task + '_Xtrain', '.npy'))
        ytr = np.load(get_fn(cache_dir, task + '_ytrain', '.npy'))
    else:
        df = load_dataset(corpus)
        df = preprocess_df(df, userdict=_userdict, jumanpp=use_jumanpp)
        if formal_run:
            train_df = df
            test_df = load_dataset(test_corpus)
            test_df = preprocess_df(test_df, userdict=_userdict, jumanpp=use_jumanpp)
        elif selftest:
            logger.info("---SELF TESTING---")
            train_df = deepcopy(df)
            test_df = deepcopy(df)
        else:
            train_df, test_df = train_test_split(df, random_seed=random_seed)

        pd.to_pickle(train_df, get_fn(cache_dir, task + '_train', '.pkl.gz'))
        pd.to_pickle(test_df, get_fn(cache_dir, task + '_test', '.pkl.gz'))
        Xtr = feature_extraction(train_df, surface=_f_surface, semantic=_f_semantic, ngram_n=_ns, verbose=verbose)
        Xtr = np.array(list(map(dict, Xtr)))
        ytr = get_labels(train_df)
        np.save(get_fn(cache_dir, task + '_Xtrain', '.npy'), Xtr)
        np.save(get_fn(cache_dir, task + '_ytrain', '.npy'), ytr)

    vectorizer = DictVectorizer()

    # Transform to BoW representation
    Xtr = vectorizer.fit_transform(Xtr)

    # Training the model
    if use_model_cache:
        with open(get_fn(cache_dir, 'model', '.pkl'), 'rb') as f:
            rfcv_model = pickle.load(f)
    else:
        # Train model using randomized CV
        rfcv_model = define_model(n_random_search=n_random_search)
        logger.info('Training model...')
        rfcv_model.fit(Xtr, ytr)
        logger.info('Training model... Done.')
        with open(get_fn(cache_dir, 'model', '.pkl'), 'wb') as f:
            pickle.dump(rfcv_model, f)

    # Evaluation on test split
    logger.info('Started evaluation.')

    if use_cache:
        Xts = np.load(get_fn(cache_dir, task + '_Xtest', '.npy'))
        yts = np.load(get_fn(cache_dir, task + '_ytest', '.npy'))
    else:
        Xts = feature_extraction(test_df, surface=_f_surface, semantic=_f_semantic, ngram_n=_ns, verbose=verbose)
        Xts = np.array(list(map(dict, Xts)))
        yts = get_labels(test_df)
        np.save(get_fn(cache_dir, task + '_Xtest', '.npy'), Xts)
        np.save(get_fn(cache_dir, task + '_ytest', '.npy'), yts)

    Xts = vectorizer.transform(Xts)

    if not formal_run:
        report, predictions, probabilities = evaluate_on_testset(rfcv_model, Xts, yts, postprocess)
        print()
        print(report)
        print()

        report_df = error_analysis(test_df, predictions, rfcv_model)
        result_fn = _report_dir / Path('result.log')
        analysis_fn = _report_dir / Path('analysis')
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
        report_df.to_csv(str(analysis_fn.with_suffix('.csv')), index=False, encoding='utf-8-sig')
        report_df.to_excel(str(analysis_fn.with_suffix('.xlsx')), sheet_name='result', index=False,
                           encoding='utf-8-sig')
    else:
        predictions, probabilities = get_preds_and_probs(rfcv_model, Xts, postprocess)

    probs_df = to_prob_df(test_df, probabilities)
    probs_fn = report_dir / Path('probs')
    probs_df.to_csv(str(probs_fn.with_suffix('.csv')), index=False, encoding='utf-8-sig')

    submission_df = to_submission_df(test_df, predictions)
    submission_fn = report_dir / Path('submission')
    submission_df.to_csv(str(submission_fn.with_suffix('.csv')), index=False, encoding='utf-8-sig')
    submission_df.to_excel(str(submission_fn.with_suffix('.xlsx')), index=False, encoding='utf-8-sig')

    modelreport_fn = _report_dir / Path('feature_importance')
    model_interpretation = get_interpretation_of_model(rfcv_model, vectorizer)
    model_interpretation.to_csv(str(modelreport_fn.with_suffix('.csv')), encoding='utf-8-sig')
    model_interpretation.to_excel(str(modelreport_fn.with_suffix('.xlsx')), sheet_name='features', encoding='utf-8-sig')

    logger.info('All done.')


#
# Command line interface
#
@click.command()
@click.argument('task', type=str)
@click.option('--formal-run', is_flag=True, default=False)
@click.option('--selftest', is_flag=True, default=False)
@click.option('--feature', '-f', type=str, multiple=True, help='eg. -f pas -f suf -f userdict')
@click.option('--cache', is_flag=True, default=False)
@click.option('--model-cache', is_flag=True, default=False)
@click.option('--jumanpp', is_flag=True, default=False)
@click.option('--postprocess', is_flag=True, default=False)
@click.option('--report-dir', type=str, default='../reports')
@click.option('--seed', type=int, help='random seed which is used for train/test split')
@click.option('--random-search', type=int, default=100, help='number of iteration of random-parameter-search')
@click.option('--verbose', '-v', is_flag=True, default=False)
def cli(task,
        cache,
        model_cache,
        report_dir,
        feature,
        postprocess,
        seed,
        verbose,
        formal_run,
        random_search,
        jumanpp,
        selftest):
    print('Features: {}'.format(feature))
    end2end(task=task, use_cache=cache, use_model_cache=model_cache, report_dir=report_dir, features=feature,
            random_seed=seed, verbose=verbose, formal_run=formal_run, n_random_search=random_search,
            use_jumanpp=jumanpp, postprocess=postprocess, selftest=selftest)


if __name__ == '__main__':
    cli()
