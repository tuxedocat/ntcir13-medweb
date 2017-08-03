import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

import krankenfinder
from krankenfinder.features.ja_semantics import *


def test_paslist():
    sf = SemanticFeatures()
    paslist = sf.get_pas_list("太郎が、お年寄りに席を譲った人に声をかけていた。")
    for pas in paslist:
        print(str(pas))
    assert (paslist)


@pytest.fixture
def predicate_dummy():
    predicate_fdic = {'ID': '（文末）',
                      'レベル': 'C',
                      '主節': True,
                      '主題格': '一人称優位',
                      '係': '文末',
                      '動態述語': True,
                      '区切': '5-5',
                      '句点': True,
                      '提題受': '30',
                      '文末': True,
                      '時制-過去': True,
                      '格要素': True,
                      '格解析結果': ':掛ける/かける:動1ガ/C/太郎/0/0/8;ヲ/C/声/5/0/8;ニ/C/人/4/0/8;ト/U/-/-/-/-;デ/U/-/-/-/-;カラ/U/-/-/-/-;ヨリ/U/-/-/-/-;マデ/U/-/-/-/-;ヘ/U/-/-/-/-;時間/U/-/-/-/-;外の関係/U/-/-/-/-;修飾/U/-/-/-/-;ノ/U/-/-/-/-;ニタイスル/U/-/-/-/-;トスル/U/-/-/-/-;ニムケル/U/-/-/-/-;ニツク/U/-/-/-/-;ニヨル/U/-/-/-/-;ニアワセル/U/-/-/-/-;ヲツウジル/U/-/-/-/-',
                      '格関係0': 'ガ:太郎',
                      '格関係4': 'ニ:人',
                      '格関係5': 'ヲ:声',
                      '正規化代表表記': '掛ける/かける?書ける/かける?欠ける/かける?賭ける/かける?駆ける/かける',
                      '正規化格解析結果-0': ':掛かる/かかる:動1ヲ/C/声/5/0/8;ニ/C/人/4/0/8',
                      '用言': '動',
                      '用言代表表記': '掛ける/かける?書ける/かける?欠ける/かける?賭ける/かける?駆ける/かける',
                      '補文ト': True,
                      '連用要素': True}
    return predicate_fdic


@pytest.fixture
def argument_dummy():
    argument_fdic = {'NE': 'PERSON:太郎',
                     'SM-主体': True,
                     'SM-人': True,
                     'ガ': True,
                     '人名': True,
                     '体言': True,
                     '係': 'ガ格',
                     '先行詞候補': True,
                     '助詞': True,
                     '区切': '0-0',
                     '名詞項候補': True,
                     '文頭': True,
                     '格要素': True,
                     '正規化代表表記': '太郎/たろう',
                     '解析格': 'ガ',
                     '連用要素': True}
    return argument_fdic


def test_get_tense(predicate_dummy):
    result = PASUtils.tense(predicate_dummy)
    assert result == '過去'


def test_get_ne(argument_dummy):
    result = PASUtils.ne(argument_dummy)
    assert result == 'PERSON'


def test_get_featuredict1():
    sf = SemanticFeatures()
    fdic = sf.pas_features("太郎が、お年寄りに席を譲った人に声をかけていた。")
    assert fdic
    print(fdic)


def test_get_featuredict2():
    sf = SemanticFeatures()
    fdic = sf.pas_features("家に帰ると、娘が熱を出していたよ。リンパも腫れてるし。明日病院に行こう。")
    assert fdic
    print(fdic)
