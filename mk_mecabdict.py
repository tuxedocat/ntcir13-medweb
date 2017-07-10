#!/usr/bin/env python3
import pathlib
import sys

import pandas as pd

COLUMNS = ['表層形', '左文脈ID', '右文脈ID', 'コスト', '品詞', '品詞細分類1', '品詞細分類2', '品詞細分類3',
           '活用形', '活用型', '原形', '読み', '発音', 'meta_normalization', 'meta_negative', 'meta_normalize_only']


def make_mecab_dict(fn: str) -> None:
    """Load source excel file, filter entries by metadata column, and writeout as CSV"""
    dataframe_dict = pd.read_excel(fn, encoding='shift-jis', sheetname=None)
    rootdir = pathlib.Path(fn).parent

    for sheet in dataframe_dict.keys():
        df = dataframe_dict.get(sheet)
        df = df.loc[(df["meta_normalize_only"] != 1), COLUMNS[0:13]]
        df = df.replace(pd.np.nan, '')
        out_fn = rootdir / pathlib.Path(sheet + '.csv')
        df.to_csv(str(out_fn), columns=COLUMNS[0:13], encoding='utf-8', index=False, header=False)
        print('Wrote {}'.format(str(out_fn)))


if __name__ == '__main__':
    _fn = sys.argv[1]
    make_mecab_dict(_fn)
