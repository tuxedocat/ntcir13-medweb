#!/usr/bin/env python3
import pathlib
import sys
import argparse

import pandas as pd

COLUMNS = ['表層形', '左文脈ID', '右文脈ID', 'コスト', '品詞', '品詞細分類1', '品詞細分類2', '品詞細分類3',
           '活用形', '活用型', '原形', '読み', '発音', 'meta_normalization', 'meta_negative', 'meta_normalize_only']


def make_mecab_dict(fn: str, exclude_unnormalized: bool = True, concatenate: bool = False) -> None:
    """Load source excel file, filter entries by metadata column, and writeout as CSV"""
    dataframe_dict = pd.read_excel(fn, encoding='shift-jis', sheetname=None)
    rootdir = pathlib.Path(fn).parent
    dfs = []

    for sheet in dataframe_dict.keys():
        df = dataframe_dict.get(sheet)

        if exclude_unnormalized:
            df = df.loc[(df["meta_normalize_only"] != 1), COLUMNS[0:13]]

        df = df.replace(pd.np.nan, '')

        if concatenate:
            dfs.append(df)
        else:
            out_fn = rootdir / pathlib.Path(sheet + '.csv')
            df.to_csv(str(out_fn), columns=COLUMNS[0:13], encoding='utf-8', index=False, header=False)
            print('Wrote {}'.format(str(out_fn)))

    if concatenate:
        dfcat = pd.concat(dfs, axis=0, ignore_index=True)
        out_fn = rootdir / pathlib.Path('dict-cat.csv')
        dfcat.to_csv(str(out_fn), columns=COLUMNS[0:13], encoding='utf-8', index=False, header=False)
        print('Wrote {}'.format(str(out_fn)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load excel-formatted dict file and export as csv for building MeCab\'s user dict.')
    parser.add_argument('input_xlsx', type=str, help='Input dictionary file.')
    parser.add_argument('-e', '--exclude-unnormalized',
                        action='store_true',
                        default=False, help='Exclude columns where meta_normalize_only == 1 if set.')
    parser.add_argument('-c', '--concatenate',
                        action='store_true',
                        default=False, help='Concatenate sheets into one CSV.')
    args = parser.parse_args()

    make_mecab_dict(args.input_xlsx, args.exclude_unnormalized, args.concatenate)
