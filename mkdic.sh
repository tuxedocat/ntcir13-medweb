#!/bin/bash
python mk_mecabdict.py data/dict.xlsx -c && \
/usr/lib/mecab/mecab-dict-index \
	-d /usr/lib/mecab/dic/mecab-ipadic-neologd \
	-u mecab-dict/cat.dic\
	-f utf-8 \
	-t utf-8 \
	data/dict-cat.csv
