from typing import *
import collections
import itertools


def get_ngrams(s: List[str], n: int = 3, padding: bool = True) -> List:
    if padding:
        seq = _pad(s, n)
    else:
        seq = s[:]

    ngrams = []
    while seq:
        _ngram = tuple(itertools.islice(seq, n))
        seq.pop(0)
        if len(_ngram) == n:
            ngrams.append(_ngram)

    return ngrams


def _pad(seq: List, n: int = 3, padstr: str = '<s{}>') -> List:
    # leftpad
    lp = []
    for i in reversed(range(n)):
        if not i == 0:
            lp.append(padstr.format('-' + str(i)))

    # rightpad
    rp = []
    for i in range(n):
        if not i == 0:
            rp.append(padstr.format('+' + str(i)))

    return lp + seq[:] + rp


def ngram_features(s: List[str], n: int = 3, padding: bool = True) -> List[Tuple[str, int]]:
    ngrams = ['/'.join(t) for t in get_ngrams(s, n, padding=padding)]
    return list(collections.Counter(ngrams).items())


def test():
    from pprint import pprint
    s1 = ["これ", "は", "もちもち", "し", "て", "もちもち", "し", "て", "いる", "猫", "です", "。"]
    wopadding_2 = get_ngrams(s1, 2, False)
    withpadding_2 = get_ngrams(s1, 2, True)
    pprint(wopadding_2)
    pprint(withpadding_2)
    wopadding_3 = get_ngrams(s1, 3, False)
    withpadding_3 = get_ngrams(s1, 3, True)
    pprint(wopadding_3)
    pprint(withpadding_3)
    pprint(ngram_features(s1, 3))


if __name__ == '__main__':
    test()
