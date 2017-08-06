from typing import *
import spacy

SELF = {'i', 'we', 'my', 'our', 'me', 'us', 'mine', 'ours'}
OTHERS = {'he', 'his', 'him', 'she', 'her', 'hers', 'they', 'their', 'them', 'theirs', 'it'}


def get_pron_type(s: str) -> str:
    if s.lower() in SELF:
        return '-SELF'
    elif s.lower() in OTHERS:
        return '-OTHERS'
    else:
        return '-NA'


def parser_func_en_suf() -> Callable[[str], List[str]]:
    en = spacy.load('en', parser=False)

    def f(s: str) -> List[str]:
        tokenized = []
        for w in en(s):
            if not w.is_punct:
                if w.lemma_ == '-PRON-':
                    pron = get_pron_type(w.norm_)
                else:
                    pron = ''
                tokenized.append('{}{}'.format(w.lemma_.lower(), pron))
        return tokenized

    return f


def sanitycheck():
    pf = parser_func_en_suf()
    for i in range(10):
        print(pf("I remember when he ate a shirasu rice bowl and got diarrhea afterwards."))
    print('-' * 80)


if __name__ == '__main__':
    sanitycheck()
