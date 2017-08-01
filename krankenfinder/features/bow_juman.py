from typing import *
import pyknp
from pyknp import Juman, Jumanpp


def parser_func_jumanpp(lemmatize: bool = True) -> Callable[[str], List[str]]:
    jumanpp = Jumanpp()
    if lemmatize:
        def f(s: str) -> List[str]:
            return [m.genkei for m in jumanpp.analysis(s)]

        return f
    else:
        def g(s: str) -> List[str]:
            return [m.midasi for m in jumanpp.analysis(s)]

        return g


def parser_func_juman(lemmatize: bool = True) -> Callable[[str], List[str]]:
    juman = Juman()
    if lemmatize:
        def f(s: str) -> List[str]:
            return [m.genkei for m in juman.analysis(s)]

        return f
    else:
        def g(s: str) -> List[str]:
            return [m.midasi for m in juman.analysis(s)]

        return g


def sanitycheck():
    print('Juman++')
    print('-' * 80)
    pf = parser_func_jumanpp()
    for i in range(10):
        print(pf("リンパ腺が腫れて痛くて仕方ない。"))
    print('-' * 80)
    pf = parser_func_jumanpp(lemmatize=False)
    for i in range(10):
        print(pf("リンパ腺が腫れて痛くて仕方ない。"))
    print('-' * 80)

    print()
    print('Juman')
    print('-' * 80)
    pf = parser_func_juman()
    for i in range(10):
        print(pf("リンパ腺が腫れて痛くて仕方ない。"))
    print('-' * 80)
    pf = parser_func_juman(lemmatize=False)
    for i in range(10):
        print(pf("リンパ腺が腫れて痛くて仕方ない。"))


if __name__ == '__main__':
    sanitycheck()
