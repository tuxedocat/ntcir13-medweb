'''Semantic features derived from KNP parser and some heuristics.'''
from typing import *
import pyknp
from pyknp.knp.blist import BList
from pyknp.knp.bunsetsu import Bunsetsu
from pyknp.knp.tlist import TList
from pyknp.knp.tag import Tag
import re
import functools

KNPFeature = Dict[str, str]


class PAS:
    def __init__(self,
                 predicate: KNPFeature,
                 ga: KNPFeature = None,
                 wo: KNPFeature = None,
                 ni: KNPFeature = None):
        self.p = predicate
        self.ga = ga
        self.wo = wo
        self.ni = ni

    def __str__(self):
        return '述語={}, ガ格={}, ヲ格={}, ニ格={}'.format(self.p, self.ga, self.wo, self.ni)


class SemanticFeatures:
    def __init__(self):
        self._knp = pyknp.KNP(jumanpp=True)

    def __del__(self):
        del self._knp
        pass

    def parse(self, s: str) -> List[Bunsetsu]:
        return self._knp.parse(s)

    def _get_tag_features(self, b: Bunsetsu) -> Optional[KNPFeature]:
        _ = b.tag_list().tag_list()[0]
        return _.features

    def _is_predicate(self, f: KNPFeature) -> bool:
        return True if '用言' in f else False

    def _get_arguments_from_predicate_feature(self,
                                              p: KNPFeature,
                                              features: List[KNPFeature]) -> List[KNPFeature]:

        def get_argument_id(s: str) -> Optional[int]:
            _i = s.replace('格関係', '')
            if _i.isnumeric():
                return int(_i)
            else:
                return None

        argument_ids = []
        for k in p.keys():
            if k.startswith('格関係'):
                i = get_argument_id(k)
                if i is not None:
                    argument_ids.append(i)
        arguments = [features[i] for i in argument_ids]
        return arguments

    def _to_PAS(self, preds: List[KNPFeature], args: List[List[KNPFeature]]) -> List[PAS]:
        for p, arglist in zip(preds, args):
            ga, wo, ni = None, None, None

            for a in arglist:
                pass

    def get_pas_list(self, s: str) -> List[PAS]:
        parsed = self.parse(s)
        features = [self._get_tag_features(b) for b in parsed]
        predicates = [f for f in features if self._is_predicate(f)]
        arguments = [self._get_arguments_from_predicate_feature(f, features) for f in predicates]
        return (predicates, arguments)


def feature_extractor_run():
    sf = SemanticFeatures()
    preds, argss = sf.get_pas_list("太郎が、お年寄りに席を譲った人に声をかけていた。")
    len(preds)
    assert (True)


if __name__ == '__main__':
    feature_extractor_run()
