"""Semantic features derived from KNP parser and some heuristics."""
from typing import *
import pyknp
from pyknp.knp.blist import BList
from pyknp.knp.bunsetsu import Bunsetsu
from pyknp.knp.tlist import TList
from pyknp.knp.tag import Tag
import re
import functools

import logging
import logging.config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)
logging.captureWarnings(True)

KNPFeature = Dict[str, str]
NOT_AVAILABLE = 'NA'


class PASUtils:
    ARGTYPE = {'ガ': 'ga', 'ヲ': 'wo', 'ニ': 'ni'}

    @classmethod
    def surface(cls, tags: KNPFeature) -> str:
        _tag = tags.get('正規化代表表記', NOT_AVAILABLE)
        surface = _tag.split('/')
        try:
            return surface[0]
        except IndexError:
            return NOT_AVAILABLE

    @classmethod
    def tense(cls, tags: KNPFeature) -> str:
        tensetags = [tag for tag in tags.keys() if tag.startswith('時制')]
        if tensetags:
            # surfaces = [tags[k] for k in tensetags]
            ts = '-'.join(tensetags).replace('時制-', '')
            return ts
        else:
            return NOT_AVAILABLE

    @classmethod
    def ne(cls, tags: KNPFeature) -> str:
        _tag = tags.get('NE')
        if not _tag:
            return NOT_AVAILABLE

        try:
            ne, surface = _tag.split(':')
            return ne
        except ValueError:
            return NOT_AVAILABLE

    @classmethod
    def argtype(cls, tags: KNPFeature) -> str:
        _tag = tags.get('解析格')
        if _tag:
            return cls.ARGTYPE.get(_tag)
        else:
            return NOT_AVAILABLE


class Predicate:
    def __init__(self, feature: KNPFeature) -> None:
        self.feature = feature
        self.surface = PASUtils.surface(feature)
        self.tense = PASUtils.tense(feature)

    def __str__(self):
        return '{}/{}'.format(self.surface, self.tense)


class Argument:
    def __init__(self, feature: KNPFeature) -> None:
        self.feature = feature
        self.surface = PASUtils.surface(feature)
        self.ne = PASUtils.ne(feature)

    def __str__(self):
        return '{}/{}'.format(self.surface, self.ne)


class PAS:
    def __init__(self, predicate: Predicate, ga: Argument = None, wo: Argument = None, ni: Argument = None) -> None:
        self.p = predicate
        self.ga = ga if ga else Argument({})
        self.wo = wo if wo else Argument({})
        self.ni = ni if ni else Argument({})
        self.features = {}

    def __str__(self):
        return '述語={}, ガ格={}, ヲ格={}, ニ格={}'.format(self.p.surface, self.ga.surface, self.wo.surface, self.ni.surface)

    def featuredict(self, exclude_NAs=True) -> Dict[str, int]:
        fdic = {}  # type: Dict[str, int]
        fdic.update({'p-surf-{}'.format(self.p.surface): 1})
        fdic.update({'p-{}'.format(self.p.tense): 1})
        fdic.update({'ga-surf-{}'.format(self.ga.surface): 1})
        fdic.update({'wo-surf-{}'.format(self.wo.surface): 1})
        fdic.update({'ni-surf-{}'.format(self.ni.surface): 1})
        fdic.update({'ga-ne-{}'.format(self.ga.ne): 1})
        fdic.update({'wo-ne-{}'.format(self.wo.ne): 1})
        fdic.update({'ni-ne-{}'.format(self.ni.ne): 1})

        if exclude_NAs:
            self.features = {k: v for k, v in fdic.items() if not k.endswith('-NA')}
        else:
            self.features = fdic
        return self.features


class SemanticFeatures:
    """意味的素性抽出器"""

    def __init__(self, verbose=False):
        self._knp = pyknp.KNP(jumanpp=False)
        self.verbose = verbose

    def __del__(self):
        del self._knp

    def parse(self, s: str) -> List[Bunsetsu]:
        try:
            if self.verbose:
                logger.debug(s)
            return self._knp.parse(s)
        except ValueError:
            logging.debug('cannnot parsed sentence {}'.format(s))
            raise

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

        arguments = []
        for i in argument_ids:
            try:
                arguments.append(features[i])
            except IndexError:
                pass
        return arguments

    def _to_PAS(self, pred: KNPFeature, args: List[KNPFeature]) -> PAS:
        ga, wo, ni = None, None, None

        predicate = Predicate(pred)

        for a in args:
            argtype = PASUtils.argtype(a)
            if argtype == 'ga':
                ga = Argument(a)
            elif argtype == 'wo':
                wo = Argument(a)
            elif argtype == 'ni':
                ni = Argument(a)

        return PAS(predicate=predicate, ga=ga, wo=wo, ni=ni)

    def get_pas_list(self, s: str) -> List[PAS]:
        parsed = self.parse(s)
        features = [self._get_tag_features(b) for b in parsed]
        predicates = [f for f in features if self._is_predicate(f)]
        arguments = [self._get_arguments_from_predicate_feature(f, features) for f in predicates]
        paslist = [self._to_PAS(p, args) for p, args in zip(predicates, arguments)]
        return paslist

    def pas_features(self, s: str) -> Dict[str, int]:
        """Main API"""
        featuredict = {}  # type: Dict[str, int]
        pas_list = self.get_pas_list(s)
        features = [p.featuredict() for p in pas_list]
        for fd in features:
            featuredict.update(fd)
        return featuredict
