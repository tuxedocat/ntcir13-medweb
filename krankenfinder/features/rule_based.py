# import some libs
import re
from typing import *

animal_flu = re.compile(r"[鳥鶏豚犬猫の]インフル")
non_target = re.compile(r"スペインの?風邪|おたふく風邪|デング熱|ジカ熱")
feels_like = re.compile(r"[鼻風邪引ひいた熱インフルエンザ花粉症]っ?ぽい")
getting_better = re.compile(r"[よ良]くなっ|回復してきた|回復し")
metaphor = re.compile(r"アイドル熱|スマホ\w+熱暴走|携帯\w+熱暴走|パソコン\w+熱暴走|\w+が熱暴走|知恵熱")
ANIMAL = '_animal'
NONTARGET = '_nontarget'
FEELSLIKE = '_feelslike'
BETTER = '_gettingbetter'
METAPHOR = '_metaphor'


class RuleBasedFeatures(object):
    def __init__(self):
        self.features_ = {}


class JaRuleBasedFeatures(RuleBasedFeatures):
    def __init__(self):
        super(JaRuleBasedFeatures, self).__init__()

    def features_from_string(self, s: str) -> Dict[str, int]:
        self.out_of_scope(s)
        self.feels_like(s)
        self.getting_better(s)
        self.metaphor(s)
        return self.features_

    def features_from_tokenized_words(self, ss: List[str]) -> Dict[str, int]:
        return self.features_

    def out_of_scope(self, s: str):
        """対象外の疾患名かどうか"""
        if animal_flu.findall(s):
            self.features_.update({ANIMAL: 1})
        else:
            self.features_.update({ANIMAL: 0})

        if non_target.findall(s):
            self.features_.update({NONTARGET: 1})
        else:
            self.features_.update({NONTARGET: 0})

    def feels_like(self, s: str):
        """Xっぽい"""
        if feels_like.findall(s):
            self.features_.update({FEELSLIKE: 1})
        else:
            self.features_.update({FEELSLIKE: 0})

    def getting_better(self, s: str):
        """回復してきているかも？"""
        if getting_better.findall(s):
            self.features_.update({BETTER: 1})
        else:
            self.features_.update({BETTER: 0})

    def metaphor(self, s: str):
        """熱暴走とかアイドル熱とか"""
        if metaphor.findall(s):
            self.features_.update({METAPHOR: 1})
        else:
            self.features_.update({METAPHOR: 0})


class EnRuleBasedFeatures(RuleBasedFeatures):
    def __init__(self):
        super(EnRuleBasedFeatures, self).__init__()


def rulebased_features_ja(s: str) -> List[Tuple[str, int]]:
    fe = JaRuleBasedFeatures()
    return list(fe.features_from_string(s).items())


def sanitycheck():
    l = [
        'ニュースでよく鳥インフルエンザのことがやっているけど、鳥インフルの蔓延はマジで困るのでやめてほしい。',
        '猫のインフルエンザってあるんだろうか。鶏みたいに。',
        'スペイン風邪って歴史的に有名だけど、今現在だとどのぐらいヤバいんだろうね。今ちょうど熱っぽいけど。',
        'インフルにかかって超つらい。',
        'インフルエンザっぽい症状だけど熱はない。',
        'インフルエンザだったみたいだけど、回復してきたっぽい。',
        '携帯が、そしてパソコンが熱暴走していて熱い。'
    ]
    exp = [
        {ANIMAL: 1, NONTARGET: 0, FEELSLIKE: 0, BETTER: 0, METAPHOR: 0},
        {ANIMAL: 1, NONTARGET: 0, FEELSLIKE: 0, BETTER: 0, METAPHOR: 0},
        {ANIMAL: 0, NONTARGET: 1, FEELSLIKE: 1, BETTER: 0, METAPHOR: 0},
        {ANIMAL: 0, NONTARGET: 0, FEELSLIKE: 0, BETTER: 0, METAPHOR: 0},
        {ANIMAL: 0, NONTARGET: 0, FEELSLIKE: 1, BETTER: 0, METAPHOR: 0},
        {ANIMAL: 0, NONTARGET: 0, FEELSLIKE: 1, BETTER: 1, METAPHOR: 0},
        {ANIMAL: 0, NONTARGET: 0, FEELSLIKE: 0, BETTER: 0, METAPHOR: 1},
    ]
    result = []
    for s in l:
        fe = JaRuleBasedFeatures()
        result.append(fe.features_from_string(s))

    for i, (r, e) in enumerate(zip(result, exp)):
        if r != e:
            print(l[i])
            print('got     \t{}'.format(r))
            print('expected\t{}'.format(e))
            raise AssertionError


if __name__ == '__main__':
    sanitycheck()
