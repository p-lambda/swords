from collections import defaultdict
from functools import lru_cache
import pickle

from ..assets import ASSETS
from ..datasets import get_dataset
from ..lemma import lemmatize
from .. import Pos, LexSubDataset
from . import LexSubGenerator, LexSubWithTargetPosGenerator

_ROGET_POS_TO_POS = {
    'v': Pos.VERB,
    'n': Pos.NOUN,
    'adj': Pos.ADJ,
    'adv': Pos.ADV
}
@lru_cache(maxsize=1)
def rogets_lemma_to_senses():
    with open(ASSETS['rogets']['fp'], 'rb') as f:
        d = pickle.load(f)
    lemma_to_senses = defaultdict(list)
    for (lemma, pos), v in d.items():
        assert lemma.strip() == lemma
        assert lemma.lower() == lemma
        pos = pos.split('/')
        pos = [_ROGET_POS_TO_POS[''.join(c for c in p if c.isalpha())] for p in pos]
        lemma_to_senses[lemma].append({
            'pos': pos,
            'substitutes': v['results']
        })
    return lemma_to_senses


class RogetsThesaurusRawGenerator(LexSubGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lemma_to_senses = rogets_lemma_to_senses()

    def substitutes_will_be_lemmatized(self):
        return False

    def generate(self, context, target, target_offset, target_pos=None):
        target = target.lower()
        if target not in self._lemma_to_senses:
            raise ValueError()
        senses = self._lemma_to_senses[target]
        subs = []
        for sense in senses:
            subs.extend(sense['substitutes'])
        return [(sub, -i) for i, sub in enumerate(subs)]


class RogetsThesaurusWithTargetLemmatizationAndPosFilteringGenerator(LexSubWithTargetPosGenerator):
    def __init__(self, *args, pos_tag_strategy='nltk', lemmatize_strategy='nltk', **kwargs):
        super().__init__(*args, pos_tag_strategy=pos_tag_strategy, **kwargs)
        self.lemmatize_strategy = lemmatize_strategy
        self._lemma_to_senses = rogets_lemma_to_senses()
        self._swords_dev = get_dataset('swords-latest_dev')
        self._swords_test = get_dataset('swords-latest_test')

    def substitutes_will_be_lemmatized(self):
        return True

    def generate_with_target_pos(self, context, target, target_offset, target_pos):
        # Lemmatize (using "ground truth" from SWORDS)
        cid = LexSubDataset.context_id(LexSubDataset.create_context(context))
        tid = LexSubDataset.target_id(LexSubDataset.create_target(cid, target, target_offset, pos=target_pos))
        split = None
        for d in [self._swords_dev, self._swords_test]:
            if d.has_target(tid):
                split = d
                break
        if split is not None:
            target_lemmatized = split.get_target(tid)['extra']['coinco_lemma']
        else:
            assert False
            target_lemmatized = lemmatize(target, target_pos=target_pos, strategy=self.lemmatize_strategy).lower()

        if target_lemmatized not in self._lemma_to_senses:
            raise ValueError()
        senses = self._lemma_to_senses[target_lemmatized]
        subs = []
        for sense in senses:
            for sub in sense['substitutes']:
                subs.append((sense['pos'], sub))
        substitutes = [(sub, -i) for i, (pos, sub) in enumerate(subs) if target_pos in pos]
        return substitutes
