from .. import Label, LexSubDataset
from ..datasets import get_dataset
from . import LexSubGenerator

class CoincoTestOracleGenerator(LexSubGenerator):
  def __init__(self, *args, threshold=1, split='test', **kwargs):
    super().__init__(*args, **kwargs)

    coinco_test = get_dataset(f'coinco_{split}')
    swords_test = get_dataset(f'swords-latest_{split}')

    self.tid_to_coinco_id = {}
    for d in [coinco_test, swords_test]:
      for tid in d.all_target_ids():
        target = d.get_target(tid)
        coinco_id = [int(e['legacy_id']) for e in target['extra']][0]
        self.tid_to_coinco_id[tid] = coinco_id

    self.coinco_id_to_substitutes = {}
    for tid in coinco_test.all_target_ids():
      target = coinco_test.get_target(tid)
      coinco_id = [int(e['legacy_id']) for e in target['extra']][0]
      substitutes = []
      for sid in coinco_test.all_substitute_ids(target_id=tid):
        substitute = coinco_test.get_substitute(sid)['substitute']
        labels = coinco_test.get_substitute_labels(sid)
        num_votes = sum([int(l in [Label.TRUE, Label.TRUE_IMPLICIT]) for l in labels])
        if num_votes >= threshold:
          substitutes.append((substitute, float(num_votes)))
      self.coinco_id_to_substitutes[coinco_id] = sorted(substitutes, key=lambda x: -x[1])

  def substitutes_will_be_lemmatized(self):
    return True

  def generate(self, context, target, target_offset, target_pos=None):
    cid = LexSubDataset.context_id(LexSubDataset.create_context(context))
    tid = LexSubDataset.target_id(LexSubDataset.create_target(cid, target, target_offset, pos=target_pos))
    return self.coinco_id_to_substitutes[self.tid_to_coinco_id[tid]]
