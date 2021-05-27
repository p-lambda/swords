from .. import Label, LexSubDataset
from ..datasets import get_dataset
from . import LexSubGenerator

class CoincoTestOracleGenerator(LexSubGenerator):
  def __init__(self, *args, threshold=1, split='test', **kwargs):
    super().__init__(*args, **kwargs)

    coinco = get_dataset(f'coinco_{split}')
    swords = get_dataset(f'swords-latest_{split}')

    self.tid_to_coinco_id = {}
    for d in [coinco, swords]:
      for tid in d.all_target_ids():
        target = d.get_target(tid)
        if d == coinco:
          coinco_ids = [int(e['legacy_id']) for e in target['extra']]
        else:
          coinco_ids = [int(i) for i in target['extra']['coinco_ids']]
        self.tid_to_coinco_id[tid] = coinco_ids[0]

    self.coinco_id_to_substitutes = {}
    for tid in coinco.all_target_ids():
      target = coinco.get_target(tid)
      coinco_id = [int(e['legacy_id']) for e in target['extra']][0]
      substitutes = []
      for sid in coinco.all_substitute_ids(target_id=tid):
        substitute = coinco.get_substitute(sid)['substitute']
        labels = coinco.get_substitute_labels(sid)
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
