from collections import defaultdict

from . import LexSubGenerator, LexSubRanker
from .. import LexSubDataset, Label

class Oracle:
  def __init__(self, d):
    if not isinstance(d, LexSubDataset):
      raise ValueError('Must provide dataset as input')
    self.d = d


class OracleWithDistractorsGenerator(Oracle, LexSubGenerator):
  def substitutes_will_be_lemmatized(self):
    return self.d.substitutes_lemmatized

  def generate(self, context, target, target_offset, target_pos=None):
    cid = LexSubDataset.context_id(LexSubDataset.create_context(context))
    tid = LexSubDataset.target_id(LexSubDataset.create_target(cid, target, target_offset, pos=target_pos))
    if not self.d.has_target(tid):
      raise ValueError()
    result = []
    for sid in self.d.all_substitute_ids(target_id=tid):
      substitute = self.d.get_substitute(sid)
      if substitute['target_id'] == tid:
        labels = self.d.get_substitute_labels(sid)
        labels = [l for l in labels if l != Label.UNSURE]
        if len(labels) == 0:
          continue
        score = sum([int(l in [Label.TRUE, Label.TRUE_IMPLICIT]) for l in labels]) / len(labels)
        result.append((substitute['substitute'], score))
    return sorted(result, key=lambda x: -x[1])


class OracleGenerator(OracleWithDistractorsGenerator):
  def __init__(self, d, *args, threshold=0.1, **kwargs):
    super().__init__(d, *args, **kwargs)
    # TODO: compute score
    self.threshold = threshold

  def generate(self, *args, **kwargs):
    result = super().generate(*args, **kwargs)
    return [(sub, score) for sub, score in result if score >= self.threshold]


class OracleTop1Generator(OracleGenerator):
  def generate(self, *args, **kwargs):
    result = super().generate(*args, **kwargs)
    return result[:1]


class OracleRanker(Oracle, LexSubRanker):
  def rank(self, context, target, target_offset, substitute, substitute_lemmatized, target_pos=None):
    raise NotImplementedError()
    """
    cid = LexSubDataset.context_id(LexSubDataset.create_context(context))
    tid = LexSubDataset.target_id(LexSubDataset.create_target(cid, target, target_offset, pos=target_pos))
    sid = LexSubDataset.substitute_id(LexSubDataset.create_substitute(tid, substitute))
    if not self.d.has_substitute(sid):
      raise ValueError()
    labels = self.d.get_substitute_labels(sid)
    num_votes = float(sum([int(l in [Label.TRUE, Label.TRUE_IMPLICIT]) for l in labels]))
    return num_votes
    """
