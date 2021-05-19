from . import LexSubGenerator

from .. import LexSubDataset
from ..assets import ASSETS, file_from_bundle
from ..datasets import semeval07

_D = semeval07('test')
_SEMEVAL07_LEGACY_ID_TO_TID = {}
for tid in _D.all_target_ids():
  target = _D.get_target(tid)
  legacy_id = target['extra']['legacy_id']
  _SEMEVAL07_LEGACY_ID_TO_TID[legacy_id] = tid
del _D


class _SemEval07Entrant(LexSubGenerator):
  ENTRANT_FP = None
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if self.ENTRANT_FP is None:
      raise NotImplementedError()
    entrant = file_from_bundle(ASSETS['semeval07_entrants']['fp'], self.ENTRANT_FP)
    lines = entrant.decode('utf-8').strip().splitlines()
    self.tid_to_substitutes = {}
    for l in lines:
      if ':::' in l:
        a, b = l.split(':::')
      else:
        a, b = l.split('::')
      legacy_id = int(a.split()[1])
      tid = _SEMEVAL07_LEGACY_ID_TO_TID.get(legacy_id)
      if tid is not None:
        substitutes = [s.strip() for s in b.split(';') if len(s.strip()) > 0]
        substitutes = [(s, -i) for i, s in enumerate(substitutes)]
        self.tid_to_substitutes[tid] = substitutes

  def substitutes_will_be_lemmatized(self):
    return True

  def generate(self, context, target, target_offset, target_pos=None):
    cid = LexSubDataset.context_id(LexSubDataset.create_context(context))
    tid = LexSubDataset.target_id(LexSubDataset.create_target(
      cid, target, target_offset, pos=target_pos))
    if tid in self.tid_to_substitutes:
      result = self.tid_to_substitutes[tid]
      return result
    else:
      raise ValueError('SemEval07Entrant will only output results for semeval07_test.')


class SemEval07HITBest(_SemEval07Entrant):
  ENTRANT_FP = 'HIT.best'
class SemEval07HITOOT(_SemEval07Entrant):
  ENTRANT_FP = 'HIT.oot'
class SemEval07HITMW(_SemEval07Entrant):
  ENTRANT_FP = 'HIT.mw'
class SemEval07IRST1Best(_SemEval07Entrant):
  ENTRANT_FP = 'IRST1.best'
class SemEval07IRST1OOT(_SemEval07Entrant):
  ENTRANT_FP = 'IRST1.oot'
class SemEval07IRST2Best(_SemEval07Entrant):
  ENTRANT_FP = 'IRST2.best'
class SemEval07IRST2OOT(_SemEval07Entrant):
  ENTRANT_FP = 'IRST2.oot'
class SemEval07KUBest(_SemEval07Entrant):
  ENTRANT_FP = 'KU.best'
class SemEval07KUOOT(_SemEval07Entrant):
  ENTRANT_FP = 'KU.oot'
class SemEval07MELBBest(_SemEval07Entrant):
  ENTRANT_FP = 'MELB.best'
class SemEval07SWAG2OOT(_SemEval07Entrant):
  ENTRANT_FP = 'SWAG2.oot'
class SemEval07SWAGOOT(_SemEval07Entrant):
  ENTRANT_FP = 'SWAG.oot'
class SemEval07TORBest(_SemEval07Entrant):
  ENTRANT_FP = 'TOR.best'
class SemEval07TOROOT(_SemEval07Entrant):
  ENTRANT_FP = 'TOR.oot'
class SemEval07UNTBest(_SemEval07Entrant):
  ENTRANT_FP = 'UNT.best'
class SemEval07UNTOOT(_SemEval07Entrant):
  ENTRANT_FP = 'UNT.oot'
class SemEval07USYDBest(_SemEval07Entrant):
  ENTRANT_FP = 'usyd.best'
class SemEval07USYDOOT(_SemEval07Entrant):
  ENTRANT_FP = 'usyd.oot'
