from collections import defaultdict
import copy
from enum import Enum
import hashlib
import json

# From UD v2: https://universaldependencies.org/u/pos/
class Pos(Enum):
  UNKNOWN = 0
  # Open class
  ADJ = 1
  ADV = 2
  INTJ = 3
  NOUN = 4
  PROPN = 5
  VERB = 6
  # Closed class
  ADP = 7
  AUX = 8
  CCONJ = 9
  DET = 10
  NUM = 11
  PART = 12
  PRON = 13
  SCONJ = 14
  # Other
  PUNCT = 15
  SYM = 16
  X = 17


# https://universaldependencies.org/tagset-conversion/en-penn-uposf.html
PTB_POS_TO_POS = """
#=>SYM
$=>SYM
''=>PUNCT
,=>PUNCT
-LRB-=>PUNCT
-RRB-=>PUNCT
.=>PUNCT
:=>PUNCT
AFX=>ADJ
CC=>CCONJ
CD=>NUM
DT=>DET
EX=>PRON
FW=>X
HYPH=>PUNCT
IN=>ADP
JJ=>ADJ
JJR=>ADJ
JJS=>ADJ
LS=>X
MD=>VERB
NIL=>X
NN=>NOUN
NNP=>PROPN
NNPS=>PROPN
NNS=>NOUN
PDT=>DET
POS=>PART
PRP=>PRON
PRP$=>DET
RB=>ADV
RBR=>ADV
RBS=>ADV
RP=>ADP
SYM=>SYM
TO=>PART
UH=>INTJ
VB=>VERB
VBD=>VERB
VBG=>VERB
VBN=>VERB
VBP=>VERB
VBZ=>VERB
WDT=>DET
WP=>PRON
WP$=>DET
WRB=>ADV
``=>PUNCT
""".strip().splitlines()
PTB_POS_TO_POS = {k:Pos[v] for k, v in [l.split('=>') for l in PTB_POS_TO_POS]}

_AIT_POS_TO_POS = {
    'UNKN': Pos.UNKNOWN,
    'VERB': Pos.VERB,
    'NOUN': Pos.NOUN,
    'PRON': Pos.PRON,
    'ADJ': Pos.ADJ,
    'ADV': Pos.ADV,
    'ADP': Pos.ADP,
    'CONJ': Pos.CCONJ,
    'DET': Pos.DET,
    'NUM': Pos.NUM,
    'PRT': Pos.PART,
    'OTH': Pos.X,
    'PUNC': Pos.PUNCT,
    'PROP': Pos.PROPN,
    'PHRS': Pos.UNKNOWN,
}
_POS_TO_AIT_POS = {v:k for k, v in _AIT_POS_TO_POS.items()}
_POS_TO_AIT_POS[Pos.UNKNOWN] = 'UNKN'
_POS_TO_AIT_POS[Pos.INTJ] = 'UNKN'
_POS_TO_AIT_POS[Pos.AUX] = 'UNKN'
_POS_TO_AIT_POS[Pos.SCONJ] = 'CONJ'
_POS_TO_AIT_POS[Pos.SYM] = 'PUNC'
assert len(_POS_TO_AIT_POS) == len(Pos)


class Label(Enum):
  FALSE = 0
  TRUE = 1
  FALSE_IMPLICIT = 2
  TRUE_IMPLICIT = 3
  UNSURE = 4


def _dict_checksum(d):
  d_json = json.dumps(d, sort_keys=True)
  return hashlib.sha1(d_json.encode('utf-8')).hexdigest()


class LexSubGenerationTask:
  def __init__(self, extra=None):
    self.__cid_to_context = {}
    self.__tid_to_target = {}
    if extra is not None:
      try:
        json.dumps(extra)
      except:
        raise ValueError('Extra information must be JSON serializable')
    self.extra = extra

  def stats(self):
    return len(self.__cid_to_context), len(self.__tid_to_target)

  def id(self):
    return 'gt:' + _dict_checksum({
      'contexts': sorted(list(self.all_context_ids())),
      'targets': sorted(list(self.all_target_ids())),
    })

  @classmethod
  def create_context(cls, context_str, extra=None):
    if type(context_str) != str or len(context_str) == 0:
      raise ValueError('Invalid context string')
    if extra is not None:
      try:
        json.dumps(extra)
      except:
        raise ValueError('Extra information must be JSON serializable')

    context = {
        'context': context_str
    }
    if extra is not None:
      context['extra'] = extra
    return context

  @classmethod
  def create_target(cls, context_id, target_str, offset, pos=None, extra=None):
    if not context_id.startswith('c:'):
      raise ValueError('Invalid context ID')
    # TODO: Make sure target_str.strip() == target_str?
    if type(target_str) != str or len(target_str) == 0:
      raise ValueError('Invalid target string')
    if type(offset) != int:
      raise ValueError('Invalid target offset')
    if pos is not None and not isinstance(pos, Pos):
      raise ValueError('Invalid target part-of-speech')
    if extra is not None:
      try:
        json.dumps(extra)
      except:
        raise ValueError('Extra information must be JSON serializable')

    target = {
        'context_id': context_id,
        'target': target_str,
        'offset': offset,
        'pos': pos,
    }
    if extra is not None:
      target['extra'] = extra
    return target

  @classmethod
  def context_id(cls, context):
    return 'c:' + _dict_checksum({
      # NOTE: Context is case-sensitive
      'context': context['context'],
    })

  @classmethod
  def target_id(cls, target):
    return 't:' + _dict_checksum({
      'context_id': target['context_id'],
      # NOTE: Target is case-insensitive (because context has case info)
      'target': target['target'].lower(),
      'offset': target['offset'],
      # NOTE: POS is an *input* to generation models, so it should be considered part of the target checksum
      'pos': None if target['pos'] is None else target['pos'].name
    })

  def has_context(self, context_id):
    return context_id in self.__cid_to_context

  def has_target(self, target_id):
    return target_id in self.__tid_to_target

  def get_context(self, context_id):
    if context_id not in self.__cid_to_context:
      raise ValueError('Invalid context ID')
    return self.__cid_to_context[context_id]

  def get_target(self, target_id):
    if target_id not in self.__tid_to_target:
      raise ValueError('Invalid target ID')
    return self.__tid_to_target[target_id]

  def add_context(self, context_or_context_str, extra=None, update_ok=False):
    if type(context_or_context_str) == dict:
      if extra is not None:
        raise ValueError()
      context = context_or_context_str
      context = self.create_context(context['context'], extra=context.get('extra'))
    else:
      context = self.create_context(context_or_context_str, extra=extra)
    cid = self.context_id(context)
    if not update_ok and cid in self.__cid_to_context:
      raise ValueError('Context ID already exists')
    self.__cid_to_context[cid] = context
    return cid

  def add_target(self, target_or_context_id, target_str=None, offset=None, pos=None, extra=None, update_ok=False):
    if type(target_or_context_id) == dict:
      if any([kwarg is not None for kwarg in [target_str, offset, pos, extra]]):
        raise ValueError()
      target = target_or_context_id
      target = self.create_target(target['context_id'], target['target'], target['offset'], pos=target['pos'], extra=target.get('extra'))
    else:
      if any([kwarg is None for kwarg in [target_str, offset]]):
        raise ValueError()
      target = self.create_target(target_or_context_id, target_str, offset, pos=pos, extra=extra)

    tid = self.target_id(target)
    if not update_ok and tid in self.__tid_to_target:
      raise ValueError('Target ID already exists')

    context_id = target['context_id']
    if not self.has_context(context_id):
      raise ValueError('Invalid context ID')
    context = self.get_context(context_id)
    if context['context'][target['offset']:target['offset']+len(target['target'])].lower() != target['target'].lower():
      raise ValueError('Target not found at offset')

    self.__tid_to_target[tid] = target
    return tid

  def all_context_ids(self):
    return self.__cid_to_context.keys()

  def all_target_ids(self):
    return self.__tid_to_target.keys()

  def get_generator_inputs(self, target_id):
    if not self.has_target(target_id):
      raise ValueError('Invalid target ID')
    target = self.get_target(target_id)
    context = self.get_context(target['context_id'])
    return {
        'context': context['context'],
        'target': target['target'],
        'target_offset': target['offset'],
        'target_pos': target['pos']
    }

  def iter_generator_input(self, batch_size=None, sort=True, sort_by='context_len_descending'):
    if sort:
      if sort_by == 'context_len_descending':
        cid_to_tids = defaultdict(list)
        for tid in self.all_target_ids():
          target = self.get_target(tid)
          cid_to_tids[target['context_id']].append(tid)
        cids_sorted = sorted(cid_to_tids.keys(), key=lambda x: -len(self.get_context(x)['context']))
        tids = []
        for cid in cids_sorted:
          tids.extend(cid_to_tids[cid])
      else:
        raise ValueError()
    else:
      tids = list(self.all_target_ids())

    if batch_size is None:
      for tid in tids:
        yield tid, self.get_generator_inputs(tid)
    else:
      for i in range(0, len(tids), batch_size):
        yield [(tid, self.get_generator_inputs(tid)) for tid in tids[i:i+batch_size]]

  def as_dict(self):
    result = {
        'contexts': copy.deepcopy(self.__cid_to_context),
        'targets': copy.deepcopy(self.__tid_to_target),
    }
    for tid, target in result['targets'].items():
      target['pos'] = None if target['pos'] is None else target['pos'].name
    if self.extra is not None:
      result['extra'] = self.extra
    return result

  @classmethod
  def from_dict(cls, d):
    i = cls(extra=d.get('extra'))
    for cid, context in d['contexts'].items():
      _cid = i.add_context(context)
      assert _cid == cid
    for tid, target in d['targets'].items():
      target['pos'] = None if target['pos'] is None else Pos[target['pos']]
      _tid = i.add_target(target)
      assert _tid == tid
    return i


class LexSubRankingTask(LexSubGenerationTask):
  def __init__(self, substitutes_lemmatized, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if type(substitutes_lemmatized) != bool:
      raise ValueError('Substitutes lemmatized must be True or False')
    self.substitutes_lemmatized = substitutes_lemmatized
    self.__sid_to_substitute = {}
    self.__tid_to_sids = defaultdict(set)

  def stats(self):
    return super().stats() + (len(self.__sid_to_substitute),)

  def id(self):
    return 'rt:' + _dict_checksum({
      'generation_task_id': super().id(),
      'substitutes': sorted(self.all_substitute_ids()),
      'substitutes_lemmatized': self.substitutes_lemmatized
    })

  @classmethod
  def create_substitute(cls, target_id, substitute_str, extra=None):
    if not target_id.startswith('t:'):
      raise ValueError('Invalid target ID')
    # TODO: Make sure substitute_str.strip() == substitute_str?
    if type(substitute_str) != str or len(substitute_str) == 0:
      raise ValueError('Invalid substitute string')
    if extra is not None:
      try:
        json.dumps(extra)
      except:
        raise ValueError('Extra information must be JSON serializable')

    substitute = {
        'target_id': target_id,
        'substitute': substitute_str
    }
    if extra is not None:
      substitute['extra'] = extra
    return substitute

  @classmethod
  def substitute_id(cls, substitute):
    return 's:' + _dict_checksum({
      'target_id': substitute['target_id'],
      # TODO: Change this? (e.g. for acronyms)?
      # NOTE: Substitute is case-insensitive (because context has case info)
      'substitute': substitute['substitute'].lower()
    })

  def has_substitute(self, substitute_id):
    return substitute_id in self.__sid_to_substitute

  def get_substitute(self, substitute_id):
    if not self.has_substitute(substitute_id):
      raise ValueError('Invalid substitute ID')
    return self.__sid_to_substitute[substitute_id]

  def add_substitute(self, substitute_or_target_id, substitute_str=None, extra=None, update_ok=False):
    if type(substitute_or_target_id) == dict:
      if any([kwarg is not None for kwarg in [substitute_str, extra]]):
        raise ValueError()
      substitute = substitute_or_target_id
      substitute = self.create_substitute(substitute['target_id'], substitute['substitute'], extra=substitute.get('extra'))
    else:
      if substitute_str is None:
        raise ValueError()
      substitute = self.create_substitute(substitute_or_target_id, substitute_str, extra=extra)

    sid = self.substitute_id(substitute)
    if not update_ok and sid in self.__sid_to_substitute:
      raise ValueError('Substitute ID already exists')

    target_id = substitute['target_id']
    if not self.has_target(target_id):
      raise ValueError('Invalid target ID')

    self.__sid_to_substitute[sid] = substitute
    self.__tid_to_sids[target_id].add(sid)
    return sid

  def all_substitute_ids(self, target_id=None):
    if target_id is not None:
      if not self.has_target(target_id):
        raise ValueError('Invalid target ID')
      return self.__tid_to_sids[target_id]
    else:
      return self.__sid_to_substitute.keys()

  def get_ranker_inputs(self, substitute_id):
    if not self.has_substitute(substitute_id):
      raise ValueError('Invalid substitute ID')
    substitute = self.get_substitute(substitute_id)
    target = self.get_target(substitute['target_id'])
    context = self.get_context(target['context_id'])
    return {
        'context': context['context'],
        'target': target['target'],
        'target_offset': target['offset'],
        'target_pos': target['pos'],
        'substitute': substitute['substitute'],
        'substitute_lemmatized': self.substitutes_lemmatized
    }

  def as_dict(self):
    result = super().as_dict()
    result.update({
      'substitutes': copy.deepcopy(self.__sid_to_substitute),
      'substitutes_lemmatized': self.substitutes_lemmatized,
    })

    return result

  @classmethod
  def from_dict(cls, d):
    i = cls(
        substitutes_lemmatized=d['substitutes_lemmatized'],
        extra=d.get('extra'))
    # TODO: Any way to use super here?
    for cid, context in d['contexts'].items():
      _cid = i.add_context(context)
      assert _cid == cid
    for tid, target in d['targets'].items():
      target['pos'] = None if target['pos'] is None else Pos[target['pos']]
      _tid = i.add_target(target)
      assert _tid == tid
    for sid, substitute in d['substitutes'].items():
      _sid = i.add_substitute(substitute)
      assert _sid == sid
    return i


class LexSubDataset(LexSubRankingTask):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__sid_to_labels = {}

  def stats(self, include_uninformative_labels=False):
    allow_list = [Label.TRUE, Label.TRUE_IMPLICIT, Label.FALSE]
    if include_uninformative_labels:
      allow_list.extend([Label.FALSE_IMPLICIT, Label.UNSURE])
    num_labels = sum([len([l for l in labels if l in allow_list]) for labels in self.__sid_to_labels.values()])
    return super().stats() + (num_labels,)

  def id(self):
    return 'd:' + _dict_checksum({
      'ranking_task_id': super().id(),
      'substitute_labels': sorted([(sid, [l.name for l in labels]) for sid, labels in self.__sid_to_labels.items()], key=lambda x: x[0])
    })

  def get_substitute_labels(self, substitute_id):
    if not self.has_substitute(substitute_id):
      raise ValueError('Invalid substitute ID')
    return self.__sid_to_labels[substitute_id]

  def add_substitute(self, substitute_or_target_id, labels_or_substitute_str, labels=None, extra=None, update_ok=False):
    if type(substitute_or_target_id) == dict:
      if any([kwarg is not None for kwarg in [labels, extra]]):
        raise ValueError()
      labels = labels_or_substitute_str
      sid = super().add_substitute(substitute_or_target_id, update_ok=update_ok)
    else:
      if any([kwarg is None for kwarg in [labels_or_substitute_str, labels]]):
        raise ValueError()
      sid = super().add_substitute(substitute_or_target_id, labels_or_substitute_str, extra=extra, update_ok=update_ok)

    if labels is None or len(labels) == 0:
      raise ValueError('Labels must not be empty')

    if sid in self.__sid_to_labels:
      old_labels = self.__sid_to_labels[sid]
      if labels[:len(old_labels)] != old_labels:
        raise ValueError('Labels should only be updated')

    self.__sid_to_labels[sid] = labels
    return sid

  def as_dict(self):
    result = super().as_dict()
    result.update({
      'substitute_labels': {sid:[l.name for l in labels] for sid, labels in self.__sid_to_labels.items()}
    })
    return result

  @classmethod
  def from_dict(cls, d):
    i = cls(
        substitutes_lemmatized=d['substitutes_lemmatized'],
        extra=d.get('extra'))
    # TODO: Any way to use super here?
    for cid, context in d['contexts'].items():
      _cid = i.add_context(context)
      assert _cid == cid
    for tid, target in d['targets'].items():
      target['pos'] = None if target['pos'] is None else Pos[target['pos']]
      _tid = i.add_target(target)
      assert _tid == tid
    for sid, substitute in d['substitutes'].items():
      _sid = i.add_substitute(substitute, [Label[l] for l in d['substitute_labels'][sid]])
      assert _sid == sid
    return i

  def as_ait(self):
    d = {
        'ss': [],
    }

    cid_to_s_attrs = {}
    for cid in self.all_context_ids():
      context = self.get_context(cid)
      s_attrs = {
          'id': cid,
          's': context['context'],
          'extra': context.get('extra'),
          'ws': []
      }
      try:
        s_attrs['split'] = context['extra']['split']
      except:
        pass
      cid_to_s_attrs[cid] = s_attrs
      d['ss'].append(s_attrs)

    tid_to_w_attrs = {}
    for tid in self.all_target_ids():
      target = self.get_target(tid)
      w_attrs = {
          'id': tid,
          'w': target['target'],
          'off': target['offset'],
          'pos': [_POS_TO_AIT_POS[target['pos']]],
          'extra': target.get('extra'),
          'wprimes': []
      }
      tid_to_w_attrs[tid] = w_attrs
      cid_to_s_attrs[target['context_id']]['ws'].append(w_attrs)

    for sid in self.all_substitute_ids():
      substitute = self.get_substitute(sid)
      labels = self.get_substitute_labels(sid)
      wp_attrs = {
          'id': sid,
          'wprime': substitute['substitute'],
          'human_labels': [l.name for l in labels]
      }
      if substitute.get('extra') is not None:
        wp_attrs['extra'] = substitute.get('extra')
      tid_to_w_attrs[substitute['target_id']]['wprimes'].append(wp_attrs)

    d['wprimes_lemmatized'] = self.substitutes_lemmatized
    if self.extra is not None:
      d['extra'] = self.extra

    return d

  @classmethod
  def from_ait(cls, d):
    i = cls(
        substitutes_lemmatized=d.get('wprimes_lemmatized', False),
        extra=d.get('extra'))
    for s_attrs in d['ss']:
      split = s_attrs.get('split')
      extra = s_attrs.get('extra')
      if split is not None:
        if extra is None:
          extra = {}
        extra['split'] = split
      cid = i.add_context(s_attrs['s'], extra=extra, update_ok=True)
      for w_attrs in s_attrs['ws']:
        try:
          pos = _AIT_POS_TO_POS[w_attrs['pos'][0]]
        except Exception as e:
          print(w_attrs['pos'])
          raise e
        # TODO: Add rest of POS list?
        tid = i.add_target(
            cid,
            w_attrs['w'],
            w_attrs['off'],
            pos=pos,
            extra=w_attrs.get('extra'),
            update_ok=True)
        for wp_attrs in w_attrs['wprimes']:
          sid = LexSubDataset.substitute_id(LexSubDataset.create_substitute(tid, wp_attrs['wprime']))
          labels = [Label[l] for l in wp_attrs['human_labels']]
          if i.has_substitute(sid):
            labels = i.get_substitute_labels(sid) + labels
          i.add_substitute(
              tid,
              wp_attrs['wprime'],
              labels,
              extra=wp_attrs.get('extra'),
              update_ok=True)

    return i


class LexSubResult:
  def __init__(self, substitutes_lemmatized):
    self.substitutes_lemmatized = substitutes_lemmatized
    self.__tid_to_substitutes = {}

  def __len__(self):
    return len(self.__tid_to_substitutes)

  def has_substitutes(self, target_id):
    return target_id in self.__tid_to_substitutes

  def get_substitutes(self, target_id):
    if not self.has_substitutes(target_id):
      raise ValueError('Invalid target ID')
    return self.__tid_to_substitutes[target_id]

  def _process_substitutes(self, target_id, substitutes):
    if not target_id.startswith('t:'):
      raise ValueError('Invalid target ID')
    processed = []
    for i, substitute in enumerate(substitutes):
      if type(substitute) in [tuple, list] and len(substitute) == 2:
        substitute, score = substitute
        try:
          score = float(score)
        except:
          raise ValueError('Invalid score')
        processed.append((substitute, score))
      elif type(substitute) == str:
        processed.append((substitute, float(-i)))
      else:
        raise ValueError('Substitute must be (str, float) tuple or str')
    processed = sorted(processed, key=lambda x: -x[1])
    return processed

  def add_substitutes(self, target_id, substitutes):
    self.__tid_to_substitutes[target_id] = self._process_substitutes(target_id, substitutes)

  def all_target_ids(self):
    return self.__tid_to_substitutes.keys()

  def iter_ranker_input(self, batch_size=None, sort=True, sort_by='context_len_descending'):
    raise NotImplementedError()
    """
    if sort:
      if sort_by == 'context_len_descending':
        cid_to_sids = defaultdict(list)
        for sid in self.all_substitute_ids(iter_ok=True):
          substitute = self.get_substitute(sid)
          cid = self.get_target(substitute['target_id'])['context_id']
          cid_to_sids[cid].append(sid)
        cids_sorted = sorted(cid_to_sids.keys(), key=lambda x: -len(self.get_context(x)['context']))
        sids = []
        for cid in cids_sorted:
          sids.extend(cid_to_sids[cid])
      else:
        raise ValueError()
    else:
      sids = self.all_substitute_ids()

    if batch_size is None:
      for sid in sids:
        yield sid, self.get_ranker_inputs(sid)
    else:
      for i in range(0, len(sids), batch_size):
        yield [(sid, self.get_ranker_inputs(sid)) for sid in sids[i:i+batch_size]]
    """

  def as_dict(self):
    return {
        'substitutes_lemmatized': self.substitutes_lemmatized,
        'substitutes': copy.deepcopy(self.__tid_to_substitutes)
    }

  @classmethod
  def from_dict(cls, d):
    i = cls(substitutes_lemmatized=d['substitutes_lemmatized'])
    for tid, substitutes in d['substitutes'].items():
      i.add_substitutes(tid, substitutes)
    return i


class LexSubNoDuplicatesResult(LexSubResult):

  def add_substitutes(self, target_id, substitutes):
    processed = self._process_substitutes(target_id, substitutes)
    if len(set([s.lower() for s, _ in processed])) != len(processed):
      raise ValueError('Duplicate substitutes encountered')
    super().add_substitutes(target_id, processed)

  @classmethod
  def from_dict(cls, d, aggregate_fn=lambda l: max(l)):
    i = cls(substitutes_lemmatized=d['substitutes_lemmatized'])
    for tid, substitutes in d['substitutes'].items():
      substitute_lowercase_to_substitutes_and_scores = defaultdict(list)
      for substitute, score in substitutes:
        substitute_lowercase_to_substitutes_and_scores[substitute.lower()].append((substitute, score))
      deduped = []
      for _, substitutes_and_scores in substitute_lowercase_to_substitutes_and_scores.items():
        substitute = sorted([sub for sub, _ in substitutes_and_scores], key=lambda x: sum(1 for c in x if x.isupper()))[-1]
        score = aggregate_fn([score for _, score in substitutes_and_scores])
        deduped.append((substitute, score))
      i.add_substitutes(tid, deduped)
    return i
