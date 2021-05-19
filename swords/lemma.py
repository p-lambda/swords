from functools import lru_cache

from . import Pos, PTB_POS_TO_POS
from .util import tokens_offsets, index_all

_POS_STRATEGY_TO_STATE = {}
@lru_cache(maxsize=1024)
def _pos_tag_raw(context, strategy='nltk'):
  if strategy == 'nltk':
    import nltk
    tokens = nltk.word_tokenize(context)
    return nltk.pos_tag(tokens)
  elif strategy == 'spacy':
    raise NotImplementedError()
  else:
    raise ValueError()


def pos_tag(context, strategy='nltk'):
  tags_raw = _pos_tag_raw(context, strategy=strategy)
  if strategy == 'nltk':
    return [(t, PTB_POS_TO_POS.get(p, Pos.UNKNOWN)) for t, p in tags_raw]
  else:
    raise ValueError()


def pos_of_target(context, target, target_offset, **kwargs):
  pos_tags = pos_tag(context, **kwargs)
  offsets = tokens_offsets(context, [t for t, _ in pos_tags])
  try:
    target_idx = offsets.index(target_offset)
    assert target.lower() == pos_tags[target_idx][0].lower()
  except ValueError:
    raise ValueError('Target does not align with tokens')
  return pos_tags[target_idx][1]


_LEMMATIZE_STRATEGY_TO_STATE = {}
def lemmatize(target, target_pos=None, context=None, target_offset=None, strategy='nltk'):
  # Infer part of speech
  if target_pos is None:
    if context is None:
      raise ValueError('Must provide either POS or context')
    else:
      if target_offset is None:
        offsets = index_all(context, target)
        if len(offsets) == 0:
          raise ValueError('Target not found in context')
        elif len(offsets) > 1:
          raise ValueError('Multiple instances of target in context')
        target_offset = offsets[0]
      target_pos = pos_of_target(context, target, target_offset)

  # Lemmatize
  if strategy == 'nltk':
    from nltk.stem import WordNetLemmatizer
    if 'nltk' not in _LEMMATIZE_STRATEGY_TO_STATE:
      _LEMMATIZE_STRATEGY_TO_STATE['nltk'] = WordNetLemmatizer()
    lemmatizer = _LEMMATIZE_STRATEGY_TO_STATE['nltk']
    nltk_pos = {
        Pos.VERB: 'v',
        Pos.NOUN: 'n',
        Pos.ADJ: 'a',
        Pos.ADV: 'r'
    }.get(target_pos, 'n')
    return lemmatizer.lemmatize(target, pos=nltk_pos)
  else:
    raise ValueError()


_DELEMMATIZE_STRATEGY_TO_STATE = {}
def delemmatize_substitute(substitute, target, target_pos=None, context=None, target_offset=None, strategy='pattern'):
  # Infer part of speech
  if target_pos is None:
    if context is None:
      raise ValueError('Must provide either POS or context')
    else:
      if target_offset is None:
        offsets = index_all(context, target)
        if len(offsets) == 0:
          raise ValueError('Target not found in context')
        elif len(offsets) > 1:
          raise ValueError('Multiple instances of target in context')
        target_offset = offsets[0]
      target_pos = pos_of_target(context, target, target_offset)

  if strategy == 'pattern':
    from .legacy import lang_utils as legacy_lang_utils
    if target_pos == Pos.NOUN:
      noun_form = legacy_lang_utils.get_noun_form(target)
      return legacy_lang_utils.handle_noun(substitute, noun_form)
    elif target_pos == Pos.VERB:
      verb_tense = legacy_lang_utils.get_tense(target)
      return legacy_lang_utils.conjugate(substitute, verb_tense)
    elif target_pos == Pos.ADJ:
      adj_form = legacy_lang_utils.get_adj_form(target)
      return legacy_lang_utils.handle_adj(substitute, adj_form)
    else:
      return substitute
  else:
    raise ValueError()
