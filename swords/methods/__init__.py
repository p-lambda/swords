from functools import lru_cache

from .. import Pos
from ..lemma import delemmatize_substitute, pos_of_target

class LexSubGenerator:
  def substitutes_will_be_lemmatized(self):
    raise NotImplementedError()

  # TODO: Batch mode
  def generate(self, context, target, target_offset, target_pos=None):
    raise NotImplementedError()

  def __call__(self, *args, **kwargs):
    return self.generate(*args, **kwargs)


class LexSubWithTargetPosGenerator(LexSubGenerator):
  def __init__(self, *args, pos_tag_strategy='nltk', **kwargs):
    super().__init__(*args, **kwargs)
    self.pos_tag_strategy = pos_tag_strategy

  def generate_with_target_pos(self, context, target, target_offset, target_pos):
    raise NotImplementedError()

  def generate(self, context, target, target_offset, target_pos=None):
    if target_pos is None:
      try:
        target_pos = pos_of_target(
            context,
            target,
            target_offset,
            strategy=self.pos_tag_strategy)
      except:
        target_pos = Pos.UNKNOWN
      assert isinstance(target_pos, Pos)

    return self.generate_with_target_pos(context, target, target_offset, target_pos)


class LexSubRanker:
  # TODO: Batch mode
  def rank(self, context, target, target_offset, substitute, substitute_lemmatized, target_pos=None):
    raise NotImplementedError()

  def __call__(self, *args, **kwargs):
    return self.rank(*args, **kwargs)


class LexSubWithDelemmatizationRanker(LexSubRanker):
  def __init__(self, *args, delemmatize_strategy='pattern', **kwargs):
    super().__init__(*args, **kwargs)
    self.delemmatize_strategy = delemmatize_strategy

  def rank_delemmatized(self, context, target, target_offset, substitute, target_pos=None):
    raise NotImplementedError()

  def rank(self, context, target, target_offset, substitute, substitute_lemmatized, target_pos=None):
    if substitute_lemmatized:
      try:
        substitute = delemmatize_substitute(
            substitute,
            target,
            target_pos=target_pos,
            context=context,
            target_offset=target_offset,
            strategy=self.delemmatize_strategy)
      except Exception as e:
        pass

    return self.rank_delemmatized(context, target, target_offset, substitute, target_pos=target_pos)


class LexSubGeneratorAsRanker(LexSubRanker):
  def __init__(self, generator, *args, **kwargs):
    if not isinstance(generator, LexSubGenerator):
      raise ValueError()

    @lru_cache()
    def generate_cached(*args, **kwargs):
      return generator(*args, **kwargs)
    self.generate_cached = generate_cached

  def rank(self, context, target, target_offset, substitute, substitute_lemmatized, target_pos=None):
    results = self.generate_cached(context, target, target_offset, target_pos=target_pos)
    for sub, score in results:
      # TODO: Handle lemmatization
      if sub == substitute:
        return score
    return float('-inf')
