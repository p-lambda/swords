import random

from . import LexSubGenerator, LexSubRanker

class NonsenseGenerator(LexSubGenerator):
  def generate(self, *args, **kwargs):
    raise NotImplementedError()


class NonsenseRanker(LexSubRanker):
  def rank(self, *args, **kwargs):
    return random.random()
