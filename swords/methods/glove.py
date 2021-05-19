from . import LexSubRanker

class GloveRanker(LexSubRanker):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    import spacy
    self.glove = spacy.load('en_vectors_web_lg')

  def rank(self, context, target, target_offset, substitute, substitute_lemmatized, target_pos=None):
    return self.glove(target).similarity(self.glove(substitute))
