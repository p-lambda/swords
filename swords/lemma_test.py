import unittest

from swords import Pos
from swords.lemma import *

class TestLemma(unittest.TestCase):
  def test_pos_tag(self):
    c = 'She went to the park.'
    t = 'went'
    self.assertEqual(pos_of_target(c, t, c.index(t)), Pos.VERB)

  def test_lemmatize(self):
    self.assertEqual(lemmatize('went', Pos.VERB), 'go')
    self.assertEqual(lemmatize('chairs', Pos.NOUN), 'chair')
    self.assertEqual(lemmatize('biggest', Pos.ADJ), 'big')
    self.assertEqual(lemmatize('went', context='She went to the park went', target_offset=4), 'go')
    self.assertEqual(lemmatize('went', context='She went to the park'), 'go')

  def test_delemmatize_substiute(self):
    c = 'She visited the better parks.'
    self.assertEqual(delemmatize_substitute('observe', 'visited', context=c), 'observed')
    self.assertEqual(delemmatize_substitute('area', 'parks', context=c), 'areas')
    self.assertEqual(delemmatize_substitute('bad', 'better', context=c), 'worse')


if __name__ == '__main__':
  unittest.main()
