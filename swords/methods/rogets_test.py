import unittest

from swords import Pos
from swords.methods.rogets import *

class TestRogetsThesaurus(unittest.TestCase):
  def test_rogets_thesaurus(self):
    context_noun = 'It is your turn.'
    context_verb = 'Hey turn around!'
    context_noun_p = 'You took two turns.'
    context_verb_pp = 'She turned and looked.'
    target = 'turn'
    target_noun_p = 'turns'
    target_verb_pp = 'turned'

    method = RogetsThesaurusRawGenerator()
    substitutes = method(context_noun, target, context_noun.index(target))
    self.assertEqual(len(substitutes), 365)
    substitutes = method(context_verb, target, context_verb.index(target))
    self.assertEqual(len(substitutes), 365)
    with self.assertRaises(ValueError):
        substitutes = method(context_noun_p, target_noun_p, context_noun_p.index(target_noun_p))
    with self.assertRaises(ValueError):
        substitutes = method(context_verb_pp, target_verb_pp, context_verb_pp.index(target_verb_pp))

    method = RogetsThesaurusWithTargetLemmatizationAndPosFilteringGenerator()
    substitutes = method(context_noun, target, context_noun.index(target))
    self.assertEqual(len(substitutes), 146)
    substitutes = method(context_verb, target, context_verb.index(target))
    self.assertEqual(len(substitutes), 219)
    substitutes = method(context_noun_p, target_noun_p, context_noun_p.index(target_noun_p))
    self.assertEqual(len(substitutes), 146)
    substitutes = method(context_verb_pp, target_verb_pp, context_verb_pp.index(target_verb_pp))
    self.assertEqual(len(substitutes), 219)


if __name__ == '__main__':
  unittest.main()
