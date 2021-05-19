import unittest

from swords.methods.wordtune import *

class TestWordtune(unittest.TestCase):
  def test_wordtune(self):
    context = 'She went to the park.'
    target = 'went'

    method = WordtuneClues()
    substitutes = method(context, target, context.index(target))
    substitutes = [s for s, _ in substitutes]
    # NOTE: As of 20/11/06
    self.assertEqual(substitutes, ['was', 'took', 'walked', 'ran', 'drove', 'started', 'came', 'headed', 'turned', 'got', 'had', 'proceeded', 'returned', 'did', 'moved'])


if __name__ == '__main__':
  unittest.main()
