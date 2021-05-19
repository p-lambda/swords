import unittest

from swords.datasets import semeval07
from swords.methods.semeval07 import *

class TestSemeval07Methods(unittest.TestCase):
  def test_semeval07_methods(self):
    semeval07_test = semeval07('test')
    # TODO: Iterate through all classes
    m = SemEval07HITBest()
    success = 0
    errors = 0
    tid_to_substitutes = {}
    for tid, kwargs in semeval07_test.iter_generator_input():
      try:
        tid_to_substitutes[tid] = m(**kwargs)
        success += 1
      except:
        errors += 1
    self.assertEqual(success, 1710)
    self.assertEqual(errors, 0)
    # TODO: Compute eval metrics from paper

if __name__ == '__main__':
  unittest.main()
