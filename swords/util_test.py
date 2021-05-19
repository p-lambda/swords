import unittest

from swords.util import *

class TestUtil(unittest.TestCase):

  def test_index_all(self):
    s = '  Hello hello  hello'
    self.assertEqual(index_all(s, 'hello'), [2, 8, 15])
    s = '  Hello '
    self.assertEqual(index_all(s, 'hello'), [2])
    s = '  hello  hell'
    self.assertEqual(index_all(s, 'hello'), [2])
    s = '  Hello hello  hello'
    self.assertEqual(index_all(s, 'hello', case_sensitive=True), [8, 15])
    s = '  Hello '
    self.assertEqual(index_all(s, 'hello', case_sensitive=True), [])
    s = '  hello  hell'
    self.assertEqual(index_all(s, 'hello', case_sensitive=True), [2])
    s = '$1 ((]]{^ $1'
    self.assertEqual(index_all(s, '$1'), [0, 10])
    s = 'Hacı Sabancı died on June 24 , 1998 in İstanbul after a two year struggle against lung cancer.'
    self.assertEqual(index_all(s, 'cancer'), [87])

  def test_normalize_text(self):
    sentence = '\n\t  This    sentence has    problems!\t\t\n\n'
    sentence = normalize_text(sentence)
    self.assertEqual(sentence.replace('has', 'had'), 'This sentence had problems!')

  def test_nltk_tokenize(self):
    sentence = 'Don\'t tokenize this incorrectly!'
    tokens = nltk_tokenize(sentence)
    self.assertEqual(tokens,
        ['Do', 'n\'t', 'tokenize', 'this', 'incorrectly', '!'])
    sentence_hat = nltk_detokenize(tokens)
    self.assertEqual(sentence_hat, sentence)

    sentence = '\"This is a quote\"'
    tokens = nltk_tokenize(sentence)
    self.assertEqual(tokens,
        ['\"', 'This', 'is', 'a', 'quote', '\"'])
    sentence_hat = nltk_detokenize(tokens)
    self.assertEqual(sentence_hat, '\" This is a quote \"')

  def test_tokens_offsets_and_residuals(self):
    examples = [
        ' This is an example.  ',
        ' Please tokenize and  align this   correctly :) \n',
    ]
    examples_tokens = [
        ['This', 'notarealtoken', 'an', 'example'],
        nltk_tokenize(examples[1]),
    ]
    expected_token_offsets = [
        [1, None, 9, 12],
        [1, 8, 17, 22, 28, 35, 45, 46],
    ]
    expected_token_residuals = [
        [' ', '', ' is ', ' ', '.  '],
        [' ', ' ', ' ', '  ', ' ', '   ', ' ', '', ' \n']
    ]

    for ex, ex_tokens, expected_off, expected_res in zip(
        examples,
        examples_tokens,
        expected_token_offsets,
        expected_token_residuals):
      ex_tokens_offs = tokens_offsets(ex, ex_tokens)
      self.assertEqual(ex_tokens_offs, expected_off)

      ex_tokens_lres, ex_rres = tokens_residuals(ex, ex_tokens)
      ex_tokens_res = ex_tokens_lres + [ex_rres]
      self.assertEqual(ex_tokens_res, expected_res)

      if None not in expected_off:
        tok_len = sum([len(t) for t in ex_tokens])
        res_len = sum([len(r) for r in ex_tokens_res])
        self.assertEqual(tok_len + res_len, len(ex))
