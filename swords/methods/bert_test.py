import unittest

from swords.methods.bert import *

class TestBertMethods(unittest.TestCase):
  def test_bert_contextual_similarity_ranker(self):
    context = 'She went to the park.'
    target = 'went'

    method = BertContextualSimilarityRanker()
    substitutes = ['travel', 'venture', 'love', 'travelled', 'ventured', 'loved']
    expected_scores = [0.518379, 0.574665, 0.529769, 0.658006, 0.816883, 0.722696]
    for substitute, expected_score in zip(substitutes, expected_scores):
      score = method(context, target, context.index(target), substitute, True)
      self.assertAlmostEqual(score, expected_score, places=4)

    method = BertContextualSimilarityWithDelemmatizationRanker()
    substitutes = ['travel', 'venture', 'love', 'travelled', 'ventured', 'loved']
    expected_scores = [0.658006, 0.816883, 0.722696, 0.658006, 0.816883, 0.722696]
    for substitute, expected_score in zip(substitutes, expected_scores):
      score = method(context, target, context.index(target), substitute, True)
      self.assertAlmostEqual(score, expected_score, places=4)

  def test_bert_infilling_generator(self):
    context = 'She went to the park.'
    target = 'went'

    subs = BertInfillingGenerator()(context, target, context.index(target))
    topk_subs = BertInfillingGenerator(top_k=10)(context, target, context.index(target))
    nucleusp_subs = BertInfillingGenerator(nucleus_p=0.95)(context, target, context.index(target))

    self.assertEqual(len(subs), 23501)
    self.assertAlmostEqual(sum([p for _, p in subs]), 1., places=4)
    self.assertEqual(len(topk_subs), 10)
    self.assertEqual(len(nucleusp_subs), 120)
    self.assertGreaterEqual(sum([p for _, p in nucleusp_subs]), 0.95)

    self.assertEqual(subs[:120], nucleusp_subs)
    self.assertEqual(subs[:10], topk_subs)
    self.assertEqual(nucleusp_subs[:10], topk_subs)

    self.assertEqual([s for s, _ in topk_subs][:4], ['drove', 'returned', 'walked', 'headed'])

    subs = BertInfillingGenerator(target_corruption='dropout', dropout_p=0.1, dropout_seed=0, nucleus_p=0.95)(context, target, context.index(target))
    self.assertEqual(len(subs), 7)
    self.assertEqual([s for s, _ in subs], ['drove', 'walked', 'was', 'took', 'came', 'goes', 'going'])

  def test_bert_based_ls(self):
    context = 'She went to the park.'
    target = 'went'
    substitute = 'ventured'

    bbls = BertBasedLexSubRanker(dropout_seed=0, include_final_embedding_layer=True, attention_dim=1)
    score = bbls(context, target, context.index(target), substitute, True)
    self.assertAlmostEqual(score, 0.7285897254943847, places=4)

    context = 'Please replacemultiplewordpieces.'
    target = 'replacemultiplewordpieces'
    substitute = 'withtheseotherwordpieces'
    score = bbls(context, target, context.index(target), substitute, True)
    self.assertAlmostEqual(score, 0.6896829462051391, places=4)

if __name__ == '__main__':
  unittest.main()
