from collections import defaultdict
import gzip
import json
import unittest

from swords import Label, LexSubDataset, LexSubGenerationTask, LexSubRankingTask, LexSubNoDuplicatesResult
from swords.datasets import *
from swords.eval import get_result, evaluate_mccarthy
from swords.methods.semeval07 import SemEval07HITOOT

class TestDatasets(unittest.TestCase):
  def test_semeval07(self):
    semeval07_trial = semeval07('trial')
    self.assertEqual(semeval07_trial.stats(include_uninformative_labels=True), (299, 300, 6540, 7125))
    self.assertEqual(semeval07_trial.id(), DATASETS['semeval07_trial']['id'])
    semeval07_test = semeval07('test')
    self.assertEqual(semeval07_test.stats(include_uninformative_labels=True), (1710, 1710, 35191, 38881))
    self.assertEqual(semeval07_test.id(), DATASETS['semeval07_test']['id'])

    # Test table 4
    # TODO: Test rest of table 1/4
    r = LexSubNoDuplicatesResult.from_dict(get_result(semeval07_test, SemEval07HITOOT).as_dict())
    metrics = evaluate_mccarthy(semeval07_test, r, mode='oot')
    self.assertEqual(metrics['oot_p'], 33.92)
    self.assertEqual(metrics['oot_r'], 33.92)
    self.assertEqual(metrics['oot_mode_p'], 46.88)
    self.assertEqual(metrics['oot_mode_r'], 46.88)

    # Test equivalence with legacy AI Thesaurus version
    with gzip.open(ASSETS['test_semeval07_test_ait']['fp'], 'r') as f:
      ait = json.load(f)
    ait_ref = LexSubDataset.from_ait(ait)
    self.assertEqual(
        LexSubGenerationTask.from_dict(semeval07_test.as_dict()).id(),
        LexSubGenerationTask.from_dict(ait_ref.as_dict()).id())
    self.assertNotEqual(
        LexSubRankingTask.from_dict(semeval07_test.as_dict()).id(),
        LexSubRankingTask.from_dict(ait_ref.as_dict()).id())
    self.assertNotEqual(semeval07_test.id(), ait_ref.id())
    semeval07_test = semeval07('test', include_negatives=False)
    self.assertEqual(semeval07_test.stats(include_uninformative_labels=True), (1710, 1710, 6873, 10563))
    self.assertEqual(ait_ref.stats(include_uninformative_labels=True), (1710, 1710, 6873, 10563))
    self.assertEqual(
        LexSubGenerationTask.from_dict(semeval07_test.as_dict()).id(),
        LexSubGenerationTask.from_dict(ait_ref.as_dict()).id())
    self.assertEqual(
        LexSubRankingTask.from_dict(semeval07_test.as_dict()).id(),
        LexSubRankingTask.from_dict(ait_ref.as_dict()).id())
    self.assertEqual(semeval07_test.id(), ait_ref.id())

  def test_twsi(self):
    twsi_all = twsi('all')
    self.assertEqual(twsi_all.stats(include_uninformative_labels=True), (25007, 25030, 1784746, 1819762))
    self.assertEqual(twsi_all.id(), DATASETS['twsi_all']['id'])

  def test_coinco(self):
    context_focus = ('Nathans_Bylichka.txt', 's-r845')

    coinco_dev = coinco('dev')
    self.assertEqual(coinco_dev.id(), DATASETS['coinco_dev']['id'])
    self.assertEqual(coinco_dev.stats(include_uninformative_labels=True), (1577, 10027, 462885, 494018))
    coinco_test = coinco('test')
    self.assertEqual(coinco_test.id(), DATASETS['coinco_test']['id'])
    self.assertEqual(coinco_test.stats(include_uninformative_labels=True), (896, 5388, 234338, 257906))

    # Parse original XML to get reference numbers
    with gzip.open(ASSETS['coinco_patched']['fp'], 'rt') as f:
      xml = f.read()
      expected_num_contexts = xml.count('<sent ')
      expected_num_targets = xml.count('<substitutions>')
      expected_num_substitutes = xml.count('<subst ')
      expected_num_labels = sum([int(d) for d in re.findall(r'freq="(\d+)" \/>', xml)])

    # Parse with contexts from XML
    coinco_dev = coinco('dev', include_surrounding_context=True, repair_context=False, include_negatives=False, skip_problematic=False)
    self.assertEqual(coinco_dev.id(), 'd:6ad1adf891f0226de02310bc75a5f9d8f04d4504')
    coinco_test = coinco('test', include_surrounding_context=True, repair_context=False, include_negatives=False, skip_problematic=False)
    self.assertEqual(coinco_test.id(), 'd:c6d3b5ba88d3db7818c3e1418d5c4eaaaca82b73')
    s1r0 = None
    for cid in coinco_dev.all_context_ids():
      context = coinco_dev.get_context(cid)
      if tuple(context['extra'][-1]['masc'][k] for k in ['document_fn', 'region_id']) == context_focus:
        s1r0 = context['context']
        break

    # Ensure number of contexts/targets/substitutes in dataset equals that of source file
    # NOTE: There is precisely *1* exact duplicate context in CoInCo (wsj_0006.txt:s-r0/s-r1)
    self.assertEqual(sum([len(d.all_context_ids()) for d in [coinco_dev, coinco_test]]), expected_num_contexts - 1)
    self.assertEqual(sum([len(d.all_target_ids()) for d in [coinco_dev, coinco_test]]), expected_num_targets)
    # NOTE: There is precisely *1* case-insensitive duplicate substitute in CoInCO (wsj_2465.txt:s-r18 "TV" vs "tv")
    self.assertEqual(sum([len(d.all_substitute_ids()) for d in [coinco_dev, coinco_test]]), expected_num_substitutes - 1)

    # Test equivalence with legacy AI Thesaurus version
    with gzip.open(ASSETS['test_coinco_test_ait']['fp'], 'r') as f:
      ait_ref = LexSubDataset.from_ait(json.load(f))
    for cid in coinco_test.all_context_ids():
      self.assertTrue(ait_ref.has_context(cid))
    for tid in coinco_test.all_target_ids():
      self.assertTrue(ait_ref.has_target(tid))
    for sid in coinco_test.all_substitute_ids():
      self.assertTrue(ait_ref.has_substitute(sid))

    # Ensure number of labels in dataset equals that of source file
    num_labels = 0
    for d in [coinco_dev, coinco_test]:
      for sid in d.all_substitute_ids():
        num_labels += len([l for l in d.get_substitute_labels(sid) if l == Label.TRUE_IMPLICIT])
    self.assertEqual(num_labels, expected_num_labels)

    # Parse with unrepaired target sentence only
    coinco_dev = coinco('dev', include_surrounding_context=False, repair_context=False, include_negatives=False, skip_problematic=False)
    self.assertEqual(coinco_dev.id(), 'd:f1a241572eb11cff385fa50899d8cb080a0c865b')
    coinco_test = coinco('test', include_surrounding_context=False, repair_context=False, include_negatives=False, skip_problematic=False)
    self.assertEqual(coinco_test.id(), 'd:05b72ec17d54f49103eb1ae08c7ab9453ada6714')
    s0r0 = None
    for cid in coinco_dev.all_context_ids():
      context = coinco_dev.get_context(cid)
      if tuple(context['extra'][-1]['masc'][k] for k in ['document_fn', 'region_id']) == context_focus:
        s0r0 = context['context']
        break

    # Parse with repaired target sentence only
    coinco_dev = coinco('dev', include_surrounding_context=False, repair_context=True, include_negatives=False, skip_problematic=False)
    self.assertEqual(coinco_dev.id(), 'd:61b0410ef2d6c72ccdf38a073975cff2627b838a')
    coinco_test = coinco('test', include_surrounding_context=False, repair_context=True, include_negatives=False, skip_problematic=False)
    self.assertEqual(coinco_test.id(), 'd:efeba6d05e31f6bf90d5b29fee3b57499954918e')
    s0r1 = None
    for cid in coinco_dev.all_context_ids():
      context = coinco_dev.get_context(cid)
      if tuple(context['extra'][-1]['masc'][k] for k in ['document_fn', 'region_id']) == context_focus:
        s0r1 = context['context']
        break

    # Parse with unrepaired surrounding context
    # NOTE: Already done above

    # Parse with repaired surrounding context
    coinco_dev = coinco('dev', include_surrounding_context=True, repair_context=True, include_negatives=False, skip_problematic=False)
    self.assertEqual(coinco_dev.stats(include_uninformative_labels=True), (1422, 9603, 65362, 98950))
    self.assertEqual(coinco_dev.id(), 'd:27b473180bf00dae6f850affd8da71f2e22b86e3')
    coinco_test = coinco('test', include_surrounding_context=True, repair_context=True, include_negatives=False, skip_problematic=False)
    self.assertEqual(coinco_test.stats(include_uninformative_labels=True), (843, 5290, 44257, 68496))
    self.assertEqual(coinco_test.id(), 'd:094121b91f815c621d38e393f3a393bc82efd6b9')
    s1r1 = None
    for cid in coinco_dev.all_context_ids():
      context = coinco_dev.get_context(cid)
      if tuple(context['extra'][-1]['masc'][k] for k in ['document_fn', 'region_id']) == context_focus:
        s1r1 = context['context']
        break

    self.assertEqual(s0r0, """‘What did you say?’,""")
    self.assertEqual(s0r1, """‘What did you say?’""")
    self.assertEqual(s1r0, """“Don’t name her ‘What did you say?’, okay? ‘What did you say?’, The next thing that comes out of your mouth is probably what she’ll respond to until we figure out how to put her back.”""")
    self.assertEqual(s1r1, """I opened my mouth to ask for an explanation, but Nepthys stopped me. “Don’t name her ‘What did you say?’, okay? The next thing that comes out of your mouth is probably what she’ll respond to until we figure out how to put her back.”""")

  def test_swords(self):
    swords_dev = get_dataset('swords-v0.6_dev', ignore_cache=True)
    self.assertEqual(swords_dev.stats(include_uninformative_labels=True), (417, 417, 24095, 72285))
    num_outliers = sum([int(len(swords_dev.get_substitute_labels(sid)) != 3) for sid in swords_dev.all_substitute_ids()])
    self.assertEqual(num_outliers, 0)

    swords_test = get_dataset('swords-v0.6_test', ignore_cache=True)
    self.assertEqual(swords_test.stats(include_uninformative_labels=True), (833, 833, 47701, 143103))
    num_outliers = sum([int(len(swords_test.get_substitute_labels(sid)) != 3) for sid in swords_test.all_substitute_ids()])
    self.assertEqual(num_outliers, 0)

    swords_test = get_dataset('swords-v0.5_test', ignore_cache=True)
    self.assertEqual(swords_test.stats(include_uninformative_labels=True), (833, 833, 47718, 145344))
    num_over = sum([int(len(swords_test.get_substitute_labels(sid)) > 3) for sid in swords_test.all_substitute_ids()])
    num_under = sum([int(len(swords_test.get_substitute_labels(sid)) < 3) for sid in swords_test.all_substitute_ids()])
    self.assertEqual(num_over, 2164)
    self.assertEqual(num_under, 17)

    swords_dev = get_dataset('swords-v0.5_dev', ignore_cache=True)
    self.assertEqual(swords_dev.stats(include_uninformative_labels=True), (417, 417, 24095, 72648))
    num_outliers = sum([int(len(swords_dev.get_substitute_labels(sid)) != 3) for sid in swords_dev.all_substitute_ids()])
    self.assertEqual(num_outliers, 363)

    swords_test = get_dataset('swords-v0.4_test', ignore_cache=True)
    self.assertEqual(swords_test.stats(include_uninformative_labels=True), (832, 832, 47652, 142955))
    num_outliers = sum([int(len(swords_test.get_substitute_labels(sid)) != 3) for sid in swords_test.all_substitute_ids()])
    self.assertEqual(num_outliers, 13)

if __name__ == '__main__':
  unittest.main()
