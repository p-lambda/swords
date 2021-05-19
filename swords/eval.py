from collections import defaultdict, Counter
import importlib
import gzip
import json
import os
import re
import subprocess
import tempfile
from tqdm import tqdm
import traceback
import urllib
import warnings

import numpy as np

from . import Label, LexSubDataset, LexSubResult, LexSubNoDuplicatesResult
from .assets import ASSETS, file_from_bundle
from .datasets import DATASETS, get_dataset
from .lemma import lemmatize
from .methods import LexSubGenerator, LexSubRanker
from .methods.bert import BertInfillingGenerator, BertContextualSimilarityRanker, BertContextualSimilarityWithDelemmatizationRanker, BertBasedLexSubRanker, BertBasedLexSubWithDelemmatizationRanker
from .methods.coinco import CoincoTestOracleGenerator
from .methods.glove import GloveRanker
from .methods.gpt3 import GPT3Generator
from .methods.nonsense import NonsenseGenerator, NonsenseRanker
from .methods.oracle import Oracle, OracleGenerator, OracleTop1Generator, OracleWithDistractorsGenerator, OracleRanker
from .methods.semeval07 import SemEval07KUOOT, SemEval07UNTOOT
#from .methods.thesaurus import ThesaurusRawGenerator, ThesaurusWithTargetLemmatizationAndPosFilteringGenerator
from .methods.wordtune import WordtuneClues, WordtuneRefine


def _load_json(fp):
  if fp.endswith('.gz'):
    open_fn = gzip.open
  else:
    open_fn = open
  with open_fn(fp, 'rt') as f:
    o = json.load(f)
  return o


# TODO: Remove this
def _cache_create_fn_factory(cls, tag):
  def create():
    fp = ASSETS[tag]['fp']
    tid_to_cached_output = _load_json(fp)
    return cls(tid_to_cached_output=tid_to_cached_output)
  return create


# TODO: Move this
GENERATORS = {
  'oracle': {
    'create': lambda d: OracleGenerator(d),
  },
  'oracle-maj': {
    'create': lambda d: OracleGenerator(d, threshold=0.5+1e-4),
  },
  'oracle-1': {
    'create': lambda d: OracleGenerator(d, threshold=0.1),
  },
  'oracle-2': {
    'create': lambda d: OracleGenerator(d, threshold=0.2),
  },
  'oracle-3': {
    'create': lambda d: OracleGenerator(d, threshold=0.3),
  },
  'oracle-4': {
    'create': lambda d: OracleGenerator(d, threshold=0.4),
  },
  'oracle-5': {
    'create': lambda d: OracleGenerator(d, threshold=0.5),
  },
  'oracle-6': {
    'create': lambda d: OracleGenerator(d, threshold=0.6),
  },
  'oracle-7': {
    'create': lambda d: OracleGenerator(d, threshold=0.7),
  },
  'oracle-8': {
    'create': lambda d: OracleGenerator(d, threshold=0.8),
  },
  'oracle-9': {
    'create': lambda d: OracleGenerator(d, threshold=0.9),
  },
  'oracle-10': {
    'create': lambda d: OracleGenerator(d, threshold=1.0),
  },
  'oracle-top1': {
    'create': lambda d: OracleTop1Generator(d),
  },
  'oracle-plus-distractors': {
    'create': lambda d: OracleWithDistractorsGenerator(d),
  },
  'semeval07-ku': {
    'create': lambda: SemEval07KUOOT(),
  },
  'semeval07-unt': {
    'create': lambda: SemEval07UNTOOT(),
  },
  #'thesaurus-raw': {
  #  'create': lambda: ThesaurusRawGenerator(),
  #},
  #'thesaurus': {
  #  'create': lambda: ThesaurusWithTargetLemmatizationAndPosFilteringGenerator(),
  #},
  'bert-infill-mask-k10': {
    'create': lambda: BertInfillingGenerator(top_k=10),
  },
  'bert-infill-mask-k50': {
    'create': lambda: BertInfillingGenerator(top_k=50),
  },
  'bert-infill-mask-k100': {
    'create': lambda: BertInfillingGenerator(top_k=100),
  },
  'bert-infill-mask-k1000': {
    'create': lambda: BertInfillingGenerator(top_k=1000),
  },
  'bert-infill-mask-p90': {
    'create': lambda: BertInfillingGenerator(nucleus_p=0.90),
  },
  'bert-infill-mask-p95': {
    'create': lambda: BertInfillingGenerator(nucleus_p=0.95),
  },
  'bert-infill-mask-p99': {
    'create': lambda: BertInfillingGenerator(nucleus_p=0.99),
  },
  'bert-infill-dropout-k10': {
    'create': lambda: BertInfillingGenerator(target_corruption='dropout', top_k=10),
  },
  'bert-infill-dropout-k50': {
    'create': lambda: BertInfillingGenerator(target_corruption='dropout', top_k=50),
  },
  'bert-infill-dropout-k100': {
    'create': lambda: BertInfillingGenerator(target_corruption='dropout', top_k=100),
  },
  'bert-infill-dropout-k1000': {
    'create': lambda: BertInfillingGenerator(target_corruption='dropout', top_k=1000),
  },
  'bert-infill-dropout-p90': {
    'create': lambda: BertInfillingGenerator(target_corruption='dropout', nucleus_p=0.90),
  },
  'bert-infill-dropout-p95': {
    'create': lambda: BertInfillingGenerator(target_corruption='dropout', nucleus_p=0.95),
  },
  'bert-infill-dropout-p99': {
    'create': lambda: BertInfillingGenerator(target_corruption='dropout', nucleus_p=0.99),
  },
  'bert-based-ls': {
    'create': lambda: BertInfillingGenerator(target_corruption='dropout', dropout_p=0.3, top_k=50),
  },
  'bert-based-ls-d7': {
    'create': lambda: BertInfillingGenerator(target_corruption='dropout', dropout_p=0.7, top_k=50),
  },
  'bert-based-ls-bert-keep': {
    'create': lambda: BertInfillingGenerator(target_corruption='dropout', dropout_p=0., top_k=50),
  },
  'bert-based-ls-bert-keep-p95': {
    'create': lambda: BertInfillingGenerator(target_corruption='dropout', dropout_p=0., nucleus_p=0.95),
  },
  'bert-based-ls-bert-mask': {
    'create': lambda: BertInfillingGenerator(target_corruption='mask', top_k=50),
  },
  'wordtune-swords-v0.8-test': {
    'create': _cache_create_fn_factory(WordtuneClues, 'methods_wordtune_swords-v0.8_test'),
  },
  'wordtune-swords-v0.8-dev-valid': {
    'create': _cache_create_fn_factory(WordtuneClues, 'methods_wordtune_swords-v0.8_dev-valid'),
  },
  'wordtune-swords-v0.8-test': {
    'create': _cache_create_fn_factory(WordtuneClues, 'methods_wordtune_swords-v0.8_test'),
  },
  'gpt3-swords-v0.8-dev-valid': {
    'create': _cache_create_fn_factory(GPT3Generator, 'methods_gpt3_swords-v0.8_dev-valid'),
  },
  'gpt3-swords-v0.8-test': {
    'create': _cache_create_fn_factory(GPT3Generator, 'methods_gpt3_swords-v0.8_test'),
  },
  'coinco-dev': {
    'create': lambda: CoincoTestOracleGenerator(threshold=1, split='dev'),
  },
  'coinco-test': {
    'create': lambda: CoincoTestOracleGenerator(threshold=1),
  },
  'coinco-test-2': {
    'create': lambda: CoincoTestOracleGenerator(threshold=2),
  },
  'coinco-test-3': {
    'create': lambda: CoincoTestOracleGenerator(threshold=3),
  },
  'coinco-test-4': {
    'create': lambda: CoincoTestOracleGenerator(threshold=4),
  },
  'coinco-test-5': {
    'create': lambda: CoincoTestOracleGenerator(threshold=5),
  },
  'coinco-test-6': {
    'create': lambda: CoincoTestOracleGenerator(threshold=6),
  },
  'swords-reannotated-test': {
    'create': lambda: OracleWithDistractorsGenerator(get_dataset('swords-v0.8-subset-human-baseline_test')),
  },
  'swords-reannotated-test-maj': {
    'create': lambda: OracleGenerator(get_dataset('swords-v0.8-subset-human-baseline_test'), threshold=0.5+1e-4),
  },
  'swords-reannotated-test-1': {
    'create': lambda: OracleGenerator(get_dataset('swords-v0.8-subset-human-baseline_test'), threshold=0.1),
  },
}


RANKERS = {
  'nonsense': {
    'create': lambda: NonsenseRanker(),
  },
  'glove': {
    'create': lambda: GloveRanker(),
  },
  'bert-contextual-raw': {
    'create': lambda: BertContextualSimilarityRanker(),
  },
  'bert-contextual': {
    'create': lambda: BertContextualSimilarityWithDelemmatizationRanker(),
  },
  'bert-based-ls-raw': {
    'create': lambda: BertBasedLexSubRanker(),
  },
  'bert-based-ls-wosp-raw': {
    'create': lambda: BertBasedLexSubRanker(compute_proposal_score=False),
  },
  'bert-based-ls-wosv-raw': {
    'create': lambda: BertBasedLexSubRanker(compute_validation_score=False),
  },
  'bert-based-ls': {
    'create': lambda: BertBasedLexSubWithDelemmatizationRanker(),
  },
  'bert-based-ls-wosp': {
    'create': lambda: BertBasedLexSubWithDelemmatizationRanker(compute_proposal_score=False),
  },
  'bert-based-ls-wosv': {
    'create': lambda: BertBasedLexSubWithDelemmatizationRanker(compute_validation_score=False),
  },
}


def _safe_divide(n, d):
  try:
    result = n / d
  except ZeroDivisionError:
    result = 0.
  return result


def _mccarthy_scoring_script_wrapper(d, r, mode, allow_abstain=True):
  # TODO: support MW mode
  assert mode in ['best', 'oot']
  scoring_script = file_from_bundle(ASSETS['semeval07']['fp'], 'scoring/score.pl').decode('utf-8')
  with tempfile.NamedTemporaryFile('w') as script_f, tempfile.NamedTemporaryFile('w') as ref_f, tempfile.NamedTemporaryFile('w') as sys_f:
    script_f.write(scoring_script)
    script_f.seek(0)

    # Create ref output
    ref_lines = []
    for tid in d.all_target_ids():
      substitutes = []
      for sid in d.all_substitute_ids(target_id=tid):
        sid_attrs = d.get_substitute(sid)
        human_labels = d.get_substitute_labels(sid)
        num_positives = sum([int(l in [Label.TRUE, Label.TRUE_IMPLICIT]) for l in human_labels])
        if num_positives > 0:
          substitutes.append((sid_attrs['substitute'], num_positives))
      substitutes = sorted(substitutes, key=lambda x: -x[1])
      substitutes = ';'.join([f'{sub} {votes}' for sub, votes in substitutes]) + ';'
      ref_lines.append(f'dummylemma.n {tid} :: {substitutes}')
    ref_f.write('\n'.join(ref_lines))
    ref_f.seek(0)

    # Create sys output
    sys_lines = []
    for tid in d.all_target_ids():
      if r.has_substitutes(tid):
        substitutes = r.get_substitutes(tid)
      else:
        if allow_abstain:
          continue
        else:
          substitutes = []
      substitutes = sorted(substitutes, key=lambda x: -x[1])
      substitutes = ';'.join([f'{sub}' for sub, _ in substitutes]) + ';'
      sep = ':::' if mode == 'oot' else '::'
      sys_lines.append(f'dummylemma.n {tid} {sep} {substitutes}')
    sys_f.write('\n'.join(sys_lines))
    sys_f.seek(0)

    cmd = 'perl {} {} {} -t {}'.format(
        script_f.name, sys_f.name, ref_f.name, mode)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = [s.decode('utf-8').strip().splitlines() for s in p.communicate()]

    return stdout, stderr


def evaluate_mccarthy(d, r, mode='oot', allow_abstain=False):
  if not isinstance(d, LexSubDataset):
    raise ValueError()
  if not isinstance(r, LexSubNoDuplicatesResult):
    raise ValueError()
  if not d.substitutes_lemmatized:
    raise ValueError()
  if not r.substitutes_lemmatized:
    raise ValueError()

  # TODO: Should this be False? I believe it is True in original script
  stdout, stderr = _mccarthy_scoring_script_wrapper(d, r, mode, allow_abstain=False)
  result = {
      f'p': None,
      f'r': None,
      f'mode_p': None,
      f'mode_r': None
  }
  if len(stdout) >= 4:
    stdout = stdout[-4:]
    try:
      result['p'], result['r'] = [float(x) for x in re.findall(r'= ([\.\d]+)', stdout[1])]
    except:
      pass
    try:
      result['mode_p'], result['mode_r'] = [float(x) for x in re.findall(r'= ([\.\d]+)', stdout[3])]
    except:
      pass
  result = {f'{mode}_{k}':v for k, v in result.items()}
  return result


def _melamud_gap_scoring_script_wrapper(d, r, aggregate='total', allow_abstain=False):
  spec = importlib.util.spec_from_file_location("GeneralizedAveragePrecision", ASSETS['eval_gap_script']['fp'])
  gap = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(gap)

  gaps = []
  for tid in d.all_target_ids():
    d_sids = d.all_substitute_ids(target_id=tid)
    d_subs = [(d.get_substitute(sid)['substitute'].lower(), d.get_substitute_labels(sid)) for sid in d_sids]
    assert len(d_subs) > 0

    if aggregate == 'total':
      agg = lambda labels: float(sum([int(l in [Label.TRUE, Label.TRUE_IMPLICIT]) for l in labels]))
    elif aggregate == 'ratio':
      agg = lambda labels: sum([int(l in [Label.TRUE, Label.TRUE_IMPLICIT]) for l in labels]) / len(labels)
    else:
      raise ValueError()

    d_subs = [(sub, agg(labels)) for sub, labels in d_subs]
    d_subs = sorted(d_subs, key=lambda x: -x[1])

    try:
      r_subs = [(s, score) for s, score in r.get_substitutes(tid)]
    except:
      r_subs = []
    if len(r_subs) == 0 and allow_abstain:
      continue

    assert len(d_subs) == len(set([s.lower() for s, _ in d_subs]))
    assert len(r_subs) == len(set([s.lower() for s, _ in r_subs]))

    gaps.append(gap.GeneralizedAveragePrecision.calc(d_subs, r_subs))

  return np.mean(gaps)


def evaluate_gap(d, r, allow_abstain=False, pct=True):
  if not isinstance(d, LexSubDataset):
    raise ValueError()
  if not isinstance(r, LexSubNoDuplicatesResult):
    raise ValueError()
  if not d.substitutes_lemmatized:
    raise ValueError()
  if not r.substitutes_lemmatized:
    raise ValueError()

  metrics = {'gap': None, 'gap_rat': None}
  multiplier = 100. if pct else 1.
  gap = _melamud_gap_scoring_script_wrapper(d, r, aggregate='total', allow_abstain=allow_abstain)
  assert gap >= 0 and gap <= 1
  metrics['gap'] = gap * multiplier

  try:
    gap = _melamud_gap_scoring_script_wrapper(d, r, aggregate='total', allow_abstain=allow_abstain)
    assert gap >= 0 and gap <= 1
    metrics['gap'] = gap * multiplier
  except:
    pass
  try:
    gap = _melamud_gap_scoring_script_wrapper(d, r, aggregate='ratio', allow_abstain=allow_abstain)
    assert gap >= 0 and gap <= 1
    metrics['gap_rat'] = gap * multiplier
  except:
    pass
  return metrics


def evaluate_pr_at_k(d, r, k, d_threshold=0.1, allow_abstain=False, pct=True):
  if not isinstance(d, LexSubDataset):
    raise ValueError()
  if not isinstance(r, LexSubNoDuplicatesResult):
    raise ValueError()
  if not d.substitutes_lemmatized:
    raise ValueError()
  if not r.substitutes_lemmatized:
    raise ValueError()

  numerator = 0
  p_denominator = 0
  r_strict_denominator = 0
  r_denominator = 0

  for tid in d.all_target_ids():
    d_sids = d.all_substitute_ids(target_id=tid)
    d_subs = [(d.get_substitute(sid)['substitute'].lower(), d.get_substitute_labels(sid)) for sid in d_sids]
    d_subs = [(sub, sum([int(l in [Label.TRUE, Label.TRUE_IMPLICIT]) for l in labels])) for sub, labels in d_subs]
    # TODO: Different ways for determining positive set besides score > 0?
    # TODO: This is a breaking change for CoInCo/SemEval07 evaluation...
    d_subs = [sub.lower() for sub, score in d_subs if score > 0]
    assert len(d_subs) == len(set(d_subs))
    d_subs = set(d_subs)

    try:
      r_subs = [(s.lower(), score) for s, score in r.get_substitutes(tid)]
    except:
      r_subs = []
    if len(r_subs) == 0 and allow_abstain:
      continue

    assert len(r_subs) == len(set([s.lower() for s, _ in r_subs]))
    assert sorted(r_subs, key=lambda x: -x[1]) == r_subs

    r_subs = r_subs[:k]

    numerator += sum([int(s in d_subs) for s, _ in r_subs])
    p_denominator += len(r_subs)
    r_strict_denominator += len(d_subs)
    r_denominator += min(len(d_subs), k)

  multiplier = 100. if pct else 1.
  metrics = {
		f'p@{k}': _safe_divide(numerator, p_denominator) * multiplier,
    f'rs@{k}': _safe_divide(numerator, r_strict_denominator) * multiplier,
    f'r@{k}': _safe_divide(numerator, r_denominator) * multiplier}

  return metrics


def preprocess_dataset_for_evaluation(dataset, verbose=False):
  _print = lambda x: print(x) if verbose else None

  changed = 0
  merged = 0
  redundant = 0

  # Lemmatize
  sid_to_substitute = {}
  sid_to_labels = defaultdict(list)
  sid_to_target = {}
  sid_to_context = {}
  for sid in dataset.all_substitute_ids():
    substitute = dataset.get_substitute(sid)
    substitute_labels = dataset.get_substitute_labels(sid)
    tid = substitute['target_id']
    target = dataset.get_target(tid)
    context = dataset.get_context(target['context_id'])

    substitute_lemma = lemmatize(
        substitute['substitute'],
        target_pos=target['pos'],
        context=context['context'],
        target_offset=target['offset']).lower().strip()

    target_lemma = lemmatize(
        target['target'],
        target_pos=target['pos'],
        context=context['context'],
        target_offset=target['offset']).lower().strip()

    sid_lemma = LexSubDataset.substitute_id({'target_id': tid, 'substitute': substitute_lemma})
    if substitute_lemma == target_lemma:
      redundant += 1
      continue
    if substitute_lemma != substitute['substitute']:
      changed += 1
    if sid_lemma in sid_to_substitute:
      merged += 1

    sid_to_context[sid_lemma] = context
    sid_to_target[sid_lemma] = target
    sid_to_substitute[sid_lemma] = substitute_lemma
    sid_to_labels[sid_lemma].extend(substitute_labels)

  total = len(dataset.all_substitute_ids())
  _print(f'Lemmatized {changed}/{total}, merged {merged}, redundant {redundant}')

  # Create lemmatized dataset
  preprocessed = LexSubDataset(substitutes_lemmatized=True)
  for sid in sid_to_substitute.keys():
    substitute = sid_to_substitute[sid]
    labels = sid_to_labels[sid]
    target = sid_to_target[sid]
    tid = LexSubDataset.target_id(target)
    context = sid_to_context[sid]

    labels = [l for l in labels if l != Label.UNSURE]
    if len(labels) == 0:
      continue

    if not preprocessed.has_context(LexSubDataset.context_id(context)):
      preprocessed.add_context(context)
    if not preprocessed.has_target(tid):
      _tid = preprocessed.add_target(target)
      assert _tid == tid
    _sid = preprocessed.add_substitute(tid, substitute, labels)
    assert _sid == sid

  return preprocessed


def preprocess_result_for_evaluation(result, dataset, verbose=False):
  _print = lambda x: print(x) if verbose else None

  changed = 0
  merged = 0
  redundant = 0
  total = 0

  r = LexSubNoDuplicatesResult(substitutes_lemmatized=True)
  for tid in result.all_target_ids():
    substitutes = result.get_substitutes(tid)
    target = dataset.get_target(tid)
    context = dataset.get_context(target['context_id'])

    target_lemma = lemmatize(
        target['target'],
        target_pos=target['pos'],
        context=context['context'],
        target_offset=target['offset']).lower().strip()

    substitute_to_max_score = {}
    for substitute_str, score in substitutes:
      substitute_lemma = lemmatize(
          substitute_str,
          target_pos=target['pos'],
          context=context['context'],
          target_offset=target['offset']).lower().strip()

      total += 1
      if substitute_lemma == target_lemma:
        redundant += 1
        continue
      if substitute_lemma != substitute_str:
        changed += 1
      if substitute_lemma in substitute_to_max_score:
        merged += 1
      max_score = substitute_to_max_score.get(substitute_lemma)
      if max_score is None or score > max_score:
        substitute_to_max_score[substitute_lemma] = score

    r.add_substitutes(tid, substitute_to_max_score.items())

  _print(f'Lemmatized {changed}/{total}, merged {merged}, redundant {redundant}')

  return r


def preprocess_result_for_optimistic_evaluation(result, dataset, verbose=False):
  _print = lambda x: print(x) if verbose else None

  if not isinstance(result, LexSubNoDuplicatesResult):
    raise ValueError()

  extra = 0
  missing = 0
  kept = 0

  r = LexSubNoDuplicatesResult(substitutes_lemmatized=True)
  for tid in dataset.all_target_ids():
    try:
      r_subs = result.get_substitutes(tid)
    except:
      r_subs = []
    r_subs, r_scores = zip(*r_subs)
    assert len(set(r_subs)) == len(r_subs)

    d_subs = [dataset.get_substitute(sid)['substitute'] for sid in dataset.all_substitute_ids(target_id=tid)]
    assert len(set(d_subs)) == len(d_subs)
    d_subs = set(d_subs)

    extra += len(set(r_subs) - set(d_subs))
    #added += len(set(d_subs) - set(r_subs))
    kept += len(set(d_subs).intersection(set(r_subs)))

    #r_sub_to_score = {sub:score for sub, score in zip(r_subs, r_scores)}
    #substitutes = [(sub, r_sub_to_score.get(sub, float('-inf'))) for sub in d_subs]
    substitutes = [(sub, score) for sub, score in zip(r_subs, r_scores) if sub in d_subs]

    r.add_substitutes(tid, substitutes)

  _print(f'Missing {missing}, filtered {extra}, kept {kept}')

  return r


def create_comparison_lists(d, r):
  # TODO: Done hastily for paper. Test!!!
  d = dataset
  ref = []
  sys = []
  sys_opt = []
  sys_bin = []

  for tid in d.all_target_ids():
    # Aggregate and threshold dataset substitutes
    d_subs = []
    for sid in d.all_substitute_ids(target_id=tid):
      sub = d.get_substitute(sid)['substitute']
      labels = d.get_substitute_labels(sid)
      # TODO: This is breaking change for CoInCo/SemEval07
      rat = sum([int(l in [Label.TRUE, Label.TRUE_IMPLICIT]) for l in labels]) / len(labels)
      if rat >= d_threshold:
        d_subs.append((sub, rat))
    if len(d_subs) == 0:
      continue

    # Aggregate system substitutes
    try:
      r_subs = [(s.lower(), score) for s, score in r.get_substitutes(tid)]
    except:
      r_subs = []
    if len(r_subs) == 0 and allow_abstain:
      continue
    r_subs = sorted(r_subs, key=lambda x: -x[1])

    comps.append((d_subs, r_subs))


def evaluate(dataset, result, allow_abstain=False, d_threshold=0.1, skip_preprocessing=False, verbose=False):
  _print = lambda x: print(x) if verbose else None

  if not skip_preprocessing:
    # NOTE: Result should come first
    # TODO: Enforce this with classes (i.e., rework LexSubNoDuplicatesResult into LexSubEvalResult or something). Right now there are no guarantees... implemented quickly for deadline
    _print('Preprocessing result for evaluation')
    result = preprocess_result_for_evaluation(result, dataset, verbose=verbose)
    _print('Preprocessing dataset for evaluation')
    dataset = preprocess_dataset_for_evaluation(dataset, verbose=verbose)

  all_metrics = {}
  all_metrics.update(evaluate_gap(dataset, result, allow_abstain=allow_abstain))

  for r in [result, result_optimistic]:
    metrics = {}
    #metrics.update(evaluate_mccarthy(dataset, result, mode='best', allow_abstain=allow_abstain))
    #metrics.update(evaluate_mccarthy(dataset, result, mode='oot', allow_abstain=allow_abstain))
    # TODO: Change GAP to take the simple lists





    

    if r == result_optimistic:
      metrics = {f'o-{k}':v for k, v in metrics.items()}
    all_metrics.update(metrics)
  return all_metrics


def get_result(dataset, generator, verbose=False):
  _print = lambda x: print(x) if verbose else None

  cls = None
  instance = None
  result = None

  # Tag
  if type(generator) == str and generator in GENERATORS:
    _print(f'Creating generator from tag: {generator}')
    if generator.startswith('oracle'):
      instance = GENERATORS[generator]['create'](dataset)
    else:
      instance = GENERATORS[generator]['create']()
  # JSON file
  elif type(generator) == str and os.path.exists(generator):
    # Load results
    _print(f'Loading generator results from file: {generator}')
    result = LexSubResult.from_dict(_load_json(generator))
  # URL file
  elif type(generator) == str and generator.startswith('http') and generator.endswith('.json'):
    _print(f'Loading generator results from URL: {generator}')
    with urllib.request.urlopen(generator) as r:
      result = LexSubResult.from_dict(json.loads(r.read().decode()))
  # Class
  elif type(generator) == type and issubclass(generator, LexSubGenerator):
    cls = generator
  # Instance
  elif isinstance(generator, LexSubGenerator):
    instance = generator
  # Dynamic module
  else:
    try:
      module, cls = generator.rsplit('.', 1)
      cls = getattr(importlib.import_module(module), cls)
      _print(f'Created generator from module: {generator}')
    except:
      raise ValueError('Unknown method')

  assert any([x is not None for x in [cls, instance, result]])

  if result is None:
    # Create instance
    if instance is None:
      _print(f'Creating instance of class: {cls}')
      assert issubclass(cls, LexSubGenerator)
      if issubclass(cls, Oracle):
        instance = cls(dataset)
      else:
        instance = cls()
    assert isinstance(instance, LexSubGenerator)

    # Generate
    _print(f'Generating')
    result = LexSubResult(substitutes_lemmatized=instance.substitutes_will_be_lemmatized())
    _tqdm = tqdm if verbose else lambda x, total: x
    traces = Counter()
    for tid, inputs in _tqdm(dataset.iter_generator_input(), total=len(dataset.all_target_ids())):
      try:
        result.add_substitutes(tid, instance(**inputs))
      except Exception as e:
        traces[traceback.format_exc()] += 1
    errors = sum(traces.values())
    if errors > 0:
      lines = []
      for trace, count in traces.most_common():
        lines.append('-' * 80)
        lines.append(str(count))
        lines.append(trace)
      traces = '\n'.join(lines)
      if len(result) == 0:
        raise Exception('System failed on every target:\n' + traces)
      warnings.warn(f'System error for {errors} targets:\n' + traces)

  assert isinstance(result, LexSubResult)
  return result


def rerank_result(dataset, result, ranker, verbose=False):
  _print = lambda x: print(x) if verbose else None

  cls = None
  instance = None
  reranked = None

  # Type
  if type(ranker) == str and ranker == 'identity':
    reranked = result
  # Tag
  elif type(ranker) == str and ranker in RANKERS:
    _print(f'Creating ranker from tag: {ranker}')
    if ranker.startswith('oracle'):
      instance = RANKERS[ranker]['create'](dataset)
    else:
      instance = RANKERS[ranker]['create']()
  # Class
  elif type(ranker) == type and issubclass(ranker, LexSubRanker):
    cls = ranker
  # Instance
  elif isinstance(ranker, LexSubRanker):
    instance = ranker
  # Dynamic module
  else:
    try:
      module, cls = ranker.rsplit('.', 1)
      cls = getattr(importlib.import_module(module), cls)
    except:
      raise ValueError('Unknown method')
    _print(f'Created ranker from module: {ranker}')

  assert any([x is not None for x in [cls, instance, reranked]])

  # Create instance
  if reranked is None:
    if instance is None:
      _print(f'Creating instance of class: {cls}')
      assert issubclass(cls, LexSubRanker)
      if issubclass(cls, Oracle):
        instance = cls(dataset)
      else:
        instance = cls()
    assert isinstance(instance, LexSubRanker)

    # Generate
    _print(f'Ranking')
    reranked = LexSubResult(substitutes_lemmatized=result.substitutes_lemmatized)
    _tqdm = tqdm if verbose else lambda x, total: x
    traces = Counter()
    success = 0
    for tid in _tqdm(result.all_target_ids(), total=len(result)):
      target = dataset.get_target(tid)
      context = dataset.get_context(target['context_id'])
      substitutes_reranked = []
      for substitute, _ in result.get_substitutes(tid):
        try:
          score = instance(
              context['context'],
              target['target'],
              target['offset'],
              substitute,
              result.substitutes_lemmatized,
              target_pos=target['pos'])
          success += 1
        except Exception as e:
          score = float('-inf')
          traces[traceback.format_exc()] += 1
        substitutes_reranked.append((substitute, score))
      reranked.add_substitutes(tid, substitutes_reranked)
    errors = sum(traces.values())
    if errors > 0:
      lines = []
      for trace, count in traces.most_common():
        lines.append('-' * 80)
        lines.append(str(count))
        lines.append(trace)
      traces = '\n'.join(lines)
      if success == 0:
        raise Exception('System failed on every substitute:\n' + traces)
      warnings.warn(f'System error for {errors} substitutes:\n' + traces)

  assert isinstance(reranked, LexSubResult)
  return reranked


if __name__ == '__main__':
  from argparse import ArgumentParser

  parser = ArgumentParser()

  parser.add_argument('dataset', type=str)
  parser.add_argument('--generator', type=str)
  parser.add_argument('--ranker', type=str)
  parser.add_argument('--output_result_json_fp', type=str)
  parser.add_argument('--output_metrics_json_fp', type=str)
  parser.add_argument('--allow_abstain', action='store_true', dest='allow_abstain')
  parser.add_argument('--skip_preprocessing', action='store_true', dest='skip_preprocessing')
  parser.add_argument('--metrics', type=str)
  parser.add_argument('--quiet', action='store_false', dest='verbose')

  parser.set_defaults(
      generator='oracle-plus-distractors',
      ranker='identity',
      output_result_json_fp=None,
      output_metrics_json_fp=None,
      allow_abstain=False,
      skip_preprocessing=False,
      metrics=None,
      verbose=True)

  args = parser.parse_args()

  # Load dataset
  print('-' * 80)
  print(f'Getting dataset {args.dataset}')
  dataset = get_dataset(args.dataset, verbose=args.verbose)

  # Load or generate results file
  print('-' * 80)
  print(f'Getting result from {args.generator}')
  result = get_result(dataset, args.generator, verbose=args.verbose)

  # Rerank
  print('-' * 80)
  print(f'Reranking result with {args.ranker}')
  result = rerank_result(dataset, result, args.ranker, verbose=args.verbose)

  # Dump to file
  if args.output_result_json_fp is not None:
    with open(args.output_result_json_fp, 'w') as f:
      f.write(json.dumps(result.as_dict()))

  # Run evaluation
  metrics = evaluate(
      dataset,
      result,
      allow_abstain=args.allow_abstain,
      skip_preprocessing=args.skip_preprocessing,
      verbose=args.verbose)

  # Dump to file
  if args.output_metrics_json_fp is not None:
    with open(args.output_metrics_json_fp, 'w') as f:
      f.write(json.dumps(metrics, indent=2))

  # Print
  metric_names = list(metrics.keys())
  if args.metrics is not None:
    allowed = [n.strip() for n in args.metrics.split(',') if len(n.strip()) > 0]
    metric_names = [n for n in metric_names if any([n.startswith(a) for a in allowed])]
  print(','.join(metric_names))
  print(','.join(['e' if metrics[n] is None else '{:.4f}'.format(metrics[n]) for n in metric_names]))
