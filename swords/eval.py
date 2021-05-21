from collections import defaultdict
import importlib
import json
import re
import subprocess
import tempfile

import numpy as np

from . import Label, LexSubDataset, LexSubNoDuplicatesResult
from .assets import ASSETS, file_from_bundle
from .datasets import get_dataset
from .run import get_result
from .lemma import lemmatize


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


def create_comparison_lists(d, r, allow_abstain=False, d_threshold=0.1):
  # TODO: Done hastily for paper. Test!!!
  ref = []
  sys = []
  sys_opt = []
  sys_bin = []

  for tid in d.all_target_ids():
    # Aggregate and threshold dataset substitutes
    d_subs_raw = []
    d_subs = []
    for sid in d.all_substitute_ids(target_id=tid):
      sub = d.get_substitute(sid)['substitute']
      d_subs_raw.append(sub)
      labels = d.get_substitute_labels(sid)
      # TODO: This is breaking change for CoInCo/SemEval07
      rat = sum([int(l in [Label.TRUE, Label.TRUE_IMPLICIT]) for l in labels]) / len(labels)
      if rat >= d_threshold:
        d_subs.append((sub, rat))
    if len(d_subs_raw) == 0:
      continue
    d_subs = sorted(d_subs, key=lambda x: -x[1])
    assert len(d_subs_raw) == len(set(d_subs_raw))
    d_subs_raw = set(d_subs_raw)

    # Aggregate system substitutes
    try:
      r_subs = r.get_substitutes(tid)
    except:
      r_subs = []
    if len(r_subs) == 0 and allow_abstain:
      continue
    r_subs = sorted(r_subs, key=lambda x: -x[1])
    
    assert sorted(r_subs, key=lambda x: -x[1]) == r_subs
    assert sorted(d_subs, key=lambda x: -x[1]) == d_subs
    
    ref.append(d_subs)
    sys.append(r_subs)
    
    r_subs_opt = [(sub, score) for sub, score in r_subs if sub in d_subs_raw]
    sys_opt.append(r_subs_opt)
    
    r_subs_bin = r_subs_opt[:]
    for sub in d_subs_raw - set([sub for sub, _ in r_subs_opt]):
      r_subs_bin.append((sub, float('-inf')))
    sys_bin.append(r_subs_bin)
  
  return ref, sys, sys_opt, sys_bin


def mean_average_precision(dataset_lists, result_lists, pct=True):
  aps = []
  for d, r in zip(dataset_lists, result_lists):
    d = set([s for s, _ in d])
    if len(d) == 0:
      continue
    ap = 0.
    num_positive = 0
    for rank, (s, _) in enumerate(r):
      if s in d:
        ap += (num_positive + 1) / (rank + 1)
        num_positive += 1
    ap /= len(d)
    aps.append(ap)
  return {
    'map': np.mean(aps) * (100. if pct else 1.)
  }


def pr_at_k(dataset_lists, result_lists, k, avg='macro', pct=True):
  if avg == 'micro':
    raise NotImplementedError()
  else:
    numerator = 0
    p_denominator = 0
    r_denominator = 0
    rs_denominator = 0
    for d, r in zip(dataset_lists, result_lists):
      d = set([s for s, _ in d])
      r = [s for s, _ in r]
      r = r[:k]

      numerator += sum([int(s in d) for s in r])
      p_denominator += len(r)
      r_denominator += min(len(d), k)
      rs_denominator += len(d)

    p = numerator / p_denominator
    r = numerator / r_denominator
    f = (2 * p * r) / (p + r)
    rs = numerator / rs_denominator
    fs = (2 * p * rs) / (p + rs)

    mul = 100 if pct else 1
    return {
      f'f@{k}': f * mul,
      f'fs@{k}': fs * mul,
      f'p@{k}': p * mul,
      f'r@{k}': r * mul,
      f'rs@{k}': rs * mul
    }


def stats(datasets_lists, result_lists):
  num_guesses = [len(r) for r in result_lists]
  return {
    'num_m': np.mean(num_guesses),
    'num_s': np.std(num_guesses)
  }


def evaluate(dataset, result, allow_abstain=False, skip_preprocessing=False, verbose=False):
    # Preprocess
    if skip_preprocessing:
        d_pre = dataset
        r_pre = result
    else:
        d_pre = preprocess_dataset_for_evaluation(dataset, verbose=verbose)
        r_pre = preprocess_result_for_evaluation(result, dataset, verbose=verbose)

    # Conceivable
    ref_c, sys, sys_lenient_c, _ = create_comparison_lists(d_pre, r_pre, d_threshold=0.1)

    # Acceptable
    ref_a, _, sys_lenient_a, _ = create_comparison_lists(d_pre, r_pre, d_threshold=0.5+1e-4)

    # Evaluate at different thresholds
    all_metrics = {}
    for pre, r, s in [
        ('lenient_a', ref_a, sys_lenient_a),
        ('lenient_c', ref_c, sys_lenient_c),
        ('strict_a', ref_a, sys),
        ('strict_c', ref_c, sys),
    ]:
        all_metrics.update({f'{pre}_{k}':v for k, v in pr_at_k(ref_a, sys_lenient_a, 10, pct=True).items()})

    # Evaluate GAP
    all_metrics.update(evaluate_gap(d_pre, r_pre))

    # Add legacy metrics
    all_metrics.update(evaluate_mccarthy(d_pre, r_pre, 'best'))
    all_metrics.update(evaluate_mccarthy(d_pre, r_pre, 'oot'))

    return all_metrics


def main(argv):
  from argparse import ArgumentParser

  parser = ArgumentParser()

  parser.add_argument('dataset', type=str)
  parser.add_argument('--result_json_fp', type=str)
  parser.add_argument('--output_metrics_json_fp', type=str)
  parser.add_argument('--allow_abstain', action='store_true', dest='allow_abstain')
  parser.add_argument('--skip_preprocessing', action='store_true', dest='skip_preprocessing')
  parser.add_argument('--metrics', type=str)
  parser.add_argument('--quiet', action='store_false', dest='verbose')

  parser.set_defaults(
      result_json_fp=None,
      output_metrics_json_fp=None,
      allow_abstain=False,
      skip_preprocessing=False,
      metrics='lenient_a_f@10,lenient_c_f@10,strict_a_f@10,strict_c_f@10',
      verbose=True)

  args = parser.parse_args(argv)

  # Load dataset
  print('-' * 80)
  print(f'Getting dataset {args.dataset}')
  dataset = get_dataset(args.dataset, verbose=args.verbose)

  # Load results file
  print('-' * 80)
  print(f'Loading result from {args.result_json_fp}')
  result = get_result(dataset, args.result_json_fp, verbose=args.verbose)

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
  if args.metrics is not None and args.metrics.strip() != 'all':
    allowed = [n.strip() for n in args.metrics.split(',') if len(n.strip()) > 0]
    metric_names = [n for n in metric_names if any([n.startswith(a) for a in allowed])]
  print(','.join(metric_names))
  print(','.join(['e' if metrics[n] is None else '{:.2f}'.format(metrics[n]) for n in metric_names]))
