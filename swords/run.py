from collections import defaultdict, Counter
import importlib
import gzip
import json
import os
from tqdm import tqdm
import traceback
import urllib
import warnings

from . import LexSubResult
from .assets import ASSETS
from .datasets import get_dataset
from .methods import LexSubGenerator, LexSubRanker
from .methods.bert import BertInfillingGenerator, BertContextualSimilarityRanker, BertContextualSimilarityWithDelemmatizationRanker, BertBasedLexSubRanker, BertBasedLexSubWithDelemmatizationRanker
from .methods.coinco import CoincoTestOracleGenerator
from .methods.glove import GloveRanker
from .methods.gpt3 import GPT3Generator
from .methods.nonsense import NonsenseGenerator, NonsenseRanker
from .methods.oracle import Oracle, OracleGenerator, OracleTop1Generator, OracleWithDistractorsGenerator, OracleRanker
#from .methods.semeval07 import SemEval07KUOOT, SemEval07UNTOOT
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


def _cache_create_fn_factory(cls, tag):
  def create():
    fp = ASSETS[tag]['fp']
    tid_to_cached_output = _load_json(fp)
    return cls(tid_to_cached_output=tid_to_cached_output)
  return create


GENERATORS = {
  'oracle-acceptable': {
    'create': lambda d: OracleGenerator(d, threshold=0.5+1e-4),
  },
  'oracle-conceivable': {
    'create': lambda d: OracleGenerator(d, threshold=0.1),
  },
  'swords-reannotated-test-acceptable': {
    'create': lambda: OracleGenerator(get_dataset('swords-v0.8-subset-human-baseline_test'), threshold=0.5+1e-4),
  },
  'swords-reannotated-test-conceivable': {
    'create': lambda: OracleGenerator(get_dataset('swords-v0.8-subset-human-baseline_test'), threshold=0.1),
  },
  'coinco-test': {
    'create': lambda: CoincoTestOracleGenerator(threshold=1),
  },
  'coinco-dev': {
    'create': lambda: CoincoTestOracleGenerator(threshold=1, split='dev'),
  },
  #'thesaurus-raw': {
  #  'create': lambda: ThesaurusRawGenerator(),
  #},
  #'thesaurus': {
  #  'create': lambda: ThesaurusWithTargetLemmatizationAndPosFilteringGenerator(),
  #},
  'bert-based-ls': {
    'create': lambda: BertInfillingGenerator(target_corruption='dropout', dropout_p=0.3, top_k=50),
  },
  'bert-based-ls-bert-keep': {
    'create': lambda: BertInfillingGenerator(target_corruption='dropout', dropout_p=0., top_k=50),
  },
  'bert-based-ls-bert-mask': {
    'create': lambda: BertInfillingGenerator(target_corruption='mask', top_k=50),
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
  # Below is for ranking
  'oracle-plus-distractors': {
    'create': lambda d: OracleWithDistractorsGenerator(d),
  },
  'swords-reannotated-test-all': {
    'create': lambda: OracleWithDistractorsGenerator(get_dataset('swords-v0.8-subset-human-baseline_test')),
  },
}


RANKERS = {
  'nonsense': {
    'create': lambda: NonsenseRanker(),
  },
  'glove': {
    'create': lambda: GloveRanker(),
  },
  'bert-contextual': {
    'create': lambda: BertContextualSimilarityWithDelemmatizationRanker(),
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
      raise ValueError(f'Unknown method: {generator}')

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
      raise ValueError(f'Unknown method: {ranker}')
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


def main(argv):
  from argparse import ArgumentParser

  parser = ArgumentParser()

  parser.add_argument('dataset', type=str)
  parser.add_argument('--generator', type=str)
  parser.add_argument('--ranker', type=str)
  parser.add_argument('--output_result_json_fp', type=str)
  parser.add_argument('--quiet', action='store_false', dest='verbose')

  parser.set_defaults(
      generator='oracle-plus-distractors',
      ranker='identity',
      output_result_json_fp=None,
      verbose=True)

  args = parser.parse_args(argv)

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
