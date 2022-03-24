from collections import defaultdict
import csv
from datetime import datetime
from functools import lru_cache
import gzip
import html
from io import StringIO
import json
import os
import pickle
import re
import traceback
import warnings
import zipfile

from . import Pos, Label, LexSubDataset, PTB_POS_TO_POS
from .assets import ASSETS, file_from_bundle
from .paths import DATASETS_CACHE_DIR
from .util import normalize_text, nltk_detokenize, tokens_offsets, index_all

_LEXELT_PATTERN = re.compile(r'<lexelt item="(.*?)">(.*?)</lexelt>', re.DOTALL)
_INSTANCE_PATTERN = re.compile(r'<instance id="(.*?)">.*?<context>(.*?)</context>.*?</instance>', re.DOTALL)
_HEAD_PATTERN = re.compile(r'<head>(.*?)</head>')
_SEMEVAL07_POS_TO_INTERNAL = {
    'n': Pos.NOUN,
    'a': Pos.ADJ,
    'r': Pos.ADV,
    'v': Pos.VERB,
}
def semeval07(split='test', include_negatives=True):
  if split not in ['trial', 'test']:
    raise ValueError()

  # Load XML
  if split == 'trial':
    xml_fp = 'trial/lexsub_trial.xml'
  else:
    xml_fp = 'test/lexsub_test.xml'
  xml = file_from_bundle(ASSETS['semeval07']['fp'], xml_fp).decode('utf-8')
  
  d = LexSubDataset(
      substitutes_lemmatized=True,
      extra={})
  legacy_id_to_tid = {}
  # NOTE: We don't use beautiful soup due to unescaping problems
  for lexelt_item, lexelt_contents in re.findall(_LEXELT_PATTERN, xml):
    lexelt_lemma, lexelt_lemma_pos = lexelt_item.split('.', 1)
    pos = _SEMEVAL07_POS_TO_INTERNAL[lexelt_lemma_pos.split('.')[-1]]

    for instance_id, instance_content in re.findall(_INSTANCE_PATTERN, lexelt_contents):
      s_id = int(instance_id)

      # Good faith attempt to fix encoding issues
      # TODO: WTF is the actual encoding here?
      s = instance_content.strip()
      s = html.unescape(s)
      s = s.replace('Â\x92', '\'')
      s = s.replace('’', '\'')
      s = s.replace('‘', '\'')
      s = s.replace('Â\x94', '"')
      s = s.replace('“', '"')
      s = s.replace('”', '"')
      s = s.replace('–', '-')
      s = s.encode('utf-8').decode('ascii', 'ignore')

      # Get word
      instance_heads = re.findall(_HEAD_PATTERN, instance_content)
      assert len(instance_heads) == 1
      w = instance_heads[0]
      assert w.strip() == w

      # Find word offset in characters
      s = normalize_text(nltk_detokenize(s.split()))
      s_tokens = s.split()
      s_w_token_id = None
      for i, t in enumerate(s_tokens):
        if '<head>' in t and '</head>' in t:
          s_w_token_id = i
          offset_in_token = t.index('<head>')
          break
      assert s_w_token_id is not None
      offsets = tokens_offsets(s, s_tokens)
      assert None not in offsets
      w_off = offsets[s_w_token_id] + offset_in_token

      # Remove indicators
      s = s.replace('<head>', '')
      s = s.replace('</head>', '')

      assert w_off >= 0 and w_off < len(s)
      assert s[w_off:w_off+len(w)] == w

      cid = LexSubDataset.context_id(LexSubDataset.create_context(s))
      if d.has_context(cid):
        extra = d.get_context(cid)['extra']
        extra['legacy_ids'].append(s_id)
        extra['raw_contexts'].append(instance_content.strip())
      else:
        extra = {
            'legacy_ids': [s_id],
            'raw_contexts': [instance_content.strip()]
        }
      cid = d.add_context(s, extra=extra, update_ok=True)
      tid = d.add_target(
          cid,
          w,
          w_off,
          pos,
          extra={
            'legacy_id': s_id,
            'lexelt_item': lexelt_item.strip(),
          })
      legacy_id_to_tid[s_id] = tid

  # Load substitutes
  if split == 'trial':
    substitutes_fp = 'trial/gold.trial'
  else:
    substitutes_fp = 'scoring/gold'
  gold = True
  substitutes_lines = file_from_bundle(ASSETS['semeval07']['fp'], substitutes_fp).decode('utf-8').strip().splitlines()

  # Load candidates
  with open(ASSETS['semeval07_candidates']['fp'], 'r') as f:
    candidates_lines = f.read().strip().splitlines()
  lexelt_to_candidates = [l.split('::') for l in candidates_lines]
  lexelt_to_candidates = {k.strip():set([' '.join(c.split()) for c in v.split(';') if len(c.strip()) > 0]) for k, v in lexelt_to_candidates}

  for l in substitutes_lines:
    if ':::' in l:
      a, b = l.split(':::')
    else:
      a, b = l.split('::')

    lexelt_item, s_id = a.split()
    lexelt_lemma, lexelt_lemma_pos = lexelt_item.split('.', 1)
    s_id = int(s_id)
    tid = legacy_id_to_tid[s_id]

    substitutes = [r.strip() for r in b.split(';') if len(r.strip()) > 0]
    num_votes = []
    for i, c in enumerate(substitutes):
      num_votes.append(int(c.split()[-1]))
      substitutes[i] = ' '.join(c.split()[:-1])

    candidates = lexelt_to_candidates['.'.join(lexelt_item.split('.')[:2])]
    assert all([s in candidates for s in substitutes])
    substitutes_lowercase = set([s.lower() for s in substitutes])
    negatives = [c for c in candidates if not c.lower() in substitutes_lowercase]

    for i, (substitute, num_vote) in enumerate(zip(substitutes, num_votes)):
      sid = LexSubDataset.substitute_id(LexSubDataset.create_substitute(tid, substitute))
      if d.has_substitute(sid):
        labels = d.get_substitute_labels(sid)
        assert all([l == Label.TRUE_IMPLICIT for l in labels])
        num_vote += len(labels)
      sid = d.add_substitute(tid, substitute, [Label.TRUE_IMPLICIT] * num_vote, update_ok=True)
    if include_negatives:
      for substitute in negatives:
        sid = LexSubDataset.substitute_id(LexSubDataset.create_substitute(tid, substitute))
        if d.has_substitute(sid):
          labels = d.get_substitute_labels(sid)
          assert all([l == Label.FALSE_IMPLICIT for l in labels])
        else:
          d.add_substitute(tid, substitute, [Label.FALSE_IMPLICIT])

  return d


def twsi(split='all', include_negatives=True):
  if split != 'all':
    raise ValueError()

  twsi_id_to_extra = defaultdict(list)
  twsi_id_to_context_str = {}
  twsi_id_target_to_offsets = defaultdict(set)
  d = LexSubDataset(substitutes_lemmatized=True)
  target_id_to_target_lemma = {}
  target_lemma_to_all_substitutes = defaultdict(list)

  with zipfile.ZipFile(ASSETS['twsi_v2']['fp'], 'r') as z:
    # Read contexts/*.contexts and contexts/raw_data/*.senselabels
    for fp in z.namelist():
      raw_context = fp.startswith('TWSI2_complete/contexts/raw_data') and fp.endswith('.senselabels')
      reg_context = fp.startswith('TWSI2_complete/contexts') and fp.endswith('.contexts')
      if not (raw_context or reg_context):
        continue
      text = z.read(fp).decode('utf-8')
      lines = [l.split('\t') for l in text.splitlines()]
      for i, l in enumerate(lines):
        if raw_context:
          _, target_lemma, target_str, twsi_id, context_str, _, _ = l
        else:
          _, target_lemma, target_str, twsi_id, context_str, _, = l
        twsi_id = int(twsi_id)

        # Detokenize context
        context_str_repaired = normalize_text(nltk_detokenize(context_str.split()))

        # Find target using indicators
        try:
          assert context_str_repaired.count('<b>') == 1
          assert context_str_repaired.count('</b>') == 1
          assert context_str_repaired.index('<b>') < context_str_repaired.index('</b>')
        except:
          continue
        target_offset = context_str_repaired.index('<b>')

        # Remove target indicators
        context_str_repaired = context_str_repaired.replace('<b>', '')
        context_str_repaired = context_str_repaired.replace('</b>', '')
        assert context_str_repaired[target_offset:target_offset+len(target_str)].lower() == target_str.lower()
        target_offsets = index_all(context_str_repaired, target_str)
        target_occurrence = target_offsets.index(target_offset)

        # Store
        if twsi_id in twsi_id_to_context_str:
          assert twsi_id_to_context_str[twsi_id] == context_str_repaired
        twsi_id_to_context_str[twsi_id] = context_str_repaired
        twsi_id_to_extra[twsi_id].append({
          'legacy_id': twsi_id,
          'fp': fp,
          'line_index': i,
          'line': l
        })
        twsi_id_target_to_offsets[(twsi_id, target_str)].add(target_offset)

    # Read corpus/wiki_title_sent.txt
    fp = 'TWSI2_complete/corpus/wiki_title_sent.txt'
    text = z.read(fp).decode('utf-8')
    lines = [l.split('\t', 3) for l in text.splitlines()]
    for i, l in enumerate(lines):
      twsi_id, _, _, context_str = l
      twsi_id = int(twsi_id)
      context_str_repaired = normalize_text(nltk_detokenize(context_str.split()))
      if twsi_id not in twsi_id_to_context_str:
        twsi_id_to_context_str[twsi_id] = context_str_repaired
      twsi_id_to_extra[twsi_id].append({
        'legacy_id': twsi_id,
        'fp': fp,
        'line_index': i,
        'line': l
      })

    # Create reverse map
    context_str_to_twsi_ids = defaultdict(list)
    for k, v in twsi_id_to_context_str.items():
      context_str_to_twsi_ids[v].append(k)

    # Read substitutions/raw_data/substitutions_per_sentence/*.turkresults
    for fp in z.namelist():
      if not fp.startswith('TWSI2_complete/substitutions/raw_data/substitutions_per_sentence'):
        continue
      if not fp.endswith('.turkresults'):
        continue
      text = z.read(fp).decode('utf-8')
      lines = [l.split('\t') for l in text.strip().splitlines()]
      assert all([len(l) == 4 for l in lines])

      for i, l in enumerate(lines):
        identifier, target_lemma, substitute_str, count = l
        count = int(count)
        target_str, identifier = identifier.split('++')
        twsi_id, source = identifier.split('||')
        twsi_id = int(twsi_id)

        # Some contexts aren't in the dataset
        if twsi_id not in twsi_id_to_context_str:
          continue

        # Resolve target offset
        context_str = twsi_id_to_context_str[twsi_id]
        target_offsets = list(twsi_id_target_to_offsets[(twsi_id, target_str)])
        assert len(target_offsets) < 2
        if len(target_offsets) == 0:
          target_offsets = index_all(context_str, target_str)
          if len(target_offsets) == 0:
            continue
          elif len(target_offsets) > 1:
            continue
          target_offset = target_offsets[0]
        assert len(target_offsets) == 1
        target_offset = target_offsets[0]

        # Add context
        cid = LexSubDataset.context_id(LexSubDataset.create_context(context_str))
        if not d.has_context(cid):
          extra = []
          for _twsi_id in context_str_to_twsi_ids[context_str]:
            extra.extend(twsi_id_to_extra[_twsi_id])
          d.add_context(context_str, extra=extra)

        # Add target
        tid = LexSubDataset.target_id(LexSubDataset.create_target(cid, target_str, target_offset, pos=Pos.NOUN))
        if not d.has_target(tid):
          d.add_target(cid, target_str, target_offset, pos=Pos.NOUN)

        # Add substitute
        extra = {
          'fp': fp,
          'line_index': i,
          'line': l
        }
        sid = LexSubDataset.substitute_id(LexSubDataset.create_substitute(tid, substitute_str))
        if d.has_substitute(sid):
          labels = d.get_substitute_labels(sid)
          assert all([l == Label.TRUE_IMPLICIT for l in labels])
          count += len(labels)
          extra = d.get_substitute(sid)['extra'] + [extra]
        else:
          extra = [extra]
        d.add_substitute(tid, substitute_str, [Label.TRUE_IMPLICIT] * count, extra=extra, update_ok=True)

        # Track information for implicit negatives
        if include_negatives:
          target_id_to_target_lemma[tid] = target_lemma
          target_lemma_to_all_substitutes[target_lemma].append(substitute_str)

  # Add implicit negatives
  if include_negatives:
    for tid in d.all_target_ids():
      target_lemma = target_id_to_target_lemma[tid]
      for substitute_str in target_lemma_to_all_substitutes[target_lemma]:
        sid = LexSubDataset.substitute_id(LexSubDataset.create_substitute(tid, substitute_str))
        if not d.has_substitute(sid):
          d.add_substitute(tid, substitute_str, [Label.FALSE_IMPLICIT] * count, extra=extra)

  return d

_COINCO_MASC_FN_TO_PATH = None
@lru_cache(maxsize=1000000)
def _coinco_masc_fn_to_text_and_regions(masc_fn):
  from bs4 import BeautifulSoup
  from intervaltree import IntervalTree

  global _COINCO_MASC_FN_TO_PATH
  if _COINCO_MASC_FN_TO_PATH is None:
    _COINCO_MASC_FN_TO_PATH = {}
    with zipfile.ZipFile(ASSETS['coinco_masc_sources']['fp'], 'r') as f:
      for path in f.namelist():
        if path.endswith('.txt'):
          fn = os.path.split(path)[-1]
          assert fn not in _COINCO_MASC_FN_TO_PATH
          _COINCO_MASC_FN_TO_PATH[fn] = path

  masc_txt_path = _COINCO_MASC_FN_TO_PATH[masc_fn]
  masc_txt = file_from_bundle(ASSETS['coinco_masc_sources']['fp'], masc_txt_path).decode('utf-8')
  masc_xml_path = masc_txt_path.replace('.txt', '-s.xml')
  masc_xml = file_from_bundle(ASSETS['coinco_masc_sources']['fp'], masc_xml_path)
  masc_xml = BeautifulSoup(masc_xml, 'xml')

  region_id_to_raw_offset = {}
  raw_regions = []
  for region_offset, region in enumerate(masc_xml.find_all('region')):
    region_id = region.attrs['xml:id']
    region = tuple([int(i) for i in region.attrs['anchors'].split()])
    assert region[1] > region[0]
    # NOTE: Turns [a, b] into [a, b)
    #region = (region[0], region[1] + 1)
    region_id_to_raw_offset[region_id] = region_offset
    raw_regions.append(region)

  tree = IntervalTree.from_tuples(raw_regions)
  tree.merge_overlaps()
  nonoverlapping_regions = sorted([tuple(i)[:2] for i in tree])
  idxs = [item for sublist in nonoverlapping_regions for item in sublist]
  assert idxs == sorted(idxs)

  return masc_txt, region_id_to_raw_offset, raw_regions, nonoverlapping_regions


def _coinco_masc_fn_and_region_id_to_text_and_bounds(masc_fn, region_id):
  txt, region_id_to_raw_offset, raw_regions, nonoverlapping_regions = _coinco_masc_fn_to_text_and_regions(masc_fn)

  raw_region_offset = region_id_to_raw_offset[region_id]
  raw_region = raw_regions[raw_region_offset]

  nonoverlapping_region_offset = None
  for i, (region_start, region_end) in enumerate(nonoverlapping_regions):
    if region_start <= raw_region[0] and raw_region[1] <= region_end:
      nonoverlapping_region_offset = i
      break
  assert nonoverlapping_region_offset is not None

  if nonoverlapping_region_offset - 1 >= 0:
    context_start = nonoverlapping_regions[nonoverlapping_region_offset - 1][0]
  else:
    context_start = nonoverlapping_regions[nonoverlapping_region_offset][0]
  if nonoverlapping_region_offset + 1 < len(nonoverlapping_regions):
    context_end = nonoverlapping_regions[nonoverlapping_region_offset + 1][1]
  else:
    context_end = nonoverlapping_regions[nonoverlapping_region_offset][1]
  context_region = (context_start, context_end)

  assert raw_region[0] >= context_region[0]
  assert raw_region[1] <= context_region[1]

  return txt, raw_region, context_region


_COINCO_DEV_SOURCE_FILES = set("""
Nathans_Bylichka.txt
enron-thread-159550.txt
A1.E2-NEW.txt
A1.E1-NEW.txt
NYTnewswire2.txt
NYTnewswire7.txt
NYTnewswire4.txt
NYTnewswire8.txt
NYTnewswire6.txt
NYTnewswire5.txt
NYTnewswire3.txt
NYTnewswire9.txt
NYTnewswire1.txt
20000410_nyt-NEW.txt
20000419_apw_eng-NEW.txt
20000415_apw_eng-NEW.txt
20000424_nyt-NEW.txt
wsj_0124.txt
wsj_0175.txt
wsj_1640.mrg-NEW.txt
""".strip().splitlines())
# NOTE: Chosen quite arbitrarily
_COINCO_DEV_UNOFFICIAL_VALIDATION_SOURCE_FILES = set("""
enron-thread-159550.txt
A1.E2-NEW.txt
A1.E1-NEW.txt
20000410_nyt-NEW.txt
20000419_apw_eng-NEW.txt
20000415_apw_eng-NEW.txt
20000424_nyt-NEW.txt
wsj_0124.txt
wsj_0175.txt
wsj_1640.mrg-NEW.txt
""".strip().splitlines())
_COINCO_DEV_UNOFFICIAL_TRAIN_SOURCE_FILES = _COINCO_DEV_SOURCE_FILES - _COINCO_DEV_UNOFFICIAL_VALIDATION_SOURCE_FILES
_COINCO_XML = None
_COINCO_ID_TO_MELAMUD_CANDIDATES = None
_COINCO_DEV_IDS = None
_COINCO_TEST_IDS = None
def coinco(
    split='test',
    include_surrounding_context=True,
    repair_context=False,
    include_negatives=True,
    skip_problematic=True):
  # TODO: Add in [FALSE_IMPLICIT] * num_annotators to labels
  from bs4 import BeautifulSoup

  if split not in ['train', 'valid', 'dev', 'test']:
    raise ValueError()

  # Load, parse and cache XML
  global _COINCO_XML
  if _COINCO_XML is None:
    with gzip.open(ASSETS['coinco_patched']['fp'], 'r') as f:
      _COINCO_XML = BeautifulSoup(f.read(), 'xml')

  # Load and cache splits
  global _COINCO_DEV_IDS
  global _COINCO_TEST_IDS
  if _COINCO_DEV_IDS is None:
    assert _COINCO_TEST_IDS is None
    with open(ASSETS['coinco_devids']['fp'], 'r') as f:
      _COINCO_DEV_IDS = set([int(i) for i in f.read().strip().splitlines()])
    with open(ASSETS['coinco_testids']['fp'], 'r') as f:
      _COINCO_TEST_IDS = set([int(i) for i in f.read().strip().splitlines()])
    assert len(_COINCO_DEV_IDS) == 10179
    assert len(_COINCO_TEST_IDS) == 5450
    assert len(_COINCO_DEV_IDS.intersection(_COINCO_TEST_IDS)) == 0

  # Load, parse and cache candidates
  global _COINCO_ID_TO_MELAMUD_CANDIDATES
  if include_negatives and _COINCO_ID_TO_MELAMUD_CANDIDATES is None:
    _COINCO_ID_TO_MELAMUD_CANDIDATES = {}

    with open(ASSETS['coinco_melamud_candidates']['fp'], 'r', encoding='ISO-8859-1') as f:
      lines = f.read().strip().splitlines()
    melamud_k_to_candidates = {}
    for l in lines:
      k, candidates = l.split('::')
      assert k not in melamud_k_to_candidates
      candidates = set([s.strip() for s in candidates.split(';') if len(s.strip()) > 0])
      melamud_k_to_candidates[k] = candidates

    with open(ASSETS['coinco_melamud_preprocessed']['fp'], 'r', encoding='ISO-8859-1') as f:
      lines = f.read().strip().splitlines()
    for l in lines:
      k, coinco_id, _, _ = l.split('\t', 3)
      coinco_id = int(coinco_id)
      if k != '..N':
        assert coinco_id not in _COINCO_ID_TO_MELAMUD_CANDIDATES
        _COINCO_ID_TO_MELAMUD_CANDIDATES[coinco_id] = melamud_k_to_candidates[k]
    assert len(_COINCO_ID_TO_MELAMUD_CANDIDATES) == 15414

  # Create dataset
  d = LexSubDataset(substitutes_lemmatized=True)
  for context_index, context_el in enumerate(_COINCO_XML.find_all('sent')):
    # Determine split
    masc_fn = context_el.attrs['MASCfile']
    if masc_fn in _COINCO_DEV_SOURCE_FILES:
      coinco_split = 'dev'
      unofficial_split = 'train' if masc_fn in _COINCO_DEV_UNOFFICIAL_TRAIN_SOURCE_FILES else 'valid'
    else:
      coinco_split = 'test'
      unofficial_split = 'test'
    if split not in [coinco_split, unofficial_split]:
      continue

    # Align context to original MASC 2.0.0 source
    masc_region_id = context_el.attrs['MASCsentID']
    masc_txt, region_bounds, context_bounds = _coinco_masc_fn_and_region_id_to_text_and_bounds(masc_fn, masc_region_id)

    # Choose to use original context from CoInCo or repaired one from MASC
    if repair_context:
      region = masc_txt[region_bounds[0]:region_bounds[1]]
      if include_surrounding_context:
        context_str = masc_txt[context_bounds[0]:context_bounds[1]]
        region_offset_in_context = region_bounds[0] - context_bounds[0]
      else:
        context_str = region
        region_offset_in_context = 0
    else:
      region = normalize_text(context_el.find('targetsentence').text)
      if include_surrounding_context:
        s_pre = normalize_text(context_el.find('precontext').text)
        s_post = normalize_text(context_el.find('postcontext').text)
        s = []
        if len(s_pre) > 0:
          s.append(s_pre)
        s_raw_i = len(s)
        s.append(region)
        if len(s_post) > 0:
          s.append(s_post)
        context_str = ' '.join(s)
        region_offset_in_context = tokens_offsets(context_str, s)[s_raw_i]
      else:
        context_str = region
        region_offset_in_context = 0
    assert region_offset_in_context >= 0

    # Tokenize
    region_tokens_els = context_el.find_all('token')
    region_tokens = [t.attrs['wordform'].strip() for t in region_tokens_els]
    region_tokens_offsets = tokens_offsets(region, region_tokens)
    # NOTE: CoInCo sometimes has an extra token at the end which we can safely ignore
    assert None not in region_tokens_offsets[:-1]

    # Add context
    cid = LexSubDataset.context_id(LexSubDataset.create_context(context_str))
    extra = {
      'legacy_id': f'{masc_fn}:{masc_region_id}',
      'split': unofficial_split,
      'masc': {
        'document_fn': masc_fn,
        'region_id': masc_region_id,
        'region_bounds': region_bounds,
        'context_bounds': context_bounds,
      },
      'coinco': {
        'split': coinco_split,
        'xml_index': context_index,
        'xml_attrs': context_el.attrs,
        'precontext': context_el.find('precontext').text.strip(),
        'targetsentence': context_el.find('targetsentence').text.strip(),
        'postcontext': context_el.find('postcontext').text.strip(),
      }
    }
    if d.has_context(cid):
      extra = d.get_context(cid)['extra'] + [extra]
    else:
      extra = [extra]
    cid = d.add_context(context_str, extra=extra, update_ok=True)

    for target_index, (target_el, target_offset) in enumerate(zip(region_tokens_els, region_tokens_offsets)):
        substitute_els = target_el.find_all('subst')

        # Skip if target token not in context (due to handful of tokenization errors in CoInCo)
        if target_offset is None:
          assert len(substitute_els) == 0
          continue

        # Skip if target not POS-tagged in MASC
        target_pos_masc = target_el.attrs['posMASC'].strip()
        if target_pos_masc == 'XXX':
          assert len(substitute_els) == 0
          continue
        else:
          # TODO: Is this problematic? I.e., posMASC='XXX' implies no labels collected
          assert len(substitute_els) > 0
        target_pos = PTB_POS_TO_POS[target_pos_masc]

        # Skip if problematic
        if skip_problematic and target_el.attrs['problematic'].strip().lower() == 'yes':
          continue

        # Check split just to be safe
        coinco_id = int(target_el.attrs['id'])
        if coinco_split == 'dev':
          assert coinco_id in _COINCO_DEV_IDS
        else:
          assert coinco_id in _COINCO_TEST_IDS

        # Add target
        target_str = target_el.attrs['wordform'].strip()
        context_offset = region_offset_in_context + target_offset
        if repair_context:
          document_offset = region_bounds[0] + target_offset
          assert masc_txt[document_offset:document_offset+len(target_str)].lower() == target_str.lower()
        else:
          document_offset = None

        tid = LexSubDataset.target_id(LexSubDataset.create_target(cid, target_str, context_offset, pos=target_pos))
        extra = {
          'legacy_id': target_el.attrs['id'],
          'masc': {
            'document_offset': document_offset,
          },
          'coinco': {
            'xml_index': target_index,
            'xml_attrs': target_el.attrs
          }
        }
        if d.has_target(tid):
          extra = d.get_target(tid)['extra'] + [extra]
        else:
          extra = [extra]
        tid = d.add_target(cid, target_str, context_offset, pos=target_pos, extra=extra, update_ok=True)

        for substitute_index, substitute_el in enumerate(substitute_els):
          # Add substitute
          substitute_str = substitute_el.attrs['lemma'].strip()
          num_votes = int(substitute_el.attrs['freq'])
          sid = LexSubDataset.substitute_id(LexSubDataset.create_substitute(tid, substitute_str))
          extra = {
            'legacy_id': substitute_index,
            'coinco': {
                'xml_index': substitute_index,
                'xml_attrs': substitute_el.attrs
            }
          }
          if d.has_substitute(sid):
            labels = d.get_substitute_labels(sid)
            assert all([l == Label.TRUE_IMPLICIT for l in labels])
            num_votes += len(labels)
            extra = d.get_substitute(sid)['extra'] + [extra]
          else:
            extra = [extra]
          labels = [Label.TRUE_IMPLICIT] * num_votes
          sid = d.add_substitute(tid, substitute_str, labels, extra=extra, update_ok=True)

  if include_negatives:
    tid_to_sids = defaultdict(list)
    for sid in d.all_substitute_ids():
      tid_to_sids[d.get_substitute(sid)['target_id']].append(sid)
    for tid in d.all_target_ids():
      target = d.get_target(tid)
      coinco_id = int(target['extra'][-1]['coinco']['xml_attrs']['id'])
      candidates = _COINCO_ID_TO_MELAMUD_CANDIDATES.get(coinco_id, [])
      substitutes = set([d.get_substitute(sid)['substitute'].lower() for sid in tid_to_sids.get(tid, [])])
      for c in candidates:
        if c.lower() not in substitutes:
          sid = LexSubDataset.substitute_id(LexSubDataset.create_substitute(tid, c))
          if d.has_substitute(sid):
            labels = d.get_substitute_labels(sid)
            assert all([l == Label.FALSE_IMPLICIT for l in labels])
          else:
            d.add_substitute(tid, c, [Label.FALSE_IMPLICIT], extra={'source': 'melamud'})

  return d


def swords(
    split='test',
    amt_csv_assets=[],
    deny_list_assets=[],
    strict_num_labels=None,
    ensure_unique_substitute_labelers=False,
    skip_control=True):
  if split not in ['dev', 'test', 'train', 'valid']:
    raise ValueError()
  if ensure_unique_substitute_labelers:
    # TODO
    raise NotImplementedError()
  assert len(amt_csv_assets) == len(set(amt_csv_assets))
  assert len(deny_list_assets) == len(set(deny_list_assets))

  # Create deny lists
  worker_deny_list = set()
  row_deny_list = set()
  substitute_deny_list = set()
  for deny_list_tag in deny_list_assets:
    if type(deny_list_tag) == tuple:
      text = file_from_bundle(ASSETS[deny_list_tag[0]]['fp'], deny_list_tag[1]).decode('utf-8')
      lines = text.strip().splitlines()
    else:
      with open(ASSETS[deny_list_tag]['fp'], 'r') as f:
        lines = f.read().strip().splitlines()

    for l in lines:
      ids = l.split()
      assert len(ids) in [1, 2, 3]
      worker_id = ids[0]
      assert worker_id.startswith('A')
      if len(ids) == 1:
        worker_deny_list.add(worker_id)
      else:
        hit_id = ids[1]
        assert len(hit_id) == 30
        if len(ids) == 2:
          row_deny_list.add((worker_id, hit_id))
        else:
          substitute_deny_list.add((worker_id, hit_id, int(ids[2])))

  # Create dataset
  coinco_d = coinco(split=split, repair_context=True)

  # Parse AMT CSVs
  d = LexSubDataset(substitutes_lemmatized=True)
  row_ids_encountered = set()
  num_labeled = 0
  num_unlabeled = 0
  for csv_tag in amt_csv_assets:
    if type(csv_tag) == tuple:
      text = file_from_bundle(ASSETS[csv_tag[0]]['fp'], csv_tag[1]).decode('utf-8')
      reader = csv.DictReader(StringIO(text))
    else:
      with open(ASSETS[csv_tag]['fp'], 'r') as f:
        reader = csv.DictReader(f)

    for row in reader:
      # Skip row if needed
      worker_id = row['WorkerId'].strip()
      row_id = (worker_id, row['HITId'].strip())
      assert row_id not in row_ids_encountered
      row_ids_encountered.add(row_id)
      if worker_id in worker_deny_list:
        continue
      if row_id in row_deny_list:
        continue

      # Filter out buggy contects from first CSV tag
      if '1109_0_300_results.csv' in csv_tag[-1] and (row['Input.s_left'] + row['Input.w'] + row['Input.s_right']) != row['Input.s']:
        continue

      # Extract target-level info
      cid = row['Input.id']
      if not coinco_d.has_context(cid):
        continue
      assert coinco_d.has_context(cid)
      tid = row['Input.target_id']
      assert coinco_d.has_target(tid)
      assert row['Input.w'] == coinco_d.get_target(tid)['target']
      #feedback = row['Answer.feedback']

      # Extract substitute-level info
      i = -1
      while f'Input.wprime_id_{i+1}' in row:
        i += 1

        # Skip substitute if specified
        substitute_id = row_id + (i,)
        if substitute_id in substitute_deny_list:
          continue

        # Get substitute
        maybe_sid = row[f'Input.wprime_id_{i}']
        substitute_str = row[f'Input.wprime_{i}']
        assert substitute_str.strip() == substitute_str
        if len(substitute_str) == 0:
          continue
        sid = LexSubDataset.substitute_id(LexSubDataset.create_substitute(tid, substitute_str))
        if sid != maybe_sid:
          assert len(maybe_sid) != 42
        source = row[f'Input.source_{i}'].strip()
        if skip_control and source in ['same', 'rand']:
          continue
        # TODO: Restore time zone to PST
        assert ' PST' in row['CreationTime']
        hit_creation_time = int(datetime.strptime(row['CreationTime'].replace(' PST', ''), '%a %b %d %H:%M:%S %Y').timestamp())

        # Get label
        labels = [row[f'Answer.candidate{i}-{l}.on'] for l in ['abstain', 'bad', 'good']]
        assert all([l in ['false', 'true'] for l in labels])
        num_true = labels.count('true')
        if num_true == 0:
          num_unlabeled += 1
          continue
        elif num_true > 1:
          raise Exception()
        label = [Label.UNSURE, Label.FALSE, Label.TRUE][labels.index('true')]
        num_labeled += 1

        # Add context to dataset
        if not d.has_context(cid):
          context = coinco_d.get_context(cid)
          _cid = d.add_context(context['context'], extra=context.get('extra'))
          assert _cid == cid

        # Add target to dataset
        if not d.has_target(tid):
          target = coinco_d.get_target(tid)
          _tid = d.add_target(cid, target['target'], target['offset'], pos=target['pos'], extra=target.get('extra'))
          assert _tid == tid

        # Add substitute to dataset
        extra = {
          'substitute_source': source,
          'label_source': substitute_id,
          'hit_creation_time': hit_creation_time,
        }
        if coinco_d.has_substitute(sid):
          extra['coinco'] = coinco_d.get_substitute(sid)['extra']
          extra['coinco_labels'] = [l.name for l in coinco_d.get_substitute_labels(sid)]

        if d.has_substitute(sid):
          # Update substitute
          labels = d.get_substitute_labels(sid) + [label]
          extra = d.get_substitute(sid)['extra'] + [extra]
        else:
          # Create substitute
          labels = [label]
          extra = [extra]
        assert len(extra) == len(labels)
        _sid = d.add_substitute(tid, substitute_str, labels, extra=extra, update_ok=True)
        assert _sid == sid

  # Warn
  if num_unlabeled > 0:
    warnings.warn(f'{num_unlabeled} / {num_unlabeled + num_labeled} substitutes were unlabeled')

  # Filter substitutes down to N labels
  if strict_num_labels is not None:
    d_strict = LexSubDataset(substitutes_lemmatized=True)
    for sid in d.all_substitute_ids():
      substitute = d.get_substitute(sid)
      labels = d.get_substitute_labels(sid)

      # Skip if we have less than N
      if len(labels) < strict_num_labels:
        continue

      # Grab the most recent N labels if we have more than N
      extra = substitute['extra']
      if len(labels) > strict_num_labels:
        assert len(labels) == len(extra)
        labels_and_extra = list(zip(labels, extra))
        labels_and_extra = sorted(labels_and_extra, key=lambda x: x[1]['hit_creation_time'])
        labels_and_extra = labels_and_extra[-strict_num_labels:]
        labels, extra = zip(*labels_and_extra)
      assert len(extra) == len(labels)
      assert len(labels) == strict_num_labels

      # Add to dataset
      tid = substitute['target_id']
      target = d.get_target(tid)
      cid = target['context_id']
      context = d.get_context(cid)
      if not d_strict.has_context(cid):
        d_strict.add_context(context)
      if not d_strict.has_target(tid):
        d_strict.add_target(target)
      d_strict.add_substitute(tid, substitute['substitute'], labels, extra=extra)

    d = d_strict

  return d


def swords_release(split='test', max_labels=10, reannotated=False, stamp='0526'):
  asset_tag = f'swords_{stamp}'
  if reannotated:
    asset_tag = f'swords_human_{stamp}'
  with open(ASSETS[asset_tag]['fp'], 'rb') as f:
    raw = pickle.load(f)

  subset_tids = None
  if split == 'subset' and not reannotated:
    subset_tids = set(swords_release(split='subset', reannotated=True, stamp=stamp).all_target_ids())

  swords_old_split = 'test' if split == 'subset' else split
  swords_old = get_dataset(f'swords-v0.8_{swords_old_split}')

  d = LexSubDataset(substitutes_lemmatized=True)
  for i, s in enumerate(raw):
    assert s['split'] in ['dev', 'test', 'subset']
    if subset_tids is None and s['split'] != split:
      continue
    if subset_tids is not None and s['target_id'] not in subset_tids:
      continue
    assert len(s['wprime']) > 0
    
    newly_collected = len(s['wprime_id']) == 0
    pos = Pos[s['pos']]
    if stamp == '0607':
      labels = s['labels']
      assert len(labels) in [3, 10]
    else:
      if reannotated:
        labels = s['step3_labels']
        assert len(labels) == 10
      elif len(s['step2_labels']) > 0:
        labels = s['step2_labels']
        assert len(labels) == 10
      else:
        labels = s['step1_labels']
        assert len(labels) == 3
    labels = [Label[l] for l in labels]
    cid = s['id']
    context_old = swords_old.get_context(cid)
    tid = s['target_id']
    target_old = swords_old.get_target(tid)
    sid = LexSubDataset.substitute_id(LexSubDataset.create_substitute(s['target_id'], s['wprime_lemma']))
    
    assert len(s['wprime_id']) in [0, 42]
    if stamp != '0607' and not newly_collected:
      assert swords_old.has_substitute(s['wprime_id'])
      assert s['wprime_id'] == sid
    assert (s['s_left'] + s['w'] + s['s_right']) == s['s']
    assert len(s['s_left']) == s['off']

    # Add context
    coinco_ids = [e['legacy_id'] for e in context_old['extra']]
    if d.has_target(cid):
      coinco_ids = d.get_context(cid)['extra']['coinco_ids'] + coinco_ids
    _cid = d.add_context(s['s'], extra={
      'coinco_ids': coinco_ids
    }, update_ok=True)
    assert _cid == cid

    # Add target
    coinco_ids = [e['legacy_id'] for e in target_old['extra']]
    coinco_lemmas = [e['coinco']['xml_attrs']['lemma'] for e in target_old['extra']]
    assert len(set(coinco_lemmas)) == 1
    coinco_lemma = coinco_lemmas[0]
    assert coinco_lemma == s['w_lemma']
    if d.has_target(tid):
      coinco_ids += d.get_target(tid)['extra']['coinco_ids']
      assert coinco_lemma == d.get_target(tid)['extra']['coinco_lemma']
    coinco_ids = sorted(list(set(coinco_ids)), key=lambda x: int(x))
    _tid = d.add_target(cid, s['w'], s['off'], pos=pos, extra={
      'coinco_ids': coinco_ids,
      'coinco_lemma': coinco_lemma
    }, update_ok=True)
    assert _tid == tid
    
    # Add substitute
    if d.has_substitute(sid):
      labels = d.get_substitute_labels(sid) + labels
    # NOTE: Handful of instances where we have 20 labels due to case-sensitivity (TV and PC)
    if max_labels is not None:
      labels = labels[:max_labels]
    assert s['wprime_lemma'] == s['wprime']
    if stamp == '0607':
      sources = s['wprime_source']
    else:
      sources = s['wprime_source'].split(';')
    _sid = d.add_substitute(tid, s['wprime_lemma'], labels, extra={
      'sources': sorted(sources),
    }, update_ok=True)
    assert _sid == sid

  return d


DATASETS = {
  'semeval07_trial': {
    'create': lambda: semeval07(split='trial'),
    'id': 'd:f1a6ea8db97a827b9c339da41bcbc8ce685457c4'
  },
  'semeval07_test': {
    'create': lambda: semeval07(split='test'),
    'id': 'd:40909f5bd227b2ea81fe4ba0a089de02465abd98'
  },
  'twsi_all': {
    'create': lambda: twsi(split='all'),
    'id': 'd:04348c5b756c5eb60f2f8ab2182d1e2d6a301bda'
  },
  'coinco_dev': {
    'create': lambda: coinco(split='dev'),
    'id': 'd:9e02efb6109b85f2e4e905061a1372d47cfa8076'
  },
  'coinco_dev-train': {
    'create': lambda: coinco(split='train'),
    'id': 'd:8c7c317c54429e378b0ac7cd3f183bd61a96fcb2'
  },
  'coinco_dev-valid': {
    'create': lambda: coinco(split='valid'),
    'id': 'd:4bb0fd60af3ecc5254ebad45c96162f2e551fdf6'
  },
  'coinco_test': {
    'create': lambda: coinco(split='test'),
    'id': 'd:55717224b4846b9e6b1eda575a0e9debce14df26'
  },
  'swords-v0.1_test': {
    'create': lambda: swords(
      split='test',
      amt_csv_assets=[
        ('swords_test_1113', 'test/1109_0_300_results.csv')
      ],
      skip_control=False),
    'id': 'd:17e5b3709108e7ca1f41e003641a9a982f5fad57'
  },
  'swords-v0.2_test': {
    'create': lambda: swords(
      split='test',
      amt_csv_assets=[
        ('swords_test_1113', 'test/1109_0_300_results.csv'),
        ('swords_test_1113', 'test/1109-2_300_600_results.csv'),
      ],
      skip_control=False),
    'id': 'd:67a501316db1057d708e4e9b70aa97c6a32288cd'
  },
  'swords-v0.3_test': {
    'create': lambda: swords(
      split='test',
      amt_csv_assets=[
        ('swords_test_1113', 'test/1109_0_300_results.csv'),
        ('swords_test_1113', 'test/1109-2_300_600_results.csv'),
        ('swords_test_1113', 'test/1109-2_600_-1_results.csv'),
      ],
      skip_control=False),
    'id': 'd:fced620015f4e426c117743d5e659961ff30f322'
  },
  'swords-v0.4_test': {
    'create': lambda: swords(
      split='test',
      amt_csv_assets=[
        ('swords_test_1113', 'test/1109_0_300_results.csv'),
        ('swords_test_1113', 'test/1109-2_300_600_results.csv'),
        ('swords_test_1113', 'test/1109-2_600_-1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect-a3_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect-a3_results.csv'),
      ],
      deny_list_assets=[
        ('swords_test_1113', 'test/1109_0_300_filter.txt'),
        ('swords_test_1113', 'test/1109-2_300_600_filter.txt'),
        ('swords_test_1113', 'test/1109-2_600_-1_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect-a1_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect-a2_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect-a3_filter.txt'),
      ]),
    'id': 'd:81b1e9ee6522e861f225ead134ed2f4e3081caa3'
  },
  'swords-v0.5_dev': {
    'create': lambda: swords(
      split='dev',
      amt_csv_assets=[
        ('swords_dev_1113', 'dev/1112_0_-1_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect-a1_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect-a2_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect_recollect-a1_results.csv'),
      ],
      deny_list_assets=[
        ('swords_dev_1113', 'dev/1112_filter.txt'),
        ('swords_dev_1113', 'dev/1112_recollect_filter.txt'),
      ]),
    'id': 'd:ea1d4323620fa05fe8b3c0c3b6db36383eaf70f2'
  },
  'swords-v0.5_test': {
    'create': lambda: swords(
      split='test',
      amt_csv_assets=[
        ('swords_test_1113', 'test/1109_0_300_results.csv'),
        ('swords_test_1113', 'test/1109-2_300_600_results.csv'),
        ('swords_test_1113', 'test/1109-2_600_-1_results.csv'),
        ('swords_test_1113', 'test/1109_collect_missing_hits_results.csv'),
        ('swords_test_1113', 'test/1109_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect-a3_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect-a3_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect_recollect_recollect-a1_results.csv'),
      ],
      deny_list_assets=[
        ('swords_test_1113', 'test/1109_0_300_filter.txt'),
        ('swords_test_1113', 'test/1109-2_300_600_filter.txt'),
        ('swords_test_1113', 'test/1109-2_600_-1_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect-a1_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect-a2_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect-a3_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect_recollect_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect_recollect_filter.txt'),
      ]),
    'id': 'd:2e82153f79a298636639cc9c9a9047b3c35b6bac'
  },
  'swords-v0.6_dev': {
    'create': lambda: swords(
      split='dev',
      amt_csv_assets=[
        ('swords_dev_1113', 'dev/1112_0_-1_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect-a1_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect-a2_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect_recollect-a1_results.csv'),
      ],
      deny_list_assets=[
        ('swords_dev_1113', 'dev/1112_filter.txt'),
        ('swords_dev_1113', 'dev/1112_recollect_filter.txt'),
      ],
      strict_num_labels=3),
    'id': 'd:3092286787cb832536b3adfcbe72b5282ae4fdc2'
  },
  'swords-v0.6_test': {
    'create': lambda: swords(
      split='test',
      amt_csv_assets=[
        ('swords_test_1113', 'test/1109_0_300_results.csv'),
        ('swords_test_1113', 'test/1109-2_300_600_results.csv'),
        ('swords_test_1113', 'test/1109-2_600_-1_results.csv'),
        ('swords_test_1113', 'test/1109_collect_missing_hits_results.csv'),
        ('swords_test_1113', 'test/1109_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect-a3_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect-a3_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect_recollect_recollect-a1_results.csv'),
      ],
      deny_list_assets=[
        ('swords_test_1113', 'test/1109_0_300_filter.txt'),
        ('swords_test_1113', 'test/1109-2_300_600_filter.txt'),
        ('swords_test_1113', 'test/1109-2_600_-1_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect-a1_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect-a2_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect-a3_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect_recollect_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect_recollect_filter.txt'),
      ],
      strict_num_labels=3),
    'id': 'd:d5a10261c4208e515972b2f94056636cb04846c2'
  },
  'swords-v0.7_dev': {
    'create': lambda: swords(
      split='dev',
      amt_csv_assets=[
        # Step 2
        ('swords_dev_1113', 'dev/1112_0_-1_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect-a1_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect-a2_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect_recollect-a1_results.csv'),
        # Step 3
        ('swords_step3_1117', 'step 3/dev/1115_dev_results.csv'),
      ],
      deny_list_assets=[
        # Spam workers
        ('swords_test_1113', 'test/worker.txt'),
        ('swords_step3_1117', 'step 3/worker.txt'),
        # Spam HITS
        ('swords_dev_1113', 'dev/1112_filter.txt'),
        ('swords_dev_1113', 'dev/1112_recollect_filter.txt'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_filter.txt'),
      ]),
    'id': 'd:5d6c2b3e350962ee96b807f877a883a14f75be0d'
  },
  'swords-v0.7_test': {
    'create': lambda: swords(
      split='test',
      amt_csv_assets=[
        # Step 2
        ('swords_test_1113', 'test/1109_0_300_results.csv'),
        ('swords_test_1113', 'test/1109-2_300_600_results.csv'),
        ('swords_test_1113', 'test/1109-2_600_-1_results.csv'),
        ('swords_test_1113', 'test/1109_collect_missing_hits_results.csv'),
        ('swords_test_1113', 'test/1109_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect-a3_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect-a3_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect_recollect_recollect-a1_results.csv'),
        # Step 3
        ('swords_step3_1117', 'step 3/test/1115_test_results.csv'),
      ],
      deny_list_assets=[
        # Spam workers
        ('swords_test_1113', 'test/worker.txt'),
        ('swords_step3_1117', 'step 3/worker.txt'),
        # Spam HITS
        ('swords_test_1113', 'test/1109_0_300_filter.txt'),
        ('swords_test_1113', 'test/1109-2_300_600_filter.txt'),
        ('swords_test_1113', 'test/1109-2_600_-1_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect-a1_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect-a2_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect-a3_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect_recollect_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect_recollect_filter.txt'),
        ('swords_step3_1117', 'step 3/test/1115_test_filter.txt'),
      ]),
      'id': 'd:621280234fe229faabef756e46e1fe4ee62d4f01'
  },
  'swords-v0.8_dev': {
    'create': lambda: swords(
      split='dev',
      amt_csv_assets=[
        # Step 2
        ('swords_dev_1113', 'dev/1112_0_-1_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect-a1_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect-a2_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect_recollect-a1_results.csv'),
        # Step 3
        ('swords_step3_1117', 'step 3/dev/1115_dev_results.csv'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_recollect-a1_results.csv'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_recollect-a2_results.csv'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_recollect-a3_results.csv'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_recollect-a4_results.csv'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_recollect-a6_results.csv'),
      ],
      deny_list_assets=[
        # Spam workers
        ('swords_test_1113', 'test/worker.txt'),
        ('swords_step3_1117', 'step 3/worker.txt'),
        # Spam HITS
        ('swords_dev_1113', 'dev/1112_filter.txt'),
        ('swords_dev_1113', 'dev/1112_recollect_filter.txt'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_filter.txt'),
      ]),
    'id': 'd:7bc0e0b6f47f0ce2cc18c4978682b5e8ee264fd7'
  },
  'swords-v0.8_dev-train': {
    'create': lambda: swords(
      split='train',
      amt_csv_assets=[
        # Step 2
        ('swords_dev_1113', 'dev/1112_0_-1_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect-a1_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect-a2_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect_recollect-a1_results.csv'),
        # Step 3
        ('swords_step3_1117', 'step 3/dev/1115_dev_results.csv'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_recollect-a1_results.csv'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_recollect-a2_results.csv'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_recollect-a3_results.csv'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_recollect-a4_results.csv'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_recollect-a6_results.csv'),
      ],
      deny_list_assets=[
        # Spam workers
        ('swords_test_1113', 'test/worker.txt'),
        ('swords_step3_1117', 'step 3/worker.txt'),
        # Spam HITS
        ('swords_dev_1113', 'dev/1112_filter.txt'),
        ('swords_dev_1113', 'dev/1112_recollect_filter.txt'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_filter.txt'),
      ]),
    'id': 'd:83cabe9983c86bb2424a89e20b7a832d7e72fc12'
  },
  'swords-v0.8_dev-valid': {
    'create': lambda: swords(
      split='valid',
      amt_csv_assets=[
        # Step 2
        ('swords_dev_1113', 'dev/1112_0_-1_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect-a1_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect-a2_results.csv'),
        ('swords_dev_1113', 'dev/1112_recollect_recollect-a1_results.csv'),
        # Step 3
        ('swords_step3_1117', 'step 3/dev/1115_dev_results.csv'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_recollect-a1_results.csv'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_recollect-a2_results.csv'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_recollect-a3_results.csv'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_recollect-a4_results.csv'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_recollect-a6_results.csv'),
      ],
      deny_list_assets=[
        # Spam workers
        ('swords_test_1113', 'test/worker.txt'),
        ('swords_step3_1117', 'step 3/worker.txt'),
        # Spam HITS
        ('swords_dev_1113', 'dev/1112_filter.txt'),
        ('swords_dev_1113', 'dev/1112_recollect_filter.txt'),
        ('swords_step3_1117', 'step 3/dev/1115_dev_filter.txt'),
      ]),
    'id': 'd:d07d376d966d6fba41143015704c6bdd2831eeb6'
  },
  'swords-v0.8_test': {
    'create': lambda: swords(
      split='test',
      amt_csv_assets=[
        # Step 2
        ('swords_test_1113', 'test/1109_0_300_results.csv'),
        ('swords_test_1113', 'test/1109-2_300_600_results.csv'),
        ('swords_test_1113', 'test/1109-2_600_-1_results.csv'),
        ('swords_test_1113', 'test/1109_collect_missing_hits_results.csv'),
        ('swords_test_1113', 'test/1109_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect-a3_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect-a3_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect_recollect-a1_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect_recollect-a2_results.csv'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect_recollect_recollect-a1_results.csv'),
        # Step 3
        ('swords_step3_1117', 'step 3/test/1115_test_results.csv'),
        ('swords_step3_1117', 'step 3/test/1115_test_recollect-a1_results.csv'),
        ('swords_step3_1117', 'step 3/test/1115_test_recollect-a2_results.csv'),
        ('swords_step3_1117', 'step 3/test/1115_test_recollect-a3_results.csv'),
      ],
      deny_list_assets=[
        # Spam workers
        ('swords_test_1113', 'test/worker.txt'),
        ('swords_step3_1117', 'step 3/worker.txt'),
        # Spam HITS
        ('swords_test_1113', 'test/1109_0_300_filter.txt'),
        ('swords_test_1113', 'test/1109-2_300_600_filter.txt'),
        ('swords_test_1113', 'test/1109-2_600_-1_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect-a1_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect-a2_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect-a3_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect_recollect_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_filter.txt'),
        ('swords_test_1113', 'test/1109_recollect_recollect_recollect_recollect_recollect_filter.txt'),
        ('swords_step3_1117', 'step 3/test/1115_test_filter.txt'),
      ]),
    'id': 'd:df8b84ad44d6fc27ff7db9602cf90a70300cd3a3'
  },
  'swords-v0.8-subset_test': {
    'id': 'd:ea901f1adfe7a766cef7445997183aa9c99fb476',
  },
  'swords-v0.8-subset-human-baseline_test': {
    'create': lambda: swords(
      split='test',
      amt_csv_assets=[
        ('swords_human_1119', 'superhuman/1119_test_superhuman_results.csv'),
        ('swords_human_1119', 'superhuman/1119_test_superhuman_recollect-a1_results.csv'),
        ('swords_human_1119', 'superhuman/1119_test_superhuman_recollect-a2_results.csv'),
        ('swords_human_1119', 'superhuman/1119_test_superhuman_recollect-a3_results.csv'),
        ('swords_human_1119', 'superhuman/1119_test_superhuman_recollect-a4_results.csv'),
        ('swords_human_1119', 'superhuman/1119_test_superhuman_recollect-a5_results.csv'),
        ('swords_human_1119', 'superhuman/1119_test_superhuman_recollect-a6_results.csv'),
        ('swords_human_1119', 'superhuman/1119_test_superhuman_recollect-a8_results.csv'),
      ],
      deny_list_assets=[
        ('swords_human_1119', 'superhuman/1119_test_superhuman_filter.txt'),
      ]),
    'id': 'd:8df8a78fa8e8b67325209cc10fb5b506da3f7d0f'
  },
  'swords-v0.9.0_dev': {
    'create': lambda: swords_release(split='dev', max_labels=None),
    'id': 'd:16697c27d513aa2637045ae177f51efa79de4e9a'
  },
  'swords-v0.9.0_test': {
    'create': lambda: swords_release(split='test', max_labels=None),
    'id': 'd:3676d10c2eb58e716505fe0c8840750627e7268a'
  },
  'swords-v0.9.0_test-subset': {
    'create': lambda: swords_release(split='subset', max_labels=None),
    'id': 'd:4525bb65d38147660e5b5eb1282ee555e5aff934'
  },
  'swords-v0.9.0_test-subset_reannotated': {
    'create': lambda: swords_release(split='subset', max_labels=None, reannotated=True),
    'id': 'd:f228b059eea7e0b2513b672fa651d20b33278f11'
  },
  'swords-v0.9.1_dev': {
    'create': lambda: swords_release(split='dev'),
    'id': 'd:16697c27d513aa2637045ae177f51efa79de4e9a'
  },
  'swords-v0.9.1_test': {
    'create': lambda: swords_release(split='test'),
    'id': 'd:018c2d54e042d91a38f33a027009e08a38e7aba8'
  },
  'swords-v0.9.1_test-subset': {
    'create': lambda: swords_release(split='subset'),
    'id': 'd:50396a093da867b578160049cead448514b339b6'
  },
  'swords-v0.9.1_test-subset_reannotated': {
    'create': lambda: swords_release(split='subset', reannotated=True),
    'id': 'd:9f438a458cd5a07a4fe1f186c1189076dc1e5333'
  },
  'swords-v1.0_dev': {
    'create': lambda: swords_release(split='dev'),
    'id': 'd:16697c27d513aa2637045ae177f51efa79de4e9a'
  },
  'swords-v1.0_test': {
    'create': lambda: swords_release(split='test'),
    'id': 'd:018c2d54e042d91a38f33a027009e08a38e7aba8'
  },
  'swords-v1.0_test-subset': {
    'create': lambda: swords_release(split='subset'),
    'id': 'd:50396a093da867b578160049cead448514b339b6'
  },
  'swords-v1.0_test-subset_reannotated': {
    'create': lambda: swords_release(split='subset', reannotated=True),
    'id': 'd:9f438a458cd5a07a4fe1f186c1189076dc1e5333'
  },
  'swords-v1.1_dev': {
    'create': lambda: swords_release(split='dev', stamp='0607'),
    'id': 'd:651209c901f26925691f9a8ceb2c8da1054cded8'
  },
  'swords-v1.1_test': {
    'create': lambda: swords_release(split='test', stamp='0607'),
    'id': 'd:76d1aba7700f7d9f9a7ddffe73b8f0e1cb2925fa'
  },
  'swords-v1.1_test-subset': {
    'create': lambda: swords_release(split='subset', stamp='0607'),
    'id': 'd:4df7c9ce25c4abb0d3d42c0ecc70e6e3a96f3932'
  },
  'swords-v1.1_test-subset_reannotated': {
    'create': lambda: swords_release(split='subset', reannotated=True, stamp='0607'),
    'id': 'd:acf8b87a1046f6fe7ec7eae30f129ccbd2c1cc1f'
  }
}


def _create_swords_subset():
  original_dataset = get_dataset('swords-v0.8_test')
  reannotated_subset = get_dataset('swords-v0.8-subset-human-baseline_test')

  original_subset = LexSubDataset(substitutes_lemmatized=False)
  for tid in reannotated_subset.all_target_ids():
    reannotated_subset_sids = reannotated_subset.all_substitute_ids(target_id=tid)
    original_dataset_sids = original_dataset.all_substitute_ids(target_id=tid)
    assert frozenset(reannotated_subset_sids) == frozenset(original_dataset_sids)

    target = original_dataset.get_target(tid)
    context = original_dataset.get_context(target['context_id'])
    original_subset.add_context(context)
    original_subset.add_target(target)
    for sid in original_dataset.all_substitute_ids(target_id=tid):
      original_subset.add_substitute(
          original_dataset.get_substitute(sid),
          original_dataset.get_substitute_labels(sid))

  return original_subset

DATASETS['swords-v0.8-subset_test']['create'] = _create_swords_subset


for tag, attrs in DATASETS.items():
  DATASETS[tag]['fp'] = os.path.join(DATASETS_CACHE_DIR, f'{tag}.json.gz')
DATASETS['swords-latest_test'] = DATASETS['swords-v1.1_test']
DATASETS['swords-latest_dev'] = DATASETS['swords-v1.1_dev']


def get_dataset(dataset, ignore_cache=False, verbose=False):
  def _print(x):
    if verbose:
      print(x)

  if dataset in DATASETS:
    attrs = DATASETS[dataset]

    # Read from cache
    fp = attrs['fp']
    if not ignore_cache:
      if os.path.exists(fp):
        _print(f'Verifying file {fp}')
        try:
          with gzip.open(fp, 'r') as f:
            d = LexSubDataset.from_dict(json.load(f))
          assert d.id() == attrs['id']
          _print('Verified')
          return d
        except:
          _print('Bad id... regenerating')

    # Generate
    _print('Generating')
    d = attrs['create']()
    if d.id() == attrs['id']:
      _print('Verified!')
    else:
      if len(attrs['id'].strip()) == 0:
        _print(d.id())
      raise Exception('Bad id')

    # Write to cache
    try:
      os.makedirs(DATASETS_CACHE_DIR, exist_ok=True)
      with gzip.open(fp, 'wt') as f:
        f.write(json.dumps(d.as_dict(), indent=2))
    except:
      _print('Failed to save to cache')
  elif os.path.exists(dataset):
    # Load from disk
    _print('Loading from {dataset}')
    if dataset.endswith('.gz'):
      open_fn = gzip.open
    else:
      open_fn = open
    with open_fn(dataset, 'r') as f:
      d = LexSubDataset.from_dict(json.load(f))
  else:
    raise ValueError('Unknown dataset')

  assert isinstance(d, LexSubDataset)
  return d


def main(argv):
  if len(argv) >= 1:
    tags = [d for d in DATASETS.keys() if d.startswith(argv[0].strip())]
  else:
    tags = DATASETS.keys()

  errors = 0
  for tag in tags:
    print('-' * 80)
    print(tag)
    try:
      get_dataset(tag, ignore_cache=False, verbose=True)
    except:
      trace = traceback.format_exc()
      print(trace)
      errors += 1

  if errors > 0:
    raise Exception('Failed to build one or more datasets')
