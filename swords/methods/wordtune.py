import json
import time

from .. import LexSubDataset
from ..assets import ASSETS
from . import LexSubGenerator

def _request_with_429_retry(
    *args,
    method='post',
    initial_backoff=1,
    backoff_multiplier=2,
    timeout=None,
    **kwargs):
  import requests
  s = time.time()
  backoff = initial_backoff
  while True:
    r = getattr(requests, method)(*args, timeout=timeout, **kwargs)
    if timeout is not None and (time.time() - s) > timeout:
      raise requests.exceptions.Timeout('Timeout')
    if r.status_code == 429:
      time.sleep(backoff)
      backoff *= backoff_multiplier
    else:
      break
  return r


class _Wordtune(LexSubGenerator):
  def __init__(self, endpoint, *args, tid_to_cached_output=None, timeout=60., **kwargs):
    super().__init__(*args, **kwargs)
    if endpoint not in ['clues', 'refine']:
      raise ValueError('Unknown endpoint')
    try:
      with open(ASSETS['methods_wordtune_headers']['fp'], 'r') as f:
        self.__headers = json.load(f)
    except:
        self.__headers = None
    self.endpoint = endpoint
    self.timeout = timeout
    self.tid_to_cached_output = tid_to_cached_output

  def substitutes_will_be_lemmatized(self):
    return False

  def raw_api_result(self, context, target, target_offset):
    if self.endpoint == 'clues':
      # Find (infill) first
      data = {
        'selection': {
          'start': target_offset,
          'wholeText': context[:target_offset] + '*' + context[target_offset+len(target):]
        }
      }
      r = _request_with_429_retry(
          'https://api.wordtune.com/find',
          headers=self.__headers,
          data=json.dumps(data),
          timeout=self.timeout)
      r.raise_for_status()
      d = r.json()
      interaction_id = d['interactionId']

      # Clues (refine with target)
      data = {
        'startIndex': target_offset,
        'endIndex': target_offset+len(target),
        'text': context
      }
      r = _request_with_429_retry(
          f'https://api.wordtune.com/find/{interaction_id}/clues',
          headers=self.__headers,
          data=json.dumps(data),
          timeout=self.timeout)
    elif self.endpoint == 'refine':
      data = {
        'selection': {
          'wholeText': context,
        }
      }
      r = _request_with_429_retry(
          'https://api.wordtune.com/refine',
          headers=self.__headers,
          data=json.dumps(data),
          timeout=self.timeout)
      raise NotImplementedError()
    else:
      raise ValueError()
    r.raise_for_status()
    d = r.json()
    return d

  def parse_raw_api_result(self, d):
    suggestions = []
    if self.endpoint == 'clues':
      for suggestion in d['drilldown']['suggestions']:
        suggestions.append((suggestion['suggestion'], suggestion['score']))
    elif self.endpoint == 'refine':
      raise NotImplementedError()
    else:
      raise ValueError()
    return sorted(suggestions, key=lambda x: -x[1])

  def generate(self, context, target, target_offset, target_pos=None):
    cid = LexSubDataset.context_id(LexSubDataset.create_context(context))
    tid = LexSubDataset.target_id(LexSubDataset.create_target(cid, target, target_offset, pos=target_pos))
    if self.tid_to_cached_output is not None:
      wordtune_output = self.tid_to_cached_output[tid]
    else:
      wordtune_output = self.raw_api_result(context, target, target_offset)
    return self.parse_raw_api_result(wordtune_output)


class WordtuneClues(_Wordtune):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, endpoint='clues', **kwargs)


class WordtuneRefine(_Wordtune):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, endpoint='refine', **kwargs)
