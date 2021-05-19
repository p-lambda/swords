from functools import lru_cache
import re

def index_all(context, w, case_sensitive=False):
  flags = 0 if case_sensitive else re.IGNORECASE
  return [m.start() for m in re.finditer(re.escape(w), context, flags=flags)]


def normalize_text(text):
  return ' '.join(text.strip().split())


_NLTK_WORD_TOKENIZER = None
def nltk_tokenize(x):
  global _NLTK_WORD_TOKENIZER
  if _NLTK_WORD_TOKENIZER is None:
    from nltk.tokenize.treebank import TreebankWordTokenizer
    _NLTK_WORD_TOKENIZER = TreebankWordTokenizer()
  x_tokens = _NLTK_WORD_TOKENIZER.tokenize(x)
  x_tokens_offsets = tokens_offsets(x, x_tokens)
  for i, off in enumerate(x_tokens_offsets):
    if off is None and '\"' in x and (x_tokens[i] == '``' or x_tokens[i] == '\'\''):
      x_tokens[i] = '\"'
  return x_tokens


_NLTK_WORD_DETOKENIZER = None
def nltk_detokenize(x_tokens):
  global _NLTK_WORD_DETOKENIZER
  if _NLTK_WORD_DETOKENIZER is None:
    from nltk.tokenize.treebank import TreebankWordDetokenizer
    _NLTK_WORD_DETOKENIZER = TreebankWordDetokenizer()
  return _NLTK_WORD_DETOKENIZER.detokenize(x_tokens)


@lru_cache(maxsize=128)
def _tokens_offsets_and_residuals_memoized(x, x_tok):
  x_remaining_off = 0
  x_remaining = x[:]

  offsets = []
  residuals = []

  for i, t in enumerate(x_tok):
    if len(t) == 0:
      warnings.warn('Encountered empty token')

    try:
      t_off_in_x_remaining = x_remaining.index(t)
      t_res = x_remaining[:t_off_in_x_remaining]
      t_off = x_remaining_off + t_off_in_x_remaining
    except:
      t_off = None
      t_res = ''

    offsets.append(t_off)
    residuals.append(t_res)

    if t_off is not None:
      trim = t_off_in_x_remaining + len(t)
      x_remaining_off += trim
      x_remaining = x_remaining[trim:]

  rres = x_remaining

  return offsets, residuals, rres


def tokens_offsets(x, x_tok):
  if type(x_tok) != tuple:
    x_tok = tuple(x_tok)
  return _tokens_offsets_and_residuals_memoized(x, x_tok)[0]


def tokens_residuals(x, x_tok):
  if type(x_tok) != tuple:
    x_tok = tuple(x_tok)
  return _tokens_offsets_and_residuals_memoized(x, x_tok)[1:]
