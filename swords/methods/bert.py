from functools import lru_cache
import numpy as np

from . import LexSubGenerator, LexSubRanker, LexSubWithDelemmatizationRanker

# NOTE: Sharing one copy of bert-large-uncased across methods that use it
_PRETRAINED_BERT_SINGLETON = None
class BertBasedMethod:
  def __init__(self):
    global _PRETRAINED_BERT_SINGLETON
    if _PRETRAINED_BERT_SINGLETON is None:
      import torch
      from transformers import BertForMaskedLM, BertTokenizer, logging as transformers_logging
      from .util import torch_device_gpu_if_available

      device = torch_device_gpu_if_available()

      transformers_logging.set_verbosity_error()
      model = BertForMaskedLM.from_pretrained('bert-large-uncased')
      model.eval()
      model.to(device)

      tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

      tokens = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
      vocab = [int(t.isalnum()) for t in tokens]
      vocab_mask = torch.tensor(vocab, dtype=torch.uint8, device=device, requires_grad=False)

      _PRETRAINED_BERT_SINGLETON = (device, model, tokenizer, vocab_mask)

    self.device, self.bert_mlm, self.tokenizer, self.vocab_mask = _PRETRAINED_BERT_SINGLETON
    self.bert = self.bert_mlm.bert

  def list_to_tensor(self, l):
    import torch
    return torch.tensor(l).unsqueeze(0).to(self.device)

  @lru_cache(maxsize=1024)
  def encode(self, context, target, target_offset):
    target_tok_len = len(self.tokenizer.encode(target, add_special_tokens=False))
    if target_tok_len == 0:
      raise ValueError('Target cannot be empty')
    context_target_enc = self.tokenizer.encode(context)
    context_prefix_enc = self.tokenizer.encode(context[:target_offset], add_special_tokens=False)
    target_tok_off = len(context_prefix_enc) + 1
    target_hat = self.tokenizer.decode(context_target_enc[target_tok_off:target_tok_off+target_tok_len])
    if ''.join(target_hat.lower().split()) != ''.join(target.lower().split()):
      raise Exception('Not equal: "{}" "{}"'.format(''.join(target_hat.lower().split()), ''.join(target.lower().split())))
    return context_target_enc, target_tok_off, target_tok_len

  def collapse_span_to_longest_constituent(self, token_ids):
    if len(token_ids) == 0:
      raise ValueError()
    elif len(token_ids) == 1:
      return token_ids
    else:
      tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
      tokens = [t[2:] if t.startswith('##') else t for t in tokens]
      longest_token = max(tokens, key=len)
      return [token_ids[tokens.index(longest_token)]]


class _BertContextualSimilarityRankerImpl(BertBasedMethod):
  def _contextual_similarity_of_substitute(self, context, target, target_offset, substitute):
    from scipy.spatial.distance import cosine as cosine_distance
    import torch

    # Encode context
    context_target_enc, target_tok_off, target_tok_len = self.encode(context, target, target_offset)

    # Compute substitute_tok_len
    substitute_enc = self.tokenizer.encode(substitute, add_special_tokens=False)

    with torch.no_grad():
      context_substitute_enc = context_target_enc[:target_tok_off] + substitute_enc + context_target_enc[target_tok_off+target_tok_len:]

      context_target_outputs, _ = self.bert(self.list_to_tensor(context_target_enc))
      context_substitute_outputs, _ = self.bert(self.list_to_tensor(context_substitute_enc))

      context_target_emb = context_target_outputs[0, target_tok_off:target_tok_off+target_tok_len].mean(dim=0).cpu().numpy()
      context_substitute_emb = context_substitute_outputs[0, target_tok_off:target_tok_off+len(substitute_enc)].mean(dim=0).cpu().numpy()

    return 1 - cosine_distance(context_target_emb, context_substitute_emb)


class BertContextualSimilarityRanker(LexSubRanker, _BertContextualSimilarityRankerImpl):
  def rank(self, context, target, target_offset, substitute, substitute_lemmatized, target_pos=None):
    return self._contextual_similarity_of_substitute(context, target, target_offset, substitute)


class BertContextualSimilarityWithDelemmatizationRanker(LexSubWithDelemmatizationRanker, _BertContextualSimilarityRankerImpl):
  def rank_delemmatized(self, context, target, target_offset, substitute, target_pos=None):
    return self._contextual_similarity_of_substitute(context, target, target_offset, substitute)


class _BertBasedLexSubProposalImpl(BertBasedMethod):
  def __init__(
      self,
      *args,
      dropout_p=0.3,
      dropout_seed=None,
      **kwargs):
    super().__init__(*args, **kwargs)

    self.dropout_p = dropout_p
    self.dropout_seed = dropout_seed

  def _logits_for_dropout_target(self, context_target_enc, target_tok_off, target_tok_len):
    import torch
    import torch.nn.functional as F

    # Find longest word piece
    if target_tok_len > 1:
      target_tokens = self.collapse_span_to_longest_constituent(
          context_target_enc[target_tok_off:target_tok_off+target_tok_len])
      context_target_enc = context_target_enc[:]
      context_target_enc[target_tok_off:target_tok_off+target_tok_len] = target_tokens
      target_tok_len = len(target_tokens)
    assert target_tok_len == 1

    # Apply dropout to target token embeddings
    with torch.no_grad():
      embeddings = self.bert.embeddings(self.list_to_tensor(context_target_enc))
      if self.dropout_seed is not None:
        torch.manual_seed(self.dropout_seed)
      embeddings_with_dropout = F.dropout(embeddings, p=self.dropout_p, training=True)
      embeddings = torch.where(
        (torch.arange(len(context_target_enc)).view(1, -1, 1) == target_tok_off).to(self.device),
        embeddings_with_dropout,
        embeddings)
      logits = self.bert_mlm(inputs_embeds=embeddings)[0][0, target_tok_off]

    return logits, context_target_enc[target_tok_off]


_BERT_MASK_TOKEN = 103
class BertInfillingGenerator(LexSubGenerator, _BertBasedLexSubProposalImpl):
  def __init__(
      self,
      *args,
      target_corruption='mask',
      top_k=None,
      nucleus_p=None,
      **kwargs):
    import torch

    if target_corruption not in ['mask', 'dropout']:
      raise ValueError()

    super().__init__(*args, **kwargs)

    self.target_corruption = target_corruption
    self.top_k = top_k
    self.nucleus_p = nucleus_p

  def substitutes_will_be_lemmatized(self):
    return False

  def generate(self, context, target, target_offset, target_pos=None):
    import torch
    import torch.nn.functional as F

    # Encode context
    context_target_enc, target_tok_off, target_tok_len = self.encode(context, target, target_offset)

    with torch.no_grad():
      if self.target_corruption == 'mask':
        # Replace target with mask token
        masked_context_enc = context_target_enc[:target_tok_off] + [_BERT_MASK_TOKEN] + context_target_enc[target_tok_off+target_tok_len:]
        logits = self.bert_mlm(self.list_to_tensor(masked_context_enc))[0][0, target_tok_off]
        target_enc = context_target_enc[target_tok_off]
      elif self.target_corruption == 'dropout':
        logits, target_enc = self._logits_for_dropout_target(context_target_enc, target_tok_off, target_tok_len)

      # Mask invalid logits from vocabulary
      logits = torch.where(self.vocab_mask, logits, torch.full_like(logits, float('-inf')))
      if target_tok_len == 1:
        logits[target_enc] = float('-inf')
      probs = F.softmax(logits, dim=-1).cpu().numpy()

    tokens = []
    scores = []
    cum_prob = 0.
    for i in np.argsort(-probs):
      if self.nucleus_p is not None and cum_prob >= self.nucleus_p:
        break
      if self.top_k is not None and len(tokens) >= self.top_k:
        break
      prob = probs[i]
      if prob <= 0.:
        break
      tokens.append(i)
      scores.append(prob)
      cum_prob += prob
    tokens = self.tokenizer.convert_ids_to_tokens(tokens)

    return list(zip(tokens, scores))


class _BertBasedLexSubRankerImpl(_BertBasedLexSubProposalImpl):
  def __init__(
      self,
      *args,
      dropout_p=0.3,
      dropout_seed=None,
      num_embedding_layers=4,
      include_final_embedding_layer=False,
      compute_validation_score=True,
      compute_proposal_score=True,
      alpha=0.01,
      attention_dim=2,
      **kwargs):
    super().__init__(*args, dropout_p=dropout_p, dropout_seed=dropout_seed, **kwargs)
    self.num_embedding_layers = num_embedding_layers
    self.include_final_embedding_layer = include_final_embedding_layer
    self.compute_validation_score = compute_validation_score
    self.compute_proposal_score = compute_proposal_score
    self.alpha = alpha
    # TODO: remove
    self.attention_dim = attention_dim

  def proposal_score(self, context_target_enc, target_tok_off, target_tok_len, substitute_enc):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
      # Compute probs
      logits, target_enc = self._logits_for_dropout_target(context_target_enc, target_tok_off, target_tok_len)
      logits = torch.where(self.vocab_mask, logits, torch.full_like(logits, float('-inf')))
      probs = F.softmax(logits, dim=-1)

      # Compute score
      # Eq. 1 from https://www.aclweb.org/anthology/P19-1328.pdf
      score = torch.log(probs[substitute_enc[0]]) - torch.log(1 - probs[target_enc])
      return score.item()

  def validation_score(self, context_target_enc, target_tok_off, target_tok_len, substitute_enc):
    import torch
    import torch.nn.functional as F

    # Collapse multi-wordpiece spans to a single wordpiece
    # TODO: Handle >1 wordpiece?
    target_enc = self.collapse_span_to_longest_constituent(
        context_target_enc[target_tok_off:target_tok_off+target_tok_len])
    substitute_enc = self.collapse_span_to_longest_constituent(substitute_enc)

    with torch.no_grad():
      # Create contexts
      co_enc = context_target_enc[:target_tok_off] + target_enc + context_target_enc[target_tok_off+target_tok_len:]
      cs_enc = context_target_enc[:target_tok_off] + substitute_enc + context_target_enc[target_tok_off+target_tok_len:]
      assert len(co_enc) == len(cs_enc)

      # Eq. 2 from https://www.aclweb.org/anthology/P19-1328.pdf
      _, _, co_hidden_states, co_attentions = self.bert(self.list_to_tensor(co_enc), output_attentions=True, output_hidden_states=True)
      _, _, cs_hidden_states = self.bert(self.list_to_tensor(cs_enc), output_hidden_states=True)

      # Compute attention scores
      co_attentions = torch.cat(co_attentions, dim=1).mean(dim=1)
      if self.attention_dim == 1:
        co_attentions = co_attentions[:, target_tok_off]
      else:
        co_attentions = co_attentions[:, :, target_tok_off]

      # Compute cosine similarities
      if self.include_final_embedding_layer:
        co_h = torch.cat(co_hidden_states[-self.num_embedding_layers:], dim=2)
        cs_h = torch.cat(cs_hidden_states[-self.num_embedding_layers:], dim=2)
      else:
        co_h = torch.cat(co_hidden_states[-(self.num_embedding_layers + 1):-1], dim=2)
        cs_h = torch.cat(cs_hidden_states[-(self.num_embedding_layers + 1):-1], dim=2)
      similarities = F.cosine_similarity(co_h, cs_h, dim=2)

      score = (co_attentions * similarities).sum()
      return score.item()

  def score(self, context, target, target_offset, substitute):
    import torch

    # Encode context
    context_target_enc, target_tok_off, target_tok_len = self.encode(context, target, target_offset)

    # Compute substitute_tok_len
    substitute_enc = self.tokenizer.encode(substitute, add_special_tokens=False)

    with torch.no_grad():
      # Compute score
      # Eq. 3 from https://www.aclweb.org/anthology/P19-1328.pdf
      if self.compute_validation_score:
        validation_score = self.validation_score(context_target_enc, target_tok_off, target_tok_len, substitute_enc)
      else:
        validation_score = 0.
      if self.compute_proposal_score:
        proposal_score = self.proposal_score(context_target_enc, target_tok_off, target_tok_len, substitute_enc)
      else:
        proposal_score = 0.
      return validation_score + self.alpha * proposal_score


class BertBasedLexSubRanker(LexSubRanker, _BertBasedLexSubRankerImpl):
  def rank(self, context, target, target_offset, substitute, substitute_lemmatized, target_pos=None):
    return self.score(context, target, target_offset, substitute)


class BertBasedLexSubWithDelemmatizationRanker(LexSubWithDelemmatizationRanker, _BertBasedLexSubRankerImpl):
  def rank_delemmatized(self, context, target, target_offset, substitute, target_pos=None):
    return self.score(context, target, target_offset, substitute)
