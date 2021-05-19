import random

from .. import Label, LexSubDataset
from ..util import index_all, normalize_text
from . import LexSubGenerator


_ENGINE_TO_MAX_TOKENS = {
  'davinci': 2048
}
class GPT3Generator(LexSubGenerator):
  def __init__(
      self,
      priming_data=None,
      tid_to_cached_output={},
      engine='davinci',
      max_num_priming_examples=None,
      max_tokens_to_generate=128,
      force_longer_continuations=True,
      frequency_penalty=None,
      presence_penalty=None,
      temperature=0,
      openai_api_key=None,
      filter_duplicates=False,
      *args,
      **kwargs):
    if engine not in _ENGINE_TO_MAX_TOKENS:
      raise ValueError()

    super().__init__()

    self.priming_data = priming_data
    self.engine = engine
    self.max_num_priming_examples = max_num_priming_examples
    self.max_tokens_to_generate = max_tokens_to_generate
    self.temperature = temperature
    self.frequency_penalty = frequency_penalty
    self.presence_penalty = presence_penalty

    self.tid_to_cached_output = tid_to_cached_output
    if 'outputs' in self.tid_to_cached_output:
      self.tid_to_cached_output = self.tid_to_cached_output['outputs']
    self.openai_api_key = openai_api_key

    self.logit_bias = None
    if force_longer_continuations:
      self.logit_bias = {
        "198": -100,   # NOTE: \n
        "201": -100,   # NOTE: \r
        "628": -100,   # NOTE: \n\n
        "50256": -100, # NOTE: <endoftext>
      }

    self.filter_duplicates = filter_duplicates

    self.tokenizer = None

  def substitutes_will_be_lemmatized(self):
    if self.priming_data is not None:
      return self.priming_data.substitutes_lemmatized
    else:
      return False
  
  def format_context(self, context, target, target_offset, *args, **kwargs):
    target_occurrence = index_all(context, target).index(target_offset)
    context = normalize_text(context)
    target_offset = index_all(context, target)[target_occurrence]
    context = context[:target_offset] + '**' + target + '**' + context[target_offset+len(target):]
    return context

  def format_target(self, context, target, target_offset, *args, **kwargs):
    return f'Q: What are appropriate substitutes for **{target}** in the above text?'

  def format_substitutes(self, substitutes):
    if substitutes is None:
      return 'A:'
    else:
      substitutes = [(s, sum([int(l in [Label.TRUE, Label.TRUE_IMPLICIT]) for l in ls])) for s, ls in substitutes]
      substitutes = sorted([(sub, score) for sub, score in substitutes if score > 0], key=lambda x: -x[1])
      return 'A: {}'.format(', '.join([s for s, _ in substitutes]))

  def create_gpt3_input(self, context, target, target_offset, *args, **kwargs):
    if self.tokenizer is None:
      from transformers import GPT2TokenizerFast
      self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Query
    context_formatted = self.format_context(context, target, target_offset)
    target_formatted = self.format_target(context, target, target_offset)
    substitutes_formatted = self.format_substitutes(None)
    examples = ['\n'.join([context_formatted, target_formatted, substitutes_formatted])]
    
    # Insert priming sequence
    max_tokens = _ENGINE_TO_MAX_TOKENS[self.engine]
    tids = set() if self.priming_data is None else set(self.priming_data.all_target_ids())
    while True:
      if len(tids) == 0:
        break
      if len(self.tokenizer.encode('\n\n'.join(examples))) + self.max_tokens_to_generate > max_tokens:
        break
      if self.max_num_priming_examples is not None and (len(examples) - 1) >= self.max_num_priming_examples:
        break

      tid = random.choice(list(tids))
      tids -= set([tid])
      generator_inputs = self.priming_data.get_generator_inputs(tid)
      substitutes = [(self.priming_data.get_substitute(sid)['substitute'], self.priming_data.get_substitute_labels(sid)) for sid in self.priming_data.all_substitute_ids(target_id=tid)]
      context_formatted = self.format_context(**generator_inputs)
      target_formatted = self.format_target(**generator_inputs)
      substitutes_formatted = self.format_substitutes(substitutes)
      examples.insert(0, '\n'.join([context_formatted, target_formatted, substitutes_formatted]))
    
    # Make sure we have enough room
    while len(self.tokenizer.encode('\n\n'.join(examples))) + self.max_tokens_to_generate > max_tokens:
      examples = examples[1:]
    assert len(examples) > 0
    
    # Create result
    result = {
      'engine': self.engine,
      'prompt': '\n\n'.join(examples),
      'max_tokens': self.max_tokens_to_generate,
      'temperature': self.temperature,
      'logprobs': 100,
      'stop': ['\n', '\r', '\n\n'],
    }
    if self.frequency_penalty is not None:
      result['frequency_penalty'] = self.frequency_penalty
    if self.presence_penalty is not None:
      result['presence_penalty'] = self.presence_penalty
    if self.logit_bias is not None:
      result['logit_bias'] = self.logit_bias
    return result

  def execute_gpt3(self, gpt3_input):
    import openai
    if self.openai_api_key is not None:
      openai.api_key = self.openai_api_key.strip()
    response = openai.Completion.create(**gpt3_input)
    return response

  def process_gpt3_output(self, gpt3_output):
    substitutes = [s.strip() for s in gpt3_output['choices'][0]['text'].strip().split(',') if len(s.strip()) > 0]
    if self.filter_duplicates:
      seen = set()
      seen_add = seen.add
      substitutes = [s for s in substitutes if not (s.lower() in seen or seen_add(s.lower()))]
    return [(s, -float(r)) for r, s in enumerate(substitutes)]

  def generate(self, context, target, target_offset, target_pos=None):
    cid = LexSubDataset.context_id(LexSubDataset.create_context(context))
    tid = LexSubDataset.target_id(LexSubDataset.create_target(cid, target, target_offset, pos=target_pos))
    gpt3_output = self.tid_to_cached_output.get(tid)
    if gpt3_output is None:
      if self.openai_api_key is not None:
        gpt3_input = self.create_gpt3_input(context, target, target_offset)
        gpt3_output = self.execute_gpt3(gpt3_input)
      else:
        raise ValueError()
    return self.process_gpt3_output(gpt3_output)
