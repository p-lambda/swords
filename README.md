# ⚔️ SWORDS: **S**tanford **Word** **S**ubstitution Benchmark

This repository houses the [**S**tanford **Word** **S**ubstitution (SWORDS) benchmark](#downloading-the-swords-benchmark). SWORDS is a benchmark for the task of _lexical substitution_ (LS), i.e. the task of replacing _target_ words in _context_ (inputs) with other contextually-synonymous _substitutes_ (outputs). For example: 

> (Input) **Context**: `The cow jumped over the moon.`
>
> (Input) **Target**: `jumped`
>
> (Output) **Substitutes**: `leapt, hurled itself, sprung, bounced, dove, hopped, vaulted`

Additionally, this repository houses the [`swords` library](#getting-started-with-the-swords-library), a platform for lexical substitution research. This platform offers:

1. A common format (JSON) for SWORD and prior LS datasets
1. Reimplementations of several common LS strategies
1. A standardized evaluation pipeline

The `swords` library was designed with a high standard of _reproducibility_. All processing steps between downloading raw data off the web and evaluating model results are preserved in the library. Docker is our weapon of choice for ensuring that these pipelines can be reproduced in the long term.

While Docker is great for reproducibility, it can be cumbersome for prototyping. Hence, we made sure our dataset format for SWORDS and past benchmarks is _portable_, allowing you to easily develop your LS strategies outside of both Docker and `swords`.

## Downloading the SWORDS benchmark

The SWORDS dev and test sets can be downloaded from the following links:

- [SWORDS development set](assets/parsed/swords-v1.0_dev.json.gz?raw=1)
- [SWORDS test set](assets/parsed/swords-v1.0_test.json.gz?raw=1)

SWORDS is published under the permissive [CC-BY-3.0-US license](https://creativecommons.org/licenses/by/3.0/us/). SWORDS includes content from the [CoInCo dataset](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/coinco/) and the [MASC corpus](http://www.anc.org/data/masc/), which are both distributed under the same license.

### SWORDS JSON format

The SWORDS dataset is distributed in a simple JSON format, containing all `contexts`, `targets`, and `substitutes` as keys in a single JSON object. Each context/target/substitute is associated with a unique ID, which is a SHA1 hash of its contents. The contents are as follows:

- Each context contains:
  - `context`: the context as plain text
  - `extra`: extra dataset-specific information about this context
- Each target contains:
  - `context_id`: the ID of the context that this target is from
  - `target`: the target word as plain text
  - `offset`: the character-level integer offset for this target in its context (helpful when the target occurs multiple times)
  - `pos`: the part of speech for this target; can be fed to LS methods as input
  - `extra`: extra dataset-specific information about this target
- Each substitute contains:
  - `target_id`: the ID of the target that this substitute is for
  - `substitute`: the substitute word as plain text
  - `extra`: extra dataset-specific information about this substitute

Labels for each substitute ID are found in the `substitute_labels` key.

Some example Python code for reading this format follows:

```py
from collections import defaultdict
import gzip
import json

# Load dataset
with gzip.open('swords-v1.0_dev.json.gz', 'r') as f:
  swords = json.load(f)

# Gather substitutes by target
tid_to_sids = defaultdict(list)
for sid, substitute in swords['substitutes'].items():
  tid_to_sids[substitute['target_id']].append(sid)

# Iterate through targets
for tid, target in swords['targets'].items():
  context = swords['contexts'][target['context_id']]
  substitutes = [swords['substitutes'][sid] for sid in tid_to_sids[tid]]
  labels = [swords['substitute_labels'][sid] for sid in tid_to_sids[tid]]
  scores = [l.count('TRUE') / len(l) for l in labels]
  print('-' * 80)
  print(context['context'])
  print('-' * 20)
  print('{} ({})'.format(target['target'], target['pos']))
  print(', '.join(['{} ({}%)'.format(substitute['substitute'], round(score * 100)) for substitute, score in sorted(zip(substitutes, scores), key=lambda x: -x[1])]))
  break
```

This should print out:

```
--------------------------------------------------------------------------------
Kim:
				I have completed the invoices for April, May and June and we owe
					Pasadena each month for a total of $3,615,910.62. I am waiting to hear
					back from Patti on May and June to make sure they are okay with her.
--------------------
total (NOUN)
amount (80%), sum (80%), sum total (60%), price (60%), balance (60%), gross (60%), figure (50%), cost (50%), full amount (40%), whole (30%), number (30%), quantum (10%), aggregate (10%), bill (10%), all (0%), entirety (0%), bulk (0%), flat out (0%), final (0%), body (0%), jackpot (0%), tale (0%), totality (0%), worth (0%), result (0%), allotment (0%), quantity (0%), budget (0%), mass (0%), the works (0%)
```

### Past benchmarks in SWORDS JSON format

For convenience, we package previous lexical substitution datasets in the same JSON format which we designed for distributing SWORDS. The CoInCo dataset ([Kremer et al. 2014](https://www.aclweb.org/anthology/E14-1057/)) can be downloaded in this format here:

- [CoInCo development set](assets/parsed/coinco_dev.json.gz?raw=1)
- [CoInCo test set](assets/parsed/coinco_test.json.gz?raw=1)

Other datasets such as SemEval07 ([McCarthy and Navigli 2007](https://www.aclweb.org/anthology/S07-1009/)) and TWSI ([Biemann 2012](https://www.aclweb.org/anthology/L12-1101/)) can be easily created by running the following scripts:

- [SemEval07](scripts/create_semeval07.sh)
- [TWSI](scripts/create_twsi.sh)

## Getting started with the `swords` library

It is highly recommend that you use the `swords` dataset through Docker. Hence, to get started, please **[install Docker](https://docs.docker.com/get-docker/)** if you do not already have it.

Next, **navigate to the `docker` directory and run `./run.sh`** which will download and host the [`swords` Docker image](https://hub.docker.com/r/chrisdonahue/swords) in the background.

Once the Docker image is running via `run.sh`, you can **interact with the `swords` command line interface (CLI) via the `cli.sh` script**. A few examples:

1. Download the required files for creating the SemEval07 dataset: `./cli.sh assets semeval07`
1. Parse the SemEval07 dataset into common JSON format: `./cli.sh parse semeval07`
1. Run BERT-based Lexical Substitution ([Zhou et al. 2019](https://www.aclweb.org/anthology/P19-1328/)) on the SWORDS development set (GPU recommended): `./cli.sh run swords-v1.0_dev --generator bert-based-ls --output_result_json_fp notebooks/swords_dev_bbls.result.json`
1. Download the required files for evaluating traditional LS metrics: `./cli.sh assets semeval07; ./cli.sh assets eval`
1. Evaluate the result from the previous step: `./cli.sh eval swords-v1.0_dev --result_json_fp notebooks/swords_dev_bbls.result.json --output_metrics_json_fp notebooks/swords_dev_bbls.metrics.json`

You can also run `./shell.sh` to launch an interactive bash shell in the Docker container, or `./notebook.sh` to launch a Jupyter notebook server with all dependencies pre-configured.

## Evaluating new lexical substitution methods on SWORDS

Here we provide start-to-finish examples for evaluating new lexical substitution methods on SWORDS, facilitating direct comparison with results from our paper. While our SWORDS file format makes it straightforward to run lexical substitution inference on SWORDS outside of Docker (examples below), we strongly recommend that _evaluation_ be run inside Docker for fair comparison.

### Generative setting

In the generative setting, lexical substitution methods must output a ranked list of substitute candidates for a given target. Below is a standalone example of a strategy which outputs 10 common verbs with random ordering, and saves the results into the format expected by our evaluation script:

```py
import gzip
import json
import random
import warnings

with gzip.open('swords-v1.0_dev.json.gz', 'r') as f:
  swords = json.load(f)

def generate(
    context,
    target,
    target_offset,
    target_pos=None):
  """Produces _substitutes_ for _target_ span within _context_

  Args:
    context: A text context, e.g., "The cow jumped over the moon".
    target: The target span, e.g., "jumped"
    target_offset: The character offset of the target word in context, e.g., 8
    target_pos: The UD part-of-speech (https://universaldependencies.org/u/pos/) for the target, e.g. "VERB"

  Returns:
    A list of substitutes and scores e.g., [(leapt, 10.), (sprung, 5.), (bounced, 1.5), ...]
  """
  # TODO: Your method here; placeholder outputs 10 common verbs
  substitutes = ['be', 'have', 'do', 'say', 'get', 'make', 'go', 'know', 'take', 'see']
  scores = [random.random() for _ in substitutes]
  return list(zip(substitutes, scores))

# NOTE: 'substitutes_lemmatized' should be True if your method produces lemmas (e.g. "jump") or False if your method produces wordforms (e.g. "jumping")
result = {'substitutes_lemmatized': True, 'substitutes': {}}
errors = 0
for tid, target in swords['targets'].items():
  context = swords['contexts'][target['context_id']]
  try:
    result['substitutes'][tid] = generate(
        context['context'],
        target['target'],
        target['offset'],
        target_pos=target.get('pos'))
  except:
    errors += 1
    continue

if errors > 0:
    warnings.warn(f'{errors} targets were not evaluated due to errors')

with open('swords-v1.0_dev_mygenerator.lsr.json', 'w') as f:
  f.write(json.dumps(result))
```

### Ranking setting

In the ranking setting, lexical substitution methods must rank the ground truth list of candidates from the dataset based on their contextual relevance for a given target. Below is a standalone example of a strategy which randomly ranks candidates, and saves the results into the format expected by our evaluation script:

```py
import gzip
import json
import random
import warnings

with gzip.open('swords-v1.0_dev.json.gz', 'r') as f:
  swords = json.load(f)

def score(
    context,
    target,
    target_offset,
    substitute,
    target_pos=None):
  """Scores a _substitute_ for _target_ span within _context_

  Args:
    context: A text context, e.g., "The cow jumped over the moon".
    target: The target span, e.g., "jumped"
    target_offset: The character offset of the target word in context, e.g., 8
    substitute: A substitute word
    target_pos: The UD part-of-speech (https://universaldependencies.org/u/pos/) for the target, e.g. "VERB"

  Returns:
    A score for this substitute
  """
  # TODO: Your method here; placeholder outputs random score
  return random.random()

# NOTE: 'substitutes_lemmatized' should be True if your method produces lemmas (e.g. "jump") or False if your method produces wordforms (e.g. "jumping")
result = {'substitutes_lemmatized': True, 'substitutes': {}}
errors = 0
for sid, substitute in swords['substitutes'].items():
  tid = substitute['target_id']
  target = swords['targets'][tid]
  context = swords['contexts'][target['context_id']]
  if tid not in result['substitutes']:
    result['substitutes'][tid] = []
  try:
    substitute_score = score(
        context['context'],
        target['target'],
        target['offset'],
        substitute['substitute'],
        target_pos=target.get('pos'))
    result['substitutes'][tid].append((substitute['substitute'], substitute_score))
  except:
    errors += 1
    continue

if errors > 0:
    warnings.warn(f'{errors} substitutes were not evaluated due to errors')

with open('swords-v1.0_dev_myranker.lsr.json', 'w') as f:
  f.write(json.dumps(result))
```

### Evaluating your lexical substiution strategy

To evaluate the example generator above, copy `swords-v1.0_dev_mygenerator.lsr.json` into the `notebooks` directory (to transfer it to Docker) and run: `./cli.sh eval swords-v1.0_dev --result_json_fp notebooks/swords-v1.0_dev_mygenerator.lsr.json --output_metrics_json_fp notebooks/mygenerator.metrics.json`

To evaluate the example ranker above, copy `swords-v1.0_dev_myranker.lsr.json` into the `notebooks` directory and run: `./cli.sh eval swords-v1.0_dev --result_json_fp notebooks/swords-v1.0_dev_myranker.lsr.json --output_metrics_json_fp notebooks/myranker.metrics.json --metrics gap_rat`
