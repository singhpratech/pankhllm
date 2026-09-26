# Training pankhllm on your own data

pankhllm trains three small decision models of its own. None of them is an LLM.

| Model | Decides | Taught by | Used when |
|---|---|---|---|
| `plan` | which catalog operation answers a question, or none | the generative planner | `decisions.planner: true` |
| `tool` | which tool an agent turn calls, or none | the agent model's clean tool calls | `decisions.tools: true` |
| `skill` | which skill file a question needs, or none | skill `examples:`, requests that name a skill, the teacher | `prompts.auto_select: true`, no trigger matched |

Each is a hashed word and bigram logistic regression over the question's shape. Training
takes well under a second per 20,000 examples; a decision takes about 0.2 ms on a CPU.

## The shortest path: hand over your files

You need your skill files, and ideally your prompt files and logs. An operations catalog is
optional. Without one, pankhllm trains skill routing, plus tool decisions from logged tool calls.

```sh
python3 notebooks/run_trainer.py --skills ./skills --prompts agent.yaml system.md \
  --logs app.jsonl more.csv --teacher ollama        # or anthropic | openai | azure
```

The runner executes the notebook headless and sets up its own Python environment on first
use. It prints the results and leaves `pankh-work/bundle/`, which is ready to serve. Running again
into the same `--out` folder adds new labels to the earlier ones; use a new folder to start fresh.

**Ask an agent to do it.** The repository ships the same skill for both coding agents:
`.claude/skills/pankhllm-train/SKILL.md` for Claude Code and `.agents/skills/pankhllm-train/SKILL.md`
for OpenAI Codex. In either, say "train pankhllm on these skills and logs". The agent checks
the files and asks before sending logs to a cloud teacher. Then it runs the trainer and
reports the numbers.

**Unlabelled logs are fine.** For skills, the teacher reads each skill's name and opening
lines and picks the one each logged question needs, or none. For operations, it plans against
your catalog. Logged tool calls and requests that name a skill are used as labels directly.

Verified here: 3 demo skills, 1 prompt file and 40 unlabelled questions, local teacher. It
took 26 seconds end to end. The teacher labelled all 40, 6 of them as needing no skill. With
so few questions the skill model decided 1 of 8 held-out questions itself (correctly) and left
the rest to triggers. It needs a few hundred logged questions before it takes over.

## The notebook

[`notebooks/train_pankhllm.ipynb`](../notebooks/train_pankhllm.ipynb) is the training system.
It runs in Google Colab or local Jupyter, top to bottom:

1. Settings, 2. get pankhllm, 3. choose the teacher,
4. **load your skills** (a folder, or a zip upload in Colab),
5. **load your logs**: question JSONL, OpenAI chat request/response logs, CSV or plain text,
6. generate more questions (templates, teacher-written),
7. label, 8. **mine and train**, 9. evaluate on held-out questions,
10. **export a bundle**: server binary, `models.json`, config, skills, `run.sh`, zipped,
11. **try it** through the stock OpenAI SDK, 12. keep it learning.

Every setting can come from an environment variable (`PANKH_TEACHER`, `PANKH_SKILLS`,
`PANKH_LOGS`, `PANKH_CATALOG`, `PANKH_WORK`, sizes), so the same notebook runs headless on a
schedule with papermill or any job runner.

| Teacher | Key | Default model | Notes |
|---|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5` | Claude; `claude-haiku-4-5` is cheaper for big label runs |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` | any model with structured outputs |
| `azure` | `AZURE_OPENAI_API_KEY` + endpoint | your deployment name | Azure OpenAI and OpenAI deployments in Azure AI Foundry |
| `ollama` | none | `gemma4:12b` | fully local; needs a GPU runtime in Colab |

In Colab, keys go in **Secrets**. The notebook passes them to the `pankhllm` process through
the environment and never writes them to a file. With a cloud teacher, the questions you
label are sent to that provider; with Ollama nothing leaves the machine.

**Verified runs, local Jupyter, Ollama teacher, laptop:**

| Run | Time | Result on held-out questions |
|---|---|---|
| 20,000 template + 323 teacher-written questions | 15 min, mostly the teacher labelling | templates 56.1% decided, 100% precision; teacher-written 59.5% decided, 97.7% (1 wrong of 44) |
| 20,000 template questions, 3 skills, 2 sample log files, bundle export, live demo | 28 s | templates 52.5% decided, 100%; 4 of 5 demo questions answered with no LLM in 2 to 26 ms |

The Anthropic, OpenAI and Azure teachers use the same provider adapters as the router, which
are covered by tests and earlier real runs. The notebook itself has not yet been run against them.

## The steps by hand

### 1. Describe what can be decided

A catalog of operations with typed parameters (see `evals/pharma-kpi/router.yaml`):

```yaml
planner:
  enabled: true
  model: teacher              # the generative model that labels
  executor: { url: "http://127.0.0.1:7000/execute" }
  unsupported_examples:       # seed the "none of these" class
    - why did calls drop last quarter
  operations:
    - name: kpi_vs_benchmark
      description: How one geography compares with its peers on a KPI.
      examples: [how is T-112 doing against my peers on TRx]
      params:
        metric: { type: enum, values: [trx, nrx, calls], aliases: { trx: [total scripts, prescriptions] } }
        geo: { type: string, pattern: '[TDR]-\d{1,3}', description: territory like T-123 }
```

Enums with aliases, patterned codes and bounded numbers matter more than anything else. At
serve time the model may only choose operations whose parameters the question fills
exactly, which is where most of the precision comes from.

### 2. Gather questions

Any JSONL file with one question per line. A `label` is optional:

```json
{"question": "who covers D-14?", "label": "coverage_owner"}
{"question": "how are we trending on new scripts in R-2"}
```

Sources, in order of value:

1. **Your application logs.** Real questions, real phrasing.
2. **Teacher-written questions.** `pankhllm augment questions.jsonl --per-op 40` asks the
   teacher for many phrasings per operation and for questions nothing should answer.
3. **Templates.** A script like `evals/pharma-kpi/gen.py` produces labelled questions at any
   volume. Hold out some templates for testing, never just some questions.
4. **Public question sets** for phrasing variety and rejection. Licenses are in
   [DATASETS.md](DATASETS.md).

Keep a test file apart from training from the start.

### 3. Label

```sh
pankhllm --config train.yaml label train.jsonl --concurrency 4
pankhllm --config train.yaml label test-questions.jsonl --out test.jsonl
pankhllm --config train.yaml label logs.jsonl --task skill --teacher-model teacher
```

Lines with a `label` are recorded as given. Lines without one go to the teacher, which
decides once; nothing is executed. The plan is validated against the catalog and invalid
plans are dropped. Without `--out`, labels go into the trace store (`store.path`) as
training rows holding only the question shape. With `--out`, they are written to a file for
evaluation and never reach training.

### 4. Train

```sh
pankhllm --config train.yaml mine --since-hours 1000000 --apply
```

The miner trains `plan`, `tool` and `skill` from every label in the store plus catalog and
skill examples. It holds out a fifth of the shapes and recommends the confidence threshold
that meets `decisions.target_precision` (0.98 by default). `--apply` saves the models into
the store. The report shows raw held-out accuracy, which is the classifier alone. At serve
time the structural checks raise precision well above it; measure that in step 5.

### 5. Evaluate

```sh
pankhllm --config train.yaml decide-eval test.jsonl --show 15
pankhllm --config train.yaml decide-eval skill-test.jsonl --task skill
```

Reports how many questions were decided without a model, how many of those were right,
the wrong ones, and decision time. Offline; no model is called. Precision is the number to
watch: a question the model does not decide still gets answered by the planner.

### 6. Export and serve

```sh
pankhllm --config train.yaml export-models models.json
```

`models.json` holds the `plan`, `tool` and `skill` models: hashed weights and labels, no
questions. Point the server at it:



```yaml
decisions: { engine: native, models_file: models.json }
store: { path: ./state/pankhllm.db, redact_questions: true }   # optional; keeps it learning
```

At start and on every reload, pankhllm uses whichever model was trained last: the one in
the store (retrained by the nightly miner) or the one in `models_file` (shipped by you).
`pankhllm serve` reloads every 5 minutes; `POST /v1/decisions/reload` reloads at once.
`GET /v1/stats` shows what is loaded. No store is needed to serve a shipped file.

## Training from production logs

With a store configured, training needs no extra work:

- Every question the planner answers with a plan that executes cleanly is recorded as a
  `plan` label. Every question it declines is an `UNSUPPORTED` label.
- Every clean agent tool call, and every direct answer to a tool-selection turn, is a
  `tool` label.
- Every request that names its skill (`"pankhllm": {"prompt": "hr-policy"}`) is a `skill`
  label.
- Run `pankhllm mine --apply` nightly (cron, Task Scheduler, a Kubernetes CronJob).

To start from existing logs, extract the questions to JSONL and run step 3. If your logs
already hold the tool or operation that was used, put it in `label` and no teacher is needed.

`decisions.mode: shadow` runs the model beside the planner without acting on it. The
miner then reports agreement per threshold, so you can switch to `act` with evidence.

## Training skill routing from skill files

Skill files keep their front matter. Add `examples:` next to `triggers:`:

```markdown
---
pankhllm:
  triggers: [call activity, trx, nrx]
  examples:
    - how is my territory doing against my peers this quarter
    - which doctors in D-12 stopped writing last month
---
# Field KPI skill
...
```

With `prompts.auto_select: true`, triggers are tried first. If none matches, the skill model
picks a skill only when it is confident and the wording is familiar; otherwise no extra
skill is sent. It needs at least two skills that are not `always`. Requests that name a
skill explicitly keep teaching it, and the nightly miner retrains it.

## Resource needs

The decision models are not LLMs. Measured on a laptop CPU (Intel i9-13900HK), one core:

| What | Measured |
|---|---|
| Plan model size | 12,545 weights, 8 labels; `models.json` with all models is 262 KB |
| Training 20,000 questions (`mine`) | 1.1 s wall, 45 MB peak memory |
| 2,000 offline decisions (`decide-eval`) | 0.46 s total; 0.22 ms p50 each |
| Server binary | 11.4 MB, no runtime dependencies |
| Server memory | 17 MB idle, 31 MB after 3,000 requests |
| Served decisions, 16 concurrent, Python client and executor | 1,753 requests/s, p50 2.0 ms, p99 6.8 ms |
| Server CPU per decided request | about 0.8 ms, including HTTP and the executor call |

No GPU is needed to train or serve the decision models. A GPU, or a hosted LLM, is needed
for the teacher during labelling, and for questions the model does not decide and must be
answered by generation.

## Privacy

- The store keeps question shapes, never raw questions, when `redact_questions: true`.
- Trained models hold hashed features only.
- `state/`, `*.db` and anything learned are git-ignored. Do not commit them: they can
  reveal business vocabulary.
- Keys stay in environment variables or Colab Secrets.

## Choosing thresholds

| Setting | Default | Meaning |
|---|---|---|
| `decisions.target_precision` | 0.98 | precision the miner's recommended threshold aims for |
| `decisions.consensus_confidence` | 0.55 | floor when the structure already agrees; the model's recommendation can raise it |
| `decisions.min_confidence` | 0.85 | confidence needed to say "nothing fits" and skip the planner |
| `decisions.min_known_words` | 0.7 | share of the question's words seen in training; below it, the planner decides |

Raise `min_known_words` if wrong decisions come from unfamiliar phrasing. Lower
`consensus_confidence` only after shadow mode shows it is safe.
