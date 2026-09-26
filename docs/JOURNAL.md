# Journal: how pankhllm's decision model was built

A record of each step, what we measured, and what we changed because of it. Dates are 2026.

## 25 Sep: a router, then a gateway that learns

- Built pankhllm as an OpenAI-compatible Rust router: intent, retrieval score, context size,
  privacy tags, cost and latency budgets; cascades on refusals and hedges.
- Added learned routes: a question's shape plus the model's tool call becomes a template, so
  a repeat question needs no model. Added cache, regex rules, planner lane, trace store,
  overnight miner, and clients for eleven languages.

## 26 Sep morning: decision models as a lane

- Compared against two decision models, Jev (hosted) and Laya (open weights). They answer a
  typed choice in about 12 ms on a GPU, but zero-shot Laya chose correctly on only 8 of 14
  questions of our catalog.
- Found what makes a small decider accurate: **structural feasibility**. Only operations
  whose parameters the question fills exactly are offered, plus a reject option.
- Measured **position bias**: the same question scored 0.90 with the operation first and
  0.52 with the reject option first. Options are now sent in catalog order, reject last.
- Result on 14 labelled questions: planner 1,743 ms p50, Laya lane 49 ms, both 14 of 14.

## 26 Sep midday: our own model

The decision lane should not depend on anyone's model. We built pankhllm's own:

1. **Teacher labels.** The generative planner's clean plans and its "unsupported" answers,
   and the agent's clean tool calls, are recorded as labels holding only the question shape.
2. **Model.** Hashed word and bigram features, multinomial logistic regression, trained by
   the miner. Deterministic, CPU only, well under a second to train.
3. **First result.** 120 teacher labels over three operations: 14 of 14 correct, 4 ms end to
   end, decisions in 0.13 to 0.26 ms.

## 26 Sep afternoon: scaling to a real domain

We wrote a pharma KPI catalog (seven operations, fictional products) and a template
generator with held-out templates. Each change below was measured on 2,000 questions from
templates the model never saw.

| Step | Change | Precision of decided questions |
|---|---|---|
| 1 | first model, 20,000 training questions | 86.47% |
| 2 | dedupe identical shapes, L2 regularisation (to stop memorising templates) | 82.83%: the reject class took over |
| 3 | reject only when nothing fits structurally; one phrase may map to several enum values; cap class weights | 98.54% |
| 4 | numbers inside a recognised phrase count as used; one shared metric vocabulary across operations | 99.92% |
| 5 | unfamiliar wording goes to the planner (`min_known_words` 0.7) | 99.92%, and 100% on teacher-written questions |

Final numbers:

| Test set | Questions | Decided | Precision | p50 |
|---|---|---|---|---|
| Held-out templates | 2,000 | 66.2% | 99.92% (1 wrong) | 0.20 ms |
| Teacher-written, teacher-labelled | 82 | 68.3% | 100% | 0.21 ms |

The one wrong decision: "trend of total scripts in T-892 for the last 13 weeks" went to
`kpi_value` instead of `kpi_trend`, because "last 13 weeks" is also a valid window.

## 26 Sep afternoon: more and different questions

- Counted the question space of the template generator: about 22,000 shapes and over
  89 million distinct questions.
- `pankhllm augment` asks the teacher for varied phrasings. The first prompt showed
  parameter regexes and the model wrote codes like "TDR-102"; it now shows descriptions.
- `pankhllm label` labels questions with the teacher without executing anything, validates
  each plan, and drops invalid ones. In the first run 82 of 242 generated questions produced
  valid labels.
- Researched which public data is legal to train on. Summary and links in
  [DATASETS.md](DATASETS.md). Licensed sources such as IQVIA Xponent are not usable
  without a license that allows it.

## 26 Sep evening: a harder test set, and what fixed it

- A second teacher-written set of 367 questions, with a better generator prompt, gave
  **93.67% precision**: 15 wrong of 237 decided. The small first set had hidden this.
- Most errors were questions about a change ("since last month", "growth") where the
  change operation's comparison parameter could not be filled, so the model picked the
  nearest operation that did fit. Added a **disagreement check**: if the model, free to choose
  among all operations, prefers one the question does not fit, the planner decides.
  Precision rose to 96.0%, and coverage on templates fell from 66% to 52%.
- Added **183 teacher labels** (half of the new set) to the 20,000 template questions.
  Coverage came back to 63% on templates, 67% and 61% on the teacher-written sets, with
  100%, 100% and 96.5% precision. Three of the four remaining disagreements are
  questionable teacher labels. Full table in [BENCHMARK-DECISIONS.md](BENCHMARK-DECISIONS.md#pharma-kpi-catalog-at-scale).
- Lesson: templates teach the structure; a few hundred real or teacher-written
  questions teach how people actually write. Production traffic provides the second for free.

## 26 Sep evening: pluggable training

- **Skill routing** learns from skill files and logs: `examples:` in front matter seed it,
  requests that name a skill label it, and the miner retrains it with the other models.
- **Notebook** for Colab or Jupyter: pick Claude, OpenAI, Azure OpenAI / AI Foundry or
  Ollama as the teacher, then generate, label, train and evaluate. Run end to end locally
  with the Ollama teacher on a small setting (3,000 template and 28 teacher labels): 50.4% of
  500 held-out questions decided, none wrong.
- **Docs**: [TRAINING.md](TRAINING.md) for the steps, [DATASETS.md](DATASETS.md) for sources.

## 26 Sep night: the notebook as the training system

- The notebook now loads skills and logs, mines, evaluates, and exports a bundle: server
  binary, `models.json`, config, skills and a start script. A last step starts the bundle and
  asks questions through the stock OpenAI Python SDK.
- New `pankhllm export-models` and `decisions.models_file`: a trained model ships as one
  262 KB file. The server uses whichever of store and file was trained last.
- Found and fixed a leak: `label --out` recorded rows that already carried a label into the
  training store. Test rows could reach training. A binary-level test now guards it.
- Full run with the local teacher: 15 minutes for 20,000 template and 323 teacher-written
  questions. Held out: templates 56.1% decided at 100%, teacher-written 59.5% at 97.7%.
- Measured resources: training 1.1 s and 45 MB on one CPU core; server 17 MB idle, 31 MB
  under load, 1,753 requests/s at p50 2.0 ms end to end. No GPU needed for the decision models.
