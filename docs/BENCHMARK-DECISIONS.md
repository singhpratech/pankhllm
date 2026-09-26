# Benchmark: how the router decides

Same machine, same 14 labelled questions, same executor, one variable: how the router decides which operation answers a question.

## Headline

| Setup | Correct | p50 end to end | Answerable first asks | Decision time |
|---|---|---|---|---|
| Generative planner (local 12B model) | 14 / 14 | 1,743 ms | 10 in 1.1 to 1.8 s | ~1.4 s |
| External decision model (Laya, same GPU) | 14 / 14 | 49 ms | 8 in 17 to 49 ms, 2 fell back | 12 ms |
| **pankhllm's own model, self-trained** | **14 / 14** | **4 ms** | **10 in 2 to 26 ms** | **0.13 to 0.26 ms** |

pankhllm's own model is a hashed-feature logistic regression trained by the router itself. The local 12B planner answered 120 training questions (none overlapping the test set: different phrasings, different territory codes, different year), each answer became a label, and `pankhllm mine --apply` trained the model in about 0.1 s. It runs in-process on the CPU. The decision times above are from a debug build.

Reproduce the self-trained row: `pankhllm --config evals/decisions/router-native.yaml warm evals/decisions/train.jsonl --replay --concurrency 2`, then `pankhllm --config evals/decisions/router-native.yaml mine --since-hours 2 --apply`, then serve and run `evals/decisions/run.py` as below.

## External decision model details

- Machine: laptop, RTX 4090 Laptop GPU (16 GB).
- Generative planner and agent: `gemma4:12b` on local Ollama, reasoning off.
- Decision engine: [Laya](https://github.com/NandhaKishorM/laya) 0.3.20 via `laya-serve`, English checkpoint, on the same GPU.
- Executor: the Python stdlib executor (`evals/decisions/executor.py`).
- Questions: `evals/decisions/questions.jsonl`: 10 answerable by a three-operation catalog, 4 that must go to the agent (why, advice, creative writing, explanation).
- Reproduce: start Ollama, `laya-serve`, the executor, `pankhllm --config evals/decisions/router.yaml serve --port 4143`, then `python3 evals/decisions/run.py 4143 evals/decisions/questions.jsonl`.

Laya alone answers a typed choice in 12 ms p50 on this GPU. End to end, including HTTP, validation and the executor, the eight fast answers took 17 to 49 ms. Two answerable questions fell back to the generative planner and were still answered correctly. The four open-ended questions went to the agent in both setups; they need generation and take 2 to 6 s with this local model.

## What it took to get there

1. **Zero-shot Laya is fast but not accurate enough on its own.** Choosing among the full catalog it picked the right operation for 8 of 14 questions, with confidences mostly between 0.35 and 0.7. Laya's own documentation says as much: near chance on custom decisions until fine-tuned.
2. **Example phrasings in the option descriptions made it worse.** Laya shares a small token budget across options; longer descriptions get truncated and confidence collapsed to zero coverage.
3. **Constraining the choice to structurally feasible operations is what works.** The slot filler first finds the operations whose parameters the question fills exactly, using every code, date and number it contains. Laya then chooses only among those and "unsupported", at a 0.55 threshold. Wrong operations rarely fit structurally, so they are never offered.
4. **Position bias is real.** The same question scored 0.90 with the operation listed first and 0.52 with "unsupported" listed first. The router sends options in catalog order with the reject option last. Averaging both orders was tested and was worse on this set.
5. **The intent gate matters.** "Why did calls fall in T-112 in 2026-08?" fits an operation structurally and Laya scored it 0.50. It never reaches the decision lane because the classifier routes reasoning questions to the agent first.

## Limits

- The self-trained model learned from 117 teacher labels over three operations. Its held-out accuracy without the structural constraint was 81%; the miner recommended a 0.90 threshold to hold 98% precision, and the structural constraint does the rest. More operations need more labels.
- Fourteen questions is a small set. Run shadow mode on real traffic before acting: `pankhllm mine` reports agreement per threshold and recommends one.
- Coverage grows with fine-tuning. `pankhllm mine --export-decisions` writes each question with the decision the slower path made, which is the training data for a checkpoint tuned to your catalog.
- Jev speaks the same `/v1/systemone` shape but was not tested here: it needs an account.

## Pharma KPI catalog at scale

A bigger test: seven operations plus a reject class (`evals/pharma-kpi/router.yaml`,
fictional products), release build, CPU only. Three held-out test sets, none seen in training:

- **Templates**: 2,000 questions from phrasing templates held out of training (`gen.py --heldout`).
- **Teacher-written A**: 82 questions a local 12B model wrote and then labelled.
- **Teacher-written B**: 367 more from a second run with a better generator prompt. Half
  were used for training in the last row below, the other half (184) for testing.

| Model trained on | Test set | Decided without a model | Precision | Decision p50 |
|---|---|---|---|---|
| 20,000 template questions | Templates | 52.5% | 100% (0 wrong of 1,049) | 0.2 ms |
| 20,000 template questions | Teacher-written A | 59.8% | 100% | 0.2 ms |
| 20,000 template questions | Teacher-written B | 54.8% | 96.0% (8 wrong of 201) | 0.2 ms |
| **+ 183 teacher-labelled questions** | Templates | **63.3%** | **100%** (0 wrong of 1,266) | 0.21 ms |
| **+ 183 teacher-labelled questions** | Teacher-written A | **67.1%** | **100%** | 0.36 ms |
| **+ 183 teacher-labelled questions** | Teacher-written B, held-out half | **61.4%** | **96.5%** (4 wrong of 113) | 0.23 ms |

Every question not decided went to the generative planner, as it would at serve time.

The four disagreements in the last row, with the teacher's label:

- "Explain why the 'prescriber_list' shows different figures for the 2023 period." The model said `UNSUPPORTED`; the teacher said `kpi_change`. The model is right: it asks why.
- "How has the number of calls for cardivex changed in T-123 over the past 12 months?" The model said `kpi_trend`; the teacher said `kpi_change`. Both are defensible.
- "Show me samples for oncora in T-123 over 8 weeks." The model said `kpi_trend`; the teacher said `kpi_value`. Both are defensible.
- "Is the frequency for cardivex in D-01 better than last year?" The model said `kpi_value`; the teacher said `kpi_change`. The model is wrong.

What the numbers say:

1. **Templates alone do not generalise to how people write.** A model trained only on
   templates was perfect on template phrasing and 96% on teacher-written phrasing.
2. **A few hundred teacher labels close most of the gap.** 183 labels added to 20,000
   template questions raised coverage by about ten points on every set without losing precision.
   In production these labels come for free: every planner decision is one.
3. **The disagreement check trades coverage for precision.** When the model, free to choose
   among all operations, prefers one the question does not fit, the planner decides. Before
   the check, the template-only model decided 66.2% of the templates at 99.92% and 64.6% of
   teacher-written B at 93.7%.

Reproduce: `python3 evals/pharma-kpi/gen.py 20000 > train.jsonl`, `gen.py 2000 --heldout > test.jsonl`,
`pankhllm --config evals/pharma-kpi/router.yaml label train.jsonl`, `... mine --since-hours 100000 --apply`,
`... decide-eval test.jsonl`. Teacher-written sets: `... augment q.jsonl --per-op 40`, then
`... label q.jsonl --out test.jsonl`. A generative teacher is not deterministic, so those sets
differ run to run. The notebook `notebooks/train_pankhllm.ipynb` runs all of it.
