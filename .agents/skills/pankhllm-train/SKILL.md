---
name: pankhllm-train
description: Train pankhllm's own decision models (skill routing, tool and operation decisions) from a user's skill files, prompt files and logs, and produce a ready-to-run server bundle. Use when the user asks to train, tune or fine-tune pankhllm, or hands over skills, prompts or logs for it.
---

# Train pankhllm from skills, prompts and logs

pankhllm trains small decision models of its own (not LLMs). A teacher LLM labels the
questions once; the models then decide in well under a millisecond on a CPU. This skill runs
`notebooks/train_pankhllm.ipynb` headless through `notebooks/run_trainer.py`.

## 1. Collect the inputs

Ask only for what is missing:

| Input | Flag | Required |
|---|---|---|
| Folder of skill `.md` files | `--skills` | yes |
| Agent or system prompt files | `--prompts` | no |
| Log files | `--logs` | no, but training is weak without them |
| Operations catalog (`router.yaml` with `planner.operations`) | `--catalog` | no; only for operation decisions |
| Teacher: `ollama`, `anthropic`, `openai`, `azure` | `--teacher` | default `ollama` |

Look at the first lines of each log file and confirm it is one of: JSONL with `question`
(optional `label`, `task`), JSONL of OpenAI chat requests or request/response pairs, CSV with a
`question` column, or plain text with one question per line. Convert other formats to JSONL
with a `question` field first.

## 2. Check privacy before choosing the teacher

With `anthropic`, `openai` or `azure`, the log questions are sent to that provider for
labelling. Say so and get an explicit yes before using a cloud teacher on real logs. `ollama`
keeps everything local; check it is running with `curl -s localhost:11434/api/tags`.
Keys come from the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`AZURE_OPENAI_API_KEY`). Never write a key to a file or print it.

## 3. Run

From the pankhllm repository root:

```sh
python3 notebooks/run_trainer.py --skills <dir> --prompts <files...> --logs <files...> \
  --teacher ollama --out pankh-work
```

Add `--bin target/release/pankhllm` when a release build exists (otherwise it builds one), and
`--catalog <router.yaml>` for operation decisions. The runner creates its own virtual
environment on first use. Expect seconds for small logs; the teacher's labelling dominates
(about 2 to 3 seconds per question with a local 12B model at `--concurrency 2`).

## 4. Report

Read the runner's output and tell the user, in a short table: questions loaded, labels per
skill or operation, and for each held-out test set the share decided without an LLM and the
precision. Point to `pankh-work/bundle/` (server, `models.json`, config, skills, `run.sh`) and
`pankh-work/pankhllm-bundle.zip`. If fewer than a few hundred questions were labelled, say that
the model will stay cautious and send most questions to the teacher or triggers until more
logs arrive.

To serve: `pankh-work/bundle/run.sh` (port 4000), then point any OpenAI-compatible client at
`http://HOST:4000/v1`.

## Rules

- Never commit `pankh-work/`, `state/`, logs, `*.db` or `models.json` built from real data.
- Never copy real questions into the repository, docs or commit messages.
