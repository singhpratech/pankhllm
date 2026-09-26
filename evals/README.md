# evals

- `domain_adversarial.py`: eleven pharma-domain attack scenarios against a live router (real models).
- `gen_pharma_questions.py`: labelled synthetic question generator for routing accuracy.
- `pharma-10k.jsonl`: 10,000 generated questions with expected intent and tier.
- `pharma.yaml`, `pharma-gpt56.yaml`: configs used for the runs recorded in `docs/EVALUATION.md`.
- `real/` (git-ignored): put labelled production questions here and run `pankhllm eval evals/real/<file>.jsonl`.

All fixtures are synthetic. Do not add real HCP, patient or customer data to this directory.
