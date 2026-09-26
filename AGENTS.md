# Notes for coding agents

pankhllm is a Rust, OpenAI-compatible LLM gateway that trains its own small decision models.

- Build and test: `cargo build --release`, `cargo test`, `cargo clippy --all-targets`.
- Adversarial and end-to-end tests live in `tests/adversarial.rs`; extend them, don't add lighter suites.
- **Training on a user's skills, prompts and logs:** follow `.agents/skills/pankhllm-train/SKILL.md`
  (Codex) or `.claude/skills/pankhllm-train/SKILL.md` (Claude Code). Both run
  `python3 notebooks/run_trainer.py`.
- Never commit `state/`, `pankh-work/`, `*.db`, logs, keys or models trained on real data.
