# rlhf-from-scratch

Hands-on RLHF tutorial and minimal code examples. This repo is focused on teaching the main steps of RLHF with compact, readable code rather than providing a production system.

What the code implements (short)

- `src/ppo/ppo_trainer.py` — a simple PPO training loop to update a language model policy.
- `src/ppo/core_utils.py` — helper routines (rollout/processing, advantage/return computation, reward wrappers).
- `src/ppo/parse_args.py` — CLI/experiment argument parsing for training runs.
- `tutorial.ipynb` — the notebook that ties the pieces together (theory, small experiments, and examples that call the code above).

What's covered in the notebook (brief)

- RLHF pipeline overview: preference data → reward model → policy optimization.
- Short demonstrations of reward modeling, PPO-based fine-tuning, and comparisons.
- Practical notes and small runnable code snippets to reproduce toy experiments.

How to try

1. Open `tutorial.ipynb` in Jupyter and run cells interactively.
2. Inspect `src/ppo/` to see how the notebook maps to the trainer and utilities.

If you want a shorter or more hands-on example (e.g., a single script to run a tiny DPO or PPO demo), tell me and I can add it.
