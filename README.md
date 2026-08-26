# HRA-Pay

**Hierarchical reinforcement learning for payment retry optimisation, with LLM-calibrated reward shaping.**

Razorpay Buildathon 2026 — Track 03, AI Revenue Recovery.

> Status: in development. Day 1 of 9 complete — environment and spec.

---

## The problem

When a payment fails, a merchant faces a compound decision, not a single one: retry or
give up, when to retry, and on which payment rail. Production retry systems generally
answer only the first two with a supervised classifier that predicts a success
probability. This project treats the whole retry sequence as a Markov Decision Process
and learns an explicit policy over *timing and channel jointly*, with a hard guardrail
layer between the learned policy and any action that touches money.

## Where AI is used, and where it deliberately is not

| Layer | Approach | Why |
|---|---|---|
| Decision policy | Branching Dueling Q-Network | Sequential credit assignment across a multi-step retry episode |
| Reward calibration | LLM, offline, cached, human-reviewed | Semantic reading of unstructured decline text, with zero runtime latency or non-determinism |
| Guardrails | Deterministic rules | **Not AI, on purpose.** Compliance limits must be provable, not learned |
| Metrics | Plain arithmetic | **Not AI, on purpose.** Nothing generative goes near the scoreboard |

## Quickstart

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
python tasks.py test
python tasks.py env-demo
```

**Linux / macOS**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
make dev
make test
make env-demo
```

`make <task>` and `python tasks.py <task>` run the same commands, so the project
behaves identically on either platform. The CPU-only torch index is worth using:
the default PyPI wheel bundles roughly 2 GB of CUDA this project never touches.

## Repository layout

```
configs/          ground-truth environment specs (committed and versioned)
src/hrapay/env/   spec validation, episode generator, Gymnasium environment
tests/            environment invariants and modelled issuer behaviours
```

## Limitations

Results are demonstrated on synthetic data with assumed success-probability
distributions, not on live Razorpay transactions. The spec encoding those assumptions
is committed in full at `configs/` so its assumptions can be inspected and disputed
directly. See `ENGINEERING_LOG.md` for the design decisions and what went wrong.
