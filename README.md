# HRA-Pay

**Hierarchical reinforcement learning for payment retry optimisation, with LLM-calibrated reward shaping.**

Razorpay Buildathon 2026 — Track 03, AI Revenue Recovery

When a payment fails, a merchant faces a compound decision, not a single one:
retry or give up, *when* to retry, and *on which rail*. Production retry systems
generally answer only the first two, with a supervised classifier that predicts a
success probability. HRA-Pay treats the whole retry sequence as a Markov Decision
Process and learns an explicit policy over **timing and channel jointly** — with a
deterministic guardrail layer between the learned policy and any action that
moves money.

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

python tasks.py test        # 91 tests
python tasks.py eval        # reproduces every number below
python tasks.py demo        # the dashboard
```

---

## Results

1,000 episodes per run. Learned agents are trained on **3 seeds each** and
reported as mean ± standard deviation across those seeds; the static baselines
are deterministic.

### Training distribution

| policy | recovered ₹ | recovery rate | wasted retries | issuer risk | time to recovery |
|---|---|---|---|---|---|
| bdq (branched) | 1,015,730 ± 41,732 | 0.693 ± 0.021 | 1,550 | **0** | 66.6h |
| flat_dqn | 1,003,601 ± 20,949 | 0.689 ± 0.017 | 1,751 | **0** | 48.8h |
| static_with_switch | 880,377 | 0.583 | 1,446 | **0** | 49.8h |
| static_schedule | 536,078 | 0.346 | 1,426 | **0** | 63.5h |

### Held-out (shifted) distribution — no agent trained on this

| policy | recovered ₹ | recovery rate | wasted retries | issuer risk | time to recovery |
|---|---|---|---|---|---|
| flat_dqn | 844,508 ± 22,809 | 0.483 ± 0.005 | 2,240 | **0** | 53.1h |
| static_with_switch | 753,958 | 0.413 | 1,689 | **0** | 49.2h |
| bdq (branched) | 765,171 ± 46,101 | 0.435 ± 0.018 | 2,051 | **0** | 66.5h |
| static_schedule | 381,354 | 0.212 | 1,563 | **0** | 55.5h |

`python tasks.py eval` prints these tables, runs a verdict check on every
architecture comparison, and writes `results/summary_{train,holdout}.csv`.

---

## What we claim, and what we refuse to claim

**Claimed.** Both learned agents beat both static baselines, on the training
distribution *and* under shift. The best learned agent recovers roughly 15% more
revenue than the best static baseline on the training distribution and 12% more
under shift. Issuer-risk exposure is **zero in every run** — no policy ever
retried a fraud-flagged authorisation, because the guard does not permit it.

**Refused.** On the training distribution the two architectures are
**indistinguishable**: their gap is smaller than the spread across training
seeds, and `compare_families` says so in the tool output rather than leaving it
to prose.

An earlier version of this project did claim BDQ won, off a 0.9% gap from one
training run each. Re-running on a second machine **flipped the sign of that
gap**. That is why every learned number here carries a seed spread, and why the
comparison tool refuses to call a difference real when it sits inside the noise.

**Reported against us.** Three things count against the system and are in the
tables above:

1. **Both learned agents spend more retries than the static baselines** — 1,550
   and 1,751 against 1,426 and 1,446 — and the gap widens under shift. They buy
   extra recovery with extra attempts.
2. **BDQ is slower than the naive baseline.** 66.6h to recovery against 49.8h. A
   merchant with tight cash timing might reasonably prefer the static schedule.
3. **The architecture ranking flips between distributions.** BDQ leads on
   training data; the flat agent leads under shift, and degrades less (29.8% vs
   37.3%). Measuring only on the training distribution would have produced the
   wrong recommendation.

---

## Is the environment rigged?

It is synthetic, and the same person wrote the environment and the agent. That is
the sharpest objection this project faces, and it is answered by measurement
rather than assurance.

`configs/spec_holdout.yaml` is a held-out distribution no agent ever trains on,
and it is adversarial rather than noisy — it **inverts** the relationships an
agent is most likely to have learned:

- **Patience stops paying.** Waiting 72h on `insufficient_funds` is worth 4.1× in
  training and 2.1× here.
- **The channel preference flips.** For `do_not_honor`, training makes UPI beat
  the original instrument 2:1. In the held-out spec the original instrument wins.
- **Payday moves** to the 5th–7th and 20th–22nd, so `is_likely_payday` actively
  misleads rather than merely going quiet.
- **A decline code appears that did not exist in training** — `mandate_revoked`.
  It has no one-hot slot (it reads as all zeros), no channel-success history, and
  was never calibrated, so it falls through to the default high friction penalty.
  This is the realistic deployment failure: a processor ships a new decline reason
  and nobody retrains.
- **The population is harder**, with higher terminal probabilities throughout.

Each of these is pinned by a test in `tests/test_shift.py`, so "the shift is
adversarial" is checked, not asserted.

**The result:** the learned agents degrade by **33.5%**, the static baselines by
**33.9%**. If the agents had memorised the training spec they would have collapsed
while fixed schedules held steady. They did not.

Both ground-truth specs are committed as fully commented YAML so their assumptions
can be read and disputed directly.

---

## Where AI is used — and where it deliberately is not

| Layer | Approach | Why |
|---|---|---|
| Decision policy | Branching Dueling Q-Network | Sequential credit assignment across a multi-step retry episode. A classifier cannot learn a timing policy. |
| Reward calibration | LLM, **offline**, cached, human-reviewed | Semantic reading of unstructured decline text, with zero runtime latency or non-determinism |
| Guardrails | Deterministic rules | **Not AI, on purpose.** Compliance limits must be provable, not learned |
| Metrics | Plain arithmetic | **Not AI, on purpose.** Nothing generative goes near the scoreboard |
| Environment dynamics | Committed YAML | **Not AI, on purpose.** The world must be auditable by a reviewer |

### The LLM runs once, offline, and a human overrode it three times

`gemini-3.7-flash` read each raw decline-reason string — **and nothing else** — and
scored how much friction a retry carries. The numeric success probabilities never
enter the prompt; `test_prompt_contains_no_success_probabilities` walks every
probability, multiplier and terminal-probability in the spec and asserts none of
them appear as a substring. Without that, "LLM calibration" would just be
laundering the ground truth into the reward.

`CalibratedFrictionTable` raises `UnreviewedTableError` and **refuses to load**
until a human sets `review.reviewed`. Training cannot start against an unreviewed
table. The review found three errors:

| code | LLM | final | why |
|---|---|---|---|
| `expired_card` | 9.0 | 4.0 | "Retrying an expired instrument is guaranteed to fail" is true, but the penalty applies to *every* action on the code, including the rail switch that actually works |
| `transaction_limit_exceeded` | 6.0 | 3.5 | Misses that limits reset on their own; 6.0 penalised patience, the exact behaviour the agent should learn |
| `do_not_honor` | 7.0 | 5.0 | Sound for repeated same-instrument retries, but this is the largest recoverable segment and an alternate rail often clears it |

All three failed for the same root cause: the prompt asked how much friction
*"retrying this decline reason"* carries, and the model answered exactly that —
but the reward applies the number to every action on that code, including the
channel switch or delayed retry that defeats the decline reason. **The model was
right; the prompt was underspecified.** The principled fix is action-conditional
friction, `penalty(decline_code, action)`, and it is listed as future work rather
than rushed.

Both the model's raw output and every human override are kept permanently in
`penalty_table.json`, so the diff is inspectable.

---

## Bounded and gated

`PolicyGuard` sits between the learned policy and the executor. Nothing in it is
learned, and no Q-value can override it.

Rules are split into classes, and the distinction is not cosmetic:

- **COMPLIANCE** — `suspected_fraud` is hard-blocked. This is not an
  expected-value question. It must not happen even if the policy is confident it
  would pay off, so it is not left to the policy.
- **FUTILITY** — `account_closed` cannot succeed on any rail. Ordinary economics
  a good agent could learn; enforced anyway so a half-trained or drifted policy
  still cannot burn money.
- **BUDGET** — max 4 attempts total, max 2 per channel.
- **VELOCITY** — after 2 attempts, timing is floored at +24h. This *escalates*
  rather than vetoing: a good retry proposed too early should be delayed, not
  discarded.

`test_fraud_retry_is_always_blocked` sweeps every macro action, timing and attempt
count and asserts the result is always ABANDON. That test is the safety claim
written down. `issuer_risk_exposure = 0` across every run is the same claim
measured end to end.

The agents train **without** the guard in the loop. Training behind it would teach
the policy that fraud retries are impossible rather than undesirable, and the
moment the guard changed the policy would be wrong. So it learns the economics
itself, and the guard stays a genuine independent check at evaluation time.

---

## Audit trail

One JSON record per decision, in `results/audit_*.jsonl`. Each record captures the
action the policy **proposed** alongside the action it was **allowed** — because
an audit log that only records what happened cannot answer the question a reviewer
actually has.

```json
{
  "decline_code": "suspected_fraud",
  "proposed": {"macro": "RETRY", "timing": "PLUS_24H", "channel": "UPI"},
  "guard": {
    "intervened": true,
    "rule": "compliance_blocked_code",
    "rule_class": "COMPLIANCE",
    "reason": "'suspected_fraud' is on the compliance block list. Retrying an
               authorisation the issuer flagged for risk is never permitted,
               regardless of expected value."
  },
  "final": {"macro": "ABANDON", "timing": "PLUS_24H", "channel": null},
  "reward_breakdown": {"abandon": 0.06, "total": 0.06}
}
```

---

## Architecture

```
configs/          ground-truth specs (train + held-out) and runtime config, all committed YAML
src/hrapay/
  env/            spec validation, seeded episode generator, Gymnasium environment
  rewards/        offline LLM calibrator, review-gated friction table, reward function
  agents/         static baselines, flat dueling DQN, branching dueling Q-network
  guard/          PolicyGuard — deterministic limits outside the learned policy
  execution/      Executor interface + simulated implementation
  audit/          JSONL decision log
  eval/           shared episode runner, metrics, comparison, shift report
app/dashboard.py  Streamlit: episode explorer, results, calibration, audit trail
tests/            91 tests
```

**The action space.** Factored into three branches — macro `{RETRY,
SWITCH_CHANNEL, ABANDON}`, timing `{NOW, +2h, +6h, +24h, +72h}`, channel `{UPI,
Credit, Debit, Netbanking, Wallet}` — following Tavakoli et al. (2018). Output
size grows additively rather than multiplicatively:

```
flat head    1 + 5 + (5 x 5)  = 31 outputs
branched     3 + 5 + 5        = 13 outputs
```

At 31 actions the flat enumeration is still tractable, which is why the two
perform comparably here. The advantage is a scaling property: a sixth payment
rail takes flat to 37 and branched to 14; a second decision dimension multiplies
one and merely extends the other.

**One shared code path.** Every policy — static, flat, branched — implements the
same `Policy` interface and runs through the same `EpisodeRunner`, the same guard,
the same executor and the same audit log. A baseline on its own code path is not
a baseline.

---

## Reproducibility

- The environment, the episode generator and both static baselines are pure NumPy
  and reproduce **bit-identically** across machines. Verified on Windows and Linux.
- Anything touching a neural network does **not**. `torch.manual_seed` does not
  guarantee an identical RNG stream across torch builds, so the learned agents'
  exact figures differ between machines. This is why every learned number is
  reported as mean ± std across seeds, and why the conclusions are stated as gaps
  rather than digits.

---

## Limitations

- **Synthetic data.** Results are demonstrated on a synthetic environment with
  assumed success-probability distributions, not on live Razorpay transactions.
  The specs encoding those assumptions are committed in full so they can be
  disputed directly, and the held-out spec exists to test whether the agent
  learned something transferable — but no synthetic result is evidence about
  production.
- **The executor is simulated.** `Executor` is an interface with a simulated
  implementation; a Razorpay test-mode implementation is designed for but not
  wired. Nothing in the decision path depends on which is used.
- **Friction is per-decline-code, not per-action.** The known flaw that caused all
  three human overrides. `penalty(decline_code, proposed_action)` is the fix.
- **DQN training is unstable here.** Greedy return peaks between steps 5k–25k and
  degrades afterwards, so training keeps the best checkpoint by periodic greedy
  evaluation rather than the last. The cause is not fully diagnosed — most likely
  the replay buffer narrowing once exploration stops. Recorded as a hypothesis.
- **Three seeds is few.** Enough to show the architecture gap is inside the noise;
  not enough to resolve a small real difference.

---

## What broke

`ENGINEERING_LOG.md` is a day-by-day record of what went wrong and what changed
because of it. The entries worth reading:

- **The guard silently crippled the baseline.** A mean time-to-recovery of exactly
  24.00h — a suspiciously round number in a column I nearly skipped — revealed
  that the original failed authorisation was consuming half of every channel's
  *retry* budget. I had handicapped the baseline my own agent had to beat, in the
  direction that flattered my agent.
- **The flat-vs-branched comparison was rigged by exploration.** Uniform-random
  over 31 flat actions picks ABANDON 3.2% of the time; uniform over three branches
  picks it 33%. The two agents were exploring different worlds. Fixing it cost the
  flat agent 0.07 of greedy return — its earlier numbers were flattered by a scheme
  the branched agent structurally could not use.
- **The last checkpoint was 37% worse than the best one.**
- **The comparison tool reported the loser as the winner.** It assumed higher is
  always better and announced `flat_dqn better` on wasted retries where flat spent
  more. An automated verdict that is confidently backwards is worse than none.
- **Three API failures in a row** — retired model, depleted quota, then a 503 —
  before the calibration call succeeded. The script now retries with backoff across
  a chain of models, and distinguishes transient failures from permanent ones.
- **I nearly edited the spec to make a test pass**, missing a threshold by 1%.

---

## References

1. Tavakoli, A., Pardo, F., & Kormushev, P. (2018). *Action Branching Architectures
   for Deep Reinforcement Learning.* AAAI 32(1). arXiv:1711.08946 — the branched
   architecture.
2. Qu, B., et al. (2025). *LLM-Enhanced Self-Evolving Reinforcement Learning for
   Multi-Step E-Commerce Payment Fraud Risk Detection.* arXiv:2509.18719 — the
   reward-shaping philosophy is adapted; its evolutionary reward-authoring loop is
   **not** replicated.
3. Mnih, V., et al. (2015). *Human-level control through deep reinforcement
   learning.* Nature 518 — the base DQN algorithm.

The novelty claimed here is application-level: a hierarchical, branched action
space applied to the joint retry-timing and channel decision, with an LLM used as
an offline, human-reviewed decline-reason calibrator. Neither underlying technique
is new, and both are cited.
