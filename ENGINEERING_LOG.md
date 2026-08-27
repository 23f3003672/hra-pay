# Engineering log

What broke, what I changed, and what I'd do differently. Written as it happened,
not reconstructed at the end.

---

## Day 1 — 26 Aug — environment and spec

**Goal:** a synthetic world that is a fair test bed, not one rigged so the agent wins.

### Decision: the ground-truth spec is a committed, versioned YAML file

The temptation was to bury the success probabilities in Python constants. I put them
in `configs/spec_train.yaml` instead, with a comment above every block explaining the
issuer behaviour it encodes. The reason is uncomfortable but important: this project's
central weakness is that I designed the world my agent is evaluated in. The only honest
response is to make that world fully legible to a reviewer, so they can judge the
assumptions directly instead of taking my word for it.

### Problem: ABANDON was a dominated action

First cut of the MDP had no notion of a genuinely unrecoverable transaction. Every
episode was recoverable with some probability, which meant "keep retrying until you run
out of attempts" strictly dominated abandoning, and the stopping rule the track asks
for would have been untestable — the agent would learn to never abandon and score well
doing it.

**Fix:** added a latent per-episode `is_terminal` flag, sampled from a per-decline-code
`terminal_prob`. When true, the environment forces P(success) = 0 no matter what the
agent does. Critically, the agent cannot observe it — it only sees the decline code,
which is correlated with but does not determine terminality (`do_not_honor` is terminal
30% of the time, `insufficient_funds` 10%, `account_closed` always). So ABANDON becomes
a real inference problem under uncertainty rather than a lookup.

This is now asserted in `test_population_contains_both_recoverable_and_terminal`, which
fails if the terminal share drifts outside 5–45% — a degenerate population in either
direction would quietly invalidate every result downstream.

### Problem: `channel_success_prior` was going to leak the answer

The state includes a historical success rate per channel. My first instinct was to read
it straight off the spec's `base_success` table. That would have handed the agent the
ground-truth generative parameters as an input feature — it would have looked like
learning while actually being a lookup, and the flat-vs-branched comparison would have
been meaningless.

**Fix:** `estimate_channel_priors` computes it by Monte Carlo over 40k sampled episodes
and sampled outcomes, marginalised over timing, payday, tier and fatigue. So it carries
real sampling noise and is a genuinely lossy summary — exactly what a merchant could
compute from their own retry logs, and nothing more. Cached to `data/channel_priors.json`,
keyed on spec version so it invalidates automatically when the spec changes.

### Small thing that would have cost hours later

`SWITCH_CHANNEL` to the channel you are already on is meaningless. Left unhandled, the
agent would have had two different action encodings for the same behaviour, which
splits the Q-value estimate for that behaviour across two outputs and slows learning
for no reason. The environment now silently rewrites it to `RETRY` and records it as
such in the audit info. Covered by `test_switch_to_current_channel_is_recorded_as_retry`.

### Known issue, deferred to Day 2 — reward scale

Rewards are currently in rupees, so a single step ranges from about -600 to +2,300
depending on transaction size. That variance will make DQN training unstable: the TD
error is dominated by how large the transaction happened to be rather than by whether
the decision was good. Noted, not fixed today. Day 2 will express reward in units of
transaction value (fraction recovered) and report dashboard figures in rupees, so the
learning signal is scale-free but the headline number stays in money.

### End of day

- 18 tests passing, covering seed reproducibility, probability bounds, guaranteed
  termination, and each specific issuer behaviour the spec claims to model.
- `make env-demo` rolls out episodes under a random policy: 1/4 recovered, total
  reward -728. Random doing badly is the expected result and the first sanity check
  that the reward function is not accidentally generous.

---

## Day 2 — 27 Aug — reward, guardrails, audit trail, baselines

**Goal:** a complete decision pipeline and the first real recovered-revenue numbers.

### The guard silently crippled the baseline, and I nearly reported it as a result

First end-to-end run produced this:

```
static_schedule       recovery_rate 0.157   mean_time_to_recovery 24.00
static_with_switch    recovery_rate 0.565   mean_time_to_recovery 49.87
```

A mean time-to-recovery of *exactly* 24.00 hours is not a statistic, it is a
symptom. It meant `static_schedule` only ever recovered on its first retry,
despite having a three-attempt schedule.

Cause: the environment seeds `_attempts_by_channel[origin_channel] = 1` to
represent the original failed authorisation, which is correct — the issuer has
seen that attempt and retry fatigue must reflect it. But `PolicyGuard` was
budgeting against the same counter, with `max_attempts_per_channel = 2`. So the
original failure consumed half of every channel's retry budget before the agent
had done anything at all. The naive baseline, which never switches channel, got
exactly one retry and then hit the cap.

**Fix:** two separate counters with explicitly different jobs.
`_attempts_by_channel` includes the original failure and drives retry fatigue.
`_retries_by_channel` counts only agent-initiated retries and is what the guard
budgets against.

```
static_schedule       recovery_rate 0.346   mean_time_to_recovery 63.51
```

The honest reading: I had accidentally handicapped the baseline my agent is
supposed to beat, in the direction that flatters my agent, and the only reason I
caught it was a suspiciously round number in a column I nearly skipped. Every
comparison in this repo now runs through one shared `EpisodeRunner` for exactly
this reason — a baseline on its own code path is not a baseline.

### Reward: scale-free, but not amount-blind

Day 1 left rewards denominated in rupees, which would have made the TD error a
function of transaction size rather than decision quality. Reward is now
expressed as a fraction of transaction value, so a full recovery is worth 1.0
whether the payment is Rs 99 or Rs 99,000.

The subtlety worth stating: this does *not* make transaction amount irrelevant,
because the retry cost has a fixed component. A Rs 3 gateway fee is 3% of a
Rs 100 payment and 0.003% of a Rs 100,000 one, so the cost/benefit of retrying
genuinely does depend on size. That is real payment economics, and it is why
`amount_norm` stays a meaningful input feature rather than dead weight.
Pinned by `test_fixed_fee_makes_small_transactions_relatively_more_expensive`.

I also got that test's assertion wrong on the first pass — I asserted a >10x
cost ratio between small and large transactions, but the variable rate applies
at every size and floors the ratio at about 8.4x. The code was right and the
test was wrong, which is worth recording because it is the failure mode that
usually goes the other way.

### PolicyGuard: why the rules are classed, not just listed

The guard separates COMPLIANCE rules from FUTILITY ones, and the distinction is
not cosmetic. Retrying an authorisation the issuer flagged as fraud is not a
matter of expected value — it must not happen even if the policy is confident
it would pay off, so it cannot be left to the policy. Futility rules are
ordinary economics that a well-trained agent could in principle learn; they are
enforced anyway so that a half-trained or drifted policy still cannot burn money.

`test_fraud_retry_is_always_blocked` sweeps every macro action, timing and
attempt count and asserts the result is always ABANDON. That test is the
submission's safety claim written down. If it fails, the claim is not true.

Velocity is deliberately an escalation rather than a veto — it pushes an
over-eager retry out to a minimum gap instead of killing it, because a good
retry proposed too early should be delayed, not discarded.

### Audit trail records what was blocked, not just what happened

Each record carries the *proposed* action alongside the *final* one. An audit
log that only records what the system did cannot answer the question a reviewer
actually has — what did the model want to do, and what stopped it. `guard`
carries the rule that fired, its class, and a human-readable reason.

Two fields renamed for honesty after the fact: `q_values` became
`policy_diagnostics`, because a static schedule has no Q-values and calling its
rule string a Q-value would be a small lie in a file whose entire purpose is
being trustworthy. `reward_breakdown` was being written as an empty dict; it now
carries the per-term decomposition, because a single reward number is not
auditable.

### End of day

47 tests passing. First real numbers, 1,000 episodes x 3 seeds:

```
policy               recovered_inr   recovery_rate   wasted_attempts   issuer_risk   ttr_h
static_schedule         536,078          0.346            1,426             0        63.5
static_with_switch      880,377          0.583            1,446             0        49.8
```

`issuer_risk_exposure = 0` for both is the guard doing its job across 6,000
episodes. The channel-switching baseline is substantially stronger than the
naive one, which is why both are reported — comparing the learned agent only
against the weaker baseline would inflate its apparent advantage.

Tomorrow: LLM reward calibration, and the flat DQN that has to beat these.
