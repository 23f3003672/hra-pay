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

---

## Day 3 — 30 Aug — LLM reward calibration and the flat DQN

**Goal:** replace the placeholder friction table with a calibrated one, and get
the first learned policy beating the baselines.

### The calibration prompt is the place this project could quietly cheat

The friction penalty is supposed to come from an LLM *reading decline text*. The
easy version of that is to hand the model everything I know about each decline
code and let it produce a sensible number. That would have been worthless: the
spec contains the ground-truth success probabilities, so a prompt containing
them would make "LLM calibration" a laundering step that copies the answer into
the reward function. Every downstream result would be circular.

So the prompt is built from `raw_reason_text` and nothing else — the free-text
string a gateway actually returns. `test_prompt_contains_no_success_probabilities`
walks every `base_success` value, every `time_multiplier` and every
`terminal_prob` in the spec and asserts none of them appear as a substring of
the prompt. It is the most important test in the repo, because it is the one
protecting against a mistake that would still *look* like it worked.

`--dry-run` prints the exact prompt so a reviewer can check this by eye rather
than taking the test's word for it.

### The review gate is enforced in code, not by good intentions

`CalibratedFrictionTable` raises `UnreviewedTableError` and refuses to load
unless `review.reviewed` is true. Training cannot start against an unreviewed
table. This matters because "a human reviews the LLM output" is the kind of
claim every project makes in a slide and almost none enforce — and the value of
the claim is entirely in the enforcement.

The table keeps the model's raw output under `llm_raw` permanently, alongside
any `human_override` with a written reason. The diff between them is the
artefact: it shows where a person actually disagreed with the model, rather than
asserting that review happened.

### Unknown decline codes default to expensive, not free

`penalty_for` returns 7.0 for a decline code the table has never seen, not 0.0.
Zero would have been the natural default and it is the wrong one: an
uncalibrated reason is one nobody has assessed, and assuming an unassessed
reason is safe to retry gets the asymmetry backwards. This is not hypothetical —
the Day 7 held-out environment introduces a decline code the table was never
calibrated on, specifically to exercise this path.

### Getting a working LLM call took three failures

The calibration call failed three times in a row, each for a different reason,
and each one changed the code.

**404 NOT_FOUND.** `gemini-2.5-flash` is retired for newly issued API keys. The
error helpfully suggested `gemini-3.6-flash` — which was not in the list of
models the key could actually reach, so following the error message would have
failed differently. Switched to `gemini-flash-latest`, a stable alias that was
in the list. That trades a pinned model for a floating one, so the script now
also records `response.model_version` — whatever concretely answered — as the
table's `source`. The request target floats so the script keeps working; the
provenance stays pinned so the table says exactly what produced it.

**429 RESOURCE_EXHAUSTED, prepayment credits depleted.** The AI Studio project
was on prepaid billing with a zero balance. Fixed outside the code by creating a
fresh project, which lands on the free tier.

**503 UNAVAILABLE, high demand.** This one was my fault. The script made a
single call to a third-party API with no retry and no alternative, so a
transient capacity spike on Google's side blocked the entire build. That is a
bad pattern anywhere and an embarrassing one in a project judged partly on
failure recovery.

Rewritten with exponential backoff (4 attempts, 2/4/8s plus jitter) across a
chain of five models. Crucially it distinguishes transient from permanent
failures: a 503 is retried, but a depleted quota or a retired model id fails
immediately and moves to the next candidate. Burning forty seconds of backoff
on an error that will never clear is worse than failing fast. The next run
logged three transient failures and then succeeded, served by `gemini-3.7-flash`.

### The human review found three errors, all caused by my prompt

The gate did its job, which was genuinely surprising — I had half expected to
read eight sensible numbers and rubber-stamp them.

Five entries were accepted as scored. Three were overridden, and all three
failed for the same reason:

| code | LLM | final | defect |
|---|---|---|---|
| `expired_card` | 9.0 | 4.0 | "Submitting on an expired instrument is guaranteed to fail" — true, but a switch to another rail does not touch the expired card, and that is the recovery that works |
| `transaction_limit_exceeded` | 6.0 | 3.5 | "Retrying before the cardholder modifies their limit will fail" — misses that limits reset on their own, so 6.0 penalised patience |
| `do_not_honor` | 7.0 | 5.0 | Sound for repeated same-instrument retries, but this is the largest recoverable segment and an alternate rail often clears it |

The model was not wrong. **My prompt was underspecified.** It asked how much
friction "retrying this decline reason" carries, and the model answered exactly
that — but the reward function applies the returned number to *every* action on
that code, including the channel switch or delayed retry that specifically
defeats the decline reason. Same number, two different questions.

The principled fix is action-conditional friction: `penalty(decline_code,
proposed_action)` rather than `penalty(decline_code)`. That is a real change to
the reward interface and is recorded as future work rather than rebuilt with
five days left. The overrides correct for it in the meantime, each with the
defect written down.

I deliberately did NOT override `insufficient_funds` at 4.0, though I would have
guessed 2.0. Overriding on a difference of opinion rather than an identified
error would make the override log meaningless — its whole value is that every
entry in it has a specific defect behind it.

### The last checkpoint was 37% worse than the best one

First trained run, reading the printed curve:

```
step 30000  mean return 0.6132
step 45000  mean return 0.5174
step 60000  mean return 0.4007
```

Return climbing to step 30k and then falling away, with epsilon already at its
0.05 floor since step 30k — so this was not exploration noise. And the trainer
was saving the *final* network.

Two things were wrong. First, the statistic: training return is measured with
exploration switched on and mixes recoverable episodes with hopeless ones, so it
is a poor guide to how the deployed greedy policy behaves. Second, and worse,
model selection: keeping the last checkpoint assumes monotone improvement, which
DQN does not provide.

Added a periodic greedy evaluation over 300 episodes on a fixed seed block the
training loop never visits, and kept the best-scoring network rather than the
last. The instrumented run made the size of the problem obvious:

```
step 30000  train 0.5594  greedy 0.5556  <- best
step 45000  train 0.5092  greedy 0.4471
step 60000  train 0.4204  greedy 0.3486
best greedy return 0.5556 at step 30,000 (final step scored 0.3486)
```

The checkpoint being shipped was 37% worse than one the same run had already
passed through. Fixing selection improved every headline metric at once,
including the one the agent had been losing on:

```
                  recovered_inr  recovery_rate  wasted  correct_abandon
last checkpoint       1,008,673        0.654    1,835        0.460
best checkpoint       1,050,257        0.702    1,552        0.497
```

Why the decline happens at all is not fully diagnosed — most likely the replay
buffer filling with on-policy data once epsilon floors, narrowing the state
distribution the network is fit on. Worth noting honestly rather than claiming a
diagnosis I have not earned. Best-checkpoint selection is the right engineering
answer regardless, and it now protects the branched agent too.

### Where the learned agent is WORSE, and why it is being reported

Final Day-3 comparison, 1,000 episodes x 3 seeds:

```
policy                recovered_inr  recovery_rate  wasted  issuer_risk  ttr_h
flat_dqn                  1,050,257        0.702    1,552        0       43.8
static_with_switch          880,377        0.583    1,446        0       49.8
static_schedule             536,078        0.346    1,426        0       63.5
```

`wasted_attempts`: 1,552 against the best baseline's 1,446. The agent still
spends about 7% more retries than the static schedule to buy its extra 12 points
of recovery rate. Smaller than the 27% gap before checkpoint selection was
fixed, but real, and whether it is a good trade depends on issuer economics this
synthetic environment only approximates. It goes in the README table and in the
video. Reporting only `recovered_inr` and `recovery_rate` would hide it.

### End of day

66 tests passing. The flat DQN is a working, evaluated fallback submission — if
the branched agent goes wrong over the next two days, a complete three-policy
comparison already exists.

Tomorrow: the Branching Dueling Q-Network, which has to beat 0.702 on the same
seeds, the same guard and the same hyperparameters, using 13 output units
instead of 31.
