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
