"""Smoke-run the environment: roll out a few episodes under a random policy.

    python -m hrapay.env.demo --episodes 5

This exists so that the environment can be inspected on its own, before any
agent exists. If the dynamics look wrong here, no amount of training will fix it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hrapay.env.generator import estimate_channel_priors
from hrapay.env.retry_env import RetryEnv
from hrapay.env.spec import EnvSpec

DEFAULT_SPEC = Path(__file__).resolve().parents[3] / "configs" / "spec_train.yaml"
PRIORS_CACHE = Path(__file__).resolve().parents[3] / "data" / "channel_priors.json"


def load_priors(spec: EnvSpec, *, refresh: bool = False) -> dict[str, dict[str, float]]:
    if PRIORS_CACHE.exists() and not refresh:
        cached = json.loads(PRIORS_CACHE.read_text())
        if cached.get("spec_version") == spec.version:
            return cached["priors"]

    priors = estimate_channel_priors(spec)
    PRIORS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PRIORS_CACHE.write_text(json.dumps({"spec_version": spec.version, "priors": priors}, indent=2))
    return priors


def main() -> None:
    ap = argparse.ArgumentParser(description="Roll out RetryEnv under a random policy.")
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh-priors", action="store_true")
    args = ap.parse_args()

    spec = EnvSpec.load(args.spec)
    print(f"Loaded spec {spec.version}: {spec.description}")
    print(f"  {len(spec.decline_codes)} decline codes, {len(spec.channels)} channels")

    priors = load_priors(spec, refresh=args.refresh_priors)
    env = RetryEnv(spec, seed=args.seed, channel_priors=priors)
    print(f"  observation dim = {env.observation_space.shape[0]}")
    print(f"  action space    = {env.action_space}\n")

    env.action_space.seed(args.seed)
    recovered = 0
    total_reward = 0.0

    for i in range(args.episodes):
        obs, info = env.reset(seed=args.seed + i)
        print(f"--- episode {i + 1}: {env.render()}")
        print(f"    (oracle: terminal={info['episode']['is_terminal_ORACLE']})")

        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            target = info["target_channel"] or "-"
            print(
                f"    {info['macro']:<14} {info['timing']:<8} -> {target:<12} "
                f"p={info['p_success_ORACLE']:.3f}  r={reward:>10,.1f}  {info['outcome']}"
            )
        recovered += int(info.get("success", False))
        print()

    print(f"random policy: recovered {recovered}/{args.episodes}, total reward {total_reward:,.0f}")


if __name__ == "__main__":
    main()
