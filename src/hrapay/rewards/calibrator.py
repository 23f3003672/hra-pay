"""Offline LLM reward calibration.

Reads the unstructured decline-reason string for each decline code and asks a
model to score how much *friction* a retry on that reason carries — issuer
irritation, compliance exposure, and the likelihood the attempt is simply
wasted. The result is cached to a versioned JSON table that the reward function
reads at training and inference time.

Three properties matter more than the model choice:

1. **It runs once, offline.** No LLM call sits inside the decision path. A
   non-deterministic network call in the middle of a money decision would add
   latency, cost, and — worse — make the policy's behaviour unauditable.

2. **The model never sees the answer.** It is shown only `raw_reason_text`,
   the free-text string a gateway would actually return. The numeric success
   probabilities in the spec are never in the prompt. If they were, this would
   not be calibration, it would be laundering the ground truth into the reward.

3. **A human reviews it before it is trusted.** The raw model output is kept
   verbatim under `llm_raw`, and any human correction is recorded next to it as
   an explicit `human_override` with a reason. The diff between the two is the
   evidence that this is a gated design rather than a model marking its own
   homework.

    python -m hrapay.rewards.calibrator            # generate the table
    python -m hrapay.rewards.calibrator --dry-run  # show the prompt, call nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hrapay.env.spec import EnvSpec

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = ROOT / "configs" / "spec_train.yaml"
DEFAULT_OUT = ROOT / "src" / "hrapay" / "rewards" / "penalty_table.json"
# A stable alias rather than a pinned version: pinned Gemini model ids are
# retired for new API keys over time, and a calibration script that stops
# working six months from now is not reproducible either. The model that
# actually answered is recorded in the table as `resolved_model`, so the
# provenance stays pinned even though the request target does not.
DEFAULT_MODEL = "gemini-flash-latest"

# Tried in order. A single hosted model is a single point of failure, and this
# script had exactly that until a 503 during a demand spike blocked the whole
# build. Flash-class models first (this is a short structured-extraction task,
# not one that needs a Pro), then progressively different backends so that an
# outage on one family does not stop the calibration.
MODEL_FALLBACKS = [
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-pro-latest",
]

# Transient conditions worth retrying. Anything else (bad key, retired model,
# depleted credits) is a real failure and must surface immediately rather than
# being buried under a minute of pointless backoff.
RETRYABLE = ("503", "UNAVAILABLE", "500", "INTERNAL", "overloaded", "high demand")
MAX_ATTEMPTS = 4

SCHEMA_VERSION = "penalty_table/v1"

SYSTEM_PROMPT = """\
You are a payments risk analyst at an Indian payment gateway. You are given the \
free-text decline reason that an issuing bank returned for a failed card or UPI \
transaction. Your job is to score how much FRICTION a retry on that reason carries.

Friction is NOT the same as "will it succeed". Friction is the cost and risk of \
making the attempt at all:
  - issuer irritation and velocity throttling from repeated authorisations
  - compliance and fraud exposure
  - the likelihood the attempt is simply wasted spend

Score friction_penalty from 0 to 10:
   0-2   benign. Retrying is routine and carries essentially no risk.
   3-5   moderate. Some issuer irritation if repeated, but a normal thing to retry.
   6-8   high. Retrying is likely wasted and starts to damage the issuer relationship.
   9-10  severe. Retrying is a compliance or fraud risk, or is certainly futile.

Be strict about the top of the range. A reason indicating suspected fraud, a \
security review, or a permanently closed account must score 9 or 10.

Return one entry per decline reason. Keep each justification to one sentence that \
a compliance reviewer could read and agree or disagree with."""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "decline_code": {"type": "string"},
                    "friction_penalty": {"type": "number"},
                    "severity": {
                        "type": "string",
                        "enum": ["BENIGN", "MODERATE", "HIGH", "SEVERE"],
                    },
                    "retry_advisable": {"type": "boolean"},
                    "justification": {"type": "string"},
                },
                "required": [
                    "decline_code",
                    "friction_penalty",
                    "severity",
                    "retry_advisable",
                    "justification",
                ],
            },
        }
    },
    "required": ["entries"],
}


def build_user_prompt(spec: EnvSpec) -> str:
    """Assemble the prompt. Contains decline text only — never probabilities."""
    lines = ["Score the friction of retrying each of the following decline reasons.\n"]
    for code in spec.decline_code_names:
        reason = " ".join(spec.decline_codes[code].raw_reason_text.split())
        lines.append(f"decline_code: {code}\nreason: {reason}\n")
    return "\n".join(lines)


def prompt_fingerprint(system: str, user: str) -> str:
    return hashlib.sha256((system + "\n---\n" + user).encode("utf-8")).hexdigest()[:16]


def load_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("\"'")
                    break
    if not key:
        raise SystemExit(
            "No GEMINI_API_KEY found.\n"
            "Put it in a .env file at the project root as:\n"
            "    GEMINI_API_KEY=your-key-here\n"
            "Or run with --deterministic to use the documented fallback table."
        )
    return key


def _is_retryable(exc: Exception) -> bool:
    text = str(exc)
    return any(marker in text for marker in RETRYABLE)


def _generate(client: Any, model: str, system: str, user: str) -> tuple[dict[str, Any], str]:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.0,
        ),
    )
    resolved = getattr(response, "model_version", None) or model
    return json.loads(response.text), str(resolved)


def call_gemini(
    system: str, user: str, model: str, api_key: str, *, verbose: bool = True
) -> tuple[dict[str, Any], str]:
    """Call Gemini with backoff, then fall back across models.

    Retries only transient failures. A depleted quota or a retired model id is
    reported straight away — silently burning 40 seconds of backoff on an error
    that will never clear is worse than failing fast.
    """
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "google-genai is not installed. Run:\n    pip install google-genai"
        ) from exc

    client = genai.Client(api_key=api_key)
    candidates = [model] + [m for m in MODEL_FALLBACKS if m != model]
    last_error: Exception | None = None

    for candidate in candidates:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return _generate(client, candidate, system, user)
            except Exception as exc:  # noqa: BLE001 - the SDK raises broadly
                last_error = exc
                if not _is_retryable(exc):
                    if verbose:
                        print(f"  {candidate}: permanent failure, moving on ({exc})")
                    break
                if attempt == MAX_ATTEMPTS:
                    if verbose:
                        print(f"  {candidate}: still unavailable after {MAX_ATTEMPTS} attempts")
                    break
                delay = (2**attempt) + random.random()
                if verbose:
                    print(f"  {candidate}: transient failure, retrying in {delay:.1f}s")
                time.sleep(delay)

    available = ""
    try:
        names = [m.name for m in client.models.list()][:15]
        available = "\n\nModels this key can reach:\n  " + "\n  ".join(names)
    except Exception:  # noqa: BLE001
        available = "\n\n(Could not list models — the key may be invalid or blocked.)"
    raise SystemExit(
        f"Calibration failed on every candidate model.\nLast error: {last_error}{available}"
        f"\n\nIf this persists, run with --deterministic to use the documented "
        f"fallback table and keep moving."
    )


# --- deterministic fallback --------------------------------------------------

DETERMINISTIC_TABLE: dict[str, tuple[float, str, bool, str]] = {
    "insufficient_funds": (
        2.0,
        "BENIGN",
        True,
        "A balance shortfall is a temporary customer-side condition; retrying after a "
        "delay is standard practice and carries no compliance risk.",
    ),
    "issuer_unavailable": (
        1.0,
        "BENIGN",
        True,
        "A network timeout at the issuer reflects infrastructure, not the customer or "
        "the transaction, so a prompt retry is expected behaviour.",
    ),
    "authentication_failed": (
        3.0,
        "MODERATE",
        True,
        "A failed or expired OTP is usually a customer usability problem, though "
        "repeated authentication failures start to look like credential testing.",
    ),
    "transaction_limit_exceeded": (
        3.0,
        "MODERATE",
        True,
        "An account limit is a hard constraint until it resets, so immediate retries "
        "are wasted spend even though the reason itself is benign.",
    ),
    "do_not_honor": (
        5.0,
        "MODERATE",
        True,
        "A generic issuer refusal carries no diagnostic information, so repeated "
        "attempts on the same instrument irritate the issuer for an unclear payoff.",
    ),
    "expired_card": (
        6.0,
        "HIGH",
        False,
        "The instrument cannot be authorised again under any circumstances; only a "
        "different payment method can succeed.",
    ),
    "suspected_fraud": (
        10.0,
        "SEVERE",
        False,
        "The issuer's risk engine has blocked this authorisation, and retrying against "
        "an active fraud control is a compliance risk regardless of expected value.",
    ),
    "account_closed": (
        9.0,
        "SEVERE",
        False,
        "A closed account will never authorise again, so every further attempt is pure "
        "cost and avoidable issuer noise.",
    ),
}


def deterministic_entries(spec: EnvSpec) -> list[dict[str, Any]]:
    out = []
    for code in spec.decline_code_names:
        if code not in DETERMINISTIC_TABLE:
            raise SystemExit(
                f"No deterministic fallback defined for decline code '{code}'. "
                f"Add one to DETERMINISTIC_TABLE or run with an API key."
            )
        penalty, severity, advisable, why = DETERMINISTIC_TABLE[code]
        out.append(
            {
                "decline_code": code,
                "friction_penalty": penalty,
                "severity": severity,
                "retry_advisable": advisable,
                "justification": why,
            }
        )
    return out


# --- table assembly ----------------------------------------------------------


def build_table(
    spec: EnvSpec, raw_entries: list[dict[str, Any]], *, model: str, fingerprint: str
) -> dict[str, Any]:
    by_code = {e["decline_code"]: e for e in raw_entries}

    missing = set(spec.decline_code_names) - set(by_code)
    if missing:
        raise SystemExit(f"Calibration returned no entry for: {sorted(missing)}")

    entries: dict[str, Any] = {}
    for code in spec.decline_code_names:
        e = by_code[code]
        entries[code] = {
            "friction_penalty": float(e["friction_penalty"]),
            "severity": e["severity"],
            "retry_advisable": bool(e["retry_advisable"]),
            "justification": e["justification"].strip(),
            "llm_raw": {
                "friction_penalty": float(e["friction_penalty"]),
                "severity": e["severity"],
                "retry_advisable": bool(e["retry_advisable"]),
            },
            "human_override": None,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "spec_version": spec.version,
        "source": model,
        "prompt_fingerprint": fingerprint,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "review": {
            "reviewed": False,
            "reviewed_by": None,
            "notes": (
                "Set reviewed=true only after a human has read every justification "
                "and either accepted the penalty or recorded a human_override with a "
                "reason. The reward function refuses to load an unreviewed table."
            ),
        },
        "entries": entries,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline LLM reward calibration.")
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true", help="print the prompt, call nothing")
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="use the documented hand-written table instead of an LLM",
    )
    args = ap.parse_args()

    spec = EnvSpec.load(args.spec)
    user_prompt = build_user_prompt(spec)
    fingerprint = prompt_fingerprint(SYSTEM_PROMPT, user_prompt)

    if args.dry_run:
        print("=== SYSTEM ===")
        print(SYSTEM_PROMPT)
        print("\n=== USER ===")
        print(user_prompt)
        print(f"\nfingerprint: {fingerprint}")
        print("\nNote: no success probabilities appear above. That is the point.")
        return

    if args.deterministic:
        entries = deterministic_entries(spec)
        source = "deterministic_fallback/v1"
    else:
        payload, resolved = call_gemini(SYSTEM_PROMPT, user_prompt, args.model, load_api_key())
        entries = payload["entries"]
        source = resolved
        if resolved != args.model:
            print(f"requested '{args.model}' -> served by '{resolved}'")

    table = build_table(spec, entries, model=source, fingerprint=fingerprint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")

    print(f"source: {source}")
    print(f"wrote:  {args.out}\n")
    print(f"{'decline_code':<28}{'penalty':>9}  {'severity':<10}{'retry?':<8}")
    print("-" * 62)
    for code, e in table["entries"].items():
        print(
            f"{code:<28}{e['friction_penalty']:>9.1f}  {e['severity']:<10}"
            f"{'yes' if e['retry_advisable'] else 'no':<8}"
        )
    print("\nNow REVIEW it. Read every justification, override anything that looks")
    print("wrong, and set review.reviewed = true. Training will refuse to start until")
    print("you do — an unreviewed table is a model marking its own homework.")


if __name__ == "__main__":
    main()
