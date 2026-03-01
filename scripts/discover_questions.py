"""
Two-stage question discovery for PR acceptance prediction.

Stage 1: Compare N pairs of (accepted, rejected) PRs side-by-side and ask the LLM
to identify observable features that distinguish each pair.
Results saved to cache/llm_pair_observations.json.

Stage 2: Aggregate all raw observations into canonical yes/no or scored questions
that can be applied to any single PR. Saves to cache/llm_questions.json.

Usage:
    # Run both stages in sequence:
    python src/discover_questions.py --stage 1 [--n-pairs 50] [--data dataset/prs.jsonl]
    python src/discover_questions.py --stage 2

    # Default (stage 1):
    python src/discover_questions.py [--n-pairs 50] [--data dataset/prs.jsonl]
"""

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import env  # loads .env before any OpenAI client is created

QUESTIONS_PATH       = ROOT / "cache" / "llm_questions.json"
OBSERVATIONS_PATH    = ROOT / "cache" / "llm_pair_observations.json"

PAIR_SYSTEM_PROMPT = """\
You compare two GitHub pull requests and identify concrete, observable differences between them.
Output ONLY a JSON array of short feature strings. No prose, no markdown.
Each string names one specific, measurable feature visible in the title or body text.
Example output: ["PR A has a checklist, PR B does not", "PR A references a specific issue number", "PR B body is empty"]"""

PAIR_USER_TEMPLATE = """\
PR A:
Title: {title_a}
Body:
{body_a}

---

PR B:
Title: {title_b}
Body:
{body_b}

List 6-10 specific, observable features that distinguish these two PRs. Output a JSON array of strings."""


def parse_dt(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_pairs(data_path: Path, n_pairs: int, deadline_days: int = 14, seed: int = 42):
    """
    Load n_pairs of (accepted, rejected) PRs. Each pair uses a distinct PR.
    Stratifies by body length for diversity.
    """
    deadline_secs = deadline_days * 86400
    rng = random.Random(seed)

    accepted, rejected = [], []
    with open(data_path) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("isCrossRepository"):
                continue
            merged = r.get("merged")
            merged_at = parse_dt(r.get("mergedAt"))
            created = parse_dt(r.get("createdAt"))
            if (merged and merged_at and created
                    and (merged_at - created).total_seconds() <= deadline_secs):
                accepted.append(r)
            elif r.get("state") == "CLOSED":
                rejected.append(r)

    def stratified_sample(prs, n):
        short  = [r for r in prs if len(r.get("body") or "") < 200]
        medium = [r for r in prs if 200 <= len(r.get("body") or "") < 1500]
        long_  = [r for r in prs if len(r.get("body") or "") >= 1500]
        per_tier = max(1, n // 3)
        sampled = (
            rng.sample(short,  min(per_tier, len(short)))
            + rng.sample(medium, min(per_tier, len(medium)))
            + rng.sample(long_,  min(n - 2 * per_tier, len(long_)))
        )
        rng.shuffle(sampled)
        return sampled[:n]

    acc_sample = stratified_sample(accepted, n_pairs)
    rej_sample = stratified_sample(rejected, n_pairs)
    return list(zip(acc_sample, rej_sample))


def _get_client():
    if not hasattr(_get_client, "_client"):
        import httpx
        from openai import OpenAI
        _get_client._client = OpenAI(
            http_client=httpx.Client(
                limits=httpx.Limits(max_connections=60, max_keepalive_connections=50)
            )
        )
    return _get_client._client


def _call_pair(pair_idx: int, pr_a: dict, pr_b: dict) -> tuple[int, list[str] | None, int, int]:
    """Compare one pair and return (pair_idx, observations, in_tok, out_tok)."""
    try:
        client = _get_client()
        user_msg = PAIR_USER_TEMPLATE.format(
            title_a=(pr_a.get("title") or "")[:300],
            body_a=(pr_a.get("body") or "")[:800],
            title_b=(pr_b.get("title") or "")[:300],
            body_b=(pr_b.get("body") or "")[:800],
        )
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": PAIR_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_completion_tokens=8000,
        )
        choice = response.choices[0]
        raw = choice.message.content or ""

        if not raw.strip():
            print(f"  [pair {pair_idx}] Empty response (finish={choice.finish_reason})",
                  file=sys.stderr)
            return pair_idx, None, 0, 0

        text = raw.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:]).rstrip("`").strip()

        observations = json.loads(text)
        usage = response.usage
        in_tok  = usage.prompt_tokens     if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        return pair_idx, observations, in_tok, out_tok

    except Exception as e:
        print(f"  [pair {pair_idx}] Failed: {e}", file=sys.stderr)
        return pair_idx, None, 0, 0


def run_stage1(data_path: Path, n_pairs: int = 50, max_workers: int = 50, budget_usd: float = 2.0):
    from ml.llm_features import INPUT_COST_PER_1K, OUTPUT_COST_PER_1K

    # Load existing observations to support restart
    existing = {}
    if OBSERVATIONS_PATH.exists():
        with open(OBSERVATIONS_PATH) as f:
            existing = json.load(f)
        print(f"[stage1] Resuming — {len(existing)} pairs already done")

    print(f"[stage1] Loading {n_pairs} pairs...")
    pairs = load_pairs(data_path, n_pairs)

    # Skip already-completed pairs
    todo = [(i, a, r) for i, (a, r) in enumerate(pairs) if str(i) not in existing]
    print(f"[stage1] {len(todo)} pairs to process ({len(pairs) - len(todo)} cached)")

    results = dict(existing)
    total_cost = 0.0
    completed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_call_pair, i, a, r): i for i, a, r in todo}

        for future in as_completed(futures):
            pair_idx, observations, in_tok, out_tok = future.result()
            call_cost = (in_tok * INPUT_COST_PER_1K + out_tok * OUTPUT_COST_PER_1K) / 1000

            with lock:
                if observations is not None:
                    results[str(pair_idx)] = {
                        "accepted_pr": pairs[pair_idx][0]["number"],
                        "rejected_pr": pairs[pair_idx][1]["number"],
                        "observations": observations,
                    }
                completed += 1
                total_cost += call_cost

                if completed % 10 == 0:
                    # Save incrementally
                    OBSERVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    tmp = OBSERVATIONS_PATH.with_suffix(".tmp")
                    with open(tmp, "w") as f:
                        json.dump(results, f, indent=2)
                    tmp.replace(OBSERVATIONS_PATH)
                    print(f"  [{completed}/{len(todo)}] cost so far: ${total_cost:.4f} / ${budget_usd:.2f}")

            if total_cost >= budget_usd:
                print(f"[stage1] Budget ${budget_usd:.2f} reached. Stopping.")
                executor.shutdown(wait=False, cancel_futures=True)
                break

    # Final save
    OBSERVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OBSERVATIONS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    successful = sum(1 for v in results.values() if v.get("observations"))
    print(f"\n[stage1] Done. {successful}/{n_pairs} pairs successful. "
          f"Total cost: ${total_cost:.4f}")
    print(f"[stage1] Observations saved to {OBSERVATIONS_PATH}")

    # Write human-readable observations file
    obs_txt_path = OBSERVATIONS_PATH.with_suffix(".txt")
    all_obs = []
    lines = []
    for i, v in sorted(results.items(), key=lambda x: int(x[0])):
        obs = v.get("observations") or []
        lines.append(f"Pair {i}  (accepted PR#{v['accepted_pr']} vs rejected PR#{v['rejected_pr']})")
        for o in obs:
            lines.append(f"  • {o}")
            all_obs.append(o)
        lines.append("")

    lines.append(f"---\nTotal: {len(all_obs)} observations across {successful} pairs")
    obs_txt_path.write_text("\n".join(lines))
    print(f"[stage1] Readable observations saved to {obs_txt_path}")
    print(f"\nTotal observations: {len(all_obs)} across {successful} pairs")
    return results


AGGREGATE_SYSTEM_PROMPT = """\
You are an expert at designing binary classification features for machine learning models.
You will be given a list of observations about differences between accepted and rejected GitHub pull requests.
Your task: abstract these observations into 12-15 canonical questions that can be applied to any single PR.
Output ONLY a JSON object with key "questions" containing an array of question objects. No prose, no markdown.
Each question object must have exactly:
  "id": snake_case identifier starting with q_ (e.g. "q_body_structured")
  "question": clear question about a single PR (not a comparison)
  "type": one of "bool" (yes/no answer) or "score" (integer 1-5)"""

AGGREGATE_USER_TEMPLATE = """\
Here are {n_obs} observations collected by comparing {n_pairs} pairs of accepted vs rejected pull requests:

{obs_block}

Based on these observations, write 12-15 canonical questions for scoring any single PR.
Requirements:
- Focus on patterns that appear across multiple pairs
- Phrase each question for a single PR (not comparisons between two PRs)
- Mix of bool and score types
- Cover: body completeness/structure, problem statement clarity, scope/complexity, automation signals, concrete examples
- Avoid redundancy — each question should capture a distinct dimension

Output format example:
{{"questions": [{{"id": "q_body_structured", "question": "Does the PR body have structured sections (e.g. Summary, Testing, Problem)?", "type": "bool"}}, ...]}}"""


def run_stage2() -> list[dict]:
    """
    Aggregate raw pair observations into canonical questions.
    Saves to cache/llm_questions.json and returns the questions list.
    """
    if not OBSERVATIONS_PATH.exists():
        print("[stage2] ERROR: No observations found. Run stage 1 first.", file=sys.stderr)
        sys.exit(1)

    with open(OBSERVATIONS_PATH) as f:
        observations_data = json.load(f)

    # Flatten all observations
    all_obs = []
    for v in observations_data.values():
        all_obs.extend(v.get("observations") or [])

    n_pairs = len(observations_data)
    n_obs = len(all_obs)
    print(f"[stage2] Loaded {n_obs} observations from {n_pairs} pairs")

    # Format observations as a numbered list
    obs_block = "\n".join(f"{i+1}. {o}" for i, o in enumerate(all_obs))

    user_msg = AGGREGATE_USER_TEMPLATE.format(
        n_obs=n_obs, n_pairs=n_pairs, obs_block=obs_block
    )

    print(f"[stage2] Calling gpt-5-nano to aggregate observations into questions...")
    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": AGGREGATE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_completion_tokens=8000,
    )

    choice = response.choices[0]
    raw = choice.message.content or ""

    if not raw.strip():
        print(f"[stage2] Empty response (finish={choice.finish_reason})", file=sys.stderr)
        sys.exit(1)

    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:]).rstrip("`").strip()

    parsed = json.loads(text)
    questions = parsed["questions"]

    usage = response.usage
    in_tok  = usage.prompt_tokens     if usage else 0
    out_tok = usage.completion_tokens if usage else 0
    from ml.llm_features import INPUT_COST_PER_1K, OUTPUT_COST_PER_1K
    cost = (in_tok * INPUT_COST_PER_1K + out_tok * OUTPUT_COST_PER_1K) / 1000
    print(f"[stage2] {len(questions)} questions generated  "
          f"(tokens: {in_tok} in / {out_tok} out, cost: ${cost:.4f})")

    # Save to cache/llm_questions.json
    QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUESTIONS_PATH, "w") as f:
        json.dump({"questions": questions, "source": "stage2_aggregation",
                   "n_observations": n_obs, "n_pairs": n_pairs}, f, indent=2)
    print(f"[stage2] Questions saved to {QUESTIONS_PATH}")

    print("\nDiscovered questions:")
    for q in questions:
        print(f"  [{q['type']:5s}] {q['id']}: {q['question']}")

    return questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Which stage to run (default: 1)")
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--data", default="dataset/prs.jsonl")
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()

    if args.stage == 1:
        run_stage1(
            data_path=ROOT / args.data,
            n_pairs=args.n_pairs,
            max_workers=args.workers,
            budget_usd=args.budget,
        )
    else:
        run_stage2()
