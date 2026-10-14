"""Summarise a vf-eval run of pack_route: per-task and overall reward, valid rate, gaps.

usage: uv run python experiments/summarize_eval.py outputs/<run-dir> [...]
"""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def load(run: Path) -> list[dict]:
    """One trace per agent run; each traces.jsonl line is an episode holding its traces."""
    episodes = [json.loads(line) for line in (run / "traces.jsonl").read_text().splitlines() if line.strip()]
    return [tr for ep in episodes for tr in ep["traces"]]


def reward_of(tr: dict) -> float | None:
    r = tr.get("rewards", {}).get("plan_quality")
    return r.get("score") if isinstance(r, dict) else r


def summarize(run: Path) -> None:
    traces = load(run)
    by_task = defaultdict(list)
    for tr in traces:
        by_task[tr["task"]["data"]["id"]].append(tr)
    print(f"## {run.name}: {len(traces)} rollouts over {len(by_task)} tasks")
    rewards, rows = [], []
    for tid in sorted(by_task):
        rs, valid, gaps, checks, errs = [], 0, [], [], 0
        for tr in by_task[tid]:
            r = reward_of(tr)
            m = tr.get("metrics", {})
            if tr.get("errors"):
                errs += 1
            rs.append(r if r is not None else 0.0)
            valid += int(m.get("valid") or 0)
            checks.append(m.get("checks_used") or 0)
            if m.get("gap_pct") is not None:
                gaps.append(m["gap_pct"])
        rewards += rs
        rows.append((tid, rs, valid, len(by_task[tid]), gaps, checks, errs))
        print(f"  {tid[:22]:22s} rewards {[round(x, 3) for x in rs]}  valid {valid}/{len(rs)}  "
              f"gap% {[round(g, 1) for g in gaps]}  checks {checks}" + (f"  errors {errs}" if errs else ""))
    all_valid = sum(r[2] for r in rows)
    all_gaps = [g for r in rows for g in r[4]]
    spread = sum(1 for r in rows if max(r[1]) - min(r[1]) > 0.05)
    print(f"  mean reward {st.mean(rewards):.3f}  valid {all_valid}/{len(rewards)}  "
          f"median gap% (valid) {st.median(all_gaps):.1f}" if all_gaps else
          f"  mean reward {st.mean(rewards):.3f}  valid {all_valid}/{len(rewards)}")
    beats = defaultdict(int)
    for tr in traces:
        for k in ("beats_stop_aware", "beats_ffd"):
            beats[k] += int(tr.get("metrics", {}).get(k) or 0)
    print(f"  beats FFD {beats['beats_ffd']}/{len(traces)}  beats stop-aware {beats['beats_stop_aware']}/{len(traces)}  "
          f"tasks with reward spread >0.05 across attempts: {spread}/{len(rows)}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        summarize(Path(arg))
