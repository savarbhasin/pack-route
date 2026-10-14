"""Build the pack_route v1 dataset from per-route files (see prepare_amazon.py).

usage: uv run python -m pack_route.build_dataset ROUTES_DIR OUT_DIR
           [--train-per-city 250] [--test 100] [--test-city DAU] [--jobs 2] [--workers 2]
           [--time-limit 180] [--stall 30] [--seed 0]

Splits by city so the test set is a city the model never trained on: train from Los Angeles
(DLA), Seattle (DSE), Chicago (DCH) and Boston (DBO), test from Austin (DAU). Each route yields at
most one candidate; its size (and van count) is drawn from SIZES and its capacity fill from
TIGHTNESS, so sizes are mixed within every split.

At 20-40 stops CP-SAT rarely proves the optimum on a laptop, so the reference is the anchor: the
joint search stops once its best plan has not improved for --stall seconds (or at --time-limit),
and the cheapest plan known becomes the reference, recorded with `optimality` and the solver's lower
bound. A candidate is kept when packing first clearly loses to it (the best stop-aware
pack-then-route plan grades <= MAX_STOP_AWARE_REWARD).

Stop any time:
- `touch OUT_DIR/STOP` lets running candidates finish, then exits (delete the file to resume);
- Ctrl-C / kill also works; at most the in-flight candidates are lost.
Every decision is appended to OUT_DIR/{train,test,rejected}.jsonl the moment it is made, and a
re-run skips every route already decided.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from pack_route import core
from pack_route.build_tasks import build_one
from pack_route.plan import Evaluation, grade

TRAIN_CITIES = ["DLA", "DSE", "DCH", "DBO"]
SIZES = [(20, 3), (25, 3), (30, 4), (40, 4)]   # (stops, vans), drawn uniformly
TIGHTNESS = (0.85, 0.95)
MAX_STOP_AWARE_REWARD = 0.9


def candidate(route_file: Path, split: str, seed: int) -> dict:
    """Deterministic task parameters for one route."""
    rid = route_file.name.removesuffix(".json.gz")
    rng = random.Random(f"{seed}:{rid}")
    stops, vans = rng.choice(SIZES)
    return {"route_id": rid, "file": str(route_file), "split": split, "stops": stops,
            "vans": vans, "tightness": round(rng.uniform(*TIGHTNESS), 2)}


def run_candidate(c: dict, time_limit: float, stall: float, workers: int) -> dict:
    t0 = time.perf_counter()
    with gzip.open(c["file"], "rt") as f:
        r = json.load(f)
    tt = {a: dict(zip(r["stops"], row)) for a, row in zip(r["stops"], r["travel_s"])}
    meta = {**{k: c[k] for k in ("route_id", "split", "stops", "vans", "tightness")},
            "station": r["route"]["station_code"], "city": r["route"]["station_code"][:3],
            "date": r["route"]["date_YYYY_MM_DD"]}
    drops = sum(1 for s, v in r["route"]["stops"].items() if v["type"] != "Station" and r["packages"].get(s))
    if drops < c["stops"]:
        return {"keep": False, "reason": f"route has only {drops} stops", **meta}
    inst = core.make_instance(c["route_id"], r["route"], r["packages"], tt, c["stops"], c["vans"],
                              c["tightness"], seed=0, sampling="spread", workers=workers)
    row = build_one(inst, time_limit, workers, stall_seconds=stall, require_proof=False)
    ref = row["reference"]
    meta |= {"seconds": round(time.perf_counter() - t0, 1), "optimality": ref["optimality"],
             "stop_aware_best_reward": round(reward_of(ref["stop_aware_best_s"], ref), 4)}
    if meta["stop_aware_best_reward"] > MAX_STOP_AWARE_REWARD:
        return {"keep": False, "reason": "pack-then-route already near the reference", **meta}
    return {"keep": True, "row": {**row, "meta": meta}, **meta}


def reward_of(cost: int, ref: dict) -> float:
    """The reward a valid plan of this cost gets (the environment's formula)."""
    ev = Evaluation(valid=True, errors=[], total_drive_s=cost, vans=[], clean_share=1.0)
    return grade(ev, ref["optimum_s"], ref["floor_s"])[0]


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file; a line cut short by a kill is skipped."""
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def append(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("routes_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--train-per-city", type=int, default=250)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--test-city", default="DAU")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--time-limit", type=float, default=180)
    ap.add_argument("--stall", type=float, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    files = {s: out / f"{s}.jsonl" for s in ("train", "test", "rejected")}
    stop_file = out / "STOP"

    quota = {c: args.train_per_city for c in TRAIN_CITIES} | {args.test_city: args.test}
    kept = {c: 0 for c in quota}
    decided = set()
    for s in ("train", "test"):
        for row in read_jsonl(files[s]):
            kept[row["meta"]["city"]] += 1
            decided.add(row["meta"]["route_id"])
    decided |= {r["route_id"] for r in read_jsonl(files["rejected"])}

    index = json.loads((args.routes_dir / "index.json").read_text())   # route -> station
    queues: dict[str, list[dict]] = {}
    for city in quota:
        paths = [args.routes_dir / f"{rid}.json.gz" for rid, st in sorted(index.items())
                 if st[:3] == city and rid not in decided]
        random.Random(f"{args.seed}:{city}").shuffle(paths)
        split = "test" if city == args.test_city else "train"
        queues[city] = [candidate(p, split, args.seed) for p in paths]
    print(f"resume: kept {kept}; candidates left "
          f"{ {c: len(q) for c, q in queues.items()} }; stop with: touch {stop_file}", flush=True)

    t_start = time.perf_counter()
    inflight: dict = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        def submit_next() -> bool:
            if stop_file.exists():
                return False
            busy = {c: sum(1 for v in inflight.values() if v["city"] == c) for c in quota}
            for city in sorted(quota, key=lambda c: kept[c] / max(1, quota[c])):   # least-filled first
                if kept[city] + busy[city] < quota[city] and queues[city]:
                    c = queues[city].pop(0)
                    inflight[pool.submit(run_candidate, c, args.time_limit, args.stall,
                                         args.workers)] = {"city": city, **c}
                    return True
            return False

        while len(inflight) < args.jobs and submit_next():
            pass
        while inflight:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                c = inflight.pop(fut)
                try:
                    res = fut.result()
                except Exception as e:   # one broken route must not stop the build
                    res = {"keep": False, "reason": f"error: {e!r}"[:300],
                           **{k: c[k] for k in ("route_id", "city", "split", "stops")}}
                if res["keep"] and kept[res["city"]] < quota[res["city"]]:
                    append(files[res["split"]], res["row"])
                    kept[res["city"]] += 1
                    status = "KEEP" + ("" if res["optimality"] == "proven" else " (anchor)")
                else:
                    if res["keep"]:
                        res |= {"keep": False, "reason": "quota already met"}
                    append(files["rejected"], {k: v for k, v in res.items() if k != "row"})
                    status = f"drop: {res['reason']}"
                print(f"[{(time.perf_counter() - t_start) / 60:6.1f} min] {res.get('city')} "
                      f"{res['route_id'][8:16]} n={res.get('stops')} {res.get('seconds', '?')}s "
                      f"{status} | kept {sum(kept.values())}/{sum(quota.values())} {kept}",
                      flush=True)
            while len(inflight) < args.jobs and submit_next():
                pass
    print(("stopped by STOP file" if stop_file.exists() else "finished") + f": kept {kept}",
          flush=True)


if __name__ == "__main__":
    main()
