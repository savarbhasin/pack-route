"""Run joint vs pack-then-route on 10 instances (one per real route) and write results.

usage: uv run python run_experiment.py DATA_DIR OUT_JSON [--stops 14] [--vans 3] [--tightness 0.9]
                                      [--sampling cluster|spread]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from pack_route.core import (check_plan, load_raw, make_instance, pack_ffd, pack_stop_aware,
                        reward, route_given_packing, solve_joint)

TIE_SEEDS = range(5)


def run_one(inst, time_limit):
    out = {"name": inst.name, "route_id": inst.route_id, "stops": inst.n,
           "packages": len(inst.pkg_vol), "vans": inst.n_vans, "capacity_cm3": inst.capacity,
           "floor_s": inst.floor()}
    joint = solve_joint(inst, time_limit=time_limit)
    assert not check_plan(inst, joint), check_plan(inst, joint)
    out["joint"] = {"cost_s": joint.cost, **joint.meta,
                    "split_stops": _splits(inst, joint)}
    ref = joint.cost

    def grade(plan):
        errs = check_plan(inst, plan)
        assert not errs, errs
        rel, g, r = reward(inst, plan.cost, ref)
        return {"cost_s": plan.cost, "gap_rel": rel, "gap_g": g, "reward": r,
                "split_stops": _splits(inst, plan)}

    a = pack_ffd(inst)
    out["ffd"] = grade(route_given_packing(inst, a, {})) if a else {"infeasible": True}
    sa = []
    for seed in TIE_SEEDS:
        assign, _ = pack_stop_aware(inst, seed)
        sa.append(grade(route_given_packing(inst, assign, {})))
    out["stop_aware"] = sa
    return out


def _splits(inst, plan):
    vans_at = {}
    for p, k in enumerate(plan.assign):
        vans_at.setdefault(inst.pkg_stop[p], set()).add(k)
    return sum(1 for v in vans_at.values() if len(v) > 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--stops", type=int, default=14)
    ap.add_argument("--vans", type=int, default=3)
    ap.add_argument("--tightness", type=float, default=0.9)
    ap.add_argument("--sampling", choices=["cluster", "spread"], default="cluster")
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=300)
    args = ap.parse_args()

    routes, pkgs, tts = load_raw(args.data_dir)
    results = []
    for i, rid in enumerate(sorted(routes)[: args.instances]):
        inst = make_instance(rid, routes[rid], pkgs[rid], tts[rid],
                             args.stops, args.vans, args.tightness, seed=i,
                             sampling=args.sampling)
        t0 = time.perf_counter()
        r = run_one(inst, args.time_limit)
        j, f = r["joint"], r["ffd"]
        sa = [s["reward"] for s in r["stop_aware"]]
        print(f"{inst.name}: pkgs={r['packages']} joint={j['cost_s']}s {j['status']} "
              f"prove={j['seconds_total']:.1f}s best@{j['seconds_to_best']:.1f}s | "
              f"ffd gap={f.get('gap_rel', float('nan')):.1%} r={f.get('reward', float('nan')):.3f} | "
              f"stop-aware r mean={statistics.mean(sa):.3f} min={min(sa):.3f} max={max(sa):.3f} "
              f"[{time.perf_counter() - t0:.0f}s]", flush=True)
        results.append(r)
        args.out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                        "results": results}, indent=1))


if __name__ == "__main__":
    main()
