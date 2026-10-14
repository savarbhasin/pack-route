"""Build pack_route tasks: real routes -> instances with a proven joint optimum and baselines.

usage: uv run python -m pack_route.build_tasks DATA_DIR OUT_JSONL
           [--stops 14] [--vans 3] [--tightness 0.9] [--sampling spread] [--instances 10]
           [--seeds-per-route 1] [--time-limit 300] [--workers 2]

DATA_DIR holds new_route_data.json, new_package_data.json and new_travel_times.json from the
Amazon Last Mile Routing Challenge (s3://amazon-last-mile-challenges, CC BY-NC 4.0).
--instances takes the first M routes (sorted); each gets --seeds-per-route instances with
distinct seeds. Each output line is {"id", "instance", "reference"}; an instance whose optimum
is not proven within the time limit is skipped. Every CP-SAT solve uses --workers threads.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
from pathlib import Path

from pack_route import core
from pack_route.plan import Parcel, PublicInstance, VanPlan, evaluate

TIE_SEEDS = range(5)
BASELINE_SECONDS = 10   # per packing solve; a large instance keeps the best packing found


def to_decilitres(inst: core.Instance, workers: int) -> core.Instance:
    """Switch volumes to the integer decilitres the agent sees, keeping the parcels packable."""
    vols = [max(1, round(v / 100)) for v in inst.pkg_vol]
    cap = max(math.ceil(inst.capacity / 100), max(vols))
    while not core._packable(vols, inst.n_vans, cap, workers):
        cap = math.ceil(cap * 1.02)
    return dataclasses.replace(inst, pkg_vol=vols, capacity=cap)


def labels(inst: core.Instance) -> tuple[list[str], list[str]]:
    w_s, w_p = len(str(inst.n)), len(str(len(inst.pkg_vol)))
    return ([f"S{i:0{max(2, w_s)}d}" for i in range(1, inst.n + 1)],
            [f"P{i:0{max(2, w_p)}d}" for i in range(1, len(inst.pkg_vol) + 1)])


def to_public(inst: core.Instance) -> PublicInstance:
    stop_ids, parcel_ids = labels(inst)
    return PublicInstance(
        stops=stop_ids,
        parcels=[Parcel(id=parcel_ids[p], stop=stop_ids[inst.pkg_stop[p] - 1], volume_dl=v)
                 for p, v in enumerate(inst.pkg_vol)],
        vans=inst.n_vans, capacity_dl=inst.capacity, travel_s=inst.tt,
    )


def to_van_plans(inst: core.Instance, plan: core.Plan) -> list[VanPlan]:
    stop_ids, parcel_ids = labels(inst)
    visits: dict[int, int] = {}
    for tour in plan.tours:
        for s in tour:
            visits[s] = visits.get(s, 0) + 1
    vans = []
    for k, tour in enumerate(plan.tours):
        if not tour:
            continue
        split = [parcel_ids[p] for p, a in enumerate(plan.assign)
                 if a == k and visits[inst.pkg_stop[p]] > 1]
        vans.append(VanPlan(route=[stop_ids[s - 1] for s in tour], parcels=split))
    return vans


def checked_cost(public: PublicInstance, inst: core.Instance, plan: core.Plan) -> int:
    """The agent-facing validator must accept every reference plan at the solver's cost."""
    ev = evaluate(public, to_van_plans(inst, plan))
    assert ev.valid and ev.total_drive_s == plan.cost, (ev.errors, ev.total_drive_s, plan.cost)
    return plan.cost


def build_one(inst: core.Instance, time_limit: float, workers: int,
              stall_seconds: float | None = None, require_proof: bool = True) -> dict | None:
    """One task row, or None when `require_proof` and the optimum is not proven in time.

    Without a proof the reference is the anchor: the cheapest plan known (the joint search's
    best, or a baseline if that is cheaper), with `optimality` "best_found" and the solver's lower bound."""
    inst = to_decilitres(inst, workers)
    joint = core.solve_joint(inst, time_limit=time_limit, workers=workers,
                             stall_seconds=stall_seconds)
    if require_proof and not joint.meta["proved"]:
        return None
    public = to_public(inst)
    a = core.pack_ffd(inst)
    ffd = core.route_given_packing(inst, a, {}, workers) if a else None
    # the fewest-splits count is solved once, then each tie-break seed picks one such packing
    first, splits = core.pack_stop_aware(inst, TIE_SEEDS[0], workers, BASELINE_SECONDS)
    packings = [first] + [core.pack_stop_aware(inst, seed, workers, BASELINE_SECONDS,
                                               best_visits=inst.n + splits)[0]
                          for seed in TIE_SEEDS[1:]]
    stop_aware = [core.route_given_packing(inst, a, {}, workers) for a in packings]
    sa_costs = [checked_cost(public, inst, p) for p in stop_aware]
    sa_best = stop_aware[sa_costs.index(min(sa_costs))]

    def dump(plan):
        return [v.model_dump() for v in to_van_plans(inst, plan)] if plan else None

    best = min([joint, sa_best] + ([ffd] if ffd else []), key=lambda p: p.cost)
    return {
        "id": inst.name,
        "instance": public.model_dump(),
        "reference": {
            # The proven optimum, or (optimality "best_found") the cheapest plan known.
            "optimum_s": checked_cost(public, inst, best),
            "optimality": "proven" if joint.meta["proved"] else "best_found",
            "lower_bound_s": math.floor(joint.meta["bound"]),
            "floor_s": inst.floor(),
            "solve_s": round(joint.meta["seconds_total"], 2),
            "ffd_s": checked_cost(public, inst, ffd) if ffd else None,
            "stop_aware_best_s": min(sa_costs),
            "stop_aware_mean_s": statistics.mean(sa_costs),
            "route_id": inst.route_id,
            # Hidden from the agent; used by the oracle check and for analysis.
            "plans": {"optimum": dump(best), "ffd": dump(ffd), "stop_aware_best": dump(sa_best)},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--stops", type=int, default=14)
    ap.add_argument("--vans", type=int, default=3)
    ap.add_argument("--tightness", type=float, default=0.9)
    ap.add_argument("--sampling", choices=["cluster", "spread"], default="spread")
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--seeds-per-route", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=300)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    routes, pkgs, tts = core.load_raw(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = args.out.open("w")   # one line per proven task, flushed as it lands
    rows = []
    k = args.seeds_per_route
    # seed = i*k + j: distinct per instance (and seed=i when k=1, as before); the name carries it
    jobs = [(rid, i * k + j) for i, rid in enumerate(sorted(routes)[: args.instances])
            for j in range(k)]
    for rid, seed in jobs:
        inst = core.make_instance(rid, routes[rid], pkgs[rid], tts[rid], args.stops, args.vans,
                                  args.tightness, seed=seed, sampling=args.sampling,
                                  workers=args.workers)
        row = build_one(inst, args.time_limit, args.workers)
        if row is None:
            print(f"{inst.name}: optimum not proven within {args.time_limit}s, skipped", flush=True)
            continue
        ref = row["reference"]
        print(f"{row['id']}: optimum {ref['optimum_s']}s (proved in {ref['solve_s']}s), "
              f"ffd {ref['ffd_s']}s, stop-aware best {ref['stop_aware_best_s']}s", flush=True)
        assert row["id"] not in {r["id"] for r in rows}, f"duplicate task id {row['id']}"
        rows.append(row)
        out.write(json.dumps(row) + "\n")
        out.flush()
    out.close()
    print(f"wrote {len(rows)} tasks to {args.out}")


if __name__ == "__main__":
    main()
