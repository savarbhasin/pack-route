"""Build pack_route tasks: real routes -> instances with a proven joint optimum and baselines.

usage: uv run python -m pack_route.build_tasks DATA_DIR OUT_JSONL
           [--stops 14] [--vans 3] [--tightness 0.9] [--sampling spread] [--instances 10]

DATA_DIR holds new_route_data.json, new_package_data.json and new_travel_times.json from the
Amazon Last Mile Routing Challenge (s3://amazon-last-mile-challenges, CC BY-NC 4.0).
Each output line is {"id", "instance", "reference"}; an instance whose optimum is not proven
within the time limit is skipped.
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


def to_decilitres(inst: core.Instance) -> core.Instance:
    """Switch volumes to the integer decilitres the agent sees, keeping the parcels packable."""
    vols = [max(1, round(v / 100)) for v in inst.pkg_vol]
    cap = max(math.ceil(inst.capacity / 100), max(vols))
    while not core._packable(vols, inst.n_vans, cap):
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


def build_one(inst: core.Instance, time_limit: float) -> dict | None:
    inst = to_decilitres(inst)
    joint = core.solve_joint(inst, time_limit=time_limit)
    if not joint.meta["proved"]:
        return None
    public = to_public(inst)
    a = core.pack_ffd(inst)
    ffd = core.route_given_packing(inst, a, {}) if a else None
    stop_aware = [core.route_given_packing(inst, core.pack_stop_aware(inst, seed)[0], {})
                  for seed in TIE_SEEDS]
    sa_costs = [checked_cost(public, inst, p) for p in stop_aware]
    sa_best = stop_aware[sa_costs.index(min(sa_costs))]

    def dump(plan):
        return [v.model_dump() for v in to_van_plans(inst, plan)] if plan else None

    return {
        "id": inst.name,
        "instance": public.model_dump(),
        "reference": {
            "optimum_s": checked_cost(public, inst, joint),
            "floor_s": inst.floor(),
            "solve_s": round(joint.meta["seconds_total"], 2),
            "ffd_s": checked_cost(public, inst, ffd) if ffd else None,
            "stop_aware_best_s": min(sa_costs),
            "stop_aware_mean_s": statistics.mean(sa_costs),
            "route_id": inst.route_id,
            # Hidden from the agent; used by the oracle check and for analysis.
            "plans": {"optimum": dump(joint), "ffd": dump(ffd), "stop_aware_best": dump(sa_best)},
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
    ap.add_argument("--time-limit", type=float, default=300)
    args = ap.parse_args()

    routes, pkgs, tts = core.load_raw(args.data_dir)
    rows = []
    for i, rid in enumerate(sorted(routes)[: args.instances]):
        inst = core.make_instance(rid, routes[rid], pkgs[rid], tts[rid], args.stops, args.vans,
                                  args.tightness, seed=i, sampling=args.sampling)
        row = build_one(inst, args.time_limit)
        if row is None:
            print(f"{inst.name}: optimum not proven within {args.time_limit}s, skipped", flush=True)
            continue
        ref = row["reference"]
        print(f"{row['id']}: optimum {ref['optimum_s']}s (proved in {ref['solve_s']}s), "
              f"ffd {ref['ffd_s']}s, stop-aware best {ref['stop_aware_best_s']}s", flush=True)
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} tasks to {args.out}")


if __name__ == "__main__":
    main()
