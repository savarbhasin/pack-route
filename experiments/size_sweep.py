"""How long does CP-SAT take to prove the joint pack+route optimum as instances grow?

usage: uv run python size_sweep.py DATA_DIR OUT_JSON
"""

import json
import sys
from pathlib import Path

from pack_route.core import check_plan, load_raw, make_instance, solve_joint

SIZES = [(20, 3), (30, 3), (45, 4)]   # (stops, vans)
PER_SIZE = 3
TIME_LIMIT = 120


def main():
    data_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    routes, pkgs, tts = load_raw(data_dir)
    rows = []
    cases = [(s, n, k, i, rid) for s in ("cluster", "spread") for n, k in SIZES
             for i, rid in enumerate(sorted(routes)[:PER_SIZE])]
    for sampling, n, k, i, rid in cases:
        inst = make_instance(rid, routes[rid], pkgs[rid], tts[rid], n, k, 0.9, seed=i,
                             sampling=sampling)
        plan = solve_joint(inst, time_limit=TIME_LIMIT)
        assert not check_plan(inst, plan)
        m = plan.meta
        gap_to_bound = (plan.cost - m["bound"]) / max(1, plan.cost - inst.floor())
        rows.append({"name": inst.name, "sampling": sampling, "stops": n, "vans": k,
                     "packages": len(inst.pkg_vol), "cost_s": plan.cost, "floor_s": inst.floor(),
                     **m, "open_gap_over_in_zone": gap_to_bound})
        print(f"{inst.name}: pkgs={len(inst.pkg_vol)} {m['status']} total={m['seconds_total']:.1f}s "
              f"best@{m['seconds_to_best']:.1f}s open-gap/in-zone={gap_to_bound:.1%}", flush=True)
        out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
