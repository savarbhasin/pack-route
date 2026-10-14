"""Summarise run_experiment.py outputs. usage: uv run python summarize.py RESULT_JSON..."""
import json, statistics as st, sys

for path in sys.argv[1:]:
    rs = json.load(open(path))["results"]
    J = [r["joint"] for r in rs]
    ffd = [r["ffd"] for r in rs]
    sa_all = [s for r in rs for s in r["stop_aware"]]
    sa_best = [max(r["stop_aware"], key=lambda s: s["reward"]) for r in rs]
    f = lambda xs, k: f"{st.mean(x[k] for x in xs):.3f} (min {min(x[k] for x in xs):.3f}, max {max(x[k] for x in xs):.3f})"
    inzone = [r["joint"]["cost_s"] - r["floor_s"] for r in rs]
    print(f"## {path}")
    print(f"proved optimal: {sum(j['proved'] for j in J)}/{len(J)}; prove time s: mean {st.mean(j['seconds_total'] for j in J):.2f}, max {max(j['seconds_total'] for j in J):.2f}")
    print(f"packages mean {st.mean(r['packages'] for r in rs):.1f}; joint cost h mean {st.mean(j['cost_s'] for j in J)/3600:.2f}; in-zone (cost-floor) min mean {st.mean(inzone)/60:.1f}")
    print(f"joint split stops mean {st.mean(j['split_stops'] for j in J):.2f}; stop-aware splits mean {st.mean(s['split_stops'] for s in sa_all):.2f}; ffd splits mean {st.mean(s['split_stops'] for s in ffd):.2f}")
    for name, xs in [("ffd", ffd), ("stop-aware (all 5 tie-breaks)", sa_all), ("stop-aware (best of 5)", sa_best)]:
        print(f"{name:32s} rel gap {f(xs,'gap_rel')}  g {f(xs,'gap_g')}  reward {f(xs,'reward')}  tasks with reward<=0.9: {sum(x['reward']<=0.9 for x in xs)}/{len(xs)}")
    print()
