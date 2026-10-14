# pack_route: does the "greedy trap" exist on real delivery data?

Experiments behind the pack_route environment (see ../README.md). Prototype for the composed-tasks idea (Traycer spec `composed-tasks`). It answers one question: on real Amazon
Last Mile routes, how much worse is **pack the vans first, then route each van** than the **joint pack+route
optimum** CP-SAT proves, and how long does the proof take?

## Problem (v0)

- One station, K identical vans, n delivery stops taken from one real 2018 route.
- Every package (real dimensions → volume) goes in exactly one van; synthetic per-van volume capacity, scaled so the
  sampled packages fill `tightness` of the fleet (the real 4.2 m³ van would hold the whole subset).
- A van visits a stop iff it carries one of its packages, so a split stop is driven to twice.
- Closed tours station → stops → station; objective = total driving seconds from the route's real travel-time matrix.
- **Not modelled yet:** time windows, service time, driver shifts.

## Plans compared

| Plan | How |
| --- | --- |
| `joint` | One CP-SAT model: package→van booleans + one optional-node `AddCircuit` per van, linked by "van carries a package of stop s ⇔ van visits s". Proven optimum = the reference |
| `ffd` | First-fit-decreasing by volume, then an optimal tour per van |
| `stop_aware` | The best packing a dispatcher could do without a map: CP-SAT minimises split stops; ties broken by 5 random seeds; then an optimal tour per van |

Every plan passes the independent `check_plan` validator (capacity, every package once, tour matches load, cost).
Grading uses PortSim's reward v3: `g = (cost − ref) / (ref − floor + 100)`, `r = 0.2 + 0.8·e^(−g/0.5)`, with
floor = each van's cheapest station→zone→station legs.

## Run

```bash
uv sync -p 3.12   # from environments/pack_route
D=data/apply   # see ../README.md for the download (CC BY-NC 4.0)
uv run python experiments/run_experiment.py $D experiments/results/main_n14_k3_t0.9.json                 # 10 instances, 14 stops, 3 vans
uv run python experiments/run_experiment.py $D experiments/results/spread_n14_k3_t0.9.json --sampling spread
uv run python experiments/summarize.py experiments/results/*.json
uv run python experiments/size_sweep.py $D experiments/results/size_sweep.json                           # solve time vs size
```

Data: `curl -O https://amazon-last-mile-challenges.s3.amazonaws.com/almrrc2021/almrrc2021-data-training/model_apply_inputs/{new_route_data,new_package_data,new_travel_times}.json`
(13 routes, ~5 MB; the full training split's travel-time file is 1.8 GB). Results are deterministic: a rerun of the
main config reproduced every cost exactly.
