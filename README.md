# Pack Route

A Verifiers v1 environment. The agent is the dispatcher at a parcel station: it decides which parcels ride in which
van **and** the order each van visits its stops, on real 2018 Amazon Last Mile routes. It is graded against the best
joint pack+route plan CP-SAT found when the task was built. Packing first and routing second loses to that plan on
every task (the "greedy trap"; see [experiments/README.md](experiments/README.md)).

## Tools 

| Tool | Returns |
| --- | --- |
| `get_orders()` | Stops, their parcels (id, volume in litres) and volume per stop |
| `get_fleet()` | Number of vans, capacity per van, depot |
| `get_travel_times(locations)` | Real driving seconds between any stops and `DEPOT` (asymmetric) |
| `check_plan(vans)` | Valid?, every rule break in words, total and per-van drive time and load. 10 calls per episode (`task.check_budget`) |
| `submit_plan(vans)` | Final, once; ends the episode |

A plan is `vans=[{"route": ["S03", "S01"], "parcels": []}, ...]`. A stop's parcels go to the one van whose route
includes it. To split a stop across vans, put it on several routes and list each of its parcels under the van that
carries it.

## Grade (PortSim reward v3)

- **Valid:** `0.2 + 0.8·e^(−g/0.5)` with `g = max(0, cost − reference) / (reference − floor + 100)`. The reference is
  `optimum_s`; the floor is the depot legs no plan can avoid. Matching or beating the reference scores 1.0.
- **Invalid:** `0.2 ×` the share of parcels that ride in one van which visits their stop and is within capacity.
- **No submission:** 0.

Metrics: `valid`, `gap_pct`, `gap_g`, `checks_used`, `beats_stop_aware`, `beats_ffd`, `beats_reference` (the plan is
cheaper than the best one the solver found).

## Tasks

Also on Hugging Face: [dev-front/pack-route-tasks](https://huggingface.co/datasets/dev-front/pack-route-tasks).

| File | Tasks | Stops / vans | Reference |
| --- | --- | --- | --- |
| `pack_route/data/v1/train.jsonl` | 80: 20 each from Los Angeles, Seattle, Chicago, Boston | 20/3, 25/3, 30/4, 40/4, mixed | best found |
| `pack_route/data/v1/test.jsonl` (default) | 10 from Austin, a city train never sees | same mix | best found; plans removed |
| `pack_route/data/tasks-v0.jsonl` | 10 (the first release) | 14/3 | proven optimum |

Each task takes one real route: a spread-out subset of its stops, their real parcels (23–155 per task) and the real
travel times between them. The fleet is synthetic: 3 or 4 vans whose capacity makes the parcels fill 85–95% of it.
Every task comes from a different route.

Each row has the public `instance` (what the tools serve), a hidden `reference` and, in v1, `meta` (city, station,
date, size, fill). The reference holds:
- `optimum_s`, the cost the grade is measured against;
- `optimality`: `proven` (v0) or `best_found` (v1);
- `lower_bound_s`, the solver's proof that no plan is cheaper (median 13% below `optimum_s` in v1);
- the floor and the FFD and stop-aware baseline costs;
- `plans`: the reference, FFD and best stop-aware plans (`null` in v1 test).

At 20–40 stops CP-SAT rarely proves the optimum on a laptop, so v1 uses the best plan found: the joint search stops
after 30 s without improvement (at most 180 s), and the cheapest of its plan and the baselines becomes the reference. A
model can beat it, which the `beats_reference` metric records. A route is kept only if the best stop-aware
pack-then-route plan grades ≤ 0.9 against it. In v1 that plan grades 0.29–0.79 (mean 0.58, 11% more driving) and FFD
0.25–0.70.

Build v1 from the full training split (6,112 routes, 5 US metros; the travel-time file is 1.8 GB):

```bash
uv sync -p 3.12
S3=https://amazon-last-mile-challenges.s3.amazonaws.com/almrrc2021/almrrc2021-data-training/model_build_inputs
mkdir -p data/build && (cd data/build && for f in route_data package_data travel_times; do curl -fO "$S3/$f.json"; done)
uv run python -m pack_route.prepare_amazon data/build data/routes         # one small file per route
uv run python -m pack_route.build_dataset data/routes pack_route/data/v1 \
    --train-per-city 20 --test 10 --jobs 2 --workers 2 --time-limit 180 --stall 30
```

`touch pack_route/data/v1/STOP` stops the build after the running tasks finish; a re-run resumes. With 2 jobs × 2
CP-SAT workers a task takes about 105 s. The build writes the test plans to `test.jsonl`; they were moved out before
publishing. v0 was built with `pack_route.build_tasks` (see its docstring), which skips instances whose optimum is not
proven.

The builders check that every reference plan, run through the agent-facing validator, is valid at exactly the
solver's cost.

## Check and run

```bash
uv run python experiments/scripted_check.py      # quality gates with a scripted model on v1 train, no API key
uv run python experiments/scripted_check.py --tasks pack_route/data/v1/test.jsonl --policies invalid nop
uv run vf-eval pack_route --env.agent.harness.id null --env.agent.runtime.type subprocess \
    --model <model> -n 10 --no-push              # a real model on v1 test; needs PRIME_API_KEY, or --client.base-url / --client.api-key-var
# other tasks: --env.taskset.tasks_path pack_route/data/v1/train.jsonl
```

Gates passing as of 2026-10-15 on all 100 tasks (v1 test checked with its plans before they were removed):
- oracle (submit the reference plan) = 1.0;
- FFD plan = its expected grade exactly (0.25–0.70 on v1);
- everything in one van = 0;
- never submitting = 0.

The `null` harness gives the model only these tools: it must reason about packing and routing without running code.
The `bash` harness would let it write and run its own solver, which makes a different task.

**Not modelled yet:** time windows, service time, driver shifts, news mid-episode.

## Licence and attribution

- **Code:** Apache-2.0 ([LICENSE](LICENSE)).
- **Tasks** (`pack_route/data/`) **and experiment results** (`experiments/results/`): derived from the
  [2021 Amazon Last Mile Routing Research Challenge dataset](https://registry.opendata.aws/amazon-last-mile-challenges)
  and released under the same licence, **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**
  (non-commercial).
- **What was derived:** stop subsets, package volumes (rounded to decilitres), travel times, and the station and date
  of each route. Fleet sizes and van capacities are synthetic.
- **Citation:** Merchán et al., "2021 Amazon Last Mile Routing Research Challenge: Data Set", *Transportation Science*
  (2022), [doi:10.1287/trsc.2022.1173](https://doi.org/10.1287/trsc.2022.1173).
- **Inspired by:** the reward formula and the "real records + solver optimum" pattern come from
  [PortSimEnv](https://github.com/adithya-s-k/FineEnvs/tree/main/07-simulation-environments) (FineEnvs, Hugging Face).
