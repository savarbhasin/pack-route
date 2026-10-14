# pack_route

A Verifiers v1 environment. The agent is the dispatcher at a parcel station: it decides which parcels ride in which
van **and** the order each van visits its stops, on real 2018 Amazon Last Mile routes. It is graded against the joint
pack+route optimum CP-SAT proved when the task was built. Packing first and routing second loses to that optimum on
every task (the "greedy trap"; see [experiments/README.md](experiments/README.md)).

## Tools (no container; one shared MCP server, per-rollout state)

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

- **Valid:** `0.2 + 0.8·e^(−g/0.5)` with `g = (cost − optimum) / (optimum − floor + 100)`. The floor is the
  depot legs no plan can avoid. The optimum scores 1.0.
- **Invalid:** `0.2 ×` the share of parcels that ride in one van which visits their stop and is within capacity.
- **No submission:** 0.

Metrics: `valid`, `gap_pct`, `gap_g`, `checks_used`, `beats_stop_aware`, `beats_ffd`.

## Tasks

`pack_route/data/tasks-v0.jsonl` holds 10 tasks: 14 stops spread across one real route, 3 vans, synthetic capacity
at 90% fill, with every optimum proven in ≤ 2.2 s. Each row carries the public instance (what the tools serve) and a
hidden `reference`: the optimum, floor, FFD and stop-aware baseline costs, and those plans. Rebuild or make more:

```bash
uv sync -p 3.12
mkdir -p data/apply && (cd data/apply && for f in new_route_data new_package_data new_travel_times; do
  curl -fO "https://amazon-last-mile-challenges.s3.amazonaws.com/almrrc2021/almrrc2021-data-training/model_apply_inputs/$f.json"; done)
uv run python -m pack_route.build_tasks data/apply pack_route/data/tasks-v0.jsonl --stops 14 --sampling spread
```

That sample is 13 routes (~5 MB). The full training split has 6,112 routes across 5 US metros, with a 1.8 GB
travel-time file.

The builder checks that every reference plan, run through the agent-facing validator, is valid at exactly the
solver's cost. It skips any instance whose optimum is not proven.

## Check and run

```bash
uv run python experiments/scripted_check.py      # quality gates with a scripted model, no API key
uv run vf-eval pack_route --env.agent.harness.id null --env.agent.runtime.type subprocess \
    --model <model> -n 10 --no-push              # a real model; needs PRIME_API_KEY, or --client.base-url / --client.api-key-var
```

Gates passing as of 2026-10-14:
- oracle (submit the optimum) = 1.0 on 10/10;
- FFD plan = the experiment's grade exactly (0.32–0.67);
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
- **What was derived:** stop subsets, package volumes (rounded to decilitres) and travel times. Van capacities are
  synthetic.
- **Citation:** Merchán et al., "2021 Amazon Last Mile Routing Research Challenge: Data Set", *Transportation Science*
  (2022), [doi:10.1287/trsc.2022.1173](https://doi.org/10.1287/trsc.2022.1173).
- **Inspired by:** the reward formula and the "real records + solver optimum" pattern come from
  [PortSimEnv](https://github.com/adithya-s-k/FineEnvs/tree/main/07-simulation-environments) (FineEnvs, Hugging Face).
