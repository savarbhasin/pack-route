"""pack_route: load parcels into vans and route them, on real Amazon Last Mile routes.

Each task is a real 2018 route's stops, parcels and travel times. The agent reads them through
tools, may price candidate plans with `check_plan`, and submits once. The grade compares the
plan's total driving time with the joint pack+route optimum CP-SAT proved when the task was
built (see `build_tasks.py`).
"""

import json
from pathlib import Path

import verifiers.v1 as vf
from pydantic import BaseModel

from pack_route.plan import PublicInstance, VanPlan, grade, litres
from pack_route.toolset import PackRouteState, PackRouteToolset

DEFAULT_TASKS = Path(__file__).with_name("data") / "tasks-v0.jsonl"

PROMPT = """You are the dispatcher at a parcel delivery station. Today {n_stops} stops need {n_parcels} parcels, \
and you have {n_vans} vans that each hold {capacity:.1f} litres.

Decide which parcels ride in which van and the order each van visits its stops. Every van starts \
and ends at DEPOT. Your goal is the lowest total driving time over all vans.

Rules:
- Every parcel rides in exactly one van, and a van's parcels must fit its capacity (sum of volumes).
- A van's route lists the stops it visits in order, each at most once, and only stops where it \
drops at least one parcel.
- By default all of a stop's parcels go in the one van whose route includes that stop. You may \
split a stop's parcels across vans: put the stop on several routes and list each of its parcels \
under the van that carries it, in that van's `parcels`.
- Vans you don't need can be left out.

Tools: `get_orders` and `get_fleet` describe the day; `get_travel_times` gives real driving times \
in seconds, which are not symmetric; `check_plan` validates and prices a plan without submitting \
it ({check_budget} calls); `submit_plan` submits your final plan, once.

`check_plan` and `submit_plan` take the plan as `vans`, a list with one entry per van used, e.g. \
vans=[{{"route": ["S03", "S01"], "parcels": []}}, {{"route": ["S02"], "parcels": []}}].
Your score depends on how close your total driving time is to the best possible plan. An invalid \
plan scores very low."""


class Reference(BaseModel):
    optimum_s: int
    floor_s: int
    solve_s: float
    ffd_s: int | None
    stop_aware_best_s: int
    stop_aware_mean_s: float
    route_id: str
    plans: dict[str, list[VanPlan] | None]
    """Reference plans (optimum, ffd, stop_aware_best). Never shown to the agent."""


class PackRouteData(vf.TaskData):
    instance: PublicInstance
    reference: Reference


class PackRouteTaskConfig(vf.TaskConfig):
    check_budget: int = 10
    """check_plan calls allowed per episode."""
    max_turns: int = 40


class PackRouteTask(vf.Task[PackRouteData, PackRouteState, PackRouteTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.id

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        trace.state.instance = self.data.instance
        trace.state.check_budget = self.config.check_budget

    @vf.stop
    async def done(self, trace: vf.Trace) -> bool:
        return trace.state.submission is not None or trace.num_turns >= self.config.max_turns

    @vf.reward(weight=1.0)
    async def plan_quality(self, trace: vf.Trace) -> float:
        sub = trace.state.submission
        ref = self.data.reference
        return grade(sub.evaluation if sub else None, ref.optimum_s, ref.floor_s)[0]

    @vf.metric
    async def plan_stats(self, trace: vf.Trace) -> dict[str, float]:
        sub, ref = trace.state.submission, self.data.reference
        ev = sub.evaluation if sub else None
        stats = {"submitted": float(sub is not None), "valid": float(bool(ev and ev.valid)),
                 "checks_used": float(trace.state.checks_used)}
        if ev and ev.valid:
            cost = ev.total_drive_s
            stats |= {"gap_pct": 100 * (cost - ref.optimum_s) / ref.optimum_s,
                      "gap_g": grade(ev, ref.optimum_s, ref.floor_s)[1],
                      "beats_stop_aware": float(cost <= ref.stop_aware_best_s),
                      "beats_ffd": float(ref.ffd_s is not None and cost <= ref.ffd_s)}
        return stats


class PackRouteConfig(vf.TasksetConfig):
    tasks_path: str = str(DEFAULT_TASKS)
    tools: vf.SharedToolsetConfig = vf.SharedToolsetConfig()
    task: PackRouteTaskConfig = PackRouteTaskConfig()


class PackRouteTaskset(vf.Taskset[PackRouteTask, PackRouteConfig]):
    @classmethod
    def toolsets(cls, config: PackRouteConfig) -> list[vf.Toolset]:
        return [PackRouteToolset(config.tools)]

    def load(self) -> list[PackRouteTask]:
        tasks = []
        for line in Path(self.config.tasks_path).read_text().splitlines():
            row = json.loads(line)
            inst = PublicInstance.model_validate(row["instance"])
            prompt = PROMPT.format(n_stops=len(inst.stops), n_parcels=len(inst.parcels),
                                   n_vans=inst.vans, capacity=litres(inst.capacity_dl),
                                   check_budget=self.config.task.check_budget)
            data = PackRouteData(id=row["id"], name=row["id"], prompt=prompt,
                                 instance=inst, reference=row["reference"])
            tasks.append(PackRouteTask(data, self.config.task))
        return tasks
