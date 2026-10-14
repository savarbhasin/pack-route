"""The dispatcher's tools. One shared server per environment worker; each rollout's instance,
check budget and submission live in its own `self.state`."""

import verifiers.v1 as vf
from pydantic import BaseModel

from pack_route.plan import DEPOT, Evaluation, PublicInstance, VanPlan, evaluate, litres


class Submission(BaseModel):
    vans: list[VanPlan]
    evaluation: Evaluation


class PackRouteState(vf.State):
    instance: PublicInstance | None = None
    check_budget: int = 10
    checks_used: int = 0
    submission: Submission | None = None


class PackRouteToolset(vf.Toolset[vf.SharedToolsetConfig, PackRouteState]):
    TOOL_PREFIX = None

    @property
    def inst(self) -> PublicInstance:
        if self.state.instance is None:
            raise RuntimeError("no instance loaded for this rollout")
        return self.state.instance

    @vf.tool
    async def get_orders(self) -> dict:
        """Today's stops and the parcels to drop at each (id and volume in litres)."""
        by_stop: dict[str, list[dict]] = {s: [] for s in self.inst.stops}
        for p in self.inst.parcels:
            by_stop[p.stop].append({"id": p.id, "volume_l": litres(p.volume_dl)})
        return {
            "depot": DEPOT,
            "stops": [{"id": s, "parcels": ps,
                       "total_volume_l": litres(sum(p.volume_dl for p in self.inst.parcels if p.stop == s))}
                      for s, ps in by_stop.items()],
        }

    @vf.tool
    async def get_fleet(self) -> dict:
        """The vans: how many, and how much each holds. Every van starts and ends at DEPOT."""
        return {"vans": self.inst.vans, "capacity_l_per_van": litres(self.inst.capacity_dl),
                "start_and_end": DEPOT}

    @vf.tool
    async def get_travel_times(self, locations: list[str]) -> dict:
        """Driving times in seconds between the given locations (stop ids and/or DEPOT).
        seconds[i][j] is the time from locations[i] to locations[j]; times are not symmetric."""
        idx = [self.inst.index(loc) for loc in locations]
        unknown = [loc for loc, i in zip(locations, idx) if i is None]
        if unknown:
            return {"error": f"unknown locations: {unknown}"}
        return {"locations": locations,
                "seconds": [[self.inst.travel_s[a][b] for b in idx] for a in idx]}

    @vf.tool
    async def check_plan(self, vans: list[VanPlan]) -> dict:
        """Validate a plan and price it without submitting. Limited calls per episode."""
        if self.state.submission is not None:
            return {"error": "a plan was already submitted"}
        left = self.state.check_budget - self.state.checks_used
        if left <= 0:
            return {"error": "no check_plan calls left; call submit_plan when ready"}
        self.state.checks_used += 1
        ev = evaluate(self.inst, vans)
        return {"valid": ev.valid, "errors": ev.errors, "total_drive_s": ev.total_drive_s,
                "vans": ev.vans, "check_plan_calls_left": left - 1}

    @vf.tool
    async def submit_plan(self, vans: list[VanPlan]) -> dict:
        """Submit the final plan. Only the first submission counts and it ends the episode."""
        if self.state.submission is not None:
            return {"error": "a plan was already submitted"}
        ev = evaluate(self.inst, vans)
        self.state.submission = Submission(vans=vans, evaluation=ev)
        return {"submitted": True, "valid": ev.valid, "errors": ev.errors,
                "total_drive_s": ev.total_drive_s}


if __name__ == "__main__":
    PackRouteToolset.run()
