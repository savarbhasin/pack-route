"""The agent-facing side of an instance: labelled stops and parcels, the plan format, the
validator that explains every rule break, and the grade.

Volumes are integer decilitres (shown to the agent as litres with one decimal) and travel
times are integer seconds, so what the agent sees is exactly what is checked.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

DEPOT = "DEPOT"
MAX_ERRORS = 20


class Parcel(BaseModel):
    id: str
    stop: str
    volume_dl: int


class PublicInstance(BaseModel):
    stops: list[str]
    """Stop ids in matrix order; the depot is row/column 0 of `travel_s`, stops follow."""
    parcels: list[Parcel]
    vans: int
    capacity_dl: int
    travel_s: list[list[int]]

    def index(self, location: str) -> int | None:
        if location == DEPOT:
            return 0
        try:
            return self.stops.index(location) + 1
        except ValueError:
            return None


class VanPlan(BaseModel):
    route: list[str] = Field(default_factory=list, description="Stop ids in visiting order; the van starts and ends at DEPOT.")
    parcels: list[str] = Field(default_factory=list, description="Only for split stops: ids of parcels this van carries at stops that are on more than one route.")


class Evaluation(BaseModel):
    valid: bool
    errors: list[str]
    total_drive_s: int | None
    vans: list[dict]
    clean_share: float
    """Share of parcels that ride in exactly one van, which visits their stop and is within capacity."""


def litres(dl: int) -> float:
    return dl / 10


def evaluate(inst: PublicInstance, vans: list[VanPlan]) -> Evaluation:
    errors: list[str] = []
    parcel_by_id = {p.id: p for p in inst.parcels}
    stops = set(inst.stops)

    if len(vans) > inst.vans:
        errors.append(f"the plan uses {len(vans)} vans but the fleet has {inst.vans}")

    on_routes: dict[str, list[int]] = {}
    for k, van in enumerate(vans):
        seen = set()
        for s in van.route:
            if s not in stops:
                errors.append(f"van {k + 1}: unknown stop {s!r}")
            elif s in seen:
                errors.append(f"van {k + 1}: visits {s} twice")
            else:
                seen.add(s)
                on_routes.setdefault(s, []).append(k)

    assign: dict[str, int] = {}
    for k, van in enumerate(vans):
        for pid in van.parcels:
            p = parcel_by_id.get(pid)
            if p is None:
                errors.append(f"van {k + 1}: unknown parcel {pid!r}")
            elif pid in assign:
                errors.append(f"parcel {pid} is listed under more than one van")
            elif k not in on_routes.get(p.stop, []):
                errors.append(f"van {k + 1}: carries {pid} but its route does not visit {p.stop}")
            else:
                assign[pid] = k
    for p in inst.parcels:
        if p.id in assign:
            continue
        holders = on_routes.get(p.stop, [])
        if not holders:
            errors.append(f"stop {p.stop} is on no route, so parcel {p.id} is not delivered")
        elif len(holders) > 1:
            errors.append(
                f"stop {p.stop} is on vans {', '.join(str(k + 1) for k in holders)}: "
                f"list parcel {p.id} under the van that carries it"
            )
        else:
            assign[p.id] = holders[0]

    van_rows, over = [], set()
    total = 0
    for k, van in enumerate(vans):
        load = sum(parcel_by_id[pid].volume_dl for pid, v in assign.items() if v == k)
        if load > inst.capacity_dl:
            over.add(k)
            errors.append(
                f"van {k + 1}: load {litres(load):.1f} L exceeds capacity {litres(inst.capacity_dl):.1f} L"
            )
        for s in dict.fromkeys(van.route):
            if s in stops and not any(v == k and parcel_by_id[pid].stop == s for pid, v in assign.items()):
                errors.append(f"van {k + 1}: visits {s} but delivers no parcel there")
        path = [0] + [i for s in van.route if (i := inst.index(s)) is not None] + [0]
        drive = sum(inst.travel_s[a][b] for a, b in zip(path, path[1:])) if van.route else 0
        total += drive
        van_rows.append({"van": k + 1, "stops": len(van.route), "load_l": litres(load),
                         "drive_s": drive})

    clean = sum(1 for p in inst.parcels if p.id in assign and assign[p.id] not in over)
    valid = not errors
    if len(errors) > MAX_ERRORS:
        errors = errors[:MAX_ERRORS] + [f"... and {len(errors) - MAX_ERRORS} more"]
    return Evaluation(valid=valid, errors=errors, total_drive_s=total if valid else None,
                      vans=van_rows, clean_share=clean / len(inst.parcels))


def grade(ev: Evaluation | None, optimum_s: int, floor_s: int) -> tuple[float, float | None]:
    """PortSim's reward v3. A valid plan scores 0.2 + 0.8·e^(−g/0.5) with
    g = (cost − optimum) / (optimum − floor + 100), so the optimum scores 1.0. An invalid plan
    scores 0.2 × its share of cleanly delivered parcels; no submission scores 0.
    Returns (reward, g)."""
    if ev is None:
        return 0.0, None
    if not ev.valid:
        return 0.2 * ev.clean_share, None
    g = max(0, ev.total_drive_s - optimum_s) / (max(0, optimum_s - floor_s) + 100)
    return 0.2 + 0.8 * math.exp(-g / 0.5), g
