"""Pack + route on real Amazon Last Mile data: joint CP-SAT optimum vs pack-then-route.

Problem (v0, deliberately small):
- One station, K identical vans, n delivery stops taken from one real route.
- Every package (real dimensions) goes into exactly one van; van volume capacity C.
- A van visits a stop iff it carries at least one of that stop's packages, so a stop
  whose packages are split across vans is driven to more than once.
- Each used van drives a closed tour station -> its stops -> station.
- Objective: total driving time in seconds, from the route's real travel-time matrix.
- Not modelled yet: time windows, service time, driver shifts.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from ortools.sat.python import cp_model


# ---------------------------------------------------------------- instances


@dataclass
class Instance:
    name: str
    route_id: str
    stops: list[str]                 # stops[0] is the station
    tt: list[list[int]]              # travel time, seconds, indexed like stops
    pkg_stop: list[int]              # package -> stop index (1..n)
    pkg_vol: list[int]               # package volume, cm3
    n_vans: int
    capacity: int                    # cm3 per van

    @property
    def n(self) -> int:
        return len(self.stops) - 1

    def floor(self) -> int:
        """Cost no plan can avoid: every van leaves the station and comes back."""
        out = min(self.tt[0][s] for s in range(1, self.n + 1))
        back = min(self.tt[s][0] for s in range(1, self.n + 1))
        return self.n_vans * (out + back)


def load_raw(data_dir: Path):
    routes = json.loads((data_dir / "new_route_data.json").read_text())
    pkgs = json.loads((data_dir / "new_package_data.json").read_text())
    tts = json.loads((data_dir / "new_travel_times.json").read_text())
    return routes, pkgs, tts


def make_instance(route_id, route, pkgs, tt, n_stops, n_vans, tightness, seed,
                  sampling: str = "cluster") -> Instance:
    """n stops from one real route: `cluster` = a seed stop + its nearest neighbours by travel
    time; `spread` = n random stops from the whole route.

    Capacity is synthetic: scaled so the sampled packages fill `tightness` of the fleet.
    The real van capacity (~4.2 m3) would hold this whole subset in one van.
    """
    rng = random.Random(seed)
    station = next(s for s, v in route["stops"].items() if v["type"] == "Station")
    drops = sorted(s for s in route["stops"] if s != station and pkgs.get(s))
    seed_stop = rng.choice(drops)
    if sampling == "cluster":
        near = sorted(drops, key=lambda s: (tt[seed_stop][s] + tt[s][seed_stop], s))[:n_stops]
    else:
        near = rng.sample(drops, n_stops)
    stops = [station] + sorted(near)
    idx = {s: i for i, s in enumerate(stops)}
    pkg_stop, pkg_vol = [], []
    for s in stops[1:]:
        for pid in sorted(pkgs[s]):
            d = pkgs[s][pid]["dimensions"]
            pkg_stop.append(idx[s])
            pkg_vol.append(max(1, round(d["depth_cm"] * d["height_cm"] * d["width_cm"])))
    total = sum(pkg_vol)
    capacity = max(math.ceil(total / (n_vans * tightness)), max(pkg_vol))
    while not _packable(pkg_vol, n_vans, capacity):   # real parcels can be bulky: grow 2% until they fit
        capacity = math.ceil(capacity * 1.02)
    m = [[round(tt[a][b]) for b in stops] for a in stops]
    return Instance(
        name=f"{route_id[8:16]}-{sampling}-n{n_stops}-k{n_vans}-t{tightness}-s{seed}",
        route_id=route_id, stops=stops, tt=m, pkg_stop=pkg_stop, pkg_vol=pkg_vol,
        n_vans=n_vans, capacity=capacity,
    )


def _packable(vols: list[int], n_vans: int, capacity: int) -> bool:
    model = cp_model.CpModel()
    x = {(p, k): model.NewBoolVar("") for p in range(len(vols)) for k in range(n_vans)}
    for p in range(len(vols)):
        model.AddExactlyOne(x[p, k] for k in range(n_vans))
    for k in range(n_vans):
        model.Add(sum(v * x[p, k] for p, v in enumerate(vols)) <= capacity)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    return solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ---------------------------------------------------------------- plans


@dataclass
class Plan:
    assign: list[int]                # package -> van
    tours: list[list[int]]           # per van, stop indices in visiting order (no station)
    cost: int
    meta: dict = field(default_factory=dict)


def tour_cost(inst: Instance, tour: list[int]) -> int:
    if not tour:
        return 0
    path = [0] + tour + [0]
    return sum(inst.tt[a][b] for a, b in zip(path, path[1:]))


def check_plan(inst: Instance, plan: Plan) -> list[str]:
    """Independent validator: returns rule breaks (empty list = valid)."""
    errs = []
    if len(plan.assign) != len(inst.pkg_vol):
        errs.append("not every package assigned")
    for k in range(inst.n_vans):
        load = sum(v for p, v in enumerate(inst.pkg_vol) if plan.assign[p] == k)
        if load > inst.capacity:
            errs.append(f"van {k} over capacity: {load} > {inst.capacity}")
        need = {inst.pkg_stop[p] for p in range(len(inst.pkg_vol)) if plan.assign[p] == k}
        if set(plan.tours[k]) != need or len(plan.tours[k]) != len(need):
            errs.append(f"van {k} tour does not match its packages")
    if sum(tour_cost(inst, t) for t in plan.tours) != plan.cost:
        errs.append("reported cost does not match tours")
    return errs


def _circuit(model, inst, nodes, prefix):
    """Optional-node circuit over station(0) + nodes. Returns (arcs, visit literals, unused literal)."""
    arcs, visit = [], {}
    all_nodes = [0] + nodes
    lit = {}
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                lit[i, j] = model.NewBoolVar(f"{prefix}a{i}_{j}")
                arcs.append((all_nodes.index(i), all_nodes.index(j), lit[i, j]))
    unused = model.NewBoolVar(f"{prefix}unused")
    arcs.append((0, 0, unused))
    for s in nodes:
        visit[s] = model.NewBoolVar(f"{prefix}v{s}")
        arcs.append((all_nodes.index(s), all_nodes.index(s), visit[s].Not()))
    model.AddCircuit(arcs)
    cost = sum(inst.tt[i][j] * l for (i, j), l in lit.items())
    return lit, visit, unused, cost


def _read_tour(solver, lit, nodes) -> list[int]:
    nxt = {i: j for (i, j), l in lit.items() if solver.Value(l)}
    tour, cur = [], nxt.get(0)
    while cur not in (None, 0):
        tour.append(cur)
        cur = nxt[cur]
    return tour


def solve_tsp(inst: Instance, nodes: list[int]) -> list[int]:
    """Optimal closed tour station -> nodes -> station (small, proves instantly)."""
    if not nodes:
        return []
    model = cp_model.CpModel()
    lit, visit, unused, cost = _circuit(model, inst, nodes, "")
    model.Add(unused == 0)
    for v in visit.values():
        model.Add(v == 1)
    model.Minimize(cost)
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 8
    solver.parameters.max_time_in_seconds = 60
    st = solver.Solve(model)
    assert st == cp_model.OPTIMAL, f"TSP not proven: {solver.StatusName(st)}"
    return _read_tour(solver, lit, nodes)


def route_given_packing(inst: Instance, assign: list[int], meta: dict) -> Plan:
    tours = []
    for k in range(inst.n_vans):
        nodes = sorted({inst.pkg_stop[p] for p, a in enumerate(assign) if a == k})
        tours.append(solve_tsp(inst, nodes))
    return Plan(assign, tours, sum(tour_cost(inst, t) for t in tours), meta)


# ---------------------------------------------------------------- sequential baselines


def pack_ffd(inst: Instance) -> list[int] | None:
    """First-fit decreasing by volume: the textbook bin-packing heuristic, blind to geography."""
    order = sorted(range(len(inst.pkg_vol)), key=lambda p: (-inst.pkg_vol[p], p))
    load = [0] * inst.n_vans
    assign = [-1] * len(inst.pkg_vol)
    for p in order:
        for k in range(inst.n_vans):
            if load[k] + inst.pkg_vol[p] <= inst.capacity:
                assign[p], load[k] = k, load[k] + inst.pkg_vol[p]
                break
        else:
            return None
    return assign


def pack_stop_aware(inst: Instance, tie_seed: int) -> tuple[list[int], int]:
    """Best packing a dispatcher could make without a map: fewest split stops.

    Many packings tie on that; a tiny random secondary objective picks one, so different
    tie_seeds give different equally-good-to-the-packer packings.
    Returns (assignment, number of split stops).
    """
    P, K, n = len(inst.pkg_vol), inst.n_vans, inst.n
    model = cp_model.CpModel()
    x = {(p, k): model.NewBoolVar(f"x{p}_{k}") for p in range(P) for k in range(K)}
    y = {(s, k): model.NewBoolVar(f"y{s}_{k}") for s in range(1, n + 1) for k in range(K)}
    for p in range(P):
        model.AddExactlyOne(x[p, k] for k in range(K))
        for k in range(K):
            model.AddImplication(x[p, k], y[inst.pkg_stop[p], k])
    for k in range(K):
        model.Add(sum(inst.pkg_vol[p] * x[p, k] for p in range(P)) <= inst.capacity)
    visits = sum(y.values())  # n + number of extra visits caused by splits
    model.Minimize(visits)
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 8
    solver.parameters.max_time_in_seconds = 60
    st = solver.Solve(model)
    assert st == cp_model.OPTIMAL, solver.StatusName(st)
    best_visits = round(solver.ObjectiveValue())
    # second pass: fix the primary objective, random tie-break among optimal packings
    model.Add(visits == best_visits)
    rng = random.Random(tie_seed)
    model.Minimize(sum(rng.randint(0, 1000) * v for v in x.values()))
    st = solver.Solve(model)
    assert st in (cp_model.OPTIMAL, cp_model.FEASIBLE), solver.StatusName(st)
    assign = [next(k for k in range(K) if solver.Value(x[p, k])) for p in range(P)]
    return assign, best_visits - n


# ---------------------------------------------------------------- joint optimum


class _Trace(cp_model.CpSolverSolutionCallback):
    def __init__(self):
        super().__init__()
        self.t0 = time.perf_counter()
        self.found: list[tuple[float, float]] = []   # (seconds, objective)

    def on_solution_callback(self):
        self.found.append((time.perf_counter() - self.t0, self.ObjectiveValue()))


def solve_joint(inst: Instance, time_limit: float = 300, workers: int = 8) -> Plan:
    P, K, n = len(inst.pkg_vol), inst.n_vans, inst.n
    nodes = list(range(1, n + 1))
    model = cp_model.CpModel()
    x = {(p, k): model.NewBoolVar(f"x{p}_{k}") for p in range(P) for k in range(K)}
    vans = [_circuit(model, inst, nodes, f"k{k}_") for k in range(K)]
    by_stop = {s: [p for p in range(P) if inst.pkg_stop[p] == s] for s in nodes}
    for p in range(P):
        model.AddExactlyOne(x[p, k] for k in range(K))
        # identical vans: package p may only use vans 0..p (removes van-relabelling symmetry)
        for k in range(p + 1, K):
            model.Add(x[p, k] == 0)
    for k, (_, visit, unused, _) in enumerate(vans):
        model.Add(sum(inst.pkg_vol[p] * x[p, k] for p in range(P)) <= inst.capacity)
        for s in nodes:
            for p in by_stop[s]:
                model.AddImplication(x[p, k], visit[s])
            model.AddBoolOr([x[p, k] for p in by_stop[s]]).OnlyEnforceIf(visit[s])
            model.AddImplication(visit[s], unused.Not())
    model.Minimize(sum(v[3] for v in vans))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = workers
    solver.parameters.max_time_in_seconds = time_limit
    trace = _Trace()
    t0 = time.perf_counter()
    st = solver.Solve(model, trace)
    wall = time.perf_counter() - t0
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"joint: {solver.StatusName(st)}")
    assign = [next(k for k in range(K) if solver.Value(x[p, k])) for p in range(P)]
    tours = [_read_tour(solver, v[0], nodes) for v in vans]
    best = solver.ObjectiveValue()
    t_best = next(t for t, o in trace.found if o <= best + 1e-6) if trace.found else wall
    return Plan(assign, tours, round(best), {
        "status": solver.StatusName(st),
        "proved": st == cp_model.OPTIMAL,
        "bound": solver.BestObjectiveBound(),
        "seconds_total": wall,
        "seconds_to_best": t_best,
    })


# ---------------------------------------------------------------- grading (PortSim formula)


def reward(inst: Instance, cost: int, ref: int) -> tuple[float, float, float]:
    """Returns (relative gap, PortSim-style floor-normalised gap g, reward).

    PortSim v3 reward: g = max(0, cost - ref) / (max(0, ref - floor) + 100), r = 0.2 + 0.8 e^(-g/0.5).
    """
    rel = (cost - ref) / ref
    g = max(0, cost - ref) / (max(0, ref - inst.floor()) + 100)
    return rel, g, 0.2 + 0.8 * math.exp(-g / 0.5)
