"""Split the full Amazon Last Mile training files into one small file per route.

usage: uv run python -m pack_route.prepare_amazon BUILD_DIR OUT_DIR

BUILD_DIR holds route_data.json, package_data.json and travel_times.json from
s3://amazon-last-mile-challenges/almrrc2021/almrrc2021-data-training/model_build_inputs/
(CC BY-NC 4.0). travel_times.json is 1.8 GB, far too large to json.load on a laptop, so it is
streamed one route at a time; package_data.json (375 MB, with NaN time windows that streaming
parsers reject) is loaded whole. Each route becomes OUT_DIR/<route_id>.json.gz:
{"route": <route_data entry>, "packages": <stop -> package -> fields>,
 "stops": [...], "travel_s": [[int seconds]]} with the matrix in `stops` order, plus
OUT_DIR/index.json mapping route id -> station code (its first three letters are the city).
Travel times are rounded to whole seconds, exactly as core.make_instance uses them.
Re-running skips routes already written.
"""

import gzip
import json
import sys
from pathlib import Path

import ijson


def main():
    build, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    routes = json.loads((build / "route_data.json").read_text())
    packages = json.loads((build / "package_data.json").read_text())
    print(f"loaded {len(routes)} routes and their packages", flush=True)

    done = {p.name.removesuffix(".json.gz") for p in out.glob("*.json.gz")}
    n = 0
    with (build / "travel_times.json").open("rb") as f:
        for rid, tt in ijson.kvitems(f, "", use_float=True):
            n += 1
            if rid not in done:
                stops = sorted(tt)
                row = {"route": routes[rid], "packages": packages[rid], "stops": stops,
                       "travel_s": [[round(tt[a][b]) for b in stops] for a in stops]}
                with gzip.open(out / f"{rid}.json.gz", "wt") as g:
                    json.dump(row, g)
            if n % 250 == 0:
                print(f"travel times: {n}/{len(routes)} routes", flush=True)
    index = {p.name.removesuffix(".json.gz"): routes[p.name.removesuffix(".json.gz")]["station_code"]
             for p in out.glob("*.json.gz")}
    (out / "index.json").write_text(json.dumps(index))
    print(f"done: {len(index)} route files in {out}, index.json maps route -> station")


if __name__ == "__main__":
    main()
