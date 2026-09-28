"""Export one weekday of Paris buses for the web viewer in docs/.

    python build_web_data.py --gtfs IDFM-gtfs.zip --cache day.pkl

Writes docs/fleet/fleet.bin.gz (typed arrays) and docs/fleet/fleet.json (manifest, lines, metadata).
The browser rebuilds positions itself: each trip is a start time plus a timing profile,
a list of (time, distance) knots along one route pattern polyline.
"""
import argparse
import gzip
import json
import os
import pickle
import time

import numpy as np

from mapbuilder.fleet import SHAPE_STRIDE, TRIP_STRIDE, Projection, build_fleet
from mapbuilder.gtfs import Feed, load_bus_day, pick_service_date
from mapbuilder.render import Style, line_color
from render_paris_bus import BBOX

HERE = os.path.dirname(os.path.abspath(__file__))
T_START, T_END = 4 * 3600, 28 * 3600
TRAIL_S = 180  # the viewer draws trails this long, so keep trips that ended just before T_START


def simplify(xy: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker; returns the indices of the kept vertices."""
    keep = np.zeros(len(xy), bool)
    keep[[0, -1]] = True
    stack = [(0, len(xy) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = xy[i], xy[j]
        ab = b - a
        n = np.hypot(*ab)
        p = xy[i + 1 : j] - a
        d = np.abs(ab[0] * p[:, 1] - ab[1] * p[:, 0]) / n if n > 1e-9 else np.hypot(p[:, 0], p[:, 1])
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return np.flatnonzero(keep)


def split_long_gaps(t: np.ndarray, d: np.ndarray, limit: int = 60_000):
    """Add knots inside gaps longer than `limit` metres so deltas fit in 16 bits (motion is unchanged)."""
    gaps = np.flatnonzero(np.diff(d) > limit)
    if not len(gaps):
        return t, d
    tt, dd = [t[: gaps[0] + 1]], [d[: gaps[0] + 1]]
    for n, g in enumerate(gaps):
        parts = int(np.ceil((d[g + 1] - d[g]) / limit))
        f = np.arange(1, parts) / parts
        tt.append(np.round(t[g] + f * (t[g + 1] - t[g])).astype(np.int64))
        dd.append(np.round(d[g] + f * (d[g + 1] - d[g])).astype(np.int64))
        end = gaps[n + 1] + 1 if n + 1 < len(gaps) else len(t)
        tt.append(t[g + 1 : end])
        dd.append(d[g + 1 : end])
    return np.concatenate(tt), np.concatenate(dd)


def route_termini(feed: Feed, route_ids: set) -> dict:
    """'A ↔ B' from each route's most common headsign in either direction (IDFM long names are mostly empty)."""
    header = feed.read("trips.txt", nrows=0).columns
    if "trip_headsign" not in header:
        return {}
    cols = ["route_id", "trip_headsign"] + (["direction_id"] if "direction_id" in header else [])
    t = feed.read("trips.txt", usecols=cols)
    t = t[t["route_id"].isin(route_ids) & t["trip_headsign"].notna()]
    if "direction_id" not in t.columns:
        t["direction_id"] = "0"
    top = t.groupby(["route_id", "direction_id"])["trip_headsign"].agg(lambda s: s.value_counts().index[0])
    out = {}
    for rid, g in top.groupby(level=0):
        ends = list(dict.fromkeys(h.strip() for h in g))
        out[rid] = " ↔ ".join(reversed(ends[:2]))
    return out


def export_stops(feed: Feed, day, route_index: dict) -> dict:
    """Bus stops of the day, merged by stop area and name, with the lines serving each."""
    info = feed.read("stops.txt", usecols=lambda c: c in ("stop_id", "stop_name", "parent_station"))
    if "parent_station" not in info.columns:
        info["parent_station"] = None
    st = day.stops.merge(info, on="stop_id", how="left")
    st["stop_name"] = st["stop_name"].fillna("").str.strip()
    st["key"] = st["parent_station"].fillna(st["stop_id"]) + "|" + st["stop_name"]
    served = day.stop_times[["trip_id", "stop_id"]].drop_duplicates().merge(day.trips[["trip_id", "route_id"]], on="trip_id")
    served = served[["stop_id", "route_id"]].drop_duplicates().merge(st[["stop_id", "key"]], on="stop_id")
    served["line"] = served["route_id"].map(route_index)
    lines_of = served.dropna(subset=["line"]).groupby("key")["line"].agg(lambda s: sorted({int(x) for x in s}))
    groups = st.groupby("key").agg(name=("stop_name", "first"), lon=("lon", "mean"), lat=("lat", "mean"))
    groups = groups.join(lines_of.rename("lines"), how="inner").sort_values("name", kind="stable")
    return dict(
        name=groups["name"].tolist(),
        lonlat=[round(float(v), 5) for xy in zip(groups["lon"], groups["lat"]) for v in xy],
        lines=groups["lines"].tolist(),
    )


def natural_key(name: str):
    digits = "".join(c for c in name if c.isdigit())
    return (0 if name.isdigit() else 1, int(digits) if digits else 0, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gtfs", required=True, help="GTFS zip or directory (agency names are read from it)")
    ap.add_argument("--weekday", default="tuesday")
    ap.add_argument("--date", help="YYYY-MM-DD, overrides --weekday")
    ap.add_argument("--cache", help="pickle file shared with render_paris_bus.py")
    ap.add_argument("--out", default=os.path.join(HERE, "docs", "fleet"))
    ap.add_argument("--tolerance", type=float, default=2.5, help="shape simplification, metres")
    a = ap.parse_args()

    t0 = time.time()
    feed = Feed(a.gtfs)
    if a.cache and os.path.exists(a.cache):
        date, fleet = pickle.load(open(a.cache, "rb"))
        day = load_bus_day(feed, date, BBOX)  # stops are not in the cache
    else:
        date = pick_service_date(feed, a.weekday, a.date)
        day = load_bus_day(feed, date, BBOX)
        fleet = build_fleet(day, Projection(*Style().center))
        if a.cache:
            pickle.dump((date, fleet), open(a.cache, "wb"))
    proj = Projection(*Style().center)
    print(f"fleet: {len(fleet.trip_start):,} trips [{time.time() - t0:.0f}s]")

    # --- lines
    routes = fleet.routes.reset_index(drop=True)
    agencies = feed.read("agency.txt")[["agency_id", "agency_name"]] if feed.has("agency.txt") else None
    names = routes.merge(agencies, how="left", on="agency_id")["agency_name"] if agencies is not None else None
    termini = route_termini(feed, set(routes["route_id"]))
    lines = []
    for i, r in routes.iterrows():
        text_of = lambda col: r.get(col) if isinstance(r.get(col), str) else ""  # noqa: E731
        name = text_of("route_short_name") or str(r["route_id"])
        color = text_of("route_color").upper() if len(text_of("route_color")) == 6 else "7F7F7F"
        text = text_of("route_text_color").upper() if len(text_of("route_text_color")) == 6 else "FFFFFF"
        glow = "".join(f"{round(c * 255):02X}" for c in line_color(r.get("route_color"), name))
        long_name = text_of("route_long_name")
        if long_name in ("", name):
            long_name = termini.get(r["route_id"], "")
        if long_name == name:
            long_name = ""
        agency = names.iloc[i] if names is not None and isinstance(names.iloc[i], str) else ""
        lines.append(
            dict(id=r["route_id"], name=name, long=long_name, color=color, text=text, glow=glow, network=agency, trips=0)
        )

    # --- trips -> (pattern, profile)
    knot_pat = (fleet.knot_d // SHAPE_STRIDE).astype(np.int64)
    keep_trips = np.flatnonzero((fleet.trip_end > T_START - TRAIL_S) & (fleet.trip_start < T_END))
    profiles: dict[tuple, int] = {}
    prof_pat, prof_t, prof_d = [], [], []
    trip_start, trip_prof, trip_route = [], [], []
    for i in keep_trips:
        k0, k1 = fleet.trip_k0[i], fleet.trip_k1[i]
        p = int(knot_pat[k0])
        start = round(fleet.trip_start[i])
        t = np.maximum.accumulate(np.round(fleet.knot_key[k0:k1] - i * TRIP_STRIDE - start)).astype(np.int64)
        t = np.maximum(t, 0)
        d = np.maximum.accumulate(np.round(fleet.knot_d[k0:k1] - p * SHAPE_STRIDE)).astype(np.int64)
        t, d = split_long_gaps(t, d)
        key = (p, t.tobytes(), d.tobytes())
        q = profiles.get(key)
        if q is None:
            q = profiles[key] = len(prof_pat)
            prof_pat.append(p)
            prof_t.append(t)
            prof_d.append(d)
        trip_start.append(start)
        trip_prof.append(q)
        trip_route.append(int(fleet.trip_route[i]))
        lines[fleet.trip_route[i]]["trips"] += 1
    order = np.argsort(trip_start, kind="stable")
    trip_start = np.array(trip_start)[order]
    trip_prof = np.array(trip_prof)[order]
    trip_route = np.array(trip_route)[order]

    # pattern polylines, simplified but keeping each vertex's distance along the original line
    shape_key = fleet.shape_key
    bounds = np.searchsorted(shape_key, np.arange(knot_pat.max() + 2) * SHAPE_STRIDE)
    used = sorted(set(prof_pat))
    remap = {p: n for n, p in enumerate(used)}
    pat_v0, pat_lonlat, pat_dist, pat_route = [0], [], [], []
    route_votes: dict[int, np.ndarray] = {}
    for q, r in zip(trip_prof, trip_route):
        route_votes.setdefault(prof_pat[q], np.zeros(len(lines), np.int64))[r] += 1
    for p in used:
        s0, s1 = bounds[p], bounds[p + 1]
        xy = np.column_stack([fleet.shape_x[s0:s1], fleet.shape_y[s0:s1]])
        cum = shape_key[s0:s1] - p * SHAPE_STRIDE
        idx = simplify(xy, a.tolerance)
        lon = xy[idx, 0] / proj.kx + proj.lon0
        lat = xy[idx, 1] / proj.ky + proj.lat0
        pat_lonlat.append(np.column_stack([lon, lat]).ravel())
        pat_dist.append(cum[idx])
        pat_v0.append(pat_v0[-1] + len(idx))
        pat_route.append(int(np.argmax(route_votes[p])))
    lonlat = np.concatenate(pat_lonlat).reshape(-1, 2)

    prof_k0 = np.concatenate([[0], np.cumsum([len(t) for t in prof_t])])
    dt = np.concatenate([np.diff(t, prepend=t[0]) for t in prof_t])
    dd = np.concatenate([np.diff(d, prepend=d[0]) for d in prof_d])
    assert dt.max() < 65536 and dd.max() < 65536

    arrays = {
        "pat_v0": np.array(pat_v0, np.uint32),
        "pat_lonlat": lonlat.ravel().astype(np.float32),
        "pat_dist": np.concatenate(pat_dist).astype(np.float32),
        "pat_route": np.array(pat_route, np.uint16),
        "prof_pat": np.array([remap[p] for p in prof_pat], np.uint16),
        "prof_k0": prof_k0.astype(np.uint32),
        "prof_d0": np.array([d[0] for d in prof_d], np.uint32),
        "prof_dt": dt.astype(np.uint16),
        "prof_dd": dd.astype(np.uint16),
        "trip_start": trip_start.astype(np.uint32),
        "trip_prof": trip_prof.astype(np.uint32),
        "trip_route": trip_route.astype(np.uint16),
    }
    os.makedirs(a.out, exist_ok=True)
    manifest, offset = [], 0
    # gzipped here and inflated by the browser, since static hosts rarely compress binary files
    with gzip.GzipFile(os.path.join(a.out, "fleet.bin.gz"), "wb", compresslevel=9, mtime=0) as f:
        for name, arr in arrays.items():
            data = arr.tobytes()
            data += b"\0" * (-len(data) % 4)
            manifest.append(dict(name=name, dtype=arr.dtype.name, offset=offset, length=len(arr)))
            f.write(data)
            offset += len(data)

    # buses on the road through the day, for the timeline
    ends = trip_start + np.array([t[-1] for t in prof_t])[trip_prof]
    steps = np.arange(T_START, T_END + 1, 300)
    activity = [int(((trip_start <= s) & (ends > s)).sum()) for s in steps]

    lo, hi = np.percentile(lonlat, 0.2, axis=0), np.percentile(lonlat, 99.8, axis=0)
    order_lines = sorted(range(len(lines)), key=lambda i: natural_key(lines[i]["name"]))
    meta = dict(
        date=date.isoformat(),
        weekday=date.strftime("%A"),
        t_start=T_START,
        t_end=T_END,
        trail=TRAIL_S,
        bounds=[round(float(lo[0]), 4), round(float(lo[1]), 4), round(float(hi[0]), 4), round(float(hi[1]), 4)],
        projection=dict(lon0=proj.lon0, lat0=proj.lat0, kx=proj.kx, ky=proj.ky),
        activity=dict(step=300, counts=activity),
        arrays=manifest,
        lines=lines,
        line_order=order_lines,
    )
    with open(os.path.join(a.out, "fleet.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
    stops = export_stops(feed, day, {rid: i for i, rid in enumerate(routes["route_id"])})
    with open(os.path.join(a.out, "stops.json"), "w") as f:
        json.dump(stops, f, ensure_ascii=False, separators=(",", ":"))
    print(
        f"{len(stops['name']):,} stops, {len(trip_start):,} trips, {len(prof_pat):,} profiles ({len(dt):,} knots), {len(used):,} patterns "
        f"({len(lonlat):,} vertices), {offset / 1e6:.1f} MB [{time.time() - t0:.0f}s]"
    )


if __name__ == "__main__":
    main()
