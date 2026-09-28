"""Turn scheduled bus trips into positions: stop times interpolated along route shapes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .gtfs import BusDay

EARTH_R = 6_371_000.0
SHAPE_STRIDE = 1e7  # metres between consecutive shapes on the global distance axis
TRIP_STRIDE = 1e6  # seconds between consecutive trips on the global time axis


class Projection:
    """Local equirectangular projection to metres, accurate enough at city scale."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = np.cos(np.radians(lat0)) * EARTH_R * np.pi / 180
        self.ky = EARTH_R * np.pi / 180

    def __call__(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky


def _cumlen(xy: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])


def project_stops(line: np.ndarray, pts: np.ndarray, tol: float = 25.0) -> np.ndarray:
    """Distance along `line` of each point, kept non-decreasing so loops don't jump ahead."""
    a, b = line[:-1], line[1:]
    ab = b - a
    seg_len2 = np.maximum((ab**2).sum(1), 1e-9)
    cum = _cumlen(line)
    out = np.empty(len(pts))
    start = 0
    for i, p in enumerate(pts):
        t = np.clip(((p - a[start:]) * ab[start:]).sum(1) / seg_len2[start:], 0, 1)
        proj = a[start:] + t[:, None] * ab[start:]
        d2 = ((proj - p) ** 2).sum(1)
        # earliest segment that is (almost) as close as the best one
        j = int(np.argmax(d2 <= d2.min() + tol**2))
        seg = start + j
        out[i] = cum[seg] + t[j] * np.sqrt(seg_len2[seg])
        start = seg
    return np.maximum.accumulate(out)


def spread_times(t: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Schedules are often rounded to the minute, so several stops share one time.

    Keep the first stop of each run as a timing anchor and interpolate the others
    by distance between anchors, so the bus moves steadily instead of teleporting.
    """
    ok = np.flatnonzero(~np.isnan(t))
    if len(ok) < 2:
        return t
    tv, dv = t[ok], d[ok]
    first_of_run = np.concatenate([[True], tv[1:] > tv[:-1]])
    at, ad = tv[first_of_run], dv[first_of_run]
    if not first_of_run[-1]:
        # the terminus shares its minute with earlier stops: let it arrive a little later
        at, ad = np.append(at, tv[-1] + 30.0), np.append(ad, dv[-1])
    return np.interp(d, ad, at)


@dataclass
class Fleet:
    """Vectorised position lookups for every trip of the day."""

    # shapes, concatenated on one axis (shape k occupies [k*SHAPE_STRIDE, ...))
    shape_key: np.ndarray
    shape_x: np.ndarray
    shape_y: np.ndarray
    # knots: (time, distance) per trip, concatenated (trip i keys at i*TRIP_STRIDE + t)
    knot_key: np.ndarray
    knot_d: np.ndarray  # global shape distance (includes the shape offset)
    knot_speed: np.ndarray  # km/h over the interval starting at this knot
    # per trip
    trip_start: np.ndarray
    trip_end: np.ndarray
    trip_route: np.ndarray  # index into routes
    trip_k0: np.ndarray
    trip_k1: np.ndarray
    routes: pd.DataFrame
    network: list[np.ndarray]  # distinct polylines (x, y metres) for the background

    def active(self, t: float) -> np.ndarray:
        return np.flatnonzero((self.trip_start <= t) & (self.trip_end > t))

    def _knot_index(self, trips: np.ndarray, t: np.ndarray) -> np.ndarray:
        k = np.searchsorted(self.knot_key, trips * TRIP_STRIDE + t, side="right") - 1
        return np.clip(k, self.trip_k0[trips], self.trip_k1[trips] - 2)

    def distance(self, trips: np.ndarray, t: np.ndarray) -> np.ndarray:
        t = np.clip(t, self.trip_start[trips], self.trip_end[trips])
        k = self._knot_index(trips, t)
        t0 = self.knot_key[k] - trips * TRIP_STRIDE
        t1 = self.knot_key[k + 1] - trips * TRIP_STRIDE
        f = np.clip((t - t0) / np.maximum(t1 - t0, 1e-6), 0, 1)
        return self.knot_d[k] + f * (self.knot_d[k + 1] - self.knot_d[k])

    def xy(self, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        j = np.clip(np.searchsorted(self.shape_key, d, side="right") - 1, 0, len(self.shape_key) - 2)
        s0, s1 = self.shape_key[j], self.shape_key[j + 1]
        f = np.clip((d - s0) / np.maximum(s1 - s0, 1e-6), 0, 1)
        return self.shape_x[j] + f * (self.shape_x[j + 1] - self.shape_x[j]), self.shape_y[j] + f * (
            self.shape_y[j + 1] - self.shape_y[j]
        )

    def speed(self, trips: np.ndarray, t: float) -> np.ndarray:
        tt = np.clip(np.full(len(trips), t), self.trip_start[trips], self.trip_end[trips])
        return self.knot_speed[self._knot_index(trips, tt)]


def build_fleet(day: BusDay, proj: Projection, min_dwell: float = 20.0, speed_window_m: float = 400.0) -> Fleet:
    stops = day.stops.set_index("stop_id")
    st = day.stop_times
    trips = day.trips.set_index("trip_id")
    routes = day.routes.reset_index(drop=True)
    route_idx = {r: i for i, r in enumerate(routes["route_id"])}

    shape_xy: dict[str, np.ndarray] = {}
    if day.shapes is not None:
        for sid, g in day.shapes.groupby("shape_id", sort=False):
            x, y = proj(g["lon"].to_numpy(), g["lat"].to_numpy())
            xy = np.column_stack([x, y])
            keep = np.concatenate([[True], np.hypot(*np.diff(xy, axis=0).T) > 0.01])
            if keep.sum() >= 2:
                shape_xy[sid] = xy[keep]

    patterns: dict[tuple, int] = {}
    pat_line: list[np.ndarray] = []
    pat_stop_d: list[np.ndarray] = []

    knot_t, knot_d, knot_v = [], [], []
    t_start, t_end, t_route, k0, k1 = [], [], [], [], []
    n_knots = 0

    grouped = st.groupby("trip_id", sort=False)
    for trip_id, g in grouped:
        stop_ids = tuple(g["stop_id"])
        if len(stop_ids) < 2:
            continue
        shape_id = trips.at[trip_id, "shape_id"] if "shape_id" in trips.columns else None
        shape_id = shape_id if isinstance(shape_id, str) and shape_id in shape_xy else None
        key = (shape_id, stop_ids)
        p = patterns.get(key)
        if p is None:
            sx, sy = proj(stops.loc[list(stop_ids), "lon"].to_numpy(), stops.loc[list(stop_ids), "lat"].to_numpy())
            spts = np.column_stack([sx, sy])
            if shape_id is not None:
                line = shape_xy[shape_id]
                sd = project_stops(line, spts)
            else:
                keep = np.concatenate([[True], np.hypot(*np.diff(spts, axis=0).T) > 0.01])
                line = spts[keep] if keep.sum() >= 2 else np.vstack([spts[0], spts[0] + 1])
                sd = project_stops(line, spts)
            p = patterns[key] = len(pat_line)
            pat_line.append(line)
            pat_stop_d.append(sd)
        sd = pat_stop_d[p]
        arr, dep = g["arr"].to_numpy(), g["dep"].to_numpy()
        arr = np.where(np.isnan(arr), dep, arr)
        dep = np.where(np.isnan(dep), arr, dep)
        t = spread_times(arr, sd)
        if np.isnan(t).any() or t[-1] <= t[0]:
            continue
        dwell = np.where(dep - arr >= min_dwell, dep - arr, 0.0)
        # knots: arrival at each stop, plus a departure knot when the bus waits
        tt, dd = [], []
        shift = 0.0
        for i in range(len(t)):
            tt.append(t[i] + shift)
            dd.append(sd[i])
            if dwell[i] > 0 and 0 < i < len(t) - 1:
                shift += dwell[i]
                tt.append(t[i] + shift)
                dd.append(sd[i])
        tt = np.maximum.accumulate(np.array(tt))
        dd = np.array(dd)
        # speed: distance over time, smoothed over a window so rounding noise doesn't dominate
        lo = np.searchsorted(dd, dd[:-1] - speed_window_m / 2, side="left")
        hi = np.maximum(np.searchsorted(dd, dd[1:] + speed_window_m / 2, side="right") - 1, np.arange(1, len(dd)))
        span_t = tt[hi] - tt[lo]
        v = np.where(span_t > 0, 3.6 * (dd[hi] - dd[lo]) / np.maximum(span_t, 1e-6), 0.0)
        v = np.append(np.where(dd[1:] == dd[:-1], 0.0, v), 0.0)  # zero while dwelling
        trip_no = len(t_start)
        knot_t.append(tt + trip_no * TRIP_STRIDE)
        knot_d.append(dd + p * SHAPE_STRIDE)
        knot_v.append(np.clip(v, 0, 90))
        t_start.append(tt[0])
        t_end.append(tt[-1])
        t_route.append(route_idx[trips.at[trip_id, "route_id"]])
        k0.append(n_knots)
        n_knots += len(tt)
        k1.append(n_knots)

    shape_key = np.concatenate([_cumlen(l) + i * SHAPE_STRIDE for i, l in enumerate(pat_line)])
    lines_xy = np.concatenate(pat_line)
    # distinct geometries for the background (patterns sharing a shape draw once)
    seen, network = set(), []
    for (sid, stop_ids), p in patterns.items():
        k = sid if sid is not None else stop_ids
        if k not in seen:
            seen.add(k)
            network.append(pat_line[p])
    return Fleet(
        shape_key=shape_key,
        shape_x=lines_xy[:, 0],
        shape_y=lines_xy[:, 1],
        knot_key=np.concatenate(knot_t),
        knot_d=np.concatenate(knot_d),
        knot_speed=np.concatenate(knot_v),
        trip_start=np.array(t_start),
        trip_end=np.array(t_end),
        trip_route=np.array(t_route),
        trip_k0=np.array(k0),
        trip_k1=np.array(k1),
        routes=routes,
        network=network,
    )
