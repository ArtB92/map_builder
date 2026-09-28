"""GTFS loading: pick a service day and extract the bus trips that touch a region."""
from __future__ import annotations

import datetime as dt
import os
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Plain GTFS bus plus the extended "bus service" range (700-799).
BUS_ROUTE_TYPES = {3} | set(range(700, 800))

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class Feed:
    """Thin reader over a GTFS zip file or an unzipped directory."""

    def __init__(self, path: str):
        self.path = path
        self._zip = zipfile.ZipFile(path) if zipfile.is_zipfile(path) else None

    def _names(self) -> dict[str, str]:
        if self._zip is None:
            return {n: os.path.join(self.path, n) for n in os.listdir(self.path)}
        # Some feeds nest files in a folder inside the zip.
        return {os.path.basename(n): n for n in self._zip.namelist() if n.endswith(".txt")}

    def has(self, name: str) -> bool:
        return name in self._names()

    def open(self, name: str):
        target = self._names()[name]
        if self._zip is None:
            return open(target, "rb")
        return self._zip.open(target)

    def read(self, name: str, **kw) -> pd.DataFrame:
        with self.open(name) as f:
            return pd.read_csv(f, dtype=str, encoding="utf-8-sig", **kw)


def parse_hms(s: pd.Series) -> np.ndarray:
    """'HH:MM:SS' (hours may exceed 24) -> seconds since midnight, NaN for blanks."""
    parts = s.fillna("").str.strip().str.split(":", expand=True)
    if parts.shape[1] < 3:
        return np.full(len(s), np.nan)
    h = pd.to_numeric(parts[0], errors="coerce")
    m = pd.to_numeric(parts[1], errors="coerce")
    sec = pd.to_numeric(parts[2], errors="coerce")
    return (h * 3600 + m * 60 + sec).to_numpy(dtype=float)


def active_services(feed: Feed, date: dt.date) -> set[str]:
    ymd = date.strftime("%Y%m%d")
    services: set[str] = set()
    if feed.has("calendar.txt"):
        cal = feed.read("calendar.txt")
        day = WEEKDAYS[date.weekday()]
        on = (cal[day] == "1") & (cal["start_date"] <= ymd) & (cal["end_date"] >= ymd)
        services |= set(cal.loc[on, "service_id"])
    if feed.has("calendar_dates.txt"):
        cd = feed.read("calendar_dates.txt")
        cd = cd[cd["date"] == ymd]
        services |= set(cd.loc[cd["exception_type"] == "1", "service_id"])
        services -= set(cd.loc[cd["exception_type"] == "2", "service_id"])
    return services


def pick_service_date(feed: Feed, weekday: str = "tuesday", date: str | None = None) -> dt.date:
    """Return the requested date, or the weekday in the feed's range with the most bus trips."""
    if date:
        return dt.datetime.strptime(date, "%Y-%m-%d").date()
    candidates: set[str] = set()
    if feed.has("calendar.txt"):
        cal = feed.read("calendar.txt")
        start, end = cal["start_date"].min(), cal["end_date"].max()
        d0 = dt.datetime.strptime(start, "%Y%m%d").date()
        d1 = dt.datetime.strptime(end, "%Y%m%d").date()
        candidates |= {(d0 + dt.timedelta(n)).strftime("%Y%m%d") for n in range((d1 - d0).days + 1)}
    if feed.has("calendar_dates.txt"):
        candidates |= set(feed.read("calendar_dates.txt", usecols=["date"])["date"])
    target = WEEKDAYS.index(weekday)
    days = sorted(
        d for d in (dt.datetime.strptime(c, "%Y%m%d").date() for c in candidates) if d.weekday() == target
    )
    if not days:
        raise SystemExit(f"No {weekday} in the feed's calendar")
    routes = feed.read("routes.txt")
    bus_routes = set(routes.loc[routes["route_type"].astype(int).isin(BUS_ROUTE_TYPES), "route_id"])
    trips = feed.read("trips.txt", usecols=["route_id", "service_id"])
    per_service = trips[trips["route_id"].isin(bus_routes)].groupby("service_id").size()
    # Only look at the first few weeks: feeds are densest near their start.
    scored = [(int(per_service.reindex(list(active_services(feed, d))).fillna(0).sum()), d) for d in days[:6]]
    best = max(scored, key=lambda x: (x[0], -x[1].toordinal()))
    return best[1]


@dataclass
class BusDay:
    date: dt.date
    routes: pd.DataFrame  # route_id, route_short_name, route_long_name, route_color, agency_id
    trips: pd.DataFrame  # trip_id, route_id, shape_id (maybe NaN), direction_id
    stop_times: pd.DataFrame  # trip_id, stop_sequence, stop_id, arr, dep
    stops: pd.DataFrame  # stop_id, lat, lon
    shapes: pd.DataFrame | None  # shape_id, lat, lon, seq
    agencies: pd.DataFrame | None


def load_bus_day(feed: Feed, date: dt.date, bbox: tuple[float, float, float, float]) -> BusDay:
    """Bus trips running on `date` with at least one stop inside bbox (lon0, lat0, lon1, lat1)."""
    services = active_services(feed, date)
    routes = feed.read("routes.txt")
    routes = routes[routes["route_type"].astype(int).isin(BUS_ROUTE_TYPES)].copy()
    trip_cols = ["trip_id", "route_id", "service_id"]
    header = feed.read("trips.txt", nrows=0).columns
    trip_cols += [c for c in ("shape_id", "direction_id", "trip_headsign") if c in header]
    trips = feed.read("trips.txt", usecols=trip_cols)
    trips = trips[trips["service_id"].isin(services) & trips["route_id"].isin(set(routes["route_id"]))]

    stops = feed.read("stops.txt", usecols=["stop_id", "stop_lat", "stop_lon"])
    stops["lat"] = stops.pop("stop_lat").astype(float)
    stops["lon"] = stops.pop("stop_lon").astype(float)
    lon0, lat0, lon1, lat1 = bbox
    inside = set(stops.loc[stops.lon.between(lon0, lon1) & stops.lat.between(lat0, lat1), "stop_id"])

    wanted = set(trips["trip_id"])
    parts = []
    with feed.open("stop_times.txt") as f:
        reader = pd.read_csv(
            f,
            dtype=str,
            encoding="utf-8-sig",
            usecols=["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"],
            chunksize=2_000_000,
        )
        for chunk in reader:
            parts.append(chunk[chunk["trip_id"].isin(wanted)])
    st = pd.concat(parts, ignore_index=True)
    touching = set(st.loc[st["stop_id"].isin(inside), "trip_id"])
    st = st[st["trip_id"].isin(touching)].copy()
    st["stop_sequence"] = st["stop_sequence"].astype(int)
    st["arr"] = parse_hms(st.pop("arrival_time"))
    st["dep"] = parse_hms(st.pop("departure_time"))
    st.sort_values(["trip_id", "stop_sequence"], inplace=True, kind="stable")
    trips = trips[trips["trip_id"].isin(touching)].reset_index(drop=True)
    routes = routes[routes["route_id"].isin(set(trips["route_id"]))].reset_index(drop=True)

    shapes = None
    if "shape_id" in trips.columns and feed.has("shapes.txt") and trips["shape_id"].notna().any():
        shapes = feed.read("shapes.txt")
        shapes = shapes[shapes["shape_id"].isin(set(trips["shape_id"].dropna()))].copy()
        shapes["lat"] = shapes.pop("shape_pt_lat").astype(float)
        shapes["lon"] = shapes.pop("shape_pt_lon").astype(float)
        shapes["seq"] = shapes.pop("shape_pt_sequence").astype(int)
        shapes.sort_values(["shape_id", "seq"], inplace=True)
    agencies = feed.read("agency.txt") if feed.has("agency.txt") else None
    stops = stops[stops["stop_id"].isin(set(st["stop_id"]))].reset_index(drop=True)
    return BusDay(date, routes, trips, st.reset_index(drop=True), stops, shapes, agencies)
