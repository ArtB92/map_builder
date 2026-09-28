"""Write a small fake GTFS feed around Paris, to exercise the pipeline without the real data."""
import csv
import io
import sys
import zipfile

import numpy as np

rng = np.random.default_rng(7)
out = sys.argv[1] if len(sys.argv) > 1 else "synthetic-gtfs.zip"
KX, KY = 73_200.0, 111_200.0  # metres per degree at Paris


def rows_to_csv(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


routes, trips, stop_times, stops, shapes = [], [], [], [], []
stop_n = 0
for r in range(120):
    a = np.array([2.35, 48.86]) + rng.normal(0, [0.07, 0.045])
    b = np.array([2.35, 48.86]) + rng.normal(0, [0.09, 0.06])
    ctrl = (a + b) / 2 + rng.normal(0, [0.02, 0.013])
    s = np.linspace(0, 1, 60)[:, None]
    curve = (1 - s) ** 2 * a + 2 * (1 - s) * s * ctrl + s**2 * b
    seg = np.hypot(np.diff(curve[:, 0]) * KX, np.diff(curve[:, 1]) * KY)
    cum = np.concatenate([[0], np.cumsum(seg)])
    n_stops = max(3, int(cum[-1] / 380))
    stop_d = np.linspace(0, cum[-1], n_stops)
    stop_ll = np.column_stack([np.interp(stop_d, cum, curve[:, 0]), np.interp(stop_d, cum, curve[:, 1])])
    ids = []
    for ll in stop_ll:
        stops.append((f"S{stop_n}", f"Stop {stop_n}", f"{ll[1]:.6f}", f"{ll[0]:.6f}"))
        ids.append(f"S{stop_n}")
        stop_n += 1
    color = "%02X%02X%02X" % tuple(rng.integers(0, 256, 3))
    rid = f"R{r}"
    routes.append((rid, "A1", str(20 + r), f"Line {20 + r}", "3", color))
    with_shape = r % 5 != 0  # some routes have no shape, to test the fallback
    for direction in (0, 1):
        sid = f"SH{r}_{direction}"
        pts = curve if direction == 0 else curve[::-1]
        if with_shape:
            for k, (lon, lat) in enumerate(pts):
                shapes.append((sid, f"{lat:.6f}", f"{lon:.6f}", str(k)))
        order = ids if direction == 0 else ids[::-1]
        headway = rng.integers(5, 16) * 60
        for dep in range(int(5.5 * 3600) + direction * 120, int(24.6 * 3600), int(headway)):
            tid = f"T{r}_{direction}_{dep}"
            trips.append((rid, "WK", tid, sid if with_shape else "", str(direction)))
            hour = dep / 3600
            rush = np.exp(-((hour - 8.5) ** 2) / 2) + np.exp(-((hour - 18) ** 2) / 2)
            v = (22 - 9 * rush) / 3.6  # m/s
            t = dep
            gaps = np.diff(stop_d)
            for k, sidx in enumerate(order):
                stamp = int(t // 60 * 60)  # minute rounding, like many real feeds
                hms = f"{stamp // 3600:02d}:{stamp % 3600 // 60:02d}:00"
                stop_times.append((tid, hms, hms, sidx, str(k + 1)))
                if k < len(gaps):
                    t += gaps[k] / (v * rng.uniform(0.6, 1.4)) + 15

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("agency.txt", rows_to_csv(["agency_id", "agency_name", "agency_url", "agency_timezone"], [("A1", "Synthetic", "http://x", "Europe/Paris")]))
    z.writestr("routes.txt", rows_to_csv(["route_id", "agency_id", "route_short_name", "route_long_name", "route_type", "route_color"], routes))
    z.writestr("trips.txt", rows_to_csv(["route_id", "service_id", "trip_id", "shape_id", "direction_id"], trips))
    z.writestr("stop_times.txt", rows_to_csv(["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"], stop_times))
    z.writestr("stops.txt", rows_to_csv(["stop_id", "stop_name", "stop_lat", "stop_lon"], stops))
    z.writestr("shapes.txt", rows_to_csv(["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"], shapes))
    z.writestr("calendar.txt", rows_to_csv(
        ["service_id", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "start_date", "end_date"],
        [("WK", "1", "1", "1", "1", "1", "0", "0", "20260901", "20261231")]))
print(out, len(trips), "trips", len(stop_times), "stop times")
