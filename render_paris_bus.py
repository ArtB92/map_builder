"""Render an animated map of every Paris bus on one weekday from an IDFM GTFS feed.

    python render_paris_bus.py --gtfs IDFM-gtfs.zip --out paris-bus.mp4
"""
import argparse
import os
import pickle
import time

from mapbuilder.fleet import Projection, build_fleet
from mapbuilder.gtfs import Feed, load_bus_day, pick_service_date
from mapbuilder.render import Scene, Style, render_still, render_video

HERE = os.path.dirname(os.path.abspath(__file__))
# Paris and the petite couronne, with a margin so lines crossing the edge are kept
BBOX = (2.10, 48.66, 2.66, 49.03)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gtfs", required=True, help="GTFS zip or directory")
    ap.add_argument("--out", required=True, help=".mp4 (or .png with --still)")
    ap.add_argument("--weekday", default="tuesday")
    ap.add_argument("--date", help="YYYY-MM-DD, overrides --weekday")
    ap.add_argument("--seconds", type=float, default=72, help="video length")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--still", help="render a single frame at HH:MM instead of a video")
    ap.add_argument("--cache", help="pickle file to cache the processed trips in")
    a = ap.parse_args()

    t0 = time.time()
    if a.cache and os.path.exists(a.cache):
        date, fleet = pickle.load(open(a.cache, "rb"))
    else:
        feed = Feed(a.gtfs)
        date = pick_service_date(feed, a.weekday, a.date)
        print(f"service day: {date:%A %Y-%m-%d}")
        day = load_bus_day(feed, date, BBOX)
        print(f"{len(day.trips):,} bus trips on {len(day.routes)} lines, {len(day.stop_times):,} stop times "
              f"({'with' if day.shapes is not None else 'without'} shapes) [{time.time() - t0:.0f}s]")
        style = Style()
        fleet = build_fleet(day, Projection(*style.center))
        if a.cache:
            pickle.dump((date, fleet), open(a.cache, "wb"))
    print(f"fleet ready: {len(fleet.trip_start):,} trips, {len(fleet.network)} geometries [{time.time() - t0:.0f}s]")

    weekday = date.strftime("%A").upper()
    subtitle = f"EVERY BUS ON A TYPICAL {weekday}"
    style = Style(seconds=a.seconds, fps=a.fps, subtitle=subtitle, date_label=date.strftime("%d %B %Y"))
    scene = Scene(fleet, style, os.path.join(HERE, "assets/geo"), os.path.join(HERE, "assets/fonts"))
    if a.still:
        hh, mm = (int(x) for x in a.still.split(":"))
        t = hh * 3600 + mm * 60
        render_still(scene, t if t >= style.t_start else t + 86400, a.out)
    else:
        render_video(scene, a.out, workers=a.workers)
    print(f"wrote {a.out} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
