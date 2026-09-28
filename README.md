# map_builder

Animated maps of transit traffic.

## Paris buses

`render_paris_bus.py` renders every bus of Paris and the petite couronne over one weekday, from the
Île-de-France Mobilités GTFS timetable, in the style of a dark "glowing network" map.

Bus positions are interpolated between scheduled stop times along each trip's route shape
(or straight between stops when a trip has no shape). Stops that share a rounded minute are
spread out by distance, so buses move smoothly instead of jumping.

Each bus takes its line's official colour, brightened for a dark map.

```
pip install -r requirements.txt
# GTFS: https://data.iledefrance-mobilites.fr/explore/dataset/offre-horaires-tc-gtfs-idfm/
python render_paris_bus.py --gtfs IDFM-gtfs.zip --out paris-bus.mp4 --cache day.pkl
python render_paris_bus.py --gtfs IDFM-gtfs.zip --still 08:30 --out preview.png --cache day.pkl
```

By default it picks the busiest Tuesday in the feed (`--weekday`, `--date` to change) and renders
04:00 to 04:00 in 72 s of 1080p at 30 fps. `--cache` saves the processed trips so later renders skip
the GTFS parsing.

To try it without the real feed: `python scripts/make_synthetic_gtfs.py synth.zip`.

## Web viewer

`docs/` is a static site (GitHub Pages ready) that replays the same day in the browser: play/pause,
a timeline to scrub through the day, a multi-select line filter with RATP / IDFM line badges, and a
pan/zoom map kept to the area the network covers. Buses are drawn with [deck.gl](https://deck.gl) over
[MapLibre](https://maplibre.org) and a CARTO dark basemap.

Its data comes from `build_web_data.py`, which writes `docs/fleet/` (about 3.3 MB gzipped): each trip is
a start time plus a timing profile along a route pattern, and the browser interpolates positions
every frame.

```
python build_web_data.py --gtfs IDFM-gtfs.zip --cache day.pkl
python -m http.server -d docs 8000   # then open http://localhost:8000
```

Line colours are the official ones from the GTFS. Line badges use Parisine, RATP's typeface, when it is
installed on the viewer's machine; since it is not freely licensed, Fira Sans (SIL OFL) stands in otherwise.
MapLibre GL JS (BSD-3-Clause) and deck.gl (MIT) are vendored in `docs/vendor`.

Department outlines in `assets/geo` come from [france-geojson](https://github.com/gregoiredavid/france-geojson)
(IGN / INSEE, Licence Ouverte). Fonts are Lato and Playfair Display (SIL Open Font License).
