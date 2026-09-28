# map_builder

Animated maps of transit traffic.

## Paris buses

`render_paris_bus.py` renders every bus of Paris and the petite couronne over one weekday, from the
Île-de-France Mobilités GTFS timetable, in the style of a dark "glowing network" map.

Bus positions are interpolated between scheduled stop times along each trip's route shape
(or straight between stops when a trip has no shape). Stops that share a rounded minute are
spread out by distance, so buses move smoothly instead of jumping.

Two looks:

- `--mode lines`: one colour per line (the official line colour, brightened for a dark map)
- `--mode speed`: trails coloured by estimated speed, from the distance and scheduled time between stops

```
pip install -r requirements.txt
# GTFS: https://data.iledefrance-mobilites.fr/explore/dataset/offre-horaires-tc-gtfs-idfm/
python render_paris_bus.py --gtfs IDFM-gtfs.zip --mode lines --out paris-bus-lines.mp4 --cache day.pkl
python render_paris_bus.py --gtfs IDFM-gtfs.zip --mode speed --out paris-bus-speed.mp4 --cache day.pkl
python render_paris_bus.py --gtfs IDFM-gtfs.zip --still 08:30 --out preview.png --cache day.pkl
```

By default it picks the busiest Tuesday in the feed (`--weekday`, `--date` to change) and renders
04:00 to 04:00 in 72 s of 1080p at 30 fps. `--cache` saves the processed trips so later renders skip
the GTFS parsing.

To try it without the real feed: `python scripts/make_synthetic_gtfs.py synth.zip`.

Department outlines in `assets/geo` come from [france-geojson](https://github.com/gregoiredavid/france-geojson)
(IGN / INSEE, Licence Ouverte). Fonts are Lato and Playfair Display (SIL Open Font License).
