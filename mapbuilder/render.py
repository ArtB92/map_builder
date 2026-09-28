"""Frame renderer: glowing buses with fading trails over a dark basemap, encoded with ffmpeg."""
from __future__ import annotations

import colorsys
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
from dataclasses import dataclass, field

import cv2
import imageio_ffmpeg
import numpy as np

from .fleet import Fleet, Projection
from .overlay import Overlay

SUB = 16  # cv2 sub-pixel precision (shift=4)

# speed colour ramp (km/h -> RGB), slow is hot, fast is cool
SPEED_STOPS = [
    (0, (1.00, 0.18, 0.28)),
    (8, (1.00, 0.45, 0.20)),
    (14, (1.00, 0.80, 0.25)),
    (20, (0.55, 1.00, 0.45)),
    (28, (0.25, 0.90, 1.00)),
    (40, (0.55, 0.60, 1.00)),
]
SPEED_BINS = [(0, 8, "Under 8 km/h"), (8, 14, "8 to 14 km/h"), (14, 20, "14 to 20 km/h"), (20, 28, "20 to 28 km/h"), (28, 999, "Over 28 km/h")]


def speed_rgb(v: np.ndarray) -> np.ndarray:
    xs = [s for s, _ in SPEED_STOPS]
    return np.stack([np.interp(v, xs, [c[i] for _, c in SPEED_STOPS]) for i in range(3)], axis=-1)


def _hex_rgb(h: str) -> tuple[float, float, float] | None:
    h = (h or "").strip().lstrip("#")
    if len(h) != 6:
        return None
    try:
        return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    except ValueError:
        return None


def line_color(route_color: str | None, name: str) -> tuple[float, float, float]:
    """Official line colour when there is one, lifted so it glows on a dark map."""
    rgb = _hex_rgb(route_color if isinstance(route_color, str) else "")
    if rgb is None or max(rgb) - min(rgb) < 0.08:  # missing, white, black or grey
        hue = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return colorsys.hsv_to_rgb(hue, 0.6, 1.0)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, min(max(l, 0.6), 0.75), max(s, 0.7))


@dataclass
class Style:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    seconds: float = 72.0
    t_start: float = 4 * 3600
    t_end: float = 28 * 3600
    center: tuple[float, float] = (2.38, 48.845)  # lon, lat drawn at map_px
    map_px: tuple[float, float] = (1255, 545)
    metres_per_px: float = 37.0
    trail_sim_seconds: float = 150.0  # how long a trail stays visible (simulated time)
    trail_gain: float = 0.8
    glow: float = 1.1
    substeps: int = 4
    mode: str = "lines"  # "lines" or "speed"
    title: str = "PARIS"
    subtitle: str = "EVERY BUS ON A TYPICAL TUESDAY"
    date_label: str = ""
    core_departments: tuple[str, ...] = ("75", "92", "93", "94")
    labels: list = field(
        default_factory=lambda: [
            ("Paris", 2.3470, 48.8584, (-200, -260)),
            ("La Défense", 2.2380, 48.8918, (-120, 70)),
            ("Saint-Denis", 2.3574, 48.9362, (70, -70)),
            ("Montreuil", 2.4485, 48.8638, (120, -50)),
            ("Créteil", 2.4556, 48.7904, (120, 40)),
            ("Orly", 2.3652, 48.7262, (-120, 60)),
        ]
    )

    @property
    def n_frames(self) -> int:
        return int(round(self.seconds * self.fps))

    def frame_time(self, i: int) -> float:
        return self.t_start + (self.t_end - self.t_start) * i / max(self.n_frames - 1, 1)


class Scene:
    def __init__(self, fleet: Fleet, style: Style, geo_dir: str, font_dir: str):
        self.fleet, self.s = fleet, style
        self.proj = Projection(*style.center)
        W, H = style.width, style.height
        self.W, self.H = W, H
        routes = fleet.routes
        names = routes.get("route_short_name", routes["route_id"]).fillna(routes["route_id"]).astype(str)
        self.route_names = names.to_numpy()
        colors = [line_color(c, n) for c, n in zip(routes.get("route_color", [None] * len(routes)), names)]
        self.route_rgb = np.array(colors, dtype=np.float32)
        uniq, self.route_color_id = np.unique(np.round(self.route_rgb * 255).astype(int), axis=0, return_inverse=True)
        self.color_table = uniq.astype(np.float32) / 255
        self.dt = (style.t_end - style.t_start) / max(style.n_frames - 1, 1)
        self.decay = float(np.exp(-self.dt / style.trail_sim_seconds))
        self._build_background(geo_dir)
        self.daily = self._daily_counts()
        self.overlay = Overlay(self, font_dir)

    # -- geometry ---------------------------------------------------------
    def to_px(self, x, y):
        s = self.s
        return s.map_px[0] + np.asarray(x) / s.metres_per_px, s.map_px[1] - np.asarray(y) / s.metres_per_px

    def lonlat_px(self, lon, lat):
        return self.to_px(*self.proj(lon, lat))

    def _poly_px(self, rings):
        out = []
        for ring in rings:
            a = np.asarray(ring, dtype=float)
            px, py = self.lonlat_px(a[:, 0], a[:, 1])
            out.append(np.round(np.column_stack([px, py]) * SUB).astype(np.int32))
        return out

    def _build_background(self, geo_dir: str):
        W, H = self.W, self.H
        bg = np.zeros((H, W, 3), np.float32)
        bg[:] = (0.035, 0.038, 0.050)
        core = np.zeros((H, W), np.uint8)
        land = np.zeros((H, W, 3), np.uint8)
        borders = np.zeros((H, W, 3), np.uint8)
        for fn in sorted(os.listdir(geo_dir)):
            if not fn.endswith(".geojson"):
                continue
            code = fn.split("-")[1]
            gj = json.load(open(os.path.join(geo_dir, fn)))
            for feat in gj.get("features", [gj]):
                geom = feat.get("geometry", feat)
                polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
                for poly in polys:
                    rings = self._poly_px(poly)
                    is_core = code in self.s.core_departments
                    fill = (44, 50, 66) if is_core else (24, 27, 36)
                    cv2.fillPoly(land, rings[:1], fill, cv2.LINE_AA, shift=4)
                    if len(rings) > 1:
                        cv2.fillPoly(land, rings[1:], (0, 0, 0), cv2.LINE_AA, shift=4)
                    if is_core:
                        cv2.fillPoly(core, rings[:1], 255, cv2.LINE_AA, shift=4)
                    cv2.polylines(borders, rings, True, (78, 86, 104), 1, cv2.LINE_AA, shift=4)
        net = np.zeros((H, W, 3), np.uint8)
        lines = []
        for line in self.fleet.network:
            px, py = self.to_px(line[:, 0], line[:, 1])
            lines.append(np.round(np.column_stack([px, py]) * SUB).astype(np.int32))
        cv2.polylines(net, lines, False, (120, 132, 156), 1, cv2.LINE_AA, shift=4)
        net_f = net.astype(np.float32) / 255
        net_glow = cv2.GaussianBlur(net_f, (0, 0), 3)

        # where moving things may appear: fade out behind the left panel and at the card edge,
        # and dim everything outside the core departments
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)
        fx = np.clip((xs - 560) / 160, 0, 1) * np.clip((W - 30 - xs) / 60, 0, 1)
        fy = np.clip((ys - 30) / 60, 0, 1) * np.clip((H - 60 - ys) / 60, 0, 1)
        frame_mask = fy[:, None] * fx[None, :]
        core_soft = cv2.GaussianBlur(core.astype(np.float32) / 255, (0, 0), 25)
        self.mask = (frame_mask * (0.3 + 0.7 * core_soft))[..., None].astype(np.float32)

        m = frame_mask[..., None].astype(np.float32)
        bg = bg + land.astype(np.float32) / 255 * m + borders.astype(np.float32) / 255 * m * 0.6
        bg = bg + (net_f * 0.22 + net_glow * 0.18) * m * (0.5 + 0.5 * self.mask)
        self.background = bg.astype(np.float32)

    def _daily_counts(self) -> np.ndarray:
        s = self.s
        ts = np.linspace(s.t_start, s.t_end, 289)
        f = self.fleet
        return np.array([np.count_nonzero((f.trip_start <= t) & (f.trip_end > t)) for t in ts])

    # -- per-frame state ----------------------------------------------------
    def bus_state(self, t: float):
        f = self.fleet
        trips = f.active(t)
        steps = np.linspace(t - self.dt, t, self.s.substeps + 1)
        pts = []
        for tt in steps:
            px, py = self.to_px(*f.xy(f.distance(trips, np.full(len(trips), tt))))
            pts.append(np.column_stack([px, py]))
        path = np.stack(pts, axis=1)  # trips x steps x 2
        return trips, path, f.speed(trips, t)

    def draw(self, trips, path, speed, trail: np.ndarray) -> np.ndarray:
        W, H = self.W, self.H
        head_u8 = np.zeros((H, W, 3), np.uint8)
        new_u8 = np.zeros((H, W, 3), np.uint8)
        ipath = np.round(path * SUB).astype(np.int32)
        if self.s.mode == "speed":
            bins = np.clip((speed / 2).astype(int), 0, 25)  # 2 km/h buckets
            keys = bins
            key_rgb = lambda k: speed_rgb(np.array(k * 2 + 1.0))  # noqa: E731
            head_rgb = lambda k: 0.45 * speed_rgb(np.array(k * 2 + 1.0)) + 0.55  # noqa: E731
        else:
            keys = self.route_color_id[self.fleet.trip_route[trips]]
            key_rgb = lambda k: self.color_table[k]  # noqa: E731
            head_rgb = lambda k: 0.8 * self.color_table[k] + 0.2  # noqa: E731
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        cuts = np.flatnonzero(np.diff(ks)) + 1
        for grp in np.split(order, cuts):
            if len(grp) == 0:
                continue
            k = keys[grp[0]]
            c = tuple(float(v) * 255 for v in key_rgb(k))
            cv2.polylines(new_u8, list(ipath[grp]), False, c, 1, cv2.LINE_AA, shift=4)
            heads = [np.repeat(ipath[g, -1:], 2, axis=0) for g in grp]
            cv2.polylines(head_u8, heads, False, tuple(float(v) * 255 for v in head_rgb(k)), 4, cv2.LINE_AA, shift=4)
        new = new_u8.astype(np.float32) / 255
        np.multiply(trail, self.decay, out=trail)
        np.maximum(trail, new, out=trail)
        head = head_u8.astype(np.float32) / 255

        small = cv2.resize(head + trail * 0.3, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.GaussianBlur(small, (0, 0), 1.5) * self.s.glow + cv2.GaussianBlur(small, (0, 0), 5) * self.s.glow
        glow = cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR)
        dyn = trail * self.s.trail_gain + glow + head
        dyn *= self.mask
        # soft shoulder instead of hard clipping, so dense areas bloom towards white,
        # then screen-blend over the basemap
        dyn = 1.0 - np.exp(-dyn * 1.2)
        return 1.0 - (1.0 - self.background) * (1.0 - dyn)

    def stats(self, trips, speed) -> dict:
        routes = self.fleet.trip_route[trips]
        counts = np.bincount(routes, minlength=len(self.route_names))
        top = np.argsort(-counts, kind="stable")[:5]
        spd = [int(np.count_nonzero((speed >= lo) & (speed < hi))) for lo, hi, _ in SPEED_BINS]
        moving = speed[speed > 0]
        return {
            "buses": len(trips),
            "lines": int(np.count_nonzero(counts)),
            "top": [(self.route_names[i], int(counts[i]), self.route_rgb[i]) for i in top if counts[i] > 0],
            "speed_bins": spd,
            "avg_speed": float(moving.mean()) if len(moving) else 0.0,
        }

    def render_frame(self, i: int, trail: np.ndarray) -> np.ndarray:
        t = self.s.frame_time(i)
        trips, path, speed = self.bus_state(t)
        img = self.draw(trips, path, speed, trail)
        img = self.overlay.compose(img, t, i, self.stats(trips, speed))
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# -- encoding ---------------------------------------------------------------
_SCENE: Scene | None = None


def _ffmpeg(path: str, s: Style) -> subprocess.Popen:
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{s.width}x{s.height}",
           "-r", str(s.fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _render_chunk(args):
    start, stop, path = args
    scene = _SCENE
    trail = np.zeros((scene.H, scene.W, 3), np.float32)
    warm = int(np.ceil(4 * scene.s.trail_sim_seconds / scene.dt))
    for i in range(max(0, start - warm), start):  # prime the trails
        scene.draw(*scene.bus_state(scene.s.frame_time(i)), trail)
    proc = _ffmpeg(path, scene.s)
    for i in range(start, stop):
        proc.stdin.write(scene.render_frame(i, trail).tobytes())
    proc.stdin.close()
    proc.wait()
    return path


def render_video(scene: Scene, out_path: str, workers: int = 4, frames: range | None = None) -> str:
    global _SCENE
    _SCENE = scene
    frames = frames or range(scene.s.n_frames)
    tmp = out_path + ".parts"
    os.makedirs(tmp, exist_ok=True)
    bounds = np.linspace(frames.start, frames.stop, workers + 1).astype(int)
    jobs = [(int(a), int(b), os.path.join(tmp, f"part{k:02d}.mp4")) for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])) if b > a]
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        parts = pool.map(_render_chunk, jobs)
    listing = os.path.join(tmp, "parts.txt")
    with open(listing, "w") as f:
        f.writelines(f"file '{os.path.abspath(p)}'\n" for p in parts)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", listing, "-c", "copy", "-movflags", "+faststart", out_path], check=True)
    for p in parts + [listing]:
        os.remove(p)
    os.rmdir(tmp)
    return out_path


def render_still(scene: Scene, t: float, out_path: str):
    """One frame at time t with fully developed trails, for quick look checks."""
    s = scene.s
    i = int(round((t - s.t_start) / scene.dt))
    trail = np.zeros((scene.H, scene.W, 3), np.float32)
    warm = int(np.ceil(4 * s.trail_sim_seconds / scene.dt))
    for j in range(max(0, i - warm), i):
        scene.draw(*scene.bus_state(s.frame_time(j)), trail)
    img = scene.render_frame(i, trail)
    cv2.imwrite(out_path, img[..., ::-1])
