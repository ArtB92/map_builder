"""Title, clock, legend, daily chart and progress bar drawn over the map."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from .render import Scene

WHITE = (240, 242, 246, 255)
MUTED = (150, 156, 168, 255)
DIM = (96, 102, 114, 255)
TRACK = (38, 41, 50, 255)


def spaced(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, spacing: float, anchor_right=False):
    """Text with letter spacing (PIL has none built in)."""
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + spacing * (len(text) - 1)
    x, y = xy
    if anchor_right:
        x -= total
    for ch, w in zip(text, widths):
        draw.text((x, y), ch, font=font, fill=fill)
        x += w + spacing
    return total


class Overlay:
    def __init__(self, scene: "Scene", font_dir: str):
        self.scene = scene
        s = scene.s
        f = lambda name, size: ImageFont.truetype(os.path.join(font_dir, name), size)  # noqa: E731
        self.f_title = f("PlayfairDisplay[wght].ttf", 68)
        self.f_title.set_variation_by_name("Bold")
        self.f_sub = f("Lato-Regular.ttf", 17)
        self.f_clock = f("Lato-Light.ttf", 118)
        self.f_stat = f("Lato-Regular.ttf", 22)
        self.f_legend = f("Lato-Regular.ttf", 19)
        self.f_small = f("Lato-Regular.ttf", 14)
        self.f_label = f("Lato-Bold.ttf", 24)
        self.f_head = f("Lato-Bold.ttf", 13)
        self.W, self.H = scene.W, scene.H
        self.chart = (70, 868, 520, 958)  # x0, y0, x1, y1
        self.static = self._static()

    def _static(self) -> Image.Image:
        s = self.scene.s
        im = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        # card: black outside, thin rounded border
        outside = Image.new("L", (self.W, self.H), 255)
        ImageDraw.Draw(outside).rounded_rectangle((14, 14, self.W - 15, self.H - 15), radius=30, fill=0)
        im.paste((0, 0, 0, 255), (0, 0), outside)
        d.rounded_rectangle((14, 14, self.W - 15, self.H - 15), radius=30, outline=(60, 64, 74, 255), width=2)

        spaced(d, (66, 44), s.title, self.f_title, WHITE, 14)
        spaced(d, (70, 132), s.subtitle, self.f_sub, MUTED, 4.2)

        for name, lon, lat, (dx, dy) in s.labels:
            px, py = (float(v) for v in self.scene.lonlat_px(lon, lat))
            lx, ly = px + dx, py + dy
            d.line((px, py, lx, ly), fill=(200, 205, 215, 150), width=1)
            d.ellipse((px - 5, py - 5, px + 5, py + 5), outline=(235, 238, 245, 220), width=2)
            w = d.textlength(name, font=self.f_label)
            tx = lx - w - 6 if dx < 0 else lx + 6
            d.text((tx, ly - 15), name, font=self.f_label, fill=(245, 246, 250, 255), stroke_width=3, stroke_fill=(8, 9, 12, 200))

        x0, y0, x1, y1 = self.chart
        spaced(d, (x0, y1 + 16), "BUSES IN SERVICE", self.f_sub, MUTED, 4.2)
        credit = "Île-de-France Mobilités GTFS  ·  France GeoJSON (IGN / INSEE)"
        if s.date_label:
            credit = f"Timetable of {s.date_label}  ·  " + credit
        d.text((self.W - 64, 1000), credit, font=self.f_small, fill=DIM, anchor="ra")
        return im

    def _chart(self, d: ImageDraw.ImageDraw, frac: float):
        x0, y0, x1, y1 = self.chart
        c = self.scene.daily.astype(float)
        c = c / max(c.max(), 1)
        xs = np.linspace(x0, x1, len(c))
        ys = y1 - c * (y1 - y0)
        cut = x0 + frac * (x1 - x0)
        pts = list(zip(xs, ys))
        d.polygon([(x0, y1)] + pts + [(x1, y1)], fill=(34, 38, 48, 255))
        past = [(x, y) for x, y in pts if x <= cut]
        if past:
            yc = float(np.interp(cut, xs, ys))
            d.polygon([(x0, y1)] + past + [(cut, yc), (cut, y1)], fill=(92, 100, 120, 255))
        d.line((x0, y1, x1, y1), fill=(70, 75, 88, 255), width=1)
        d.line((cut, y0 - 8, cut, y1), fill=(235, 238, 245, 255), width=2)

    def compose(self, img: np.ndarray, t: float, i: int, st: dict) -> np.ndarray:
        s = self.scene.s
        im = self.static.copy()
        d = ImageDraw.Draw(im)
        hh, mm = int(t // 3600) % 24, int(t % 3600 // 60)
        d.text((62, 150), f"{hh:02d}:{mm:02d}", font=self.f_clock, fill=WHITE)
        spaced(d, (70, 300), f"{st['buses']:,} buses in service".replace(",", " "), self.f_stat, (225, 228, 235, 255), 1.6)
        spaced(d, (70, 336), f"{st['lines']} bus lines running", self.f_stat, (225, 228, 235, 255), 1.6)
        if s.mode == "speed":
            spaced(d, (70, 372), f"average speed {st['avg_speed']:.1f} km/h", self.f_stat, (225, 228, 235, 255), 1.6)

        # legend
        y = 640
        from .render import SPEED_BINS, speed_rgb

        if s.mode == "speed":
            spaced(d, (70, y - 40), "BUS SPEED", self.f_head, MUTED, 3.5)
            rows = [(label, n, speed_rgb(np.array((lo + min(hi, 40)) / 2))) for (lo, hi, label), n in zip(SPEED_BINS, st["speed_bins"])]
            scale = max(max(st["speed_bins"]), 1)
        else:
            spaced(d, (70, y - 40), "BUSIEST LINES RIGHT NOW", self.f_head, MUTED, 3.5)
            rows = [(f"Bus {name}", n, rgb) for name, n, rgb in st["top"]]
            scale = max([n for _, n, _ in rows] + [1])
        for label, n, rgb in rows:
            col = tuple(int(v * 255) for v in rgb) + (255,)
            d.ellipse((72, y + 5, 84, y + 17), fill=col)
            d.text((96, y), label, font=self.f_legend, fill=(230, 232, 238, 255))
            d.text((330, y), f"{n}", font=self.f_legend, fill=(230, 232, 238, 255), anchor="ra")
            d.rectangle((72, y + 28, 330, y + 32), fill=TRACK)
            d.rectangle((72, y + 28, 72 + 258 * n / scale, y + 32), fill=col)
            y += 44

        frac = i / max(s.n_frames - 1, 1)
        self._chart(d, frac)
        # progress bar
        bx0, bx1, by = 64, self.W - 64, 1040
        d.line((bx0, by, bx1, by), fill=(60, 64, 74, 255), width=3)
        bx = bx0 + frac * (bx1 - bx0)
        d.line((bx0, by, bx, by), fill=(170, 175, 185, 255), width=3)
        d.ellipse((bx - 7, by - 7, bx + 7, by + 7), fill=WHITE)

        ov = np.asarray(im, dtype=np.float32) / 255
        a = ov[..., 3:4]
        return img * (1 - a) + ov[..., :3] * a
