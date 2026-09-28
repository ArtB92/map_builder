/* Paris by Bus: every scheduled bus of one weekday, replayed on a map.
 *
 * fleet.bin.gz holds, per trip, a start time and a timing profile: (time, distance) knots
 * along one route pattern polyline. Positions are interpolated here, every frame, for the
 * buses running at the current time.
 */
(() => {
  "use strict";

  const SPEED = 900; // simulated seconds per real second: the day plays in 96 s
  const START_AT = 7.5 * 3600;
  const CORE_VIEW = [[2.22, 48.775], [2.50, 48.945]]; // Paris and the petite couronne
  const FIT_PADDING = () =>
    innerWidth > 760 ? { top: 110, bottom: 150, left: 60, right: 400 } : { top: 200, bottom: 190, left: 20, right: 20 };

  const $ = (id) => document.getElementById(id);
  const TYPES = { uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array, int32: Int32Array, float32: Float32Array };

  // ---------------------------------------------------------------- loading
  async function fetchWithProgress(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: ${res.status}`);
    const total = +res.headers.get("content-length") || 0;
    if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      onProgress(Math.min(got / total, 1));
    }
    const out = new Uint8Array(got);
    let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.length; }
    return out;
  }

  async function gunzip(bytes) {
    // Some servers already decode .gz with Content-Encoding; only inflate real gzip data.
    if (!(bytes[0] === 0x1f && bytes[1] === 0x8b)) return bytes.buffer;
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Response(stream).arrayBuffer();
  }

  async function loadFleet() {
    const stopsReq = fetch("fleet/stops.json").then((r) => r.json());
    const meta = await (await fetch("fleet/fleet.json")).json();
    const gz = await fetchWithProgress("fleet/fleet.bin.gz", (p) => ($("progress").style.width = `${p * 90}%`));
    $("loading-text").textContent = "Preparing 90,000 trips";
    const buf = await gunzip(gz);
    const A = {};
    for (const a of meta.arrays) A[a.name] = new TYPES[a.dtype](buf, a.offset, a.length);
    const F = prepare(meta, A);
    F.stops = prepareStops(await stopsReq, F.lines);
    return F;
  }

  function prepareStops(S, lines) {
    const n = S.name.length;
    const networks = S.lines.map((ls) => [...new Set(ls.map((i) => lines[i].network).filter(Boolean))]);
    return { n, name: S.name, lonlat: Float32Array.from(S.lonlat), lines: S.lines, networks, folded: S.name.map(fold) };
  }

  function prepare(meta, A) {
    const K = A.prof_dt.length;
    const PT = new Float64Array(K); // knot time since trip start (s)
    const PD = new Float64Array(K); // knot distance along the pattern (m)
    const Q = A.prof_k0.length - 1;
    for (let q = 0; q < Q; q++) {
      let t = 0, d = A.prof_d0[q];
      for (let k = A.prof_k0[q]; k < A.prof_k0[q + 1]; k++) {
        t += A.prof_dt[k];
        d += A.prof_dd[k];
        PT[k] = t;
        PD[k] = d;
      }
    }
    const N = A.trip_start.length;
    const tripEnd = new Float64Array(N);
    let maxDur = 0;
    for (let i = 0; i < N; i++) {
      const dur = PT[A.prof_k0[A.trip_prof[i] + 1] - 1];
      tripEnd[i] = A.trip_start[i] + dur;
      if (dur > maxDur) maxDur = dur;
    }
    const lines = meta.lines;
    const rgb = (hex) => [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)];
    const glow = new Uint8Array(lines.length * 3);
    lines.forEach((l, i) => glow.set(rgb(l.glow), i * 3));

    // bounding box of each line, to frame a selection
    const P = A.pat_route.length;
    const lineBox = lines.map(() => [180, 90, -180, -90]);
    for (let p = 0; p < P; p++) {
      const b = lineBox[A.pat_route[p]];
      for (let v = A.pat_v0[p]; v < A.pat_v0[p + 1]; v++) {
        const x = A.pat_lonlat[2 * v], y = A.pat_lonlat[2 * v + 1];
        if (x < b[0]) b[0] = x;
        if (y < b[1]) b[1] = y;
        if (x > b[2]) b[2] = x;
        if (y > b[3]) b[3] = y;
      }
    }
    return { meta, A, PT, PD, N, P, tripEnd, maxDur, lines, glow, lineBox };
  }

  // ---------------------------------------------------------------- positions
  function makeKinematics(F) {
    const { A, PT, PD } = F;
    const K0 = A.prof_k0, V0 = A.pat_v0, VD = A.pat_dist, LL = A.pat_lonlat;

    // distance along the pattern after `tt` seconds of profile q
    function distAt(q, tt) {
      let lo = K0[q], hi = K0[q + 1] - 1;
      if (tt <= PT[lo]) return PD[lo];
      if (tt >= PT[hi]) return PD[hi];
      while (hi - lo > 1) {
        const m = (lo + hi) >> 1;
        if (PT[m] <= tt) lo = m; else hi = m;
      }
      const t0 = PT[lo], t1 = PT[lo + 1];
      return PD[lo] + ((tt - t0) / (t1 - t0)) * (PD[lo + 1] - PD[lo]);
    }

    // inverse of distAt: when the bus of profile q passed distance d (latest knot time on ties)
    function timeAt(q, d) {
      let lo = K0[q], hi = K0[q + 1] - 1;
      if (d <= PD[lo]) return PT[lo];
      if (d >= PD[hi]) return PT[hi];
      while (hi - lo > 1) {
        const m = (lo + hi) >> 1;
        if (PD[m] <= d) lo = m; else hi = m;
      }
      const d0 = PD[lo], d1 = PD[lo + 1];
      return PT[lo] + ((d - d0) / (d1 - d0)) * (PT[lo + 1] - PT[lo]);
    }

    // first vertex index of pattern p whose distance is > d
    function vertexAfter(p, d) {
      let lo = V0[p], hi = V0[p + 1];
      while (lo < hi) {
        const m = (lo + hi) >> 1;
        if (VD[m] <= d) lo = m + 1; else hi = m;
      }
      return lo;
    }

    function pointAt(p, d, out, o) {
      const v0 = V0[p], v1 = V0[p + 1] - 1;
      let j = vertexAfter(p, d);
      if (j <= v0) { out[o] = LL[2 * v0]; out[o + 1] = LL[2 * v0 + 1]; return; }
      if (j > v1) { out[o] = LL[2 * v1]; out[o + 1] = LL[2 * v1 + 1]; return; }
      const a = j - 1, span = VD[j] - VD[a];
      const f = span > 0 ? (d - VD[a]) / span : 0;
      out[o] = LL[2 * a] + f * (LL[2 * j] - LL[2 * a]);
      out[o + 1] = LL[2 * a + 1] + f * (LL[2 * j + 1] - LL[2 * a + 1]);
    }

    return { distAt, timeAt, vertexAfter, pointAt };
  }

  // Growable typed buffers reused frame to frame.
  class Buf {
    constructor(Type, n) { this.Type = Type; this.a = new Type(n); }
    ensure(n) {
      if (n > this.a.length) {
        const b = new this.Type(Math.max(n, this.a.length * 1.5) | 0);
        b.set(this.a);
        this.a = b;
      }
      return this.a;
    }
  }

  function makeFrameBuilder(F, K) {
    const { A, tripEnd, maxDur, glow } = F;
    const trail = F.meta.trail;
    const TB = F.meta.t_start - 3600; // timestamps are stored relative to this, for float32 precision
    const start = A.trip_start, prof = A.trip_prof, route = A.trip_route, profPat = A.prof_pat;
    const LL = A.pat_lonlat, VD = A.pat_dist;

    const heads = { pos: new Buf(Float32Array, 16384), col: new Buf(Uint8Array, 32768), route: new Buf(Uint16Array, 8192) };
    const tr = { pos: new Buf(Float32Array, 262144), time: new Buf(Float32Array, 131072), col: new Buf(Uint8Array, 524288), starts: new Buf(Uint32Array, 8192) };

    function lowerBound(arr, x) {
      let lo = 0, hi = arr.length;
      while (lo < hi) {
        const m = (lo + hi) >> 1;
        if (arr[m] < x) lo = m + 1; else hi = m;
      }
      return lo;
    }

    return function build(T, visible) {
      const i0 = lowerBound(start, T - trail - maxDur);
      const i1 = lowerBound(start, T + 1);
      let nh = 0, np = 0, nv = 0;
      for (let i = i0; i < i1; i++) {
        const end = tripEnd[i];
        if (end <= T - trail) continue;
        const r = route[i];
        if (!visible[r]) continue;
        const q = prof[i], p = profPat[q], s = start[i];
        const tHead = Math.min(T, end) - s;
        const tTail = Math.max(T - trail, s) - s;
        const dHead = K.distAt(q, tHead);
        const cr = glow[3 * r], cg = glow[3 * r + 1], cb = glow[3 * r + 2];

        if (T < end) {
          const hp = heads.pos.ensure(2 * nh + 2), hc = heads.col.ensure(4 * nh + 4);
          K.pointAt(p, dHead, hp, 2 * nh);
          hc[4 * nh] = cr; hc[4 * nh + 1] = cg; hc[4 * nh + 2] = cb; hc[4 * nh + 3] = 255;
          heads.route.ensure(nh + 1)[nh] = r;
          nh++;
        }

        // trail: the tail point, the pattern vertices in between, then the head
        const dTail = K.distAt(q, tTail);
        if (dHead - dTail < 1) continue;
        const va = K.vertexAfter(p, dTail), vb = K.vertexAfter(p, dHead);
        const n = 2 + Math.max(0, vb - va);
        const pos = tr.pos.ensure(2 * (nv + n)), tim = tr.time.ensure(nv + n), col = tr.col.ensure(4 * (nv + n));
        tr.starts.ensure(np + 2)[np] = nv;
        K.pointAt(p, dTail, pos, 2 * nv);
        tim[nv] = s + tTail - TB;
        nv++;
        for (let v = va; v < vb; v++) {
          if (VD[v] <= dTail || VD[v] >= dHead) continue;
          pos[2 * nv] = LL[2 * v];
          pos[2 * nv + 1] = LL[2 * v + 1];
          tim[nv] = s + K.timeAt(q, VD[v]) - TB;
          nv++;
        }
        K.pointAt(p, dHead, pos, 2 * nv);
        tim[nv] = s + tHead - TB;
        nv++;
        for (let k = tr.starts.a[np]; k < nv; k++) {
          col[4 * k] = cr; col[4 * k + 1] = cg; col[4 * k + 2] = cb; col[4 * k + 3] = 255;
        }
        np++;
      }
      return {
        heads: {
          length: nh,
          attributes: {
            getPosition: { value: heads.pos.a.slice(0, 2 * nh), size: 2 },
            getFillColor: { value: heads.col.a.slice(0, 4 * nh), size: 4, normalized: true },
          },
        },
        headRoute: heads.route.a.slice(0, nh),
        trails: {
          length: np,
          startIndices: tr.starts.a.slice(0, np),
          attributes: {
            getPath: { value: tr.pos.a.slice(0, 2 * nv), size: 2 },
            getTimestamps: { value: tr.time.a.slice(0, nv), size: 1 },
            getColor: { value: tr.col.a.slice(0, 4 * nv), size: 4, normalized: true },
          },
        },
        currentTime: T - TB,
        count: nh,
      };
    };
  }

  // ---------------------------------------------------------------- UI helpers
  const pad = (n) => String(n).padStart(2, "0");
  const clock = (t) => `${pad(Math.floor(t / 3600) % 24)}:${pad(Math.floor(t / 60) % 60)}`;
  const fold = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  function badge(line, cls = "") {
    return `<span class="badge ${cls}" style="background:#${line.color};color:#${line.text}">${esc(line.name)}</span>`;
  }

  // ---------------------------------------------------------------- app
  async function main() {
    const map = new maplibregl.Map({
      container: "map",
      style: baseStyle(),
      bounds: CORE_VIEW,
      fitBoundsOptions: { padding: 20 },
      minZoom: 8,
      maxZoom: 17.5,
      dragRotate: false,
      pitchWithRotate: false,
      touchPitch: false,
      attributionControl: false,
    });
    // not "load": it waits for every basemap tile, and never fires if the tile server is unreachable
    const styleReady = new Promise((ok) => map.once("style.load", ok));
    map.touchZoomRotate.disableRotation();
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
    map.addControl(
      new maplibregl.AttributionControl({
        compact: true,
        customAttribution:
          'Timetable <a href="https://data.iledefrance-mobilites.fr/explore/dataset/offre-horaires-tc-gtfs-idfm/" target="_blank" rel="noopener">Île-de-France Mobilités</a>',
      }),
      "bottom-right",
    );

    let F;
    try {
      F = await loadFleet();
    } catch (err) {
      $("loading-text").textContent = `Could not load the timetable (${err.message})`;
      throw err;
    }
    $("progress").style.width = "100%";
    const { meta, lines } = F;
    const date = new Date(`${meta.date}T12:00:00`);
    $("subtitle").textContent = `Every bus of Île-de-France Mobilités on ${date.toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric" })}`;

    // keep the camera over the area the network covers
    const [x0, y0, x1, y1] = meta.bounds;
    map.setMaxBounds([[x0 - 0.12, y0 - 0.08], [x1 + 0.12, y1 + 0.08]]);

    const K = makeKinematics(F);
    const build = makeFrameBuilder(F, K);

    // ------------------------------------------------ state
    const state = {
      t: START_AT,
      target: START_AT,
      playing: false,
      selected: new Set(), // lines picked in the line filter
      stopSel: new Set(), // stops picked in the stop filter
      showStops: false,
      filtered: false,
      visible: new Uint8Array(lines.length).fill(1),
      frame: null,
      dirty: true,
      hover: -1,
    };

    // ------------------------------------------------ deck layers
    const ADDITIVE = {
      blend: true,
      blendColorOperation: "add",
      blendColorSrcFactor: "src-alpha",
      blendColorDstFactor: "one",
      blendAlphaOperation: "add",
      blendAlphaSrcFactor: "one",
      blendAlphaDstFactor: "one-minus-src-alpha",
      depthWriteEnabled: false,
      depthCompare: "always",
    };
    const patterns = Array.from({ length: F.P }, (_, i) => i);
    const { A } = F;
    const networkColor = (p) => {
      const r = A.pat_route[p];
      const c = F.glow.subarray(3 * r, 3 * r + 3);
      const a = !state.filtered ? 30 : state.visible[r] ? 130 : 5;
      return [c[0], c[1], c[2], a];
    };

    const overlay = new deck.MapboxOverlay({
      interleaved: true,
      layers: [],
      getTooltip: tooltip,
      getCursor: ({ isHovering }) => (isHovering ? "pointer" : "grab"),
      onClick: (info) => {
        if (info.layer && (info.layer.id === "stops" || info.layer.id === "picked-stops")) toggleStop(info.object);
      },
    });
    await styleReady;
    map.addControl(overlay);

    const ST = F.stops;
    const allStops = Array.from({ length: ST.n }, (_, i) => i);
    let stopCache = { version: -1, data: allStops };
    function stopData() {
      if (stopCache.version !== state.selectionVersion) {
        const data = state.filtered ? allStops.filter((i) => ST.lines[i].some((r) => state.visible[r])) : allStops;
        stopCache = { version: state.selectionVersion, data };
      }
      return stopCache.data;
    }

    function zoomScale() {
      const z = map.getZoom();
      return Math.min(Math.max(1 + (z - 10.5) * 0.45, 0.7), 3.2);
    }

    function render() {
      if (state.dirty || !state.frame) {
        state.frame = build(state.t, state.visible);
        state.dirty = false;
      }
      const fr = state.frame;
      const s = zoomScale();
      const filtered = state.filtered;
      overlay.setProps({
        layers: [
          new deck.PathLayer({
            id: "network",
            data: patterns,
            beforeId: "labels",
            getPath: (p) => A.pat_lonlat.subarray(2 * A.pat_v0[p], 2 * A.pat_v0[p + 1]),
            positionFormat: "XY",
            _pathType: "open",
            getColor: networkColor,
            getWidth: (p) => (filtered && state.visible[A.pat_route[p]] ? 2 : 1),
            widthUnits: "pixels",
            widthScale: filtered ? Math.max(s, 1) : 1,
            pickable: true,
            autoHighlight: true,
            highlightColor: [255, 255, 255, 90],
            updateTriggers: { getColor: [state.selectionVersion], getWidth: [state.selectionVersion] },
            parameters: ADDITIVE,
          }),
          new deck.ScatterplotLayer({
            id: "stops",
            data: stopData(),
            beforeId: "labels",
            visible: state.showStops,
            getPosition: (i) => [ST.lonlat[2 * i], ST.lonlat[2 * i + 1]],
            radiusUnits: "pixels",
            getRadius: 1,
            radiusScale: Math.max(1.6, 2.1 * s),
            stroked: true,
            filled: true,
            getFillColor: [12, 13, 18, 255],
            getLineColor: [230, 232, 238, 170],
            lineWidthUnits: "pixels",
            getLineWidth: 1,
            lineWidthScale: Math.max(0.8, 0.7 * s),
            pickable: true,
            autoHighlight: true,
            highlightColor: [255, 255, 255, 255],
          }),
          new deck.TripsLayer({
            id: "trails",
            data: fr.trails,
            beforeId: "labels",
            _pathType: "open",
            positionFormat: "XY",
            currentTime: fr.currentTime,
            trailLength: meta.trail,
            fadeTrail: true,
            widthUnits: "pixels",
            getWidth: 1,
            widthScale: 1.6 * s,
            opacity: 0.9,
            capRounded: true,
            jointRounded: true,
            parameters: ADDITIVE,
          }),
          new deck.ScatterplotLayer({
            id: "halo",
            data: fr.heads,
            beforeId: "labels",
            radiusUnits: "pixels",
            getRadius: 1,
            radiusScale: 4 * s,
            opacity: 0.12,
            stroked: false,
            parameters: ADDITIVE,
          }),
          new deck.ScatterplotLayer({
            id: "heads",
            data: fr.heads,
            beforeId: "labels",
            radiusUnits: "pixels",
            getRadius: 1,
            radiusScale: 1.4 * s,
            opacity: 1,
            stroked: false,
            pickable: true,
            parameters: ADDITIVE,
          }),
          new deck.ScatterplotLayer({
            id: "picked-stops",
            data: [...state.stopSel],
            beforeId: "labels",
            getPosition: (i) => [ST.lonlat[2 * i], ST.lonlat[2 * i + 1]],
            radiusUnits: "pixels",
            getRadius: 1,
            radiusScale: 6 + 1.5 * s,
            stroked: true,
            filled: true,
            getFillColor: [12, 13, 18, 230],
            getLineColor: [255, 255, 255, 255],
            lineWidthUnits: "pixels",
            getLineWidth: 2.5,
            pickable: true,
          }),
        ],
      });
      $("count").textContent = fr.count.toLocaleString("en-GB");
      $("clock").textContent = clock(state.t);
    }

    const TIP_STYLE = { background: "rgba(16,18,26,.94)", color: "#eef0f4", borderRadius: "8px", padding: "8px 10px", fontSize: "13px", border: "1px solid rgba(255,255,255,.09)" };
    function tooltip(info) {
      if (!info.layer || info.index < 0) return null;
      if (info.layer.id === "stops" || info.layer.id === "picked-stops") {
        const i = info.object;
        const ls = ST.lines[i].filter((r) => !state.filtered || state.visible[r] || state.stopSel.has(i));
        return {
          html: `<div class="tip-stop"><b>${esc(ST.name[i])}</b>${lineBadges(ls, 12)}<small>${state.stopSel.has(i) ? "Click to remove from the filter" : "Click to show only the lines stopping here"}</small></div>`,
          style: TIP_STYLE,
        };
      }
      let r;
      if (info.layer.id === "heads") r = state.frame && state.frame.headRoute[info.index];
      else if (info.layer.id === "network") r = A.pat_route[info.object];
      if (r === undefined || r === null) return null;
      const l = lines[r];
      return {
        html: `<div class="tip">${badge(l)}<div>${esc(l.long || `Bus ${l.name}`)}<small>${esc(l.network)}</small></div></div>`,
        style: TIP_STYLE,
      };
    }

    // ------------------------------------------------ timeline
    const range = $("time");
    range.min = meta.t_start;
    range.max = meta.t_end;
    const setRange = (t) => {
      range.value = Math.round(t);
      range.style.setProperty("--p", `${((t - meta.t_start) / (meta.t_end - meta.t_start)) * 100}%`);
    };
    const ticks = $("ticks");
    for (let h = meta.t_start; h <= meta.t_end; h += 4 * 3600) {
      const s = document.createElement("span");
      s.style.left = `${((h - meta.t_start) / (meta.t_end - meta.t_start)) * 100}%`;
      s.textContent = clock(h);
      ticks.appendChild(s);
    }
    let scrubbing = false;
    range.addEventListener("input", () => {
      state.target = +range.value;
      scrubbing = true;
      setRange(state.target);
    });
    range.addEventListener("change", () => (scrubbing = false));

    const canvas = $("activity");
    let activity = meta.activity.counts;
    function drawActivity() {
      const dpr = devicePixelRatio || 1;
      const w = canvas.clientWidth, h = canvas.clientHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      const g = canvas.getContext("2d");
      g.scale(dpr, dpr);
      const max = Math.max(...meta.activity.counts, 1);
      const scaleMax = state.filtered ? Math.max(...activity, 1) : max;
      g.beginPath();
      g.moveTo(0, h - 12);
      activity.forEach((c, i) => g.lineTo((i / (activity.length - 1)) * w, h - 12 - (c / scaleMax) * (h - 16)));
      g.lineTo(w, h - 12);
      g.closePath();
      const grad = g.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, "rgba(255,255,255,0.22)");
      grad.addColorStop(1, "rgba(255,255,255,0.03)");
      g.fillStyle = grad;
      g.fill();
    }
    function computeActivity() {
      if (!state.filtered) return (activity = meta.activity.counts);
      const step = meta.activity.step, n = meta.activity.counts.length;
      const diff = new Int32Array(n + 1);
      for (let i = 0; i < F.N; i++) {
        if (!state.visible[A.trip_route[i]]) continue;
        const a = Math.max(0, Math.ceil((A.trip_start[i] - meta.t_start) / step));
        const b = Math.min(n, Math.ceil((F.tripEnd[i] - meta.t_start) / step));
        if (b > a) { diff[a]++; diff[b]--; }
      }
      const out = [];
      let c = 0;
      for (let i = 0; i < n; i++) out.push((c += diff[i]));
      activity = out;
    }
    new ResizeObserver(drawActivity).observe(canvas);

    // ------------------------------------------------ play / pause
    const player = document.querySelector(".player");
    function setPlaying(on) {
      state.playing = on;
      player.classList.toggle("playing", on);
      $("play").setAttribute("aria-label", on ? "Pause" : "Play");
      if (on) state.target = state.t;
    }
    $("play").addEventListener("click", () => setPlaying(!state.playing));
    addEventListener("keydown", (e) => {
      if (e.code !== "Space" || e.target.closest("input, button")) return;
      e.preventDefault();
      setPlaying(!state.playing);
    });

    // ------------------------------------------------ line filter
    const filter = $("filter"), dropdown = $("dropdown"), options = $("options"), search = $("search");
    const byNetwork = new Map();
    for (const i of meta.line_order) {
      const net = lines[i].network || "Other";
      if (!byNetwork.has(net)) byNetwork.set(net, []);
      byNetwork.get(net).push(i);
    }
    const networks = [...byNetwork.keys()].sort((a, b) => (a === "RATP" ? -1 : b === "RATP" ? 1 : a.localeCompare(b, "fr")));
    const CHECK = '<svg viewBox="0 0 12 12"><path d="m2.5 6.2 2.3 2.3 4.7-5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    let html = "";
    for (const net of networks) {
      const ids = byNetwork.get(net);
      html += `<button class="group" data-network="${esc(net)}" title="Select every line of ${esc(net)}"><span>${esc(net)}</span><span class="n">${ids.length}</span></button>`;
      for (const i of ids) {
        const l = lines[i];
        html += `<button class="option" role="option" aria-selected="false" data-line="${i}" data-network="${esc(net)}" data-search="${esc(fold(`${l.name} ${l.long} ${net}`))}">` +
          `<span class="check">${CHECK}</span>${badge(l)}<span class="meta">${esc(l.long)}</span><span class="trips">${l.trips.toLocaleString("en-GB")} trips</span></button>`;
      }
    }
    options.innerHTML = html;
    const optionEls = [...options.querySelectorAll(".option")];
    const groupEls = [...options.querySelectorAll(".group")];

    const panels = [
      { root: filter, dropdown, toggle: $("filter-toggle"), search },
      { root: $("stop-filter"), dropdown: $("stop-dropdown"), toggle: $("stop-toggle"), search: $("stop-search") },
    ];
    function openDropdown(panel, open) {
      for (const p of panels) {
        const on = p === panel && open;
        p.dropdown.hidden = !on;
        p.root.classList.toggle("open", on);
        p.toggle.setAttribute("aria-expanded", on);
      }
      if (open && innerWidth > 760) panel.search.focus();
    }
    for (const p of panels) p.toggle.addEventListener("click", () => openDropdown(p, p.dropdown.hidden));
    document.addEventListener("pointerdown", (e) => {
      for (const p of panels) if (!p.dropdown.hidden && !p.root.contains(e.target)) openDropdown(p, false);
    });
    addEventListener("keydown", (e) => {
      if (e.key === "Escape") for (const p of panels) if (!p.dropdown.hidden) openDropdown(p, false);
    });

    search.addEventListener("input", () => {
      const q = fold(search.value.trim());
      const shown = new Set();
      for (const el of optionEls) {
        const hit = !q || el.dataset.search.includes(q);
        el.hidden = !hit;
        if (hit) shown.add(el.dataset.network);
      }
      for (const g of groupEls) g.hidden = !shown.has(g.dataset.network);
      let empty = options.querySelector(".empty");
      if (!shown.size && !empty) {
        empty = document.createElement("div");
        empty.className = "empty";
        options.appendChild(empty);
      }
      if (empty) {
        empty.hidden = shown.size > 0;
        empty.textContent = `No line matches “${search.value.trim()}”`;
      }
    });

    options.addEventListener("click", (e) => {
      const opt = e.target.closest(".option");
      const grp = e.target.closest(".group");
      if (opt) {
        const i = +opt.dataset.line;
        state.selected.has(i) ? state.selected.delete(i) : state.selected.add(i);
      } else if (grp) {
        const ids = optionEls.filter((el) => el.dataset.network === grp.dataset.network && !el.hidden).map((el) => +el.dataset.line);
        const all = ids.every((i) => state.selected.has(i));
        for (const i of ids) all ? state.selected.delete(i) : state.selected.add(i);
      } else return;
      selectionChanged();
    });
    $("select-all").addEventListener("click", () => { state.selected.clear(); selectionChanged(); });

    state.selectionVersion = 0;
    function selectionChanged() {
      state.selectionVersion++;
      const sel = state.selected;
      // a line shows when it passes both filters: picked (or no line picked) and serving a picked stop (or no stop picked)
      const viaStops = new Uint8Array(lines.length).fill(state.stopSel.size ? 0 : 1);
      for (const i of state.stopSel) for (const r of ST.lines[i]) viaStops[r] = 1;
      for (let r = 0; r < lines.length; r++) state.visible[r] = (!sel.size || sel.has(r)) && viaStops[r] ? 1 : 0;
      state.filtered = sel.size > 0 || state.stopSel.size > 0;
      for (const el of optionEls) el.setAttribute("aria-selected", sel.has(+el.dataset.line));
      $("selection-count").textContent = sel.size ? `${sel.size} selected` : "";
      const v = $("filter-value");
      if (!sel.size) v.textContent = "All lines";
      else {
        const ids = [...sel].sort((a, b) => meta.line_order.indexOf(a) - meta.line_order.indexOf(b));
        v.innerHTML = ids.slice(0, 6).map((i) => badge(lines[i], "small")).join("") + (ids.length > 6 ? `<span class="more">+${ids.length - 6}</span>` : "");
      }
      computeActivity();
      drawActivity();
      state.dirty = true;
      if (state.filtered) {
        const b = [180, 90, -180, -90];
        for (let i = 0; i < lines.length; i++) {
          if (!state.visible[i]) continue;
          const lb = F.lineBox[i];
          b[0] = Math.min(b[0], lb[0]); b[1] = Math.min(b[1], lb[1]);
          b[2] = Math.max(b[2], lb[2]); b[3] = Math.max(b[3], lb[3]);
        }
        if (b[0] < b[2]) map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: FIT_PADDING(), maxZoom: 14.5, duration: 900 });
      }
    }

    // ------------------------------------------------ stop filter
    const stopOptions = $("stop-options"), stopSearch = $("stop-search");
    const MAX_RESULTS = 80;
    function lineBadges(ids, max) {
      const sorted = [...ids].sort((a, b) => lineRank[a] - lineRank[b]);
      return `<span class="badges">${sorted.slice(0, max).map((r) => badge(lines[r], "small")).join("")}${sorted.length > max ? `<span class="more">+${sorted.length - max}</span>` : ""}</span>`;
    }
    const lineRank = new Int32Array(lines.length);
    meta.line_order.forEach((r, k) => (lineRank[r] = k));
    function stopRow(i) {
      return `<button class="option stop-row" role="option" aria-selected="${state.stopSel.has(i)}" data-stop="${i}">` +
        `<span class="check">${CHECK}</span><span class="stop-name"><b>${esc(ST.name[i])}</b><small>${esc(ST.networks[i].slice(0, 2).join(" · "))}</small></span>${lineBadges(ST.lines[i], 4)}</button>`;
    }
    function renderStopOptions() {
      const q = fold(stopSearch.value.trim());
      let html = "";
      const picked = [...state.stopSel];
      if (picked.length) html += `<div class="group">Selected</div>${picked.map(stopRow).join("")}`;
      if (!q) {
        html += `<div class="empty">Type a stop name, or click a stop on the map.</div>`;
      } else {
        const starts = [], contains = [];
        for (let i = 0; i < ST.n && starts.length < MAX_RESULTS; i++) {
          if (state.stopSel.has(i)) continue;
          const f = ST.folded[i];
          if (f.startsWith(q)) starts.push(i);
          else if (contains.length < MAX_RESULTS && f.includes(q)) contains.push(i);
        }
        const hits = starts.concat(contains).slice(0, MAX_RESULTS);
        html += hits.length
          ? `<div class="group">Stops</div>${hits.map(stopRow).join("")}`
          : `<div class="empty">No stop matches “${esc(stopSearch.value.trim())}”</div>`;
      }
      stopOptions.innerHTML = html;
    }
    stopSearch.addEventListener("input", renderStopOptions);
    stopOptions.addEventListener("click", (e) => {
      const opt = e.target.closest(".option");
      if (opt) toggleStop(+opt.dataset.stop);
    });
    $("stop-clear").addEventListener("click", () => {
      state.stopSel.clear();
      stopsChanged();
    });
    function toggleStop(i) {
      state.stopSel.has(i) ? state.stopSel.delete(i) : state.stopSel.add(i);
      stopsChanged();
    }
    function stopsChanged() {
      const n = state.stopSel.size;
      const v = $("stop-value");
      if (!n) v.textContent = "Any stop";
      else {
        const names = [...state.stopSel].map((i) => ST.name[i]);
        v.innerHTML = `<span class="chip">${esc(names[0])}</span>` + (n > 1 ? `<span class="more">+${n - 1}</span>` : "");
      }
      $("stop-count").textContent = n ? `${n} selected` : "";
      renderStopOptions();
      selectionChanged();
    }
    renderStopOptions();

    const showStops = $("show-stops");
    showStops.addEventListener("click", () => {
      state.showStops = !state.showStops;
      showStops.setAttribute("aria-pressed", state.showStops);
      $("show-stops-label").textContent = state.showStops ? "Hide stops" : "Show stops";
      state.zoomDirty = true;
    });

    // ------------------------------------------------ loop
    map.on("zoom", () => (state.zoomDirty = true));
    let last = performance.now();
    function tick(now) {
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      let moved = false;
      if (state.playing && !scrubbing) {
        state.t += dt * SPEED;
        if (state.t >= meta.t_end) state.t = meta.t_start;
        state.target = state.t;
        setRange(state.t);
        moved = true;
      } else if (Math.abs(state.target - state.t) > 0.5) {
        // ease towards the scrubbed time so dragging the timeline stays smooth
        const k = 1 - Math.exp(-dt * 14);
        state.t += (state.target - state.t) * k;
        moved = true;
      } else if (state.t !== state.target) {
        state.t = state.target;
        moved = true;
      }
      if (moved) state.dirty = true;
      if (moved || state.zoomDirty || state.dirty || !state.frame) {
        state.zoomDirty = false;
        render();
      }
      requestAnimationFrame(tick);
    }
    setRange(state.t);
    render();
    drawActivity();
    requestAnimationFrame(tick);
    $("loading").classList.add("done");
    setTimeout(() => $("loading").remove(), 600);
    window.__app = { state, map, setPlaying, build }; // handy from the console
  }

  // Dark basemap drawn from OpenFreeMap's keyless vector tiles (OpenMapTiles schema).
  function baseStyle() {
    const name = ["coalesce", ["get", "name:fr"], ["get", "name"]];
    const roadWidth = (base) => ["interpolate", ["exponential", 1.6], ["zoom"], 9, base * 0.3, 13, base, 17, base * 6];
    return {
      version: 8,
      glyphs: "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf",
      sources: {
        omt: {
          type: "vector",
          url: "https://tiles.openfreemap.org/planet",
          attribution: '<a href="https://openfreemap.org" target="_blank" rel="noopener">OpenFreeMap</a> © <a href="https://www.openmaptiles.org/" target="_blank" rel="noopener">OpenMapTiles</a> © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors',
        },
        departments: { type: "geojson", data: "geo/departments.geojson" },
      },
      layers: [
        { id: "background", type: "background", paint: { "background-color": "#07080c" } },
        { id: "park", type: "fill", source: "omt", "source-layer": "park", paint: { "fill-color": "#0b110e", "fill-opacity": 0.8 } },
        { id: "wood", type: "fill", source: "omt", "source-layer": "landcover", filter: ["in", ["get", "class"], ["literal", ["wood", "grass"]]], paint: { "fill-color": "#0a0f0c", "fill-opacity": 0.7 } },
        { id: "water", type: "fill", source: "omt", "source-layer": "water", paint: { "fill-color": "#0d1522" } },
        { id: "waterway", type: "line", source: "omt", "source-layer": "waterway", minzoom: 11, paint: { "line-color": "#0d1522", "line-width": 1.2 } },
        { id: "building", type: "fill", source: "omt", "source-layer": "building", minzoom: 14, paint: { "fill-color": "#101218", "fill-opacity": ["interpolate", ["linear"], ["zoom"], 14, 0, 15, 0.9] } },
        { id: "rail", type: "line", source: "omt", "source-layer": "transportation", filter: ["==", ["get", "class"], "rail"], minzoom: 11, paint: { "line-color": "#1b1d24", "line-width": 1, "line-dasharray": [3, 2] } },
        { id: "road-minor", type: "line", source: "omt", "source-layer": "transportation", minzoom: 12, filter: ["in", ["get", "class"], ["literal", ["minor", "service", "tertiary"]]], layout: { "line-cap": "round", "line-join": "round" }, paint: { "line-color": "#16181e", "line-width": roadWidth(0.8) } },
        { id: "road-major", type: "line", source: "omt", "source-layer": "transportation", filter: ["in", ["get", "class"], ["literal", ["primary", "secondary", "trunk"]]], layout: { "line-cap": "round", "line-join": "round" }, paint: { "line-color": "#1d2027", "line-width": roadWidth(1.2) } },
        { id: "road-motorway", type: "line", source: "omt", "source-layer": "transportation", filter: ["==", ["get", "class"], "motorway"], layout: { "line-cap": "round", "line-join": "round" }, paint: { "line-color": "#24272f", "line-width": roadWidth(1.5) } },
        { id: "departments", type: "line", source: "departments", paint: { "line-color": "#ffffff", "line-opacity": 0.14, "line-width": 1, "line-dasharray": [3, 2] } },
        // buses are drawn just below this layer, so labels stay readable
        {
          id: "labels", type: "symbol", source: "omt", "source-layer": "transportation_name", minzoom: 14,
          layout: { "symbol-placement": "line", "text-field": name, "text-font": ["Noto Sans Regular"], "text-size": 11 },
          paint: { "text-color": "#6c7280", "text-halo-color": "#07080c", "text-halo-width": 1.4 },
        },
        {
          id: "place-minor", type: "symbol", source: "omt", "source-layer": "place", minzoom: 11,
          filter: ["in", ["get", "class"], ["literal", ["suburb", "quarter", "neighbourhood", "village"]]],
          layout: { "text-field": name, "text-font": ["Noto Sans Regular"], "text-size": ["interpolate", ["linear"], ["zoom"], 11, 10, 15, 13], "text-transform": "uppercase", "text-letter-spacing": 0.08 },
          paint: { "text-color": "#7d8391", "text-halo-color": "#07080c", "text-halo-width": 1.5 },
        },
        {
          id: "place-town", type: "symbol", source: "omt", "source-layer": "place",
          filter: ["in", ["get", "class"], ["literal", ["city", "town"]]],
          layout: { "text-field": name, "text-font": ["Noto Sans Bold"], "text-size": ["interpolate", ["linear"], ["zoom"], 8, 11, 14, 16] },
          paint: { "text-color": "#a3a9b5", "text-halo-color": "#07080c", "text-halo-width": 1.6 },
        },
      ],
    };
  }

  main().catch((err) => {
    console.error(err);
    $("loading-text").textContent = `Something went wrong: ${err.message}`;
  });
})();
