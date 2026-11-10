/* WR Doppelgänger website: finder, compare panel, movement player, draw-a-route.
 * Data: window.WRS (finder, inline), window.MOVE (assets/movement.js),
 *       window.RouteModel (assets/route_features.js), window.SITE (featured pair).
 */
(function () {
  "use strict";
  var M = window.MOVE, WRS = window.WRS, RM = window.RouteModel, SITE = window.SITE;
  var C = { star: "#2a78d6", dbl: "#eb6834", ink: "#0b0b0b", ink2: "#52514e", muted: "#898781",
            grid: "#e1e0d9", field: "#f3f6f1", yard: "#dbe3d5", scrim: "#9aa394", band: "#ebeae4" };
  var FONT = getComputedStyle(document.body).fontFamily;
  var SVGNS = "http://www.w3.org/2000/svg";
  var NAMES = WRS.map(function (w) { return w.name; });
  var byName = {};
  WRS.forEach(function (w) { byName[w.name] = w; });

  function $(id) { return document.getElementById(id); }
  function el(tag, text, cls) {
    var e = document.createElement(tag);
    if (text !== undefined && text !== null) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }
  function svg(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function decode(a) {
    var p = [];
    for (var i = 0; i < a.length; i += 2) p.push([a[i] / 10, a[i + 1] / 10]);
    return p;
  }
  function fmt(v, d) { return isNaN(v) ? "–" : v.toFixed(d); }
  function pct(v) { return (v * 100).toFixed(v < 0.1 ? 1 : 0) + "%"; }
  function goTo(id) { $(id).scrollIntoView({ behavior: "smooth", block: "start" }); }
  function teamOf(name) { return M.wr[name] ? M.wr[name].team : ""; }
  function fillPlayers(sel, list, withNone) {
    sel.innerHTML = "";
    if (withNone) sel.appendChild(new Option("— nobody —", ""));
    list.forEach(function (n) { sel.appendChild(new Option(n + " (" + teamOf(n) + ")", n)); });
  }

  /* ------------------------------------------------------------ nav + tabs */
  var links = document.querySelectorAll("nav a[href^='#']");
  if ("IntersectionObserver" in window) {
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        links.forEach(function (a) { a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id); });
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    document.querySelectorAll("main section[id]").forEach(function (s) { obs.observe(s); });
  }
  document.querySelectorAll(".tabs").forEach(function (bar) {
    var btns = bar.querySelectorAll("button");
    btns.forEach(function (b) {
      b.addEventListener("click", function () {
        btns.forEach(function (o) {
          o.setAttribute("aria-selected", o === b ? "true" : "false");
          $(o.dataset.tab).hidden = o !== b;
        });
      });
    });
  });

  /* ------------------------------------------------------------ radar (SVG) */
  function radar(host, a, b) {
    host.innerHTML = "";
    var s = svg("svg", { viewBox: "0 0 360 300", role: "img",
                         "aria-label": "Radar of six movement traits, percentiles" }, host);
    var cx = 180, cy = 150, R = 100, n = M.traits.length;
    function pt(i, v) {
      var ang = -Math.PI / 2 + i * 2 * Math.PI / n;
      return [cx + Math.cos(ang) * R * v / 100, cy + Math.sin(ang) * R * v / 100];
    }
    [25, 50, 75, 100].forEach(function (v) {
      var ring = [];
      for (var i = 0; i < n; i++) ring.push(pt(i, v).join(","));
      svg("polygon", { points: ring.join(" "), fill: "none", stroke: C.grid }, s);
    });
    for (var i = 0; i < n; i++) {
      var e = pt(i, 100), lab = pt(i, 122);
      svg("line", { x1: cx, y1: cy, x2: e[0], y2: e[1], stroke: C.grid }, s);
      var t = svg("text", { x: lab[0], y: lab[1] + 4, "text-anchor": Math.abs(lab[0] - cx) < 5 ? "middle" :
                            (lab[0] > cx ? "start" : "end"), "font-size": 12, fill: C.ink2 }, s);
      t.textContent = M.traits[i];
    }
    [[a, C.star], [b, C.dbl]].forEach(function (pair) {
      if (!pair[0]) return;
      var vals = pair[0].pct, pts = vals.map(function (v, k) { return pt(k, v).join(","); });
      svg("polygon", { points: pts.join(" "), fill: pair[1], "fill-opacity": 0.12, stroke: pair[1],
                       "stroke-width": 2.2, "stroke-linejoin": "round" }, s);
      vals.forEach(function (v, k) {
        var p = pt(k, v), dot = svg("circle", { cx: p[0], cy: p[1], r: 4, fill: pair[1],
                                               stroke: "#fff", "stroke-width": 1.5 }, s);
        svg("title", {}, dot).textContent = pair[0].name + ": " + M.traits[k] + " " + Math.round(v) + "th percentile";
      });
    });
  }

  /* ---------------------------------------------- speed-over-time (SVG) */
  function speedChart(host, series) {
    host.innerHTML = "";
    var Wd = 440, H = 250, L = 38, Rm = 14, T = 12, B = 34, tmax = M.t[M.t.length - 1], ymax = 9;
    var s = svg("svg", { viewBox: "0 0 " + Wd + " " + H, role: "img",
                         "aria-label": "Average speed by time since the snap" }, host);
    function X(t) { return L + (Wd - L - Rm) * t / tmax; }
    function Y(v) { return T + (H - T - B) * (1 - v / ymax); }
    for (var v = 0; v <= ymax; v += 3) {
      svg("line", { x1: L, x2: Wd - Rm, y1: Y(v), y2: Y(v), stroke: C.grid }, s);
      svg("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end", "font-size": 11, fill: C.muted }, s).textContent = v;
    }
    for (var t = 0; t <= tmax + 1e-9; t += 1) {
      svg("text", { x: X(t), y: H - B + 16, "text-anchor": "middle", "font-size": 11, fill: C.muted }, s).textContent = t + " s";
    }
    svg("text", { x: L, y: H - 4, "font-size": 11, fill: C.ink2 }, s).textContent = "seconds since snap · speed in yd/s";
    var up = M.t.map(function (t, i) { return X(t) + "," + Y(M.league.q75[i]); });
    var dn = M.t.map(function (t, i) { return X(t) + "," + Y(M.league.q25[i]); }).reverse();
    svg("polygon", { points: up.concat(dn).join(" "), fill: C.band }, s);
    svg("polyline", { points: M.t.map(function (t, i) { return X(t) + "," + Y(M.league.q50[i]); }).join(" "),
                      fill: "none", stroke: C.muted, "stroke-width": 1.2 }, s);
    series.forEach(function (sr) {
      svg("polyline", { points: M.t.map(function (t, i) { return X(t) + "," + Y(sr.y[i]); }).join(" "),
                        fill: "none", stroke: sr.color, "stroke-width": 2.6, "stroke-linejoin": "round" }, s);
    });
    // crosshair + readout
    var cross = svg("line", { y1: T, y2: H - B, stroke: C.ink2, "stroke-width": 1, visibility: "hidden" }, s);
    var dots = series.map(function (sr) {
      return svg("circle", { r: 4.5, fill: sr.color, stroke: "#fff", "stroke-width": 1.5, visibility: "hidden" }, s);
    });
    var tip = svg("text", { x: L + 8, y: T + 14, "font-size": 12, fill: C.ink, visibility: "hidden" }, s);
    var hit = svg("rect", { x: L, y: T, width: Wd - L - Rm, height: H - T - B, fill: "transparent" }, s);
    function move(ev) {
      var box = s.getBoundingClientRect(), px = (ev.clientX - box.left) * Wd / box.width;
      var i = Math.round(Math.min(Math.max((px - L) / (Wd - L - Rm), 0), 1) * (M.t.length - 1));
      cross.setAttribute("x1", X(M.t[i])); cross.setAttribute("x2", X(M.t[i]));
      cross.setAttribute("visibility", "visible");
      var parts = [M.t[i].toFixed(1) + " s"];
      series.forEach(function (sr, k) {
        dots[k].setAttribute("cx", X(M.t[i])); dots[k].setAttribute("cy", Y(sr.y[i]));
        dots[k].setAttribute("visibility", "visible");
        parts.push(sr.name.split(" ").slice(-1)[0] + " " + sr.y[i].toFixed(1));
      });
      parts.push("median WR " + M.league.q50[i].toFixed(1));
      tip.textContent = parts.join("  ·  ");
      tip.setAttribute("visibility", "visible");
    }
    function hide() {
      [cross, tip].concat(dots).forEach(function (e) { e.setAttribute("visibility", "hidden"); });
    }
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerleave", hide);
  }

  /* ---------------------------------------------------------------- finder */
  var F = { a: SITE.star, b: null };
  var sel = $("wr-select"), search = $("wr-search"), built = $("built-only"), rows = $("wr-rows"), who = $("wr-who");

  function finderFill(filter) {
    var f = (filter || "").toLowerCase(), keep = sel.value;
    sel.innerHTML = "";
    WRS.forEach(function (w) {
      if (f && w.name.toLowerCase().indexOf(f) < 0) return;
      sel.appendChild(new Option(w.name + " (" + w.team + ")", w.name));
    });
    if (keep && byName[keep] && (!f || keep.toLowerCase().indexOf(f) >= 0)) sel.value = keep;
  }
  function matches(w) {
    return w.m.filter(function (m) { return !built.checked || m[5]; }).slice(0, 5);
  }
  function finderShow() {
    var w = byName[sel.value];
    rows.innerHTML = "";
    if (!w) { who.textContent = "No receiver matches that name."; return; }
    F.a = w.name;
    who.textContent = w.routes + " routes · 2021 APY " + w.apy + " · Signature tier: " + w.tier +
      (w.hub ? " · hub (common profile)" : "");
    var list = matches(w);
    if (!list.length) {
      var tr = el("tr"), td = el("td", "No built-alike match in his top " + w.m.length + ".");
      td.colSpan = 6; tr.appendChild(td); rows.appendChild(tr);
      F.b = null; compareShow(); return;
    }
    if (!list.some(function (m) { return m[0] === F.b; })) F.b = list[0][0];
    list.forEach(function (m, i) {
      var tr = el("tr", null, m[0] === F.b ? "pick sel" : "pick"), name = el("td", null, "dbl");
      tr.tabIndex = 0;
      tr.setAttribute("aria-label", "Compare with " + m[0]);
      name.appendChild(document.createTextNode(m[0] + " "));
      name.appendChild(el("span", m[1], "team"));
      var sim = el("td", null, "num"), bar = el("span", null, "bar");
      bar.style.width = Math.max(4, Math.round(m[2] * 60)) + "px";
      sim.appendChild(bar); sim.appendChild(document.createTextNode(m[2].toFixed(2)));
      var flags = el("td");
      if (m[4]) flags.appendChild(el("span", "Stable", "badge ok"));
      if (m[6]) flags.appendChild(el("span", "Hub", "badge hub"));
      if (m[5]) flags.appendChild(el("span", "Built like", "badge built"));
      [el("td", String(i + 1)), name, sim, el("td", m[3] + "%", "num"), flags, el("td", m[7], "num")]
        .forEach(function (c) { tr.appendChild(c); });
      function pick() { F.b = m[0]; finderShow(); }
      tr.addEventListener("click", pick);
      tr.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
      rows.appendChild(tr);
    });
    compareShow();
  }
  function compareShow() {
    var a = M.wr[F.a], b = F.b ? M.wr[F.b] : null, w = byName[F.a];
    var m = w && F.b ? w.m.filter(function (x) { return x[0] === F.b; })[0] : null;
    $("cmp-a").textContent = F.a;
    $("cmp-b").textContent = F.b || "—";
    $("cmp-meta").textContent = m ? "Similarity " + m[2].toFixed(2) + " · in both halves " + m[3] + "% of splits · " +
      (m[4] ? "stable" : "not stable") + " · " + w.apy + " vs " + m[7] : "";
    radar($("cmp-radar"), { name: F.a, pct: a.pct }, b ? { name: F.b, pct: b.pct } : null);
    var series = [{ name: F.a, y: a.speed, color: C.star }];
    if (b) series.push({ name: F.b, y: b.speed, color: C.dbl });
    speedChart($("cmp-speed"), series);
  }
  function finderSet(a, b) {
    search.value = "";
    finderFill("");
    sel.value = a; F.b = b || null;
    finderShow();
  }
  finderFill();
  sel.value = SITE.star;
  F.b = SITE.dbl;
  search.addEventListener("input", function () { finderFill(search.value); finderShow(); });
  sel.addEventListener("change", function () { F.b = null; finderShow(); });
  built.addEventListener("change", finderShow);
  $("cmp-watch").addEventListener("click", function () { watchSet(F.a, F.b); goTo("watch"); });
  $("cmp-draw").addEventListener("click", function () { drawLoadReal(F.a); goTo("draw"); });
  finderShow();

  document.querySelectorAll("tr[data-a]").forEach(function (tr) {
    tr.classList.add("pick");
    tr.tabIndex = 0;
    function go() { finderSet(tr.dataset.a, tr.dataset.b); goTo("finder"); }
    tr.addEventListener("click", go);
    tr.addEventListener("keydown", function (e) { if (e.key === "Enter") go(); });
  });

  /* ------------------------------------------------------- field drawing */
  function setupCanvas(cv) {
    var dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    }
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h };
  }
  /* view: {x0, x1, y0, y1} in yards (x = inside, y = downfield); equal scale, centred. */
  function makeView(g, v) {
    var sc = Math.min(g.w / (v.x1 - v.x0), g.h / (v.y1 - v.y0));
    var ox = (g.w - sc * (v.x1 - v.x0)) / 2, oy = (g.h - sc * (v.y1 - v.y0)) / 2;
    return {
      sc: sc,
      px: function (x) { return ox + (x - v.x0) * sc; },
      py: function (y) { return g.h - oy - (y - v.y0) * sc; },
      yd: function (px, py) { return [v.x0 + (px - ox) / sc, v.y0 + (g.h - oy - py) / sc]; },
      v: v
    };
  }
  function drawField(g, vw) {
    var ctx = g.ctx, v = vw.v;
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, g.w, g.h);
    ctx.fillStyle = C.field;
    ctx.fillRect(vw.px(v.x0), vw.py(v.y1), (v.x1 - v.x0) * vw.sc, (v.y1 - v.y0) * vw.sc);
    ctx.font = "11px " + FONT;
    for (var y = Math.ceil(v.y0 / 5) * 5; y <= v.y1 - 1.5; y += 5) {
      ctx.strokeStyle = y === 0 ? C.scrim : C.yard;
      ctx.lineWidth = y === 0 ? 1.6 : 1;
      ctx.beginPath(); ctx.moveTo(vw.px(v.x0), vw.py(y)); ctx.lineTo(vw.px(v.x1), vw.py(y)); ctx.stroke();
      ctx.fillStyle = C.muted;
      ctx.fillText(y === 0 ? "snap" : (y > 0 ? "+" : "") + y + " yd", vw.px(v.x0) + 6, vw.py(y) - 4);
    }
    ctx.fillStyle = C.muted;
    ctx.textAlign = "right";
    ctx.fillText("middle of field →", vw.px(v.x1) - 6, vw.py(v.y1) + 14);
    ctx.textAlign = "left";
    ctx.fillText("← sideline", vw.px(v.x0) + 6, vw.py(v.y1) + 14);
    ctx.fillStyle = C.ink;
    ctx.beginPath(); ctx.arc(vw.px(0), vw.py(0), 5, 0, 2 * Math.PI); ctx.fill();
  }
  function pointAt(p, t) {                                    // position at time t (s)
    var f = t * 10, i = Math.floor(f);
    if (i >= p.length - 1) return p[p.length - 1];
    var r = f - i;
    return [p[i][0] + r * (p[i + 1][0] - p[i][0]), p[i][1] + r * (p[i + 1][1] - p[i][1])];
  }
  function drawRoute(ctx, vw, p, t, color, opts) {
    opts = opts || {};
    var last = Math.min(Math.floor(t * 10), p.length - 1);
    ctx.strokeStyle = color; ctx.lineWidth = opts.width || 2; ctx.globalAlpha = opts.alpha || 0.85;
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(vw.px(p[0][0]), vw.py(p[0][1]));
    for (var i = 1; i <= last; i++) ctx.lineTo(vw.px(p[i][0]), vw.py(p[i][1]));
    var head = pointAt(p, t);
    ctx.lineTo(vw.px(head[0]), vw.py(head[1]));
    ctx.stroke();
    ctx.globalAlpha = 1;
    if (opts.ticks) {                                         // dot every 0.5 s: wider gaps = faster
      ctx.fillStyle = color;
      for (i = 5; i <= last; i += 5) {
        ctx.beginPath(); ctx.arc(vw.px(p[i][0]), vw.py(p[i][1]), 2.6, 0, 2 * Math.PI); ctx.fill();
      }
    }
    var done = t >= (p.length - 1) / 10;
    ctx.beginPath(); ctx.arc(vw.px(head[0]), vw.py(head[1]), done ? 4.5 : 5, 0, 2 * Math.PI);
    if (done) { ctx.fillStyle = "#fff"; ctx.fill(); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke(); }
    else { ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke(); }
  }

  /* ------------------------------------------------------ watch them move */
  var Wt = { t: 0, playing: false, speed: 1, routes: [], tmax: 3, last: null };
  var wA = $("watch-a"), wB = $("watch-b"), wN = $("watch-n"), wPlay = $("watch-play"),
      wRate = $("watch-rate"), wScrub = $("watch-scrub"), wTime = $("watch-time"), wCv = $("watch-canvas");
  fillPlayers(wA, NAMES, false);
  fillPlayers(wB, NAMES, true);

  function watchRoutes() {
    var n = +wN.value, out = [];
    [[wA.value, C.star, "a"], [wB.value, C.dbl, "b"]].forEach(function (x) {
      if (!x[0]) return;
      M.wr[x[0]].routes.slice(0, n).forEach(function (r) {
        var p = decode(r);
        out.push({ p: p, color: x[1], who: x[2], speed: RM.routeFeatures(p).speed });
      });
    });
    return out;
  }
  function watchSet(a, b) {
    wA.value = a; wB.value = b || "";
    watchReset(true);
  }
  function watchReset(autoplay) {
    Wt.routes = watchRoutes();
    Wt.tmax = Math.max.apply(null, Wt.routes.map(function (r) { return (r.p.length - 1) / 10; }));
    Wt.tmax = Math.min(Wt.tmax, 6);
    wScrub.max = Wt.tmax.toFixed(1);
    Wt.t = 0;
    $("watch-a-name").textContent = wA.value;
    $("watch-b-name").textContent = wB.value || "—";
    $("watch-b-row").style.opacity = wB.value ? 1 : 0.35;
    if (autoplay) watchPlay(true); else watchDraw();
  }
  function watchView() {
    var xs = [-10, 10], ys = [-2, 12];
    Wt.routes.forEach(function (r) {
      r.p.forEach(function (q) {
        xs[0] = Math.min(xs[0], q[0]); xs[1] = Math.max(xs[1], q[0]);
        ys[0] = Math.min(ys[0], q[1]); ys[1] = Math.max(ys[1], q[1]);
      });
    });
    return { x0: xs[0] - 2, x1: xs[1] + 2, y0: ys[0] - 2, y1: ys[1] + 3 };
  }
  function liveSpeed(who) {
    var vals = [];
    Wt.routes.forEach(function (r) {
      if (r.who !== who) return;
      var i = Math.round(Wt.t * 10);
      if (i < r.speed.length) vals.push(r.speed[i]);
    });
    if (!vals.length) return null;
    return { v: vals.reduce(function (x, y) { return x + y; }, 0) / vals.length, n: vals.length };
  }
  function watchDraw() {
    var g = setupCanvas(wCv), vw = makeView(g, watchView());
    drawField(g, vw);
    Wt.routes.forEach(function (r) { drawRoute(g.ctx, vw, r.p, Wt.t, r.color, { ticks: true }); });
    wScrub.value = Wt.t.toFixed(1);
    wTime.textContent = Wt.t.toFixed(1) + " s";
    [["a", "watch-a-speed", "watch-a-bar"], ["b", "watch-b-speed", "watch-b-bar"]].forEach(function (x) {
      var s = liveSpeed(x[0]);
      $(x[1]).textContent = s ? s.v.toFixed(1) + " yd/s · " + s.n + " still running" : "all routes ended";
      $(x[2]).style.width = s ? Math.min(100, s.v / 10 * 100) + "%" : "0";
    });
  }
  function frame(now) {
    if (!Wt.playing) return;
    if (Wt.last !== null) Wt.t += (now - Wt.last) / 1000 * Wt.speed;
    Wt.last = now;
    if (Wt.t >= Wt.tmax) { Wt.t = Wt.tmax; watchPlay(false); }
    watchDraw();
    if (Wt.playing) requestAnimationFrame(frame);
  }
  function watchPlay(on) {
    if (on && Wt.t >= Wt.tmax) Wt.t = 0;
    Wt.playing = on; Wt.last = null;
    wPlay.textContent = on ? "❚❚ Pause" : "▶ Play";
    wPlay.setAttribute("aria-pressed", on ? "true" : "false");
    if (on) requestAnimationFrame(frame); else watchDraw();
  }
  wPlay.addEventListener("click", function () { watchPlay(!Wt.playing); });
  $("watch-restart").addEventListener("click", function () { Wt.t = 0; watchPlay(true); });
  wRate.addEventListener("change", function () { Wt.speed = +wRate.value; });
  wScrub.addEventListener("input", function () { watchPlay(false); Wt.t = +wScrub.value; watchDraw(); });
  [wA, wB, wN].forEach(function (s) { s.addEventListener("change", function () { watchReset(true); }); });
  $("watch-swap").addEventListener("click", function () {
    var w = byName[wA.value], best = w && w.m.length ? w.m[0][0] : "";
    wB.value = best; watchReset(true);
  });
  window.addEventListener("resize", function () { watchDraw(); drawRender(); });
  watchSet(SITE.star, SITE.dbl);
  watchPlay(false);
  Wt.t = Wt.tmax; watchDraw();                                 // show full routes until the user presses play

  /* ------------------------------------------------------- draw a route */
  var D = { stroke: null, path: null, real: null, near: null, t: null, anim: false, drawing: false };
  var dCv = $("draw-canvas"), dTime = $("draw-time"), dTimeLab = $("draw-time-label"),
      dReal = $("draw-real"), dOut = $("draw-results");
  var DRAW_VIEW = { x0: -15, x1: 15, y0: -3, y1: 25 };
  fillPlayers(dReal, NAMES, false);
  dReal.value = SITE.star;

  // Pre-compute the standardized features of every embedded real route (for "closest real route").
  var BANK = [];
  NAMES.forEach(function (n) {
    M.wr[n].routes.forEach(function (r, k) {
      var p = decode(r);
      BANK.push({ name: n, k: k, p: p, z: zscore(RM.routeFeatures(p).values) });
    });
  });
  function zscore(v) {
    var m = M.model;
    return v.map(function (x, j) {
      if (isNaN(x)) x = j === 5 ? 0 : m.kept_median;
      x = Math.min(Math.max(x, m.lo[j]), m.hi[j]);
      return (x - m.mean[j]) / m.sd[j];
    });
  }
  function nearest(z, skip) {                                // skip = the loaded real route itself
    var best = null, bd = Infinity;
    BANK.forEach(function (b) {
      if (skip && b.name === skip.name && b.k === skip.k) return;
      var d = 0;
      for (var j = 0; j < z.length; j++) d += (z[j] - b.z[j]) * (z[j] - b.z[j]);
      if (d < bd) { bd = d; best = b; }
    });
    return best;
  }
  function pace() { return document.querySelector("input[name=pace]:checked").value; }

  function drawViewFor() {
    if (D.drawing || !D.path) return DRAW_VIEW;
    var v = { x0: DRAW_VIEW.x0, x1: DRAW_VIEW.x1, y0: DRAW_VIEW.y0, y1: DRAW_VIEW.y1 };
    [D.path, D.near ? D.near.p : []].forEach(function (p) {
      p.forEach(function (q) {
        v.x0 = Math.min(v.x0, q[0] - 2); v.x1 = Math.max(v.x1, q[0] + 2);
        v.y0 = Math.min(v.y0, q[1] - 2); v.y1 = Math.max(v.y1, q[1] + 2);
      });
    });
    return v;
  }
  function drawRender() {
    var g = setupCanvas(dCv), vw = makeView(g, drawViewFor()), ctx = g.ctx;
    D.vw = vw;
    drawField(g, vw);
    if (D.drawing && D.stroke) {
      ctx.strokeStyle = C.ink2; ctx.lineWidth = 2.5; ctx.lineCap = "round"; ctx.lineJoin = "round";
      ctx.beginPath();
      D.stroke.forEach(function (q, i) {
        var x = vw.px(q[0] - D.stroke[0][0]), y = vw.py(q[1] - D.stroke[0][1]);
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      });
      ctx.stroke();
      return;
    }
    if (!D.path) {
      var wide = g.w > 520;
      ctx.fillStyle = C.ink2; ctx.textAlign = "center"; ctx.font = (wide ? 15 : 14) + "px " + FONT;
      ctx.fillText(wide ? "Press and drag to draw a route from the snap dot" : "Press and drag from the dot",
                   g.w / 2, vw.py(18.6));
      ctx.font = "13px " + FONT; ctx.fillStyle = C.muted;
      ctx.fillText(wide ? "Draw at your own pace: slow down where he would brake" : "Slow down where he would brake",
                   g.w / 2, vw.py(18.6) + 20);
      ctx.textAlign = "left";
      return;
    }
    var t = D.t === null ? 99 : D.t;
    if (D.near) drawRoute(ctx, vw, D.near.p, 99, C.dbl, { width: 1.8, alpha: 0.75 });
    var f = D.feats;
    if (f.cutAt >= 0 && t * 10 >= f.cutAt) {                  // highlight the sharpest cut
      ctx.strokeStyle = C.ink; ctx.globalAlpha = 0.14; ctx.lineWidth = 12; ctx.beginPath();
      for (var i = f.cutAt; i <= Math.min(f.cutAt + 10, D.path.length - 1); i++) {
        var x = vw.px(D.path[i][0]), y = vw.py(D.path[i][1]);
        if (i === f.cutAt) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke(); ctx.globalAlpha = 1;
    }
    drawRoute(ctx, vw, D.path, t, C.star, { width: 2.6, ticks: true, alpha: 1 });
  }

  function drawAnimate() {
    D.t = 0; D.anim = true;
    var start = null;
    function step(now) {
      if (!D.anim) return;
      if (start === null) start = now;
      D.t = (now - start) / 1000;
      if (D.t >= (D.path.length - 1) / 10) { D.t = null; D.anim = false; }
      drawRender();
      if (D.anim) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function analyze(animate) {
    if (D.stroke && !D.real) {
      var st = D.stroke.map(function (q) { return [q[0] - D.stroke[0][0], q[1] - D.stroke[0][1], q[2]]; });
      D.path = RM.resample(st, +dTime.value, pace());
    }
    if (!D.path) return;
    D.feats = RM.routeFeatures(D.path);
    var preds = RM.predict(M.model, D.feats.values);
    var z = zscore(D.feats.values);
    D.near = nearest(z, D.realKey);
    renderResults(preds);
    if (animate) drawAnimate(); else drawRender();
  }

  var FEAT_ROWS = [
    ["Release speed (1.0 s)", "yd/s", 1], ["Release burst", "yd/s²", 1], ["Top speed", "yd/s", 1],
    ["Average speed", "yd/s", 1], ["Hardest braking", "yd/s²", 1], ["Sharpest cut", "°", 0],
    ["Speed kept in cut", "", 2], ["Has a real cut", "", 0], ["Straightness", "", 2]];

  function renderResults(preds) {
    dOut.innerHTML = "";
    var chance = 1 / preds.length, v = D.feats.values;
    var head = el("h3", D.real ? "Who ran this route? The model's guess" : "Who runs it like this?");
    dOut.appendChild(head);
    if (D.real) {
      var rank = preds.map(function (p) { return p.name; }).indexOf(D.real) + 1;
      dOut.appendChild(el("p", "This is a real route by " + D.real + ". The model ranks him #" + rank +
        " of " + preds.length + (rank <= 5 ? " (in the top 5)." : ". One route is a weak clue, so misses are common."), "who"));
    }
    var list = el("ol", null, "preds");
    var top = preds[0].p;
    preds.slice(0, 5).forEach(function (p) {
      var li = el("li"), name = el("span", p.name, "pname"), team = el("span", " " + teamOf(p.name), "team");
      var bar = el("span", null, "pbar"); bar.style.width = Math.max(3, p.p / top * 100) + "%";
      if (p.name === D.real) li.className = "truth";
      var val = el("span", pct(p.p) + " · " + (p.p / chance).toFixed(1) + "× chance", "pval");
      var b = el("button", "Watch", "mini");
      b.setAttribute("aria-label", "Watch " + p.name + " move");
      b.addEventListener("click", function () { watchSet(p.name, D.near ? D.near.name : ""); goTo("watch"); });
      var row = el("div", null, "prow"); row.appendChild(name); row.appendChild(team);
      li.appendChild(row); li.appendChild(bar); li.appendChild(val); li.appendChild(b);
      list.appendChild(li);
    });
    dOut.appendChild(list);
    if (D.near) {
      var nr = el("p", null, "who");
      nr.appendChild(el("span", "Closest real route ", null));
      nr.appendChild(el("b", D.near.name, "dblc"));
      nr.appendChild(document.createTextNode(" (" + teamOf(D.near.name) + "), drawn in orange" +
        (D.real ? " (another route, not this one)." : ".")));
      dOut.appendChild(nr);
    }
    var warn = [];
    if (v[2] > 11) warn.push("Top speed " + v[2].toFixed(1) + " yd/s is faster than any WR in the data. Lengthen the route time.");
    if (v[0] < 1) warn.push("Almost no movement in the first second. Real releases reach about 4 yd/s by 1.0 s.");
    warn.forEach(function (w) { dOut.appendChild(el("p", w, "note")); });

    var tbl = el("table", null, "feat");
    var thead = el("tr");
    ["Your route", "", "Typical WR (middle 50%)"].forEach(function (h, i) {
      var th = el("th", h); if (i === 1) th.className = "num"; thead.appendChild(th);
    });
    tbl.appendChild(thead);
    FEAT_ROWS.forEach(function (r, j) {
      var tr = el("tr"), q = M.typical[j];
      var val = j === 7 ? (v[j] ? "yes" : "no") : fmt(j === 6 ? v[j] * 100 : v[j], j === 6 ? 0 : r[2]) +
                (j === 6 && !isNaN(v[j]) ? "%" : (r[1] ? " " + r[1] : ""));
      var range = el("td", null, "range");
      if (j !== 7) {
        var trk = el("span", null, "track"), band = el("span", null, "band"), mark = el("span", null, "mark");
        var lo = q.p05, hi = q.p95, pos = function (x) { return Math.min(Math.max((x - lo) / (hi - lo), 0), 1) * 100; };
        band.style.left = pos(q.p25) + "%"; band.style.width = (pos(q.p75) - pos(q.p25)) + "%";
        trk.appendChild(band);
        if (!isNaN(v[j])) { mark.style.left = pos(v[j]) + "%"; trk.appendChild(mark); }
        range.appendChild(trk);
      } else {
        range.textContent = Math.round(q.p50 * 100) + "% of routes have one";
      }
      [el("td", r[0]), el("td", val, "num"), range].forEach(function (c) { tr.appendChild(c); });
      tbl.appendChild(tr);
    });
    dOut.appendChild(tbl);
  }

  function toYards(ev) {
    var box = dCv.getBoundingClientRect();
    return D.vw.yd(ev.clientX - box.left, ev.clientY - box.top);
  }
  dCv.addEventListener("pointerdown", function (ev) {
    ev.preventDefault();
    dCv.setPointerCapture(ev.pointerId);
    D.anim = false; D.real = null; D.realKey = null; D.path = null; D.near = null; D.t = null;
    var q = toYards(ev);
    D.stroke = [[q[0], q[1], performance.now()]];
    D.drawing = true;
    drawRender();
  });
  dCv.addEventListener("pointermove", function (ev) {
    if (!D.drawing) return;
    var q = toYards(ev), last = D.stroke[D.stroke.length - 1];
    if (Math.hypot(q[0] - last[0], q[1] - last[1]) < 0.05) return;
    D.stroke.push([q[0], q[1], performance.now()]);
    drawRender();
  });
  function endStroke() {
    if (!D.drawing) return;
    D.drawing = false;
    var st = D.stroke;
    if (st.length < 4 || st[st.length - 1][2] - st[0][2] < 250) {
      D.stroke = null; drawRender();
      dOut.innerHTML = "";
      dOut.appendChild(el("p", "That was too short. Press, drag for a second or more, then let go.", "note"));
      return;
    }
    analyze(true);
  }
  dCv.addEventListener("pointerup", endStroke);
  dCv.addEventListener("pointercancel", endStroke);
  dTime.addEventListener("input", function () {
    dTimeLab.textContent = (+dTime.value).toFixed(1) + " s";
    if (D.stroke && !D.real) analyze(false);
  });
  document.querySelectorAll("input[name=pace]").forEach(function (r) {
    r.addEventListener("change", function () { if (D.stroke && !D.real) analyze(false); });
  });
  $("draw-clear").addEventListener("click", function () {
    D = { stroke: null, path: null, real: null, near: null, t: null, anim: false, drawing: false };
    dOut.innerHTML = "";
    dOut.appendChild(el("p", "Draw a route on the field to see who runs it most like that.", "who"));
    drawRender();
  });
  $("draw-replay").addEventListener("click", function () { if (D.path) drawAnimate(); });
  function drawLoadReal(name) {
    dReal.value = name;
    var rs = M.wr[name].routes, k = Math.floor(Math.random() * rs.length);
    D.anim = false; D.stroke = null; D.real = name; D.path = decode(rs[k]); D.t = null;
    D.realKey = { name: name, k: k };
    analyze(true);
  }
  $("draw-load").addEventListener("click", function () { drawLoadReal(dReal.value); });
  dTimeLab.textContent = (+dTime.value).toFixed(1) + " s";
  drawRender();
})();
