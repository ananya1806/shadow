/* Movement features from positions only, and the browser "who ran this route?" model.
 *
 * routeFeatures() mirrors xy_features() in src/route_model.py step by step, so a
 * route gets the same numbers in Python (training) and here (the website).
 * Input: array of [x, y] in yards, one point per 0.1 s, point 0 = the snap.
 */
(function (root) {
  var FPS = 10, RELEASE = 10, WINDOW = 10, CUT_MIN_SPEED = 3.0, CUT_DEGREES = 45,
      DECEL_FLOOR = -10.0;

  function routeFeatures(p) {
    var n = p.length, i, s = [], heading = [], ds = [], path = 0;
    for (i = 0; i < n; i++) {
      var a = p[Math.max(i - 1, 0)], b = p[Math.min(i + 1, n - 1)];
      var div = (i === 0 || i === n - 1) ? 1 : 2;               // central difference inside
      var vx = (b[0] - a[0]) / div, vy = (b[1] - a[1]) / div;
      s.push(Math.hypot(vx, vy) * FPS);
      heading.push(Math.atan2(vy, vx) * 180 / Math.PI);
      if (i > 0) path += Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]);
    }
    for (i = 1; i < n; i++) ds.push((s[i] - s[i - 1]) * FPS);

    var best = -1, at = -1;
    for (var t = RELEASE; t < n - WINDOW; t++) {
      if (s[t] > CUT_MIN_SPEED) {
        var d = heading[t + WINDOW] - heading[t] + 180;
        var turn = Math.abs((((d % 360) + 360) % 360) - 180);    // Python-style modulo
        if (turn > best) { best = turn; at = t; }
      }
    }
    var hasCut = best > CUT_DEGREES ? 1 : 0, kept = NaN;
    if (hasCut) {
      var lo = Infinity;
      for (i = at; i <= at + WINDOW; i++) lo = Math.min(lo, s[i]);
      kept = Math.min(Math.max(lo / Math.max(s[at - 5], 0.1), 0), 1.2);
    }
    var line = Math.hypot(p[n - 1][0] - p[0][0], p[n - 1][1] - p[0][1]);
    var sMax = Math.max.apply(null, s), sMean = s.reduce(function (x, y) { return x + y; }, 0) / n;
    return {
      speed: s,
      values: [s[RELEASE], Math.max.apply(null, ds.slice(0, RELEASE)), sMax, sMean,
               Math.max(Math.min.apply(null, ds), DECEL_FLOOR), at >= 0 ? best : NaN, kept, hasCut,
               path === 0 ? 1 : Math.min(line / path, 1)],
      cutAt: hasCut ? at : -1
    };
  }

  /* Probabilities for every WR. model = export() from route_model.py. */
  function predict(model, values) {
    var x = values.slice();
    if (isNaN(x[5])) x[5] = 0;
    if (isNaN(x[6])) x[6] = model.kept_median;
    var z = x.map(function (v, j) {
      v = Math.min(Math.max(v, model.lo[j]), model.hi[j]);
      return (v - model.mean[j]) / model.sd[j];
    });
    var logits = model.W.map(function (w, k) {
      var sum = model.b[k];
      for (var j = 0; j < z.length; j++) sum += w[j] * z[j];
      return sum;
    });
    var m = Math.max.apply(null, logits), tot = 0;
    var e = logits.map(function (l) { var v = Math.exp(l - m); tot += v; return v; });
    return e.map(function (v, k) { return { name: model.classes[k], p: v / tot }; })
            .sort(function (a, b) { return b.p - a.p; });
  }

  /* Resample a drawn stroke [[x, y, ms], ...] to 10 points per second over `seconds`.
     pace "mine" keeps where the user sped up or slowed down; "even" spreads it evenly. */
  function resample(stroke, seconds, pace) {
    var n = Math.max(2, Math.round(seconds * FPS) + 1), key = [0], i;
    for (i = 1; i < stroke.length; i++) {
      key.push(pace === "even"
        ? key[i - 1] + Math.hypot(stroke[i][0] - stroke[i - 1][0], stroke[i][1] - stroke[i - 1][1])
        : stroke[i][2] - stroke[0][2]);
    }
    var total = key[key.length - 1] || 1, out = [], j = 0;
    for (i = 0; i < n; i++) {
      var k = total * i / (n - 1);
      while (j < key.length - 2 && key[j + 1] < k) j++;
      var span = key[j + 1] - key[j], f = span > 0 ? (k - key[j]) / span : 0;
      f = Math.min(Math.max(f, 0), 1);
      out.push([stroke[j][0] + f * (stroke[j + 1][0] - stroke[j][0]),
                stroke[j][1] + f * (stroke[j + 1][1] - stroke[j][1])]);
    }
    return out;
  }

  var api = { routeFeatures: routeFeatures, predict: predict, resample: resample };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.RouteModel = api;
})(this);
