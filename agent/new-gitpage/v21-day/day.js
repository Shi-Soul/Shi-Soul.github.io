/*
 * V21：竖向滚动驱动横向轨道（原生滚动 + sticky，带平滑惯性），底色与文字色沿“一天”插值。
 * 窄屏或减少动效时退回竖向排版，颜色仍随页面滚动变化。
 */
window.Day = {
  // 一天的色标：[进度, 背景, 文字, 次文字, 强调]
  stops: [
    [0.0, "#f3e8ee", "#26222c", "#5b5563", "#c46a4b"],
    [0.25, "#eef2f6", "#1f2430", "#525a68", "#3f78b5"],
    [0.5, "#f8f5ec", "#24221c", "#5a564b", "#c58a2a"],
    [0.75, "#f1ddd3", "#2a2022", "#64504f", "#c25a3f"],
    [0.9, "#5b4a6e", "#f2ecf5", "#cfc4da", "#ffb38a"],
    [1.0, "#161c2b", "#e9edf6", "#a7b0c4", "#9cc3ff"],
  ],
  mix(a, b, u) {
    const h = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
    const x = h(a), y = h(b);
    return `rgb(${x.map((v, i) => Math.round(v + (y[i] - v) * u)).join(",")})`;
  },
  paint(p) {
    const s = Day.stops; let i = 0; while (i < s.length - 2 && p > s[i + 1][0]) i++;
    const u = Math.min(1, Math.max(0, (p - s[i][0]) / (s[i + 1][0] - s[i][0])));
    const r = document.documentElement.style;
    r.setProperty("--bg", Day.mix(s[i][1], s[i + 1][1], u));
    r.setProperty("--ink", Day.mix(s[i][2], s[i + 1][2], u));
    r.setProperty("--ink2", Day.mix(s[i][3], s[i + 1][3], u));
    r.setProperty("--acc", Day.mix(s[i][4], s[i + 1][4], u));
    const dark = p > 0.83;
    r.setProperty("--line", dark ? "rgba(255,255,255,.18)" : "rgba(35,34,42,.14)");
    r.setProperty("--card", dark ? "rgba(255,255,255,.12)" : "rgba(255,255,255,.55)");
    r.setProperty("--muted", dark ? "#9aa3b8" : "#8c8a95");
  },
  init(journey, track, sun) {
    const vertical = matchMedia("(max-width: 800px)").matches || (window.Site && Site.calm);
    document.body.classList.toggle("vertical", vertical);
    let span = 0, cur = 0;
    const measure = () => {
      if (vertical) return;
      span = Math.max(0, track.scrollWidth - innerWidth);
      journey.style.height = span + innerHeight + "px";
    };
    const progress = () => {
      if (vertical) { const m = document.documentElement.scrollHeight - innerHeight; return m > 0 ? scrollY / m : 0; }
      const top = journey.offsetTop; return span ? Math.min(1, Math.max(0, (scrollY - top) / span)) : 0;
    };
    const dot = sun && sun.querySelector(".dot");
    (function frame() {
      const p = progress();
      cur += (p - cur) * (vertical ? 1 : 0.12);          // 平滑惯性
      if (!vertical) track.style.transform = `translate3d(${-cur * span}px,0,0)`;
      Day.paint(cur);
      if (dot) dot.style.left = cur * 100 + "%";
      requestAnimationFrame(frame);
    })();
    addEventListener("resize", measure);
    measure();
    return measure;
  },
};
