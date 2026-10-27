/* V10：把研究画成一条训练曲线。论文是曲线上的 checkpoint（取 data.js 的 date），履历节点标在 x 轴上。 */
window.Run = {
  t: (s) => { const [y, m] = String(s).split(/[.\-]/).map(Number); return y + ((m || 6) - 0.5) / 12; },
  plot(host, tip, onPick) {
    const { S, esc, venue, media } = Site;
    const W = 1000, H = 330, L = 40, R = 170, T = 24, B = 62;
    const now = new Date(), tNow = now.getFullYear() + now.getMonth() / 12, t0 = 2021.5, t1 = tNow + 1;
    const X = (t) => L + (t - t0) / (t1 - t0) * (W - L - R), Y = (v) => T + (1 - v) * (H - T - B);
    const f = (t) => 0.05 + 0.82 / (1 + Math.exp(-(t - 2024.7) * 1.5));          // 平滑曲线
    const n = (t) => Math.sin(t * 41.3) * 0.5 + Math.sin(t * 97.1 + 1.3) * 0.3 + Math.sin(t * 13.7) * 0.2; // 确定性“噪声”
    const pts = (a, b, g) => { const o = []; for (let i = 0; i <= 160; i++) { const t = a + (b - a) * i / 160; o.push([X(t), Y(g(t))]); } return o; };
    const d = (p) => "M" + p.map((q) => q[0].toFixed(1) + "," + q[1].toFixed(1)).join("L");
    const up = pts(t0, tNow, (t) => f(t) + 0.035 + 0.012 * n(t)), lo = pts(t0, tNow, (t) => f(t) - 0.035 - 0.012 * n(t + 3));
    let s = "";
    for (let v = 0; v <= 1; v += 0.25) s += `<line class="grid" x1="${L}" x2="${W - R + 40}" y1="${Y(v)}" y2="${Y(v)}"/>`;
    for (let y = Math.ceil(t0); y <= Math.floor(t1); y++) s += `<line class="grid" x1="${X(y)}" x2="${X(y)}" y1="${T}" y2="${H - B}"/><text x="${X(y)}" y="${H - B + 16}" text-anchor="middle">${y}</text>`;
    s += `<line class="ax" x1="${L}" x2="${W - R + 40}" y1="${H - B}" y2="${H - B}"/><line class="ax" x1="${L}" x2="${L}" y1="${T}" y2="${H - B}"/>`;
    s += `<text x="${L - 8}" y="${T + 4}" text-anchor="end">1.0</text><text x="${L - 8}" y="${H - B}" text-anchor="end">0</text>`;
    s += `<path class="band" d="${d(up)} L${lo.reverse().map((q) => q[0].toFixed(1) + "," + q[1].toFixed(1)).join("L")} Z"/>`;
    s += `<path class="curve raw" style="stroke-width:1;opacity:.35" d="${d(pts(t0, tNow, (t) => f(t) + 0.03 * n(t * 1.7)))}"/>`;
    s += `<path class="curve main" d="${d(pts(t0, tNow, f))}"/>`;
    s += `<path class="future" d="${d(pts(tNow, t1, (t) => f(t) + (t - tNow) * 0.05))}"/>`;
    s += `<text class="fut-l" x="${X(t1) + 8}" y="${Y(f(t1) + 0.05) + 4}">general-purpose robots?</text>`;
    // 履历节点
    const ev = S.education.concat(S.experience).map((e) => ({ t: Run.t(e.when.split(/\s*[–-]\s*/)[0]), label: e.title.replace(/ in .*/, "").replace("Research Intern", "Intern") + " · " + e.org.split(",")[0].replace("Shanghai Jiao Tong University", "SJTU").replace("Yunqi Academy of Engineering", "Yunqi") })).sort((a, b) => a.t - b.t);
    ev.forEach((e, i) => { s += `<g class="ev"><line x1="${X(e.t)}" x2="${X(e.t)}" y1="${T}" y2="${H - B + 22 + (i % 2) * 14}"/><text x="${X(e.t) + 4}" y="${H - B + 32 + (i % 2) * 14}">${esc(e.label)}</text></g>`; });
    // checkpoints
    const span = tNow - t0;
    const cks = S.publications.filter((p) => p.date).map((p) => ({ p, t: Run.t(p.date) })).sort((a, b) => a.t - b.t);
    cks.forEach((c, i) => {
      // 曲线单调上升：左上和右下是空的，标签交替放两侧并拉短引线
      const x = X(c.t), y = Y(f(c.t)), ul = i % 2 === 0, lx = ul ? x - 16 : x + 16, ly = ul ? y - 16 : y + 24;
      s += `<g class="ck" data-i="${S.publications.indexOf(c.p)}" style="--t:${((c.t - t0) / span * 2.4).toFixed(2)}"><line x1="${x}" y1="${y}" x2="${ul ? lx + 2 : lx - 2}" y2="${ul ? ly + 4 : ly - 12}" stroke="var(--line2)"/><circle cx="${x}" cy="${y}" r="5"/><text x="${lx}" y="${ly}" text-anchor="${ul ? "end" : "start"}">${esc(c.p.short)}</text></g>`;
    });
    const svg = host.querySelector("svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.innerHTML = s;
    // 曲线描出来
    const main = svg.querySelector(".main"), len = main.getTotalLength();
    const calm = Site.calm;
    if (!calm) { main.style.strokeDasharray = len; main.style.strokeDashoffset = len; main.style.transition = "stroke-dashoffset 2.4s cubic-bezier(.4,0,.2,1)"; }
    const go = () => { host.classList.add("go"); main.style.strokeDashoffset = 0; };
    if (calm || !window.IntersectionObserver) go();
    else { const io = new IntersectionObserver((es) => { if (es[0].isIntersecting) { go(); io.disconnect(); } }, { threshold: 0.35 }); io.observe(host); }
    // 悬停 checkpoint
    svg.querySelectorAll(".ck").forEach((g) => {
      const p = S.publications[g.dataset.i];
      g.addEventListener("mouseenter", () => {
        tip.innerHTML = `${media(p)}<div class="t"><b>${esc(p.short)}</b><span>${p.date} · ${esc(venue(p))}</span></div>`;
        const r = g.querySelector("circle").getBoundingClientRect(), hr = host.getBoundingClientRect();
        tip.style.left = Math.min(hr.width - 250, Math.max(6, r.left - hr.left - 120)) + "px";
        tip.style.top = (r.top - hr.top > hr.height / 2 ? r.top - hr.top - 200 : r.bottom - hr.top + 10) + "px";
        tip.classList.add("on"); onPick && onPick(p, true);
      });
      g.addEventListener("mouseleave", () => { tip.classList.remove("on"); onPick && onPick(p, false); });
      g.addEventListener("click", () => onPick && onPick(p, "go"));
    });
  },
  // 履历写成 config.yaml
  yaml() {
    const { S, esc } = Site;
    const k = (x) => `<span class="k">${x}</span>`, v = (x) => `<span class="s">${esc(x)}</span>`;
    const item = (e, a, b) => `  - ${k(a + ":")} ${v(e.title)}\n    ${k(b + ":")} ${v(e.org)}\n    ${k("period:")} ${v(e.when)}` + (e.detail ? `\n    ${k("note:")} ${v(e.detail)}` : "");
    return `<span class="c"># config.yaml: path so far (full CV linked below)</span>\n${k("education:")}\n${S.education.map((e) => item(e, "degree", "school")).join("\n")}\n${k("experience:")}\n${S.experience.map((e) => item(e, "role", "org")).join("\n")}\n${k("objective:")} ${v("general-purpose robots")}`;
  },
};
