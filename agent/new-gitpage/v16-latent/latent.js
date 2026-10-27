/* V16：研究的“嵌入空间”。分类中心排成一圈，论文位置 = 所属分类中心的平均 + 确定性抖动；
   加载时从随机位置收敛（像 t-SNE 迭代），之后静止。 */
window.Latent = {
  color: (i) => `var(--c${i % 5})`,
  catIndex: (id) => Site.S.categories.findIndex((c) => c.id === id),
  // 每个分类周围的淡色关键词（只在 V16 用，加论文不需要改这里）
  keywords: {
    tracking: ["motion imitation", "kung fu", "text → motion", "whole-body"],
    loco: ["balance", "narrow terrain", "quadruped", "ZMP"],
    s2r: ["domain gap", "lifelong", "real-world data"],
    gen: ["autoregressive", "trajectories", "language"],
    marl: ["self-play", "Nash", "offline data"],
  },
  hash(s) { let h = 2166136261; for (const c of s) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return ((h >>> 0) % 1000) / 1000; },
  draw(host, tip, onCat, onPick) {
    const { S, esc, venue, media } = Site;
    const W = 720, H = 470, cx = W / 2, cy = H / 2 + 4, RX = 205, RY = 145;
    const cats = S.categories.map((c, i) => { const a = -Math.PI / 2 + i * 2 * Math.PI / S.categories.length; return { ...c, i, a, x: cx + RX * Math.cos(a), y: cy + RY * Math.sin(a) }; });
    const pts = S.publications.map((p) => {
      const cs = (p.cats || []).map(Latent.catIndex).filter((i) => i >= 0);
      const mx = cs.reduce((s, i) => s + cats[i].x, 0) / cs.length, my = cs.reduce((s, i) => s + cats[i].y, 0) / cs.length;
      const h = Latent.hash(p.title), h2 = Latent.hash(p.short + "y");
      return { p, c: cs[0], x: mx + (h - .5) * 40, y: my + (h2 - .5) * 30, sx: 40 + h2 * (W - 80), sy: 40 + h * (H - 80) };
    });
    let s = "";
    // 每个分类一个固定的圈；标签和关键词放在圈的外侧（远离中心）
    cats.forEach((c) => {
      const ox = Math.cos(c.a), oy = Math.sin(c.a);
      s += `<g class="grp" data-c="${c.id}"><circle class="hull" cx="${c.x}" cy="${c.y}" r="54" fill="${Latent.color(c.i)}" stroke="${Latent.color(c.i)}"/>`;
      (Latent.keywords[c.id] || []).forEach((k, j, all) => { const a = c.a + Math.PI + (j - (all.length - 1) / 2) * 0.42; s += `<text class="kw" x="${c.x + Math.cos(a) * 30}" y="${c.y + Math.sin(a) * 30 + (j - (all.length - 1) / 2) * 13 + 4}" text-anchor="middle">${esc(k)}</text>`; });
      s += `<text class="clab" x="${c.x + ox * 66}" y="${c.y + oy * 66 + 4}" text-anchor="${ox > 0.35 ? "start" : ox < -0.35 ? "end" : "middle"}" fill="${Latent.color(c.i)}">${esc(c.name)}</text></g>`;
    });
    pts.forEach((q, k) => { s += `<g class="pt" data-k="${k}" transform="translate(${q.sx},${q.sy})"><circle r="6" fill="${Latent.color(q.c)}"/><text x="10" y="4">${esc(q.p.short)}</text></g>`; });
    const svg = host.querySelector("svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.innerHTML = s;
    host.querySelector(".legend").innerHTML = cats.map((c) => `<span data-c="${c.id}"><i style="background:${Latent.color(c.i)}"></i>${esc(c.name)}</span>`).join("");
    // 收敛动画
    const it = host.querySelector("#iter"), G = [...svg.querySelectorAll(".pt")];
    const place = (u) => G.forEach((g, k) => { const q = pts[k], e = 1 - Math.pow(1 - u, 3); g.setAttribute("transform", `translate(${q.sx + (q.x - q.sx) * e},${q.sy + (q.y - q.sy) * e})`); });
    const finish = () => { place(1); it.textContent = "1000"; host.classList.add("ready"); }; // 收敛后再显示簇
    if (Site.calm) finish();
    else {
      const io = new IntersectionObserver((es) => { if (!es[0].isIntersecting) return; io.disconnect();
        const t0 = performance.now(), D = 1800;
        (function step(now) { const u = Math.min(1, (now - t0) / D); place(u); it.textContent = String(Math.round(u * 1000)); if (u < 1) requestAnimationFrame(step); else finish(); })(t0);
      }, { threshold: 0.3 });
      io.observe(host);
    }
    // 交互
    G.forEach((g, k) => {
      const q = pts[k];
      g.addEventListener("mouseenter", () => {
        tip.innerHTML = `${media(q.p)}<div class="t"><b>${esc(q.p.short)}</b><span>${esc(venue(q.p))}</span></div>`;
        const r = g.getBoundingClientRect(), hr = host.getBoundingClientRect();
        tip.style.left = Math.min(hr.width - 240, Math.max(6, r.left - hr.left - 110)) + "px";
        tip.style.top = (r.top - hr.top > hr.height / 2 ? r.top - hr.top - 190 : r.bottom - hr.top + 8) + "px";
        tip.classList.add("on"); onPick(q.p, true);
      });
      g.addEventListener("mouseleave", () => { tip.classList.remove("on"); onPick(q.p, false); });
      g.addEventListener("click", () => onCat(S.categories[q.c].id, q.p));
    });
    svg.querySelectorAll(".grp").forEach((g) => g.addEventListener("click", () => onCat(g.dataset.c)));
    host.querySelectorAll(".legend span").forEach((l) => l.addEventListener("click", () => onCat(l.dataset.c)));
    return {
      highlight(f) {
        svg.querySelectorAll(".grp").forEach((g) => { g.classList.toggle("on", g.dataset.c === f); g.classList.toggle("dim", !["all", "selected", g.dataset.c].includes(f)); });
        G.forEach((g, k) => { const p = pts[k].p, show = f === "all" || (f === "selected" ? p.selected : (p.cats || []).includes(f)); g.classList.toggle("dim", !show); });
      },
      hot(p, on) { G.forEach((g, k) => g.classList.toggle("hot", on && pts[k].p === p)); },
    };
  },
};
