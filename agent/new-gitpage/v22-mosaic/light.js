/*
 * V22：柔和的光场。几团淡彩色光在低分辨率画布上缓慢游移（CSS 再模糊），其中一团带一点“注意力”，
 * 会慢慢移向鼠标。砖块也会被鼠标轻轻吸引。
 */
function Light(canvas) {
  const ctx = canvas.getContext("2d"), calm = window.Site && Site.calm, S = 0.18;
  let W, H, mx = null, my = null, t = 0;
  const blobs = [
    { c: [255, 210, 186], x: 0.2, y: 0.25, r: 0.42, sp: 0.00031, ph: 0 },
    { c: [196, 220, 255], x: 0.75, y: 0.3, r: 0.46, sp: 0.00024, ph: 2 },
    { c: [204, 240, 222], x: 0.6, y: 0.8, r: 0.4, sp: 0.00028, ph: 4 },
    { c: [226, 214, 255], x: 0.3, y: 0.75, r: 0.38, sp: 0.00035, ph: 1, follow: true },
  ];
  function resize() { W = Math.ceil(innerWidth * S); H = Math.ceil(innerHeight * S); canvas.width = W; canvas.height = H; }
  resize(); addEventListener("resize", resize);
  addEventListener("pointermove", (e) => { mx = e.clientX / innerWidth; my = e.clientY / innerHeight; });
  function draw() {
    ctx.fillStyle = "#fbfaf8"; ctx.fillRect(0, 0, W, H);
    for (const b of blobs) {
      let x = b.x + Math.sin(t * b.sp + b.ph) * 0.12, y = b.y + Math.cos(t * b.sp * 1.3 + b.ph) * 0.1;
      if (b.follow && mx != null) { b.fx = (b.fx ?? x) + (mx - (b.fx ?? x)) * 0.012; b.fy = (b.fy ?? y) + (my - (b.fy ?? y)) * 0.012; x = b.fx; y = b.fy; }
      const R = b.r * Math.max(W, H), g = ctx.createRadialGradient(x * W, y * H, 0, x * W, y * H, R);
      g.addColorStop(0, `rgba(${b.c},.95)`); g.addColorStop(1, `rgba(${b.c},0)`);
      ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    }
  }
  if (calm) return draw();
  (function loop() { t += 16; draw(); requestAnimationFrame(loop); })();
}
function Magnet(root) {
  if (window.Site && Site.calm) return;
  root.addEventListener("pointermove", (e) => {
    const tile = e.target.closest(".t:not(.foot)"); if (!tile) return;
    const r = tile.getBoundingClientRect(), dx = (e.clientX - r.left) / r.width - 0.5, dy = (e.clientY - r.top) / r.height - 0.5;
    tile.style.transform = `translate(${dx * 5}px, ${dy * 5}px)`;
  });
  root.addEventListener("pointerout", (e) => { const tile = e.target.closest(".t"); if (tile && !tile.contains(e.relatedTarget)) tile.style.transform = ""; });
}
