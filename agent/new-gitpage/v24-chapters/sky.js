/*
 * V24：深色天空、地平线微光、极慢的极光；偶尔有远处的光点沿地平线缓缓划过。
 * 画布分辨率减半再放大，保持安静、省电。
 */
function Sky(canvas) {
  const ctx = canvas.getContext("2d"), calm = window.Site && Site.calm, S = 0.5;
  let W, H, t = 0, travelers = [];
  function resize() { W = Math.ceil(innerWidth * S); H = Math.ceil(innerHeight * S); canvas.width = W; canvas.height = H; }
  resize(); addEventListener("resize", resize);
  function draw() {
    const hy = H * 0.64;
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, "#060912"); g.addColorStop(0.55, "#0b1326"); g.addColorStop(0.64, "#13213f"); g.addColorStop(0.66, "#080c17"); g.addColorStop(1, "#05070d");
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    // 极光：几条缓慢起伏的柔光带
    ctx.globalCompositeOperation = "lighter";
    [[110, 200, 210, 0.05, 0], [140, 120, 230, 0.04, 2.1], [90, 170, 255, 0.035, 4.2]].forEach(([r, gg, b, a, ph]) => {
      for (let k = 0; k < 3; k++) {
        ctx.beginPath();
        for (let x = 0; x <= W; x += 8) {
          const y = hy * (0.42 + 0.1 * k) + Math.sin(x * 0.006 + t * 0.00022 + ph) * 26 + Math.sin(x * 0.013 - t * 0.00031 + ph) * 12;
          x ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.strokeStyle = `rgba(${r},${gg},${b},${a})`; ctx.lineWidth = 34 - k * 8; ctx.stroke();
      }
    });
    // 地平线微光
    const hg = ctx.createRadialGradient(W / 2, hy, 0, W / 2, hy, W * 0.6);
    hg.addColorStop(0, "rgba(150,185,255,.22)"); hg.addColorStop(1, "rgba(150,185,255,0)");
    ctx.fillStyle = hg; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "rgba(190,210,255,.55)"; ctx.fillRect(0, hy, W, 1);
    // 远处缓缓移动的光点
    if (!calm && Math.random() < 0.004 && travelers.length < 3) travelers.push({ x: -10, v: 0.08 + Math.random() * 0.1, y: hy - 2 - Math.random() * 3 });
    travelers = travelers.filter((p) => p.x < W + 10);
    for (const p of travelers) { p.x += p.v; ctx.fillStyle = "rgba(220,235,255,.9)"; ctx.beginPath(); ctx.arc(p.x, p.y, 1.2, 0, 6.283); ctx.fill(); }
    ctx.globalCompositeOperation = "source-over";
  }
  if (calm) return draw();
  (function loop() { t += 16; draw(); requestAnimationFrame(loop); })();
}
