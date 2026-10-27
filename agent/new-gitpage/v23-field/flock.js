/*
 * V23：一群极小的光点（boids）。对齐与聚合的倾向随时间慢慢增强：从无序到缓慢流动的群体。
 * 鼠标附近轻轻避开；Flock.attract(x, y) 让群体缓缓聚向某处（悬停论文时用）。
 */
window.Flock = (function () {
  let target = null;
  function start(canvas, n) {
    const ctx = canvas.getContext("2d"), calm = window.Site && Site.calm;
    let W, H, dpr, t = 0, mouse = null, visible = true;
    const B = [];
    function resize() { dpr = Math.min(2, devicePixelRatio || 1); W = innerWidth; H = innerHeight; canvas.width = W * dpr; canvas.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); }
    resize(); addEventListener("resize", resize);
    for (let i = 0; i < (n || 160); i++) B.push({ x: Math.random() * W, y: Math.random() * H, vx: Math.random() - .5, vy: Math.random() - .5 });
    addEventListener("pointermove", (e) => (mouse = { x: e.clientX, y: e.clientY }));
    document.addEventListener("visibilitychange", () => (visible = !document.hidden));
    function step() {
      t += 1;
      const learn = 1 - Math.exp(-t / 1500);           // 0 → 1，约半分钟
      const wA = 0.012 + 0.05 * learn, wC = 0.0004 + 0.0016 * learn, wS = 0.05, noise = 0.08 * (1 - learn) + 0.012, R = 70, R2 = R * R, maxV = 0.55;
      for (const b of B) {
        let ax = 0, ay = 0, cx = 0, cy = 0, sx = 0, sy = 0, k = 0;
        for (const o of B) {
          if (o === b) continue;
          const dx = o.x - b.x, dy = o.y - b.y, d2 = dx * dx + dy * dy;
          if (d2 < R2) { k++; ax += o.vx; ay += o.vy; cx += dx; cy += dy; if (d2 < 260) { sx -= dx; sy -= dy; } }
        }
        if (k) { b.vx += (ax / k - b.vx) * wA + (cx / k) * wC; b.vy += (ay / k - b.vy) * wA + (cy / k) * wC; }
        b.vx += sx * wS * 0.02 + (Math.random() - .5) * noise; b.vy += sy * wS * 0.02 + (Math.random() - .5) * noise;
        if (target) { b.vx += (target.x - b.x) * 0.00012; b.vy += (target.y - b.y) * 0.00012; }
        if (mouse) { const dx = b.x - mouse.x, dy = b.y - mouse.y, d2 = dx * dx + dy * dy; if (d2 < 9000) { b.vx += dx / Math.sqrt(d2 + 1) * 0.04; b.vy += dy / Math.sqrt(d2 + 1) * 0.04; } }
        const v = Math.hypot(b.vx, b.vy); if (v > maxV) { b.vx *= maxV / v; b.vy *= maxV / v; }
        b.x += b.vx; b.y += b.vy;
        if (b.x < -20) b.x = W + 20; if (b.x > W + 20) b.x = -20; if (b.y < -20) b.y = H + 20; if (b.y > H + 20) b.y = -20;
      }
    }
    function draw() {
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "rgba(170,205,255,.55)";
      for (const b of B) { ctx.beginPath(); ctx.arc(b.x, b.y, 1.15, 0, 6.283); ctx.fill(); }
    }
    if (calm) { for (let i = 0; i < 600; i++) step(); draw(); return; }
    (function loop() { if (visible) { step(); draw(); } requestAnimationFrame(loop); })();
  }
  return { start, attract: (x, y) => (target = x == null ? null : { x, y }) };
})();
