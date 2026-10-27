/*
 * V20：几只小 agent 在背景里画极淡的墨线。
 * 一开始转向噪声大（线条抖动），之后逐渐学会顺着一个平滑的流场走（线条变流畅）；旧墨迹会慢慢褪去。
 */
function Ink(canvas) {
  if (window.Site && Site.calm) return;
  const ctx = canvas.getContext("2d");
  let W, H, dpr, t = 0, visible = true;
  const N = 7, ag = [];
  function resize() {
    const r = canvas.getBoundingClientRect(); dpr = Math.min(2, devicePixelRatio || 1);
    W = r.width; H = r.height; canvas.width = W * dpr; canvas.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize(); addEventListener("resize", resize);
  for (let i = 0; i < N; i++) ag.push({ x: Math.random() * W, y: Math.random() * H, a: Math.random() * 6.28 });
  const field = (x, y) => Math.sin(x * 0.004 + t * 0.0004) * 1.4 + Math.cos(y * 0.005 - t * 0.0003) * 1.4;
  document.addEventListener("visibilitychange", () => (visible = !document.hidden));
  (function step() {
    if (visible) {
      t += 1;
      const learn = 1 - Math.exp(-t / 900);                 // 学习进度：0 → 1
      const noise = 0.5 * (1 - learn) + 0.02, pull = 0.01 + 0.07 * learn;
      if (t % 5 === 0) { ctx.fillStyle = "rgba(246,246,243,0.035)"; ctx.fillRect(0, 0, W, H); }
      ctx.strokeStyle = "rgba(47,93,80,0.13)"; ctx.lineWidth = 1;
      ctx.beginPath();
      for (const p of ag) {
        let d = field(p.x, p.y) - p.a; d = Math.atan2(Math.sin(d), Math.cos(d));
        p.a += d * pull + (Math.random() - 0.5) * noise;
        const nx = p.x + Math.cos(p.a) * 0.9, ny = p.y + Math.sin(p.a) * 0.9;
        ctx.moveTo(p.x, p.y); ctx.lineTo(nx, ny);
        p.x = nx; p.y = ny;
        if (p.x < -10 || p.x > W + 10 || p.y < -10 || p.y > H + 10) { p.x = Math.random() * W; p.y = Math.random() * H; }
      }
      ctx.stroke();
    }
    requestAnimationFrame(step);
  })();
}
