/*
 * V19：页面上唯一会动的东西，一个会学习跟随的光点。
 * 起初探索噪声大、增益小，在页边游走；跟随光标的经验越多，噪声衰减、增益变大，越来越稳。
 */
function Agent(canvas) {
  if (window.Site && Site.calm) return;
  const ctx = canvas.getContext("2d");
  let W, H, dpr, x, y, vx = 0, vy = 0, exp = 0, mouse = null, lastMove = 0, t = 0;
  function resize() {
    dpr = Math.min(2, window.devicePixelRatio || 1); W = innerWidth; H = innerHeight;
    canvas.width = W * dpr; canvas.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (x == null) { x = W * 0.82; y = H * 0.3; }
  }
  resize(); addEventListener("resize", resize);
  addEventListener("pointermove", (e) => { mouse = { x: e.clientX, y: e.clientY }; lastMove = performance.now(); });
  addEventListener("pointerleave", () => (mouse = null));
  const gauss = () => (Math.random() + Math.random() + Math.random() - 1.5) * 1.15;
  (function step(now) {
    t += 1;
    const following = mouse && now - lastMove < 4000;
    // 没在跟随时，目标是页边一个缓慢移动的点
    const tx = following ? mouse.x + 18 : W * 0.86 + Math.sin(t / 260) * 40, ty = following ? mouse.y - 14 : H * 0.32 + Math.cos(t / 330) * 60;
    if (following) exp += 0.016;
    const learn = 1 - Math.exp(-exp / 6);
    const k = 0.0012 + 0.016 * learn, noise = 0.9 * (1 - learn) + 0.04;
    vx = (vx + (tx - x) * k + gauss() * noise) * 0.9;
    vy = (vy + (ty - y) * k + gauss() * noise) * 0.9;
    x += vx; y += vy;
    ctx.clearRect(0, 0, W, H);
    const g = ctx.createRadialGradient(x, y, 0, x, y, 34);
    g.addColorStop(0, "rgba(176,120,58,.28)"); g.addColorStop(1, "rgba(176,120,58,0)");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 34, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "rgba(140,92,40,.85)"; ctx.beginPath(); ctx.arc(x, y, 2.6, 0, Math.PI * 2); ctx.fill();
    requestAnimationFrame(step);
  })(performance.now());
}
