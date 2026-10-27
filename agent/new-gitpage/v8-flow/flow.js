/* V8 的流场：粒子沿缓慢变化的向量场流动，鼠标附近形成涡旋。 */
function Flow(canvas, opts) {
  opts = opts || {};
  const ctx = canvas.getContext("2d");
  const calm = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const colors = ["70,224,196", "139,123,255", "255,143,177"];
  let W = 0, H = 0, dpr = 1, parts = [], mouse = { x: -1e4, y: -1e4 }, visible = true, t = 0;

  function resize() {
    const r = canvas.getBoundingClientRect();
    dpr = Math.min(2, window.devicePixelRatio || 1);
    W = r.width; H = r.height;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0a0d16"; ctx.fillRect(0, 0, W, H);
    const n = Math.min(opts.max || 1400, Math.round((W * H) / 900));
    parts = Array.from({ length: n }, spawn);
  }
  function spawn() {
    return { x: Math.random() * W, y: Math.random() * H, life: 60 + Math.random() * 240, c: colors[(Math.random() * colors.length) | 0], s: 0.6 + Math.random() * 1.2 };
  }
  // 几个正弦叠加出平滑、随时间缓慢变化的角度场
  const field = (x, y) => Math.sin(x * 0.0021 + t * 0.00031) * 1.7 + Math.cos(y * 0.0027 - t * 0.00023) * 1.7 + Math.sin((x + y) * 0.0011 + t * 0.00017) * 1.2;

  function step() {
    t += 16;
    ctx.fillStyle = "rgba(10,13,22,0.08)";
    ctx.fillRect(0, 0, W, H);
    ctx.lineWidth = 1.1;
    for (const p of parts) {
      const a = field(p.x, p.y);
      let vx = Math.cos(a) * p.s, vy = Math.sin(a) * p.s;
      const dx = p.x - mouse.x, dy = p.y - mouse.y, d2 = dx * dx + dy * dy;
      if (d2 < 170 * 170) {
        const d = Math.sqrt(d2) + 1, f = (1 - d / 170) * 3.2;
        vx += (-dy / d) * f + (dx / d) * f * 0.35;
        vy += (dx / d) * f + (dy / d) * f * 0.35;
      }
      const nx = p.x + vx, ny = p.y + vy;
      ctx.strokeStyle = `rgba(${p.c},${0.25 + Math.min(0.55, Math.hypot(vx, vy) * 0.18)})`;
      ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(nx, ny); ctx.stroke();
      p.x = nx; p.y = ny;
      if (--p.life < 0 || p.x < -5 || p.x > W + 5 || p.y < -5 || p.y > H + 5) Object.assign(p, spawn());
    }
  }
  function loop() { if (visible) step(); requestAnimationFrame(loop); }

  resize();
  window.addEventListener("resize", resize);
  window.addEventListener("pointermove", (e) => { const r = canvas.getBoundingClientRect(); mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top; });
  window.addEventListener("pointerleave", () => { mouse.x = mouse.y = -1e4; });
  if (window.IntersectionObserver) new IntersectionObserver((es) => (visible = es[0].isIntersecting)).observe(canvas);
  if (calm) { for (let i = 0; i < 160; i++) step(); } else loop();
}

// 卡片聚光：把指针位置写进 CSS 变量
function Spotlight(root) {
  root.addEventListener("pointermove", (e) => {
    const c = e.target.closest(".pub"); if (!c) return;
    const r = c.getBoundingClientRect();
    c.style.setProperty("--mx", e.clientX - r.left + "px"); c.style.setProperty("--my", e.clientY - r.top + "px");
  });
}
