/* 工牌的弹簧摆：可以拖拽甩动，鼠标移动和页面滚动也会让它晃起来。 */
function Badge(lanyard, badge) {
  if (window.Site && Site.calm) return;
  let th = 0, om = 0, drag = null, last = performance.now(), lastY = window.scrollY;
  const K = 38, C = 3.2, MAX = 0.9;
  badge.addEventListener("pointerdown", (e) => {
    const r = lanyard.getBoundingClientRect();
    drag = { cx: r.left + r.width / 2, top: r.top, x: e.clientX, th0: th };
    badge.setPointerCapture(e.pointerId);
  });
  badge.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const len = Math.max(200, e.clientY - drag.top);
    const target = Math.max(-MAX, Math.min(MAX, Math.atan2(drag.cx - e.clientX, len)));
    om = (target - th) * 30; th = target;
  });
  const up = () => (drag = null);
  badge.addEventListener("pointerup", up); badge.addEventListener("pointercancel", up);
  window.addEventListener("pointermove", (e) => {
    if (drag) return;
    const r = badge.getBoundingClientRect(), near = Math.max(0, 1 - Math.abs(e.clientY - (r.top + r.height / 2)) / 500);
    om -= (e.movementX || 0) * 0.0025 * near; // 往右划，工牌往右摆（rotate 正角是向左）
  });
  window.addEventListener("scroll", () => { const dy = window.scrollY - lastY; lastY = window.scrollY; om += dy * 0.004; }, { passive: true });
  (function step(now) {
    const dt = Math.min(0.033, (now - last) / 1000); last = now;
    if (!drag) {
      const idle = Math.sin(now / 1400) * 0.012;
      om += (-K * (th - idle) - C * om) * dt;
      th = Math.max(-MAX, Math.min(MAX, th + om * dt));
    }
    lanyard.style.transform = `rotate(${th}rad)`;
    badge.style.transform = `perspective(800px) rotateY(${om * 4}deg)`;
    requestAnimationFrame(step);
  })(last);
}
