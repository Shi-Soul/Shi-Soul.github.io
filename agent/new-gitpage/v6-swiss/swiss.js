/* V6 的小动效：名字逐字升起、红点光标、分隔线滚动时画出。 */
window.Swiss = {
  name(h1) {
    let k = 0;
    const line = (t, last) => `<span class="ln">${[...t].map((c) => `<span class="ch" style="--k:${k++}">${c}</span>`).join("")}${last ? `<span class="ch red" style="--k:${k++}">.</span>` : ""}</span>`;
    h1.innerHTML = line("Weiji") + line("Xie", true);
  },
  cursor(dot) {
    if (window.Site && Site.calm) return;
    let x = -100, y = -100, cx = x, cy = y;
    window.addEventListener("pointermove", (e) => { x = e.clientX; y = e.clientY; });
    document.addEventListener("pointerover", (e) => dot.classList.toggle("big", !!e.target.closest("a, button, .pub")));
    (function loop() { cx += (x - cx) * 0.18; cy += (y - cy) * 0.18; dot.style.transform = `translate(${cx}px, ${cy}px)`; requestAnimationFrame(loop); })();
  },
  rules() {
    const els = document.querySelectorAll("[data-rule]");
    if (!window.IntersectionObserver || (window.Site && Site.calm)) return els.forEach((e) => e.classList.add("in"));
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }));
    els.forEach((e) => io.observe(e));
  },
};
