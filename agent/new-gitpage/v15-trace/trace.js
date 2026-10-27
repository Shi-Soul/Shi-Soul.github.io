/* V15 两页共用：按顺序执行 tool call（滚动到才执行），显示耗时，最后标记任务完成。 */
window.Trace = {
  sig: (fn, args) => `<span class="fn">${fn}</span>(${Object.entries(args || {}).map(([k, v]) => `${k}=<span class="s">${typeof v === "string" ? `"${Site.esc(v)}"` : v}</span>`).join(", ")})`,
  run(status) {
    const steps = [...document.querySelectorAll(".step[data-run]")];
    let left = steps.length;
    const exec = (s) => new Promise((done) => {
      if (s.dataset.ran) return done(); s.dataset.ran = "1";
      status.className = "status run"; status.lastChild.textContent = " running";
      s.classList.add("running");
      const ms = Site.calm ? 0 : 300 + Math.round(Math.random() * 500);
      setTimeout(() => {
        s.classList.remove("running"); s.classList.add("done");
        s.querySelector(".dur").textContent = (ms / 1000 || 0.1).toFixed(1) + "s";
        if (--left === 0) { status.className = "status ok"; status.lastChild.textContent = ` completed · ${steps.length} steps`; }
        else { status.className = "status"; status.lastChild.textContent = " waiting"; }
        done();
      }, ms);
    });
    if (Site.calm || !window.IntersectionObserver) return steps.forEach(exec);
    let q = Promise.resolve();
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (!e.isIntersecting) return; io.unobserve(e.target);
      const i = steps.indexOf(e.target);
      steps.slice(0, i + 1).forEach((s) => { q = q.then(() => exec(s)); });
    }), { rootMargin: "0px 0px -12% 0px" });
    steps.forEach((s) => io.observe(s));
  },
};
