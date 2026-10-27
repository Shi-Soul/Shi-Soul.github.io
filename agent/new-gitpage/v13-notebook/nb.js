/* V13 两页共用：滚动到单元格时“执行”它（In [ ] → In [*] → In [n]，内核 idle/busy），以及几个输出渲染器。 */
window.NB = {
  count: 0,
  run(kernel) {
    const cells = [...document.querySelectorAll(".cell[data-code]")];
    const exec = (c) => {
      if (c.dataset.ran) return; c.dataset.ran = "1";
      const p = c.querySelector(".prompt"); p.textContent = "In [*]:"; c.classList.add("active"); kernel.classList.add("busy"); kernel.lastChild.textContent = " busy";
      setTimeout(() => {
        NB.count += 1; p.textContent = `In [${NB.count}]:`;
        const o = c.querySelector(".prompt.o"); if (o) o.textContent = `Out[${NB.count}]:`;
        c.classList.add("done"); c.classList.remove("active"); kernel.classList.remove("busy"); kernel.lastChild.textContent = " idle";
      }, Site.calm ? 0 : 380 + Math.random() * 300);
    };
    if (Site.calm || !window.IntersectionObserver) return cells.forEach(exec);
    // 按顺序执行：前面的没跑完不跑后面的
    let q = Promise.resolve();
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (!e.isIntersecting) return; io.unobserve(e.target);
      const i = cells.indexOf(e.target);
      q = q.then(() => Promise.all(cells.slice(0, i + 1).filter((c) => !c.dataset.ran).map((c, k) => new Promise((r) => setTimeout(() => { exec(c); setTimeout(r, 450); }, k * 500)))));
    }), { rootMargin: "0px 0px -15% 0px" });
    cells.forEach((c) => io.observe(c));
  },
  df(cols, rows, idx) {
    const { esc } = Site;
    return `<table class="df"><thead><tr><th></th>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${rows.map((r, i) => `<tr><th class="idx">${idx ? idx[i] : i}</th>${r.map((v, j) => `<td${j === 0 ? ' class="t"' : ""}>${v}</td>`).join("")}</tr>`).join("")}</tbody></table><div class="df-shape">${rows.length} rows × ${cols.length} columns</div>`;
  },
  hl(src) { // 极简高亮：每行拆成代码与注释两段，只给代码段上色
    const code = (t) => t
      .replace(/(&quot;.*?&quot;)/g, '<span class="s">$1</span>')
      .replace(/\b(from|import|as|for|in|print|True|False|None)\b(?![^<]*<\/span>)/g, '<span class="k">$1</span>')
      .replace(/\.(\w+)\(/g, '.<span class="f">$1</span>(')
      .replace(/=(\d+)\b/g, '=<span class="n">$1</span>');
    return Site.esc(src).split("\n").map((line) => {
      const i = line.indexOf("#");
      return i < 0 ? code(line) : code(line.slice(0, i)) + `<span class="c">${line.slice(i)}</span>`;
    }).join("\n");
  },
  fill() {
    document.querySelectorAll(".cell[data-code]").forEach((c) => {
      const src = c.querySelector(".src"); src.innerHTML = NB.hl(src.textContent.trim());
    });
  },
};
