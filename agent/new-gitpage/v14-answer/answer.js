/* V14 两页共用：打出问题、来源卡片依次出现、回答段落逐段淡入、[n] 引用悬浮预览。 */
window.Answer = {
  num: (p) => Site.S.publications.indexOf(p) + 1,
  cite: (p) => `<a class="cite" href="${Site.esc(p.links.project || p.links.pdf)}" data-ref="${Answer.num(p)}">${Answer.num(p)}</a>`,
  sources(el) {
    const { S, esc, venue, media } = Site;
    el.innerHTML = S.publications.map((p, i) => `<a class="src" style="--k:${i}" href="${esc(p.links.project || p.links.pdf)}">${media(p)}<div class="t"><b><sup>${i + 1}</sup>${esc(p.short)}</b><span>${esc(venue(p))}</span></div></a>`).join("");
  },
  // 依次：打字 → 来源 → 段落
  play(qEl, query, srcWrap, ans) {
    const parts = [...ans.children];
    const show = () => { srcWrap.classList.add("ready"); parts.forEach((p, i) => setTimeout(() => p.classList.add("on"), Site.calm ? 0 : 350 + i * 160)); };
    if (Site.calm) { qEl.textContent = query; return show(); }
    let i = 0; qEl.innerHTML = '<span></span><i class="cur"></i>';
    const t = qEl.firstChild;
    (function type() { t.textContent = query.slice(0, ++i); if (i < query.length) setTimeout(type, 28); else setTimeout(show, 250); })();
  },
  popovers() {
    const { S, esc, venue, media } = Site;
    const pop = document.createElement("div"); pop.className = "pop"; pop.setAttribute("role", "tooltip"); document.body.appendChild(pop);
    document.addEventListener("mouseover", (e) => {
      const c = e.target.closest(".cite"); if (!c) return;
      const p = S.publications[c.dataset.ref - 1];
      pop.innerHTML = `${media(p)}<div class="t"><b>${esc(p.short)}</b><span>${esc(venue(p))}</span><div>${esc(p.note)}</div></div>`;
      const r = c.getBoundingClientRect();
      pop.style.left = Math.max(8, Math.min(window.scrollX + r.left - 120, window.scrollX + document.documentElement.clientWidth - 270)) + "px";
      pop.style.top = window.scrollY + r.bottom + 8 + "px";
      pop.classList.add("on");
    });
    document.addEventListener("mouseout", (e) => { if (e.target.closest(".cite")) pop.classList.remove("on"); });
  },
};
