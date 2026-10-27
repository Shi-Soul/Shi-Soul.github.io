/* V12 两页共用：顶部（路径、标签、标签页）和右侧栏（简介、联系方式、项目主页、给 LLM 的 Markdown）。 */
window.Card = {
  head(el, page) {
    const { P, S, catName } = Site;
    const tags = ["robotics", "reinforcement-learning", "embodied-ai"].concat(S.categories.map((c) => c.name.toLowerCase().replace(/\s+/g, "-"))).filter((v, i, a) => a.indexOf(v) === i);
    el.innerHTML = `<div class="wrap">
      <div class="crumb"><span class="av" style="background-image:url(${P.photo})" aria-hidden="true"></span><span>people /</span><b>weiji-xie</b><button class="copy" type="button" id="copyname" title="Copy name">⧉</button></div>
      <div class="tags">${tags.map((t) => `<span>${t}</span>`).join("")}</div>
      <nav class="tabsnav" aria-label="Site"><a href="index.html"${page === "card" ? ' aria-current="page"' : ""}><i class="fa-regular fa-file-lines"></i>Model card</a><a href="cv.html"${page === "cv" ? ' aria-current="page"' : ""}><i class="fa-solid fa-list-check"></i>Training details (CV)</a></nav>
    </div>`;
    el.querySelector("#copyname").addEventListener("click", (e) => { navigator.clipboard && navigator.clipboard.writeText(P.name); e.target.textContent = "✓"; });
  },
  side(el) {
    const { P, S, esc } = Site;
    const spaces = S.publications.filter((p) => p.links.project);
    el.innerHTML = `<div class="box"><div class="who"><div class="img" style="background-image:url(${P.photo})" role="img" aria-label="Photo of Weiji Xie"></div><div><b>${P.name}</b><span>M.S. student, SJTU APEX Lab<br>Research intern, TeleAI<br>${P.location}</span></div></div></div>
      <div class="box"><h4>Use this researcher</h4><div class="contacts">${Site.contacts({ labels: true, emailText: true })}</div><small>${esc(P.contactNote)}</small></div>
      <div class="box spaces"><h4>Spaces using this researcher</h4>${spaces.map((p) => `<a href="${esc(p.links.project)}">${esc(p.short)}<span>${p.date || p.year}</span></a>`).join("")}</div>
      <div class="box"><h4>For agents</h4><button class="btn" type="button" id="md">Copy profile as Markdown</button><small>llms.txt-style summary of this page, generated from the same data.</small></div>`;
    el.querySelector("#md").addEventListener("click", (e) => Site.copyMarkdown(e.target));
  },
  code(pre) {
    pre.querySelector("button").addEventListener("click", (e) => { const c = pre.cloneNode(true); c.querySelector("button").remove(); navigator.clipboard && navigator.clipboard.writeText(c.textContent.trim()); e.target.textContent = "Copied"; });
  },
};
