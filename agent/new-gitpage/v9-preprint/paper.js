/* V9 两页共用：arXiv 侧边戳、参考文献、引用悬浮预览、作者简介。 */
window.Paper = {
  stamp(el) {
    const d = new Date(), yymm = String(d.getFullYear()).slice(2) + String(d.getMonth() + 1).padStart(2, "0");
    const day = d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
    el.textContent = `arXiv:${yymm}.00249v3 [cs.RO] ${day}`;
  },
  num: (p) => Site.S.publications.indexOf(p) + 1,
  cite: (p) => `<a class="cite" href="#ref-${Paper.num(p)}" data-ref="${Paper.num(p)}">[${Paper.num(p)}]</a>`,
  ref(p) {
    const { esc, me } = Site, i = p.title.indexOf(":");
    const venue = /review/i.test(p.venue) ? "under review" : `in <i>Proc. ${esc(p.venue)}</i>`;
    const ln = Object.entries(p.links).map(([k, u]) => `<a href="${esc(u)}">[${k === "project" ? "project" : "pdf"}]</a>`).join(" ");
    return `“${esc(p.title)},” ${me(p.authors)}, ${venue}, ${p.year}.<span class="ln">${ln}</span>`;
  },
  refs(el) {
    el.innerHTML = Site.S.publications.map((p, i) => `<li id="ref-${i + 1}"><span class="n">[${i + 1}]</span><span>${Paper.ref(p)}</span></li>`).join("");
  },
  // 悬停 [n] 显示完整引用
  popovers() {
    const pop = document.createElement("div"); pop.className = "pop"; pop.setAttribute("role", "tooltip"); document.body.appendChild(pop);
    document.addEventListener("mouseover", (e) => {
      const c = e.target.closest(".cite"); if (!c) return;
      const p = Site.S.publications[c.dataset.ref - 1];
      pop.innerHTML = `<b>[${c.dataset.ref}]</b> ${Paper.ref(p)}`;
      const r = c.getBoundingClientRect();
      pop.style.left = Math.max(8, Math.min(window.scrollX + r.left - 20, window.scrollX + document.documentElement.clientWidth - 430)) + "px";
      pop.style.top = window.scrollY + r.bottom + 6 + "px";
      pop.classList.add("on");
    });
    document.addEventListener("mouseout", (e) => { if (e.target.closest(".cite")) pop.classList.remove("on"); });
  },
  contacts(el) {
    el.innerHTML = Site.contacts({ labels: true, emailText: true });
  },
  // IEEE 式作者简介（不用代词）
  bio(el) {
    const P = Site.P;
    el.innerHTML = `<div class="img" style="background-image:url(${P.photo})" role="img" aria-label="Photo of Weiji Xie"></div>
      <p><b>Weiji Xie</b> received the B.Eng. degree in artificial intelligence from Shanghai Jiao Tong University, Shanghai, China, in 2025 (GPA 4.03/4.30, top 5%), and is currently pursuing the M.S. degree in computer science there with the APEX Lab, advised by Prof. Weinan Zhang. Since 2024, also a research intern at TeleAI, China Telecom, working with Dr. Chenjia Bai. Research interests include robot learning, reinforcement learning and embodied AI, toward general-purpose robots. Recipient of the ASC Student Supercomputer Challenge First Prize (2024); reviewer for TPAMI and ICRA.</p>`;
  },
};
