/*
 * 各版本共用的渲染小工具：联系方式（含微信二维码弹窗）、论文列表、缩略图 + 悬停视频。
 * 版本页只负责排版和样式，内容全部来自 data.js。
 */
(function () {
  "use strict";
  const S = window.SITE, P = S.profile;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const Site = {
    S, P, esc,
    me: (authors) => esc(authors).replace(/Weiji Xie/g, '<b class="me">$&</b>'),
    venue: (p) => (/review/i.test(p.venue) ? "Under review" : `${p.venue} ${p.year}`),
    isTop: (p) => !/review/i.test(p.venue),
    main: () => S.publications.filter((p) => p.tier === "main"),
    others: () => S.publications.filter((p) => p.tier !== "main"),
    rest: (p) => { const i = p.title.indexOf(":"); return i > 0 ? p.title.slice(i + 1).trim() : p.title; },

    // 缩略图：有 image 用图；有 video 时悬停才加载播放；都没有就生成文字占位
    media(p, cls) {
      cls = cls || "";
      if (!p.image) return `<div class="pm pm-ph ${cls}" aria-hidden="true"><span>${esc(p.short)}</span></div>`;
      const v = p.video ? ` data-video="${esc(p.video)}"` : "";
      const badge = p.video ? `<span class="pm-play" aria-hidden="true">▶</span>` : "";
      return `<div class="pm ${cls}"${v}><img src="${esc(p.image)}" alt="${esc(p.short)}" loading="lazy">${badge}</div>`;
    },
    links(p, labels) {
      labels = Object.assign({ project: "Project", pdf: "Paper", code: "Code" }, labels || {});
      return Object.entries(p.links).map(([k, u]) => `<a href="${esc(u)}">${labels[k] || k}</a>`).join("");
    },

    // 联系方式：保留原站全部入口（邮箱、Scholar、GitHub、LinkedIn、X、微信）
    contacts(opts) {
      opts = opts || {};
      const L = P.links, lab = (t) => (opts.labels ? `<span>${t}</span>` : "");
      const items = [
        `<a href="mailto:${P.email}" title="Email"><i class="fa-solid fa-envelope"></i>${lab(opts.emailText ? P.email : "Email")}</a>`,
        `<a href="${L.scholar}" title="Google Scholar"><i class="ai ai-google-scholar"></i>${lab("Scholar")}</a>`,
        `<a href="${L.github}" title="GitHub"><i class="fa-brands fa-github"></i>${lab("GitHub")}</a>`,
        `<a href="${L.linkedin}" title="LinkedIn"><i class="fa-brands fa-linkedin"></i>${lab("LinkedIn")}</a>`,
        `<a href="${L.x}" title="X"><i class="fa-brands fa-x-twitter"></i>${lab("X")}</a>`,
        `<button type="button" data-wechat title="WeChat"><i class="fa-brands fa-weixin"></i>${lab("WeChat")}</button>`,
      ];
      return items.join("");
    },

    init() {
      // 悬停播放
      document.querySelectorAll(".pm[data-video]").forEach((el) => {
        let vid = null;
        const on = () => {
          if (!vid) {
            vid = document.createElement("video");
            Object.assign(vid, { muted: true, loop: true, playsInline: true, preload: "none" });
            vid.src = el.dataset.video;
            el.appendChild(vid);
          }
          el.classList.add("playing");
          const pr = vid.play(); if (pr && pr.catch) pr.catch(() => {});
        };
        const off = () => { el.classList.remove("playing"); if (vid) vid.pause(); };
        const host = el.closest("[data-hover]") || el;
        host.addEventListener("mouseenter", on); host.addEventListener("mouseleave", off);
      });
      // 微信二维码
      let modal = null;
      const close = () => modal && modal.classList.remove("open");
      document.addEventListener("click", (e) => {
        const b = e.target.closest("[data-wechat]");
        if (b) {
          if (!modal) {
            modal = document.createElement("div");
            modal.className = "wx-modal";
            modal.setAttribute("role", "dialog");
            modal.setAttribute("aria-label", "WeChat QR code");
            modal.innerHTML = `<div class="wx-card"><img src="${esc(P.wechatQR)}" alt="WeChat QR code"><p>${esc(P.contactNote)}</p><button type="button" class="wx-close" aria-label="Close">×</button></div>`;
            modal.addEventListener("click", (ev) => { if (ev.target === modal || ev.target.closest(".wx-close")) close(); });
            document.body.appendChild(modal);
          }
          modal.classList.add("open");
          modal.querySelector(".wx-close").focus();
        }
      });
      document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
    },

    // 明暗切换（只记在本浏览器）
    themeToggle(btn) {
      const root = document.documentElement;
      try { const t = localStorage.getItem("wx-theme"); if (t) root.dataset.theme = t; } catch (e) {}
      btn && btn.addEventListener("click", () => {
        const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
        root.dataset.theme = dark ? "light" : "dark";
        try { localStorage.setItem("wx-theme", root.dataset.theme); } catch (e) {}
        window.Demos && Demos.refresh();
      });
    },
  };
  window.Site = Site;
})();
