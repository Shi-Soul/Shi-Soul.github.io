/* 草稿阶段的版本切换器：右下角的小浮条，方便对比。定稿后删掉这个 <script> 即可。 */
(function () {
  if (window.self !== window.top) return; // 在草稿总览页的预览 iframe 里不显示
  const versions = [["v1-console", "V1"], ["v2-editorial", "V2"], ["v3-atlas", "V3"], ["v4-notebook", "V4"], ["v5-badge", "V5"], ["v6-swiss", "V6"], ["v7-manpage", "V7"], ["v8-flow", "V8"], ["v9-preprint", "V9"], ["v10-run", "V10"], ["v11-drawing", "V11"], ["v12-modelcard", "V12"], ["v13-notebook", "V13"], ["v14-answer", "V14"], ["v15-trace", "V15"], ["v16-latent", "V16"], ["v17-loop", "V17"], ["v18-selfimprove", "V18"], ["v19-letter", "V19"], ["v20-gallery", "V20"], ["v21-day", "V21"], ["v22-mosaic", "V22"], ["v23-field", "V23"], ["v24-chapters", "V24"]];
  const here = location.pathname.split("/").filter(Boolean).slice(-2).find((s) => /^v\d-/.test(s)) || "";
  const bar = document.createElement("nav");
  bar.setAttribute("aria-label", "Draft versions");
  bar.innerHTML = `<a href="../index.html" title="All drafts">drafts</a>` +
    versions.map(([dir, n]) => `<a href="../${dir}/index.html"${dir === here ? ' aria-current="page"' : ""}>${n}</a>`).join("");
  bar.style.cssText = "position:fixed;right:14px;bottom:14px;z-index:9999;display:flex;gap:2px;max-width:calc(100vw - 28px);overflow-x:auto;scrollbar-width:none;padding:4px;border-radius:999px;background:rgba(20,20,20,.86);backdrop-filter:blur(6px);font:500 12px/1 ui-monospace,SFMono-Regular,Menlo,monospace;box-shadow:0 6px 20px rgba(0,0,0,.25)";
  bar.querySelectorAll("a").forEach((a) => {
    const cur = a.hasAttribute("aria-current");
    a.style.cssText = `flex:none;color:${cur ? "#111" : "#ddd"};background:${cur ? "#fff" : "transparent"};padding:7px 10px;border-radius:999px;text-decoration:none`;
  });
  document.body.appendChild(bar);
})();
