/* V11 两页共用：图框分区刻度、标题栏、光标坐标（mm）、DETAIL A 尺寸线。 */
window.Drawing = {
  frame(el) {
    const cols = 8, rows = 6;
    let h = "";
    for (let i = 0; i < cols; i++) { const x = `calc(${((i + 0.5) / cols) * 100}% - 3px)`; h += `<span style="left:${x};top:1px">${i + 1}</span><span style="left:${x};bottom:1px">${i + 1}</span>`; }
    for (let j = 0; j < rows; j++) { const y = `calc(${((j + 0.5) / rows) * 100}% - 7px)`; const c = "ABCDEF"[j]; h += `<span style="top:${y};left:3px">${c}</span><span style="top:${y};right:3px">${c}</span>`; }
    el.innerHTML = h;
  },
  titleBlock(el, sheet, title) {
    const d = new Date(), date = d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0");
    const rev = "ABCDEFGHIJ"[Site.S.news.length - 1] || "A";
    el.innerHTML = `<div class="row"><div>Title</div><div class="big">${title}</div></div>
      <div class="row"><div>Subject</div><div>${Site.esc(Site.P.tagline)}</div></div>
      <div class="row"><div>Drawn by</div><div>Weiji Xie</div></div>
      <div class="row"><div>Org.</div><div>SJTU · APEX Lab / TeleAI</div></div>
      <div class="split"><div><b>DATE</b><span>${date}</span></div><div><b>SCALE</b><span>1:1</span></div><div><b>REV</b><span>${rev}</span></div></div>
      <div class="split"><div><b>SHEET</b><span>${sheet}</span></div><div style="grid-column:span 2"><b>CURSOR (MM)</b><span id="xy">X 000.0 · Y 000.0</span></div></div>`;
    // 96 dpi 下 1 mm ≈ 3.78 px
    const xy = el.querySelector("#xy");
    window.addEventListener("pointermove", (e) => { xy.textContent = `X ${((e.pageX) / 3.78).toFixed(1).padStart(5, "0")} · Y ${((e.pageY) / 3.78).toFixed(1).padStart(5, "0")}`; });
  },
  // 照片的尺寸标注：35 × 45
  detail(svg) {
    const L = 26, T = 26, W = 160, H = 206;
    svg.innerHTML = `<g class="dim">
      <line x1="${L}" y1="10" x2="${L + W}" y2="10" style="--len:${W}"/><line x1="${L}" y1="4" x2="${L}" y2="${T - 6}" style="--len:30"/><line x1="${L + W}" y1="4" x2="${L + W}" y2="${T - 6}" style="--len:30"/>
      <path d="M${L + 7} 7 L${L} 10 L${L + 7} 13 M${L + W - 7} 7 L${L + W} 10 L${L + W - 7} 13" style="--len:40"/>
      <line x1="10" y1="${T}" x2="10" y2="${T + H}" style="--len:${H}"/><line x1="4" y1="${T}" x2="${L - 6}" y2="${T}" style="--len:30"/><line x1="4" y1="${T + H}" x2="${L - 6}" y2="${T + H}" style="--len:30"/>
      <path d="M7 ${T + 7} L10 ${T} L13 ${T + 7} M7 ${T + H - 7} L10 ${T + H} L13 ${T + H - 7}" style="--len:40"/>
      </g><text x="${L + W / 2 - 8}" y="7">35</text><text x="-4" y="${T + H / 2}" transform="rotate(-90 -4 ${T + H / 2})" text-anchor="middle">45</text>`;
  },
};
