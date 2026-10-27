/* V18 两页共用：给每条记录算一个稳定的“提交哈希”，并把论文和履历按日期排成两条泳道。 */
window.Git = {
  hash(s) { let h = 2166136261; for (const c of s) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return (h >>> 0).toString(16).padStart(8, "0").slice(0, 7); },
  key: (d) => String(d).replace(/\s.*$/, "").replace(/[^\d.]/g, "").padEnd(7, "0"),
  // 论文（research 泳道）+ 履历起点（career 泳道），按时间倒序
  rows(f) {
    const { S } = Site;
    const papers = Site.filter(f).map((p) => ({ t: "r", d: p.date || String(p.year), p }));
    const path = S.education.concat(S.experience).map((e) => ({ t: "p", d: e.when.split(/\s*[–-]\s*/)[0], e }));
    return papers.concat(path).sort((a, b) => Git.key(b.d).localeCompare(Git.key(a.d)));
  },
};
