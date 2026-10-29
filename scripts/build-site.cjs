const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assert = require("node:assert/strict");

const root = path.resolve(__dirname, "..");
const output = path.join(root, "dist");
assert.equal(path.relative(root, output), "dist");
if (fs.existsSync(output)) assert(!fs.lstatSync(output).isSymbolicLink(), "dist must not be a symlink");
fs.rmSync(output, { recursive: true, force: true });
fs.mkdirSync(output);

// Publish site files and downloadable reading notes, not repository configuration.
for (const file of ["index.html", "cv.html", "data.js", "site.js", "site.css", "style.css", "robots.txt", ".nojekyll", "404.html"]) {
  fs.copyFileSync(path.join(root, file), path.join(output, file));
}
for (const dir of ["media", "cv", "publications", "agent"]) {
  fs.cpSync(path.join(root, dir), path.join(output, dir), {
    recursive: true,
    filter: (source) => !/\.(?:py|cjs)$/i.test(source),
  });
}

const escape = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const entries = fs.readdirSync(path.join(root, "agent"), { withFileTypes: true })
  .filter((entry) => entry.isDirectory() && fs.existsSync(path.join(root, "agent", entry.name, "index.html")))
  .sort((a, b) => a.name.localeCompare(b.name))
  .map((entry) => {
    const html = fs.readFileSync(path.join(root, "agent", entry.name, "index.html"), "utf8");
    return {
      url: `/agent/${entry.name}/`,
      title: html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1].trim() || entry.name,
      description: html.match(/<meta\s+name="description"\s+content="([^"]*)"/i)?.[1] || "",
    };
  });
const cards = entries.map((entry, index) => `<li><a class="entry" href="${entry.url}" aria-labelledby="entry-${index + 1}">
  <span class="number" aria-hidden="true">${String(index + 1).padStart(2, "0")}</span>
  <div><h3 id="entry-${index + 1}">${escape(entry.title)}</h3><p class="description">${escape(entry.description)}</p><span class="path">${entry.url}</span></div>
  <span class="arrow" aria-hidden="true">→</span></a></li>`).join("\n");
const directoryFile = path.join(output, "agent/index.html");
let directory = fs.readFileSync(directoryFile, "utf8");
directory = directory.replace(/<span class="count">[\s\S]*?<\/span>/, `<span class="count">共 ${entries.length} 篇</span>`)
  .replace(/<ol class="entries">[\s\S]*?<\/ol>/, `<ol class="entries">${cards}</ol>`);
assert(!directory.includes("{%") && !directory.includes("{{"), "Unrendered template in agent directory");
fs.writeFileSync(directoryFile, directory);

// Keep the reading area directly accessible but out of search results.
// Do not block crawling in robots.txt: crawlers need to read these noindex tags.
const readingRoot = path.join(output, "agent");
for (const file of fs.readdirSync(readingRoot, { recursive: true }).filter((file) => /\.html$/i.test(file))) {
  const target = path.join(readingRoot, file);
  const html = fs.readFileSync(target, "utf8")
    .replace(/<meta\b(?=[^>]*\bname=["']robots["'])[^>]*>/gi, "")
    .replace(/<head\b[^>]*>/i, '$&<meta name="robots" content="noindex, nofollow">');
  fs.writeFileSync(target, html);
}

const context = { window: {} };
vm.runInNewContext(fs.readFileSync(path.join(root, "data.js"), "utf8"), context);
const site = context.window.SITE;
const origin = "https://shi-soul.github.io";
const urls = ["/", "/cv.html"];
fs.writeFileSync(path.join(output, "sitemap.xml"), `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${urls.map((url) => `<url><loc>${origin}${url}</loc><lastmod>${site.lastUpdated}</lastmod></url>`).join("")}</urlset>\n`);

for (const ref of [site.profile.photo, site.profile.wechatQR, ...site.publications.map((p) => p.image), ...site.education.concat(site.experience).map((r) => r.logo)].filter(Boolean)) {
  assert(fs.existsSync(path.join(output, ref)), `Missing published media: ${ref}`);
}
for (const file of ["index.html", "cv.html"]) {
  const html = fs.readFileSync(path.join(output, file), "utf8");
  for (const [, script] of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) new vm.Script(script);
  const markup = html.replace(/(<script\b[^>]*>)[\s\S]*?<\/script>/gi, "$1</script>");
  for (const [, ref] of markup.matchAll(/(?:src|href)="([^"]+)"/g)) {
    if (/^(?:https?:|mailto:|#)/.test(ref)) continue;
    assert(fs.existsSync(path.resolve(output, ref.split("#")[0])), `Missing ${file} resource: ${ref}`);
  }
}
console.log(`Static site ready: ${output}; ${entries.length} reading pages; ${site.publications.length} papers.`);
