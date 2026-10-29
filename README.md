# Weiji Xie · 谢炜基

个人主页：[shi-soul.github.io](https://shi-soul.github.io/)。

站点使用原生 HTML、CSS 和 JavaScript，无框架、无第三方构建依赖。GitHub Actions 使用 Node.js 22 打包并发布，不需要 Ruby、Jekyll、Python 或 Docker。

## 修改内容

- `data.js`：简介、联系方式、新闻、论文、经历、合作者链接和兴趣。首页与 CV 共用。
- `index.html` / `cv.html`：页面结构。
- `style.css` / `site.css`：页面样式与共用组件样式。
- `site.js`：论文筛选、作者链接、视频悬停播放、微信弹窗等交互。
- `media/`：照片、论文缩略图、机构 logo 和微信二维码。

简介中的 `statement` 支持 `**加粗**`、`==黄色底==`、`==**加粗并加黄色底**==`，只需修改引号中的文本。`affiliation` 中的链接仍使用 HTML。更新内容后修改 `data.js` 顶部的 `lastUpdated`，首页和 CV 页脚会同步显示该日期。

添加论文时复制 `publications` 中的一条，修改字段；`selectedOrder` 控制 Selected 列表顺序。首页上有 GitHub Code 链接的论文会自动显示 Stars。

## 检查与发布

安装 Node.js 22 或更新版本后，在仓库根目录运行（无需 `npm install`）：

```sh
npm test
npm run build
```

`dist/` 是完整的静态站点；首页和 CV 也可直接打开根目录的 HTML 预览。构建脚本会自动生成 `/agent/` 阅读目录和 `sitemap.xml`，并检查主页的本地资源。

推送到 `master` 后，`.github/workflows/deploy.yml` 会测试、打包，再发布到 `gh-pages`。GitHub Pages 保持从 `gh-pages` 根目录发布即可。不要手动编辑 `dist/` 或 `gh-pages`。

## 保留的内容与旧链接

- `/cv/` 跳转到 `/cv.html`；`/publications/` 跳转到首页的 All publications。
- 原草稿入口 `/agent/new-gitpage/home/` 跳转到新主页。
- `agent/` 下的教程、调研和设计草稿保持独立；新建 `agent/<目录名>/index.html` 会自动加入阅读目录。
- 阅读区域不设站内入口，不加入 sitemap；构建时为该区域的所有 HTML 添加 `noindex, nofollow`。robots.txt 允许抓取，以便搜索引擎读取禁止收录标记。这是隐藏入口，不是访问控制；已知准确网址的人仍可访问。
- `agent/new-gitpage/v0-original/` 保存旧主页的静态快照，无需 Jekyll 即可浏览。
- `assets/` 只保留旧草稿仍引用的微信二维码和论文 PDF。
- `Archive/` 是本地资料目录，不参与构建和发布。

旧 Jekyll 源码仍可在 Git 历史的 `f4141a3` 提交及以前找到。旧主题的 MIT 许可保留在 `LICENSE`。
