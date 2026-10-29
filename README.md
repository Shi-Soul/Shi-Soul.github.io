# Weiji Xie · 谢炜基

个人主页：[shi-soul.github.io](https://shi-soul.github.io/)。

站点使用原生 HTML、CSS 和 JavaScript，无框架、无第三方构建依赖。预览首页和 CV 只需浏览器；打包发布只需 Node.js 22 或更新版本，无需 npm、Ruby、Jekyll、Python 或 Docker。

## 修改内容

- `data.js`：简介、联系方式、新闻、论文、经历、合作者链接和兴趣。首页与 CV 共用。
- `index.html` / `cv.html`：页面结构。
- `style.css` / `site.css`：页面样式与共用组件样式。
- `site.js`：论文筛选、作者链接、视频悬停播放、微信弹窗等交互。
- `media/`：统一存放照片、论文缩略图、机构 logo、微信二维码和 PDF。
- `media/fonts/`：本地正文、手写和等宽字体，以及保持原样的 Font Awesome / Academicons 图标字体；来源与许可在 `NOTICE.txt`。

首页与 CV 预加载首屏字体，只有正式字体加载成功后才显示文字，没有超时回退。字体加载缓慢或失败时文字保持隐藏，背景和照片照常显示。

简介中的 `statement` 支持 `**加粗**`、`==黄色底==`、`==**加粗并加黄色底**==`，只需修改引号中的文本。`affiliation` 中的链接仍使用 HTML。更新内容后修改 `data.js` 顶部的 `lastUpdated`，首页和 CV 页脚会同步显示该日期。

添加论文时复制 `publications` 中的一条，修改字段；`selectedOrder` 控制 Selected 列表顺序。首页上有 GitHub Code 链接的论文会自动显示 Stars。

演示用 `image` 指定静态封面、`video` 指定视频；`playback: "viewport"` 会在进入视口时加载并循环播放，离屏或切到后台时暂停。省略 `playback` 则保留悬停播放。两段 KungfuBot demo 已预先转成 H.264 MP4，日常打包无需 FFmpeg。

## 检查与发布

在仓库根目录直接运行，无需安装任何包：

```sh
node scripts/build-site.cjs
```

`dist/` 是完整的静态站点；首页和 CV 也可直接打开根目录的 HTML 预览。构建脚本会自动生成 `/agent/` 阅读目录和 `sitemap.xml`，并检查主页的本地资源。

推送到 `master` 后，`.github/workflows/deploy.yml` 会直接运行打包脚本并发布到 `gh-pages`。脚本会检查页面脚本语法和本地资源。GitHub Pages 保持从 `gh-pages` 根目录发布即可。不要手动编辑 `dist/` 或 `gh-pages`。
