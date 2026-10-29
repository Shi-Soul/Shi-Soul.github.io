/*
 * 所有版本共用的一份内容。改这里，所有版本（含 CV 页）同时更新。
 *
 * 新加一篇论文 = 在 publications 里复制一条改字段：
 *   selected: 是否出现在默认的 "Selected" 标签下
 *   cats:     所属分类（见 categories），决定出现在哪些筛选标签里
 *   image:    缩略图（放进 ../media/，没有就留空，会自动生成文字占位）
 *   video:    可选，鼠标悬停时才加载播放（可以直接填项目主页上的视频地址）
 *   note:     一句话说明
 *   date:     首次公开的年月（V10 的时间轴用；TextOp、TrajGen 的月份待核对）
 */
window.SITE = {
  profile: {
    name: "Weiji Xie",
    nameZh: "谢炜基",
    role: "M.S. student · SJTU",
    tagline: "Robot learning · Reinforcement learning · Embodied AI",
    statement: "I study how robots can learn physical skills from data, simulation and interaction, and carry them reliably into the real world. Long term, I want to build general-purpose robots that can do anything for people.",
    affiliation: 'M.S. student at <a href="https://www.sjtu.edu.cn/">SJTU</a>, <a href="https://apex.sjtu.edu.cn/">APEX Lab</a> (advisor <a href="https://wnzhang.net/">Prof. Weinan Zhang</a>) · Research intern at <a href="https://www.teleai.com.cn/">TeleAI</a> with <a href="https://baichenjia.github.io/">Dr. Chenjia Bai</a>',
    // 旧版本（V1–V3、V7）用的简介
    bio: 'M.S. student at <a href="https://www.sjtu.edu.cn/">SJTU</a> (<a href="https://apex.sjtu.edu.cn/">APEX Lab</a>, advised by <a href="https://wnzhang.net/">Prof. Weinan Zhang</a>) and research intern at <a href="https://www.teleai.com.cn/">TeleAI</a> with <a href="https://baichenjia.github.io/">Dr. Chenjia Bai</a>. I want to build general-purpose robots.',
    location: "Shanghai, China",
    photo: "../media/photo.jpg",
    email: "xieweiji249@sjtu.edu.cn",
    links: {
      scholar: "https://scholar.google.com/citations?user=Ey2NlGMAAAAJ",
      github: "https://github.com/Shi-Soul",
      linkedin: "https://www.linkedin.com/in/weiji-xie-bb054b2b2",
      x: "https://x.com/Shi_Soul",
    },
    wechatQR: "../../../media/wechat-qr.png",
    contactNote: "Feel free to add my WeChat or drop an email.",
    cv: "../../../cv/", // 旧版本用；新版本链接到各自的 cv.html
  },

  categories: [
    { id: "tracking", name: "Motion Tracking" },
    { id: "loco", name: "Locomotion" },
    { id: "s2r", name: "Sim-to-Real" },
    { id: "gen", name: "Generative Models" },
    { id: "marl", name: "Multi-Agent RL" },
  ],

  publications: [
    {
      short: "TextOp", date: "2026.01", venue: "Under review", year: 2026, selected: true, cats: ["tracking", "gen"], tier: "main",
      title: "TextOp: Real-time Interactive Text-Driven Humanoid Robot Motion Generation and Control",
      authors: "Weiji Xie*, Jiakun Zheng*, Jinrui Han, Jiyuan Shi, Weinan Zhang†, Chenjia Bai†, Xuelong Li",
      note: "Type a sentence; the robot moves, in real time.",
      links: { project: "https://text-op.github.io/" },
      image: "../media/textop.jpg", video: "https://text-op.github.io/static/videos/demo_big.mp4",
    },
    {
      short: "KungfuBot", date: "2025.06", venue: "NeurIPS", year: 2025, selected: true, cats: ["tracking", "s2r"], tier: "main",
      title: "KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills",
      authors: "Weiji Xie*, Jinrui Han*, Jiakun Zheng*, Huanyu Li, Xinzhe Liu, Jiyuan Shi, Weinan Zhang, Chenjia Bai†, Xuelong Li†",
      note: "Kung fu and dance on a real humanoid.",
      links: { project: "https://kungfubot.github.io/", pdf: "https://arxiv.org/pdf/2506.12851" },
      image: "../media/kungfubot.jpg", video: "https://kungfubot.github.io/static/videos/360%20spin.mp4",
    },
    {
      short: "KungfuBot2", date: "2025.09", venue: "Under review", year: 2025, selected: true, cats: ["tracking"],
      title: "KungfuBot2: Learning Versatile Motion Skills for Humanoid Whole-Body Control",
      authors: "Jinrui Han, Weiji Xie, Jiakun Zheng, Jiyuan Shi, Weinan Zhang, Ting Xiao, Chenjia Bai",
      note: "One policy, many motion skills.",
      links: { project: "https://kungfubot2-humanoid.github.io/", pdf: "https://arxiv.org/pdf/2509.16638" },
      image: "../media/kungfubot2.jpg",
    },
    {
      short: "Narrow Terrain", date: "2025.02", venue: "IROS", year: 2025, selected: true, cats: ["loco"], tier: "main",
      title: "Humanoid Whole-Body Locomotion on Narrow Terrain via Dynamic Balance and Reinforcement Learning",
      authors: "Weiji Xie, Chenjia Bai†, Jiyuan Shi, Junkai Yang, Yunfei Ge, Weinan Zhang†, Xuelong Li",
      note: "Whole-body balance on terrain as narrow as a foot.",
      links: { project: "https://whole-body-loco.github.io/", pdf: "https://arxiv.org/pdf/2502.17219" },
      image: "../media/narrow.jpg",
    },
    {
      short: "LoopSR", date: "2024.09", venue: "IROS", year: 2025, selected: true, cats: ["loco", "s2r"],
      title: "LoopSR: Looping Sim-and-Real for Lifelong Policy Adaptation of Legged Robots",
      authors: "Peilin Wu, Weiji Xie, Jiahang Cao, Hang Lai, Weinan Zhang",
      note: "Real-world experience flows back into simulation, for life.",
      links: { project: "https://peilinwu.site/looping-sim-and-real.github.io/", pdf: "https://arxiv.org/pdf/2409.17992" },
      image: "../media/loopsr.jpg",
    },
    {
      short: "OFF-FSP", date: "2024.03", venue: "AAAI", year: 2026, selected: true, cats: ["marl"],
      title: "Offline Fictitious Self-Play for Competitive Games",
      authors: "Jingxiao Chen, Weiji Xie, Weinan Zhang†, Yong Yu, Ying Wen†",
      note: "Self-play from cheap offline data, tested on a real robot game.",
      links: { pdf: "https://arxiv.org/pdf/2403.00841" },
      image: "../media/ofsp.jpg",
    },
    {
      short: "TrajGen", date: "2024.06", venue: "Under review", year: 2024, selected: false, cats: ["gen"],
      title: "A Pre-Trained Trajectory Generative Model on Road Network",
      authors: "Weiji Xie, Yufan Zhang, Rui Zhang, Zhouyue Sun, Tianyu Liu, Guanjie Zheng, Zhenhui Li, Weinan Zhang",
      note: "A pre-trained generative model for city-scale vehicle trajectories.",
      links: { pdf: "../../../media/traffic_model.pdf" },
      image: "../media/trajgen.jpg",
    },
  ],

  // TODO: 日期请核对
  news: [
    { date: "2026.01", text: "TextOp released: real-time text-driven humanoid control." },
    { date: "2025.11", text: "OFF-FSP accepted to AAAI 2026." },
    { date: "2025.09", text: "KungfuBot accepted to NeurIPS 2025." },
    { date: "2025.09", text: "Started my M.S. at SJTU." },
    { date: "2025.06", text: "Two papers accepted to IROS 2025." },
  ],

  education: [
    { when: "2025.09 – now", title: "M.S. in Computer Science and Technology", org: "Shanghai Jiao Tong University", detail: "APEX Lab, advised by Prof. Weinan Zhang" },
    { when: "2021.09 – 2025.06", title: "B.Eng. in Artificial Intelligence", org: "Shanghai Jiao Tong University", detail: "GPA 4.03 / 4.30 · Rank top 5%" },
  ],
  experience: [
    { when: "2024.10 – now", title: "Research Intern", org: "TeleAI, China Telecom", detail: "With Dr. Chenjia Bai" },
    { when: "2024.06 – 2024.08", title: "Research Intern", org: "Yunqi Academy of Engineering" },
  ],
  honors: [
    { year: 2024, text: "ASC Student Supercomputer Challenge, First Prize & Application Innovation Award" },
    { year: 2024, text: "SJTU B-Class Excellent Scholarship" },
    { year: 2023, text: "SJTU B-Class Excellent Scholarship" },
    { year: 2023, text: "Outstanding Youth League Member, SJTU" },
  ],
  service: ["Reviewer: TPAMI, ICRA"],
  misc: [
    { k: "Languages", v: "Chinese, English" },
    { k: "Hobbies", v: "Reading math, philosophy and social science; badminton." },
  ],
};

// 旧版本（V1–V3、V7）用的精简履历
window.SITE.cv = [
  { when: "2025 –", what: "M.S. Computer Science, SJTU" },
  { when: "2024 –", what: "Research Intern, TeleAI" },
  { when: "2021 – 25", what: "B.Eng. Artificial Intelligence, SJTU" },
];
