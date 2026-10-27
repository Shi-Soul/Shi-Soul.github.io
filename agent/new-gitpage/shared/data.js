/*
 * 所有版本共用的一份内容。改这里，所有版本同时更新。
 *
 * 新加一篇论文 = 在 publications 里复制一条改字段：
 *   tier:  "main"（主：一作/共一的代表作，带缩略图大卡片） | "other"（次：紧凑列表）
 *   image: 缩略图（放进 ../media/，没有就留空，会自动生成文字占位）
 *   video: 可选，鼠标悬停时才加载播放（可以直接填项目主页上的视频地址）
 *   note:  一句话说明（一行以内）
 */
window.SITE = {
  profile: {
    name: "Weiji Xie",
    nameZh: "谢炜基",
    role: "M.S. student · SJTU",
    tagline: "Humanoid whole-body control · Reinforcement learning",
    taglineZh: "人形机器人全身控制 · 强化学习",
    bio: 'M.S. student at <a href="https://www.sjtu.edu.cn/">SJTU</a> (<a href="https://apex.sjtu.edu.cn/">APEX Lab</a>, advised by <a href="https://wnzhang.net/">Prof. Weinan Zhang</a>) and research intern at <a href="https://www.teleai.com.cn/">TeleAI</a> with <a href="https://baichenjia.github.io/">Dr. Chenjia Bai</a>. I want to build general-purpose robots.',
    bioZh: '<a href="https://www.sjtu.edu.cn/">上海交通大学</a>计算机硕士生（<a href="https://apex.sjtu.edu.cn/">APEX 实验室</a>，导师<a href="https://wnzhang.net/">张伟楠</a>教授），在 <a href="https://www.teleai.com.cn/">TeleAI</a> 实习，与<a href="https://baichenjia.github.io/">白辰甲</a>老师合作。目标是造出通用机器人。',
    location: "Shanghai, China",
    photo: "../media/photo.jpg",
    email: "xieweiji249@sjtu.edu.cn",
    links: {
      scholar: "https://scholar.google.com/citations?user=Ey2NlGMAAAAJ",
      github: "https://github.com/Shi-Soul",
      linkedin: "https://www.linkedin.com/in/weiji-xie-bb054b2b2",
      x: "https://x.com/Shi_Soul",
    },
    wechatQR: "../../../assets/img/wechat-qr.png",
    contactNote: "Feel free to add my WeChat or drop an email.",
    cv: "../../../cv/",
  },

  publications: [
    {
      short: "TextOp", tier: "main", venue: "Under review", year: 2026,
      title: "TextOp: Real-time Interactive Text-Driven Humanoid Robot Motion Generation and Control",
      authors: "Weiji Xie*, Jiakun Zheng*, Jinrui Han, Jiyuan Shi, Weinan Zhang†, Chenjia Bai†, Xuelong Li",
      note: "Type a sentence; the humanoid moves, in real time.", noteZh: "边打字，机器人边动。",
      links: { project: "https://text-op.github.io/" },
      image: "../media/textop.jpg", video: "https://text-op.github.io/static/videos/demo_big.mp4",
    },
    {
      short: "KungfuBot", tier: "main", venue: "NeurIPS", year: 2025,
      title: "KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills",
      authors: "Weiji Xie*, Jinrui Han*, Jiakun Zheng*, Huanyu Li, Xinzhe Liu, Jiyuan Shi, Weinan Zhang, Chenjia Bai†, Xuelong Li†",
      note: "Kung fu and dance on a real humanoid.", noteZh: "让人形机器人打功夫、跳舞。",
      links: { project: "https://kungfubot.github.io/", pdf: "https://arxiv.org/pdf/2506.12851" },
      image: "../media/kungfubot.jpg", video: "https://kungfubot.github.io/static/videos/360%20spin.mp4",
    },
    {
      short: "Narrow Terrain", tier: "main", venue: "IROS", year: 2025,
      title: "Humanoid Whole-Body Locomotion on Narrow Terrain via Dynamic Balance and Reinforcement Learning",
      authors: "Weiji Xie, Chenjia Bai†, Jiyuan Shi, Junkai Yang, Yunfei Ge, Weinan Zhang†, Xuelong Li",
      note: "Whole-body balance on terrain as narrow as a foot.", noteZh: "在一脚宽的地形上保持全身平衡。",
      links: { project: "https://whole-body-loco.github.io/", pdf: "https://arxiv.org/pdf/2502.17219" },
      image: "../media/narrow.jpg",
    },
    {
      short: "KungfuBot2", tier: "other", venue: "Under review", year: 2025,
      title: "KungfuBot2: Learning Versatile Motion Skills for Humanoid Whole-Body Control",
      authors: "Jinrui Han, Weiji Xie, Jiakun Zheng, Jiyuan Shi, Weinan Zhang, Ting Xiao, Chenjia Bai",
      note: "One policy, many motion skills.", noteZh: "一个策略，多种动作技能。",
      links: { project: "https://kungfubot2-humanoid.github.io/", pdf: "https://arxiv.org/pdf/2509.16638" },
      image: "../media/kungfubot2.jpg",
    },
    {
      short: "LoopSR", tier: "other", venue: "IROS", year: 2025,
      title: "LoopSR: Looping Sim-and-Real for Lifelong Policy Adaptation of Legged Robots",
      authors: "Peilin Wu, Weiji Xie, Jiahang Cao, Hang Lai, Weinan Zhang",
      note: "Lifelong sim-and-real adaptation for legged robots.", noteZh: "仿真与真实循环，终身适应。",
      links: { project: "https://peilinwu.site/looping-sim-and-real.github.io/", pdf: "https://arxiv.org/pdf/2409.17992" },
      image: "../media/loopsr.jpg",
    },
    {
      short: "OFSP", tier: "other", venue: "AAAI", year: 2026,
      title: "Offline Fictitious Self-Play for Competitive Games",
      authors: "Jingxiao Chen, Weiji Xie, Weinan Zhang†, Yong Yu, Ying Wen†",
      note: "Nash equilibria from offline data.", noteZh: "只用离线数据逼近纳什均衡。",
      links: { pdf: "https://arxiv.org/pdf/2403.00841" },
    },
    {
      short: "TrajGen", tier: "other", venue: "Under review", year: 2024,
      title: "A Pre-Trained Trajectory Generative Model on Road Network",
      authors: "Weiji Xie, Yufan Zhang, Rui Zhang, Zhouyue Sun, Tianyu Liu, Guanjie Zheng, Zhenhui Li, Weinan Zhang",
      note: "Pre-trained generative model for road-network trajectories.", noteZh: "路网上的预训练轨迹生成模型。",
      links: { pdf: "../../../assets/pdf/traffic_model.pdf" },
    },
  ],

  // TODO: 日期请核对
  news: [
    { date: "2026.01", text: "TextOp released.", textZh: "TextOp 发布。" },
    { date: "2025.11", text: "OFSP accepted to AAAI 2026.", textZh: "OFSP 被 AAAI 2026 接收。" },
    { date: "2025.09", text: "KungfuBot accepted to NeurIPS 2025.", textZh: "KungfuBot 被 NeurIPS 2025 接收。" },
    { date: "2025.09", text: "Started M.S. at SJTU.", textZh: "开始在上海交大读硕士。" },
    { date: "2025.06", text: "Two papers accepted to IROS 2025.", textZh: "两篇论文被 IROS 2025 接收。" },
  ],

  cv: [
    { when: "2025 –", what: "M.S. Computer Science, SJTU" },
    { when: "2024 –", what: "Research Intern, TeleAI" },
    { when: "2021 – 25", what: "B.Eng. Artificial Intelligence, SJTU" },
  ],
  awards: ["ASC Student Supercomputer Challenge, First Prize (2024)", "SJTU Excellent Scholarship (2023, 2024)"],
  service: "Reviewer: TPAMI, ICRA",
};
