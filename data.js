/*
 * 主页的全部内容。改这里，首页和 CV 页同时更新。
 * 论文列表由简历与 Google Scholar 整理，2026-10-28。
 *
 * 新加一篇论文 = 在 publications 里复制一条改字段：
 *   selected: 是否出现在默认的 "Selected" 标签下（顺序见 selectedOrder）
 *   cats:     所属分类（见 categories），决定出现在哪些筛选标签里
 *   image:    缩略图（放进 media/，没有就留空，会自动生成文字占位）
 *   video:    可选，鼠标悬停时才加载播放（可以直接填项目主页上的视频地址）
 *   links:    project / pdf / code，有哪个填哪个
 *   note:     一句话说明
 *   date:     首次公开的年月（取 arXiv 编号），分类标签按它倒序
 */
window.SITE = {
  lastUpdated: "2026-10-29", // 更新内容后修改此日期，首页与 CV 页脚同步显示。
  profile: {
    name: "Weiji Xie",
    nameZh: "谢炜基",
    role: "M.S. student · SJTU",
    tagline: "Robot learning · Reinforcement learning · Embodied AI",
    // 简介格式：**加粗**，==黄色底==，==**加粗并加黄色底**==；删除标记即恢复普通文字。
    affiliation: 'I’m an M.S. student at <a href="https://www.sjtu.edu.cn/">Shanghai Jiao Tong University (SJTU)</a> supervised by <a href="https://wnzhang.net/">Prof. Weinan Zhang</a>. I am currently interning at the <a href="https://www.teleai.com.cn/">TeleAI</a> and working closely with <a href="https://baichenjia.github.io/">Dr. Chenjia Bai</a>.',
    statement: "My research interests include **Reinforcement Learning** and **Robotics**, with a current focus on **Humanoid Robots**. Fundamentally, I want to build ==General-Purpose Robots== that can do anything for people.",
    location: "Shanghai, China",
    photo: "media/photo.webp",
    email: "xieweiji249@sjtu.edu.cn",
    links: {
      scholar: "https://scholar.google.com/citations?user=Ey2NlGMAAAAJ",
      github: "https://github.com/Shi-Soul",
      linkedin: "https://www.linkedin.com/in/weiji-xie-bb054b2b2",
      x: "https://x.com/Shi_Soul",
    },
    wechatQR: "media/wechat-qr.png",
    contactNote: "Feel free to add my WeChat or drop an email.",
    collabNote: "Always happy to chat and open to collaboration.",
  },

  // Selected 标签下的展示顺序（填 short 名，自己调）；其他标签按 date 倒序
  selectedOrder: ["TTC", "TextOp", "KungfuBot", "HALO", "KungfuBot2", "DBHL"],

  categories: [
    { id: "tracking", name: "Motion Tracking" },
    { id: "locomanip", name: "Loco-Manipulation" },
    { id: "loco", name: "Locomotion" },
    // { id: "misc", name: "Others" },
  ],

  // 合作者主页：姓名完全匹配时，首页和 CV 自动添加链接；部分使用机构/GitHub/Scholar 备用入口。
  // 来源、候选匹配及未找到的作者见 agent/new-gitpage/coauthors-review.md。
  coauthors: {
    "Junsong Wu": "https://jason-wjs.github.io/",
    // "Zibo Zhou": "https://github.com/ZiboKNOW",
    // "Yueru Chen": "https://github.com/YueruChen111",
    "Weinan Zhang": "https://wnzhang.net/",
    "Chenjia Bai": "https://baichenjia.github.io/",
    "Jiyuan Shi": "https://scholar.google.com/citations?user=aVte5j4AAAAJ",
    "Xuelong Li": "https://scholar.google.com/citations?user=ahUibskAAAAJ",
    "Xiangyu Miao": "https://xiangyumiao.pages.dev/",
    "Jinrui Han": "https://bwrooney82.github.io/",
    // "Zehao Yu": "https://github.com/yzhperseverance",
    // "Jiakun Zheng": "https://github.com/Gabril17",
    "Peilin Wu": "https://peilinwu.site/",
    "Jingxiao Chen": "https://timerchen.github.io/",
    "Xinzhe Liu": "https://xinzheliu.github.io/",
    "Yunfei Ge": "https://gecloud24.github.io/",
    // "Jiahang Cao": "https://apex.sjtu.edu.cn/members/jhcao@apexlab.org",
    "Hang Lai": "https://scholar.google.com/citations?user=WYdaCd0AAAAJ",
    "Yong Yu": "https://scholar.google.com/citations?user=-84M1m0AAAAJ&hl=en",
    "Ying Wen": "https://yingwen.io/",
    "Xingyi Wang": "https://scholar.google.com/citations?user=w7dfT-gAAAAJ",
    // "Chenyun Zhang": "https://github.com/chenyunzhanggit",
  },

  publications: [
    {
      short: "TTC", date: "2026.10", venue: "Under review", year: 2026, selected: true, cats: ["tracking", "locomanip"],
      title: "Test-Time Compute for Humanoid Behavior Foundation Models",
      authors: "Weiji Xie*, Junsong Wu*, Zibo Zhou, Yueru Chen, Xingyi Wang, Zhenhua Xiong, Weinan Zhang, Chenjia Bai†",
      note: "Search better commands in simulation at test time; a frozen behavior foundation model executes more precisely, no retraining.",
      links: { project: "https://bfmttc.github.io/", pdf: "https://arxiv.org/abs/2610.14896", code: "https://github.com/Gamma-Robot/bfm_ttc" },
      image: "media/ttc.jpg", video: "https://bfmttc.github.io/assets/videos/command-optimization-loop.mp4",
    },
    {
      short: "KODA", date: "2026.10", venue: "Under review", year: 2026, cats: ["locomanip"],
      title: "KODA: Kinematics–Observation Data Alignment for Humanoid Learning from Human Egocentric Demonstration",
      authors: "Zibo Zhou*, Yueru Chen*, Junsong Wu, Weiji Xie, Qingyao Xu, Sheng Yin, Shiqing Wang, Jiyuan Shi, Siheng Chen, Chenjia Bai†, Xuelong Li†",
      note: "Turn human egocentric demonstrations into aligned training data for humanoid VLA policies.",
      links: { project: "https://koda-humanoid.github.io/koda/", pdf: "https://arxiv.org/abs/2610.21152", code: "https://github.com/Gamma-Robot/KODA" },
      image: "media/koda.jpg",
    },
    {
      short: "NEXUS", date: "2026.09", venue: "Under review", year: 2026, cats: ["loco"],
      title: "NEXUS: Perceptive Whole-Body Control for Terrain-Adaptive Teleoperation",
      authors: "Xiangyu Miao, Junsong Wu, Jiyuan Shi, Weiji Xie, Jinrui Han, Xingyi Wang, Weinan Zhang†, Chenjia Bai†, Xuelong Li",
      note: "The operator walks on flat ground; the humanoid adapts to stairs and slopes on its own.",
      links: { project: "https://nexus-humanoid.github.io/", pdf: "https://arxiv.org/abs/2609.39000", code: "https://github.com/Gamma-Robot/NEXUS" },
      image: "media/nexus.jpg",
    },
    {
      short: "OASIS", date: "2026.06", venue: "Under review", year: 2026, cats: ["locomanip"],
      title: "OASIS: From Simulation Data Collection to Real-World Humanoid Loco-Manipulation",
      authors: "Zehao Yu, Jiakun Zheng, Weiji Xie, Jiyuan Shi, Chenyun Zhang, Chenjia Bai, Xuelong Li",
      note: "Collect loco-manipulation data in simulation, deploy zero-shot on a real humanoid.",
      links: { project: "https://oasis-humanoid.github.io/", pdf: "https://arxiv.org/abs/2606.08548", code: "https://github.com/TeleHuman/OASIS" },
      image: "media/oasis.jpg", video: "https://oasis-humanoid.github.io/static/videos/asset/cup.mp4",
    },
    {
      short: "HALO", date: "2026.03", venue: "IROS", year: 2026, selected: false, cats: ["tracking"],
      title: "HALO: Closing Sim-to-Real Gap for Heavy-loaded Humanoid Agile Motion Skills via Differentiable Simulation",
      authors: "Xingyi Wang*, Chenyun Zhang*, Weiji Xie*, Chao Yu, Wei Song, Chenjia Bai†, Shiqiang Zhu†",
      note: "Agile motions under heavy payloads, with differentiable simulation to identify what the simulator gets wrong.",
      links: { project: "https://mwondering.github.io/halo-humanoid/", pdf: "https://arxiv.org/abs/2603.15084" },
      image: "media/halo.jpg", video: "https://mwondering.github.io/halo-humanoid/static/videos/sysid_demo_01.mp4",
    },
    {
      short: "TextOp", date: "2026.02", venue: "CoRL", year: 2026, selected: true, cats: ["tracking"],
      title: "TextOp: Real-time Interactive Text-Driven Humanoid Robot Motion Generation and Control",
      authors: "Weiji Xie*, Jiakun Zheng*, Jinrui Han, Jiyuan Shi, Weinan Zhang†, Chenjia Bai†, Xuelong Li",
      note: "Type a sentence; the robot moves, in real time. Change the command mid-motion and it follows.",
      links: { project: "https://text-op.github.io/", pdf: "https://arxiv.org/abs/2602.07439", code: "https://github.com/TeleHuman/TextOp" },
      image: "media/textop.jpg", video: "https://text-op.github.io/static/videos/demo_big.mp4",
    },
    {
      short: "KungfuBot2", date: "2025.09", venue: "ICRA", year: 2026, selected: true, cats: ["tracking"],
      title: "KungfuBot2: Learning Versatile Motion Skills for Humanoid Whole-Body Control",
      authors: "Jinrui Han, Weiji Xie, Jiakun Zheng, Jiyuan Shi, Weinan Zhang, Ting Xiao, Chenjia Bai",
      note: "One policy, many skills: minute-long martial arts and dance sequences, including unseen motions.",
      links: { project: "https://kungfubot2-humanoid.github.io/", pdf: "https://arxiv.org/abs/2509.16638", code: "https://github.com/TeleHuman/PBHC" },
      image: "media/kungfubot2.jpg",
    },
    {
      short: "KungfuBot", date: "2025.06", venue: "NeurIPS", year: 2025, selected: true, cats: ["tracking"],
      title: "KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills",
      authors: "Weiji Xie*, Jinrui Han*, Jiakun Zheng*, Huanyu Li, Xinzhe Liu, Jiyuan Shi, Weinan Zhang, Chenjia Bai†, Xuelong Li†",
      note: "Spin kicks and flying kicks on a real Unitree G1, learned from human video.",
      links: { project: "https://kungfubot.github.io/", pdf: "https://arxiv.org/abs/2506.12851", code: "https://github.com/TeleHuman/PBHC" },
      image: "media/kungfubot.jpg", video: "https://kungfubot.github.io/static/videos/360%20spin.mp4",
    },
    {
      short: "DBHL", date: "2025.02", venue: "IROS", year: 2025, selected: false, cats: ["loco"],
      title: "Humanoid Whole-Body Locomotion on Narrow Terrain via Dynamic Balance and Reinforcement Learning",
      authors: "Weiji Xie, Chenjia Bai†, Jiyuan Shi, Junkai Yang, Yunfei Ge, Weinan Zhang†, Xuelong Li",
      note: "Proprioception-only walking on narrow beams and obstacles, with balance recovery under pushes.",
      links: { project: "https://whole-body-loco.github.io/", pdf: "https://arxiv.org/abs/2502.17219" },
      image: "media/narrow.jpg",
    },
    {
      short: "LoopSR", date: "2024.09", venue: "IROS", year: 2025, cats: ["loco"],
      title: "LoopSR: Looping Sim-and-Real for Lifelong Policy Adaptation of Legged Robots",
      authors: "Peilin Wu, Weiji Xie, Jiahang Cao, Hang Lai, Weinan Zhang",
      note: "Real-world experience flows back into simulation, so a legged robot keeps improving for life.",
      links: { project: "https://peilinwu.site/looping-sim-and-real.github.io/", pdf: "https://arxiv.org/abs/2409.17992", code: "https://github.com/aldeW64/LoopSR" },
      image: "media/loopsr.jpg",
    },
    {
      short: "OFF-FSP", date: "2024.03", venue: "AAAI", year: 2026, cats: ["misc"],
      title: "Offline Fictitious Self-Play for Competitive Games",
      authors: "Jingxiao Chen, Weiji Xie, Weinan Zhang†, Yong Yu, Ying Wen†",
      note: "Self-play from cheap offline data, tested on a real robot ball-defence game.",
      links: { pdf: "https://arxiv.org/abs/2403.00841" },
      image: "media/ofsp.jpg",
    },
  ],

  // 日期为大致月份，请核对。文中出现的论文 short 名会自动链接到项目主页；其他链接写在 newsLinks 里
  newsLinks: { PBHC: "https://github.com/TeleHuman/PBHC" },
  news: [
    { date: "2026.10", text: "TTC, KODA and NEXUS released on arXiv." },
    { date: "2026.08", text: "TextOp accepted to CoRL 2026." },
    { date: "2026.06", text: "HALO accepted to IROS 2026. OASIS released." },
    { date: "2026.01", text: "KungfuBot2 accepted to ICRA 2026." },
    { date: "2026.01", text: "TextOp released." },
    { date: "2025.11", text: "OFF-FSP accepted to AAAI 2026." },
    { date: "2025.09", text: "KungfuBot accepted to NeurIPS 2025. Started my M.S. at SJTU." },
    { date: "2025.06", text: "KungfuBot released with open source code PBHC." },
    { date: "2025.06", text: "DBHL and LoopSR accepted to IROS 2025." },
  ],

  // 经历：logo 填 media/ 下的图；没有 logo 时用 mono 里的两个字母做徽章
  education: [
    { when: "2025.09 – 2028.03 (expected)", title: "M.S. in Computer Science and Technology", org: "Shanghai Jiao Tong University", logo: "media/logo-sjtu.jpg",
      advisor: { role: "Advisor", name: "Prof. Weinan Zhang", url: "https://wnzhang.net/" }, detail: "Reinforcement Learning and Embodied AI" },
    { when: "2021.09 – 2025.06", title: "B.Eng. in Artificial Intelligence", org: "Shanghai Jiao Tong University", logo: "media/logo-sjtu.jpg",
      detail: "GPA 4.07 / 4.30 · Rank top 5%" },
  ],
  experience: [
    { when: "2024.10 – now", title: "Research Intern", org: "TeleAI, China Telecom", logo: "media/logo-teleai.jpg",
      advisor: { role: "Supervisor", name: "Dr. Chenjia Bai", url: "https://baichenjia.github.io/" }, detail: "RL-based humanoid motion control; VLA-based loco-manipulation" },
    { when: "2024.06 – 2024.08", title: "Research Intern", org: "Yunqi Academy of Engineering",
      advisor: { role: "Advisor", name: "Prof. Zhenhui Li", url: "https://jessielzh.com/" }, detail: "City Intelligence" },
    { when: "2022.08 – 2025.06", title: "Undergraduate Researcher", org: "APEX Lab, Shanghai Jiao Tong University", orgUrl: "https://apex.sjtu.edu.cn/", orgLinkText: "APEX Lab", logo: "media/logo-apex.png", logoDark: true,
      advisor: { role: "Advisor", name: "Prof. Weinan Zhang", url: "https://wnzhang.net/" }, detail: "Multi-agent RL; Legged robot locomotion" },
  ],
  honors: [
    { year: 2024, text: "ASC Student Supercomputer Challenge, First Prize & Application Innovation Award" },
    { year: "2022–2024", text: "SJTU B-Class Excellent Scholarship (three times)" },
    { year: "2021–2024", text: "Fuguang Undergraduate Scholarship (four times)" },
    { year: 2023, text: "Outstanding Youth League Member, SJTU" },
  ],
  service: ["Conference reviewer: CoRL, ICRA, ICLR", "Journal reviewer: TPAMI, RA-L"],
  misc: [
    { k: "Tools", v: "Python (PyTorch, JAX), C++, MuJoCo, Isaac Lab / Sim / Gym, ROS" },
    { k: "Languages", v: "Chinese, English" },
    { k: "Hobbies", v: "Reading books about math, philosophy, social science, etc." },
  ],
};
