/*
 * V1 首屏的交互小人：canvas 画的人形机器人，按输入的文字指令切换动作（呼应 TextOp）。
 *
 *   Demos.mount(el, "textop")       -> instance
 *   instance.command("wave both hands")
 *
 * 颜色读取容器上的 CSS 变量：--demo-bg --demo-fg --demo-accent --demo-muted --demo-grid --demo-font
 */
(function () {
  "use strict";

  // ---------- utils ----------
  const TAU = Math.PI * 2;
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const lerp = (a, b, u) => a + (b - a) * u;
  const ease = (u) => u * u * (3 - 2 * u);
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- humanoid (side view) ----------
  // 角度：肢体相对竖直向下，正值朝前(+x)。手臂相对躯干；腿为绝对角；膝盖为负(向后弯)。
  const L = { torso: 0.3, neck: 0.05, head: 0.062, ua: 0.15, fa: 0.14, th: 0.23, sh: 0.23 };
  const BASE = { tx: 0, ls: 0.1, le: 0.2, rs: -0.1, re: 0.2, lh: 0.05, lk: -0.05, rh: -0.05, rk: -0.05, lift: 0, flip: 1 };
  const P = (o) => Object.assign({}, BASE, o);
  const POSE = {
    stand: P({}),
    guard: P({ tx: 0.05, ls: 0.7, le: 1.9, rs: 1.0, re: 1.7, lh: 0.35, lk: -0.5, rh: -0.3, rk: -0.35 }),
    kick: P({ tx: -0.38, ls: -0.5, le: 0.6, rs: 0.9, re: 1.6, lh: 1.75, lk: -0.05, rh: -0.08, rk: -0.08 }),
    chamber: P({ tx: -0.1, ls: 0.4, le: 1.8, rs: 0.9, re: 1.7, lh: 1.4, lk: -1.9, rh: -0.05, rk: -0.1 }),
    punch: P({ tx: 0.18, ls: 1.57, le: 0.02, rs: 0.3, re: 2.2, lh: 0.45, lk: -0.65, rh: -0.4, rk: -0.25 }),
    punch2: P({ tx: 0.22, ls: 0.3, le: 2.2, rs: 1.6, re: 0.02, lh: 0.45, lk: -0.65, rh: -0.4, rk: -0.25 }),
    squat: P({ tx: 0.4, ls: 1.45, le: 0.1, rs: 1.35, re: 0.15, lh: 1.25, lk: -2.1, rh: 1.15, rk: -2.0 }),
    waveA: P({ ls: 2.75, le: 0.55, rs: 2.65, re: 0.35, lh: 0.08, rh: -0.08 }),
    waveB: P({ ls: 2.75, le: -0.35, rs: 2.65, re: -0.45, lh: 0.08, rh: -0.08 }),
    crouch: P({ tx: 0.25, ls: -0.6, le: 0.3, rs: -0.5, re: 0.3, lh: 0.8, lk: -1.4, rh: 0.7, rk: -1.3 }),
    air: P({ tx: -0.05, ls: 2.9, le: 0.1, rs: 2.8, re: 0.1, lh: 0.35, lk: -0.6, rh: 0.1, rk: -0.5, lift: 0.22 }),
    dance1: P({ tx: -0.15, ls: 2.4, le: 0.8, rs: -0.6, re: 0.4, lh: 0.5, lk: -0.9, rh: -0.15, rk: -0.1 }),
    dance2: P({ tx: 0.15, ls: -0.6, le: 0.4, rs: 2.4, re: 0.8, lh: 0.1, lk: -0.1, rh: 0.45, rk: -0.9 }),
    shrug: P({ ls: 0.7, le: 1.6, rs: 0.6, re: 1.7, tx: -0.05 }),
  };
  const spin = (o) => Object.assign({}, POSE.kick, { flip: -1, lift: 0.08 }, o);

  const MOTIONS = {
    kungfu: [
      [POSE.guard, 0.55], [POSE.chamber, 0.22], [POSE.kick, 0.4], [POSE.chamber, 0.2], [POSE.guard, 0.5],
      [POSE.punch, 0.18], [POSE.punch2, 0.2], [POSE.punch, 0.2], [POSE.guard, 0.5],
      [POSE.crouch, 0.25], [spin({ lh: 0.9, lk: -1.6 }), 0.22], [spin(), 0.3], [P(Object.assign({}, POSE.guard, { flip: 1 })), 0.35],
    ],
    dance: [[POSE.dance1, 0.45], [POSE.dance2, 0.45], [POSE.dance1, 0.45], [POSE.waveA, 0.4], [POSE.dance2, 0.45]],
    jump: [[POSE.stand, 0.4], [POSE.crouch, 0.35], [POSE.air, 0.3], [POSE.air, 0.12], [POSE.crouch, 0.25], [POSE.stand, 0.45]],
    wave: [[POSE.waveA, 0.32], [POSE.waveB, 0.32]],
    punch: [[POSE.guard, 0.3], [POSE.punch, 0.18], [POSE.guard, 0.18], [POSE.punch2, 0.18], [POSE.guard, 0.3]],
    squat: [[POSE.stand, 0.5], [POSE.squat, 0.7], [POSE.squat, 0.5], [POSE.stand, 0.6]],
    spinkick: [[POSE.guard, 0.45], [POSE.crouch, 0.25], [spin({ lh: 0.9, lk: -1.6 }), 0.22], [spin(), 0.35], [P(Object.assign({}, POSE.guard, { flip: 1 })), 0.4]],
    shrug: [[POSE.shrug, 0.6], [POSE.stand, 0.6]],
    stand: [[POSE.stand, 1]],
  };
  // 每段的时长写在“目标位姿”上：[pose, 从上一位姿过渡到此位姿的时间]
  function keyframePose(frames, t) {
    let total = 0;
    for (const f of frames) total += f[1];
    t = ((t % total) + total) % total;
    for (let i = 0; i < frames.length; i++) {
      const [p, d] = frames[i];
      if (t <= d) {
        const prev = frames[(i - 1 + frames.length) % frames.length][0];
        return lerpPose(prev, p, ease(t / d));
      }
      t -= d;
    }
    return frames[0][0];
  }
  function lerpPose(a, b, u) {
    const o = {};
    for (const k in BASE) o[k] = lerp(a[k], b[k], u);
    return o;
  }
  function walkPose(t, speed) {
    const ph = t * (speed || 7);
    const s = Math.sin(ph), c = Math.cos(ph);
    return P({
      tx: 0.08,
      lh: 0.42 * s, lk: -0.9 * Math.max(0, -c) - 0.08,
      rh: -0.42 * s, rk: -0.9 * Math.max(0, c) - 0.08,
      ls: -0.45 * s, le: 0.5, rs: 0.45 * s, re: 0.5,
      lift: 0.012 * Math.abs(Math.sin(ph)),
    });
  }
  const motionPose = (name, t) => (name === "walk" ? walkPose(t) : keyframePose(MOTIONS[name] || MOTIONS.stand, t));

  function fk(p) {
    const pel = { x: 0, y: 0 };
    const neck = { x: L.torso * Math.sin(p.tx), y: -L.torso * Math.cos(p.tx) };
    const hd = L.neck + L.head;
    const head = { x: neck.x + hd * Math.sin(p.tx), y: neck.y - hd * Math.cos(p.tx) };
    const arm = (s, e) => {
      const a1 = p.tx + s, a2 = a1 + e;
      const el = { x: neck.x + L.ua * Math.sin(a1), y: neck.y + L.ua * Math.cos(a1) };
      return [el, { x: el.x + L.fa * Math.sin(a2), y: el.y + L.fa * Math.cos(a2) }];
    };
    const leg = (h, k) => {
      const kn = { x: L.th * Math.sin(h), y: L.th * Math.cos(h) };
      return [kn, { x: kn.x + L.sh * Math.sin(h + k), y: kn.y + L.sh * Math.cos(h + k) }];
    };
    const [le, lw] = arm(p.ls, p.le), [re, rw] = arm(p.rs, p.re);
    const [lk, lf] = leg(p.lh, p.lk), [rk, rf] = leg(p.rh, p.rk);
    const J = { pel, neck, head, le, lw, re, rw, lk, lf, rk, rf };
    const ground = Math.max(lf.y, rf.y);
    for (const k in J) { J[k] = { x: J[k].x * p.flip, y: J[k].y - ground - p.lift }; }
    return J;
  }

  function drawFigure(ctx, J, ox, oy, s, col, opt) {
    opt = opt || {};
    const X = (pt) => ox + pt.x * s, Y = (pt) => oy + pt.y * s;
    const seg = (a, b) => { ctx.moveTo(X(a), Y(a)); ctx.lineTo(X(b), Y(b)); };
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    const w = opt.width || Math.max(2, s * 0.032);
    if (opt.dashed) ctx.setLineDash([w * 1.2, w * 1.4]);
    // 后侧肢体
    ctx.strokeStyle = opt.backColor || col; ctx.lineWidth = w;
    ctx.globalAlpha = (opt.alpha == null ? 1 : opt.alpha) * (opt.dashed ? 1 : 0.55);
    ctx.beginPath(); seg(J.pel, J.rk); seg(J.rk, J.rf); seg(J.neck, J.re); seg(J.re, J.rw); ctx.stroke();
    ctx.globalAlpha = opt.alpha == null ? 1 : opt.alpha;
    ctx.strokeStyle = col;
    ctx.beginPath(); seg(J.pel, J.neck); seg(J.pel, J.lk); seg(J.lk, J.lf); seg(J.neck, J.le); seg(J.le, J.lw); ctx.stroke();
    ctx.setLineDash([]);
    if (!opt.dashed) {
      ctx.lineWidth = w * 0.8;
      ctx.beginPath(); ctx.arc(X(J.head), Y(J.head), L.head * s, 0, TAU); ctx.stroke();
      if (opt.joints) {
        ctx.fillStyle = opt.joints;
        for (const k of ["pel", "neck", "le", "re", "lk", "rk"]) { ctx.beginPath(); ctx.arc(X(J[k]), Y(J[k]), w * 0.62, 0, TAU); ctx.fill(); }
      }
    }
    ctx.globalAlpha = 1;
  }

  // ---------- scenes ----------
  function floor(ctx, w, h, gy, C, scroll) {
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    const vx = w / 2, n = 14, sp = w / 7;
    ctx.globalAlpha = 0.7;
    ctx.beginPath();
    for (let i = -n; i <= n; i++) { const x = vx + i * sp - (scroll || 0) % sp; ctx.moveTo(x, gy); ctx.lineTo(vx + (x - vx) * 2.4, h); }
    for (let k = 1; k < 5; k++) { const y = gy + (h - gy) * Math.pow(k / 5, 1.8); ctx.moveTo(0, y); ctx.lineTo(w, y); }
    ctx.stroke(); ctx.globalAlpha = 1;
  }
  function label(ctx, txt, x, y, C, size, color, align) {
    ctx.font = `${size}px ${C.font}`; ctx.fillStyle = color || C.muted; ctx.textAlign = align || "left"; ctx.textBaseline = "alphabetic";
    ctx.fillText(txt, x, y);
  }

  const SCENES = {
    textop: {
      poster: 3.2,
      cmds: [
        ["walk forward", "walk"], ["wave both hands", "wave"], ["throw two punches", "punch"],
        ["squat down slowly", "squat"], ["do a spin kick", "spinkick"], ["jump up", "jump"],
      ],
      parse(text) {
        const s = text.toLowerCase();
        const table = [
          [/walk|run|forward|go|走/, "walk"], [/wave|hello|hi\b|hey|挥/, "wave"], [/punch|box|hit|拳/, "punch"],
          [/squat|sit|down|蹲/, "squat"], [/kick|kung|spin|踢|功夫/, "spinkick"], [/jump|hop|跳/, "jump"],
          [/dance|groove|舞/, "dance"],
        ];
        for (const [re, m] of table) if (re.test(s)) return m;
        return "shrug";
      },
      draw(ctx, w, h, t, C, st) {
        const small = h < 230, head = small ? 44 : 80; // 小画布用紧凑的指令栏
        const gy = h * 0.88, s = Math.max(40, Math.min(h * 0.62, (gy - head) / 1.15, w * 0.45));
        // 指令队列：外部输入优先，否则循环示例
        if (!st.init) { st.init = true; st.cur = "stand"; st.prev = "stand"; st.switchT = -9; st.text = ""; st.typedAt = 0; st.idx = -1; st.nextAuto = 0.2; }
        if (st.pending) {
          const txt = st.pending; st.pending = null;
          st.text = txt; st.typedAt = t; st.manual = t; st.queued = this.parse(txt); st.applyAt = t + 0.05;
        } else if (t >= st.nextAuto && (!st.manual || t - st.manual > 8)) {
          const first = st.idx < 0; // 第一条指令直接处于“已输入、已在动”的状态，静态封面也有内容
          st.idx = (st.idx + 1) % this.cmds.length;
          const [txt, m] = this.cmds[st.idx];
          st.text = txt; st.typedAt = first ? t - 10 : t; st.queued = m;
          st.applyAt = first ? t : t + txt.length * 0.045 + 0.15; st.nextAuto = st.applyAt + 2.8;
          st.lead = first ? 1.4 : 0;
        }
        if (st.queued && t >= st.applyAt) {
          st.prevPose = st.lead ? null : st.lastPose || POSE.stand; st.cur = st.queued; st.queued = null; st.switchT = t - (st.lead || 0); st.lead = 0;
        }
        const local = t - st.switchT;
        let pose = motionPose(st.cur, local);
        if (local < 0.45 && st.prevPose) pose = lerpPose(st.prevPose, pose, ease(local / 0.45));
        st.lastPose = pose;
        const scroll = st.cur === "walk" ? local * w * 0.18 : 0;
        floor(ctx, w, h, gy, C, scroll);
        drawFigure(ctx, fk(pose), w / 2, gy, s, C.fg, { joints: C.accent });
        // prompt
        const shown = st.manual && st.typedAt === st.manual ? st.text : st.text.slice(0, Math.floor((t - st.typedAt) / 0.045));
        const pad = small ? 8 : 12, bh = small ? 22 : 30, fs = small ? 11 : 13;
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.strokeRect(pad + 0.5, pad + 0.5, w - pad * 2 - 1, bh);
        label(ctx, "›", pad + 8, pad + bh / 2 + fs / 2 - 1, C, fs + 1, C.accent);
        label(ctx, shown + (Math.floor(t * 2) % 2 ? "▍" : " "), pad + 22, pad + bh / 2 + fs / 2 - 1, C, fs, C.fg);
        // motion tokens streaming
        const n = Math.min(24, Math.floor(Math.max(0, local) * 9));
        for (let i = 0; i < 24; i++) {
          ctx.fillStyle = i < n ? C.accent : C.grid; ctx.globalAlpha = i < n ? 0.25 + 0.75 * (i / 24) : 0.6;
          ctx.fillRect(pad + i * ((w - pad * 2) / 24), pad + bh + (small ? 5 : 8), (w - pad * 2) / 24 - 2, small ? 3 : 4);
        }
        ctx.globalAlpha = 1;
        if (!small) label(ctx, "motion tokens → tracker → robot", pad, pad + bh + 28, C, 10);
        st.info = { motion: st.cur, pose };
      },
    },

  };

  // ---------- runtime ----------
  const instances = [];
  function readColors(el) {
    const cs = getComputedStyle(el);
    const v = (n, d) => (cs.getPropertyValue(n).trim() || d);
    return {
      bg: v("--demo-bg", "transparent"), fg: v("--demo-fg", "#222"), accent: v("--demo-accent", "#e4572e"),
      muted: v("--demo-muted", "#888"), grid: v("--demo-grid", "rgba(128,128,128,.25)"),
      font: v("--demo-font", "ui-monospace, SFMono-Regular, Menlo, monospace"),
    };
  }
  function mount(el, kind, opts) {
    opts = opts || {};
    const scene = SCENES[kind];
    if (!scene) return null;
    const canvas = document.createElement("canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", opts.label || kind + " demo animation");
    canvas.style.cssText = "display:block;width:100%;height:100%";
    el.appendChild(canvas);
    const inst = {
      el, canvas, kind, scene, st: {}, visible: false, hover: false, mode: opts.mode || "auto",
      t: scene.poster || 0, colors: readColors(el), frame: 0,
      command(text) { this.st.pending = text; this.hover = true; },
      get info() { return this.st.info; },
    };
    const resize = () => {
      const r = el.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.max(1, Math.round(r.width * dpr)); canvas.height = Math.max(1, Math.round(r.height * dpr));
      inst.w = r.width; inst.h = r.height; inst.dpr = dpr; render(inst);
    };
    if (window.ResizeObserver) new ResizeObserver(resize).observe(el);
    if (window.IntersectionObserver) new IntersectionObserver((es) => es.forEach((e) => (inst.visible = e.isIntersecting))).observe(el);
    else inst.visible = true;
    if (inst.mode === "hover") {
      el.addEventListener("mouseenter", () => (inst.hover = true));
      el.addEventListener("mouseleave", () => (inst.hover = false));
      el.addEventListener("focusin", () => (inst.hover = true));
      el.addEventListener("focusout", () => (inst.hover = false));
    }
    instances.push(inst);
    resize();
    return inst;
  }
  function render(inst) {
    if (!inst.w) return;
    const ctx = inst.canvas.getContext("2d");
    ctx.setTransform(inst.dpr, 0, 0, inst.dpr, 0, 0);
    ctx.clearRect(0, 0, inst.w, inst.h);
    if (inst.colors.bg !== "transparent") { ctx.fillStyle = inst.colors.bg; ctx.fillRect(0, 0, inst.w, inst.h); }
    inst.scene.draw(ctx, inst.w, inst.h, inst.t, inst.colors, inst.st);
  }
  let lastNow = 0;
  function tick(now) {
    const dt = Math.min(0.05, (now - (lastNow || now)) / 1000); lastNow = now;
    for (const inst of instances) {
      const playing = inst.visible && (inst.mode === "hover" ? inst.hover : !reduceMotion || inst.hover);
      if (++inst.frame % 30 === 0) inst.colors = readColors(inst.el);
      if (playing) { inst.t += dt; render(inst); }
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
  if (window.matchMedia) {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const re = () => instances.forEach((i) => { i.colors = readColors(i.el); render(i); });
    mq.addEventListener ? mq.addEventListener("change", re) : mq.addListener(re);
  }

  window.Demos = { mount, refresh: () => instances.forEach((i) => { i.colors = readColors(i.el); render(i); }) };
})();
