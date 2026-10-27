/*
 * 程序化占位动画：在没有真实 demo GIF/视频之前，用 canvas 画出每篇论文“在做什么”。
 *
 *   Demos.mount(el, kind, { mode: "auto" | "hover" })  -> instance
 *   instance.command("wave both hands")                 (仅 textop：外部输入指令)
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
  function rng(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
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
    kungfu: {
      poster: 1.25,
      draw(ctx, w, h, t, C, st) {
        const gy = h * 0.84, s = Math.min(h * 0.68, w * 0.5);
        floor(ctx, w, h, gy, C);
        const ref = fk(motionPose("kungfu", t + 0.07));
        const pol = fk(motionPose("kungfu", t));
        st.trail = st.trail || [];
        if (!st.last || t - st.last > 0.035) { st.trail.push(pol); st.last = t; if (st.trail.length > 7) st.trail.shift(); }
        st.trail.forEach((J, i) => drawFigure(ctx, J, w / 2, gy, s, C.fg, { alpha: 0.06 + i * 0.025 }));
        drawFigure(ctx, ref, w / 2, gy, s, C.accent, { dashed: true, width: Math.max(1.2, s * 0.012), alpha: 0.9 });
        drawFigure(ctx, pol, w / 2, gy, s, C.fg, { joints: C.accent });
        const err = Math.abs(ref.lf.y - pol.lf.y) + Math.abs(ref.lw.x - pol.lw.x);
        label(ctx, "- - reference   ── policy", 12, 20, C, 11);
        label(ctx, "track err " + (err * 10).toFixed(3), w - 12, 20, C, 11, C.muted, "right");
      },
    },

    versatile: {
      poster: 0.9,
      draw(ctx, w, h, t, C, st) {
        const gy = h * 0.8, s = Math.min(h * 0.56, w * 0.2);
        floor(ctx, w, h, gy, C);
        const skills = ["kungfu", "dance", "jump"];
        const cycle = Math.floor(t / 4.2);
        skills.forEach((m, i) => {
          const x = w * (0.2 + i * 0.3);
          drawFigure(ctx, fk(motionPose(m, t + i * 0.37)), x, gy, s, i === 1 ? C.accent : C.fg, { joints: i === 1 ? C.fg : C.accent });
          const id = String(((cycle * 97 + i * 131 + 17) % 900) + 100);
          label(ctx, "skill #" + id, x, gy + 20, C, 11, C.muted, "center");
        });
        label(ctx, "π(a | s, motion)   one policy · 10k+ clips", 12, 20, C, 11);
      },
    },

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

    narrow: {
      poster: 0.6,
      draw(ctx, w, h, t, C) {
        const vx = w / 2, vy = h * 0.18, by = h * 1.02;
        const bwNear = w * 0.07, bwFar = w * 0.006;
        // 地面坠落感
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.globalAlpha = 0.5;
        ctx.beginPath();
        for (let i = -10; i <= 10; i++) { ctx.moveTo(vx, vy); ctx.lineTo(vx + i * w * 0.18, by + h * 0.4); }
        ctx.stroke(); ctx.globalAlpha = 1;
        // 横梁
        ctx.fillStyle = C.grid;
        ctx.beginPath(); ctx.moveTo(vx - bwFar, vy); ctx.lineTo(vx + bwFar, vy); ctx.lineTo(vx + bwNear, by); ctx.lineTo(vx - bwNear, by); ctx.closePath(); ctx.fill();
        ctx.strokeStyle = C.muted; ctx.globalAlpha = 0.6;
        ctx.beginPath();
        for (let k = 0; k < 12; k++) {
          const u = ((k / 12 + t * 0.12) % 1), d = Math.pow(u, 2.2);
          const y = vy + (by - vy) * d, hw = lerp(bwFar, bwNear, d);
          ctx.moveTo(vx - hw, y); ctx.lineTo(vx + hw, y);
        }
        ctx.stroke(); ctx.globalAlpha = 1;
        // 背面人形
        const ph = t * 4.2, s = h * 0.62, footY = h * 0.84;
        const sway = Math.sin(ph) * s * 0.025, roll = -Math.sin(ph) * 0.06;
        const bob = Math.abs(Math.cos(ph)) * s * 0.012;
        const pel = { x: vx + sway, y: footY - s * 0.46 - bob };
        const hw = s * 0.055, sw = s * 0.11;
        const neck = { x: pel.x + Math.sin(roll) * s * 0.3, y: pel.y - Math.cos(roll) * s * 0.3 };
        ctx.lineCap = "round"; ctx.lineWidth = Math.max(2, s * 0.032); ctx.strokeStyle = C.fg;
        const ln = (a, b) => { ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); };
        ctx.beginPath();
        ln(pel, neck);
        const shL = { x: neck.x - sw * Math.cos(roll), y: neck.y - sw * Math.sin(roll) }, shR = { x: neck.x + sw * Math.cos(roll), y: neck.y + sw * Math.sin(roll) };
        ln(shL, shR);
        const armA = 1.25 + Math.sin(ph + 0.6) * 0.25, armB = 1.25 - Math.sin(ph + 0.6) * 0.25;
        const elL = { x: shL.x - Math.sin(armA) * s * 0.15, y: shL.y + Math.cos(armA) * s * 0.15 };
        const elR = { x: shR.x + Math.sin(armB) * s * 0.15, y: shR.y + Math.cos(armB) * s * 0.15 };
        ln(shL, elL); ln(elL, { x: elL.x - Math.sin(armA - 0.3) * s * 0.14, y: elL.y + Math.cos(armA - 0.3) * s * 0.14 });
        ln(shR, elR); ln(elR, { x: elR.x + Math.sin(armB - 0.3) * s * 0.14, y: elR.y + Math.cos(armB - 0.3) * s * 0.14 });
        const hipL = { x: pel.x - hw, y: pel.y }, hipR = { x: pel.x + hw, y: pel.y };
        const liftL = Math.max(0, Math.sin(ph)) * s * 0.07, liftR = Math.max(0, -Math.sin(ph)) * s * 0.07;
        const fL = { x: vx - s * 0.018, y: footY - liftL }, fR = { x: vx + s * 0.018, y: footY - liftR };
        const kn = (hp, f, lift, dir) => ({ x: (hp.x + f.x) / 2 + dir * (s * 0.012 + lift * 0.5), y: (hp.y + f.y) / 2 - lift * 0.2 });
        const kL = kn(hipL, fL, liftL, -1), kR = kn(hipR, fR, liftR, 1);
        ln(hipL, hipR); ln(hipL, kL); ln(kL, fL); ln(hipR, kR); ln(kR, fR);
        ctx.stroke();
        ctx.lineWidth = Math.max(1.5, s * 0.025);
        ctx.beginPath(); ctx.arc(neck.x + Math.sin(roll) * s * 0.11, neck.y - Math.cos(roll) * s * 0.11, s * 0.062, 0, TAU); ctx.stroke();
        // CoM / ZMP
        const com = { x: pel.x + (neck.x - pel.x) * 0.3, y: pel.y - s * 0.08 };
        const zmp = { x: vx + sway * 0.4, y: footY };
        ctx.strokeStyle = C.accent; ctx.fillStyle = C.accent; ctx.setLineDash([3, 4]); ctx.lineWidth = 1;
        ctx.beginPath(); ln(com, zmp); ctx.stroke(); ctx.setLineDash([]);
        ctx.beginPath(); ctx.arc(com.x, com.y, 4.5, 0, TAU); ctx.fill();
        ctx.beginPath(); ctx.ellipse(zmp.x, zmp.y, 7, 2.5, 0, 0, TAU); ctx.fill();
        label(ctx, "CoM", com.x + 9, com.y + 4, C, 11, C.accent);
        label(ctx, "ZMP ∈ support", zmp.x + 12, zmp.y + 4, C, 11, C.accent);
        label(ctx, "beam width ≈ foot width", 12, 20, C, 11);
      },
    },

    loopsr: {
      poster: 0.4,
      draw(ctx, w, h, t, C, st) {
        const gy = h * 0.74, s = Math.min(w, h * 1.6) * 0.55;
        const scroll = t * s * 0.9, loop = Math.floor(t / 3.2);
        const rough = 0.012 + (loop % 4) * 0.008;
        const ter = (x) => gy - s * rough * (Math.sin((x + scroll) * 0.025) + 0.6 * Math.sin((x + scroll) * 0.061 + 1.3));
        ctx.strokeStyle = C.muted; ctx.lineWidth = 1.5; ctx.beginPath();
        for (let x = 0; x <= w; x += 4) x === 0 ? ctx.moveTo(x, ter(x)) : ctx.lineTo(x, ter(x));
        ctx.stroke();
        ctx.fillStyle = C.grid; ctx.globalAlpha = 0.5; ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
        // 机器狗
        const cx = w * 0.42, bodyL = s * 0.42, legL = s * 0.16, bodyY = gy - legL * 1.75;
        const ph = t * 8;
        const hips = [[-bodyL / 2 + 6, 0], [bodyL / 2 - 6, Math.PI], [-bodyL / 2 + 6 + 10, Math.PI], [bodyL / 2 - 6 - 10, 0]];
        const pitch = Math.sin(ph * 2) * 0.015;
        hips.forEach(([hx, off], i) => {
          const H = { x: cx + hx, y: bodyY + hx * pitch };
          const p = ph + off, fx = H.x + Math.cos(p) * legL * 0.45, fy = ter(fx) - Math.max(0, Math.sin(p)) * legL * 0.35;
          const dx = fx - H.x, dy = fy - H.y, d = clamp(Math.hypot(dx, dy), 1, legL * 1.98);
          const a = Math.atan2(dy, dx), b = Math.acos(clamp(d / (2 * legL), -1, 1));
          const K = { x: H.x + legL * Math.cos(a + b), y: H.y + legL * Math.sin(a + b) };
          ctx.strokeStyle = i >= 2 ? C.muted : C.fg; ctx.lineWidth = Math.max(2, s * 0.028); ctx.lineCap = "round";
          ctx.beginPath(); ctx.moveTo(H.x, H.y); ctx.lineTo(K.x, K.y); ctx.lineTo(fx, fy); ctx.stroke();
        });
        ctx.fillStyle = C.fg; ctx.save(); ctx.translate(cx, bodyY); ctx.rotate(pitch);
        roundRect(ctx, -bodyL / 2, -s * 0.07, bodyL, s * 0.1, s * 0.03); ctx.fill();
        ctx.fillStyle = C.accent; ctx.fillRect(bodyL / 2 - s * 0.05, -s * 0.05, s * 0.03, s * 0.025);
        ctx.restore();
        // sim <-> real loop
        const lx = w - Math.min(150, w * 0.36), ly = 16, lw = Math.min(138, w * 0.34), lh = 52;
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.ellipse(lx + lw / 2, ly + lh / 2, lw / 2 - 18, lh / 2 - 8, 0, 0, TAU); ctx.stroke();
        const u = (t / 3.2) % 1, ang = u * TAU - Math.PI;
        const dot = { x: lx + lw / 2 + Math.cos(ang) * (lw / 2 - 18), y: ly + lh / 2 + Math.sin(ang) * (lh / 2 - 8) };
        ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(dot.x, dot.y, 4, 0, TAU); ctx.fill();
        label(ctx, "SIM", lx, ly + lh / 2 + 4, C, 11, u < 0.5 ? C.muted : C.accent, "left");
        label(ctx, "REAL", lx + lw, ly + lh / 2 + 4, C, 11, u >= 0.5 ? C.muted : C.accent, "right");
        label(ctx, "lifelong loop #" + (loop + 1) + "  ·  policy v" + (loop + 1), 12, 20, C, 11);
      },
    },

    selfplay: {
      poster: 6,
      draw(ctx, w, h, t, C, st) {
        const cx = w / 2, cy = h * 0.56, r = Math.min(w * 0.36, h * 0.42);
        const V = [0, 1, 2].map((i) => ({ x: cx + r * Math.cos(-Math.PI / 2 + i * TAU / 3), y: cy + r * Math.sin(-Math.PI / 2 + i * TAU / 3) }));
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(V[0].x, V[0].y); V.forEach((v) => ctx.lineTo(v.x, v.y)); ctx.closePath(); ctx.stroke();
        ["π₁ rock", "π₂ paper", "π₃ scissors"].forEach((n, i) => label(ctx, n, V[i].x, V[i].y + (i === 0 ? -8 : 18), C, 11, C.muted, "center"));
        // 虚拟博弈：平均策略向纳什均衡收敛
        const epoch = Math.floor(t / 9);
        if (st.epoch !== epoch) {
          const R = rng(epoch * 7 + 3); const a = R(), b = R() * (1 - a);
          st.epoch = epoch; st.x = [a, b, 1 - a - b]; st.n = 3; st.path = []; st.t0 = t;
        }
        const steps = Math.floor((t - st.t0) * 60);
        const M = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]];
        while (st.path.length < steps && st.path.length < 520) {
          const pay = M.map((row) => row[0] * st.x[0] + row[1] * st.x[1] + row[2] * st.x[2]);
          const br = pay.indexOf(Math.max.apply(null, pay));
          st.n += 1; st.x = st.x.map((v, i) => v + ((i === br ? 1 : 0) - v) / st.n);
          st.path.push(st.x.slice());
        }
        const pt = (x) => ({ x: x[0] * V[0].x + x[1] * V[1].x + x[2] * V[2].x, y: x[0] * V[0].y + x[1] * V[1].y + x[2] * V[2].y });
        ctx.strokeStyle = C.fg; ctx.lineWidth = 1.4;
        st.path.forEach((x, i) => {
          if (!i) return; const a = pt(st.path[i - 1]), b = pt(x);
          ctx.globalAlpha = 0.15 + 0.85 * (i / st.path.length); ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        });
        ctx.globalAlpha = 1;
        const c = pt([1 / 3, 1 / 3, 1 / 3]);
        ctx.strokeStyle = C.accent; ctx.beginPath(); ctx.arc(c.x, c.y, 7, 0, TAU); ctx.stroke();
        label(ctx, "Nash", c.x + 11, c.y + 4, C, 11, C.accent);
        if (st.path.length) { const p = pt(st.path[st.path.length - 1]); ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(p.x, p.y, 4.5, 0, TAU); ctx.fill(); }
        label(ctx, "offline data → best response → average policy", 12, 20, C, 11);
      },
    },

    traffic: {
      poster: 2,
      draw(ctx, w, h, t, C, st) {
        if (!st.nodes || st.w !== w || st.h !== h) {
          const R = rng(11), cols = 7, rows = 5; st.w = w; st.h = h; st.nodes = [];
          for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++)
            st.nodes.push({ x: w * (0.08 + 0.84 * i / (cols - 1)) + (R() - 0.5) * w * 0.05, y: h * (0.2 + 0.7 * j / (rows - 1)) + (R() - 0.5) * h * 0.06, i, j });
          st.adj = st.nodes.map((n) => st.nodes.map((m, k) => (Math.abs(m.i - n.i) + Math.abs(m.j - n.j) === 1 ? k : -1)).filter((k) => k >= 0));
          st.cars = [0, 1, 2, 3, 4, 5].map((k) => ({ path: [Math.floor(R() * st.nodes.length)], R: rng(100 + k) }));
        }
        ctx.strokeStyle = C.grid; ctx.lineWidth = 3; ctx.lineCap = "round"; ctx.beginPath();
        st.adj.forEach((a, i) => a.forEach((k) => { if (k > i) { ctx.moveTo(st.nodes[i].x, st.nodes[i].y); ctx.lineTo(st.nodes[k].x, st.nodes[k].y); } }));
        ctx.stroke();
        const speed = 1.4;
        st.cars.forEach((car, ci) => {
          const need = Math.floor(t * speed) + 2;
          while (car.path.length < need) { const last = car.path[car.path.length - 1], nb = st.adj[last]; car.path.push(nb[Math.floor(car.R() * nb.length)]); }
          const k = Math.floor(t * speed), u = t * speed - k;
          const col = ci === 0 ? C.accent : C.fg;
          ctx.strokeStyle = col; ctx.lineWidth = ci === 0 ? 2.5 : 1.5; ctx.globalAlpha = 0.65; ctx.beginPath();
          for (let q = Math.max(0, k - 4); q <= k; q++) { const n = st.nodes[car.path[q]]; q === Math.max(0, k - 4) ? ctx.moveTo(n.x, n.y) : ctx.lineTo(n.x, n.y); }
          const a = st.nodes[car.path[k]], b = st.nodes[car.path[k + 1]];
          const p = { x: lerp(a.x, b.x, u), y: lerp(a.y, b.y, u) };
          ctx.lineTo(p.x, p.y); ctx.stroke(); ctx.globalAlpha = 1;
          ctx.fillStyle = col; ctx.beginPath(); ctx.arc(p.x, p.y, ci === 0 ? 4.5 : 3, 0, TAU); ctx.fill();
        });
        label(ctx, "generated trajectories on a road graph", 12, 20, C, 11);
      },
    },
  };

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }

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

  // 有真实素材就用素材，否则挂程序化动画
  function media(el, pub, opts) {
    if (pub.media) {
      const isVideo = /\.(mp4|webm|mov)$/i.test(pub.media);
      const m = document.createElement(isVideo ? "video" : "img");
      m.src = pub.media; m.alt = pub.short + " demo";
      if (isVideo) { m.muted = true; m.loop = true; m.playsInline = true; m.autoplay = !(opts && opts.mode === "hover"); }
      m.style.cssText = "display:block;width:100%;height:100%;object-fit:cover";
      el.appendChild(m);
      if (isVideo && opts && opts.mode === "hover") {
        el.addEventListener("mouseenter", () => m.play()); el.addEventListener("mouseleave", () => m.pause());
      }
      return null;
    }
    return mount(el, pub.demo, Object.assign({ label: pub.short + " illustration" }, opts));
  }

  window.Demos = { mount, media, refresh: () => instances.forEach((i) => { i.colors = readColors(i.el); render(i); }) };
})();
