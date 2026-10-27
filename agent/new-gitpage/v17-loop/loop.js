/* V17 的 agent–environment 循环：一个点沿循环缓慢移动，每转一圈 t 加一。 */
window.Loop = {
  draw(host, photo) {
    const W = 560, H = 380;
    const svg = host.querySelector("svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    // 循环路径：agent → action 弧 → environment → state 弧 → agent；直线段藏在方框后面
    const path = "M 390 85 C 490 85, 490 285, 390 285 L 170 285 C 70 285, 70 85, 170 85 Z";
    svg.innerHTML = `
      <defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="context-stroke"/></marker></defs>
      <path d="M 390 85 C 490 85, 490 285, 394 285" class="arr" stroke="var(--agent)" marker-end="url(#ah)"/>
      <path d="M 170 285 C 70 285, 70 85, 166 85" class="arr" stroke="var(--env)" marker-end="url(#ah)"/>
      <path d="M 170 268 C 104 268, 104 102, 166 102" class="arr" stroke="var(--rew)" stroke-dasharray="4 4" marker-end="url(#ah)"/>
      <path id="lp" d="${path}" fill="none" stroke="none"/>
      <circle id="tok" r="5.5" fill="var(--rew)"/>
      <rect class="box ag" x="170" y="40" width="220" height="90" rx="12"/>
      <image href="${photo}" x="184" y="51" width="52" height="68" preserveAspectRatio="xMidYMin slice"/>
      <text class="pi" x="252" y="80">π<tspan font-size="14" dy="5">θ</tspan></text><text class="ttl" x="284" y="78">Agent</text>
      <text class="sub" x="252" y="106">Weiji Xie</text>
      <rect class="box en" x="170" y="240" width="220" height="90" rx="12"/>
      <text class="ttl" x="192" y="278">Environment</text><text class="sub" x="192" y="304">robots · data · real world</text>
      <text class="lab" x="380" y="170" text-anchor="end" fill="var(--agent)">action aₜ <tspan class="v">build, test, ship</tspan></text>
      <text class="lab" x="180" y="196" fill="var(--env)">state sₜ₊₁ <tspan class="v">what changed</tspan></text>
      <text class="lab" x="180" y="219" fill="var(--rew)">reward rₜ <tspan class="v">what worked</tspan></text>
      <text class="ticker" x="${W / 2}" y="${H - 14}" text-anchor="middle" id="tick">t = 0</text>`;
    const lp = svg.querySelector("#lp"), tok = svg.querySelector("#tok"), tick = svg.querySelector("#tick"), L = lp.getTotalLength();
    const LAP = 7000;
    if (Site.calm) { const pt = lp.getPointAtLength(0); tok.setAttribute("cx", pt.x); tok.setAttribute("cy", pt.y); tick.textContent = "t = 2026"; return; }
    const t0 = performance.now();
    (function step(now) {
      const u = ((now - t0) % LAP) / LAP, pt = lp.getPointAtLength(u * L);
      tok.setAttribute("cx", pt.x); tok.setAttribute("cy", pt.y);
      tick.textContent = `t = ${Math.floor((now - t0) / LAP)} · keep improving`;
      requestAnimationFrame(step);
    })(t0);
  },
};
