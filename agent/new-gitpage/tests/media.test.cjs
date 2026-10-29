const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

// Exercise the real media binding with controlled browser media events.
function setup(play = () => Promise.resolve(), initialize = false) {
  const classes = new Set(), hoverEvents = {}, mediaEvents = {};
  let video;
  const host = { addEventListener: (name, listener) => { hoverEvents[name] = listener; } };
  const el = {
    dataset: { video: "https://example.test/demo.mp4" },
    classList: { add: (name) => classes.add(name), remove: (name) => classes.delete(name) },
    closest: () => host,
    appendChild: (child) => { video = child; },
  };
  const document = {
    querySelectorAll: () => el.dataset.bound ? [] : [el],
    addEventListener() {},
    createElement: () => ({
      paused: true,
      play() { this.paused = false; return play(); },
      pause() { this.paused = true; },
      addEventListener: (name, listener) => { mediaEvents[name] = listener; },
    }),
  };
  const window = { SITE: { profile: {} } };
  const source = fs.readFileSync(path.join(__dirname, "../home/site.js"), "utf8");
  vm.runInNewContext(source, { window, document });
  if (initialize) window.Site.init();
  else window.Site.bindMedia(document);
  return {
    enter: () => hoverEvents.mouseenter(),
    leave: () => hoverEvents.mouseleave(),
    emit: (name) => mediaEvents[name]?.(),
    visible: () => classes.has("playing"),
    video: () => video,
  };
}

test("keep thumbnail visible until the video actually starts playing", () => {
  const f = setup();
  f.enter();
  assert.equal(f.visible(), false);
  f.emit("playing");
  assert.equal(f.visible(), true);
});

test("a rejected play request leaves the thumbnail visible", async () => {
  const f = setup(() => Promise.reject(new Error("playback blocked")));
  f.enter();
  await Promise.resolve();
  assert.equal(f.visible(), false);
});

test("a media error restores the thumbnail after playback starts", () => {
  const f = setup();
  f.enter();
  f.emit("playing");
  f.emit("error");
  assert.equal(f.visible(), false);
});

test("buffering shows the thumbnail and playback resumes the video", () => {
  const f = setup();
  f.enter();
  f.emit("playing");
  f.emit("waiting");
  assert.equal(f.visible(), false);
  f.emit("playing");
  assert.equal(f.visible(), true);
});

test("late playback after mouseleave cannot reveal or keep playing the video", () => {
  const f = setup();
  f.enter();
  f.leave();
  f.video().paused = false;
  f.emit("playing");
  assert.equal(f.visible(), false);
  assert.equal(f.video().paused, true);
});

test("an old rejected request cannot hide a newer successful playback", async () => {
  let rejectOld, calls = 0;
  const old = new Promise((resolve, reject) => { rejectOld = reject; });
  const f = setup(() => ++calls === 1 ? old : Promise.resolve());
  f.enter();
  f.leave();
  f.enter();
  f.emit("playing");
  rejectOld(new Error("superseded request"));
  await Promise.resolve();
  assert.equal(f.visible(), true);
});

test("CV initialization uses the same thumbnail fallback on playback failure", async () => {
  const f = setup(() => Promise.reject(new Error("video unavailable")), true);
  f.enter();
  await Promise.resolve();
  assert.equal(f.visible(), false);
});
