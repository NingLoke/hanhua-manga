/* 一键汉化漫画 4.0 —— 界面脚本（原生 JS，无外部依赖、无构建步骤，离线可用）
 * 结构：错误角标 → 工具 → 后端通信（POST /api/<方法>、SSE /events）→ 状态 → toast / 弹窗 / tooltip
 *      → 主题 / 路由 / 顶栏侧栏 / 拖放 / 快捷键 → 任务 / 校对 / 术语表 / 设置 / 关于 → 启动
 * URL 参数：t=令牌；theme=light|dark（截图用）；shot=1（截图模式：关动画、不连 SSE）
 */
(function () {
"use strict";

// ===================================================================== 错误角标（最先装：后面任何脚本错误都写到左下角并记日志）
const Q = new URLSearchParams(location.search);
const TOKEN = Q.get("t") || "";
const SHOT = Q.has("shot");
const ERR_BOX = document.getElementById("jsErrors");
let errCount = 0;

function reportError(message, source, line, stack) {
  const msg = String(message || "未知错误").slice(0, 300);
  errCount++;
  if (errCount > 30) return;
  try {
    ERR_BOX.hidden = false;
    const d = document.createElement("div");
    const where = source ? `（${String(source).split("/").pop().split("?")[0]}:${line || 0}）` : "";
    d.textContent = "界面出错：" + msg + where;
    ERR_BOX.appendChild(d);
    while (ERR_BOX.childElementCount > 4) ERR_BOX.firstElementChild.remove();
    ERR_BOX.title = "点击关闭（详细信息已写入日志）";
  } catch (_) { /* 角标本身出错就算了 */ }
  try {
    fetch("/api/ui_error", {
      method: "POST", headers: { "Content-Type": "application/json", "X-Token": TOKEN },
      body: JSON.stringify({ message: msg, source: String(source || ""), line: Number(line) || 0, stack: String(stack || "").slice(0, 2000) }),
    }).catch(() => {});
  } catch (_) { /* 忽略 */ }
}
window.onerror = (msg, src, line, col, err) => { reportError((err && err.message) || msg, src, line, err && err.stack); };
window.addEventListener("unhandledrejection", ev => {
  const r = ev.reason;
  if (r && r.isApi) { ev.preventDefault(); toast(r.message, "error"); return; }
  reportError((r && r.message) || String(r), "promise", 0, r && r.stack);
});
ERR_BOX.addEventListener("click", () => { ERR_BOX.hidden = true; ERR_BOX.textContent = ""; });

// ===================================================================== 工具
const $ = (s, r) => (r || document).querySelector(s);
const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ESC[c]);
const ico = (n, c) => `<svg class="${c || "ico"}" aria-hidden="true"><use href="#i-${n}"/></svg>`;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const sleep = ms => new Promise(r => setTimeout(r, ms));
const basename = p => String(p || "").split(/[\\/]/).filter(Boolean).pop() || String(p || "");
const isTyping = t => !!t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));

function el(html) { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; }
function debounce(fn, ms) {
  let t = 0, args = null;
  const d = (...a) => { args = a; clearTimeout(t); t = setTimeout(() => { t = 0; fn(...args); }, ms); };
  d.flush = () => { if (t) { clearTimeout(t); t = 0; return fn(...args); } };
  d.cancel = () => { clearTimeout(t); t = 0; };
  return d;
}
function getPath(o, path) { return path.split(".").reduce((x, k) => (x == null ? undefined : x[k]), o); }
function patchOf(path, value) {
  const keys = path.split("."), root = {};
  let cur = root;
  keys.forEach((k, i) => { cur[k] = i === keys.length - 1 ? value : {}; cur = cur[k]; });
  return root;
}
function lsGet(k) { try { return localStorage.getItem(k); } catch (_) { return null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); } catch (_) { /* 隐私模式 */ } }
function fmtBytes(n) {
  n = Math.max(0, Number(n) || 0);
  if (n >= 1024 ** 3) return (n / 1024 ** 3).toFixed(n >= 100 * 1024 ** 3 ? 0 : n >= 10 * 1024 ** 3 ? 1 : 2) + " GB";
  if (n >= 1024 ** 2) return (n / 1024 ** 2).toFixed(n >= 100 * 1024 ** 2 ? 0 : 1) + " MB";
  return Math.round(n / 1024) + " KB";
}
const fmtGB = gb => (Number(gb) >= 10 ? Number(gb).toFixed(0) : Number(gb || 0).toFixed(1)) + " GB";
function fmtDur(s) {
  s = Math.max(0, Math.round(Number(s) || 0));
  if (s < 60) return s + " 秒";
  const m = Math.floor(s / 60);
  if (m < 60) return m + " 分" + (m < 10 && s % 60 ? " " + (s % 60) + " 秒" : "");
  const h = Math.floor(m / 60);
  return h + " 小时" + (m % 60 ? " " + (m % 60) + " 分" : "");
}
function shortPath(p) {
  const s = String(p || "");
  if (s.length <= 64) return s;
  const parts = s.split(/[\\/]/);
  return parts.length > 3 ? parts[0] + "\\…\\" + parts.slice(-2).join("\\") : "…" + s.slice(-60);
}
function rgbToHex(c) {
  if (!Array.isArray(c) || c.length !== 3) return "";
  return "#" + c.map(v => clamp(Math.round(v), 0, 255).toString(16).padStart(2, "0")).join("");
}
function hexToRgb(h) {
  const m = /^#?([0-9a-f]{6})$/i.exec(h || "");
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// ===================================================================== 后端通信
class ApiErr extends Error { constructor(m) { super(m); this.isApi = true; } }

async function api(name, params, opt) {
  opt = opt || {};
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), opt.timeout || 90000);
  let res;
  try {
    res = await fetch("/api/" + name, {
      method: "POST", headers: { "Content-Type": "application/json", "X-Token": TOKEN },
      body: JSON.stringify(params || {}), signal: ctl.signal,
    });
  } catch (e) {
    throw new ApiErr(e && e.name === "AbortError" ? "后台处理超时，请稍后重试" : "连接不上后台程序，请确认软件仍在运行");
  } finally { clearTimeout(timer); }
  let data = null;
  try { data = await res.json(); } catch (_) { /* 不是 JSON */ }
  if (!data) throw new ApiErr(`后台返回了无法识别的内容（HTTP ${res.status}）`);
  if (!data.ok) throw new ApiErr(data.error || "操作失败");
  return data.data;
}
/** 调接口，失败自动弹出中文 toast；返回 undefined 表示失败。 */
async function call(name, params, opt) {
  try { return await api(name, params, opt); } catch (e) {
    if (!(opt && opt.quiet)) toast((e && e.message) || String(e), "error");
    return undefined;
  }
}
/** 按钮忙碌态（转圈 + 禁用），防止重复点击。 */
async function busy(btn, fn) {
  if (btn) {
    if (btn.classList.contains("busy")) return undefined;
    btn.classList.add("busy"); btn.disabled = true;
  }
  try { return await fn(); } finally { if (btn && btn.isConnected) { btn.classList.remove("busy"); btn.disabled = false; } }
}

const STAGES = ["pending", "detected", "recognized", "translated", "cleaned", "rendered"];
const STAGE_NO = { pending: 0, detected: 1, recognized: 2, translated: 3, cleaned: 4, rendered: 5, error: 9 };
const STAGE_LABEL = { pending: "等待中", detected: "已检测", recognized: "已识别", translated: "已翻译", cleaned: "已去字", rendered: "已完成", error: "出错" };
const atLeast = (stage, target) => (STAGE_NO[stage] || 0) >= STAGE_NO[target] && stage !== "error";

function imgUrl(jobId, index, kind, ver, w) {
  let u = `/img/${encodeURIComponent(jobId)}/${index}/${kind}?t=${encodeURIComponent(TOKEN)}`;
  if (ver != null && ver !== "") u += "&v=" + encodeURIComponent(ver);
  if (w) u += "&w=" + w;
  return u;
}
const pageVer = p => (p ? `${p.version || 0}.${STAGE_NO[p.stage] || 0}` : "");
const fontUrl = (id, text) => `/font-preview?id=${encodeURIComponent(id || "")}&t=${encodeURIComponent(TOKEN)}&text=${encodeURIComponent(text || "")}`;

// 所有图片加载完成后淡入（.ok），加载失败的封面保留占位
document.addEventListener("load", e => {
  const t = e.target;
  if (t && t.tagName === "IMG") {
    t.classList.add("ok");
    const ph = t.parentNode && t.parentNode.querySelector(".cover-ph");
    if (ph) ph.hidden = true;
  }
}, true);
document.addEventListener("error", e => {
  const t = e.target;
  if (t && t.tagName === "IMG") { t.classList.remove("ok"); t.classList.add("broken"); }
}, true);

// ===================================================================== 状态
const S = {
  ready: false, app: {}, settings: null, defaults: null, enums: {}, hardware: {}, engines: [], fonts: [],
  formats: { books: [], pages: [] }, paths: {}, engine: { running: false, label: "", error: "" },
  jobs: new Map(),      // id → summary（插入顺序 = 队列顺序）
  pages: new Map(),     // job_id → [{index,name,stage,version,error}]
  sel: null,            // 任务页当前选中的任务
  llm: null,            // llm_status()
  presets: null,        // api_presets()
  testRes: {},          // 引擎测试结果 id → {ok, msg}
};
const VIEWS = {};
const ACT = {};

// ===================================================================== toast
const TOAST_ICONS = { success: "check-circle", error: "alert-circle", warning: "alert", info: "info" };
function toast(message, type, opt) {
  type = type || "info"; opt = opt || {};
  const root = $("#toasts");
  const acts = opt.actions || [];
  const t = el(`<div class="toast ${type}" role="${type === "error" ? "alert" : "status"}">${ico(TOAST_ICONS[type] || "info")}
    <div class="toast-body">${esc(message)}</div>${acts.map((a, i) => `<button class="toast-act" data-i="${i}">${esc(a.label)}</button>`).join("")}
    <button class="toast-x" aria-label="关闭">${ico("x")}</button></div>`);
  let timer = 0;
  const close = () => {
    if (!t.isConnected || t.classList.contains("out")) return;
    clearTimeout(timer); t.classList.add("out"); setTimeout(() => t.remove(), 240);
  };
  const arm = () => { clearTimeout(timer); timer = setTimeout(close, opt.duration || (type === "error" ? 7000 : acts.length ? 6500 : 3600)); };
  t.addEventListener("click", e => {
    const b = e.target.closest(".toast-act");
    if (b) { close(); try { acts[+b.dataset.i].run(); } catch (err) { reportError(err.message, "toast", 0, err.stack); } }
    else if (e.target.closest(".toast-x")) close();
  });
  t.addEventListener("mouseenter", () => clearTimeout(timer));
  t.addEventListener("mouseleave", arm);
  root.appendChild(t);
  while (root.childElementCount > 4) root.firstElementChild.remove();
  arm();
  return close;
}

// ===================================================================== 弹窗（危险操作确认 / 三选一）
let modalClose = null;
/**
 * modal({title, text, html, extra, icon, danger, ok, cancel, buttons:[{label,value,kind}], read(root,value)}) → Promise<value>
 * Esc / 点遮罩 = 取消（false）。
 */
function modal(o) {
  if (modalClose) modalClose(false);
  return new Promise(resolve => {
    const root = $("#modalRoot");
    const buttons = o.buttons || [
      { label: o.cancel || "取消", value: false },
      { label: o.ok || "确定", value: true, kind: o.danger ? "danger" : "primary" },
    ];
    const cls = k => (k === "primary" ? " btn-primary" : k === "danger" ? " btn-danger-solid" : k === "ghost" ? " btn-ghost" : "");
    root.innerHTML = `<div class="modal-backdrop"></div>
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="mdTitle">
        <div class="modal-body"><div class="modal-icon${o.danger ? " danger" : ""}">${ico(o.icon || (o.danger ? "alert" : "info"))}</div>
          <div style="flex:1;min-width:0"><div class="modal-title" id="mdTitle">${esc(o.title || "")}</div>
            ${o.text ? `<div class="modal-text">${esc(o.text)}</div>` : ""}${o.html ? `<div class="modal-text">${o.html}</div>` : ""}
            ${o.extra ? `<div class="modal-extra">${o.extra}</div>` : ""}</div></div>
        <div class="modal-foot">${buttons.map((b, i) => `<button class="btn${cls(b.kind)}" data-i="${i}">${esc(b.label)}</button>`).join("")}</div>
      </div>`;
    root.hidden = false;
    const prev = document.activeElement;
    const done = v => {
      if (!modalClose) return;
      modalClose = null;
      document.removeEventListener("keydown", onKey, true);
      const out = o.read ? o.read(root, v) : v;
      root.hidden = true; root.innerHTML = "";
      try { if (prev && prev.focus) prev.focus(); } catch (_) { /* 忽略 */ }
      resolve(out);
    };
    function onKey(e) { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); done(false); } }
    document.addEventListener("keydown", onKey, true);
    root.querySelector(".modal-foot").addEventListener("click", e => {
      const b = e.target.closest("button");
      if (b) done(buttons[+b.dataset.i].value);
    });
    root.querySelector(".modal-backdrop").addEventListener("click", () => done(false));
    modalClose = done;
    const focus = root.querySelector(".modal-foot .btn-primary, .modal-foot .btn-danger-solid") || root.querySelector(".modal-foot button:last-child");
    setTimeout(() => { if (focus && focus.isConnected) focus.focus(); }, 20);
  });
}
const confirmDanger = (title, text, ok, extra) => modal({ title, text, danger: true, ok: ok || "删除", extra });

// ===================================================================== tooltip（data-tip，可带 data-kbd）
const TIP = $("#tooltip");
let tipTimer = 0, tipFor = null;
function hideTip() { clearTimeout(tipTimer); tipFor = null; TIP.hidden = true; }
function showTip(t) {
  if (!t.isConnected || !t.dataset.tip) return;
  TIP.innerHTML = esc(t.dataset.tip) + (t.dataset.kbd ? `<span class="kbd">${esc(t.dataset.kbd)}</span>` : "");
  TIP.hidden = false;
  const r = t.getBoundingClientRect(), w = TIP.offsetWidth, h = TIP.offsetHeight;
  let x = r.left + r.width / 2 - w / 2, y = r.bottom + 8;
  if (y + h > innerHeight - 6) y = r.top - h - 8;
  TIP.style.left = clamp(x, 6, innerWidth - w - 6) + "px";
  TIP.style.top = Math.max(6, y) + "px";
}
document.addEventListener("mouseover", e => {
  if (SHOT) return;
  const t = e.target.closest ? e.target.closest("[data-tip]") : null;
  if (t === tipFor) return;
  hideTip();
  if (t) { tipFor = t; tipTimer = setTimeout(() => showTip(t), 420); }
});
document.addEventListener("mousedown", hideTip, true);
document.addEventListener("scroll", hideTip, true);
window.addEventListener("blur", hideTip);

// ===================================================================== 主题
const THEMES = { auto: ["monitor", "外观：跟随系统"], light: ["sun", "外观：浅色"], dark: ["moon", "外观：深色"] };
const mqDark = matchMedia("(prefers-color-scheme: dark)");
let themeOverride = Q.get("theme");      // 截图 / 调试用，点按钮后失效
const themePref = () => themeOverride || getPath(S.settings, "general.theme") || lsGet("hanhua.theme") || "auto";
function applyTheme(pref, animate) {
  const eff = pref === "light" || pref === "dark" ? pref : (mqDark.matches ? "dark" : "light");
  const root = document.documentElement;
  if (animate && !SHOT) { root.classList.add("theme-anim"); setTimeout(() => root.classList.remove("theme-anim"), 450); }
  root.setAttribute("data-theme", eff);
  const [icn, tip] = THEMES[pref] || THEMES.auto;
  const b = $("#themeBtn");
  b.innerHTML = ico(icn);
  b.dataset.tip = tip + "（点击切换）";
}
mqDark.addEventListener("change", () => { if (themePref() === "auto") applyTheme("auto", true); });
ACT["theme"] = async () => {
  const order = ["auto", "light", "dark"];
  const next = order[(order.indexOf(themePref()) + 1) % 3];
  themeOverride = null;
  lsSet("hanhua.theme", next);
  applyTheme(next, true);
  hideTip();
  await saveSettings({ general: { theme: next } }, { quiet: true });
};

// ===================================================================== 设置保存（即改即存）
let savedTimer = 0;
async function saveSettings(patch, opt) {
  const s = await call("save_settings", { patch }, opt);
  if (s) { applySettings(s, true); flashSaved(); }
  return s;
}
function flashSaved() {
  $$(".set-saved").forEach(x => x.classList.add("show"));
  clearTimeout(savedTimer);
  savedTimer = setTimeout(() => $$(".set-saved").forEach(x => x.classList.remove("show")), 1600);
}
function applySettings(s, fromSelf) {
  if (!s) return;
  const prevTheme = getPath(S.settings, "general.theme");
  S.settings = s;
  if (!themeOverride && s.general && s.general.theme !== prevTheme) {
    lsSet("hanhua.theme", s.general.theme);
    if (prevTheme !== undefined) applyTheme(s.general.theme, true);
  }
  renderTopbar();
  updateGlossBadge();
  for (const v of Object.values(VIEWS)) if (v.mounted && v.onSettings) v.onSettings(s, fromSelf);
}

// ===================================================================== 路由
let current = null, currentArgs = [];
function parseHash() {
  const p = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(s => { try { return decodeURIComponent(s); } catch (_) { return s; } });
  return { name: p[0] || "tasks", args: p.slice(1) };
}
const hashOf = (name, args) => "#/" + [name].concat(args || []).map(x => encodeURIComponent(String(x))).join("/");
function go(path) { const h = "#/" + path; if (location.hash === h) route(); else location.hash = h; }
function setRouteArgs(args) {
  currentArgs = args;
  const h = hashOf(current, args);
  if (location.hash !== h) history.replaceState(null, "", h);
}
let routing = false;
async function route() {
  if (!S.ready || routing) return;
  routing = true;
  try {
    let { name, args } = parseHash();
    if (!VIEWS[name]) name = "tasks";
    if (current && current !== name && VIEWS[current].leave && !(await VIEWS[current].leave())) {
      history.replaceState(null, "", hashOf(current, currentArgs));
      return;
    }
    const prev = current;
    current = name; currentArgs = args;
    $$(".nav-item").forEach(a => a.classList.toggle("active", a.dataset.route === name));
    $$(".view").forEach(v => v.classList.toggle("active", v.dataset.view === name));
    const v = VIEWS[name];
    if (!v.mounted) { v.mounted = true; v.mount($("#view-" + name)); }
    if (prev && prev !== name && VIEWS[prev].hide) VIEWS[prev].hide();
    hideTip();
    await v.show(args, prev !== name);
    document.title = (v.title ? v.title + " · " : "") + "一键汉化漫画 4.0.5";
  } catch (e) {
    reportError(e.message, "route", 0, e.stack);
  } finally { routing = false; }
}
window.addEventListener("hashchange", route);

// ===================================================================== 顶栏 / 侧栏
function renderTopbar() {
  $("#demoBadge").hidden = !S.app.demo;
  const hw = S.hardware || {};
  const chip = $("#hwChip");
  $("#hwChipText").textContent = hw.chip || "CPU";
  chip.classList.toggle("cpu", hw.device === "cpu");
  chip.dataset.tip = `${hw.device_label || "CPU"} 加速 · 点击查看硬件详情与版本推荐`;
  const prof = getPath(S.settings, "hardware.profile");
  $$("#profileSeg button").forEach(b => b.classList.toggle("on", b.dataset.v === prof));
}
$("#profileSeg").addEventListener("click", async e => {
  const b = e.target.closest("button[data-v]");
  if (!b || b.classList.contains("on")) return;
  const names = { fast: "极速", balanced: "均衡", quality: "高质量" };
  const s = await saveSettings({ hardware: { profile: b.dataset.v } });
  if (s) toast(`已切换到「${names[b.dataset.v]}」档位，新开始的任务生效`, "success", { duration: 2600 });
});
$("#themeBtn").setAttribute("data-act", "theme");
ACT["go-about"] = () => go("about");
ACT["reload"] = () => location.reload();

function overall() {
  const o = { total: 0, done: 0, active: 0, running: null, paused: 0, queued: 0, error: 0, doneJobs: 0 };
  for (const j of S.jobs.values()) {
    if (j.status === "running" || j.status === "queued" || j.status === "paused") {
      o.active++; o.total += j.total || 0; o.done += (j.done || 0) + (j.failed || 0);
    }
    if (j.status === "running" && !o.running) o.running = j;
    if (j.status === "paused") o.paused++;
    if (j.status === "queued") o.queued++;
    if (j.status === "error") o.error++;
    if (j.status === "done") o.doneJobs++;
  }
  return o;
}
function renderEngineCard() {
  const o = overall();
  const dot = $("#engineDot"), st = $("#engineState"), sub = $("#engineSub"), btn = $("#engineToggle");
  dot.className = "engine-dot" + (o.running ? " on" : o.paused || o.queued ? " paused" : "");
  if (S.engine.error) { st.textContent = "引擎不可用"; sub.textContent = "详情见页面顶部提示"; }
  else if (o.running) {
    const r = o.running;
    st.textContent = "正在汉化";
    sub.textContent = `${r.title} · ${r.done || 0}/${r.total || 0} 页` + (r.speed_ppm ? ` · ${r.speed_ppm} 页/分` : "");
  } else if (o.paused) { st.textContent = "已暂停"; sub.textContent = `${o.paused} 本书暂停中，点下面继续`; }
  else if (o.queued) { st.textContent = "排队中"; sub.textContent = `${o.queued} 本书等待开始`; }
  else { st.textContent = "空闲"; sub.textContent = S.jobs.size ? "全部完成，可以继续拖入漫画" : "拖入漫画即可开始"; }
  sub.title = sub.textContent;
  $("#engineFill").style.width = (o.total ? (o.done / o.total) * 100 : 0).toFixed(1) + "%";
  btn.hidden = !(o.running || o.paused || o.queued) || !!S.engine.error;
  btn.dataset.mode = o.running ? "pause" : "resume";
  btn.innerHTML = o.running ? `${ico("pause")}<span>全部暂停</span>` : `${ico("play")}<span>全部继续</span>`;
  btn.dataset.tip = o.running ? "暂停所有任务（进度会保存）" : "继续所有暂停的任务";
  const nb = $("#navTaskBadge");
  nb.hidden = !o.active; nb.textContent = o.active;
}
ACT["toggle-engine"] = async btn => {
  await busy(btn, async () => {
    if (btn.dataset.mode === "pause") { if (await call("pause")) toast("已暂停全部任务，进度已保存", "info"); }
    else if (await call("resume")) { await call("start", {}, { quiet: true }); }
  });
};
function updateGlossBadge() {
  const n = (getPath(S.settings, "translate.glossary") || []).length;
  const b = $("#navGlossBadge");
  b.hidden = !n; b.textContent = n > 999 ? "999+" : n;
}
function renderEngineBanner() {
  const b = $("#engineBanner");
  b.hidden = !S.engine.error;
  if (S.engine.error) b.innerHTML = `${ico("alert-circle")}<span>${esc(S.engine.error)}。界面可以照常浏览，但暂时无法开始新的汉化，请查看日志或重新安装。</span><button class="btn btn-sm" data-act="open-logs">打开日志</button>`;
}

// ===================================================================== 实时事件（SSE）
let es = null, connTimer = 0;
function connectEvents() {
  if (SHOT || es) return;
  es = new EventSource("/events?t=" + encodeURIComponent(TOKEN));
  es.onmessage = m => {
    let batch;
    try { batch = JSON.parse(m.data); } catch (_) { return; }
    for (const e of Array.isArray(batch) ? batch : [batch]) {
      try { onEvent(e); } catch (err) { reportError(err.message, "event:" + (e && e.type), 0, err.stack); }
    }
  };
  es.onerror = () => {
    if (!connTimer) connTimer = setTimeout(() => { $("#connBanner").hidden = false; }, 1600);
  };
}
const resync = debounce(async () => {
  const jobs = await call("list_jobs", {}, { quiet: true });
  if (jobs) setJobs(jobs);
  const s = await call("get_settings", {}, { quiet: true });
  if (s) applySettings(s, false);
  for (const id of Array.from(S.pages.keys())) if (!S.jobs.has(id)) S.pages.delete(id);
  for (const v of Object.values(VIEWS)) if (v.mounted && v.onResync) v.onResync();
  loadLlm();
}, 250);

function onEvent(e) {
  if (!e || typeof e !== "object") return;
  switch (e.type) {
    case "hello":
      clearTimeout(connTimer); connTimer = 0; $("#connBanner").hidden = true;
      resync();
      break;
    case "resync": resync(); break;
    case "job": if (e.job && e.job.id) upsertJob(e.job); break;
    case "removed": removeJob(e.job_id); break;
    case "page": onPageEvent(e); break;
    case "engine":
      S.engine.running = !!e.running;
      if (e.device) S.engine.label = e.device;
      renderEngineCard();
      break;
    case "op": onOpEvent(e); break;
    case "toast": if (e.message) toast(e.message, e.level || "info"); break;
    case "settings": applySettings(e.settings, false); break;
    case "llm": onLlmEvent(e); break;
    default: break;
  }
}
function onOpEvent(e) {
  const job = S.jobs.get(e.job_id);
  const title = job ? `《${job.title}》` : "";
  if (e.state === "error") { toast(`${title}${e.message || "操作失败"}`, "error"); return; }
  if (e.state !== "done") return;
  if (e.op === "export" && e.path) {
    toast(`${title}导出完成`, "success", { actions: [
      { label: "打开", run: () => call("open_path", { path: e.path }) },
      { label: "在文件夹中显示", run: () => call("reveal", { path: e.path }) },
    ] });
  } else toast(`${title}${e.message || "完成"}`, "success");
}

// ---- 任务 / 页面数据 ------------------------------------------------------------------
const removedJobIds = new Set();
function setJobs(list) {
  const m = new Map();
  for (const j of list || []) if (j && j.id && !j.removed && !removedJobIds.has(j.id)) m.set(j.id, j);
  S.jobs = m;
  if (S.sel && !m.has(S.sel)) S.sel = null;
  renderEngineCard();
  for (const v of Object.values(VIEWS)) if (v.mounted && v.onJobs) v.onJobs();
}
function upsertJob(j) {
  if (j.removed) { removeJob(j.id); return; }
  if (removedJobIds.has(j.id)) return;
  const isNew = !S.jobs.has(j.id);
  const old = S.jobs.get(j.id);
  S.jobs.set(j.id, j);
  if (old && old.status !== "done" && j.status === "done" && !SHOT) {
    toast(`《${j.title}》汉化完成` + (j.failed ? `（${j.failed} 页出错，已用原图）` : ""), j.failed ? "warning" : "success", {
      actions: j.out_path ? [{ label: "打开", run: () => call("open_path", { path: j.out_path }) },
                            { label: "校对", run: () => openEditor(j.id) }] : [{ label: "校对", run: () => openEditor(j.id) }],
    });
  }
  renderEngineCard();
  for (const v of Object.values(VIEWS)) if (v.mounted && v.onJob) v.onJob(j, isNew, old);
}
function removeJob(id) {
  removedJobIds.add(id);
  if (!S.jobs.has(id)) return;
  S.jobs.delete(id);
  S.pages.delete(id);
  if (S.sel === id) S.sel = null;
  renderEngineCard();
  for (const v of Object.values(VIEWS)) if (v.mounted && v.onRemoved) v.onRemoved(id);
}
async function loadPages(jobId, force) {
  if (!jobId) return [];
  if (!force && S.pages.has(jobId)) return S.pages.get(jobId);
  const list = await call("list_pages", { job_id: jobId }, { quiet: true });
  if (!list) return S.pages.get(jobId) || [];
  S.pages.set(jobId, list);
  return list;
}
function onPageEvent(e) {
  const list = S.pages.get(e.job_id);
  let prevStage = null;
  if (list) {
    const p = list[e.index];
    if (p) { prevStage = p.stage; p.stage = e.stage; p.version = e.version; }
    else if (e.index === list.length) list.push({ index: e.index, name: "", stage: e.stage, version: e.version, error: "" });
  }
  for (const v of Object.values(VIEWS)) if (v.mounted && v.onPage) v.onPage(e, prevStage);
}
function openEditor(jobId, index) {
  go("editor/" + encodeURIComponent(jobId) + (index != null ? "/" + index : ""));
}

// ===================================================================== 添加文件 / 拖放
function reportAdded(r) {
  const jobs = (r && r.jobs) || [], skipped = (r && r.skipped) || [];
  for (const j of jobs) { removedJobIds.delete(j.id); upsertJob(j); }
  if (jobs.length) {
    S.sel = jobs[jobs.length - 1].id;
    if (VIEWS.tasks.mounted) VIEWS.tasks.select(S.sel);
    toast(jobs.length === 1 ? `已添加《${jobs[0].title}》，开始汉化` : `已添加 ${jobs.length} 本，开始汉化`, "success");
  }
  if (skipped.length) {
    const s = skipped.slice(0, 2).map(x => (x.path ? basename(x.path) + "：" : "") + x.reason).join("；");
    toast(`${skipped.length} 项没有添加（${s}${skipped.length > 2 ? " 等" : ""}）`, "warning");
  }
  if (!jobs.length && !skipped.length) toast("没有找到可以处理的漫画文件", "warning");
}
async function addPaths(paths) {
  if (!paths.length) return;
  const r = await call("add_paths", { paths });
  if (r) { reportAdded(r); if (current !== "tasks") go("tasks"); }
}
ACT["pick-files"] = async btn => {
  if (!S.app.window && S.app.demo) { $("#fileInput").click(); return; }
  await busy(btn, async () => { const r = await call("pick_files"); if (r && r.paths && r.paths.length) { reportAdded(r); if (current !== "tasks") go("tasks"); } });
};
ACT["pick-folder"] = async btn => {
  if (!S.app.window && S.app.demo) { addPaths(["演示文件夹（浏览器模式）"]); return; }
  await busy(btn, async () => { const r = await call("pick_folder"); if (r && r.paths && r.paths.length) { reportAdded(r); if (current !== "tasks") go("tasks"); } });
};
$("#fileInput").addEventListener("change", e => {
  const names = Array.from(e.target.files || []).map(f => f.name);
  e.target.value = "";
  addPaths(names);
});

let dragDepth = 0;
const hasFiles = e => Array.from((e.dataTransfer && e.dataTransfer.types) || []).includes("Files");
function hideDrop() { $("#dropOverlay").hidden = true; const dz = $("#dropzone"); if (dz) dz.classList.remove("hot"); }
window.addEventListener("dragenter", e => {
  if (!hasFiles(e)) return;
  e.preventDefault(); dragDepth++;
  $("#dropOverlay").hidden = false;
  const dz = $("#dropzone"); if (dz) dz.classList.add("hot");
});
window.addEventListener("dragover", e => { if (hasFiles(e)) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; } });
window.addEventListener("dragleave", e => { if (!hasFiles(e)) return; dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) hideDrop(); });
window.addEventListener("drop", e => {
  if (!hasFiles(e)) return;
  e.preventDefault(); dragDepth = 0; hideDrop();
  const files = Array.from(e.dataTransfer.files || []);
  if (!files.length) return;
  if (S.app.window) {            // 桌面程序：pywebview 在 Python 侧拿到完整路径并添加，这里只给反馈
    toast(`正在添加 ${files.length} 项…`, "info", { duration: 1800 });
    if (current !== "tasks") go("tasks");
  } else if (S.app.demo) addPaths(files.map(f => f.name));
  else toast("浏览器调试模式拿不到文件的完整路径，请在桌面程序里拖入", "warning");
});

// ===================================================================== 全局点击 / 快捷键
document.addEventListener("click", e => {
  const t = e.target.closest("[data-act]");
  if (!t || t.disabled || t.getAttribute("aria-disabled") === "true") return;
  const f = ACT[t.dataset.act];
  if (!f) return;
  e.preventDefault();
  Promise.resolve().then(() => f(t, e)).catch(err => {
    if (err && err.isApi) toast(err.message, "error"); else reportError(err && err.message, "act:" + t.dataset.act, 0, err && err.stack);
  });
});
document.addEventListener("keydown", e => {
  if (modalClose) return;
  const ctrl = e.ctrlKey || e.metaKey, k = (e.key || "").toLowerCase();
  if (ctrl && k === "s") { e.preventDefault(); if (current === "editor") VIEWS.editor.save(); return; }
  if (ctrl && k === "o") { e.preventDefault(); ACT[e.shiftKey ? "pick-folder" : "pick-files"](null); return; }
  if (ctrl && !e.shiftKey && !e.altKey && /^[1-5]$/.test(k)) { e.preventDefault(); go(["tasks", "editor", "glossary", "settings", "about"][+k - 1]); return; }
  const v = VIEWS[current];
  if (v && v.onKey) v.onKey(e);
});
document.addEventListener("keyup", e => { const v = VIEWS[current]; if (v && v.onKeyUp) v.onKeyUp(e); });
window.addEventListener("beforeunload", e => {
  if (VIEWS.editor.mounted && VIEWS.editor.isDirty && VIEWS.editor.isDirty()) { e.preventDefault(); e.returnValue = ""; }
});
ACT["open-logs"] = () => call("open_logs");
ACT["open-data"] = () => call("open_data_dir");
ACT["go"] = t => go(t.dataset.to);

// ===================================================================== 空状态插画（内联 SVG，颜色跟主题走）
const ART = {
  drop: `<svg class="dz-art" viewBox="0 0 168 118" aria-hidden="true">
    <rect class="art-paper2" x="30" y="20" width="60" height="82" rx="8" transform="rotate(-9 60 61)"/>
    <rect class="art-paper" x="56" y="10" width="66" height="90" rx="9"/>
    <path class="art-line" d="M68 28h30M68 38h42M68 48h24"/>
    <rect class="art-accent-soft" x="66" y="60" width="46" height="28" rx="7"/>
    <path class="art-accent-stroke" d="M76 74h26"/>
    <path class="art-accent" d="M112 46c0-6.6 5.4-12 12-12h16c6.6 0 12 5.4 12 12v5c0 6.6-5.4 12-12 12h-7l-9 8v-8c-6.6 0-12-5.4-12-12z"/>
    <path d="M122 47.5h20M132 43v4.5M124.5 47.5c1.4 4 4 6.7 7.8 8.2M139.5 47.5c-1.4 4.2-4 7-8 8.6" stroke="#fff" stroke-width="2" fill="none" stroke-linecap="round"/>
    <circle class="art-accent" cx="40" cy="98" r="12"/>
    <path d="M40 104V92M35 96.5l5-5 5 5" stroke="#fff" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`,
  preview: `<svg class="empty-art" viewBox="0 0 148 112" aria-hidden="true">
    <rect class="art-paper" x="20" y="12" width="108" height="88" rx="10"/>
    <path class="art-accent-soft" d="M74 12h44a10 10 0 0 1 10 10v68a10 10 0 0 1-10 10H74z"/>
    <path class="art-line" d="M34 32h26M34 42h18M86 32h26M86 42h18"/>
    <path class="art-accent-stroke" d="M74 6v100"/>
    <circle class="art-white" cx="74" cy="56" r="11" stroke="var(--border-strong)" stroke-width="1.5"/>
    <path d="M70 52l-3.5 4 3.5 4M78 52l3.5 4-3.5 4" stroke="var(--accent)" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <rect class="art-paper2" x="32" y="66" width="30" height="20" rx="5"/><rect class="art-accent" x="86" y="66" width="30" height="20" rx="5" opacity=".85"/>
  </svg>`,
  editor: `<svg class="empty-art" viewBox="0 0 148 112" aria-hidden="true">
    <rect class="art-paper2" x="18" y="16" width="70" height="86" rx="9"/>
    <rect class="art-paper" x="30" y="8" width="70" height="88" rx="9"/>
    <rect class="art-accent-soft" x="40" y="22" width="34" height="22" rx="11"/>
    <path class="art-accent-stroke" d="M48 33h18"/>
    <path class="art-line" d="M42 60h44M42 70h30"/>
    <path class="art-accent" d="M104 34l18 18-30 30H74V64z"/>
    <path d="M98 40l18 18" stroke="#fff" stroke-width="2.2" stroke-linecap="round"/>
  </svg>`,
  book: `<svg class="empty-art" viewBox="0 0 148 112" aria-hidden="true">
    <path class="art-paper" d="M22 20c14-6 30-6 52 4v76c-22-10-38-10-52-4z"/>
    <path class="art-paper" d="M126 20c-14-6-30-6-52 4v76c22-10 38-10 52-4z"/>
    <path class="art-line" d="M34 38c10-3 18-2 28 2M34 50c10-3 18-2 28 2M86 40c10-4 18-5 28-2"/>
    <rect class="art-accent-soft" x="84" y="50" width="32" height="20" rx="6"/>
    <path class="art-accent-stroke" d="M91 60h18"/>
    <circle class="art-accent" cx="114" cy="88" r="11"/>
    <path d="M114 83v10M109 88h10" stroke="#fff" stroke-width="2.3" stroke-linecap="round"/>
  </svg>`,
  error: `<svg class="empty-art" viewBox="0 0 148 112" aria-hidden="true">
    <rect class="art-paper" x="30" y="14" width="88" height="84" rx="12"/>
    <path class="art-line" d="M46 36h40M46 48h56M46 60h30"/>
    <circle cx="104" cy="78" r="16" fill="var(--red)"/>
    <path d="M104 70v9M104 85v.5" stroke="#fff" stroke-width="3" stroke-linecap="round"/>
  </svg>`,
};
const emptyHtml = (art, title, sub, actions) => `<div class="empty">${ART[art] || ""}<div class="empty-title">${esc(title)}</div>${sub ? `<div class="empty-sub">${esc(sub)}</div>` : ""}${actions ? `<div class="dz-actions" style="margin:6px 0 0">${actions}</div>` : ""}</div>`;

// ===================================================================== 本地翻译模型（任务页横幅、设置页、关于页共用）
const LATER_KEY = "hanhua.llmLater";
const laterActive = () => { const t = Number(lsGet(LATER_KEY) || 0); return t > 0 && Date.now() - t < 3 * 864e5; };
const dlPct = d => (d && d.total ? clamp((d.done / d.total) * 100, 0, 100) : 0);
function dlText(d) {
  if (!d) return "";
  if (d.message && !d.speed) return (d.done ? fmtBytes(d.done) + " / " + fmtBytes(d.total) + " · " : "") + d.message;
  const parts = [`${fmtBytes(d.done)} / ${fmtBytes(d.total)}`];
  if (d.speed > 0) {
    parts.push(fmtBytes(d.speed) + "/s");
    if (d.total > d.done) parts.push("剩余约 " + fmtDur((d.total - d.done) / d.speed));
  }
  return parts.join(" · ");
}
let llmSeq = 0;
async function loadLlm() {
  const seq = ++llmSeq;
  const st = await call("llm_status", {}, { quiet: true, timeout: 30000 });
  if (!st || seq !== llmSeq) return;
  const cur = S.llm && S.llm.download;      // 事件里的进度可能比快照更新
  if (cur && st.download && cur.state === "downloading" && st.download.state === "downloading"
      && cur.model_id === st.download.model_id && cur.done > st.download.done) st.download = cur;
  S.llm = st;
  llmChanged();
}
function llmChanged() { for (const v of Object.values(VIEWS)) if (v.mounted && v.onLlm) v.onLlm(); }
function patchDownload(d) {
  const boxes = $$("[data-dl]").filter(b => b.dataset.dl === d.model_id);
  for (const b of boxes) {
    const f = b.querySelector(".dl-fill"), t = b.querySelector(".dl-text");
    if (f) f.style.width = dlPct(d).toFixed(1) + "%";
    if (t) t.textContent = dlText(d);
  }
  return boxes.length > 0;
}
function onLlmEvent(e) {
  if (!S.llm) { loadLlm(); return; }
  const prev = S.llm.download;
  S.llm.download = e;
  if (e.state === "downloading") {
    const same = prev && prev.state === "downloading" && prev.model_id === e.model_id;
    if (!(same && patchDownload(e))) llmChanged();
    return;
  }
  if (e.state === "done") toast(e.message || "模型下载完成", "success");
  else if (e.state === "error") toast(e.message || "模型下载失败", "error", { actions: [{ label: "重试", run: () => llmDownload(e.model_id) }] });
  else if (e.state === "canceled") toast(e.message || "已取消下载", "info");
  loadLlm();
}
const llmModel = id => ((S.llm && S.llm.models) || []).find(m => m.id === id);
async function llmDownload(id, btn) {
  const r = await busy(btn, () => call("llm_download", { model_id: id }));
  if (!r) return;
  const m = llmModel(id);
  if (S.llm && !(S.llm.download && S.llm.download.state === "downloading")) {
    S.llm.download = { model_id: id, name: m ? m.name : id, state: "downloading", done: (m && m.partial_bytes) || 0,
                       total: (m && m.size) || 0, speed: 0, message: "正在选择最快的下载源…" };
    llmChanged();
  }
}
ACT["llm-dl"] = btn => llmDownload(btn.dataset.id, btn);
ACT["llm-cancel"] = btn => busy(btn, () => call("llm_cancel"));
ACT["llm-later"] = () => {
  lsSet(LATER_KEY, String(Date.now()));
  llmChanged();
  toast("好的。之后可以在「设置 → 翻译」里随时下载本地模型", "info");
};
ACT["llm-del"] = async btn => {
  const m = llmModel(btn.dataset.id);
  const name = m ? m.name : btn.dataset.id;
  if (!(await confirmDanger("删除这个模型？", `将删除「${name}」的模型文件${m && m.size_gb ? `（${fmtGB(m.size_gb)}）` : ""}，需要时可以重新下载。`))) return;
  const st = await busy(btn, () => call("llm_delete", { model_id: btn.dataset.id }));
  if (st) { S.llm = st; llmChanged(); toast(`已删除「${name}」`, "success"); }
};
ACT["llm-import"] = async btn => {
  const r = await busy(btn, () => call("llm_import"));
  if (r && r.imported) { S.llm = r.status; llmChanged(); toast(`已导入「${r.imported.name}」并设为当前模型`, "success"); }
};

// ===================================================================== 任务页
const STATUS = { running: ["running", "汉化中"], queued: ["queued", "排队中"], paused: ["paused", "已暂停"],
                 done: ["done", "已完成"], error: ["error", "出错"], canceled: ["canceled", "已取消"] };
const BAR_ORDER = ["rendered", "cleaned", "translated", "recognized", "detected", "error"];
const TRIVIAL_MSG = new Set(["", "排队中", "已暂停", "已取消", "处理中", "全部完成", "已完成", "汉化中"]);
function kindLabel(k) {
  k = String(k || "").toLowerCase();
  return ({ folder: "文件夹", dir: "文件夹", images: "图片", image: "图片" })[k] || k.toUpperCase() || "漫画";
}
function jobParts(j) {
  const [cls, label] = STATUS[j.status] || ["queued", j.status || ""];
  const total = j.total || 0, done = j.done || 0, failed = j.failed || 0, sc = j.stage_counts || {};
  const run = j.status === "running";
  const stats = [`<span><b>${done}</b> / ${total} 页</span>`];
  if (j.layout_label) stats.push(`<span>${esc(j.layout_label)}</span>`);
  if (run && j.speed_ppm) stats.push(`<span>${j.speed_ppm} 页/分</span>`);
  if (run && j.eta_s) stats.push(`<span>剩余约 ${fmtDur(j.eta_s)}</span>`);
  if (j.status === "done" && j.speed_ppm) stats.push(`<span>平均 ${j.speed_ppm} 页/分</span>`);
  if (failed) stats.push(`<span class="err">${failed} 页出错</span>`);
  const meta = `<span class="ellipsis" title="${esc(j.source)}">${esc(shortPath(j.source))}</span>`
    + (j.out_path ? `<span class="sep">·</span><span class="ellipsis" title="${esc(j.out_path)}">输出：${esc(basename(j.out_path))}</span>` : "");
  const msg = String(j.message || "");
  return {
    top: `<div class="job-title ellipsis" title="${esc(j.title)}">${esc(j.title)}</div><span class="tag ${cls}">${esc(label)}</span>`,
    meta, stats: stats.join(""), actions: jobActions(j),
    widths: BAR_ORDER.map(k => (total ? ((sc[k] || 0) / total) * 100 : 0).toFixed(2) + "%"),
    msg: TRIVIAL_MSG.has(msg) || msg === label ? "" : msg,
  };
}
function jobActions(j) {
  const id = esc(j.id), st = j.status, a = [];
  const ib = (act, icon, tip, cls) => `<button class="btn-icon sm${cls || ""}" data-act="${act}" data-id="${id}" data-tip="${tip}">${ico(icon)}</button>`;
  if (st === "paused" || st === "canceled" || (st === "error" && !j.failed && j.total)) {
    a.push(`<button class="btn btn-sm btn-primary" data-act="job-resume" data-id="${id}">${ico("play")}继续</button>`);
  }
  if (j.failed && st !== "running") a.push(`<button class="btn btn-sm" data-act="job-retry" data-id="${id}">${ico("refresh")}重试出错页</button>`);
  if (st === "done" && j.out_path) a.push(`<button class="btn btn-sm" data-act="job-open" data-id="${id}">${ico("external")}打开</button>`);
  if (st === "running" || st === "queued") a.push(ib("job-pause", "pause", "暂停（进度会保存）"));
  if ((j.done || 0) > 0) a.push(ib("job-edit", "edit", "校对"));
  if (st === "done") {
    if (j.out_path) a.push(ib("job-reveal", "folder", "在文件夹中显示"));
    a.push(ib("job-export", "download", "重新导出"), ib("job-rerender", "type", "按当前排版设置重新嵌字全书"));
  }
  if (st === "running" || st === "queued" || st === "paused") a.push(ib("job-cancel", "stop", "取消任务"));
  a.push(ib("job-remove", "trash", "删除任务", " danger"));
  return a.join("");
}
function jobCardHtml(j) {
  const p = jobParts(j);
  return `<div class="job${j.id === S.sel ? " selected" : ""}" data-id="${esc(j.id)}">
    <div class="cover">${(j.total || 0) > 0 ? `<img alt="" loading="lazy" src="${imgUrl(j.id, j.cover || 0, "src", "c", 200)}">` : ""}<div class="cover-ph">${ico("book")}</div><span class="fmt">${esc(kindLabel(j.kind))}</span></div>
    <div class="job-body"><div class="job-top">${p.top}</div><div class="job-meta">${p.meta}</div>
      <div class="progress${j.status === "running" ? " running" : ""}">${BAR_ORDER.map((k, i) => `<i class="s-${k}" style="width:${p.widths[i]}"></i>`).join("")}</div>
      <div class="job-foot"><div class="job-stats">${p.stats}</div><div class="job-actions">${p.actions}</div></div>
      ${p.msg ? `<div class="job-msg${j.status === "error" ? " error" : ""}">${esc(p.msg)}</div>` : ""}</div></div>`;
}
function patchJobCard(card, j) {
  const p = jobParts(j);
  const set = (sel, html) => { const n = card.querySelector(sel); if (n && n._h !== html) { n.innerHTML = html; n._h = html; } };
  set(".job-top", p.top); set(".job-meta", p.meta); set(".job-stats", p.stats); set(".job-actions", p.actions);
  const bar = card.querySelector(".progress");
  bar.classList.toggle("running", j.status === "running");
  p.widths.forEach((w, i) => { if (bar.children[i]) bar.children[i].style.width = w; });
  let msg = card.querySelector(".job-msg");
  if (p.msg) {
    if (!msg) { msg = document.createElement("div"); card.querySelector(".job-body").appendChild(msg); }
    msg.className = "job-msg" + (j.status === "error" ? " error" : "");
    msg.textContent = p.msg;
  } else if (msg) msg.remove();
}
function thumbHtml(jid, p, cur) {
  const r = p.stage === "rendered";
  const tip = `第 ${p.index + 1} 页 · ${STAGE_LABEL[p.stage] || p.stage}${p.error ? "：" + p.error : ""}`;
  return `<div class="thumb${r ? "" : " pending"}${cur ? " cur" : ""}" data-i="${p.index}" title="${esc(tip)}">${r ? `<img alt="" loading="lazy" src="${imgUrl(jid, p.index, "thumb", pageVer(p))}">` : ""}<span class="no">${p.index + 1}</span><span class="sdot s-${p.stage}"></span></div>`;
}
const skelThumbs = n => Array.from({ length: n }, () => `<div class="thumb skel"></div>`).join("");
const FORMAT_CHIPS = ["文件夹", "JPG · PNG · WEBP", "ZIP · CBZ", "RAR · CBR", "7Z · CB7", "PDF", "EPUB"];
const TIPS3 = [["upload", "拖入漫画", "文件夹、压缩包、PDF、EPUB 都能直接拖进来，一次可以拖多本"],
               ["sparkle", "全自动汉化", "检测气泡 → 识别原文 → 翻译 → 去字 → 嵌字，按电脑配置自动加速"],
               ["edit", "校对与导出", "逐页对照修改译文和样式，保存后立即重新排版，一键导出成品"]];

function renderBootSkeleton() {
  const v = $("#view-tasks");
  v.classList.add("active");
  const job = `<div class="job" style="animation:none"><div class="skel" style="width:62px;height:86px"></div><div class="job-body">
    <div class="skel skel-line" style="width:42%"></div><div class="skel skel-line" style="width:68%"></div>
    <div class="skel skel-line" style="height:6px"></div><div class="skel skel-line" style="width:30%"></div></div></div>`;
  v.innerHTML = `<div class="tasks"><div class="tasks-main"><div class="skel" style="height:96px;border-radius:14px"></div>
    <div class="section-label"><span class="skel skel-line" style="width:110px"></span></div><div class="job-list">${job + job + job}</div></div>
    <aside class="tasks-side"><div class="card" style="padding:16px"><div class="skel skel-line" style="width:40%;margin-bottom:14px"></div>
    <div class="skel" style="aspect-ratio:.72;border-radius:10px"></div></div></aside></div>`;
}

VIEWS.tasks = {
  title: "任务",
  pv: { job: null, index: -1, pinned: false, pos: 50 },
  mount(root) {
    root.innerHTML = `<div class="tasks"><div class="tasks-main"><div id="llmBanner"></div><div id="dzWrap"></div><div id="jobHead"></div>
      <div class="job-list" id="jobList"></div><div id="jobTips"></div></div><aside class="tasks-side"><div class="preview" id="pvWrap"></div></aside></div>`;
    const list = $("#jobList", root);
    list.addEventListener("click", e => {
      if (e.target.closest("button, a")) return;
      const c = e.target.closest(".job");
      if (c) this.select(c.dataset.id);
    });
    list.addEventListener("dblclick", e => {
      if (e.target.closest("button")) return;
      const c = e.target.closest(".job");
      if (c) openEditor(c.dataset.id);
    });
    $("#dzWrap", root).addEventListener("click", e => { if (!e.target.closest("button") && e.target.closest(".dropzone")) ACT["pick-files"](null); });
    const side = $("#pvWrap", root);
    side.addEventListener("click", e => { const t = e.target.closest(".thumb"); if (t && !t.classList.contains("skel")) this.setPreview(+t.dataset.i, true); });
    side.addEventListener("dblclick", e => {
      const t = e.target.closest(".thumb, .compare");
      if (!t || !this.pv.job) return;
      openEditor(this.pv.job, t.classList.contains("thumb") ? +t.dataset.i : (this.pv.index >= 0 ? this.pv.index : undefined));
    });
    side.addEventListener("pointerdown", e => this.cmpDown(e));
  },
  show() { this.renderAll(); },
  renderAll() { this.renderBanner(); this.renderDropzone(); this.renderJobs(); this.renderPreview(); },
  ensureSel() {
    if (S.sel && S.jobs.has(S.sel)) return S.sel;
    const jobs = Array.from(S.jobs.values());
    const j = jobs.find(x => x.status === "running") || jobs.find(x => (x.done || 0) > 0) || jobs[0];
    S.sel = j ? j.id : null;
    return S.sel;
  },
  select(id) {
    if (!S.jobs.has(id)) return;
    S.sel = id;
    $$("#jobList .job").forEach(c => c.classList.toggle("selected", c.dataset.id === id));
    if (this.pv.job !== id) this.renderPreview();
  },
  renderDropzone() {
    const box = $("#dzWrap");
    if (!box) return;
    const compact = S.jobs.size > 0;
    box.innerHTML = compact
      ? `<div class="dropzone compact" id="dropzone"><div class="dz-icon">${ico("upload")}</div><div class="dz-text"><div class="dz-title">拖入更多漫画</div>
          <div class="dz-sub">文件夹、图片、压缩包、PDF、EPUB · 每次拖入创建独立任务，各版本分别保存</div></div>
          <div class="dz-actions"><button class="btn btn-primary" data-act="pick-files" data-tip="添加文件" data-kbd="Ctrl+O">${ico("file-plus")}添加文件</button>
          <button class="btn" data-act="pick-folder" data-tip="添加文件夹" data-kbd="Ctrl+Shift+O">${ico("folder-plus")}添加文件夹</button></div></div>`
      : `<div class="dropzone" id="dropzone">${ART.drop}<div class="dz-title">把漫画拖到这里，一键汉化</div>
          <div class="dz-sub">自动识别气泡、翻译、去字、嵌字 · 支持一次拖入多本</div>
          <div class="dz-actions"><button class="btn btn-primary btn-lg" data-act="pick-files">${ico("file-plus")}添加文件</button>
          <button class="btn btn-lg" data-act="pick-folder">${ico("folder-plus")}添加文件夹</button></div>
          <div class="dz-formats">${FORMAT_CHIPS.map(f => `<span class="chip">${f}</span>`).join("")}</div></div>`;
  },
  renderHead() {
    const head = $("#jobHead");
    if (!head) return;
    const n = S.jobs.size, o = overall();
    head.innerHTML = n ? `<div class="section-label"><span>任务队列 · ${n} 本</span><span>${o.doneJobs ? `<button class="link-btn" data-act="clear-done">清除已完成（${o.doneJobs}）</button>` : ""}</span></div>` : "";
  },
  renderJobs() {
    const list = $("#jobList");
    if (!list) return;
    this.ensureSel();
    this.renderHead();
    list.innerHTML = Array.from(S.jobs.values()).map(jobCardHtml).join("");
    $("#jobTips").innerHTML = S.jobs.size ? "" : `<div class="tips3">${TIPS3.map(([i, t, s], k) => `<div class="tip3"><div class="tip3-ico">${ico(i)}</div><div><b>${k + 1}. ${t}</b><span>${s}</span></div></div>`).join("")}</div>`;
  },
  onJobs() { if (current === "tasks") this.renderAll(); else this.renderJobs(); },
  onJob(j, isNew) {
    const list = $("#jobList");
    if (!list) return;
    if (isNew) { this.renderDropzone(); this.renderJobs(); if (!this.pv.job) this.renderPreview(); return; }
    const card = list.querySelector(`.job[data-id="${CSS.escape(j.id)}"]`);
    if (!card) { this.renderJobs(); return; }
    patchJobCard(card, j);
    this.renderHead();
    if (j.id === this.pv.job) this.renderFollow();
  },
  onRemoved(id) {
    const card = $(`#jobList .job[data-id="${CSS.escape(id)}"]`);
    const after = () => { if (card) card.remove(); this.renderHead(); if (!S.jobs.size) { this.renderDropzone(); this.renderJobs(); } };
    if (card) { card.classList.add("leaving"); setTimeout(after, 190); } else after();
    if (this.pv.job === id) { this.pv.job = null; this.renderPreview(); }
  },
  onSettings() { this.renderBanner(); },
  onLlm() { this.renderBanner(); },
  onResync() { if (this.pv.job) loadPages(this.pv.job, true).then(() => { this.renderThumbs(); this.choosePreview(); }); },

  // ---- 首次使用引导：推荐下载本地翻译模型 ----
  renderBanner() {
    const box = $("#llmBanner");
    if (!box) return;
    const st = S.llm, local = getPath(S.settings, "translate.local") || {};
    const dl = st && st.download;
    if (dl && dl.state === "downloading") {
      box.innerHTML = `<div class="banner info llm-banner" data-dl="${esc(dl.model_id)}">${ico("download")}<div class="banner-text">
        <div class="llm-banner-title"><b>正在下载本地翻译模型 ${esc(dl.name || dl.model_id)}</b><span class="dl-text">${esc(dlText(dl))}</span></div>
        <div class="progress dl-bar"><i class="dl-fill" style="width:${dlPct(dl).toFixed(1)}%"></i></div>
        <div class="llm-banner-sub">支持断点续传；下载完成前，翻译会先用 API 服务或免费在线引擎。</div></div>
        <div class="llm-banner-acts"><button class="btn btn-sm" data-act="llm-cancel">${ico("x")}取消</button></div></div>`;
      return;
    }
    const rec = st && st.recommend && st.recommend.model;
    const need = st && local.enabled !== false && !(st.installed || []).length && rec;
    if (!need || laterActive()) { box.innerHTML = ""; return; }
    box.innerHTML = `<div class="banner info llm-banner">${ico("sparkle")}<div class="banner-text">
      <b>推荐下载本地翻译模型 ${esc(rec.name)}（约 ${fmtGB(rec.size_gb)}），获得最佳翻译效果</b>
      <div class="llm-banner-sub">${esc(st.recommend.reason || "")}离线运行、不限次数，整页上下文翻译更通顺。</div></div>
      <div class="llm-banner-acts"><button class="btn btn-primary btn-sm" data-act="llm-dl" data-id="${esc(rec.id)}">${ico("download")}立即下载</button>
      <button class="btn btn-ghost btn-sm" data-act="llm-later">稍后</button></div></div>`;
  },

  // ---- 实时预览（前后对比滑块）+ 页面缩略图 ----
  renderPreview() {
    const wrap = $("#pvWrap");
    if (!wrap) return;
    const j = S.jobs.get(this.ensureSel());
    if (!j) { this.pv.job = null; wrap.innerHTML = `<div class="card">${emptyHtml("preview", "实时预览", "开始汉化后，这里会实时显示每一页的前后对比")}</div>`; return; }
    if (this.pv.job !== j.id) Object.assign(this.pv, { job: j.id, index: -1, pinned: false });
    wrap.innerHTML = `<div class="card"><div class="card-head"><div style="min-width:0"><div class="card-title">${ico("compare")}实时预览</div><div class="pv-title ellipsis" id="pvTitle">${esc(j.title)}</div></div>
        <div class="head-actions" style="flex-wrap:nowrap"><button class="btn-icon sm" data-act="pv-step" data-d="-1" data-tip="上一张已完成的页">${ico("chev-l")}</button>
        <button class="btn-icon sm" data-act="pv-step" data-d="1" data-tip="下一张已完成的页">${ico("chev-r")}</button>
        <button class="btn btn-sm" data-act="pv-edit">${ico("edit")}校对</button></div></div>
        <div id="pvBox"><div class="compare-empty"><span class="spinner"></span></div></div>
        <div class="pv-foot"><span id="pvInfo">拖动滑块对比 · 双击进入校对</span><span id="pvFollow"></span></div></div>
      <div class="card"><div class="card-head"><div class="card-title">${ico("image")}页面</div><span class="muted" id="thCount"></span></div><div class="thumbs" id="thumbs">${skelThumbs(8)}</div></div>`;
    const jid = j.id;
    loadPages(jid).then(() => { if (this.pv.job === jid && $("#thumbs")) { this.renderThumbs(); this.choosePreview(); } });
  },
  renderThumbs() {
    const box = $("#thumbs");
    if (!box) return;
    const jid = this.pv.job, pages = S.pages.get(jid) || [];
    this.renderCount();
    if (!pages.length) { box.innerHTML = `<div class="muted" style="grid-column:1/-1;padding:4px 2px">还没有页面</div>`; return; }
    const MAX = 400;
    box.innerHTML = pages.slice(0, MAX).map(p => thumbHtml(jid, p, p.index === this.pv.index)).join("")
      + (pages.length > MAX ? `<div class="muted" style="grid-column:1/-1">还有 ${pages.length - MAX} 页，可在校对页查看</div>` : "");
  },
  renderCount() {
    const c = $("#thCount"), pages = S.pages.get(this.pv.job) || [];
    if (c) c.textContent = pages.length ? `${pages.filter(p => p.stage === "rendered").length} / ${pages.length} 已完成` : "";
  },
  choosePreview() {
    const pages = S.pages.get(this.pv.job) || [];
    const ok = i => pages[i] && pages[i].stage === "rendered";
    if (!(this.pv.pinned && ok(this.pv.index))) {
      let last = -1;
      for (const p of pages) if (p.stage === "rendered") last = p.index;
      this.pv.index = last;
      this.pv.pinned = false;
    }
    this.renderCompare();
  },
  setPreview(index, pinned) {
    this.pv.index = index;
    this.pv.pinned = !!pinned;
    $$("#thumbs .thumb.cur").forEach(t => t.classList.remove("cur"));
    const t = $(`#thumbs .thumb[data-i="${index}"]`);
    if (t) t.classList.add("cur");
    this.renderCompare();
  },
  renderFollow() {
    const f = $("#pvFollow"), j = S.jobs.get(this.pv.job);
    if (f) f.innerHTML = this.pv.pinned && j && j.status === "running" ? `<button class="link-btn" data-act="pv-follow">跟随最新进度</button>` : "";
  },
  renderCompare() {
    const box = $("#pvBox");
    if (!box) return;
    const j = S.jobs.get(this.pv.job), pages = S.pages.get(this.pv.job) || [];
    const p = pages[this.pv.index];
    $("#pvTitle").textContent = j ? j.title + (p ? ` · 第 ${p.index + 1} / ${pages.length} 页` : "") : "";
    this.renderFollow();
    if (!j || !p || p.stage !== "rendered") {
      const wait = j && (j.status === "running" || j.status === "queued");
      const sel = pages[this.pv.index];
      const text = sel && sel.stage !== "rendered" ? `第 ${sel.index + 1} 页还没完成（${STAGE_LABEL[sel.stage] || sel.stage}）`
        : wait ? "第一页完成后会显示在这里" : "这本书还没有完成的页面";
      box.innerHTML = `<div class="compare-empty"><div>${wait ? `<span class="spinner"></span><div style="margin-top:10px">${esc(text)}</div>` : esc(text)}</div></div>`;
      return;
    }
    const v = pageVer(p), key = `${j.id}:${p.index}:${v}`;
    const cur = box.querySelector(".compare");
    if (cur && cur.dataset.key === key) return;
    const outUrl = imgUrl(j.id, p.index, "out", v, 1100), srcUrl = imgUrl(j.id, p.index, "src", "s", 1100);
    const html = `<div class="compare" data-key="${esc(key)}" style="--pos:${this.pv.pos}%"><img class="cmp-bot" alt="汉化后" src="${outUrl}"><img class="cmp-top" alt="原图" src="${srcUrl}">
      <div class="cmp-line"></div><div class="cmp-knob">${ico("compare")}</div><span class="cmp-label l">原图</span><span class="cmp-label r">汉化后</span></div>`;
    const swap = () => { if (this.pv.job === j.id && box.isConnected && this.pv.index === p.index) box.innerHTML = html; };
    if (!cur || SHOT) { swap(); return; }
    let n = 0;                                   // 先在后台加载好两张图再切换，跟随进度时不闪白
    const ready = () => { if (++n === 2) swap(); };
    for (const u of [outUrl, srcUrl]) { const im = new Image(); im.onload = im.onerror = ready; im.src = u; }
  },
  cmpDown(e) {
    const c = e.target.closest(".compare");
    if (!c || e.button !== 0) return;
    e.preventDefault();
    c.setPointerCapture(e.pointerId);
    c.classList.add("dragging");
    const move = ev => {
      const r = c.getBoundingClientRect();
      this.pv.pos = clamp(((ev.clientX - r.left) / r.width) * 100, 0, 100);
      c.style.setProperty("--pos", this.pv.pos.toFixed(2) + "%");
    };
    const up = () => {
      c.classList.remove("dragging");
      c.removeEventListener("pointermove", move); c.removeEventListener("pointerup", up); c.removeEventListener("pointercancel", up);
    };
    move(e);
    c.addEventListener("pointermove", move); c.addEventListener("pointerup", up); c.addEventListener("pointercancel", up);
  },
  step(d) {
    const pages = S.pages.get(this.pv.job) || [];
    const done = pages.filter(p => p.stage === "rendered").map(p => p.index);
    if (!done.length) return;
    let i = done.indexOf(this.pv.index);
    i = i < 0 ? (d > 0 ? 0 : done.length - 1) : clamp(i + d, 0, done.length - 1);
    this.setPreview(done[i], true);
  },
  onPage(e, prevStage) {
    if (e.job_id !== this.pv.job) return;
    const pages = S.pages.get(e.job_id), p = pages && pages[e.index];
    if (!p) return;
    const t = $(`#thumbs .thumb[data-i="${e.index}"]`);
    if (t) {
      const n = el(thumbHtml(e.job_id, p, e.index === this.pv.index));
      if (p.stage === "rendered" && prevStage !== "rendered" && !SHOT) n.classList.add("flash");
      t.replaceWith(n);
    }
    this.renderCount();
    if (p.stage === "rendered" && (!this.pv.pinned || e.index === this.pv.index)) {
      if (!this.pv.pinned) this.pv.index = e.index;
      const cur = $("#thumbs .thumb.cur");
      if (cur && +cur.dataset.i !== this.pv.index) cur.classList.remove("cur");
      const nt = $(`#thumbs .thumb[data-i="${this.pv.index}"]`);
      if (nt) nt.classList.add("cur");
      this.renderCompare();
    }
  },
  onKey(e) {
    if (isTyping(e.target) || e.ctrlKey || e.altKey || e.metaKey) return;
    if (e.key === "ArrowLeft") { e.preventDefault(); this.step(-1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); this.step(1); }
  },
};
ACT["pv-step"] = btn => VIEWS.tasks.step(+btn.dataset.d);
ACT["pv-edit"] = () => { const pv = VIEWS.tasks.pv; if (pv.job) openEditor(pv.job, pv.index >= 0 ? pv.index : undefined); };
ACT["pv-follow"] = () => { VIEWS.tasks.pv.pinned = false; VIEWS.tasks.choosePreview(); };

// ---- 任务卡片上的操作 ----
const jobOf = btn => S.jobs.get(btn.dataset.id);
ACT["job-pause"] = btn => busy(btn, () => call("pause", { job_id: btn.dataset.id }));
ACT["job-resume"] = btn => busy(btn, async () => { if (await call("resume", { job_id: btn.dataset.id })) await call("start", {}, { quiet: true }); });
ACT["job-retry"] = btn => busy(btn, async () => { if (await call("retry", { job_id: btn.dataset.id })) toast("正在重新处理出错的页面", "info"); });
ACT["job-open"] = btn => { const j = jobOf(btn); if (j && j.out_path) return call("open_path", { path: j.out_path }); };
ACT["job-reveal"] = btn => { const j = jobOf(btn); if (j) return call("reveal", { path: j.out_path || j.source }); };
ACT["job-edit"] = btn => openEditor(btn.dataset.id);
ACT["job-export"] = btn => busy(btn, async () => { if (await call("export", { job_id: btn.dataset.id })) toast("正在导出…完成后会提示", "info", { duration: 2200 }); });
ACT["job-rerender"] = async btn => {
  const j = jobOf(btn);
  if (!j || !(await modal({ title: "按当前设置重新嵌字？", icon: "type", ok: "重新嵌字",
    text: `将按现在的排版设置（字体、字号、行距、描边等）重新排版《${j.title}》的全部页面；手动改过的译文和样式会保留。` }))) return;
  await busy(btn, () => call("rerender_all", { job_id: j.id }));
};
ACT["job-cancel"] = async btn => {
  const j = jobOf(btn);
  if (!j || !(await confirmDanger("取消这个任务？", `《${j.title}》会停止处理；已完成的页面会保留，之后可以点「继续」接着做。`, "取消任务"))) return;
  await busy(btn, () => call("cancel", { job_id: j.id }));
};
ACT["job-remove"] = async btn => {
  const j = jobOf(btn);
  if (!j) return;
  const r = await modal({
    title: "删除这个任务？", danger: true, ok: "删除",
    text: `删除《${j.title}》的任务和处理进度。原始漫画和已导出的成品会保留，再次拖入会创建新任务。`,
  });
  if (!r) return;
  const card = btn.closest(".job");
  if (card) card.classList.add("leaving");
  if (await call("remove", { job_id: j.id, delete_files: true })) { removeJob(j.id); toast(`已删除《${j.title}》`, "success"); }
  else if (card) card.classList.remove("leaving");
};
ACT["clear-done"] = async () => {
  const done = Array.from(S.jobs.values()).filter(j => j.status === "done");
  if (!done.length || !(await modal({ title: `清除 ${done.length} 个已完成的任务？`, ok: "清除", icon: "check-circle",
    text: "从列表中移除，保留成品和中间结果；再次拖入会创建新的独立任务。" }))) return;
  for (const j of done) if (await call("remove", { job_id: j.id, delete_files: false })) removeJob(j.id);
  toast("已清除已完成的任务", "success");
};

// ===================================================================== 校对页
const LAYERS = [["src", "原图", "1"], ["clean", "去字", "2"], ["out", "成品", "3"]];
const blkKind = b => (b.is_sfx ? "拟声词" : b.kind === "free" ? "气泡外" : "气泡");
const fontName = id => { const f = (S.fonts || []).find(x => x.id === id); return f ? f.name : (id || "默认"); };
function autoGrow(t) { t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight + 2, 240) + "px"; }
function defaultFontFor(b) {
  const ts = getPath(S.settings, "typeset") || {};
  return b.is_sfx ? ts.font_sfx : b.kind === "free" ? ts.font_free : ts.font_dialog;
}
const styled = b => b.font || b.font_size || b.out_direction || b.align || b.fill || b.stroke || b.stroke_width != null;

VIEWS.editor = {
  title: "校对",
  st: { job: null, index: -1, page: null, blocks: [], orig: "[]", sel: -1, layer: "out", z: 1, tx: 0, ty: 0,
        fitted: true, boxes: true, seq: 0, has: {} },
  openStyle: new Set(),
  mount(root) {
    root.innerHTML = `<div class="editor" id="edMain">
      <div class="ed-strip" id="edStrip"></div>
      <div class="ed-center">
        <div class="ed-toolbar">
          <select class="select sm ed-job" id="edJob" aria-label="选择要校对的书"></select>
          <div class="pager"><button class="btn-icon sm" data-act="ed-step" data-d="-1" data-tip="上一页" data-kbd="←">${ico("chev-l")}</button>
            <span class="pg" id="edPg">– / –</span><button class="btn-icon sm" data-act="ed-step" data-d="1" data-tip="下一页" data-kbd="→">${ico("chev-r")}</button></div>
          <div class="seg seg-sm" id="edLayer" role="radiogroup" aria-label="显示图层">${LAYERS.map(([v, l, k]) => `<button data-v="${v}" data-tip="显示${l}" data-kbd="${k}">${l}</button>`).join("")}</div>
          <button class="btn-icon sm on" id="edBoxBtn" data-act="ed-boxes" data-tip="显示 / 隐藏文字框" data-kbd="B">${ico("scan")}</button>
          <span class="grow"></span>
        </div>
        <div class="ed-canvas" id="edCanvas">
          <div class="zoom-ctl"><button class="btn-icon sm" data-act="ed-zoom" data-d="0.8" data-tip="缩小" data-kbd="-">${ico("zoom-out")}</button>
            <span class="zv" id="edZv">100%</span><button class="btn-icon sm" data-act="ed-zoom" data-d="1.25" data-tip="放大" data-kbd="+">${ico("zoom-in")}</button>
            <button class="btn-icon sm" data-act="ed-fit" data-tip="适应窗口" data-kbd="0">${ico("fit")}</button></div>
          <div class="ed-stage" id="edStage" data-layer="out"><img class="base" alt="" draggable="false"><img class="ov ov-clean" alt="" draggable="false"><img class="ov ov-out" alt="" draggable="false"><div class="ed-boxes" id="edBoxes"></div></div>
          <div class="ed-peek" id="edPeek" hidden>正在看原图</div>
          <div class="ed-hint"><span><span class="kbd">空格</span>按住看原图</span><span><span class="kbd">←</span><span class="kbd">→</span>翻页</span><span><span class="kbd">Ctrl</span>+滚轮 缩放</span><span><span class="kbd">Ctrl+S</span>应用</span></div>
          <div class="ed-overlay" id="edOverlay" hidden></div>
        </div>
      </div>
      <div class="ed-panel">
        <div class="ed-panel-head"><h3>${ico("type")}文字块 <span class="muted" id="edCount"></span></h3><span class="dirty-pill" id="edDirty" hidden>未保存</span></div>
        <div class="ed-notice" id="edNotice" hidden></div>
        <div class="ed-blocks" id="edBlocks"></div>
        <div class="ed-actions">
          <button class="btn btn-primary btn-block" id="edSave" data-act="ed-save" data-tip="保存修改并重新排版本页" data-kbd="Ctrl+S">${ico("save")}应用并重排</button>
          <div class="row"><button class="btn" data-act="ed-retranslate">${ico("translate")}重新翻译本页</button><button class="btn" data-act="ed-redetect">${ico("scan")}重新检测本页</button></div>
          <div class="row"><button class="btn btn-ghost btn-sm" id="edRevert" data-act="ed-revert" disabled>${ico("undo")}撤销未保存的修改</button></div>
        </div>
      </div>
    </div><div class="ed-empty" id="edEmpty" hidden></div>`;
    this.canvas = $("#edCanvas", root);
    this.stage = $("#edStage", root);
    this.reloadSoon = debounce(() => this.reload(), 300);
    this.canvas.addEventListener("wheel", e => this.onWheel(e), { passive: false });
    this.canvas.addEventListener("pointerdown", e => this.onDown(e));
    this.canvas.addEventListener("dblclick", e => { if (!e.target.closest(".bx, .ed-hint, .zoom-ctl, .ed-overlay")) this.toggleZoom(e); });
    $("#edStrip", root).addEventListener("click", e => { const it = e.target.closest(".strip-item"); if (it) this.open(this.st.job, +it.dataset.i); });
    $("#edJob", root).addEventListener("change", e => this.open(e.target.value, null));
    $("#edLayer", root).addEventListener("click", e => { const b = e.target.closest("button[data-v]"); if (b) this.setLayer(b.dataset.v); });
    const blocks = $("#edBlocks", root);
    blocks.addEventListener("input", e => this.onField(e));
    blocks.addEventListener("change", e => this.onField(e));
    blocks.addEventListener("click", e => this.onBlockClick(e));
    blocks.addEventListener("focusin", e => { const c = e.target.closest(".blk"); if (c && isTyping(e.target)) this.select(+c.dataset.i, "canvas"); });
    blocks.addEventListener("toggle", e => {
      const c = e.target.closest && e.target.closest(".blk");
      if (c && e.target.tagName === "DETAILS") { const i = +c.dataset.i; if (e.target.open) this.openStyle.add(i); else this.openStyle.delete(i); }
    }, true);
    const [, clean, out] = $$("img", this.stage);
    for (const img of [clean, out]) img.addEventListener("error", () => { if (img.getAttribute("src")) this.onLayerMissing(img === out ? "out" : "clean"); });
    $("img.base", this.stage).addEventListener("load", e => {
      const p = this.st.page;
      if (p && !(p.width && p.height)) { p.width = e.target.naturalWidth; p.height = e.target.naturalHeight; this.sizeStage(); if (this.st.fitted) this.fit(); }
    });
    window.addEventListener("blur", () => this.peek(false));
    if (window.ResizeObserver) new ResizeObserver(() => { if (this.st.fitted && this.st.page && current === "editor") this.fit(); }).observe(this.canvas);
  },
  jobList() { return Array.from(S.jobs.values()).filter(j => (j.total || 0) > 0); },
  defaultJob() {
    const jobs = this.jobList();
    const sel = S.jobs.get(S.sel);
    if (sel && (sel.done || 0) > 0) return sel.id;
    const j = jobs.find(x => (x.done || 0) > 0) || jobs.find(x => x.status === "running") || jobs[0];
    return j ? j.id : null;
  },
  defaultIndex(jid) {
    const pages = S.pages.get(jid) || [];
    const p = pages.find(x => x.stage === "rendered") || pages.find(x => atLeast(x.stage, "translated"));
    return p ? p.index : 0;
  },
  async show(args) {
    if (!this.jobList().length) { this.showEmpty(true); return; }
    this.showEmpty(false);
    let [jid, idx] = args || [];
    if (!jid || !S.jobs.has(jid)) jid = this.st.job && S.jobs.has(this.st.job) ? this.st.job : this.defaultJob();
    let index = idx != null && idx !== "" && !isNaN(+idx) ? +idx : (jid === this.st.job && this.st.index >= 0 ? this.st.index : null);
    this.renderJobSelect(jid);
    await this.open(jid, index, true);
    if (this.st.fitted) requestAnimationFrame(() => this.fit());
  },
  hide() { this.peek(false); },
  leave() {
    if (!this.isDirty()) return Promise.resolve(true);
    return this.askSave().then(async r => {
      if (r === "cancel") return false;
      if (r === "save") return this.save();
      this.revert();
      return true;
    });
  },
  askSave() {
    return modal({
      title: "本页有未保存的修改", text: "要先应用这些修改（会重新排版本页）吗？", icon: "save",
      buttons: [{ label: "取消", value: "cancel" }, { label: "放弃修改", value: "discard", kind: "ghost" }, { label: "应用并继续", value: "save", kind: "primary" }],
    }).then(v => v || "cancel");
  },
  showEmpty(on) {
    $("#edMain").hidden = on;
    const e = $("#edEmpty");
    e.hidden = !on;
    if (on) e.innerHTML = emptyHtml("editor", "还没有可以校对的漫画", "在「任务」页添加漫画；翻译完成的页面会出现在这里，可以逐页修改译文和样式",
      `<button class="btn btn-primary" data-act="go" data-to="tasks">${ico("plus")}去添加漫画</button>`);
  },
  renderJobSelect(jid) {
    const sel = $("#edJob");
    if (!sel) return;
    const cur = jid || this.st.job;
    sel.innerHTML = this.jobList().map(j => `<option value="${esc(j.id)}"${j.id === cur ? " selected" : ""}>${esc(j.title)}（${j.done || 0}/${j.total || 0}）</option>`).join("");
  },
  async open(jid, index, force) {
    const st = this.st;
    if (!jid) return false;
    if (!force && jid === st.job && index === st.index) return true;
    if (st.page && this.isDirty() && (jid !== st.job || index !== st.index)) {
      const r = await this.askSave();
      if (r === "cancel") { this.renderJobSelect(); return false; }
      if (r === "save" && !(await this.save())) { this.renderJobSelect(); return false; }
    }
    const jobChanged = jid !== st.job;
    const pages = await loadPages(jid, jobChanged);
    if (index == null) index = this.defaultIndex(jid);
    index = clamp(index | 0, 0, Math.max(0, pages.length - 1));
    st.job = jid; st.index = index;
    if (jobChanged) { this.openStyle.clear(); this.renderJobSelect(); this.renderStrip(); } else this.markStrip();
    if (current === "editor") setRouteArgs([jid, index]);
    this.updatePager();
    if (!pages.length) { this.applyPage(null); return false; }
    const seq = ++st.seq;
    const t = setTimeout(() => this.overlay(`<span class="spinner lg"></span>`), 160);
    const page = await call("get_page", { job_id: jid, index }, { quiet: true });
    clearTimeout(t);
    if (seq !== st.seq) return false;
    if (!page) {
      this.overlay(emptyHtml("error", "这一页读取失败", "可能是任务已被删除，或中间结果被清理了", `<button class="btn" data-act="ed-reload">${ico("refresh")}重试</button>`));
      return false;
    }
    this.overlay("");
    if (!jobChanged && index !== this.lastIndex) this.openStyle.clear();
    this.lastIndex = index;
    this.applyPage(page, false);
    return true;
  },
  overlay(html) { const o = $("#edOverlay"); o.innerHTML = html; o.hidden = !html; },
  sizeStage() {
    const [W, H] = this.dims();
    this.stage.style.width = W + "px";
    this.stage.style.height = H + "px";
  },
  applyPage(page, keep) {
    const st = this.st;
    st.page = page;
    st.blocks = page ? (page.blocks || []).map(b => Object.assign({}, b)) : [];
    st.orig = JSON.stringify(st.blocks);
    st.sel = keep ? Math.min(st.sel, st.blocks.length - 1) : -1;
    const [base, clean, out] = $$("img", this.stage);
    if (!page) {
      for (const im of [base, clean, out]) im.removeAttribute("src");
      $("#edBoxes").innerHTML = "";
      this.renderBlocks(); this.renderNotice(); this.updateDirty();
      return;
    }
    const pages = S.pages.get(st.job);
    if (pages && pages[page.index]) { pages[page.index].stage = page.stage; pages[page.index].version = page.version; this.patchStrip(page.index); }
    const v = pageVer(page);
    st.has = { clean: atLeast(page.stage, "cleaned"), out: page.stage === "rendered" };
    this.sizeStage();
    const srcU = imgUrl(st.job, page.index, "src", "s");
    if (base.getAttribute("src") !== srcU) base.src = srcU;
    const setOv = (im, ok, kind) => { const u = ok ? imgUrl(st.job, page.index, kind, v) : ""; if (!u) im.removeAttribute("src"); else if (im.getAttribute("src") !== u) im.src = u; };
    setOv(clean, st.has.clean, "clean");
    setOv(out, st.has.out, "out");
    this.setLayer(st.layer, true);
    if (!keep) { if (st.fitted) this.fit(); else this.applyTransform(); }
    this.renderBoxes(); this.renderBlocks(); this.renderNotice(); this.updatePager(); this.updateDirty();
  },
  onLayerMissing(kind) {
    this.st.has[kind] = false;
    this.setLayer(this.st.layer, true);
  },
  effLayer() {
    const st = this.st, L = st.layer;
    if (L === "out" && !st.has.out) return st.has.clean ? "clean" : "src";
    if (L === "clean" && !st.has.clean) return "src";
    return L;
  },
  setLayer(v, silent) {
    const st = this.st;
    if (!silent) {
      if (v !== "src" && !st.has[v] && st.page) toast(v === "out" ? "这一页还没完成嵌字，暂时没有成品图" : "这一页还没完成去字", "info", { duration: 2200 });
      st.layer = v;
    }
    const eff = this.effLayer();
    this.stage.dataset.layer = eff;
    $$("#edLayer button").forEach(b => {
      b.classList.toggle("on", b.dataset.v === eff);
      b.classList.toggle("dim", b.dataset.v !== "src" && !st.has[b.dataset.v]);
    });
  },
  toggleBoxes() {
    this.st.boxes = !this.st.boxes;
    this.canvas.classList.toggle("hide-boxes", !this.st.boxes);
    $("#edBoxBtn").classList.toggle("on", this.st.boxes);
  },
  peek(on) {
    if (!this.canvas) return;
    this.canvas.classList.toggle("peek", !!on);
    $("#edPeek").hidden = !on;
  },

  // ---- 画布：缩放 / 平移 ----
  dims() {
    const p = this.st.page || {}, b = $("img.base", this.stage);
    return [p.width || (b && b.naturalWidth) || 1000, p.height || (b && b.naturalHeight) || 1400];
  },
  applyTransform() {
    const st = this.st;
    this.stage.style.transform = `translate(${st.tx.toFixed(1)}px, ${st.ty.toFixed(1)}px) scale(${st.z})`;
    this.stage.style.setProperty("--z", st.z);
    $("#edZv").textContent = Math.round(st.z * 100) + "%";
  },
  fit() {
    const st = this.st, [W, H] = this.dims(), r = this.canvas.getBoundingClientRect();
    if (!r.width || !r.height) return;
    // 上方给浮动缩放控件留 54px，下方给快捷键提示条留 50px
    const pad = 26, top = 54, hint = 50, cw = Math.max(60, r.width - pad * 2), ch = Math.max(60, r.height - top - hint);
    const tall = H / W > 2.6;                     // 韩漫长条：按宽度适应，从顶部开始
    st.z = clamp(tall ? Math.min(cw / W, 1.5) : Math.min(cw / W, ch / H), 0.03, 4);
    st.tx = (r.width - W * st.z) / 2;
    st.ty = tall ? top : top + Math.max(0, (ch - H * st.z) / 2);
    st.fitted = true;
    this.applyTransform();
  },
  zoomAt(factor, cx, cy) {
    const st = this.st, nz = clamp(st.z * factor, 0.05, 8);
    if (cx == null) { const r = this.canvas.getBoundingClientRect(); cx = r.width / 2; cy = r.height / 2; }
    st.tx = cx - ((cx - st.tx) * nz) / st.z;
    st.ty = cy - ((cy - st.ty) * nz) / st.z;
    st.z = nz; st.fitted = false;
    this.applyTransform();
  },
  toggleZoom(e) {
    if (!this.st.fitted) { this.fit(); return; }
    const r = this.canvas.getBoundingClientRect();
    this.zoomAt(1 / this.st.z, e.clientX - r.left, e.clientY - r.top);
  },
  onWheel(e) {
    if (!this.st.page) return;
    e.preventDefault();
    const r = this.canvas.getBoundingClientRect();
    if (e.ctrlKey || e.metaKey) { this.zoomAt(Math.exp(-e.deltaY * 0.0018), e.clientX - r.left, e.clientY - r.top); return; }
    const k = e.deltaMode === 1 ? 32 : 1;
    this.st.tx -= (e.shiftKey ? e.deltaY : e.deltaX) * k;
    this.st.ty -= (e.shiftKey ? 0 : e.deltaY) * k;
    this.st.fitted = false;
    this.applyTransform();
  },
  onDown(e) {
    if ((e.button !== 0 && e.button !== 1) || e.target.closest(".ed-hint, .ed-overlay, .zoom-ctl")) return;
    const bx = e.target.closest(".bx");
    if (bx && e.button === 0) { this.select(+bx.dataset.i, "list"); return; }
    e.preventDefault();
    const st = this.st, sx = e.clientX, sy = e.clientY, tx0 = st.tx, ty0 = st.ty;
    let moved = false;
    this.canvas.setPointerCapture(e.pointerId);
    const move = ev => {
      const dx = ev.clientX - sx, dy = ev.clientY - sy;
      if (!moved && Math.abs(dx) + Math.abs(dy) < 4) return;
      if (!moved) { moved = true; this.canvas.classList.add("panning"); }
      st.tx = tx0 + dx; st.ty = ty0 + dy; st.fitted = false;
      this.applyTransform();
    };
    const up = () => {
      this.canvas.classList.remove("panning");
      this.canvas.removeEventListener("pointermove", move); this.canvas.removeEventListener("pointerup", up); this.canvas.removeEventListener("pointercancel", up);
      if (!moved && e.button === 0) this.select(-1);
    };
    this.canvas.addEventListener("pointermove", move); this.canvas.addEventListener("pointerup", up); this.canvas.addEventListener("pointercancel", up);
  },

  // ---- 文字块 ----
  select(i, scroll) {
    const st = this.st;
    st.sel = i;
    $$("#edBoxes .bx").forEach(b => b.classList.toggle("sel", +b.dataset.i === i));
    $$("#edBlocks .blk").forEach(b => b.classList.toggle("sel", +b.dataset.i === i));
    if (i < 0) return;
    if (scroll === "list") { const c = $(`#edBlocks .blk[data-i="${i}"]`); if (c) c.scrollIntoView({ block: "nearest", behavior: SHOT ? "auto" : "smooth" }); }
    if (scroll === "canvas") this.revealBox(i);
  },
  selectStep(d) {
    const n = this.st.blocks.length;
    if (!n) return;
    this.select(this.st.sel < 0 ? (d > 0 ? 0 : n - 1) : clamp(this.st.sel + d, 0, n - 1), "list");
    this.revealBox(this.st.sel);
  },
  revealBox(i) {
    const st = this.st, b = st.blocks[i];
    if (!b || !b.box) return;
    const r = this.canvas.getBoundingClientRect(), m = 36, hint = 44;
    const [x1, y1, x2, y2] = b.box;
    const L = st.tx + x1 * st.z, T = st.ty + y1 * st.z, R = st.tx + x2 * st.z, B = st.ty + y2 * st.z;
    let dx = 0, dy = 0;
    if (L < m) dx = m - L; else if (R > r.width - m) dx = r.width - m - R;
    if (T < m) dy = m - T; else if (B > r.height - hint) dy = r.height - hint - B;
    if (dx || dy) { st.tx += dx; st.ty += dy; st.fitted = false; this.applyTransform(); }
  },
  renderBoxes() {
    const st = this.st;
    $("#edBoxes").innerHTML = st.blocks.map((b, i) => {
      const [x1, y1, x2, y2] = b.box || [0, 0, 0, 0];
      return `<div class="bx${i === st.sel ? " sel" : ""}${b.skip ? " skip" : ""}${b.overflow ? " over" : ""}" data-i="${i}" style="left:${x1}px;top:${y1}px;width:${Math.max(1, x2 - x1)}px;height:${Math.max(1, y2 - y1)}px"><span class="bx-no">${i + 1}</span></div>`;
    }).join("");
  },
  canEdit() { const p = this.st.page; return !!p && atLeast(p.stage, "translated"); },
  renderBlocks() {
    const st = this.st, box = $("#edBlocks");
    $("#edCount").textContent = st.blocks.length ? `${st.blocks.length} 个` : "";
    if (!st.page) { box.innerHTML = emptyHtml("editor", "这本书还没有页面", ""); return; }
    if (!st.blocks.length) {
      const busyStage = !atLeast(st.page.stage, "detected") && st.page.stage !== "error";
      box.innerHTML = `<div class="empty" style="padding:30px 12px">${busyStage ? `<span class="spinner lg"></span><div class="empty-title">正在检测文字…</div>`
        : `<div class="empty-title">这一页没有检测到文字</div><div class="empty-sub">如果其实有文字没被识别，可以点下面的「重新检测本页」</div>`}</div>`;
      return;
    }
    const editable = this.canEdit();
    box.innerHTML = st.blocks.map((b, i) => this.blockHtml(b, i, editable)).join("");
    $$("textarea", box).forEach(autoGrow);
  },
  blockHtml(b, i, editable) {
    const dis = editable ? "" : " disabled";
    const fontOpts = `<option value="">自动（${esc(fontName(defaultFontFor(b)))}）</option>`
      + (S.fonts || []).map(f => `<option value="${esc(f.id)}"${f.id === b.font ? " selected" : ""}>${esc(f.name)}</option>`).join("");
    const seg = (f, opts) => `<div class="seg seg-sm" data-f="${f}">${opts.map(([v, l]) => `<button type="button" data-v="${v}" class="${(b[f] || "") === v ? "on" : ""}"${dis}>${l}</button>`).join("")}</div>`;
    const conf = b.ocr_conf ? `<span class="conf" data-tip="原文识别的置信度">${Math.round(b.ocr_conf * 100)}%</span>` : "";
    const warn = b.overflow ? `<span class="blk-warn" data-tip="最小字号也放不下，已强制缩小；可以精简译文或调小字号">${ico("alert")}放不下</span>` : "";
    const fill = rgbToHex(b.fill), stroke = rgbToHex(b.stroke);
    return `<div class="blk${i === this.st.sel ? " sel" : ""}${b.skip ? " skip" : ""}" data-i="${i}">
      <div class="blk-head"><span class="blk-no">${i + 1}</span><span class="tag plain">${blkKind(b)}</span>${warn}<span class="grow"></span>${conf}
        <label class="check" data-tip="不擦除、不嵌字，保留原图上的文字"><input type="checkbox" data-f="skip"${b.skip ? " checked" : ""}${dis}>保留原文</label></div>
      <div class="blk-fields">
        <div><label class="fl">原文</label><textarea class="textarea src" rows="1" data-f="src_text" spellcheck="false"${dis}>${esc(b.src_text)}</textarea></div>
        <div><label class="fl">译文</label><textarea class="textarea" rows="1" data-f="translation" spellcheck="false" placeholder="留空 = 只擦除不嵌字"${dis}>${esc(b.translation)}</textarea></div>
        <details${this.openStyle.has(i) ? " open" : ""}><summary>${ico("chev-d")}样式${styled(b) ? `<span class="muted"> · 已自定义</span>` : ""}</summary>
          <div class="style-grid">
            <span>字体</span><div class="row"><select class="select sm" data-f="font" style="flex:1;min-width:0"${dis}>${fontOpts}</select></div>
            <span>字号</span><div class="row"><input class="input sm num" type="number" min="4" max="2000" step="1" data-f="font_size" value="${b.font_size || ""}" placeholder="${b.render_font_size || "自动"}"${dis}><span class="muted">px · 空 = 自动</span></div>
            <span>方向</span><div class="row">${seg("out_direction", [["", "自动"], ["v", "竖排"], ["h", "横排"]])}</div>
            <span>对齐</span><div class="row">${seg("align", [["", "自动"], ["center", "居中"], ["start", "靠前"], ["end", "靠后"]])}</div>
            <span>颜色</span><div class="row"><input type="color" class="color-pick${fill ? "" : " auto"}" data-f="fill" value="${fill || "#111111"}" data-tip="文字颜色"${dis}>
              <input type="color" class="color-pick${stroke ? "" : " auto"}" data-f="stroke" value="${stroke || "#ffffff"}" data-tip="描边颜色"${dis}>
              <input class="input sm num" type="number" min="0" max="1" step="0.01" data-f="stroke_width" value="${b.stroke_width != null ? b.stroke_width : ""}" placeholder="描边" data-tip="描边宽度（相对字号，例如 0.12）"${dis}>
              <button type="button" class="link-btn" data-f="color_auto"${dis}>自动</button></div>
          </div></details>
      </div></div>`;
  },
  onField(e) {
    const t = e.target, f = t.dataset.f, card = t.closest(".blk");
    if (!f || !card) return;
    if (e.type === "input" && (t.type === "checkbox" || t.tagName === "SELECT")) return;
    if (e.type === "change" && (t.tagName === "TEXTAREA" || t.type === "number" || t.type === "color")) return;
    const i = +card.dataset.i, b = this.st.blocks[i];
    if (!b) return;
    switch (f) {
      case "src_text": case "translation": b[f] = t.value; autoGrow(t); break;
      case "skip": {
        b.skip = t.checked;
        card.classList.toggle("skip", b.skip);
        const bx = $(`#edBoxes .bx[data-i="${i}"]`);
        if (bx) bx.classList.toggle("skip", b.skip);
        break;
      }
      case "font": b.font = t.value || null; break;
      case "font_size": b.font_size = t.value === "" ? null : clamp(Math.round(+t.value) || 4, 4, 2000); break;
      case "stroke_width": b.stroke_width = t.value === "" ? null : clamp(+t.value || 0, 0, 1); break;
      case "fill": case "stroke": b[f] = hexToRgb(t.value); t.classList.remove("auto"); break;
      default: return;
    }
    this.updateDirty();
  },
  onBlockClick(e) {
    const t = e.target, card = t.closest(".blk");
    if (!card) return;
    const i = +card.dataset.i, b = this.st.blocks[i];
    const sb = t.closest(".seg[data-f] button");
    if (sb) {
      if (sb.disabled) return;
      b[sb.parentElement.dataset.f] = sb.dataset.v || null;
      $$("button", sb.parentElement).forEach(x => x.classList.toggle("on", x === sb));
      this.updateDirty();
      return;
    }
    if (t.closest('[data-f="color_auto"]')) {
      b.fill = null; b.stroke = null; b.stroke_width = null;
      $$(".color-pick", card).forEach(c => c.classList.add("auto"));
      const sw = card.querySelector('[data-f="stroke_width"]');
      if (sw) sw.value = "";
      this.updateDirty();
      return;
    }
    if (!isTyping(t) && !t.closest("summary, label, button, input")) this.select(i, "canvas");
  },
  isDirty() { return !!this.st.page && JSON.stringify(this.st.blocks) !== this.st.orig; },
  updateDirty() {
    const d = this.isDirty();
    $("#edDirty").hidden = !d;
    $("#edRevert").disabled = !d;
  },
  revert() {
    this.st.blocks = JSON.parse(this.st.orig);
    this.renderBoxes(); this.renderBlocks(); this.updateDirty();
  },
  renderNotice() {
    const p = this.st.page, box = $("#edNotice");
    let html = "";
    if (p && p.stage === "error") html = `<div class="banner error">${ico("alert-circle")}<div class="banner-text">这一页处理出错：${esc(p.error || "未知原因")}。可以点「重新检测本页」再试一次。</div></div>`;
    else if (p && !atLeast(p.stage, "translated")) html = `<div class="banner warn">${ico("alert")}<div class="banner-text">这一页还在处理中（${STAGE_LABEL[p.stage] || p.stage}），翻译完成后就能校对，完成时会自动刷新。</div></div>`;
    else if (p && p.stage !== "rendered") html = `<div class="banner info">${ico("info")}<div class="banner-text">这一页还没嵌字完成；现在修改译文后点「应用并重排」会立即排版。</div></div>`;
    else if (p) {
      const n = this.st.blocks.filter(b => b.overflow).length;
      if (n) html = `<div class="banner warn">${ico("alert")}<div class="banner-text">有 ${n} 个文字块放不下，已自动缩小字号；可以精简译文或调小字号。</div></div>`;
    }
    box.innerHTML = html;
    box.hidden = !html;
    $("#edSave").disabled = !this.canEdit();
    const rt = $('.ed-actions [data-act="ed-retranslate"]'), rd = $('.ed-actions [data-act="ed-redetect"]');
    if (rt) rt.disabled = !p || !(atLeast(p.stage, "recognized") || p.stage === "error");
    if (rd) rd.disabled = !p;
  },
  updatePager() {
    const pages = S.pages.get(this.st.job) || [];
    $("#edPg").textContent = pages.length ? `第 ${this.st.index + 1} / ${pages.length} 页` : "– / –";
    const [prev, next] = $$('.pager [data-act="ed-step"]');
    if (prev) prev.disabled = this.st.index <= 0;
    if (next) next.disabled = this.st.index >= pages.length - 1;
  },
  stripHtml(p) {
    const url = p.stage === "rendered" ? imgUrl(this.st.job, p.index, "thumb", pageVer(p)) : imgUrl(this.st.job, p.index, "src", "s", 240);
    return `<div class="strip-item${p.index === this.st.index ? " cur" : ""}" data-i="${p.index}" title="第 ${p.index + 1} 页 · ${esc(STAGE_LABEL[p.stage] || p.stage)}"><img alt="" loading="lazy" src="${url}"><span class="no">${p.index + 1}</span><span class="sdot s-${p.stage}"></span></div>`;
  },
  renderStrip() {
    const pages = S.pages.get(this.st.job) || [];
    $("#edStrip").innerHTML = pages.slice(0, 3000).map(p => this.stripHtml(p)).join("");
    this.markStrip();
  },
  markStrip() {
    $$("#edStrip .strip-item.cur").forEach(x => x.classList.remove("cur"));
    const it = $(`#edStrip .strip-item[data-i="${this.st.index}"]`);
    if (it) { it.classList.add("cur"); it.scrollIntoView({ block: "nearest" }); }
  },
  patchStrip(i) {
    const p = (S.pages.get(this.st.job) || [])[i], it = $(`#edStrip .strip-item[data-i="${i}"]`);
    if (!p || !it) return;
    const n = el(this.stripHtml(p)), img = it.querySelector("img"), nimg = n.querySelector("img");
    if (img && nimg && img.getAttribute("src") === nimg.getAttribute("src")) {
      it.querySelector(".sdot").className = `sdot s-${p.stage}`;
      it.title = n.title;
    } else it.replaceWith(n);
  },
  step(d) {
    const pages = S.pages.get(this.st.job) || [];
    const i = clamp(this.st.index + d, 0, pages.length - 1);
    if (i !== this.st.index) this.open(this.st.job, i);
  },
  async reload() {
    const st = this.st;
    if (!st.job || st.index < 0 || this.isDirty()) return;
    const seq = ++st.seq;
    const page = await call("get_page", { job_id: st.job, index: st.index }, { quiet: true });
    if (page && seq === st.seq && !this.isDirty()) this.applyPage(page, true);
  },
  async save() {
    const st = this.st;
    if (!st.page) return false;
    if (!this.canEdit()) { toast("这一页还没翻译完，稍后再来校对", "warning"); return false; }
    let ok = false;
    await busy($("#edSave"), async () => {
      const page = await call("update_page", { job_id: st.job, index: st.index, blocks: st.blocks }, { timeout: 120000 });
      if (page) { this.applyPage(page, true); ok = true; toast("已应用修改并重新排版", "success", { duration: 2200 }); }
    });
    return ok;
  },
  onKey(e) {
    if (isTyping(e.target)) {
      if (e.key === "Escape") e.target.blur();
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey || !this.st.job) return;
    const k = e.key;
    if (k === "ArrowLeft" || k === "PageUp") { e.preventDefault(); this.step(-1); }
    else if (k === "ArrowRight" || k === "PageDown") { e.preventDefault(); this.step(1); }
    else if (k === "ArrowUp") { e.preventDefault(); this.selectStep(-1); }
    else if (k === "ArrowDown") { e.preventDefault(); this.selectStep(1); }
    else if (k === " ") { e.preventDefault(); if (!e.repeat) this.peek(true); }
    else if (k === "1" || k === "2" || k === "3") this.setLayer(LAYERS[+k - 1][0]);
    else if (k === "b" || k === "B") this.toggleBoxes();
    else if (k === "0") this.fit();
    else if (k === "+" || k === "=") this.zoomAt(1.25);
    else if (k === "-" || k === "_") this.zoomAt(0.8);
    else if (k === "Escape") this.select(-1);
    else if (k === "Enter" && this.st.sel >= 0) {
      const ta = $(`#edBlocks .blk[data-i="${this.st.sel}"] textarea[data-f="translation"]`);
      if (ta && !ta.disabled) { e.preventDefault(); ta.focus(); }
    }
  },
  onKeyUp(e) { if (e.key === " ") this.peek(false); },
  onPage(e) {
    if (e.job_id !== this.st.job) return;
    this.patchStrip(e.index);
    const p = this.st.page;
    if (e.index === this.st.index && p && (e.version !== p.version || e.stage !== p.stage)) this.reloadSoon();
  },
  onJob(j) { if (j.id === this.st.job || !$(`#edJob option[value="${CSS.escape(j.id)}"]`)) this.renderJobSelect(); },
  onJobs() { this.renderJobSelect(); },
  onRemoved(id) {
    if (id !== this.st.job) { this.renderJobSelect(); return; }
    Object.assign(this.st, { job: null, index: -1, page: null, blocks: [], orig: "[]", sel: -1 });
    if (current === "editor") this.show([]);
  },
  onResync() { if (this.st.job && current === "editor") loadPages(this.st.job, true).then(() => { this.renderStrip(); this.reload(); }); },
  onSettings(s, fromSelf) { if (!fromSelf && this.st.page && !this.isDirty()) this.renderBlocks(); },
};
ACT["ed-step"] = btn => VIEWS.editor.step(+btn.dataset.d);
ACT["ed-zoom"] = btn => VIEWS.editor.zoomAt(+btn.dataset.d);
ACT["ed-fit"] = () => VIEWS.editor.fit();
ACT["ed-boxes"] = () => VIEWS.editor.toggleBoxes();
ACT["ed-save"] = () => VIEWS.editor.save();
ACT["ed-revert"] = () => { VIEWS.editor.revert(); toast("已撤销本页未保存的修改", "info", { duration: 2000 }); };
ACT["ed-reload"] = () => { const st = VIEWS.editor.st; VIEWS.editor.open(st.job, st.index, true); };
ACT["ed-retranslate"] = async btn => {
  const ed = VIEWS.editor, st = ed.st;
  if (!st.page) return;
  if (ed.isDirty() && !(await modal({ title: "重新翻译本页？", danger: true, ok: "重新翻译", text: "本页有未保存的修改，重新翻译会用新的机器译文覆盖它们。" }))) return;
  await busy(btn, async () => {
    const page = await call("retranslate_page", { job_id: st.job, index: st.index }, { timeout: 240000 });
    if (page) { ed.applyPage(page, true); toast("本页已重新翻译并排版", "success"); }
  });
};
ACT["ed-redetect"] = async btn => {
  const ed = VIEWS.editor, st = ed.st;
  if (!st.page || !(await modal({ title: "重新检测本页？", danger: true, ok: "重新检测", icon: "scan",
    text: "会重新检测文字块、识别原文并翻译；这一页手动改过的译文和样式都会丢失。" }))) return;
  await busy(btn, async () => {
    const page = await call("redetect_page", { job_id: st.job, index: st.index }, { timeout: 240000 });
    if (page) { ed.applyPage(page, false); toast(`重新检测完成，找到 ${(page.blocks || []).length} 个文字块`, "success"); }
  });
};

// ===================================================================== 术语表
VIEWS.glossary = {
  title: "术语表",
  items: [], q: "",
  mount(root) {
    root.innerHTML = `<div class="scroll" style="flex:1"><div class="page">
      <div class="page-head"><div><h1 class="page-title">术语表</h1><div class="page-sub">人名、地名、招式等的固定译法；翻译时自动套用，本地模型和 API 还会把它当作上下文参考</div></div>
        <div class="head-actions"><span class="set-saved">${ico("check")}已保存</span>
          <button class="btn" data-act="gl-import">${ico("upload")}导入 CSV</button><button class="btn" data-act="gl-export">${ico("download")}导出 CSV</button>
          <button class="btn btn-primary" data-act="gl-add">${ico("plus")}添加词条</button></div></div>
      <div class="card gl-card"><div class="gl-toolbar"><div class="search">${ico("search")}<input class="input" id="glSearch" type="search" placeholder="搜索原文、译文或备注"></div><span class="gl-count" id="glCount"></span></div>
        <div class="gl-table"><div class="gl-row head"><span>原文</span><span>译文</span><span>备注</span><span></span></div><div class="gl-body" id="glBody"></div></div></div>
      <div class="gl-tips">${ico("info", "ico sm")}<span>CSV 至少两列：原文、译文（第三列备注可选），第一行可以是表头；Excel 另存的 GBK 编码也能识别。导入时同名原文会被更新。</span></div>
    </div></div>`;
    this.saveSoon = debounce(() => this.save(), 600);
    $("#glSearch", root).addEventListener("input", debounce(e => { this.q = e.target.value.trim().toLowerCase(); this.renderRows(); }, 120));
    const body = $("#glBody", root);
    body.addEventListener("input", e => {
      const t = e.target, row = t.closest(".gl-row");
      if (!row || !t.dataset.f) return;
      const it = this.items[+row.dataset.i];
      if (it) { it[t.dataset.f] = t.value; this.saveSoon(); }
    });
    body.addEventListener("keydown", e => {
      if (e.key !== "Enter" || !e.target.matches(".input")) return;
      e.preventDefault();
      const next = e.target.closest(".gl-row").nextElementSibling;
      const inp = next && next.querySelector(`[data-f="${e.target.dataset.f}"]`);
      if (inp) inp.focus(); else ACT["gl-add"]();
    });
  },
  show() { this.load(); },
  load() {
    this.items = (getPath(S.settings, "translate.glossary") || []).map(x => ({ src: x.src || "", dst: x.dst || "", note: x.note || "" }));
    this.renderRows();
  },
  clean() { return this.items.filter(x => x.src.trim()).map(x => ({ src: x.src, dst: x.dst, note: x.note })); },
  onSettings(s, fromSelf) {
    const body = $("#glBody");
    if (fromSelf || (body && body.contains(document.activeElement))) return;
    const incoming = JSON.stringify((getPath(s, "translate.glossary") || []).map(x => ({ src: x.src, dst: x.dst, note: x.note || "" })));
    if (incoming !== JSON.stringify(this.clean())) this.load();
  },
  async save() {
    const r = await call("save_glossary", { items: this.clean() });
    if (r && S.settings) { S.settings.translate.glossary = r; updateGlossBadge(); flashSaved(); }
  },
  renderRows() {
    const body = $("#glBody");
    if (!body) return;
    const q = this.q, LIMIT = 500, rows = [];
    let shown = 0;
    this.items.forEach((it, i) => {
      if (q && !`${it.src}\n${it.dst}\n${it.note}`.toLowerCase().includes(q)) return;
      if (shown++ >= LIMIT) return;
      rows.push(`<div class="gl-row${it._fresh ? " fresh" : ""}" data-i="${i}"><input class="input" data-f="src" value="${esc(it.src)}" placeholder="原文，如 ルフィ" spellcheck="false">
        <input class="input" data-f="dst" value="${esc(it.dst)}" placeholder="译文，如 路飞" spellcheck="false"><input class="input" data-f="note" value="${esc(it.note)}" placeholder="备注（可选）">
        <button class="btn-icon sm danger" data-act="gl-del" data-tip="删除这条">${ico("trash")}</button></div>`);
      delete it._fresh;
    });
    const total = this.items.length;
    $("#glCount").textContent = q ? `找到 ${shown} 条 · 共 ${total} 条` : `共 ${total} 条`;
    if (!total) body.innerHTML = emptyHtml("book", "术语表还是空的", "添加常用译名后，所有任务的翻译都会自动统一用词",
      `<button class="btn btn-primary" data-act="gl-add">${ico("plus")}添加第一条</button><button class="btn" data-act="gl-import">${ico("upload")}导入 CSV</button>`);
    else if (!shown) body.innerHTML = `<div class="empty" style="padding:30px"><div class="empty-title">没有匹配的词条</div><div class="empty-sub">换个关键词试试</div></div>`;
    else body.innerHTML = rows.join("") + (shown > LIMIT ? `<div class="gl-more">还有 ${shown - LIMIT} 条没有显示，请用搜索缩小范围</div>` : "");
  },
};
ACT["gl-add"] = () => {
  const v = VIEWS.glossary;
  if (v.q) { v.q = ""; const s = $("#glSearch"); if (s) s.value = ""; }
  v.items.unshift({ src: "", dst: "", note: "", _fresh: true });
  v.renderRows();
  const first = $('#glBody .gl-row[data-i="0"] [data-f="src"]');
  if (first) first.focus();
};
ACT["gl-del"] = btn => {
  const v = VIEWS.glossary, i = +btn.closest(".gl-row").dataset.i;
  const [it] = v.items.splice(i, 1);
  v.renderRows();
  v.save();
  if (it && it.src.trim()) toast(`已删除「${it.src}」`, "info", { actions: [{ label: "撤销", run: () => { v.items.splice(i, 0, it); v.renderRows(); v.save(); } }] });
};
ACT["gl-import"] = () => $("#csvInput").click();
$("#csvInput").addEventListener("change", async e => {
  const f = e.target.files && e.target.files[0];
  e.target.value = "";
  if (!f) return;
  if (f.size > 8 * 1024 * 1024) { toast("文件太大了（超过 8 MB）", "error"); return; }
  const buf = await f.arrayBuffer();
  let text;
  try { text = new TextDecoder("utf-8", { fatal: true }).decode(buf); }
  catch (_) { try { text = new TextDecoder("gbk").decode(buf); } catch (__) { text = new TextDecoder("utf-8").decode(buf); } }
  const has = (getPath(S.settings, "translate.glossary") || []).length;
  let replace = false;
  if (has) {
    const r = await modal({ title: `导入「${f.name}」`, icon: "upload", text: `当前已有 ${has} 条词条，要怎么导入？`,
      buttons: [{ label: "取消", value: false }, { label: "替换全部", value: "replace", kind: "ghost" }, { label: "合并（同名更新）", value: "merge", kind: "primary" }] });
    if (!r) return;
    replace = r === "replace";
  }
  const res = await call("import_glossary", { text, replace });
  if (!res) return;
  if (S.settings) S.settings.translate.glossary = res.items;
  updateGlossBadge();
  if (VIEWS.glossary.mounted) VIEWS.glossary.load();
  toast(`导入完成：新增 ${res.added} 条，更新 ${res.updated} 条`, "success");
});
ACT["gl-export"] = async btn => {
  const r = await busy(btn, () => call("export_glossary"));
  if (!r) return;
  if (r.path) { toast(`已导出 ${r.count} 条到「${basename(r.path)}」`, "success", { actions: [{ label: "在文件夹中显示", run: () => call("reveal", { path: r.path }) }] }); return; }
  if (S.app.window) return;                          // 桌面程序里取消了保存对话框
  const csv = r.csv.charCodeAt(0) === 0xfeff ? r.csv : "﻿" + r.csv;   // 带 BOM，Excel 打开不乱码
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  a.download = "术语表.csv";
  document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1500);
  toast(`已导出 ${r.count} 条`, "success");
};

// ===================================================================== 设置
const pct = v => Math.round(v * 100) + "%";
const SECTIONS = [
  { id: "general", icon: "monitor", title: "常规", sub: "外观与启动行为", rows: [
    { k: "general.theme", t: "seg", label: "外观", desc: "跟随系统会随 Windows 的深色 / 浅色模式自动切换", opts: [["auto", "跟随系统"], ["light", "浅色"], ["dark", "深色"]] },
    { k: "general.auto_resume", t: "switch", label: "启动时继续未完成的任务", desc: "上次关闭软件时没做完的书，下次打开自动接着做" },
    { k: "general.notify_done", t: "switch", label: "完成时通知", desc: "一本书汉化完成后弹出系统通知" },
  ] },
  { id: "hardware", icon: "chip", title: "性能与硬件", sub: "画质档位决定用多大的模型和修复分辨率", rows: [
    { k: "hardware.profile", t: "cards", label: "画质档位", opts: [
      ["fast", "极速", "bolt", "小模型 + 较低修复分辨率，适合轻薄本和老电脑"],
      ["balanced", "均衡", "gauge", "速度与画质兼顾，大多数电脑推荐"],
      ["quality", "高质量", "sparkle", "大模型 + 高分辨率修复，需要较好的显卡"]] },
    { k: "hardware.device", t: "select", label: "运行设备", desc: "自动：NVIDIA 显卡用 CUDA，AMD / Intel 显卡用 DirectML，否则用 CPU",
      opts: [["auto", "自动选择（推荐）"], ["cuda", "NVIDIA CUDA"], ["dml", "DirectML（AMD / Intel 显卡）"], ["cpu", "仅用 CPU"]] },
    { k: "hardware.cpu_threads", t: "num", label: "CPU 线程数", desc: "0 = 自动", min: 0, max: 256, suffix: "线程" },
    { k: "hardware.gpu_id", t: "num", label: "显卡编号", desc: "有多块显卡时指定用哪一块（0 = 第一块）", min: 0, max: 16 },
  ] },
  { id: "source", icon: "file-text", title: "语言与阅读方向", sub: "原文语言、竖排列和气泡的先后顺序", rows: [
    { k: "source.lang", t: "select", label: "原文语言", opts: [["ja", "日语（日漫）"], ["ko", "韩语（韩漫）"], ["en", "英语（美漫）"], ["zh", "中文（国漫 · 繁简转换）"], ["auto", "自动识别"]] },
    { k: "source.reading_order", t: "seg", label: "阅读方向（竖排列 / 气泡）", desc: "竖排每列从上到下，按这里选择的方向换列；同页气泡也按此排序。每次导入独立保存", opts: [["ltr", "从左到右"], ["rtl", "从右到左"]] },
  ] },
  { id: "translate", icon: "translate", title: "翻译", custom: true },
  { id: "typeset", icon: "type", title: "排版与字体", sub: "译文嵌回气泡时的字体和版式", rows: [
    { k: "typeset.font_dialog", t: "font", label: "对白字体", desc: "气泡里的台词" },
    { k: "typeset.font_free", t: "font", label: "旁白字体", desc: "气泡外的旁白、招牌等" },
    { k: "typeset.font_sfx", t: "font", label: "拟声词字体", desc: "「轰」「砰」这类效果字" },
    { k: "typeset.direction", t: "seg", label: "气泡内文字方向", desc: "自动 = 跟随原文；竖排每列从上到下，横排每行从左到右", opts: [["auto", "自动"], ["v", "竖排"], ["h", "横排"]] },
    { k: "typeset.min_font_px", t: "num", label: "最小字号", desc: "0 = 按页面分辨率自动", min: 0, max: 400, suffix: "px" },
    { k: "typeset.max_font_px", t: "num", label: "最大字号", desc: "0 = 自动", min: 0, max: 400, suffix: "px" },
    { k: "typeset.line_spacing", t: "range", label: "行距", min: 0.8, max: 2, step: 0.02, fmt: v => v.toFixed(2) + " 倍" },
    { k: "typeset.letter_spacing", t: "range", label: "字距", min: -0.1, max: 0.5, step: 0.01, fmt: pct },
    { k: "typeset.stroke_free", t: "range", label: "旁白描边", desc: "气泡外文字的描边宽度（相对字号）", min: 0, max: 0.4, step: 0.01, fmt: pct },
    { k: "typeset.bubble_margin", t: "range", label: "气泡内留白", desc: "文字离气泡边缘的距离（相对气泡短边）", min: 0, max: 0.4, step: 0.01, fmt: pct },
    { k: "typeset.punct_style", t: "seg", label: "引号风格", opts: [["cjk", "「直角引号」"], ["western", "“弯引号”"]] },
    { k: "typeset.keep_original_color", t: "switch", label: "保留原文颜色", desc: "彩色字、白字黑底等按原样还原" },
  ] },
  { id: "inpaint", icon: "layers", title: "去字", sub: "只在文字像素范围内修改，绝不动画面其它部分", rows: [
    { k: "inpaint.mode", t: "cards", label: "去字方式", opts: [
      ["auto", "智能（推荐）", "sparkle", "纯色气泡直接填色，复杂背景用 AI 修复"],
      ["lama", "全部 AI 修复", "layers", "效果最自然，速度最慢"],
      ["fill", "全部填色", "bolt", "最快；复杂背景上可能留下色块"]] },
    { k: "inpaint.max_side", t: "num", label: "修复分辨率上限", desc: "0 = 跟随画质档位；数值越大越清晰、越慢", min: 0, max: 8192, step: 64, suffix: "px" },
  ] },
  { id: "output", icon: "download", title: "输出", sub: "成品的格式、画质和保存位置", rows: [
    { k: "output.format", t: "select", label: "输出格式", opts: [["same", "与原文件相同"], ["folder", "图片文件夹"], ["cbz", "CBZ 漫画包"], ["zip", "ZIP 压缩包"], ["pdf", "PDF"], ["epub", "EPUB"]] },
    { k: "output.image_format", t: "select", label: "图片格式", opts: [["same", "与原图相同"], ["jpg", "JPG"], ["png", "PNG（无损）"], ["webp", "WEBP"]] },
    { k: "output.jpeg_quality", t: "range", label: "JPG 质量", min: 50, max: 100, step: 1, fmt: v => String(v) },
    { k: "output.webp_quality", t: "range", label: "WEBP 质量", min: 50, max: 100, step: 1, fmt: v => String(v) },
    { k: "output.dir", t: "dir", field: "dir", label: "输出目录", desc: "留空 = 和源文件放在同一个文件夹", placeholder: "和源文件同目录" },
    { k: "output.suffix", t: "text", label: "文件名后缀", desc: "例如「海边的约定 01_汉化.cbz」", placeholder: "_汉化" },
    { k: "output.keep_workdir", t: "switch", label: "保留中间结果", desc: "保留后可以随时回到校对页修改；关闭可节省磁盘空间" },
    { k: "output.work_dir", t: "dir", field: "work_dir", label: "工作目录", desc: "存放中间结果的位置（体积较大，可以放到空间大的盘）", placeholder: "默认（数据目录）" },
  ] },
  { id: "data", icon: "folder", title: "数据与诊断", custom: true },
];
const TR_ROWS = [
  { k: "translate.target", t: "seg", label: "译成", opts: [["zh-Hans", "简体中文"], ["zh-Hant", "繁體中文"]] },
  { k: "translate.sfx_mode", t: "select", label: "拟声词", desc: "「轰」「砰」这类效果字的处理方式", opts: [["translate", "翻译并替换"], ["keep", "保留原样"], ["erase", "只擦除，不嵌字"]] },
  { k: "translate.translate_free_text", t: "switch", label: "翻译气泡外的文字", desc: "旁白、招牌、手写字等" },
  { k: "translate.concurrency", t: "num", label: "同时翻译的请求数", desc: "在线引擎 / API 的并发数；经常被限流时调小", min: 1, max: 8 },
  { k: "translate.timeout", t: "num", label: "单次请求超时", min: 3, max: 600, suffix: "秒" },
  { k: "translate.max_retries", t: "num", label: "失败重试次数", min: 0, max: 10, suffix: "次" },
];
const ROW_BY_KEY = {};
for (const s of SECTIONS) for (const r of s.rows || []) ROW_BY_KEY[r.k] = r;
for (const r of TR_ROWS) ROW_BY_KEY[r.k] = r;
ROW_BY_KEY["translate.local.gpu_layers"] = { min: -1, max: 999 };
const FONT_SAMPLE = { "typeset.font_dialog": "你在说什么？快跑！", "typeset.font_free": "那一年夏天，我们相遇了", "typeset.font_sfx": "轰隆隆！砰！" };
const FONT_SLOT = { "typeset.font_dialog": "dialog", "typeset.font_free": "free", "typeset.font_sfx": "sfx" };
const BACKEND_LABEL = { cuda: "NVIDIA CUDA", vulkan: "Vulkan 显卡加速", cpu: "CPU" };
const isUnavailable = e => !!e && e.kind === "free" && (["edge", "tencent"].includes(e.id) || /暂不可用|失效|下线|过期/.test(e.note || ""));

function coerce(k, raw) {
  const d = getPath(S.defaults, k);
  if (typeof d === "boolean") return !!raw;
  if (typeof d === "number") {
    let n = Number(raw);
    if (raw === "" || !isFinite(n)) n = d;
    const r = ROW_BY_KEY[k];
    if (r && r.min != null) n = clamp(n, r.min, r.max);
    return Number.isInteger(d) ? Math.round(n) : n;
  }
  return raw == null ? "" : String(raw);
}
function rowHtml(r) {
  const v = getPath(S.settings, r.k), k = esc(r.k);
  const same = x => String(x) === String(v);
  let ctl = "";
  switch (r.t) {
    case "switch": ctl = `<label class="switch"><input type="checkbox" data-k="${k}"${v ? " checked" : ""}><span class="track"></span></label>`; break;
    case "select": ctl = `<select class="select" data-k="${k}">${r.opts.map(([x, l]) => `<option value="${esc(x)}"${same(x) ? " selected" : ""}>${esc(l)}</option>`).join("")}</select>`; break;
    case "seg": ctl = `<div class="seg" data-k="${k}" role="radiogroup">${r.opts.map(([x, l]) => `<button type="button" data-v="${esc(x)}"${same(x) ? ' class="on"' : ""}>${esc(l)}</button>`).join("")}</div>`; break;
    case "num": ctl = `<div class="input-group"><input class="input num" type="number" data-k="${k}" value="${esc(v)}" min="${r.min}" max="${r.max}" step="${r.step || 1}">${r.suffix ? `<span class="input-suffix">${r.suffix}</span>` : ""}</div>`; break;
    case "range": ctl = `<div class="range-wrap"><input class="range" type="range" data-k="${k}" min="${r.min}" max="${r.max}" step="${r.step}" value="${esc(v)}"><span class="range-val">${esc(r.fmt ? r.fmt(+v) : v)}</span></div>`; break;
    case "text": ctl = `<input class="input text" type="text" data-k="${k}" value="${esc(v)}" placeholder="${esc(r.placeholder || "")}">`; break;
    case "dir": ctl = `<div class="dir-ctl"><input class="input mono" type="text" readonly data-k="${k}" value="${esc(v)}" placeholder="${esc(r.placeholder || "")}" title="${esc(v || r.placeholder || "")}">
      <button class="btn" data-act="pick-dir" data-field="${r.field}">${ico("folder")}选择…</button>${v ? `<button class="btn-icon" data-act="clear-dir" data-k="${k}" data-tip="恢复默认">${ico("x")}</button>` : ""}</div>`; break;
    case "font": ctl = `<div class="font-pick" data-k="${k}"><button type="button" class="font-btn" data-act="font-menu"><span class="fname">${esc(fontName(v))}</span><img class="fprev" alt="" src="${fontUrl(v, FONT_SAMPLE[r.k])}">${ico("chev-d")}</button></div>`; break;
    case "cards": ctl = `<div class="opt-cards" data-k="${k}">${r.opts.map(([x, l, i, d]) => `<button type="button" class="opt-card${same(x) ? " on" : ""}" data-v="${esc(x)}"><b>${ico(i)}${esc(l)}</b><span>${esc(d)}</span></button>`).join("")}</div>`; break;
    default: break;
  }
  return `<div class="set-row${r.t === "cards" ? " stack" : ""}" data-row="${k}"><div><div class="set-label">${esc(r.label)}</div>${r.desc ? `<div class="set-desc">${esc(r.desc)}</div>` : ""}</div><div class="set-ctl">${ctl}</div></div>`;
}
function testResHtml(id) {
  const r = S.testRes[id], a = `data-res="${esc(id)}"`;
  if (!r) return `<span class="test-res" ${a}></span>`;
  if (r.ok == null) return `<span class="test-res" ${a}><span class="spinner sm"></span><span class="ellipsis">${esc(r.msg)}</span></span>`;
  return `<span class="test-res ${r.ok ? "ok" : "bad"}" ${a} title="${esc(r.msg)}">${ico(r.ok ? "check-circle" : "alert-circle")}<span class="ellipsis">${esc(r.msg)}</span></span>`;
}
function setTestRes(id, ok, msg) {
  S.testRes[id] = { ok, msg };
  $$(`[data-res="${CSS.escape(id)}"]`).forEach(x => { x.outerHTML = testResHtml(id); });
}
function closeFontMenus() { $$(".font-menu").forEach(m => m.remove()); }
document.addEventListener("pointerdown", e => { if (!e.target.closest(".font-pick")) closeFontMenus(); }, true);

VIEWS.settings = {
  title: "设置",
  timers: {},
  mount(root) {
    root.innerHTML = `<div class="scroll" id="setScroll" style="flex:1"><div class="page">
      <div class="page-head"><div><h1 class="page-title">设置</h1><div class="page-sub">修改即时保存；每次新导入建立独立任务，使用当时的设置。已有版本不会随全局设置改变</div></div>
        <div class="head-actions"><span class="set-saved">${ico("check")}已自动保存</span></div></div>
      <div class="settings"><nav class="set-nav" id="setNav">${SECTIONS.map(s => `<a href="#/settings/${s.id}" data-sec="${s.id}">${ico(s.icon)}${s.title}</a>`).join("")}
        <div class="nav-foot"><button class="btn btn-sm btn-ghost btn-block" data-act="reset-all">${ico("undo")}恢复全部默认</button></div></nav>
      <div class="set-sections" id="setSections"></div></div></div></div>`;
    this.scroller = $("#setScroll", root);
    this.scroller.addEventListener("scroll", () => this.spy(), { passive: true });
    for (const ev of ["wheel", "pointerdown", "keydown"]) this.scroller.addEventListener(ev, () => { this.pendingScroll = null; }, { passive: true });
    $("#setNav", root).addEventListener("click", e => {
      const a = e.target.closest("a[data-sec]");
      if (!a) return;
      e.preventDefault();
      this.scrollTo(a.dataset.sec);
      setRouteArgs([a.dataset.sec]);
    });
    const sec = $("#setSections", root);
    sec.addEventListener("change", e => this.onChange(e));
    sec.addEventListener("input", e => this.onInput(e));
    sec.addEventListener("click", e => this.onClick(e));
    sec.addEventListener("toggle", e => { if (e.target.classList && e.target.classList.contains("set-adv")) this.advOpen = e.target.open; }, true);
    sec.addEventListener("dragstart", e => {
      const it = e.target.closest && e.target.closest(".eng[draggable]");
      if (!it) return;
      this.dragEng = it.dataset.eng;
      it.classList.add("dragging");
      e.dataTransfer.effectAllowed = "move";
      e.dataTransfer.setData("text/plain", it.dataset.eng);
    });
    sec.addEventListener("dragover", e => {
      const it = e.target.closest(".eng");
      if (!this.dragEng || !it || it.classList.contains("off")) return;
      e.preventDefault();
      $$(".eng.drag-over").forEach(x => { if (x !== it) x.classList.remove("drag-over"); });
      it.classList.add("drag-over");
    });
    sec.addEventListener("drop", e => {
      const it = e.target.closest(".eng");
      if (!this.dragEng || !it || it.dataset.eng === this.dragEng) return;
      e.preventDefault();
      const order = (getPath(S.settings, "translate.engines") || []).filter(x => x !== this.dragEng);
      const at = order.indexOf(it.dataset.eng);
      if (at < 0) return;
      const r = it.getBoundingClientRect();
      order.splice(at + (e.clientY > r.top + r.height / 2 ? 1 : 0), 0, this.dragEng);
      this.saveEngines(order);
    });
    sec.addEventListener("dragend", () => { this.dragEng = null; $$(".eng.dragging, .eng.drag-over").forEach(x => x.classList.remove("dragging", "drag-over")); });
    this.render();
  },
  async show(args) {
    if (!S.llm) loadLlm();
    this.ensurePresets();
    const id = args && args[0];
    // 字体 / 模型表异步到达后布局会变：3 秒内再校正一次滚动位置（用户自己滚动则取消）
    this.pendingScroll = id ? { id, until: Date.now() + 3000 } : null;
    if (!id) { this.spy(); return; }
    requestAnimationFrame(() => this.scrollTo(id));
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => requestAnimationFrame(() => this.rescroll()));
  },
  rescroll() {
    const p = this.pendingScroll;
    if (p && Date.now() < p.until && current === "settings") this.scrollTo(p.id);
  },
  hide() { closeFontMenus(); this.flushAll(); },
  async ensurePresets() {
    if (S.presets) return;
    const p = await call("api_presets", {}, { quiet: true });
    if (p) { S.presets = p; this.rerender("api"); }
  },
  render() {
    const box = $("#setSections");
    if (!box) return;
    box.innerHTML = SECTIONS.map(s => (s.id === "translate" ? this.translateHtml() : s.id === "data" ? this.dataHtml() : this.cardHtml(s))).join("");
    this.spy();
  },
  cardHtml(s) {
    return `<section class="card set-card" id="sec-${s.id}"><div class="card-head"><div><div class="card-title">${ico(s.icon)}${s.title}</div>${s.sub ? `<div class="card-sub">${s.sub}</div>` : ""}</div>
      <button class="link-btn" data-act="reset-sec" data-sec="${s.id}">恢复默认</button></div><div class="set-rows">${s.rows.map(rowHtml).join("")}</div></section>`;
  },
  translateHtml() { return this.localCardHtml() + this.apiCardHtml() + this.onlineCardHtml() + this.optsCardHtml(); },
  localCardHtml() {
    const on = !!getPath(S.settings, "translate.local.enabled");
    return `<section class="card set-card${on ? "" : " off"}" id="sec-translate" data-sw="translate.local.enabled">
      <div class="card-head"><div><div class="card-title">${ico("chip")}本地大模型<span class="lic-badge">推荐</span></div>
        <div class="card-sub">软件自带 llama.cpp 引擎，按电脑配置推荐模型；离线运行、不限次数、整页上下文 + 术语表，效果最好</div></div>
        <label class="switch" data-tip="开启 / 关闭本地大模型"><input type="checkbox" data-k="translate.local.enabled"${on ? " checked" : ""}><span class="track"></span></label></div>
      <div class="set-rows" id="localBody">${this.localBodyHtml()}</div></section>`;
  },
  localBodyHtml() {
    const st = S.llm;
    if (!st) return `<div class="set-row stack"><div class="llm-hw">${`<div class="hw-item"><span class="skel skel-line" style="width:40%"></span><span class="skel skel-line" style="width:70%"></span></div>`.repeat(4)}</div><div class="skel" style="height:150px;border-radius:12px"></div></div>`;
    const hw = st.hardware || {}, rec = st.recommend, recM = rec && rec.model, loc = getPath(S.settings, "translate.local") || {};
    const gpuMode = !!rec && rec.backend !== "cpu";
    const budget = gpuMode ? (hw.vram_gb || 0) - 3.1 : (hw.ram_gb || 0) - 6;
    const dl = st.download && st.download.state === "downloading" ? st.download : null;
    const models = st.models || [], active = st.active ? llmModel(st.active) : null;
    const hwHtml = `<div class="llm-hw">
      <div class="hw-item"><span>${ico("monitor", "ico sm")}显卡</span><b title="${esc(hw.gpu_full || "")}">${esc(hw.gpu || "未检测到独立显卡")}</b></div>
      <div class="hw-item"><span>${ico("chip", "ico sm")}显存</span><b>${hw.vram_gb ? fmtGB(hw.vram_gb) : "—"}</b></div>
      <div class="hw-item"><span>${ico("layers", "ico sm")}内存</span><b>${hw.ram_gb ? fmtGB(hw.ram_gb) : "—"}</b></div>
      <div class="hw-item"><span>${ico("bolt", "ico sm")}运行方式</span><b>${rec ? esc(BACKEND_LABEL[rec.backend] || rec.backend) + (rec.gpu_layers > 0 ? " · 整模型进显卡" : "") : "—"}</b></div></div>`;
    const recHtml = recM ? `<div class="rec">${ico("sparkle")}<div><b>为这台电脑推荐：${esc(recM.name)}（约 ${fmtGB(recM.size_gb)}）</b><p>${esc(rec.reason)}</p></div></div>`
      : `<div class="rec warn">${ico("alert")}<div><b>这台电脑不太适合运行本地模型</b><p>建议改用下面的「API 服务」。</p></div></div>`;
    const rows = models.map(m => this.modelRowHtml(m, st, dl, budget, gpuMode)).join("");
    const autoName = st.selected === "auto" && active ? `（当前：${active.name}）` : recM ? `（推荐 ${recM.name}）` : "";
    const opts = `<option value="auto"${st.selected === "auto" ? " selected" : ""}>自动推荐${esc(autoName)}</option>`
      + models.map(m => `<option value="${esc(m.id)}"${m.id === st.selected ? " selected" : ""}${m.installed ? "" : " disabled"}>${esc(m.name)}${m.installed ? "" : "（未下载）"}</option>`).join("");
    const backs = st.backends || [];
    const bOpts = [["auto", "自动"], ["cuda", "NVIDIA CUDA"], ["vulkan", "Vulkan（AMD / Intel 显卡）"], ["cpu", "CPU"]].map(([v, l]) => {
      const missing = v !== "auto" && !backs.includes(v);
      return `<option value="${v}"${loc.backend === v ? " selected" : ""}${missing ? " disabled" : ""}>${l}${missing ? "（本版本未附带）" : ""}</option>`;
    }).join("");
    const state = active ? `<span class="tag done">已就绪</span>` : dl ? `<span class="tag running">下载中</span>` : `<span class="tag paused">还没有可用的模型</span>`;
    return `<div class="set-row stack">${hwHtml}${recHtml}</div>
      <div class="set-row stack"><div class="llm-head"><div><div class="set-label">模型</div><div class="set-desc">自动从 Hugging Face / ModelScope（国内镜像）中选最快的源下载，支持断点续传和完整性校验</div></div>
        <button class="btn btn-sm" data-act="llm-import" data-tip="已经有 .gguf 模型文件？导入后即可使用">${ico("upload")}导入模型</button></div>
        <div class="llm-table">${rows || `<div class="llm-row"><span class="muted">没有可用的模型清单</span></div>`}</div></div>
      <div class="set-row"><div><div class="set-label">使用的模型</div><div class="set-desc">「自动推荐」会在已下载的模型里选这台电脑能流畅运行、效果最好的一个</div></div>
        <div class="set-ctl"><select class="select" data-k="translate.local.model">${opts}</select></div></div>
      <div class="set-row"><div><div class="set-label">测试本地模型 ${state}</div><div class="set-desc">启动本地模型翻译一句话；第一次启动需要 10~60 秒加载模型</div></div>
        <div class="set-ctl">${testResHtml("local")}<button class="btn" data-act="test-engine" data-eng="local">${ico("bolt")}测试</button></div></div>
      <details class="set-adv"${this.advOpen ? " open" : ""}><summary>${ico("chev-d")}高级选项</summary>
        <div class="set-row"><div><div class="set-label">运行后端</div><div class="set-desc">一般保持自动；显卡驱动有问题时可以改用 CPU</div></div><div class="set-ctl"><select class="select" data-k="translate.local.backend">${bOpts}</select></div></div>
        <div class="set-row"><div><div class="set-label">放进显卡的层数</div><div class="set-desc">-1 = 自动（放得下就全部放进显卡）；显存不够时调小</div></div><div class="set-ctl"><input class="input num" type="number" min="-1" max="999" step="1" data-k="translate.local.gpu_layers" value="${esc(loc.gpu_layers)}"></div></div>
        <div class="set-row"><div><div class="set-label">上下文长度</div><div class="set-desc">越大越能参考整页上下文，但占用更多显存 / 内存</div></div><div class="set-ctl"><select class="select" data-k="translate.local.ctx">${[4096, 8192, 16384, 32768].map(n => `<option value="${n}"${loc.ctx === n ? " selected" : ""}>${n / 1024}K</option>`).join("")}</select></div></div>
      </details>`;
  },
  modelRowHtml(m, st, dl, budget, gpuMode) {
    const mine = dl && dl.model_id === m.id ? dl : null;
    const active = st.active === m.id;
    const tooBig = !m.custom && !m.installed && budget > 0 && m.need_gb > budget;
    const badges = (m.recommended ? `<span class="lic-badge">推荐</span>` : "")
      + (active ? `<span class="tag running plain">正在使用</span>` : m.installed ? `<span class="tag done plain">已下载</span>` : "")
      + (tooBig ? `<span class="llm-warn">${gpuMode ? "显存" : "内存"}可能不够</span>` : "");
    const id = esc(m.id);
    const act = mine ? `<button class="btn btn-sm" data-act="llm-cancel">${ico("x")}取消</button>`
      : m.installed ? `<button class="btn-icon sm danger" data-act="llm-del" data-id="${id}" data-tip="删除模型文件">${ico("trash")}</button>`
      : `<button class="btn btn-sm${m.recommended ? " btn-primary" : ""}" data-act="llm-dl" data-id="${id}"${dl ? " disabled" : ""}>${ico("download")}${m.partial_bytes ? "继续下载" : "下载"}</button>`;
    const need = m.custom ? "自己导入的模型" : `需约 ${fmtGB(m.need_gb)} ${gpuMode ? "显存" : "内存"}`;
    const extra = mine ? `<div class="llm-prog" data-dl="${id}"><div class="progress"><i class="dl-fill" style="width:${dlPct(mine).toFixed(1)}%"></i></div><span class="dl-text">${esc(dlText(mine))}</span></div>`
      : m.partial_bytes && !m.installed ? `<div class="llm-part">已下载 ${fmtBytes(m.partial_bytes)}（共 ${fmtGB(m.size_gb)}），点「继续下载」接着下</div>` : "";
    return `<div class="llm-row${m.recommended ? " rec" : ""}${active ? " active" : ""}"><div class="llm-name"><b>${esc(m.name)}</b>${badges}</div>
      <div class="llm-size">${fmtGB(m.size_gb)}<span>${need}</span></div><div class="llm-act">${act}</div>${extra}</div>`;
  },
  apiCardHtml() {
    const a = getPath(S.settings, "translate.api") || {};
    const presets = S.presets || [];
    const p = presets.find(x => x.id === a.provider) || {};
    const opts = (presets.length ? presets : [{ id: a.provider, name: a.provider }])
      .map(x => `<option value="${esc(x.id)}"${x.id === a.provider ? " selected" : ""}>${esc(x.name)}</option>`).join("");
    return `<section class="card set-card${a.enabled ? "" : " off"}" id="sec-api" data-sw="translate.api.enabled">
      <div class="card-head"><div><div class="card-title">${ico("bolt")}API 服务（填 Key）</div>
        <div class="card-sub">DeepSeek、硅基流动、OpenAI、Gemini、OpenRouter 或任何 OpenAI 兼容接口；本地模型不可用时自动接替</div></div>
        <label class="switch" data-tip="开启 / 关闭 API 服务"><input type="checkbox" data-k="translate.api.enabled"${a.enabled ? " checked" : ""}><span class="track"></span></label></div>
      <div class="set-rows">
        <div class="set-row"><div><div class="set-label">服务商</div><div class="set-desc">选好后自动使用对应的服务地址</div></div><div class="set-ctl"><select class="select" data-k="translate.api.provider">${opts}</select></div></div>
        <div class="set-row"><div><div class="set-label">服务地址</div><div class="set-desc">留空 = 使用服务商的默认地址</div></div><div class="set-ctl"><input class="input text mono" type="text" spellcheck="false" data-k="translate.api.base_url" value="${esc(a.base_url)}" placeholder="${esc(p.base_url || "https://…/v1")}"></div></div>
        <div class="set-row"><div><div class="set-label">API Key</div><div class="set-desc">只保存在这台电脑上，不会上传到任何地方</div></div><div class="set-ctl"><input class="input text mono" type="password" id="apiKey" autocomplete="off" spellcheck="false" data-k="translate.api.api_key" value="${esc(a.api_key)}" placeholder="sk-…"><button class="btn-icon" data-act="toggle-key" data-tip="显示 / 隐藏 Key">${ico("eye")}</button></div></div>
        <div class="set-row"><div><div class="set-label">模型</div><div class="set-desc">${p.model ? `留空 = ${esc(p.model)}` : "填写模型名称，或点「获取模型列表」挑选"}</div><div class="api-chips" id="apiChips"></div></div>
          <div class="set-ctl"><input class="input text mono" type="text" spellcheck="false" list="apiModelList" data-k="translate.api.model" value="${esc(a.model)}" placeholder="${esc(p.model || "模型名称")}"><datalist id="apiModelList"></datalist>
          <button class="btn" data-act="api-models">${ico("refresh")}获取模型列表</button></div></div>
        <div class="set-row"><div><div class="set-label">测试连接</div><div class="set-desc">用当前填写的地址、Key 和模型翻译一句话</div></div><div class="set-ctl">${testResHtml("api")}<button class="btn" data-act="test-engine" data-eng="api">${ico("bolt")}测试</button></div></div>
      </div></section>`;
  },
  onlineCardHtml() {
    const tr = getPath(S.settings, "translate") || {};
    const free = (S.engines || []).filter(e => e.kind === "free");
    const enabled = (tr.engines || []).filter(id => free.some(e => e.id === id));
    const ids = enabled.concat(free.map(e => e.id).filter(id => !enabled.includes(id)));
    const items = ids.map(id => {
      const e = free.find(x => x.id === id), on = enabled.includes(id), bad = isUnavailable(e), rank = enabled.indexOf(id) + 1, eid = esc(id);
      return `<div class="eng${on ? "" : " off"}" data-eng="${eid}"${on ? ' draggable="true"' : ""}>
        <span class="grip"${on ? ' data-tip="拖动调整顺序"' : ""}>${ico("grip")}</span>
        <label class="switch"><input type="checkbox" data-eng-toggle="${eid}"${on ? " checked" : ""}${bad && !on ? " disabled" : ""}><span class="track"></span></label>
        <div style="min-width:0"><div class="eng-name">${esc(e.name)}${on ? `<span class="eng-rank">第 ${rank} 顺位</span>` : ""}${bad ? `<span class="tag error plain">暂不可用</span>` : ""}</div><div class="eng-desc">${esc(e.note || "")}</div></div>
        <div class="eng-tools">${testResHtml(id)}<button class="btn btn-sm" data-act="test-engine" data-eng="${eid}"${bad && !on ? " disabled" : ""}>测试</button>${on ? `
          <button class="btn-icon sm" data-act="eng-move" data-eng="${eid}" data-d="-1" data-tip="上移"${rank === 1 ? " disabled" : ""}>${ico("up")}</button>
          <button class="btn-icon sm" data-act="eng-move" data-eng="${eid}" data-d="1" data-tip="下移"${rank === enabled.length ? " disabled" : ""}>${ico("down")}</button>` : ""}</div></div>`;
    }).join("");
    return `<section class="card set-card${tr.online_fallback ? "" : " off"}" id="sec-online" data-sw="translate.online_fallback">
      <div class="card-head"><div><div class="card-title">${ico("translate")}免费在线翻译（兜底）</div><div class="card-sub">本地模型和 API 都不可用时才会使用；需要联网，按顺序自动切换，质量一般</div></div>
        <label class="switch" data-tip="开启 / 关闭免费在线兜底"><input type="checkbox" data-k="translate.online_fallback"${tr.online_fallback ? " checked" : ""}><span class="track"></span></label></div>
      <div class="set-rows"><div class="set-row stack"><div class="eng-list" id="engList">${items || `<div class="muted">没有可用的在线引擎</div>`}</div></div></div></section>`;
  },
  optsCardHtml() {
    const n = (getPath(S.settings, "translate.glossary") || []).length;
    return `<section class="card set-card" id="sec-topts"><div class="card-head"><div><div class="card-title">${ico("sliders")}翻译选项</div><div class="card-sub">对所有翻译方式都生效</div></div>
      <button class="link-btn" data-act="reset-sec" data-sec="translate">恢复默认</button></div><div class="set-rows">${TR_ROWS.map(rowHtml).join("")}
      <div class="set-row"><div><div class="set-label">术语表</div><div class="set-desc">${n ? `共 ${n} 条固定译法，翻译时自动套用` : "还没有词条；添加人名、地名等固定译法，全书用词统一"}</div></div>
        <div class="set-ctl"><button class="btn" data-act="go" data-to="glossary">${ico("book")}管理术语表</button></div></div></div></section>`;
  },
  dataHtml() {
    return `<section class="card set-card" id="sec-data"><div class="card-head"><div><div class="card-title">${ico("folder")}数据与诊断</div><div class="card-sub">日志、缓存和中间结果的位置</div></div></div><div class="set-rows">
      <div class="set-row"><div style="min-width:0"><div class="set-label">数据目录</div><div class="set-desc mono selectable ellipsis" title="${esc(S.paths.data || "")}">${esc(S.paths.data || "")}</div></div><div class="set-ctl"><button class="btn" data-act="open-data">${ico("folder")}打开</button></div></div>
      <div class="set-row"><div><div class="set-label">运行日志</div><div class="set-desc">遇到问题时把日志发给客服，可以更快找到原因</div></div><div class="set-ctl"><button class="btn" data-act="open-logs">${ico("file-text")}打开日志文件夹</button></div></div>
      <div class="set-row"><div><div class="set-label">恢复全部默认设置</div><div class="set-desc">术语表会保留；API Key 会被清空</div></div><div class="set-ctl"><button class="btn btn-danger" data-act="reset-all">${ico("undo")}恢复默认</button></div></div>
    </div></section>`;
  },
  rerender(part) {
    if (!$("#setSections")) return;
    if (part === "local") {
      const b = $("#localBody");
      if (b && !b.contains(document.activeElement)) b.innerHTML = this.localBodyHtml();
    } else {
      const id = { api: "#sec-api", online: "#sec-online", opts: "#sec-topts" }[part];
      const html = { api: () => this.apiCardHtml(), online: () => this.onlineCardHtml(), opts: () => this.optsCardHtml() }[part];
      const cur = id && $(id);
      if (cur && !cur.contains(document.activeElement)) cur.outerHTML = html();
    }
  },
  refreshRow(k) {
    const row = $(`.set-row[data-row="${CSS.escape(k)}"]`), r = ROW_BY_KEY[k];
    if (row && r) row.outerHTML = rowHtml(r);
  },
  updateOff() {
    $$("#setSections .set-card[data-sw]").forEach(c => c.classList.toggle("off", !getPath(S.settings, c.dataset.sw)));
  },
  sync() {
    const root = $("#setSections");
    if (!root) return;
    for (const t of $$("[data-k]", root)) {
      const k = t.dataset.k;
      if (t.tagName === "BUTTON" || t === document.activeElement || this.timers[k]) continue;
      const v = getPath(S.settings, k);
      if (t.matches(".seg, .opt-cards")) $$(":scope > button", t).forEach(x => x.classList.toggle("on", String(x.dataset.v) === String(v)));
      else if (t.classList.contains("font-pick")) {
        const n = t.querySelector(".fname"), im = t.querySelector(".fprev"), u = fontUrl(v, FONT_SAMPLE[k]);
        if (n) n.textContent = fontName(v);
        if (im && im.getAttribute("src") !== u) im.src = u;
      } else if (t.type === "checkbox") t.checked = !!v;
      else if (t.type === "range") {
        t.value = v;
        const lab = t.parentElement.querySelector(".range-val"), r = ROW_BY_KEY[k];
        if (lab) lab.textContent = r && r.fmt ? r.fmt(+v) : v;
      } else if (v != null && String(t.value) !== String(v)) t.value = v;
    }
    this.updateOff();
  },
  onSettings(s, fromSelf) {
    if (!$("#setSections")) return;
    this.sync();
    if (!fromSelf) { this.rerender("online"); this.rerender("opts"); }
  },
  onLlm() { this.rerender("local"); requestAnimationFrame(() => this.rescroll()); },
  onChange(e) {
    const t = e.target;
    if (t.dataset.engToggle) { this.toggleEngine(t.dataset.engToggle, t.checked); return; }
    const k = t.dataset.k;
    if (!k) return;
    if (/^(range|text|password|number)$/.test(t.type)) { this.flushKey(k, t); return; }
    this.saveKey(k, t.type === "checkbox" ? t.checked : t.value);
  },
  onInput(e) {
    const t = e.target, k = t.dataset.k;
    if (!k || !/^(range|text|password|number)$/.test(t.type)) return;
    if (t.type === "range") {
      const lab = t.parentElement.querySelector(".range-val"), r = ROW_BY_KEY[k];
      if (lab) lab.textContent = r && r.fmt ? r.fmt(+t.value) : t.value;
    }
    clearTimeout(this.timers[k]);
    this.timers[k] = setTimeout(() => { delete this.timers[k]; this.saveKey(k, t.value); }, t.type === "range" ? 250 : 600);
  },
  onClick(e) {
    const b = e.target.closest(".seg[data-k] > button, .opt-cards[data-k] > .opt-card");
    if (!b || b.classList.contains("on")) return;
    $$(":scope > button", b.parentElement).forEach(x => x.classList.toggle("on", x === b));
    this.saveKey(b.parentElement.dataset.k, b.dataset.v);
  },
  flushKey(k, t) {
    if (this.timers[k]) { clearTimeout(this.timers[k]); delete this.timers[k]; this.saveKey(k, t.value); }
  },
  async flushAll() {
    for (const k of Object.keys(this.timers)) {
      clearTimeout(this.timers[k]);
      delete this.timers[k];
      const t = $(`#setSections [data-k="${CSS.escape(k)}"]`);
      if (t) await this.saveKey(k, t.value);
    }
  },
  async saveKey(k, raw) {
    const v = coerce(k, raw);
    let patch = patchOf(k, v);
    if (k === "translate.api.provider") {
      const p = (S.presets || []).find(x => x.id === v) || {};
      patch = { translate: { api: { provider: v, base_url: "", model: p.model || "" } } };
    }
    const s = await saveSettings(patch);
    if (!s) { this.sync(); return; }
    this.sync();
    if (k === "translate.api.provider") this.rerender("api");
    if (k === "translate.local.model" || k === "translate.local.enabled") loadLlm();
    if (ROW_BY_KEY[k] && ROW_BY_KEY[k].t === "dir") this.refreshRow(k);
  },
  toggleEngine(id, on) {
    const order = (getPath(S.settings, "translate.engines") || []).slice(), i = order.indexOf(id);
    if (on && i < 0) order.push(id);
    if (!on && i >= 0) {
      if (order.length === 1) { toast("至少保留一个在线引擎；不想用在线翻译，请关闭上面的总开关", "warning"); this.rerender("online"); return; }
      order.splice(i, 1);
    }
    this.saveEngines(order);
  },
  async saveEngines(order) { await saveSettings({ translate: { engines: order } }); this.rerender("online"); },
  spy() {
    if (!this.scroller) return;
    const top = this.scroller.getBoundingClientRect().top + 100;
    let cur = SECTIONS[0].id;
    for (const s of SECTIONS) { const c = $("#sec-" + s.id); if (c && c.getBoundingClientRect().top <= top) cur = s.id; }
    $$("#setNav a").forEach(a => a.classList.toggle("on", a.dataset.sec === cur));
  },
  scrollTo(id) {
    const c = $("#sec-" + id);
    if (!c) return;
    const top = c.getBoundingClientRect().top - this.scroller.getBoundingClientRect().top + this.scroller.scrollTop - 14;
    this.scroller.scrollTo({ top, behavior: SHOT ? "auto" : "smooth" });
    $$("#setNav a").forEach(a => a.classList.toggle("on", a.dataset.sec === id));
  },
  onKey(e) { if (e.key === "Escape") closeFontMenus(); },
};
ACT["font-menu"] = btn => {
  const pick = btn.closest(".font-pick"), had = pick.querySelector(".font-menu");
  closeFontMenus();
  if (had) return;
  const k = pick.dataset.k, cur = getPath(S.settings, k), sample = FONT_SAMPLE[k], slot = FONT_SLOT[k];
  const tags = f => f.tags || [];
  const fonts = (S.fonts || []).slice().sort((a, b) => (tags(b).includes(slot) ? 1 : 0) - (tags(a).includes(slot) ? 1 : 0));
  const menu = el(`<div class="font-menu" role="listbox">${fonts.map(f => `<button type="button" class="font-opt${f.id === cur ? " on" : ""}" data-font="${esc(f.id)}">
      <span class="fname"><span>${esc(f.name)}</span><span class="muted">${tags(f).includes(slot) ? "推荐" : tags(f).includes("user") ? "已添加" : ""}</span></span>
      <img alt="" loading="lazy" src="${fontUrl(f.id, sample)}"></button>`).join("")}
    <div class="menu-foot"><button type="button" class="btn btn-sm btn-ghost btn-block" data-act="add-font">${ico("plus")}添加字体文件（TTF / OTF）…</button></div></div>`);
  pick.appendChild(menu);
  const on = menu.querySelector(".font-opt.on");
  if (on) on.scrollIntoView({ block: "nearest" });
  menu.addEventListener("click", e => {
    const o = e.target.closest(".font-opt");
    if (!o) return;
    closeFontMenus();
    VIEWS.settings.saveKey(k, o.dataset.font);
  });
};
ACT["add-font"] = async btn => {
  const r = await busy(btn, () => call("add_font"));
  if (!r) return;
  if (r.fonts) S.fonts = r.fonts;
  closeFontMenus();
  if (r.added && r.added.length) toast(`已添加 ${r.added.length} 个字体，可以在字体列表里选用`, "success");
  if (r.errors && r.errors.length) toast(r.errors.join("；"), "warning");
};
ACT["test-engine"] = async btn => {
  const id = btn.dataset.eng;
  if (VIEWS.settings.mounted) await VIEWS.settings.flushAll();
  await busy(btn, async () => {
    setTestRes(id, null, id === "local" ? "正在启动本地模型…" : "测试中…");
    try {
      const r = await api("test_engine", { engine_id: id }, { timeout: id === "local" ? 300000 : 90000 });
      setTestRes(id, !!r.ok, r.message);
    } catch (e) { setTestRes(id, false, e.message); }
  });
};
ACT["eng-move"] = btn => {
  const order = (getPath(S.settings, "translate.engines") || []).slice();
  const i = order.indexOf(btn.dataset.eng), j = i + Number(btn.dataset.d);
  if (i < 0 || j < 0 || j >= order.length) return;
  [order[i], order[j]] = [order[j], order[i]];
  VIEWS.settings.saveEngines(order);
};
ACT["api-models"] = async btn => {
  await VIEWS.settings.flushAll();
  const a = getPath(S.settings, "translate.api") || {};
  await busy(btn, async () => {
    const r = await call("api_models", { base_url: a.base_url || "", api_key: a.api_key || "", provider: a.provider || "" }, { timeout: 30000 });
    if (!r) return;
    const list = r.models || [];
    const dlist = $("#apiModelList");
    if (dlist) dlist.innerHTML = list.map(m => `<option value="${esc(m)}">`).join("");
    const chips = $("#apiChips");
    if (chips) chips.innerHTML = list.slice(0, 40).map(m => `<button type="button" class="chip${m === a.model ? " on" : ""}" data-act="api-pick" data-m="${esc(m)}">${esc(m)}</button>`).join("")
      + (list.length > 40 ? `<span class="muted">等共 ${list.length} 个</span>` : "");
    toast(list.length ? `获取到 ${list.length} 个模型，点一下即可选用` : "这个服务没有返回模型列表，请手动填写模型名称", list.length ? "success" : "warning");
  });
};
ACT["api-pick"] = async btn => {
  const inp = $('#setSections [data-k="translate.api.model"]');
  if (inp) inp.value = btn.dataset.m;
  $$("#apiChips .chip").forEach(c => c.classList.toggle("on", c === btn));
  await VIEWS.settings.saveKey("translate.api.model", btn.dataset.m);
};
ACT["toggle-key"] = btn => {
  const i = $("#apiKey");
  if (!i) return;
  i.type = i.type === "password" ? "text" : "password";
  btn.classList.toggle("on", i.type === "text");
};
ACT["pick-dir"] = async btn => {
  const r = await busy(btn, () => call("pick_output_dir", { field: btn.dataset.field }));
  if (r && r.path) { applySettings(r.settings, true); flashSaved(); VIEWS.settings.refreshRow("output." + btn.dataset.field); }
};
ACT["clear-dir"] = btn => VIEWS.settings.saveKey(btn.dataset.k, "");
ACT["reset-sec"] = async btn => {
  const id = btn.dataset.sec, s = SECTIONS.find(x => x.id === id);
  const text = id === "translate" ? "翻译相关的设置（本地模型选择、API 地址和 Key、在线引擎顺序等）会恢复默认；术语表会保留。"
    : `「${s ? s.title : id}」里的设置会恢复成默认值。`;
  if (!(await modal({ title: "恢复默认设置？", text, danger: true, ok: "恢复默认", icon: "undo" }))) return;
  const r = await call("reset_settings", { section: id });
  if (r) { applySettings(r, true); VIEWS.settings.render(); flashSaved(); toast("已恢复默认", "success"); loadLlm(); }
};
ACT["reset-all"] = async () => {
  if (!(await modal({ title: "恢复全部默认设置？", text: "所有设置都会恢复成刚安装时的样子（API Key 会被清空），术语表会保留。", danger: true, ok: "全部恢复", icon: "undo" }))) return;
  const r = await call("reset_settings", {});
  if (r) { applySettings(r, true); applyTheme(themePref(), true); if (VIEWS.settings.mounted) VIEWS.settings.render(); toast("已恢复全部默认设置", "success"); loadLlm(); }
};

// ===================================================================== 关于
const LICENSES = [
  ["AI 模型", [
    ["文字 / 气泡检测", "RT-DETR-v2（漫画 / 韩漫 / 国漫 / 美漫微调）", "Apache-2.0"],
    ["日文识别", "manga-ocr", "Apache-2.0"],
    ["韩文 / 英文 / 中文识别", "PaddleOCR PP-OCR", "Apache-2.0"],
    ["去字修复", "LaMa · AnimeMangaInpainting", "Apache-2.0"],
    ["本地翻译模型", "Qwen 通义千问（GGUF）", "Apache-2.0"],
  ]],
  ["字体", [
    ["对白", "思源黑体 Noto Sans SC / TC", "OFL-1.1"],
    ["旁白", "霞鹜文楷 LXGW WenKai", "OFL-1.1"],
    ["拟声词", "站酷快乐体 ZCOOL KuaiLe", "OFL-1.1"],
    ["毛笔", "马善政毛笔 Ma Shan Zheng", "OFL-1.1"],
  ]],
  ["运行库", [
    ["AI 推理", "ONNX Runtime", "MIT"],
    ["本地大模型引擎", "llama.cpp", "MIT"],
    ["图像处理", "OpenCV", "Apache-2.0"],
    ["数值计算 / 图片读写", "NumPy · Pillow", "BSD · MIT-CMU"],
    ["PDF 读写", "pypdfium2 · PDFium", "Apache-2.0 · BSD"],
    ["界面窗口", "pywebview · Microsoft WebView2", "BSD-3-Clause"],
    ["网络 / 文本", "requests · jaconv", "Apache-2.0 · MIT"],
  ]],
];
const SHORTCUTS = [
  [["Ctrl", "O"], "添加文件"], [["Ctrl", "Shift", "O"], "添加文件夹"], [["Ctrl", "1~5"], "切换页面"],
  [["←", "→"], "上一页 / 下一页（校对、预览）"], [["空格"], "校对：按住看原图"], [["Ctrl", "S"], "校对：应用并重排"],
  [["1", "2", "3"], "校对：原图 / 去字 / 成品"], [["↑", "↓"], "校对：切换文字块"], [["B"], "校对：显示 / 隐藏文字框"],
  [["Ctrl", "滚轮"], "校对：缩放画布"], [["0"], "校对：适应窗口"], [["Esc"], "关闭弹窗 / 取消选择"],
];
const PROFILE_NAME = { fast: "极速", balanced: "均衡", quality: "高质量" };

VIEWS.about = {
  title: "关于",
  hw: null,
  mount(root) { root.innerHTML = `<div class="scroll" style="flex:1"><div class="page" id="aboutPage"></div></div>`; },
  async show() {
    this.render();
    if (!S.llm) loadLlm();
    if (!this.hw) {
      const hw = await call("hardware", {}, { quiet: true, timeout: 30000 });
      if (hw) { this.hw = hw; this.render(); }
    }
  },
  onLlm() { if (current === "about") this.render(); },
  onSettings() { if (current === "about") this.render(); },
  render() {
    const box = $("#aboutPage");
    if (!box) return;
    const a = S.app || {}, hw = this.hw;
    const skel = n => Array.from({ length: n }, (_, i) => `<span class="skel skel-line" style="width:${50 + ((i * 17) % 40)}%"></span>`).join("");
    const gpus = hw && hw.gpus && hw.gpus.length ? hw.gpus.map(g => `${esc(g.name)}${g.vram_mb ? `（${fmtGB(g.vram_mb / 1024)}）` : ""}`).join("<br>") : "未检测到独立显卡";
    const prov = hw ? (hw.providers || []).map(p => esc(String(p).replace("ExecutionProvider", ""))).join(" · ") : "";
    const kv = hw ? `<div class="kv">
        <span>显卡</span><div>${gpus}</div>
        <span>处理器</span><div>${esc(hw.cpu_name || "未知")}（${hw.cpu_threads || 1} 线程）</div>
        <span>内存</span><div>${hw.ram_gb ? fmtGB(hw.ram_gb) : "未知"}</div>
        <span>加速方式</span><div>${esc(hw.device_label || "")}<span class="muted"> · ${esc(hw.chip || "")}</span></div>
        <span>推理引擎</span><div>ONNX Runtime ${esc(hw.ort_version || "")}<div class="muted">${prov}</div></div>
        <span>发行版</span><div>${esc(hw.variant_label || a.variant_label || "")} · v${esc(hw.version || a.version || "")}</div>
        <span>Python</span><div class="mono">${esc(hw.python || "")}</div></div>`
      : `<div class="kv">${["显卡", "处理器", "内存", "加速方式", "推理引擎", "发行版"].map(k => `<span>${k}</span><div>${skel(1)}</div>`).join("")}</div>`;
    const rec = hw && hw.recommend, curProf = getPath(S.settings, "hardware.profile");
    const recs = [];
    if (rec) {
      recs.push(`<div class="rec${rec.current_ok ? "" : " warn"}">${ico(rec.current_ok ? "check-circle" : "alert")}<div>
        <b>${rec.current_ok ? `当前的「${esc(hw.variant_label || a.variant_label || "")}」适合这台电脑` : `建议改用「${esc(rec.variant_label)}」`}</b><p>${esc(rec.reason)}</p></div></div>`);
      recs.push(`<div class="rec${curProf === rec.profile ? "" : " warn"}">${ico("gauge")}<div><b>推荐画质档位：${PROFILE_NAME[rec.profile] || rec.profile}</b>
        <p>${curProf === rec.profile ? "已经在使用推荐的档位。" : `当前是「${PROFILE_NAME[curProf] || curProf}」，按这台电脑的配置可以用「${PROFILE_NAME[rec.profile]}」。`}</p>
        ${curProf !== rec.profile ? `<button class="btn btn-sm" data-act="apply-profile" data-v="${esc(rec.profile)}">改用推荐档位</button>` : ""}</div></div>`);
    }
    const lr = S.llm && S.llm.recommend;
    if (lr && lr.model) {
      const m = llmModel(lr.model.id), ok = !!(m && m.installed);
      recs.push(`<div class="rec${ok ? "" : " warn"}">${ico("translate")}<div><b>推荐本地翻译模型：${esc(lr.model.name)}（约 ${fmtGB(lr.model.size_gb)}）</b><p>${esc(lr.reason)}</p>
        ${ok ? "" : `<button class="btn btn-sm btn-primary" data-act="go" data-to="settings/translate">${ico("download")}去下载</button>`}</div></div>`);
    }
    const lic = LICENSES.map(([g, items]) => `<div class="lic-group">${g}</div>` + items.map(([use, name, l]) =>
      `<div class="lic"><div class="grow"><div class="lic-name">${esc(name)}</div><div class="lic-use">${esc(use)}</div></div><span class="lic-badge${l.startsWith("OFL") ? " ofl" : ""}">${esc(l)}</span></div>`).join("")).join("");
    const keys = SHORTCUTS.map(([ks, d]) => `<div>${ks.map(k => `<span class="kbd">${esc(k)}</span>`).join("")}</div><div>${esc(d)}</div>`).join("");
    box.innerHTML = `<div class="card about-hero"><svg class="about-logo" aria-hidden="true"><use href="#i-logo"/></svg>
        <div style="position:relative;min-width:0"><div class="about-name">一键汉化漫画</div>
          <div class="about-ver"><span class="chip mono">v${esc(a.version || "4.0.0")}</span><span class="tag done plain">${esc(a.variant_label || "")}</span>${a.demo ? `<span class="tag paused plain">演示模式</span>` : ""}</div>
          <div class="about-tag">检测 → 识别 → 翻译 → 去字 → 嵌字，一键完成。本地优先：翻译模型在你的电脑上运行，漫画不出本机。</div></div>
        <span style="flex:1"></span><div class="head-actions" style="position:relative"><button class="btn" data-act="open-logs">${ico("file-text")}日志</button><button class="btn" data-act="open-data">${ico("folder")}数据目录</button></div></div>
      <div class="about-grid">
        <div class="card"><div class="card-head"><div class="card-title">${ico("chip")}本机硬件</div></div><div class="card-body">${kv}</div></div>
        <div class="card"><div class="card-head"><div class="card-title">${ico("sparkle")}为这台电脑推荐</div></div><div class="card-body"><div class="rec-list">${recs.join("") || skel(4)}</div></div></div>
      </div>
      <div class="card" style="margin-top:16px"><div class="card-head"><div class="card-title">${ico("keyboard")}快捷键</div></div><div class="card-body"><div class="shortcuts">${keys}</div></div></div>
      <div class="card" style="margin-top:16px"><div class="card-head"><div><div class="card-title">${ico("shield")}第三方组件与授权</div>
        <div class="card-sub muted" style="margin-top:2px">只使用允许商用的开源授权（Apache-2.0 / MIT / BSD / OFL），授权全文随软件附带</div></div></div>
        <div class="card-body"><div class="lic-list">${lic}</div></div></div>`;
  },
};
ACT["apply-profile"] = async btn => {
  const s = await saveSettings({ hardware: { profile: btn.dataset.v } });
  if (s) toast(`已改用「${PROFILE_NAME[btn.dataset.v]}」档位，新开始的任务生效`, "success");
};

// ===================================================================== 启动
async function boot() {
  applyTheme(themePref(), false);
  renderBootSkeleton();
  let b = null, err = null;
  for (let i = 0; i < 3 && !b; i++) {
    try { b = await api("bootstrap", {}, { timeout: 30000 }); }
    catch (e) { err = e; if (i < 2) await sleep(700 * (i + 1)); }
  }
  const vt = $("#view-tasks");
  if (!b) {
    vt.innerHTML = `<div style="flex:1;display:grid;place-items:center">${emptyHtml("error", "界面没能连上后台程序", (err && err.message) || "请关闭软件后重新打开",
      `<button class="btn btn-primary" data-act="reload">${ico("refresh")}重新加载</button>`)}</div>`;
    return;
  }
  S.app = b.app || {};
  S.defaults = b.defaults || {};
  S.enums = b.enums || {};
  S.hardware = b.hardware || {};
  S.engines = b.engines || [];
  S.fonts = b.fonts || [];
  S.formats = b.formats || S.formats;
  S.paths = b.paths || {};
  S.engine = { running: !!(b.engine && b.engine.running), label: (b.engine && b.engine.label) || "", error: (b.engine && b.engine.error) || "" };
  applySettings(b.settings, false);
  applyTheme(themePref(), false);
  setJobs(b.jobs || []);
  renderEngineBanner();
  vt.innerHTML = "";
  vt.classList.remove("active");
  S.ready = true;
  connectEvents();
  await route();
  loadLlm();
}
boot();
})();
