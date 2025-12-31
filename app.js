// Running Form Analysis Web App (MVP)
// Browser-only: video never uploaded; analysis runs in the browser.
// Important: This is a demo MVP. Production quality needs more robust event detection (initial contact / toe-off)
// and camera calibration (pixel-to-real distance, lens distortion, camera angle compensation, 3D/multi-view, etc).

import { I18N, DEFAULT_LANG } from "./i18n.js?v=3";

const tf = window.tf;
const poseDetection = window.poseDetection;

const DEFAULT_SAMPLE_FPS = 24;
const DEFAULT_MIN_STEP_SEC = 0.3;
const DIRECTION_MODE = "auto";
const MODEL_TYPE_AUTO = "auto";
const MODEL_TYPE_LITE = "lite";
const MODEL_TYPE_FULL = "full";
const MOBILE_MODEL_TYPE = MODEL_TYPE_LITE;
const DESKTOP_MODEL_TYPE = MODEL_TYPE_LITE; // default all to lite for speed
const MOBILE_FPS_FALLBACK = 30;
const MIN_POSE_SCORE = 0.2;
const MIN_DURATION_SEC = 3;
const MIN_DETECTION_RATIO = 0.2;
const MIN_DETECTED_FRAMES = 8;
const MIN_CONTACTS = 4;
const MIN_LONG_EDGE = 1920;
const MIN_SHORT_EDGE = 1080;
const NORMAL_FPS = 30;
const SLOWMO_FPS = 240;
const NORMAL_FPS_TOLERANCE = 6;
const SLOWMO_FPS_TOLERANCE = 20;
const SLOWMO_FACTOR = 8;
const FPS_PROBE_MS_MOBILE = 350;
const FPS_PROBE_MS_DESKTOP = 600;
const FPS_PROBE_TIMEOUT_EXTRA_MS = 600;
const FPS_MIN_FRAMES_MOBILE = 4;
const FPS_MIN_FRAMES_DESKTOP = 8;
const MIN_DIRECTION_SAMPLES = 8;
const MAX_DIRECTION_FLIP_RATIO = 0.25;
const MAX_SAMPLES_MOBILE = 150;
const MAX_SAMPLES_DESKTOP = 240;
const INFER_LONG_EDGE_MOBILE = 384;
const INFER_LONG_EDGE_DESKTOP = 640;
const MODEL_URL_PATTERN = /blazepose/i;
const POSE_CONNECTIONS = (() => {
  try {
    return poseDetection.util.getAdjacentPairs(poseDetection.SupportedModels.BlazePose);
  } catch (err) {
    return [];
  }
})();

const $ = (sel) => document.querySelector(sel);

const els = {
  file: $("#videoFile"),
  video: $("#video"),
  overlay: $("#overlay"),
  btnAnalyze: $("#btnAnalyze"),
  btnDownload: $("#btnDownload"),
  langButtons: document.querySelectorAll("[data-lang]"),
  progressBar: $("#progressBar"),
  status: $("#status"),
  log: $("#log"),
  summary: $("#summary"),
  flags: $("#flags"),
  metricsTable: $("#metricsTable tbody"),
  advice: $("#advice"),
  modelType: $("#modelType"),
  modelHint: $("#modelHint"),
};

let poseDetector = null;
let overlayCtx = null;
let lastAnalysis = null;
let currentLang = DEFAULT_LANG;
let lastStatus = null;
let lastRender = null;
let modelTypeSelection = MODEL_TYPE_LITE;
let modelTypeResolved = null;
let modelBackend = null;
let inferenceCanvas = null;
let inferenceCtx = null;
let fetchPatched = false;

function getValueByPath(obj, key) {
  return key.split(".").reduce((acc, part) => (acc && acc[part] !== undefined ? acc[part] : null), obj);
}

function interpolate(str, vars = {}) {
  return str.replace(/\{(\w+)\}/g, (_, key) => (vars[key] !== undefined ? String(vars[key]) : `{${key}}`));
}

function t(key, vars = {}) {
  const current = getValueByPath(I18N[currentLang], key);
  const fallback = getValueByPath(I18N[DEFAULT_LANG], key);
  const value = current !== null ? current : fallback;
  if (typeof value !== "string") return "";
  return interpolate(value, vars);
}

function getArray(key, vars = {}) {
  const current = getValueByPath(I18N[currentLang], key);
  const fallback = getValueByPath(I18N[DEFAULT_LANG], key);
  const value = current !== null ? current : fallback;
  if (!Array.isArray(value)) return [];
  return value.map((item) => interpolate(String(item), vars));
}

function applyTranslations() {
  document.documentElement.lang = currentLang;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.dataset.i18n;
    const text = t(key);
    const attr = el.dataset.i18nAttr;
    if (attr) {
      el.setAttribute(attr, text);
    } else {
      el.textContent = text;
    }
  });

  els.langButtons.forEach((btn) => {
    const isActive = btn.dataset.lang === currentLang;
    btn.classList.toggle("active", isActive);
    btn.setAttribute("aria-pressed", isActive ? "true" : "false");
  });

  if (lastStatus) {
    setStatus(lastStatus.key, lastStatus.vars, { silent: true });
  }

  if (lastRender?.type === "analysis") {
    renderResults(lastRender.analysis);
  }
  if (lastRender?.type === "issue") {
    renderIssue(lastRender.key, lastRender.vars);
  }
  if (lastRender?.type === "analyzing") {
    els.summary.textContent = t("results.analyzing");
  }
}

function installFetchProgress() {
  if (fetchPatched || typeof fetch !== "function") return;
  fetchPatched = true;
  const originalFetch = fetch;
  globalThis.fetch = async (input, init) => {
    const url = typeof input === "string" ? input : input?.url;
    const isModel = typeof url === "string" && MODEL_URL_PATTERN.test(url);
    if (!isModel) return originalFetch(input, init);

    const name = (() => {
      try {
        const u = new URL(url);
        const parts = u.pathname.split("/").filter(Boolean);
        return parts.slice(-3).join("/") || url;
      } catch (err) {
        return url;
      }
    })();

    const res = await originalFetch(input, init);
    const lenHeader = res.headers?.get?.("content-length");
    const total = lenHeader ? parseInt(lenHeader, 10) : null;
    if (!res.body || typeof res.body.tee !== "function") {
      return res;
    }

    const [streamForUse, streamForMonitor] = res.body.tee();
    if (total && Number.isFinite(total) && total > 0) {
      const reader = streamForMonitor.getReader();
      let received = 0;
      let lastPct = 0;
      const pump = () => reader.read().then(({ done, value }) => {
        if (done) {
          log(t("log.modelDownloadProgress", { name, pct: 100 }));
          return;
        }
        received += value?.byteLength || 0;
        const pct = Math.floor((received / total) * 100);
        if (pct >= lastPct + 10) {
          lastPct = pct;
          log(t("log.modelDownloadProgress", { name, pct }));
        }
        return pump();
      }).catch(() => {});
      pump();
    }

    return new Response(streamForUse, {
      status: res.status,
      statusText: res.statusText,
      headers: res.headers,
    });
  };
}

function setLanguage(lang) {
  const next = I18N[lang] ? lang : DEFAULT_LANG;
  if (next === currentLang) return;
  currentLang = next;
  try {
    localStorage.setItem("strideiq-lang", currentLang);
  } catch (err) {
    // Ignore storage errors.
  }
  try {
    const params = new URLSearchParams(window.location.search);
    params.set("lang", currentLang);
    const nextUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, "", nextUrl);
  } catch (err) {
    // Ignore URL update errors.
  }
  applyTranslations();
}

function initLanguage() {
  const params = new URLSearchParams(window.location.search);
  const urlLang = params.get("lang");
  if (I18N[urlLang]) {
    currentLang = urlLang;
    try {
      localStorage.setItem("strideiq-lang", currentLang);
    } catch (err) {
      // Ignore storage errors.
    }
    applyTranslations();
    els.langButtons.forEach((btn) => {
      btn.addEventListener("click", () => setLanguage(btn.dataset.lang));
    });
    return;
  }

  let stored = null;
  try {
    stored = localStorage.getItem("strideiq-lang");
  } catch (err) {
    stored = null;
  }
  currentLang = I18N[stored] ? stored : DEFAULT_LANG;
  applyTranslations();
  els.langButtons.forEach((btn) => {
    btn.addEventListener("click", () => setLanguage(btn.dataset.lang));
  });
}

function isMobileDevice() {
  if (navigator.userAgentData?.mobile) return true;
  const ua = navigator.userAgent || "";
  return /Android|iPhone|iPad|iPod|IEMobile|Mobile/i.test(ua);
}

function resolveModelType(type) {
  if (type === MODEL_TYPE_AUTO) {
    return isMobileDevice() ? MOBILE_MODEL_TYPE : DESKTOP_MODEL_TYPE;
  }
  return type;
}

function getModelTypeLabel(type) {
  const label = t(`modelTypes.${type}`);
  return label || type;
}

function resetPoseDetector() {
  if (poseDetector && typeof poseDetector.dispose === "function") {
    poseDetector.dispose();
  }
  poseDetector = null;
  modelTypeResolved = null;
  modelBackend = null;
}

function setModelTypeSelection(next) {
  if (![MODEL_TYPE_AUTO, MODEL_TYPE_LITE, MODEL_TYPE_FULL].includes(next)) {
    modelTypeSelection = MODEL_TYPE_AUTO;
  } else {
    modelTypeSelection = next;
  }
  try {
    localStorage.setItem("strideiq-model-type", modelTypeSelection);
  } catch (err) {
    // Ignore storage errors.
  }
  if (els.modelType) {
    els.modelType.value = modelTypeSelection;
  }
  if (poseDetector) {
    resetPoseDetector();
  }
  updateModelHint();
}

function initModelType() {
  let stored = null;
  try {
    stored = localStorage.getItem("strideiq-model-type");
  } catch (err) {
    stored = null;
  }
  setModelTypeSelection(stored || MODEL_TYPE_LITE);
  if (els.modelType) {
    els.modelType.addEventListener("change", (event) => {
      setModelTypeSelection(event.target.value);
    });
  }
  updateModelHint();
}

function updateModelHint() {
  if (!els.modelHint) return;
  const resolved = resolveModelType(modelTypeSelection);
  const backend = modelBackend || "-";
  els.modelHint.textContent = t("input.modelHint", {
    modelType: getModelTypeLabel(resolved),
    backend,
  });
}

function log(msg) {
  const ts = new Date().toISOString().replace("T", " ").replace("Z", "");
  els.log.textContent += `[${ts}] ${msg}\n`;
  els.log.scrollTop = els.log.scrollHeight;
}

function setStatus(key, vars = {}, options = {}) {
  const msg = t(key, vars);
  els.status.textContent = msg;
  if (!options.silent) {
    log(msg);
  }
  lastStatus = { key, vars };
}

function setProgress(pct) {
  const clamped = Math.max(0, Math.min(100, pct));
  els.progressBar.style.width = `${clamped}%`;
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function yieldToUI() {
  // Give the UI a chance to update during long loops.
  await sleep(0);
}

async function ensureCanvasReady() {
  if (!overlayCtx) {
    overlayCtx = els.overlay.getContext("2d");
  }
  // Canvas pixel size should match video.videoWidth / video.videoHeight.
  const vw = els.video.videoWidth || 0;
  const vh = els.video.videoHeight || 0;
  if (vw > 0 && vh > 0) {
    els.overlay.width = vw;
    els.overlay.height = vh;
  }
}

async function loadModel() {
  const nextModelType = resolveModelType(modelTypeSelection);
  if (poseDetector && modelTypeResolved === nextModelType) {
    setStatus("status.modelAlreadyLoaded");
    return;
  }
  if (poseDetector) {
    resetPoseDetector();
  }
  setStatus("status.modelLoading", { modelType: getModelTypeLabel(nextModelType) });
  setProgress(5);
  log(t("log.modelInitProgress", { stage: "backend" }));

  try {
    const ok = await tf.setBackend("webgl");
    if (!ok) throw new Error("webgl backend unavailable");
    await tf.ready();
  } catch (err) {
    log(t("log.backendFallback", { error: String(err) }));
    await tf.setBackend("cpu");
    await tf.ready();
  }
  modelBackend = tf.getBackend();
  setProgress(10);
  log(t("log.modelInitProgress", { stage: "downloading model" }));

  installFetchProgress();

  try {
    poseDetector = await poseDetection.createDetector(poseDetection.SupportedModels.BlazePose, {
      runtime: "tfjs",
      modelType: nextModelType,
      enableSmoothing: false,
    });
    modelTypeResolved = nextModelType;
    setStatus("status.modelLoaded", {
      modelType: getModelTypeLabel(nextModelType),
      backend: modelBackend,
    });
    setProgress(15);
    log(t("log.modelInitProgress", { stage: "model ready" }));
    updateModelHint();
  } catch (err) {
    log(t("log.modelInitFailed", { error: String(err) }));
    resetPoseDetector();
    setStatus("status.modelLoadFailed", { error: String(err) });
    alert(t("alerts.modelLoadFailed"));
    throw err;
  }

  setProgress(8);
  if (els.video.videoWidth > 0) {
    await ensureCanvasReady();
  }
  maybeEnableAnalyze();
}

function maybeEnableAnalyze() {
  const hasFile = !!els.file.files?.[0];
  // Enable as soon as a file is chosen; analyze() will still guard if metadata isn't ready.
  els.btnAnalyze.disabled = !hasFile;
}

function resetOutputs() {
  els.summary.classList.remove("muted");
  els.flags.classList.remove("muted");
  els.advice.classList.remove("muted");
  els.summary.textContent = t("results.analyzing");
  els.flags.innerHTML = "";
  els.metricsTable.innerHTML = "";
  els.advice.innerHTML = "";
  els.btnDownload.disabled = true;
  lastAnalysis = null;
  lastRender = { type: "analyzing" };
}

function clearOverlay() {
  if (!overlayCtx) return;
  overlayCtx.clearRect(0, 0, els.overlay.width, els.overlay.height);
}

function drawConnectors(ctx, landmarks, connections) {
  if (!connections.length) return;
  ctx.save();
  ctx.strokeStyle = "#3ad0ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (const [i, j] of connections) {
    const a = landmarks[i];
    const b = landmarks[j];
    if (!a || !b) continue;
    if (!Number.isFinite(a.x) || !Number.isFinite(a.y) || !Number.isFinite(b.x) || !Number.isFinite(b.y)) continue;
    if (Number.isFinite(a.score) && a.score < MIN_POSE_SCORE) continue;
    if (Number.isFinite(b.score) && b.score < MIN_POSE_SCORE) continue;
    ctx.moveTo(a.x * ctx.canvas.width, a.y * ctx.canvas.height);
    ctx.lineTo(b.x * ctx.canvas.width, b.y * ctx.canvas.height);
  }
  ctx.stroke();
  ctx.restore();
}

function drawLandmarks(ctx, landmarks) {
  ctx.save();
  ctx.fillStyle = "#5be0ff";
  const radius = 2.5;
  for (const lm of landmarks) {
    if (!lm) continue;
    if (!Number.isFinite(lm.x) || !Number.isFinite(lm.y)) continue;
    if (Number.isFinite(lm.score) && lm.score < MIN_POSE_SCORE) continue;
    ctx.beginPath();
    ctx.arc(lm.x * ctx.canvas.width, lm.y * ctx.canvas.height, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function ensureInferenceCanvas(vw, vh) {
  if (!inferenceCanvas) {
    inferenceCanvas = document.createElement("canvas");
    inferenceCtx = inferenceCanvas.getContext("2d");
  }
  const longEdge = Math.max(vw, vh);
  const shortEdge = Math.min(vw, vh);
  if (longEdge === 0 || shortEdge === 0) return null;
  const targetLong = isMobileDevice() ? INFER_LONG_EDGE_MOBILE : INFER_LONG_EDGE_DESKTOP;
  const scale = Math.min(1, targetLong / longEdge);
  const outW = Math.max(1, Math.round(vw * scale));
  const outH = Math.max(1, Math.round(vh * scale));
  if (inferenceCanvas.width !== outW || inferenceCanvas.height !== outH) {
    inferenceCanvas.width = outW;
    inferenceCanvas.height = outH;
  }
  return { canvas: inferenceCanvas, ctx: inferenceCtx, width: outW, height: outH };
}

function drawOverlay(landmarks, direction) {
  if (!overlayCtx || !landmarks) return;

  const ctx = overlayCtx;
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);

  // Skeleton
  drawConnectors(ctx, landmarks, POSE_CONNECTIONS);
  drawLandmarks(ctx, landmarks);

  // Gravity / reference lines
  const hipL = landmarks[23];
  const hipR = landmarks[24];
  const hip = hipL && hipR ? midpoint(hipL, hipR) : null;
  const ankleL = landmarks[27];
  const ankleR = landmarks[28];
  let pick = null;
  if (ankleL && ankleR) {
    pick = ankleL.y > ankleR.y ? ankleL : ankleR;
  } else {
    pick = ankleL || ankleR;
  }

  if (!hip || !Number.isFinite(hip.x) || !pick || !Number.isFinite(pick.x)) {
    return;
  }

  ctx.save();
  ctx.lineWidth = 2;

  // Center-of-mass line (hip x)
  ctx.strokeStyle = "rgba(255, 255, 255, 0.60)";
  ctx.beginPath();
  ctx.moveTo(hip.x * els.overlay.width, 0);
  ctx.lineTo(hip.x * els.overlay.width, els.overlay.height);
  ctx.stroke();

  // Landing line (picked ankle x)
  ctx.strokeStyle = "rgba(255, 0, 0, 0.65)";
  ctx.beginPath();
  ctx.moveTo(pick.x * els.overlay.width, 0);
  ctx.lineTo(pick.x * els.overlay.width, els.overlay.height);
  ctx.stroke();

  // Direction marker
  const arrowY = 28;
  const arrowX = 18;
  ctx.fillStyle = "rgba(255,255,255,0.75)";
  ctx.font = "14px ui-monospace, monospace";
  ctx.fillText(direction > 0 ? ">" : "<", arrowX, arrowY);

  ctx.restore();
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
}

function dist2D(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function angleDeg(a, b, c) {
  // Angle ABC (at b) in degrees.
  const abx = a.x - b.x;
  const aby = a.y - b.y;
  const cbx = c.x - b.x;
  const cby = c.y - b.y;

  const dot = abx * cbx + aby * cby;
  const ab = Math.hypot(abx, aby);
  const cb = Math.hypot(cbx, cby);
  if (ab === 0 || cb === 0) return NaN;

  const cos = dot / (ab * cb);
  const cosClamped = Math.max(-1, Math.min(1, cos));
  const ang = Math.acos(cosClamped);
  return (ang * 180) / Math.PI;
}

function percentile(arr, p) {
  const xs = arr.filter((x) => Number.isFinite(x)).slice().sort((a, b) => a - b);
  if (xs.length === 0) return NaN;
  const idx = (xs.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return xs[lo];
  return xs[lo] + (xs[hi] - xs[lo]) * (idx - lo);
}

function median(arr) {
  return percentile(arr, 0.5);
}

function mean(arr) {
  const xs = arr.filter((x) => Number.isFinite(x));
  if (xs.length === 0) return NaN;
  return xs.reduce((s, x) => s + x, 0) / xs.length;
}

function directionSign(landmarks) {
  const nose = landmarks?.[0];
  const hip = landmarks ? midpoint(landmarks[23], landmarks[24]) : null;
  if (!nose || !hip) return 0;
  const dx = nose.x - hip.x;
  if (!Number.isFinite(dx) || Math.abs(dx) < 0.015) return 0;
  return dx > 0 ? 1 : -1;
}

function estimateDirectionAuto(landmarks) {
  // Use nose relative to hips to infer facing direction.
  // Right (larger x) means facing/running right.
  const sign = directionSign(landmarks);
  return sign || 1;
}

function directionFromUI(landmarks) {
  if (DIRECTION_MODE === "right") return 1;
  if (DIRECTION_MODE === "left") return -1;
  return estimateDirectionAuto(landmarks);
}

function findContactPeaks(yArr, timeArr, minStepSec) {
  // Use local maxima of heel_y (lowest point) as initial contact.
  // Larger y means closer to the bottom of the frame (closer to ground).
  const thr = percentile(yArr, 0.80); // top 20% as candidates
  if (!Number.isFinite(thr)) return [];

  const peaks = [];
  let lastPeakT = -Infinity;

  for (let i = 1; i < yArr.length - 1; i++) {
    const y0 = yArr[i - 1], y1 = yArr[i], y2 = yArr[i + 1];
    const t = timeArr[i];
    if (!Number.isFinite(y0) || !Number.isFinite(y1) || !Number.isFinite(y2) || !Number.isFinite(t)) continue;
    if (y1 < thr) continue; // not low enough
    if (y1 > y0 && y1 >= y2) {
      if (t - lastPeakT >= minStepSec) {
        peaks.push(i);
        lastPeakT = t;
      }
    }
  }
  return peaks;
}

function classifyFootStrike(heel, toe) {
  // Larger y is closer to ground. If heel is noticeably lower, call it heel strike.
  if (!heel || !toe) return "unknown";
  const dy = heel.y - toe.y;
  if (!Number.isFinite(dy)) return "unknown";
  if (dy > 0.012) return "heel";
  if (dy < -0.012) return "forefoot";
  return "midfoot";
}

function normalizePose(pose, vw, vh) {
  if (!pose || !Array.isArray(pose.keypoints) || pose.keypoints.length === 0) return null;
  const score = Number.isFinite(pose.score)
    ? pose.score
    : mean(pose.keypoints.map((kp) => (Number.isFinite(kp.score) ? kp.score : 0)));
  if (!Number.isFinite(score) || score < MIN_POSE_SCORE) return null;
  return pose.keypoints.map((kp) => {
    const x = Number.isFinite(kp.x) && vw > 0 ? kp.x / vw : NaN;
    const y = Number.isFinite(kp.y) && vh > 0 ? kp.y / vh : NaN;
    return {
      x: clamp01(x),
      y: clamp01(y),
      score: Number.isFinite(kp.score) ? kp.score : 0,
    };
  });
}

function interpretOverstride(ratio) {
  if (!Number.isFinite(ratio)) return { label: t("interpret.overstride.unknown"), level: "na" };
  if (ratio >= 0.20) return { label: t("interpret.overstride.severe"), level: "bad" };
  if (ratio >= 0.12) return { label: t("interpret.overstride.moderate"), level: "warn" };
  if (ratio >= 0.06) return { label: t("interpret.overstride.mild"), level: "warn" };
  return { label: t("interpret.overstride.good"), level: "good" };
}

function interpretKnee(angle) {
  if (!Number.isFinite(angle)) return { label: t("interpret.knee.unknown"), level: "na" };
  if (angle >= 170) return { label: t("interpret.knee.locked"), level: "bad" };
  if (angle >= 160) return { label: t("interpret.knee.extended"), level: "warn" };
  return { label: t("interpret.knee.good"), level: "good" };
}

function interpretTrunkLean(deg) {
  // Positive: forward (aligned with running direction), negative: backward.
  if (!Number.isFinite(deg)) return { label: t("interpret.trunk.unknown"), level: "na" };
  if (deg < -2) return { label: t("interpret.trunk.backward"), level: "bad" };
  if (deg < 0) return { label: t("interpret.trunk.slightBackward"), level: "warn" };
  if (deg <= 12) return { label: t("interpret.trunk.reasonable"), level: "good" };
  return { label: t("interpret.trunk.largeForward"), level: "warn" };
}

function interpretRetraction(speed) {
  // Positive: pulling back before contact. Negative: still reaching forward.
  if (!Number.isFinite(speed)) return { label: t("interpret.retraction.unknown"), level: "na" };
  if (speed < -0.02) return { label: t("interpret.retraction.forwardReach"), level: "bad" };
  if (speed < 0.02) return { label: t("interpret.retraction.slowPull"), level: "warn" };
  return { label: t("interpret.retraction.good"), level: "good" };
}

function formatNum(x, digits = 2) {
  if (!Number.isFinite(x)) return "--";
  return x.toFixed(digits);
}

function computeContactMetrics(frames, contacts, leg, direction, timeScale = 1) {
  // leg: "L" or "R"
  const idxHipL = 23, idxHipR = 24;
  const idxShoulderL = 11, idxShoulderR = 12;
  const idxKnee = leg === "L" ? 25 : 26;
  const idxAnkle = leg === "L" ? 27 : 28;
  const idxHeel = leg === "L" ? 29 : 30;
  const idxToe = leg === "L" ? 31 : 32;

  const out = [];

  for (const i of contacts) {
    const f = frames[i];
    if (!f?.landmarks) continue;
    const L = f.landmarks;

    const hip = midpoint(L[idxHipL], L[idxHipR]);
    const shoulder = midpoint(L[idxShoulderL], L[idxShoulderR]);

    const knee = L[idxKnee];
    const ankle = L[idxAnkle];
    const heel = L[idxHeel];
    const toe = L[idxToe];

    // Leg length (approx).
    const legLen = dist2D(hip, knee) + dist2D(knee, ankle);

    // Overstride: ankle ahead of hip along running direction.
    const overstride = (ankle.x - hip.x) * direction;
    const overstrideRatio = legLen > 0 ? overstride / legLen : NaN;

    // Knee angle at contact.
    const kneeAngle = angleDeg(hip, knee, ankle);

    // Trunk lean relative to vertical, sign aligned to running direction.
    const dx = (shoulder.x - hip.x) * direction;
    const dy = hip.y - shoulder.y; // positive if shoulder above hip
    const trunkLeanDeg = dy !== 0 ? (Math.atan(dx / dy) * 180) / Math.PI : NaN;

    const strike = classifyFootStrike(heel, toe);

    // Retraction speed: slope of (ankle - hip) along direction in last 0.12s.
    const windowSec = 0.12 * timeScale;
    const w = f.sampleFps ? Math.max(2, Math.floor(f.sampleFps * windowSec)) : 2;
    const j0 = Math.max(0, i - w);
    const f0 = frames[j0];
    let retractSpeed = NaN;
    if (f0?.landmarks) {
      const L0 = f0.landmarks;
      const hip0 = midpoint(L0[idxHipL], L0[idxHipR]);
      const ankle0 = L0[idxAnkle];
      const rel0 = (ankle0.x - hip0.x) * direction;
      const rel1 = (ankle.x - hip.x) * direction;
      const dt = (f.t - f0.t) / timeScale;
      if (Number.isFinite(dt) && dt > 0) {
        // rel decreasing => pulling back => positive retractionScore
        retractSpeed = -(rel1 - rel0) / dt;
      }
    }

    out.push({
      t: f.t,
      leg,
      overstrideRatio,
      kneeAngle,
      trunkLeanDeg,
      strike,
      retractSpeed,
      legLen,
      overstride,
    });
  }

  return out;
}

function buildFlags(summary) {
  if (!summary.contactCount) {
    return [{
      level: "warn",
      text: t("flags.noContacts"),
    }];
  }

  const flags = [];

  if (Number.isFinite(summary.overstrideRatioMedian)) {
    if (summary.overstrideRatioMedian >= 0.12) {
      flags.push({ level: "bad", text: t("flags.overstrideBad") });
    } else {
      flags.push({ level: "good", text: t("flags.overstrideGood") });
    }
  } else {
    flags.push({ level: "warn", text: t("flags.overstrideUnknown") });
  }

  if (Number.isFinite(summary.kneeAngleMedian)) {
    if (summary.kneeAngleMedian >= 165) {
      flags.push({ level: "bad", text: t("flags.kneeBad") });
    } else {
      flags.push({ level: "good", text: t("flags.kneeGood") });
    }
  } else {
    flags.push({ level: "warn", text: t("flags.kneeUnknown") });
  }

  if (Number.isFinite(summary.trunkLeanMedian)) {
    if (summary.trunkLeanMedian < -1) {
      flags.push({ level: "bad", text: t("flags.trunkBad") });
    } else {
      flags.push({ level: "good", text: t("flags.trunkGood") });
    }
  } else {
    flags.push({ level: "warn", text: t("flags.trunkUnknown") });
  }

  if (Number.isFinite(summary.heelStrikeRate)) {
    if (summary.heelStrikeRate >= 0.6) {
      flags.push({ level: "warn", text: t("flags.heelMostly") });
    } else if (summary.heelStrikeRate <= 0.2) {
      flags.push({ level: "good", text: t("flags.heelMostlyNon") });
    } else {
      flags.push({ level: "warn", text: t("flags.heelMixed") });
    }
  } else {
    flags.push({ level: "warn", text: t("flags.heelUnknown") });
  }

  if (Number.isFinite(summary.retractSpeedMedian)) {
    if (summary.retractSpeedMedian < 0.02) {
      flags.push({ level: "warn", text: t("flags.retractionSlow") });
    } else {
      flags.push({ level: "good", text: t("flags.retractionGood") });
    }
  } else {
    flags.push({ level: "warn", text: t("flags.retractionUnknown") });
  }

  return flags;
}

function buildAdvice(summary) {
  if (!summary.contactCount) {
    return [
      t("advice.noContacts"),
    ];
  }

  const out = [];

  if (Number.isFinite(summary.overstrideRatioMedian)) {
    if (summary.overstrideRatioMedian >= 0.12) {
      out.push(t("advice.overstridePriority1"));
      out.push(t("advice.overstridePriority2"));
    } else {
      out.push(t("advice.overstrideGood"));
    }
  } else {
    out.push(t("advice.overstrideUnknown"));
  }

  if (Number.isFinite(summary.kneeAngleMedian) && summary.kneeAngleMedian >= 165) {
    out.push(t("advice.knee"));
  }

  if (Number.isFinite(summary.trunkLeanMedian) && summary.trunkLeanMedian < -1) {
    out.push(t("advice.trunk"));
  }

  if (Number.isFinite(summary.heelStrikeRate) && Number.isFinite(summary.overstrideRatioMedian)) {
    if (summary.heelStrikeRate >= 0.6 && summary.overstrideRatioMedian < 0.12) {
      out.push(t("advice.heelOk"));
    } else if (summary.heelStrikeRate >= 0.6 && summary.overstrideRatioMedian >= 0.12) {
      out.push(t("advice.heelOverstride"));
    }
  }

  out.push(t("advice.strength"));

  return out;
}

function renderIssuesContent(title, issues, suggestions) {
  els.summary.textContent = title;

  els.flags.innerHTML = "";
  for (const issue of issues) {
    const li = document.createElement("li");
    li.textContent = issue;
    li.classList.add("bad");
    els.flags.appendChild(li);
  }

  els.advice.innerHTML = "";
  for (const suggestion of suggestions) {
    const li = document.createElement("li");
    li.textContent = suggestion;
    els.advice.appendChild(li);
  }

  els.metricsTable.innerHTML = `<tr><td colspan="3" class="muted">${escapeHtml(t("results.notAvailable"))}</td></tr>`;
  els.btnDownload.disabled = true;
}

function renderIssue(key, vars = {}) {
  const title = t(`issues.${key}.title`, vars);
  const issues = getArray(`issues.${key}.issues`, vars);
  const suggestions = getArray(`issues.${key}.suggestions`, vars);
  renderIssuesContent(title, issues, suggestions);
  lastRender = { type: "issue", key, vars };
}

function renderResults(analysis) {
  const s = analysis.summary;
  analysis.meta.notes = getArray("meta.notes");

  const durationLabel = analysis.meta.slowMoFactor > 1
    ? t("results.durationSlowmo", {
      duration: formatNum(analysis.meta.durationSec, 1),
      factor: analysis.meta.slowMoFactor,
      realDuration: formatNum(analysis.meta.realDurationSec, 1),
    })
    : t("results.durationNormal", {
      duration: formatNum(analysis.meta.durationSec, 1),
    });

  const directionLabel = analysis.meta.direction > 0
    ? t("results.directionRight")
    : t("results.directionLeft");
  const directionModeLabel = analysis.meta.directionMode === "auto"
    ? t("results.directionModeAuto")
    : analysis.meta.directionMode;

  els.summary.textContent =
    t("results.summaryLine1", {
      sampleFps: analysis.meta.sampleFps,
      durationLabel,
    }) +
    "\n" +
    t("results.summaryLine2", {
      left: analysis.contacts.left.length,
      right: analysis.contacts.right.length,
      total: analysis.contacts.all.length,
    }) +
    "\n" +
    t("results.summaryLine3", {
      directionLabel,
      directionMode: directionModeLabel,
    }) +
    "\n\n" +
    t("results.summaryNote");

  // Flags
  const flags = buildFlags(s);
  els.flags.innerHTML = "";
  for (const f of flags) {
    const li = document.createElement("li");
    li.textContent = f.text;
    if (f.level === "bad") li.classList.add("bad");
    if (f.level === "good") li.classList.add("good");
    els.flags.appendChild(li);
  }

  // Metrics table
  const rows = [
    {
      name: t("metrics.overstride"),
      val: formatNum(s.overstrideRatioMedian, 3),
      interp: interpretOverstride(s.overstrideRatioMedian).label,
    },
    {
      name: t("metrics.knee"),
      val: formatNum(s.kneeAngleMedian, 1),
      interp: interpretKnee(s.kneeAngleMedian).label,
    },
    {
      name: t("metrics.trunk"),
      val: formatNum(s.trunkLeanMedian, 1),
      interp: interpretTrunkLean(s.trunkLeanMedian).label,
    },
    {
      name: t("metrics.heelStrike"),
      val: `${formatNum(s.heelStrikeRate * 100, 1)}%`,
      interp: s.heelStrikeRate >= 0.6
        ? t("metrics.mostlyHeel")
        : (s.heelStrikeRate <= 0.2 ? t("metrics.mostlyNonHeel") : t("metrics.mixedStrike")),
    },
    {
      name: t("metrics.retraction"),
      val: formatNum(s.retractSpeedMedian, 3),
      interp: interpretRetraction(s.retractSpeedMedian).label,
    },
  ];

  els.metricsTable.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${escapeHtml(r.name)}</td><td class="mono">${escapeHtml(r.val)}</td><td>${escapeHtml(r.interp)}</td>`;
    els.metricsTable.appendChild(tr);
  }

  // Advice
  const adv = buildAdvice(s);
  els.advice.innerHTML = "";
  for (const a of adv) {
    const li = document.createElement("li");
    li.textContent = a;
    els.advice.appendChild(li);
  }

  els.btnDownload.disabled = false;
  lastRender = { type: "analysis", analysis };
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function estimateVideoFps(video) {
  const supportsRvfc = typeof video.requestVideoFrameCallback === "function";
  const supportsQuality = typeof video.getVideoPlaybackQuality === "function";
  log(`Debug: FPS probe start (rvfc=${supportsRvfc}, quality=${supportsQuality}, readyState=${video.readyState}, paused=${video.paused})`);
  if (!supportsRvfc && !supportsQuality) {
    if (isMobileDevice()) {
      log(`Debug: FPS probe unsupported on this browser; using mobile fallback ${MOBILE_FPS_FALLBACK}fps`);
      return MOBILE_FPS_FALLBACK;
    }
    return NaN;
  }

  const mobile = isMobileDevice();
  const sampleMs = mobile ? FPS_PROBE_MS_MOBILE : FPS_PROBE_MS_DESKTOP;
  const minFrames = mobile ? FPS_MIN_FRAMES_MOBILE : FPS_MIN_FRAMES_DESKTOP;
  const timeoutMs = sampleMs + FPS_PROBE_TIMEOUT_EXTRA_MS;
  const originalTime = video.currentTime;
  const wasPaused = video.paused;
  const wasMuted = video.muted;
  const wasPlaybackRate = video.playbackRate;

  video.muted = true;
  video.playbackRate = 1;

  let frameCount = 0;
  let start = null;
  const startWall = performance.now();
  let rafId = null;
  const qualityStart = supportsQuality ? video.getVideoPlaybackQuality().totalVideoFrames : 0;
  let qualityEnd = qualityStart;
  let cleaned = false;

  const onFrame = (now) => {
    if (!start) start = now;
    frameCount += 1;
    if (now - start < sampleMs) {
      rafId = video.requestVideoFrameCallback(onFrame);
    }
  };

  if (supportsRvfc) {
    rafId = video.requestVideoFrameCallback(onFrame);
  }

  const playStart = performance.now();
  try {
    await video.play();
  } catch (err) {
    log(`Debug: video.play() failed during FPS probe: ${String(err)}`);
    // Autoplay can still fail; fall through and return NaN.
  }
  log(`Debug: video.play() resolved in ${Math.round(performance.now() - playStart)}ms (paused=${video.paused}, readyState=${video.readyState})`);

  const cleanup = () => {
    if (cleaned) return;
    cleaned = true;
    if (supportsRvfc && typeof video.cancelVideoFrameCallback === "function" && rafId !== null) {
      video.cancelVideoFrameCallback(rafId);
    }
    video.pause();
    video.currentTime = originalTime;
    video.muted = wasMuted;
    video.playbackRate = wasPlaybackRate;
    if (!wasPaused) {
      try {
        video.play();
      } catch (err) {
        // Ignore resume errors.
      }
    }
  };

  const probe = async () => {
    try {
      await sleep(sampleMs + 150);
      const elapsedMs = start ? performance.now() - start : (performance.now() - startWall);
      let fps = NaN;
      let method = "none";
      if (elapsedMs > 0) {
        if (supportsRvfc && frameCount >= minFrames) {
          fps = frameCount / (elapsedMs / 1000);
          method = "rvfc";
        } else if (supportsQuality) {
          qualityEnd = video.getVideoPlaybackQuality().totalVideoFrames;
          const diff = qualityEnd - qualityStart;
          if (diff >= minFrames) {
            fps = diff / (elapsedMs / 1000);
            method = "playbackQuality";
          }
        }
      }

      log(
        `Debug: FPS probe done method=${method}, frames=${frameCount}, qualityDelta=${qualityEnd - qualityStart}, elapsed=${Math.round(elapsedMs)}ms, fps=${formatNum(fps, 2)}`
      );
      return { fps, method, elapsedMs };
    } catch (err) {
      log(`Debug: FPS probe error: ${String(err)}`);
      return { fps: mobile ? MOBILE_FPS_FALLBACK : NaN, method: "error", elapsedMs: performance.now() - startWall };
    } finally {
      cleanup();
    }
  };

  const timeout = new Promise((resolve) => {
    setTimeout(() => {
      log(`Debug: FPS probe timeout after ${timeoutMs}ms; using fallback ${MOBILE_FPS_FALLBACK}fps`);
      cleanup();
      resolve({ fps: MOBILE_FPS_FALLBACK, method: "timeout", elapsedMs: timeoutMs });
    }, timeoutMs);
  });

  const result = await Promise.race([probe(), timeout]);

  if (!Number.isFinite(result.fps) && mobile) {
    log(`Debug: FPS probe returned NaN; using mobile fallback ${MOBILE_FPS_FALLBACK}fps`);
    return MOBILE_FPS_FALLBACK;
  }

  return result.fps;
}

function downloadJson(obj, filename) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function seekTo(video, tSec) {
  return new Promise((resolve, reject) => {
    const onSeeked = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("video seek error"));
    };
    const cleanup = () => {
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
    };
    video.addEventListener("seeked", onSeeked, { once: true });
    video.addEventListener("error", onError, { once: true });
    video.currentTime = tSec;
  });
}

async function analyze() {
  if (!els.file.files?.[0]) {
    alert(t("alerts.chooseVideo"));
    return;
  }
  if (!els.video.duration || !Number.isFinite(els.video.duration)) {
    alert(t("alerts.videoNotLoaded"));
    return;
  }

  resetOutputs();
  setProgress(0);
  setStatus("status.checkingRequirements");

  const duration = els.video.duration;
  const vw = els.video.videoWidth || 0;
  const vh = els.video.videoHeight || 0;
  const longEdge = Math.max(vw, vh);
  const shortEdge = Math.min(vw, vh);

  if (longEdge < MIN_LONG_EDGE || shortEdge < MIN_SHORT_EDGE) {
    renderIssue("resolutionLow", {
      minLong: MIN_LONG_EDGE,
      minShort: MIN_SHORT_EDGE,
      width: vw,
      height: vh,
    });
    setStatus("status.stoppedResolutionLow");
    setProgress(0);
    return;
  }

  // Reset to the start so FPS sampling always has playable frames (prevents end-of-video FPS=null on repeat runs).
  await seekTo(els.video, 0);

  setStatus("status.checkingFps");
  const fpsStart = performance.now();
  let inputFps = await estimateVideoFps(els.video);
  if (!Number.isFinite(inputFps) && isMobileDevice()) {
    inputFps = MOBILE_FPS_FALLBACK;
    log(`Debug: Using mobile FPS fallback ${inputFps}fps after probe failed`);
  }
  log(`Debug: estimateVideoFps total time=${Math.round(performance.now() - fpsStart)}ms, result=${formatNum(inputFps, 2)}`);
  if (!Number.isFinite(inputFps)) {
    renderIssue("fpsUnknown");
    setStatus("status.stoppedFpsUnknown");
    setProgress(0);
    return;
  }

  const isNormalFps = Math.abs(inputFps - NORMAL_FPS) <= NORMAL_FPS_TOLERANCE;
  const isSlowMoFps = Math.abs(inputFps - SLOWMO_FPS) <= SLOWMO_FPS_TOLERANCE;
  if (!isNormalFps && !isSlowMoFps) {
    renderIssue("fpsUnsupported", {
      fps: formatNum(inputFps, 1),
    });
    setStatus("status.stoppedFpsUnsupported");
    setProgress(0);
    return;
  }

  const slowMoFactor = isSlowMoFps ? SLOWMO_FACTOR : 1;
  const realDuration = duration / slowMoFactor;
  if (realDuration < MIN_DURATION_SEC) {
    renderIssue("videoTooShort", {
      min: MIN_DURATION_SEC.toFixed(1),
      detected: formatNum(realDuration, 2),
      video: formatNum(duration, 2),
    });
    setStatus("status.stoppedVideoTooShort");
    setProgress(0);
    return;
  }

  if (!poseDetector) {
    try {
      await loadModel();
    } catch (err) {
      return;
    }
  }

  const sampleFps = DEFAULT_SAMPLE_FPS;
  const minStepSec = DEFAULT_MIN_STEP_SEC * slowMoFactor;
  const dtNominal = 1 / Math.max(5, Math.min(60, sampleFps));
  const maxSamples = isMobileDevice() ? MAX_SAMPLES_MOBILE : MAX_SAMPLES_DESKTOP;
  let steps = Math.max(1, Math.ceil(duration / dtNominal));
  if (steps > maxSamples) {
    steps = maxSamples;
  }
  const dt = duration / Math.max(1, steps);

  setStatus("status.analysisStarted", {
    duration: duration.toFixed(2),
    sampleFps,
    steps,
  });
  await ensureCanvasReady();
  const infer = ensureInferenceCanvas(vw, vh);
  if (infer) {
    log(t("log.analysisConfig", {
      steps,
      sampleFps,
      width: infer.width,
      height: infer.height,
      modelType: getModelTypeLabel(modelTypeResolved || resolveModelType(modelTypeSelection)),
      backend: modelBackend || "-",
    }));
  }

  // Frame records
  const frames = [];
  let direction = 1;
  let directionLocked = false;
  let detectedFrames = 0;
  let directionLeft = 0;
  let directionRight = 0;

  // Sample frames using seek + estimatePoses.
  // Long videos may take time; consider trimming for faster analysis.
  for (let k = 0; k <= steps; k++) {
    const timeSec = Math.min(duration, k * dt);

    // Seek
    await seekTo(els.video, timeSec);

    // Detect
    let poses = null;
    let landmarks = null;
    const infer = ensureInferenceCanvas(vw, vh);
    try {
      if (infer) {
        infer.ctx.drawImage(els.video, 0, 0, infer.width, infer.height);
        poses = await poseDetector.estimatePoses(infer.canvas, {
          maxPoses: 1,
          flipHorizontal: false,
        });
        const pose = poses?.[0] ?? null;
        landmarks = normalizePose(pose, infer.width, infer.height);
      }
    } catch (err) {
      log(t("log.detectError", { time: timeSec.toFixed(2), error: String(err) }));
      poses = null;
      landmarks = null;
    }
    if (landmarks) {
      detectedFrames += 1;
      const sign = directionSign(landmarks);
      if (sign === 1) directionRight += 1;
      if (sign === -1) directionLeft += 1;
    }

    if (landmarks && !directionLocked) {
      direction = directionFromUI(landmarks);
      directionLocked = true;
    }

    frames.push({
      t: timeSec,
      landmarks,
      sampleFps,
    });

    if (landmarks) {
      drawOverlay(landmarks, direction);
    } else {
      clearOverlay();
    }

    if (k % 8 === 0) {
      setProgress(15 + (k / Math.max(1, steps)) * 70); // 15-85 for analysis loop
      await yieldToUI();
    }
    if (steps > 0 && k > 0 && k % Math.max(1, Math.floor(steps / 10)) === 0) {
      const pct = Math.round((k / steps) * 100);
      log(t("log.analysisProgress", { pct }));
    }
  }

  setProgress(90);
  setStatus("status.computingSummary");

  const detectionRatio = frames.length ? detectedFrames / frames.length : 0;
  if (detectedFrames === 0) {
    renderIssue("noRunner");
    setStatus("status.stoppedNoRunner");
    setProgress(100);
    return;
  }

  if (detectedFrames < MIN_DETECTED_FRAMES || detectionRatio < MIN_DETECTION_RATIO) {
    renderIssue("lowDetection");
    setStatus("status.stoppedLowDetection");
    setProgress(100);
    return;
  }

  const directionSamples = directionLeft + directionRight;
  if (directionSamples < MIN_DIRECTION_SAMPLES) {
    renderIssue("directionUnclear");
    setStatus("status.stoppedDirectionUnclear");
    setProgress(100);
    return;
  }

  if (directionLeft > 0 && directionRight > 0) {
    const flipRatio = Math.min(directionLeft, directionRight) / directionSamples;
    if (flipRatio > MAX_DIRECTION_FLIP_RATIO) {
      renderIssue("directionInconsistent");
      setStatus("status.stoppedDirectionInconsistent");
      setProgress(100);
      return;
    }
  }

  if (directionSamples > 0) {
    direction = directionRight >= directionLeft ? 1 : -1;
  }

  // Build y arrays for heel peaks
  const leftHeelY = frames.map((f) => f.landmarks?.[29]?.y ?? NaN);
  const rightHeelY = frames.map((f) => f.landmarks?.[30]?.y ?? NaN);
  const timeArr = frames.map((f) => f.t);

  const leftPeaks = findContactPeaks(leftHeelY, timeArr, minStepSec);
  const rightPeaks = findContactPeaks(rightHeelY, timeArr, minStepSec);

  // Compute metrics at peaks
  const leftMetrics = computeContactMetrics(frames, leftPeaks, "L", direction, slowMoFactor);
  const rightMetrics = computeContactMetrics(frames, rightPeaks, "R", direction, slowMoFactor);
  const allMetrics = [...leftMetrics, ...rightMetrics].sort((a, b) => a.t - b.t);
  if (allMetrics.length < MIN_CONTACTS) {
    renderIssue("fewStrides", {
      left: leftMetrics.length,
      right: rightMetrics.length,
    });
    setStatus("status.stoppedFewStrides");
    setProgress(100);
    return;
  }

  const overstrideRatios = allMetrics.map((m) => m.overstrideRatio);
  const kneeAngles = allMetrics.map((m) => m.kneeAngle);
  const trunkLeans = allMetrics.map((m) => m.trunkLeanDeg);
  const retractSpeeds = allMetrics.map((m) => m.retractSpeed);

  const heelStrikeRate = (() => {
    const xs = allMetrics.map((m) => m.strike).filter((x) => x !== "unknown");
    if (xs.length === 0) return NaN;
    const heel = xs.filter((x) => x === "heel").length;
    return heel / xs.length;
  })();

  const analysis = {
    meta: {
      createdAt: new Date().toISOString(),
      durationSec: duration,
      realDurationSec: realDuration,
      inputFpsEstimate: inputFps,
      slowMoFactor,
      sampleFps,
      minStepSec,
      direction,
      directionMode: DIRECTION_MODE,
      model: {
        name: "TensorFlow.js BlazePose",
        runtime: "tfjs",
        modelType: modelTypeResolved,
        backend: modelBackend,
      },
      notes: getArray("meta.notes"),
    },
    contacts: {
      left: leftMetrics,
      right: rightMetrics,
      all: allMetrics,
    },
    summary: {
      overstrideRatioMedian: median(overstrideRatios),
      kneeAngleMedian: median(kneeAngles),
      trunkLeanMedian: median(trunkLeans),
      heelStrikeRate,
      retractSpeedMedian: median(retractSpeeds),
      contactCount: allMetrics.length,
    },
  };

  lastAnalysis = analysis;
  renderResults(analysis);

  setProgress(100);
  setStatus("status.analysisDone");
}

els.file.addEventListener("change", () => {
  const file = els.file.files?.[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  els.video.src = url;
  els.video.load();

  setStatus("status.videoSelected", {
    name: file.name,
    size: Math.round(file.size / 1024 / 1024),
  });
  setProgress(0);
  maybeEnableAnalyze();

  els.video.onloadedmetadata = async () => {
    setStatus("status.videoLoaded", {
      width: els.video.videoWidth,
      height: els.video.videoHeight,
      duration: els.video.duration.toFixed(2),
    });
    await ensureCanvasReady();
    maybeEnableAnalyze();
  };
  els.video.onerror = () => {
    els.btnAnalyze.disabled = true;
  };
});

els.btnAnalyze.addEventListener("click", async () => {
  els.btnAnalyze.disabled = true;
  try {
    await analyze();
  } catch (e) {
    console.error(e);
    setStatus("status.analysisFailed", { error: String(e) });
    alert(t("alerts.analysisFailed", { error: String(e) }));
  } finally {
    els.btnAnalyze.disabled = false;
  }
});

els.btnDownload.addEventListener("click", () => {
  if (!lastAnalysis) return;
  downloadJson(lastAnalysis, "run-gait-analysis.json");
});

// Initial UI state
initLanguage();
initModelType();
setStatus("status.waitingVideo");
setProgress(0);
maybeEnableAnalyze();
