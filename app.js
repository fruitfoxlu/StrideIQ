// 跑姿分析 Web App（MVP）
// 純前端：影片不需要上傳到伺服器；分析在瀏覽器端完成。
// 重要：這是示範用 MVP。實務上若要接近商用品質，需要更完整的事件偵測（initial contact / toe-off）
// 與更嚴謹的相機校正（像素到實際距離、鏡頭畸變、拍攝角度補償、3D/多視角等）。

import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm";

const $ = (sel) => document.querySelector(sel);

const els = {
  file: $("#videoFile"),
  video: $("#video"),
  overlay: $("#overlay"),
  btnLoadModel: $("#btnLoadModel"),
  btnAnalyze: $("#btnAnalyze"),
  btnDownload: $("#btnDownload"),
  sampleFps: $("#sampleFps"),
  minStepSec: $("#minStepSec"),
  directionMode: $("#directionMode"),
  progressBar: $("#progressBar"),
  status: $("#status"),
  log: $("#log"),
  summary: $("#summary"),
  flags: $("#flags"),
  metricsTable: $("#metricsTable tbody"),
  advice: $("#advice"),
};

let poseLandmarker = null;
let drawingUtils = null;
let overlayCtx = null;
let lastAnalysis = null;

function log(msg) {
  const t = new Date().toISOString().replace("T", " ").replace("Z", "");
  els.log.textContent += `[${t}] ${msg}\n`;
  els.log.scrollTop = els.log.scrollHeight;
}

function setStatus(msg) {
  els.status.textContent = `狀態：${msg}`;
  log(msg);
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
  // 讓 UI 有機會更新（避免長時間同步阻塞造成卡死的錯覺）
  await sleep(0);
}

async function ensureCanvasReady() {
  if (!overlayCtx) {
    overlayCtx = els.overlay.getContext("2d");
  }
  // Canvas 實際像素尺寸要跟 video.videoWidth / video.videoHeight 一致
  const vw = els.video.videoWidth || 0;
  const vh = els.video.videoHeight || 0;
  if (vw > 0 && vh > 0) {
    els.overlay.width = vw;
    els.overlay.height = vh;
    drawingUtils = new DrawingUtils(overlayCtx);
  }
}

async function loadModel() {
  if (poseLandmarker) {
    setStatus("模型已載入（略過）");
    return;
  }
  setStatus("載入模型中（WASM + 模型檔）...");
  setProgress(3);

  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

  // 先嘗試 GPU，失敗再 fallback CPU（某些環境 GPU delegate 可能不可用）
  try {
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    setStatus("模型已載入（GPU delegate）");
  } catch (err) {
    log(`GPU delegate 初始化失敗，改用 CPU。原因：${err}`);
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    setStatus("模型已載入（CPU delegate）");
  }

  setProgress(8);
  if (els.video.videoWidth > 0) {
    await ensureCanvasReady();
  }
  maybeEnableAnalyze();
}

function maybeEnableAnalyze() {
  const hasFile = !!els.file.files?.[0];
  const hasVideo = els.video.src && els.video.readyState >= 1;
  const hasModel = !!poseLandmarker;
  els.btnAnalyze.disabled = !(hasFile && hasVideo && hasModel);
}

function resetOutputs() {
  els.summary.classList.remove("muted");
  els.flags.classList.remove("muted");
  els.advice.classList.remove("muted");
  els.summary.textContent = "分析中...";
  els.flags.innerHTML = "";
  els.metricsTable.innerHTML = "";
  els.advice.innerHTML = "";
  els.btnDownload.disabled = true;
  lastAnalysis = null;
}

function clearOverlay() {
  if (!overlayCtx) return;
  overlayCtx.clearRect(0, 0, els.overlay.width, els.overlay.height);
}

function drawOverlay(landmarks, direction) {
  if (!overlayCtx || !drawingUtils || !landmarks) return;

  const ctx = overlayCtx;
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);

  // skeleton
  drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    lineWidth: 3,
  });
  drawingUtils.drawLandmarks(landmarks, {
    radius: (d) => 2 + 2 * clamp01(d.from?.z ?? 0),
  });

  // gravity / reference lines
  const hip = midpoint(landmarks[23], landmarks[24]);
  const ankleL = landmarks[27];
  const ankleR = landmarks[28];

  // 選較低（更接近地面）的腳來畫「落地線」作為視覺參考
  const pick = (ankleL?.y ?? 0) > (ankleR?.y ?? 0) ? ankleL : ankleR;

  ctx.save();
  ctx.lineWidth = 2;

  // 重心線（髖部 x）
  ctx.strokeStyle = "rgba(255, 255, 255, 0.60)";
  ctx.beginPath();
  ctx.moveTo(hip.x * els.overlay.width, 0);
  ctx.lineTo(hip.x * els.overlay.width, els.overlay.height);
  ctx.stroke();

  // 落地點線（選定腳的 x）
  ctx.strokeStyle = "rgba(255, 0, 0, 0.65)";
  ctx.beginPath();
  ctx.moveTo(pick.x * els.overlay.width, 0);
  ctx.lineTo(pick.x * els.overlay.width, els.overlay.height);
  ctx.stroke();

  // 方向標記
  const arrowY = 28;
  const arrowX = 18;
  ctx.fillStyle = "rgba(255,255,255,0.75)";
  ctx.font = "14px ui-monospace, monospace";
  ctx.fillText(direction > 0 ? "→" : "←", arrowX, arrowY);

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
  // angle ABC (at b) in degrees
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

function estimateDirectionAuto(landmarks) {
  // 方向：以 nose 相對於 hips 的左右判斷。
  // 右側（x較大）視為「面向/跑向右」。
  const nose = landmarks?.[0];
  const hip = landmarks ? midpoint(landmarks[23], landmarks[24]) : null;
  if (!nose || !hip) return 1;
  const dx = nose.x - hip.x;
  if (!Number.isFinite(dx) || Math.abs(dx) < 0.015) return 1;
  return dx > 0 ? 1 : -1;
}

function directionFromUI(landmarks) {
  const mode = els.directionMode.value;
  if (mode === "right") return 1;
  if (mode === "left") return -1;
  return estimateDirectionAuto(landmarks);
}

function findContactPeaks(yArr, timeArr, minStepSec) {
  // 以 heel_y 的局部最大值（最低點）近似 initial contact。
  // y 座標越大代表越靠近畫面下方（越接近地面）。
  const thr = percentile(yArr, 0.80); // 取 top 20% 作為候選
  if (!Number.isFinite(thr)) return [];

  const peaks = [];
  let lastPeakT = -Infinity;

  for (let i = 1; i < yArr.length - 1; i++) {
    const y0 = yArr[i - 1], y1 = yArr[i], y2 = yArr[i + 1];
    const t = timeArr[i];
    if (!Number.isFinite(y0) || !Number.isFinite(y1) || !Number.isFinite(y2) || !Number.isFinite(t)) continue;
    if (y1 < thr) continue; // 不夠低（不夠接近地面）
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
  // y 越大越靠近地面。若 heel 明顯更低 -> heel strike
  if (!heel || !toe) return "unknown";
  const dy = heel.y - toe.y;
  if (!Number.isFinite(dy)) return "unknown";
  if (dy > 0.012) return "heel";
  if (dy < -0.012) return "forefoot";
  return "midfoot";
}

function interpretOverstride(ratio) {
  if (!Number.isFinite(ratio)) return { label: "未知", level: "na" };
  if (ratio >= 0.20) return { label: "顯著（高機率 overstride）", level: "bad" };
  if (ratio >= 0.12) return { label: "中等（可能 overstride）", level: "warn" };
  if (ratio >= 0.06) return { label: "輕微（可再優化）", level: "warn" };
  return { label: "良好（落點接近髖下）", level: "good" };
}

function interpretKnee(angle) {
  if (!Number.isFinite(angle)) return { label: "未知", level: "na" };
  if (angle >= 170) return { label: "非常接近打直", level: "bad" };
  if (angle >= 160) return { label: "偏伸直", level: "warn" };
  return { label: "有彈性（較佳）", level: "good" };
}

function interpretTrunkLean(deg) {
  // 正值：向前（沿跑步方向），負值：向後
  if (!Number.isFinite(deg)) return { label: "未知", level: "na" };
  if (deg < -2) return { label: "後仰（可能影響推進效率）", level: "bad" };
  if (deg < 0) return { label: "略後仰", level: "warn" };
  if (deg <= 12) return { label: "合理前傾範圍", level: "good" };
  return { label: "前傾較大（視配速與個體差異）", level: "warn" };
}

function interpretRetraction(speed) {
  // 正值：落地前腳在「往後拉」；負值：還在往前伸（常見於 overstride）
  if (!Number.isFinite(speed)) return { label: "未知", level: "na" };
  if (speed < -0.02) return { label: "明顯往前伸（回收/回拉不足）", level: "bad" };
  if (speed < 0.02) return { label: "回拉偏慢（可優化）", level: "warn" };
  return { label: "有回拉（較佳）", level: "good" };
}

function formatNum(x, digits = 2) {
  if (!Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

function computeContactMetrics(frames, contacts, leg, direction) {
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

    // leg length (approx)
    const legLen = dist2D(hip, knee) + dist2D(knee, ankle);

    // overstride: ankle ahead of hip along running direction
    const overstride = (ankle.x - hip.x) * direction;
    const overstrideRatio = legLen > 0 ? overstride / legLen : NaN;

    // knee angle at contact
    const kneeAngle = angleDeg(hip, knee, ankle);

    // trunk lean relative to vertical, sign aligned to running direction
    const dx = (shoulder.x - hip.x) * direction;
    const dy = hip.y - shoulder.y; // positive if shoulder above hip
    const trunkLeanDeg = dy !== 0 ? (Math.atan(dx / dy) * 180) / Math.PI : NaN;

    const strike = classifyFootStrike(heel, toe);

    // retraction speed: slope of (ankle - hip) along direction in last 0.12s
    const w = f.sampleFps ? Math.max(2, Math.floor(f.sampleFps * 0.12)) : 2;
    const j0 = Math.max(0, i - w);
    const f0 = frames[j0];
    let retractSpeed = NaN;
    if (f0?.landmarks) {
      const L0 = f0.landmarks;
      const hip0 = midpoint(L0[idxHipL], L0[idxHipR]);
      const ankle0 = L0[idxAnkle];
      const rel0 = (ankle0.x - hip0.x) * direction;
      const rel1 = (ankle.x - hip.x) * direction;
      const dt = f.t - f0.t;
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
  const flags = [];

  if (summary.overstrideRatioMedian >= 0.12) {
    flags.push({ level: "bad", text: "GRAVITY BACK / 步幅偏長：落地點在髖前方（overstride）機率高" });
  } else {
    flags.push({ level: "good", text: "落點相對接近髖下（overstride 風險較低）" });
  }

  if (summary.kneeAngleMedian >= 165) {
    flags.push({ level: "bad", text: "KNEE LOCKED：接觸瞬間膝角偏大（偏伸直/接近打直）" });
  } else {
    flags.push({ level: "good", text: "膝角保有彈性（較佳）" });
  }

  if (summary.trunkLeanMedian < -1) {
    flags.push({ level: "bad", text: "TORSO BACKWARD TILT：軀幹偏後仰（堆疊偏後）" });
  } else {
    flags.push({ level: "good", text: "軀幹堆疊尚可（未見明顯後仰）" });
  }

  if (summary.heelStrikeRate >= 0.6) {
    flags.push({ level: "warn", text: "HEEL LANDING：多數接觸為後足先著地（本身不一定是問題，需搭配 overstride 判讀）" });
  } else if (summary.heelStrikeRate <= 0.2) {
    flags.push({ level: "good", text: "著地較偏中足/前足（以本工具估算）" });
  } else {
    flags.push({ level: "warn", text: "著地型態混合（以本工具估算）" });
  }

  if (summary.retractSpeedMedian < 0.02) {
    flags.push({ level: "warn", text: "LOWER LEG SLOW PULLING：落地前腳回拉速度偏慢（可能與 overstride / 協調性有關）" });
  } else {
    flags.push({ level: "good", text: "落地前有回拉（較佳）" });
  }

  return flags;
}

function buildAdvice(summary) {
  const out = [];

  if (summary.overstrideRatioMedian >= 0.12) {
    out.push("優先處理：縮短步幅、提高步頻 3–7%，把「腳往下放在髖下」當作主要 cue（避免伸腳去踩地）。");
    out.push("可操作訓練：節拍器 1 分鐘正常 + 1 分鐘 +5% 步頻，做 6–10 組；每週 2 次，連續 3–4 週。");
  } else {
    out.push("步幅/落點：目前落點相對接近髖下，可把優化重點放在力量與彈性回彈能力（而非刻意改著地型態）。");
  }

  if (summary.kneeAngleMedian >= 165) {
    out.push("膝角：避免落地瞬間膝蓋接近打直。通常「步幅變短」會自然改善膝角，不要單獨硬凹膝蓋。");
  }

  if (summary.trunkLeanMedian < -1) {
    out.push("軀幹：用『耳朵—肩—髖』垂直堆疊、從腳踝微前傾（不是從腰折）來改善後仰。");
  }

  if (summary.heelStrikeRate >= 0.6 && summary.overstrideRatioMedian < 0.12) {
    out.push("著地：後足先著地在很多配速下很常見；若落點已接近髖下，通常不必強迫改前足著地（避免小腿/阿基里斯過載）。");
  } else if (summary.heelStrikeRate >= 0.6 && summary.overstrideRatioMedian >= 0.12) {
    out.push("著地：與其把目標放在『改前足』，更應先把落地點拉回髖下；足部著地型態常會隨步幅縮短自然改變。");
  }

  out.push("力量/彈性（對跑步經濟性常更關鍵）：每週 2 次下肢力量（分腿蹲、髖伸、提踵）+ 1 次輕量增強式（踝彈跳/跳繩），連做 8–12 週再評估。");

  return out;
}

function renderResults(analysis) {
  const s = analysis.summary;

  els.summary.textContent =
    `完成分析：取樣 ${analysis.meta.sampleFps} fps，影片長度 ${formatNum(analysis.meta.durationSec, 1)} 秒。` +
    `\n有效接觸事件：左 ${analysis.contacts.left.length}、右 ${analysis.contacts.right.length}（共 ${analysis.contacts.all.length}）。` +
    `\n方向：${analysis.meta.direction > 0 ? "面向/跑向右（→）" : "面向/跑向左（←）"}（${analysis.meta.directionMode}）。` +
    `\n\n提醒：以下數值以 2D 估算，會受鏡頭角度、距離、畫面裁切、遮擋、衣著、跑台/戶外等影響。`;

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
      name: "Overstride ratio（落地點在髖前的比例）",
      val: formatNum(s.overstrideRatioMedian, 3),
      interp: interpretOverstride(s.overstrideRatioMedian).label,
    },
    {
      name: "Knee angle @ contact（膝角，度）",
      val: formatNum(s.kneeAngleMedian, 1),
      interp: interpretKnee(s.kneeAngleMedian).label,
    },
    {
      name: "Trunk lean（軀幹傾角，度；+向前 / -向後）",
      val: formatNum(s.trunkLeanMedian, 1),
      interp: interpretTrunkLean(s.trunkLeanMedian).label,
    },
    {
      name: "Heel-strike rate（後足先著地比例）",
      val: `${formatNum(s.heelStrikeRate * 100, 1)}%`,
      interp: s.heelStrikeRate >= 0.6 ? "多數後足" : (s.heelStrikeRate <= 0.2 ? "多數非後足" : "混合"),
    },
    {
      name: "Retraction speed（落地前回拉速度，越大越好）",
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
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
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
  if (!poseLandmarker) {
    await loadModel();
  }
  if (!els.file.files?.[0]) {
    alert("請先選擇影片檔");
    return;
  }
  if (!els.video.duration || !Number.isFinite(els.video.duration)) {
    alert("影片尚未載入完成，請稍候再試");
    return;
  }

  resetOutputs();
  setProgress(0);

  const sampleFps = Number(els.sampleFps.value || 15);
  const minStepSec = Number(els.minStepSec.value || 0.25);

  const duration = els.video.duration;
  const dt = 1 / Math.max(5, Math.min(60, sampleFps));
  const steps = Math.max(1, Math.ceil(duration / dt));

  setStatus(`分析開始：duration=${duration.toFixed(2)}s, sampleFps=${sampleFps}, steps≈${steps}`);
  await ensureCanvasReady();

  // frame records
  const frames = [];
  let direction = 1;
  let directionLocked = false;

  // 用 seek + detectForVideo 逐幀取樣
  // 注意：對很長的影片會比較花時間；實務上可加「只分析某段」或「先裁切」等功能。
  for (let k = 0; k <= steps; k++) {
    const t = Math.min(duration, k * dt);

    // Seek
    await seekTo(els.video, t);

    // detect
    let result = null;
    try {
      result = poseLandmarker.detectForVideo(els.video, t * 1000);
    } catch (err) {
      log(`detectForVideo error @${t.toFixed(2)}s: ${err}`);
      result = null;
    }

    const landmarks = result?.landmarks?.[0] ?? null;

    if (landmarks && !directionLocked) {
      direction = directionFromUI(landmarks);
      directionLocked = true;
    }

    frames.push({
      t,
      landmarks,
      sampleFps,
    });

    if (landmarks) {
      drawOverlay(landmarks, direction);
    } else {
      clearOverlay();
    }

    if (k % 8 === 0) {
      setProgress((k / steps) * 80); // analysis portion 0-80
      await yieldToUI();
    }
  }

  setProgress(82);
  setStatus("計算步態事件與摘要...");

  // Build y arrays for heel peaks
  const leftHeelY = frames.map((f) => f.landmarks?.[29]?.y ?? NaN);
  const rightHeelY = frames.map((f) => f.landmarks?.[30]?.y ?? NaN);
  const timeArr = frames.map((f) => f.t);

  const leftPeaks = findContactPeaks(leftHeelY, timeArr, minStepSec);
  const rightPeaks = findContactPeaks(rightHeelY, timeArr, minStepSec);

  // compute metrics at peaks
  const leftMetrics = computeContactMetrics(frames, leftPeaks, "L", direction);
  const rightMetrics = computeContactMetrics(frames, rightPeaks, "R", direction);
  const allMetrics = [...leftMetrics, ...rightMetrics].sort((a, b) => a.t - b.t);

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
      sampleFps,
      minStepSec,
      direction,
      directionMode: els.directionMode.value,
      model: {
        name: "MediaPipe PoseLandmarker Lite",
        modelUrl: MODEL_URL,
        wasmRoot: WASM_ROOT,
      },
      notes: [
        "All metrics are 2D approximations in normalized image coordinates.",
        "Contact events are detected by local maxima of heel y (lowest point) and can be noisy.",
        "Do not use as medical diagnosis. Use as training feedback only.",
      ],
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
  setStatus("完成");
}

els.btnLoadModel.addEventListener("click", async () => {
  try {
    await loadModel();
  } catch (e) {
    console.error(e);
    setStatus(`模型載入失敗：${e}`);
    alert("模型載入失敗，請檢查網路或改用 Chrome/Edge 再試。");
  }
});

els.file.addEventListener("change", () => {
  const file = els.file.files?.[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  els.video.src = url;
  els.video.load();

  setStatus(`已選擇影片：${file.name}（${Math.round(file.size / 1024 / 1024)} MB）`);
  setProgress(0);

  els.video.onloadedmetadata = async () => {
    setStatus(`影片已載入：${els.video.videoWidth}x${els.video.videoHeight}, duration=${els.video.duration.toFixed(2)}s`);
    await ensureCanvasReady();
    maybeEnableAnalyze();
  };
});

els.btnAnalyze.addEventListener("click", async () => {
  els.btnAnalyze.disabled = true;
  try {
    await analyze();
  } catch (e) {
    console.error(e);
    setStatus(`分析失敗：${e}`);
    alert("分析失敗。建議換短一點的影片、降低取樣 FPS，或換 Chrome/Edge 再試。");
  } finally {
    els.btnAnalyze.disabled = false;
  }
});

els.btnDownload.addEventListener("click", () => {
  if (!lastAnalysis) return;
  downloadJson(lastAnalysis, "run-gait-analysis.json");
});

// 初始 UI 狀態
setStatus("等待使用者操作：先『載入模型』→ 選擇影片 → 開始分析");
setProgress(0);
maybeEnableAnalyze();
