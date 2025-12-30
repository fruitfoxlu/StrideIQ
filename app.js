// Running Form Analysis Web App (MVP)
// Browser-only: video never uploaded; analysis runs in the browser.
// Important: This is a demo MVP. Production quality needs more robust event detection (initial contact / toe-off)
// and camera calibration (pixel-to-real distance, lens distortion, camera angle compensation, 3D/multi-view, etc).

import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm";
const DEFAULT_SAMPLE_FPS = 24;
const DEFAULT_MIN_STEP_SEC = 0.3;
const DIRECTION_MODE = "auto";
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
const MIN_DIRECTION_SAMPLES = 8;
const MAX_DIRECTION_FLIP_RATIO = 0.25;

const $ = (sel) => document.querySelector(sel);

const els = {
  file: $("#videoFile"),
  video: $("#video"),
  overlay: $("#overlay"),
  btnAnalyze: $("#btnAnalyze"),
  btnDownload: $("#btnDownload"),
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
  els.status.textContent = `Status: ${msg}`;
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
    drawingUtils = new DrawingUtils(overlayCtx);
  }
}

async function loadModel() {
  if (poseLandmarker) {
    setStatus("Model already loaded (skipped).");
    return;
  }
  setStatus("Loading model (WASM + model file)...");
  setProgress(3);

  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

  // Try GPU first, fall back to CPU if the delegate is unavailable.
  try {
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    setStatus("Model loaded (GPU delegate).");
  } catch (err) {
    log(`GPU delegate init failed, falling back to CPU. Reason: ${err}`);
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    setStatus("Model loaded (CPU delegate).");
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
  els.btnAnalyze.disabled = !(hasFile && hasVideo);
}

function resetOutputs() {
  els.summary.classList.remove("muted");
  els.flags.classList.remove("muted");
  els.advice.classList.remove("muted");
  els.summary.textContent = "Checking video requirements...";
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

  // Skeleton
  drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    lineWidth: 3,
  });
  drawingUtils.drawLandmarks(landmarks, {
    radius: (d) => 2 + 2 * clamp01(d.from?.z ?? 0),
  });

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

function interpretOverstride(ratio) {
  if (!Number.isFinite(ratio)) return { label: "Unknown", level: "na" };
  if (ratio >= 0.20) return { label: "Severe (high likelihood of overstride)", level: "bad" };
  if (ratio >= 0.12) return { label: "Moderate (possible overstride)", level: "warn" };
  if (ratio >= 0.06) return { label: "Mild (could improve)", level: "warn" };
  return { label: "Good (landing near the hip)", level: "good" };
}

function interpretKnee(angle) {
  if (!Number.isFinite(angle)) return { label: "Unknown", level: "na" };
  if (angle >= 170) return { label: "Very close to locked", level: "bad" };
  if (angle >= 160) return { label: "Extended", level: "warn" };
  return { label: "Flexed (better)", level: "good" };
}

function interpretTrunkLean(deg) {
  // Positive: forward (aligned with running direction), negative: backward.
  if (!Number.isFinite(deg)) return { label: "Unknown", level: "na" };
  if (deg < -2) return { label: "Backward lean (may reduce propulsion)", level: "bad" };
  if (deg < 0) return { label: "Slight backward lean", level: "warn" };
  if (deg <= 12) return { label: "Reasonable forward lean", level: "good" };
  return { label: "Large forward lean (pace dependent)", level: "warn" };
}

function interpretRetraction(speed) {
  // Positive: pulling back before contact. Negative: still reaching forward.
  if (!Number.isFinite(speed)) return { label: "Unknown", level: "na" };
  if (speed < -0.02) return { label: "Clear forward reach (low retraction)", level: "bad" };
  if (speed < 0.02) return { label: "Slow pullback (could improve)", level: "warn" };
  return { label: "Good pullback", level: "good" };
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
      text: "No valid contact events detected. Try a clearer side-view video with good lighting.",
    }];
  }

  const flags = [];

  if (Number.isFinite(summary.overstrideRatioMedian)) {
    if (summary.overstrideRatioMedian >= 0.12) {
      flags.push({ level: "bad", text: "Overstride risk: landing ahead of the hip." });
    } else {
      flags.push({ level: "good", text: "Landing is close to the hip (lower overstride risk)." });
    }
  } else {
    flags.push({ level: "warn", text: "Overstride: unable to estimate." });
  }

  if (Number.isFinite(summary.kneeAngleMedian)) {
    if (summary.kneeAngleMedian >= 165) {
      flags.push({ level: "bad", text: "Knee near locked at contact." });
    } else {
      flags.push({ level: "good", text: "Knee has some flex at contact." });
    }
  } else {
    flags.push({ level: "warn", text: "Knee angle: unable to estimate." });
  }

  if (Number.isFinite(summary.trunkLeanMedian)) {
    if (summary.trunkLeanMedian < -1) {
      flags.push({ level: "bad", text: "Backward torso lean (stacking behind)." });
    } else {
      flags.push({ level: "good", text: "Torso stack looks reasonable (no clear backward lean)." });
    }
  } else {
    flags.push({ level: "warn", text: "Trunk lean: unable to estimate." });
  }

  if (Number.isFinite(summary.heelStrikeRate)) {
    if (summary.heelStrikeRate >= 0.6) {
      flags.push({ level: "warn", text: "Heel landing is common; interpret with overstride." });
    } else if (summary.heelStrikeRate <= 0.2) {
      flags.push({ level: "good", text: "More mid/forefoot landing (estimated)." });
    } else {
      flags.push({ level: "warn", text: "Mixed strike pattern (estimated)." });
    }
  } else {
    flags.push({ level: "warn", text: "Foot strike type: unable to estimate." });
  }

  if (Number.isFinite(summary.retractSpeedMedian)) {
    if (summary.retractSpeedMedian < 0.02) {
      flags.push({ level: "warn", text: "Slow pullback before contact (may relate to overstride)." });
    } else {
      flags.push({ level: "good", text: "Good pullback before contact." });
    }
  } else {
    flags.push({ level: "warn", text: "Leg retraction: unable to estimate." });
  }

  return flags;
}

function buildAdvice(summary) {
  if (!summary.contactCount) {
    return [
      "No valid contact events detected. Try a clearer side-view video with minimal occlusion and steady framing.",
    ];
  }

  const out = [];

  if (Number.isFinite(summary.overstrideRatioMedian)) {
    if (summary.overstrideRatioMedian >= 0.12) {
      out.push("Priority: shorten stride and raise cadence 3-7%. Cue: place the foot down under the hip (avoid reaching).");
      out.push("Training: metronome intervals - 1 min normal + 1 min +5% cadence, 6-10 sets, 2x/week for 3-4 weeks.");
    } else {
      out.push("Stride/landing: landing is close to the hip; focus on strength and elastic rebound rather than forcing a strike change.");
    }
  } else {
    out.push("Overstride: unable to estimate. Try a clearer side-view video with visible hips and feet.");
  }

  if (Number.isFinite(summary.kneeAngleMedian) && summary.kneeAngleMedian >= 165) {
    out.push("Knee angle: avoid a near-locked knee at contact. Shorter stride usually improves this; do not force the knee alone.");
  }

  if (Number.isFinite(summary.trunkLeanMedian) && summary.trunkLeanMedian < -1) {
    out.push("Torso: stack ear-shoulder-hip vertically and lean slightly from the ankles (not the waist).");
  }

  if (Number.isFinite(summary.heelStrikeRate) && Number.isFinite(summary.overstrideRatioMedian)) {
    if (summary.heelStrikeRate >= 0.6 && summary.overstrideRatioMedian < 0.12) {
      out.push("Foot strike: heel striking is common at many paces; if landing is near the hip, no need to force forefoot (avoid calf/Achilles overload).");
    } else if (summary.heelStrikeRate >= 0.6 && summary.overstrideRatioMedian >= 0.12) {
      out.push("Foot strike: prioritize moving the landing back under the hip; strike pattern often shifts naturally.");
    }
  }

  out.push("Strength/elasticity (often most important for running economy): 2x/week lower-body strength (split squat, hip extension, calf raises) + 1x light plyometrics (ankle hops/jump rope) for 8-12 weeks.");

  return out;
}

function renderIssues(title, issues, suggestions) {
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

  els.metricsTable.innerHTML = "<tr><td colspan=\"3\" class=\"muted\">Not available</td></tr>";
  els.btnDownload.disabled = true;
}

function renderResults(analysis) {
  const s = analysis.summary;

  const durationLabel = analysis.meta.slowMoFactor > 1
    ? `video length ${formatNum(analysis.meta.durationSec, 1)} s (slow-mo x${analysis.meta.slowMoFactor} => real time ${formatNum(analysis.meta.realDurationSec, 1)} s)`
    : `video length ${formatNum(analysis.meta.durationSec, 1)} s`;

  els.summary.textContent =
    `Analysis complete: sampled at ${analysis.meta.sampleFps} fps, ${durationLabel}.` +
    `\nValid contact events: L ${analysis.contacts.left.length}, R ${analysis.contacts.right.length} (total ${analysis.contacts.all.length}).` +
    `\nDirection: ${analysis.meta.direction > 0 ? "Facing/running right (>)" : "Facing/running left (<)"} (${analysis.meta.directionMode}).` +
    "\n\nReminder: metrics are 2D estimates and can be affected by camera angle, distance, cropping, occlusion, clothing, treadmill vs outdoor, etc.";

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
      name: "Overstride ratio (landing ahead of hip, normalized)",
      val: formatNum(s.overstrideRatioMedian, 3),
      interp: interpretOverstride(s.overstrideRatioMedian).label,
    },
    {
      name: "Knee angle @ contact (deg)",
      val: formatNum(s.kneeAngleMedian, 1),
      interp: interpretKnee(s.kneeAngleMedian).label,
    },
    {
      name: "Trunk lean (deg; +forward / -backward)",
      val: formatNum(s.trunkLeanMedian, 1),
      interp: interpretTrunkLean(s.trunkLeanMedian).label,
    },
    {
      name: "Heel-strike rate",
      val: `${formatNum(s.heelStrikeRate * 100, 1)}%`,
      interp: s.heelStrikeRate >= 0.6 ? "Mostly heel" : (s.heelStrikeRate <= 0.2 ? "Mostly non-heel" : "Mixed"),
    },
    {
      name: "Retraction speed (pullback before contact; higher is better)",
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

async function estimateVideoFps(video) {
  const supportsRvfc = typeof video.requestVideoFrameCallback === "function";
  const supportsQuality = typeof video.getVideoPlaybackQuality === "function";
  if (!supportsRvfc && !supportsQuality) return NaN;

  const sampleMs = 600;
  const minFrames = 8;
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

  try {
    await video.play();
  } catch (err) {
    // Autoplay can still fail; fall through and return NaN.
  }

  await sleep(sampleMs + 150);

  if (supportsRvfc && typeof video.cancelVideoFrameCallback === "function" && rafId !== null) {
    video.cancelVideoFrameCallback(rafId);
  }

  video.pause();

  const elapsedMs = start ? performance.now() - start : (performance.now() - startWall);
  let fps = NaN;
  if (elapsedMs > 0) {
    if (supportsRvfc && frameCount >= minFrames) {
      fps = frameCount / (elapsedMs / 1000);
    } else if (supportsQuality) {
      const qualityEnd = video.getVideoPlaybackQuality().totalVideoFrames;
      const diff = qualityEnd - qualityStart;
      if (diff >= minFrames) {
        fps = diff / (elapsedMs / 1000);
      }
    }
  }

  video.currentTime = originalTime;
  video.muted = wasMuted;
  video.playbackRate = wasPlaybackRate;
  if (!wasPaused) {
    try {
      await video.play();
    } catch (err) {
      // Ignore resume errors.
    }
  }

  return fps;
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
    alert("Please choose a video file first.");
    return;
  }
  if (!els.video.duration || !Number.isFinite(els.video.duration)) {
    alert("Video is not fully loaded yet. Please try again in a moment.");
    return;
  }

  resetOutputs();
  setProgress(0);
  setStatus("Checking video requirements...");

  const duration = els.video.duration;
  const vw = els.video.videoWidth || 0;
  const vh = els.video.videoHeight || 0;
  const longEdge = Math.max(vw, vh);
  const shortEdge = Math.min(vw, vh);

  if (longEdge < MIN_LONG_EDGE || shortEdge < MIN_SHORT_EDGE) {
    renderIssues(
      "Video resolution too low.",
      [
        `Minimum required: ${MIN_LONG_EDGE}x${MIN_SHORT_EDGE} (1080p).`,
        `Detected: ${vw}x${vh}.`,
      ],
      [
        "Record or export at 1080p or higher.",
        "Avoid heavy cropping or digital zoom.",
      ],
    );
    setStatus("Analysis stopped: resolution too low.");
    setProgress(0);
    return;
  }

  setStatus("Checking video frame rate...");
  const inputFps = await estimateVideoFps(els.video);
  if (!Number.isFinite(inputFps)) {
    renderIssues(
      "Unable to verify frame rate.",
      ["The browser could not read the video FPS."],
      [
        `Use ${NORMAL_FPS} fps (normal) or ${SLOWMO_FPS} fps (slow-motion).`,
        "Try Chrome or Edge if this keeps happening.",
      ],
    );
    setStatus("Analysis stopped: FPS unknown.");
    setProgress(0);
    return;
  }

  const isNormalFps = Math.abs(inputFps - NORMAL_FPS) <= NORMAL_FPS_TOLERANCE;
  const isSlowMoFps = Math.abs(inputFps - SLOWMO_FPS) <= SLOWMO_FPS_TOLERANCE;
  if (!isNormalFps && !isSlowMoFps) {
    renderIssues(
      "Unsupported frame rate.",
      [
        `Supported: ${NORMAL_FPS} fps (normal) or ${SLOWMO_FPS} fps (slow-motion).`,
        `Detected: ${formatNum(inputFps, 1)} fps.`,
      ],
      [
        `Export at ${NORMAL_FPS} fps or ${SLOWMO_FPS} fps.`,
        "Avoid 60/120 fps exports or time-lapse.",
      ],
    );
    setStatus("Analysis stopped: unsupported FPS.");
    setProgress(0);
    return;
  }

  const slowMoFactor = isSlowMoFps ? SLOWMO_FACTOR : 1;
  const realDuration = duration / slowMoFactor;
  if (realDuration < MIN_DURATION_SEC) {
    renderIssues(
      "Video too short to analyze.",
      [
        `Minimum real-time duration is ${MIN_DURATION_SEC.toFixed(1)} seconds.`,
        `Detected: ${formatNum(realDuration, 2)} seconds (video length ${formatNum(duration, 2)}s).`,
      ],
      [
        "Record 3-8 seconds of steady, side-view running.",
        "Ensure at least 4-6 full strides are visible.",
      ],
    );
    setStatus("Analysis stopped: video too short.");
    setProgress(0);
    return;
  }

  if (!poseLandmarker) {
    await loadModel();
  }

  const sampleFps = DEFAULT_SAMPLE_FPS;
  const minStepSec = DEFAULT_MIN_STEP_SEC * slowMoFactor;
  const dt = 1 / Math.max(5, Math.min(60, sampleFps));
  const steps = Math.max(1, Math.ceil(duration / dt));

  setStatus(`Analysis started: duration=${duration.toFixed(2)}s, sampleFps=${sampleFps}, steps~${steps}`);
  await ensureCanvasReady();

  // Frame records
  const frames = [];
  let direction = 1;
  let directionLocked = false;
  let detectedFrames = 0;
  let directionLeft = 0;
  let directionRight = 0;

  // Sample frames using seek + detectForVideo.
  // Long videos may take time; consider trimming for faster analysis.
  for (let k = 0; k <= steps; k++) {
    const t = Math.min(duration, k * dt);

    // Seek
    await seekTo(els.video, t);

    // Detect
    let result = null;
    try {
      result = poseLandmarker.detectForVideo(els.video, t * 1000);
    } catch (err) {
      log(`detectForVideo error @${t.toFixed(2)}s: ${err}`);
      result = null;
    }

    const landmarks = result?.landmarks?.[0] ?? null;
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
  setStatus("Computing gait events and summary...");

  const detectionRatio = frames.length ? detectedFrames / frames.length : 0;
  if (detectedFrames === 0) {
    renderIssues(
      "No runner detected.",
      ["No person could be detected in the video."],
      [
        "Use a clear side-view shot with the full body visible.",
        "Increase lighting and avoid motion blur.",
      ],
    );
    setStatus("Analysis stopped: no person detected.");
    setProgress(100);
    return;
  }

  if (detectedFrames < MIN_DETECTED_FRAMES || detectionRatio < MIN_DETECTION_RATIO) {
    renderIssues(
      "Low detection quality.",
      [
        "Too few frames contained reliable pose landmarks.",
        "Video may be blurry, too dark, or the runner is too small in frame.",
      ],
      [
        "Use a brighter, sharper video and keep the runner larger in frame.",
        "Avoid heavy occlusion and fast camera motion.",
      ],
    );
    setStatus("Analysis stopped: low detection quality.");
    setProgress(100);
    return;
  }

  const directionSamples = directionLeft + directionRight;
  if (directionSamples < MIN_DIRECTION_SAMPLES) {
    renderIssues(
      "Direction unclear.",
      ["The runner's facing direction could not be determined reliably."],
      [
        "Keep the runner fully visible in a clear side view.",
        "Avoid heavy occlusion or extreme camera angles.",
      ],
    );
    setStatus("Analysis stopped: direction unclear.");
    setProgress(100);
    return;
  }

  if (directionLeft > 0 && directionRight > 0) {
    const flipRatio = Math.min(directionLeft, directionRight) / directionSamples;
    if (flipRatio > MAX_DIRECTION_FLIP_RATIO) {
      renderIssues(
        "Inconsistent running direction.",
        ["The runner appears to change direction within the clip."],
        [
          "Use a single-direction clip (left-to-right or right-to-left).",
          "Avoid out-and-back segments or camera flips.",
        ],
      );
      setStatus("Analysis stopped: inconsistent direction.");
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
    renderIssues(
      "Too few strides detected.",
      [
        "Not enough clear contact events were found.",
        `Detected: L ${leftMetrics.length}, R ${rightMetrics.length}.`,
      ],
      [
        "Use a longer clip with several strides (3-5 seconds).",
        "Keep the runner fully visible from head to feet.",
      ],
    );
    setStatus("Analysis stopped: too few strides.");
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
        name: "MediaPipe PoseLandmarker Lite",
        modelUrl: MODEL_URL,
        wasmRoot: WASM_ROOT,
      },
      notes: [
        "All metrics are 2D approximations in normalized image coordinates.",
        "Contact events are detected by local maxima of heel y (lowest point) and can be noisy.",
        "Time-based metrics are normalized using the slow-motion factor when applicable.",
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
  setStatus("Done.");
}

els.file.addEventListener("change", () => {
  const file = els.file.files?.[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  els.video.src = url;
  els.video.load();

  setStatus(`Selected video: ${file.name} (${Math.round(file.size / 1024 / 1024)} MB)`);
  setProgress(0);

  els.video.onloadedmetadata = async () => {
    setStatus(`Video loaded: ${els.video.videoWidth}x${els.video.videoHeight}, duration=${els.video.duration.toFixed(2)}s`);
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
    setStatus(`Analysis failed: ${e}`);
    alert("Analysis failed. Try a shorter video, lower sampling FPS, or Chrome/Edge.");
  } finally {
    els.btnAnalyze.disabled = false;
  }
});

els.btnDownload.addEventListener("click", () => {
  if (!lastAnalysis) return;
  downloadJson(lastAnalysis, "run-gait-analysis.json");
});

// Initial UI state
setStatus("Waiting for input: choose video -> Start Analysis");
setProgress(0);
maybeEnableAnalyze();
