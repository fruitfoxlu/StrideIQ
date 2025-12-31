# Running Form Analysis Web App (MVP, browser-only)

This is a deployable MVP with the following features:

- Uses **TensorFlow.js BlazePose (Web)** to detect 33 pose landmarks in the browser (TF.js + pose-detection vendored under `vendor/`)
- Simple, rule-based analysis (overstride, knee angle, trunk lean, heel-strike rate, pre-contact retraction speed)
- Video stays local (no upload), suitable for PoC and early product validation
- Defaults: 24 FPS sampling and a 0.30s minimum step interval to reduce double counting

Note: This is a demo MVP. Production quality usually needs:
- More reliable gait event detection (initial contact / toe-off)
- Camera calibration and scale conversion (pixels to cm/deg)
- Multi-view or 3D to reduce angle bias and occlusion
- Data quality controls (lighting, framing, treadmill vs outdoor)

## Quick Use

1) Record a side-view clip that meets the requirements below.
2) Serve the folder over HTTP (see Local Development).
3) Open the app, choose the video, click Start Analysis, and optionally download the JSON.
4) Use the language toggle in the header to switch English/中文.
5) Share language-specific links with `?lang=en` or `?lang=zh-Hant`.

## Project Structure

- `index.html`: UI
- `style.css`: styling
- `app.js`: BlazePose model loading, frame sampling, metrics, and UI updates
- `i18n.js`: language strings for English/Traditional Chinese
- `Dockerfile`: container for static hosting

## Video Requirements

- Side view, single runner, consistent direction (left-to-right or right-to-left)
- 1080p or higher at 30 fps (video your browser can decode), or 240 fps slow-mo (iPhone Slo-mo)
- At least 3 seconds of running (4-6 strides)

## Local Development (recommended)

Because ES modules are restricted on `file://`, serve over HTTP:

### Option A: Python (simplest)

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

### Option B: Node (optional)

```bash
npx serve .
```

## Docker (optional)

```bash
docker build -t run-gait-webapp .
docker run --rm -p 8080:80 run-gait-webapp
# open http://localhost:8080
```

## Deployment Options

Because all analysis runs in the browser, any static host works:

- GitHub Pages
- Cloudflare Pages
- Netlify / Vercel
- GCS Static Website + Cloud CDN
- S3 Static Website + CloudFront

## Data & Privacy

- Video analysis runs in the browser; nothing is uploaded unless you add a backend
- Model and runtime assets are fetched from CDN URLs in `app.js` (you can self-host if needed)

## License

- App code: add your own LICENSE as needed
- TensorFlow.js / model assets: follow Google/TensorFlow licensing terms

## Feedback

Email: fruitfoxlu@gmail.com
