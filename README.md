# 跑姿分析 Web App（MVP，純前端）

這是一個「可部署」的最小可行版本（MVP），功能是：

- 使用 **MediaPipe PoseLandmarker（Web）** 在瀏覽器端偵測人體 33 個姿勢點
- 針對跑姿做「簡化規則」分析（overstride、膝角、軀幹傾角、後足著地比例、落地前回拉速度）
- 影片 **不需要上傳到伺服器**（適合先做 PoC/產品流程驗證，並降低雲端成本）

> 注意：這是示範用 MVP。若要接近商用品質，通常需要：
> - 更可靠的步態事件偵測（initial contact / toe-off）
> - 相機校正與尺度換算（像素→公分/度）
> - 多視角或 3D（降低視角偏差與遮擋）
> - 資料品質控管（光線、取景、跑台/戶外差異）

## 專案結構

- `index.html`：UI
- `style.css`：樣式
- `app.js`：MediaPipe 模型載入、逐幀分析、指標計算、結果輸出
- `Dockerfile`：可用容器部署到 Cloud Run / ECS / 任意容器平台

## 本機啟動（推薦）

由於瀏覽器對 `file://` 的 ES module 會有限制，請用本機 HTTP server：

### 方式 A：Python（最簡單）

```bash
python3 -m http.server 8000
# 然後開啟 http://localhost:8000
```

### 方式 B：Node（可選）

```bash
npx serve .
```

## Docker 部署（可直接上雲）

```bash
docker build -t run-gait-webapp .
docker run --rm -p 8080:80 run-gait-webapp
# 打開 http://localhost:8080
```

## 上線部署建議

### 1) 靜態網站（最省錢、最簡單）

因為分析都在瀏覽器端跑，你只要能把這三個檔案提供成靜態網站即可：

- GitHub Pages
- Cloudflare Pages
- Netlify / Vercel
- GCS Static Website + Cloud CDN（GCP）
- S3 Static Website + CloudFront（AWS）

### 2) Cloud Run（你需要「容器化」時）

你可以直接用 `Dockerfile` 上 Cloud Run。此情境常見於：
- 你要加登入、後端存檔、付費、管理後台
- 你想把模型改成後端推論（但成本與延遲會增加）

## 資料與隱私

- 影片分析在瀏覽器端進行，不會上傳到伺服器（除非你自行加後端）
- 模型與 wasm 會從 CDN / Google Storage 下載（可改成自己 host）

## 授權

- 本 repo 程式碼：你可自由修改使用（建議自行加 LICENSE）
- MediaPipe / 模型：依 Google/MediaPipe 原授權條款使用
