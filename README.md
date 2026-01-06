# yolo_backend

此專案為簡單的人流（行人）檢測與追蹤範例，使用 YOLOv8 與 ByteTrack 進行檢測與追蹤，會輸出每個影片的「人流統計報告」。

**主要檔案**

**功能簡述**
- 批次處理 `video/` 中的所有影片
- 使用 YOLO 只偵測行人（類別 0）
- 使用 ByteTrack 分配唯一 ID 並統計出現過的不同人數
- 輸出：每支影片的標註影片（可選）與 `tracking_results/人流统计报告.txt`

**快速開始（建議）**

---

**Web 上傳介面（可選）**

本專案也提供一個簡單的 Web 上傳頁面，方便非開發者上傳影片到專案的 `video/` 資料夾。實作檔案：`web_uploader.py` 以及 `templates/index.html`。

啟動方式（在專案根目錄並啟用虛擬環境）：

```bash
pip install flask
python web_uploader.py
```

預設會在 `http://127.0.0.1:5000/` 提供上傳頁面（本地瀏覽器訪問）。上傳後的影片會存放到專案 `video/` 資料夾，之後可由 `检测人流.py` 批次處理。

注意事項：
- 若要公開對外使用，請替換 `app.secret_key`，並使用反向代理（如 nginx）與 HTTPS，加上適當的檔案大小限制與驗證。
- 若從遠程訪問，使用 `http://<server_ip>:5000/`（將 `<server_ip>` 替換為服務器 IP）。

1. 建議建立虛擬環境並啟用：

```bash
python -m venv .venv
source .venv/bin/activate
```

2. 安裝必要套件：

```bash
pip install -U pip
pip install ultralytics opencv-python-headless pillow numpy lap
```

3. 將欲分析的影片放到 `video/`（若無此資料夾請建立）：

```bash
mkdir -p video
# 把影片複製或上傳到 video/ 內
```

4. 執行腳本：

```bash
python 检测人流.py
```

執行後會在 `tracking_results/` 產生：
- `*_人流统计.mp4`（帶標註的輸出影片，若輸出被啟用）
- `人流统计报告.txt`（所有影片的統計摘要）

**若只需要報表而不輸出標註影片**
- 目前腳本預設會寫出標註影片。如果你只需要 `人流统计报告.txt`，有兩種選擇：
	1. 在執行前暫時將 `检测人流.py` 中寫檔（`out.write(frame)`）相關程式碼註解或移除。
	2. 我可以幫你修改腳本，新增命令列參數 `--no-video`（或 `--save-video False`）以在執行時選擇是否輸出影片。

**常見問題 & 排錯**
- 如果出現 `No module named 'lap'`，請執行：

```bash
pip install lap
```

- 如果影片無法開啟，請檢查檔案副檔名與檔案是否損毀，或者使用 `ffmpeg` 轉檔到常見格式再重試。

**範例流程**

```bash
# 1. 建立 venv、安裝套件
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install ultralytics opencv-python-headless pillow numpy lap

# 2. 把要檢測的影片放到 video/，再執行
python 检测人流.py

# 3. 查看結果
ls tracking_results
cat tracking_results/人流统计报告.txt
```

API
curl http://127.0.0.1:5000/api/latest_result


