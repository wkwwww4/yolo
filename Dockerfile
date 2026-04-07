# 1. 使用輕量級的 Python 3.9 作為基底
FROM python:3.9-slim

# 2. 設定容器內的工作目錄
WORKDIR /app

# 3. 安裝 OpenCV 與 YOLO 執行時需要的系統底層套件
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. 複製 requirements.txt 進來
COPY requirements.txt .

# 5. 【關鍵步驟】先安裝所有套件，接著馬上「強制重裝」 headless 版本蓋過去！
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless

# 6. 將你專案內的所有檔案 (包含 yolov8n.pt, 所有的 .py 檔) 複製到容器內
COPY . .

# 7. 曝露 Flask 預設的 5000 埠
EXPOSE 5000

# 8. 設定容器啟動時執行的指令
CMD ["python", "web_uploader.py"]