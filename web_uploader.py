from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import subprocess
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import time
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'video')
app.config['RESULTS_FOLDER'] = os.path.join(os.getcwd(), 'tracking_results')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5 GB limit
app.secret_key = 'change-me-for-production'

# 全域 YOLO 模型（延遲載入）
yolo_model = None
last_detection_result = None  # 保存最後一次檢測結果

def get_yolo_model():
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO('yolov8n.pt')
    return yolo_model

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'hevc', 'h265'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('沒有上傳檔案')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('未選擇檔案')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            flash(f'上傳成功: {filename}')
            return redirect(url_for('index'))
        else:
            flash('不支援的檔案類型')
            return redirect(request.url)

    # GET: 列出目前 video/ 以及 tracking_results/ 中的檔案
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    files = sorted([f for f in os.listdir(app.config['UPLOAD_FOLDER']) if not f.startswith('.')])
    results = sorted([f for f in os.listdir(app.config['RESULTS_FOLDER']) if not f.startswith('.')])
    return render_template('index.html', files=files, results=results)

@app.route('/video/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/results/<path:filename>')
def download_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, as_attachment=True)


@app.route('/run_detection', methods=['POST'])
def run_detection():
    """在背景啟動 检测人流.py 並將輸出寫入 tracking_results/detection_run.log"""
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    log_path = os.path.join(app.config['RESULTS_FOLDER'], 'detection_run.log')
    python_exec = sys.executable
    script_path = os.path.join(os.getcwd(), '检测人流.py')

    try:
        # 以背景程序啟動，將 stdout/stderr 附加到 log
        log_file = open(log_path, 'ab')
        process = subprocess.Popen([python_exec, script_path], stdout=log_file, stderr=log_file)
        flash(f'已啟動檢測（PID {process.pid}），請稍候查看 tracking_results/detection_run.log')
    except Exception as e:
        flash(f'啟動檢測失敗: {e}')

    return redirect(url_for('index'))


@app.route('/camera')
def camera():
    """實時攝像頭檢測頁面"""
    return render_template('camera.html')


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """接收圖像並運行 YOLO 檢測，返回標註結果"""
    global last_detection_result
    
    if 'image' not in request.files:
        return jsonify({'error': '沒有圖像'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '空圖像'}), 400

    try:
        start_time = time.time()
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': '圖像解碼失敗'}), 400

        # 運行 YOLO 檢測（只檢測人類，類別 0）
        model = get_yolo_model()
        results = model(frame, classes=[0], conf=0.5, verbose=False)

        # 繪製檢測框
        people_count = 0
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            people_count = len(boxes)

            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 繪製人數統計
        cv2.putText(frame, f'People: {people_count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        inference_time = (time.time() - start_time) * 1000

        # 編碼為 JPEG base64
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 保存最後的檢測結果
        last_detection_result = {
            'image_base64': img_base64,
            'people_count': people_count,
            'inference_time': inference_time,
            'timestamp': time.time()
        }

        return jsonify({
            'image': img_base64,
            'people_count': people_count,
            'inference_time': inference_time
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/latest_result')
def get_latest_result():
    """取得最後一次檢測結果"""
    if last_detection_result is None:
        return jsonify({'error': '尚無結果'}), 404
    
    return jsonify({
        'people_count': last_detection_result['people_count'],
        'inference_time': last_detection_result['inference_time'],
        'timestamp': last_detection_result['timestamp']
    })


@app.route('/api/latest_image')
def get_latest_image():
    """下載最後一次檢測的圖像"""
    if last_detection_result is None:
        return jsonify({'error': '尚無結果'}), 404
    
    img_base64 = last_detection_result['image_base64']
    img_bytes = base64.b64decode(img_base64)
    return send_file(
        BytesIO(img_bytes),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name=f"detection_{int(last_detection_result['timestamp'])}.jpg"
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
