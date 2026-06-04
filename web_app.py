# -*- coding: utf-8 -*-
import argparse
import contextlib
import io
import json
import os
import threading
import time
import traceback

import cv2
import pymysql
import yaml
from flask import Flask, Response, jsonify, render_template_string, request

from run_realtime_detection_yolo_new_3 import run_realtime_detection
from src.identification.DrugMatcher import DrugMatcher
from src.identification.Recog import PharmaceuticalBottleClassifier
from src.identification.rknn_ocr_adapter import RknnOCRRecognizer


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no")


def init_db(db_config):
    kwargs = dict(
        host=db_config.get("host", "192.168.137.1"),
        user=db_config.get("user", "root"),
        password=db_config.get("password", "root"),
        database=db_config.get("database", "medicine_db"),
        charset=db_config.get("charset", "utf8"),
        port=int(db_config.get("port", 3306)),
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        return pymysql.connect(**kwargs)
    except pymysql.err.OperationalError as exc:
        if exc.args and exc.args[0] == 1115 and kwargs["charset"].lower() == "utf8mb4":
            kwargs["charset"] = "utf8"
            return pymysql.connect(**kwargs)
        raise


def jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


class WebRuntime:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.stop_event = threading.Event()
        self.trigger_event = threading.Event()
        self.last_frame_jpeg = None
        self.last_result = None
        self.status = {
            "state": "idle",
            "processing": False,
            "frame_count": 0,
            "fps": 0.0,
            "trigger_mode": "manual",
            "started_at": None,
            "last_result_at": None,
            "last_error": None,
            "trigger_count": 0,
        }

    def snapshot_status(self):
        with self.lock:
            data = dict(self.status)
            data["thread_alive"] = bool(self.thread and self.thread.is_alive())
            data["frame_available"] = self.last_frame_jpeg is not None
            data["result_available"] = self.last_result is not None
            return data

    def update_status(self, data):
        with self.lock:
            self.status.update(jsonable(data))

    def update_frame(self, frame):
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with self.lock:
            self.last_frame_jpeg = encoded.tobytes()

    def update_result(self, payload):
        with self.lock:
            self.last_result = jsonable(payload)
            self.status["last_result_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.status["processing"] = False

    def get_frame(self):
        with self.lock:
            return self.last_frame_jpeg

    def get_result(self):
        with self.lock:
            return self.last_result

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return False, "already running"
            self.stop_event.clear()
            self.trigger_event.clear()
            self.status.update({
                "state": "starting",
                "processing": False,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_error": None,
            })
            self.thread = threading.Thread(target=self._run_detection, daemon=True)
            self.thread.start()
            return True, "started"

    def stop(self):
        self.stop_event.set()
        with self.lock:
            self.status["state"] = "stopping"
        return True, "stopping"

    def trigger(self):
        with self.lock:
            alive = bool(self.thread and self.thread.is_alive())
            if not alive:
                return False, "detection is not running"
            if self.status.get("processing"):
                return False, "recognition is still processing"
            self.status["trigger_count"] = int(self.status.get("trigger_count", 0)) + 1
            self.status["processing"] = True
        self.trigger_event.set()
        return True, "manual trigger queued"

    def _run_detection(self):
        conn = None
        try:
            self.update_status({"state": "loading_models", "processing": False})
            self._set_default_env()
            with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            runtime_config = config.get("runtime", {}) or {}
            recognition_workers = int(os.environ.get(
                "YILIAO_RECOGNITION_WORKERS",
                runtime_config.get("recognition_workers", 1),
            ))
            quiet_ocr = env_bool("YILIAO_QUIET_OCR", bool(runtime_config.get("quiet_ocr", True)))
            ocr_instance_count = int(os.environ.get(
                "YILIAO_OCR_INSTANCES",
                runtime_config.get("ocr_instances", 1),
            ))

            conn = init_db(config.get("db_config", {}))
            tables = config["table_config"]
            init_log_stream = io.StringIO() if quiet_ocr else None
            with contextlib.redirect_stdout(init_log_stream) if init_log_stream is not None else contextlib.nullcontext():
                drug_matcher = DrugMatcher(
                    conn,
                    drug_table=tables["drug_table"],
                    drug_column=tables["drug_column"],
                    patient_table=tables["patient_table"],
                    patient_column=tables["patient_column"],
                    cache_drugs=True,
                )
                classifier = PharmaceuticalBottleClassifier(db_conn=conn, device="npu")

            det_model_path = os.path.join(PROJECT_ROOT, "src", "identification", "Det_bs32.rknn")
            rec_model_path = os.path.join(PROJECT_ROOT, "model_ocr_0526.rknn")
            cls_model_path = os.path.join(PROJECT_ROOT, "model_cls_bs32.rknn")
            ocr_recognizers = [
                RknnOCRRecognizer(
                    det_model_path=det_model_path,
                    rec_model_path=rec_model_path,
                    cls_model_path=cls_model_path,
                    det_batch_size=32,
                    rec_batch_size=16,
                    cls_batch_size=32,
                )
                for _ in range(max(1, ocr_instance_count))
            ]

            self.update_status({"state": "running", "processing": False})
            run_realtime_detection(
                username=config["RTSP"]["username"],
                password=config["RTSP"]["password"],
                ip_address=config["RTSP"]["ip_address"],
                port=config["RTSP"]["port"],
                channel=config["RTSP"]["channel"],
                model=None,
                checkpoint_path=config["model"]["checkpoint_path"],
                num_classes=config["model"]["num_classes"],
                ocr_recognizer=ocr_recognizers,
                drug_matcher=drug_matcher,
                classifier=classifier,
                length=3,
                tile_size=config["segmentor"]["tile_size"],
                overlap=config["segmentor"]["overlap"],
                target_fps=config["segmentor"]["target_fps"],
                batch_frames=config["segmentor"]["batch_frames"],
                output_type=config["display"]["output_type"],
                overlay_alpha=config["display"]["overlay_alpha"],
                display_scale=config["display"]["display_scale"],
                save_video=False,
                output_path=config["saving"]["output_path"],
                save_fps=config["saving"]["save_fps"],
                device="npu",
                trigger_interval=999999.0,
                recognition_workers=recognition_workers,
                classifier_thread_safe=False,
                quiet_ocr=quiet_ocr,
                headless=True,
                trigger_mode="manual",
                manual_trigger_event=self.trigger_event,
                stop_event=self.stop_event,
                status_callback=self.update_status,
                result_callback=self.update_result,
                frame_callback=self.update_frame,
            )
        except Exception as exc:
            self.update_status({
                "state": "error",
                "processing": False,
                "last_error": repr(exc),
            })
            traceback.print_exc()
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            with self.lock:
                if self.status.get("state") not in ("error",):
                    self.status["state"] = "stopped"
                self.status["processing"] = False

    @staticmethod
    def _set_default_env():
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        os.environ.setdefault("YILIAO_MIXED_OCR_BATCH", "1")
        os.environ.setdefault("YILIAO_FEATURE_CACHE", "./single_image_feature_cache.pkl")
        os.environ.setdefault("YILIAO_VERBOSE_RUNTIME", "1")
        os.environ.setdefault("YILIAO_RUNTIME_LOGS", "1")
        os.environ.setdefault("YILIAO_COLLECT_TIMING", "1")
        os.environ.setdefault("YILIAO_QUIET_OCR", "0")
        os.environ.setdefault("YILIAO_CLS_MODEL", "./model_cls_bs32.rknn")
        os.environ.setdefault("YILIAO_CLS_BATCH_SIZE", "32")


runtime = WebRuntime()
app = Flask(__name__)


PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>药品识别控制台</title>
  <style>
    body { margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #101418; color: #edf2f7; }
    header { height: 48px; display: flex; align-items: center; padding: 0 16px; background: #18202a; border-bottom: 1px solid #2a3441; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 12px; padding: 12px; }
    .video { background: #050608; min-height: 600px; display: flex; align-items: center; justify-content: center; overflow: hidden; }
    .video img { max-width: 1024px; max-height: 600px; width: auto; height: auto; }
    .panel { background: #18202a; border: 1px solid #2a3441; border-radius: 6px; padding: 12px; }
    .row { display: flex; gap: 8px; margin-bottom: 10px; }
    button { flex: 1; height: 40px; border: 0; border-radius: 4px; color: #fff; background: #2563eb; font-size: 15px; cursor: pointer; }
    button.secondary { background: #475569; }
    button.danger { background: #b91c1c; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    .kv { display: grid; grid-template-columns: 110px 1fr; gap: 6px; font-size: 14px; margin: 10px 0; }
    pre { white-space: pre-wrap; word-break: break-word; max-height: 320px; overflow: auto; background: #0b1117; padding: 10px; border-radius: 4px; }
    .results { max-height: 420px; overflow: auto; display: grid; gap: 8px; }
    .result-card { background: #0b1117; border: 1px solid #2a3441; border-radius: 5px; padding: 8px; font-size: 13px; }
    .result-title { font-weight: 700; margin-bottom: 5px; color: #93c5fd; }
    .muted { color: #94a3b8; }
    .pill { display: inline-block; padding: 2px 6px; border-radius: 999px; background: #334155; margin-right: 4px; }
    h1 { font-size: 17px; margin: 0; }
    h2 { font-size: 15px; margin: 12px 0 8px; }
    .ok { color: #22c55e; }
    .warn { color: #facc15; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } .video { min-height: 320px; } }
  </style>
</head>
<body>
  <header><h1>药品识别控制台</h1></header>
  <main>
    <section class="video">
      <img id="video" src="/video_feed" alt="YOLO实时画面">
    </section>
    <aside class="panel">
      <div class="row">
        <button id="startBtn" onclick="postApi('/api/start')">启动检测</button>
        <button class="danger" onclick="postApi('/api/stop')">停止</button>
      </div>
      <div class="row">
        <button id="triggerBtn" onclick="postApi('/api/trigger')">触发识别</button>
        <button class="secondary" onclick="refreshAll()">刷新</button>
      </div>
      <h2>状态</h2>
      <div class="kv" id="statusBox"></div>
      <h2>最新识别结果</h2>
      <div class="results">
        <div class="result-card">
          <div class="result-title">药瓶输出结果</div>
          <div id="bottleBox" class="muted">暂无结果</div>
        </div>
        <div class="result-card">
          <div class="result-title">药袋标签病人匹配结果</div>
          <div id="bagBox" class="muted">暂无结果</div>
        </div>
        <div class="result-card">
          <div class="result-title">输液袋输出结果</div>
          <div id="infusionBox" class="muted">暂无结果</div>
        </div>
        <div class="result-card">
          <div class="result-title">最终查询匹配结果</div>
          <div id="matchBox" class="muted">暂无结果</div>
        </div>
      </div>
    </aside>
  </main>
  <script>
    async function postApi(path) {
      const resp = await fetch(path, {method: 'POST'});
      const data = await resp.json();
      await refreshAll();
      if (!resp.ok) alert(data.message || data.error || '请求失败');
    }
    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }
    function emptyText(items) {
      return !items || items.length === 0;
    }
    function renderList(id, html) {
      document.getElementById(id).innerHTML = html || '<span class="muted">暂无结果</span>';
    }
    function renderResult(payload) {
      if (!payload) {
        renderList('bottleBox', '');
        renderList('bagBox', '');
        renderList('infusionBox', '');
        renderList('matchBox', '');
        return;
      }
      const bottles = payload.bottles || [];
      renderList('bottleBox', bottles.map(item => `
        <div>
          <span class="pill">#${esc(item.index)}</span>
          <b>${esc(item.final_medicine || '未分类')}</b>
          <div>OCR：${esc(item.ocr_text || '')}</div>
          <div>候选：${esc((item.candidates || []).join('，'))}</div>
          <div>方式：${esc(item.classification_method || '')}，置信度：${Number(item.confidence || 0).toFixed(4)}</div>
          <div class="muted">${esc(item.decision_reason || item.status || '')}</div>
        </div>`).join('<hr>'));

      const bags = payload.bags || [];
      renderList('bagBox', bags.map(item => `
        <div>
          <span class="pill">#${esc(item.index)}</span>
          <b>${esc(item.patient_name || '未匹配到病人')}</b>
          <div>OCR：${esc(item.ocr_text || '')}</div>
          <div class="muted">状态：${esc(item.status || '')}</div>
        </div>`).join('<hr>'));

      const infusions = payload.infusions || payload.structured_infusions || [];
      renderList('infusionBox', infusions.map(item => `
        <div>
          <span class="pill">#${Number(item.index || 0) + 1}</span>
          <b>${esc(item.liquid || '未识别液体')}</b>
          <div>浓度：${esc(item.concentration || '')}，容量：${esc(item.volume || '')}</div>
          <div>OCR：${esc(item.raw_text || '')}</div>
          <div class="muted">状态：${esc(item.status || '')}</div>
        </div>`).join('<hr>'));

      const match = payload.database_match || {};
      const validation = match.validation || {};
      const medicines = payload.recognized_medicines || [];
      const statusText = {
        matched: '匹配正确',
        mismatch: '匹配不一致',
        batch_not_found: '病人批次不存在',
        skipped: '跳过数据库比对',
        error: '数据库比对异常'
      }[match.status] || (match.status || '暂无');
      renderList('matchBox', `
        <div><b>${esc(statusText)}</b></div>
        <div>病人：${esc(payload.patient_name || '未识别')}</div>
        <div>识别药品：${esc(medicines.join('，') || '无')}</div>
        <div>数据库药品：${esc((validation.actual || []).join('，') || '无')}</div>
        <div>缺少：${esc((validation.missing || []).join('，') || '无')}</div>
        <div>多余：${esc((validation.extra || []).join('，') || '无')}</div>
        <div class="muted">${esc(match.message || '')}</div>
      `);
    }
    async function refreshAll() {
      const status = await (await fetch('/api/status')).json();
      const rows = [
        ['状态', status.state],
        ['处理中', status.processing ? '是' : '否'],
        ['帧数', status.frame_count],
        ['FPS', Number(status.fps || 0).toFixed(1)],
        ['画面', status.frame_available ? '有' : '等待中'],
        ['触发次数', status.trigger_count],
        ['最近结果', status.last_result_at || '暂无'],
        ['错误', status.last_error || '无']
      ];
      document.getElementById('statusBox').innerHTML = rows.map(([k, v]) => `<div>${k}</div><div>${v}</div>`).join('');
      document.getElementById('startBtn').disabled = ['starting', 'loading_models', 'running'].includes(status.state);
      document.getElementById('triggerBtn').disabled = status.state !== 'running' || status.processing;
      const result = await (await fetch('/api/result')).json();
      renderResult(result.result);
    }
    setInterval(refreshAll, 1000);
    refreshAll();
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/api/health")
def health():
    return jsonify(ok=True, framework="flask")


@app.get("/api/status")
def status():
    return jsonify(runtime.snapshot_status())


@app.get("/api/result")
def result():
    return jsonify(result=runtime.get_result())


@app.post("/api/start")
def start():
    ok, message = runtime.start()
    return jsonify(ok=ok, message=message), (200 if ok else 409)


@app.post("/api/stop")
def stop():
    ok, message = runtime.stop()
    return jsonify(ok=ok, message=message)


@app.post("/api/trigger")
def trigger():
    ok, message = runtime.trigger()
    return jsonify(ok=ok, message=message), (200 if ok else 409)


@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = runtime.get_frame()
            if frame is None:
                time.sleep(0.2)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.08)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--start", action="store_true", help="start detection when Flask starts")
    args = parser.parse_args()
    if args.start:
        runtime.start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
