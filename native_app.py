# -*- coding: utf-8 -*-
import argparse
import contextlib
import gc
import io
import os
import queue
import subprocess
import threading
import time
import traceback

import cv2
import numpy as np
import pymysql
import yaml
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UI_WIDTH = 1024
UI_HEIGHT = 600


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no")


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


def build_bottle_visualization(bottles, width=1024, height=600):
    items = []
    for bottle in bottles or []:
        image = bottle.pop("det_visualization", None)
        if not isinstance(image, np.ndarray) or image.size == 0:
            continue
        items.append((bottle, image))
    if not items:
        return None

    panel_width = max(180, width // min(3, len(items)))
    panel_height = max(220, height // ((len(items) + 2) // 3))
    canvas = np.full((height, width, 3), (24, 30, 40), dtype=np.uint8)
    for index, (item, image) in enumerate(items):
        row, col = divmod(index, 3)
        x1, y1 = col * panel_width, row * panel_height
        x2, y2 = min(width, x1 + panel_width), min(height, y1 + panel_height)
        label_h = 42
        available_w = max(1, x2 - x1 - 8)
        available_h = max(1, y2 - y1 - label_h - 8)
        scale = min(available_w / image.shape[1], available_h / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        px = x1 + (x2 - x1 - resized.shape[1]) // 2
        py = y1 + label_h + (available_h - resized.shape[0]) // 2
        canvas[py:py + resized.shape[0], px:px + resized.shape[1]] = resized
        label = (
            f"Bottle {item.get('index')}: DET={item.get('det_region_count', 0)} "
            f"REC={item.get('rec_nonempty_count', 0)}"
        )
        cv2.putText(canvas, label, (x1 + 8, y1 + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 235, 242), 1)
    return canvas


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


def create_local_bottle_db_matcher(
        drug_matcher_cls,
        db_conn,
        local_drug_names,
        drug_table="drugs",
        drug_column="medicine_name",
        patient_table="patients",
        patient_column="name",
):
    class LocalBottleDbMatcher(drug_matcher_cls):
        def __init__(self):
            super().__init__(
                db_conn,
                drug_table=drug_table,
                drug_column=drug_column,
                patient_table=patient_table,
                patient_column=patient_column,
                cache_drugs=False,
            )
            self._drug_names = list(dict.fromkeys(name for name in local_drug_names if name))
            if not self._drug_names:
                raise ValueError("local bottle template drug names are empty")
            self._load_patient_names()

    return LocalBottleDbMatcher()


def release_models(classifier, ocr_recognizers):
    for recognizer in ocr_recognizers:
        release = getattr(recognizer, "release", None)
        if callable(release):
            try:
                release()
            except Exception:
                pass
    if classifier is not None:
        release = getattr(classifier, "release", None)
        if callable(release):
            try:
                release()
            except Exception:
                pass
    gc.collect()


def find_font_path():
    env_path = os.environ.get("YILIAO_UI_FONT", "").strip()
    candidates = [
        env_path,
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    try:
        output = subprocess.check_output(
            ["fc-match", "-f", "%{file}", "Noto Sans CJK SC"],
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        )
        path = output.decode("utf-8", errors="ignore").strip()
        if path and os.path.exists(path):
            return path
    except Exception:
        pass
    return None


FONT_PATH = find_font_path()
FONT_CACHE = {}


def get_font(size):
    key = int(size)
    if key not in FONT_CACHE:
        if FONT_PATH:
            FONT_CACHE[key] = ImageFont.truetype(FONT_PATH, key)
        else:
            FONT_CACHE[key] = ImageFont.load_default()
    return FONT_CACHE[key]


def text_width(text, font):
    bbox = font.getbbox(str(text))
    return bbox[2] - bbox[0]


def wrap_paragraph(text, font, max_width):
    text = str(text)
    if not text:
        return [""]
    lines = []
    current = ""
    for char in text:
        if char == "\n":
            lines.append(current)
            current = ""
            continue
        candidate = current + char
        if current and text_width(candidate, font) > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    lines.append(current)
    return lines


def format_infusion_line(item):
    index = int(item.get("index", 0)) + 1
    return (
        f"输液袋 {index}: 液体={item.get('liquid')}, "
        f"浓度={item.get('concentration')}, 容量={item.get('volume')}, "
        f"状态={item.get('status')}"
    )


def format_result(payload):
    if not payload:
        return "暂无识别结果"

    lines = []
    lines.append("【最终查询匹配结果】")
    match = payload.get("database_match") or {}
    validation = match.get("validation") or {}
    lines.append(f"匹配状态: {match.get('status') or '无'}")
    if match.get("message"):
        lines.append(f"说明: {match.get('message')}")
    if match.get("batch_id") is not None:
        lines.append(f"批次ID: {match.get('batch_id')}")
    lines.append("")
    patient_names = payload.get("patient_names") or []
    patient_name = payload.get("patient_name")
    lines.append(f"患者: {patient_names or ([patient_name] if patient_name else [])}")
    lines.append(f"识别药品: {payload.get('recognized_medicines') or []}")
    lines.append(f"数据库药品: {validation.get('actual') or []}")
    lines.append(f"缺少: {validation.get('missing') or []}")
    lines.append(f"多余: {validation.get('extra') or []}")
    lines.append("")

    lines.append("【药瓶输出结果】")
    bottles = payload.get("bottles") or []
    if bottles:
        for item in bottles:
            lines.append(f"药瓶 {item.get('index')}: {item.get('final_medicine')}")
            lines.append(f"  OCR: {item.get('ocr_text') or ''}")
            lines.append(
                f"  DET文字区域: {item.get('det_region_count', 0)}, "
                f"REC非空结果: {item.get('rec_nonempty_count', 0)}"
            )
            if item.get("ocr_empty_reason"):
                lines.append(f"  OCR空结果原因: {item.get('ocr_empty_reason')}")
            candidates = item.get("candidates") or []
            if candidates:
                lines.append(f"  候选: {candidates}")
            lines.append(
                f"  方式: {item.get('classification_method') or ''}, "
                f"置信度: {float(item.get('confidence') or 0.0):.4f}"
            )
            if item.get("decision_reason") or item.get("status"):
                lines.append(f"  决策: {item.get('decision_reason') or item.get('status')}")
    else:
        lines.append("无")
    lines.append("")

    lines.append("【药袋标签病人匹配结果】")
    bags = payload.get("bags") or []
    if bags:
        for item in bags:
            lines.append(f"药袋 {item.get('index')}: 患者={item.get('patient_name')}")
            lines.append(f"  OCR: {item.get('ocr_text') or ''}")
            if item.get("status"):
                lines.append(f"  状态: {item.get('status')}")
    else:
        lines.append("无")
    lines.append("")

    lines.append("【输液袋输出结果】")
    infusions = payload.get("infusions") or payload.get("structured_infusions") or []
    if infusions:
        for item in infusions:
            lines.append(format_infusion_line(item))
            if item.get("raw_text"):
                lines.append(f"  OCR: {item.get('raw_text')}")
    else:
        lines.append("无")
    lines.append("")

    timing = payload.get("timing_sec") or {}
    if timing:
        lines.append("【用时】")
        for key, value in timing.items():
            lines.append(f"{key}: {value}s")
    return "\n".join(lines)


class NativeRuntime:
    def __init__(self, event_queue):
        self.event_queue = event_queue
        self.lock = threading.Lock()
        self.thread = None
        self.stop_event = threading.Event()
        self.trigger_event = threading.Event()
        self.last_result = None
        self.latest_detection_frame = None
        self.latest_bottle_frame = None
        self._auto_trigger_active = False
        self.requested_trigger_mode = "manual"
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

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            self.stop_event.clear()
            self.trigger_event.clear()
            self.last_result = None
            self.latest_detection_frame = None
            self.latest_bottle_frame = None
            self._auto_trigger_active = False
            self.status.update({
                "state": "starting",
                "processing": False,
                "frame_count": 0,
                "fps": 0.0,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_result_at": None,
                "last_error": None,
            })
            self.thread = threading.Thread(target=self._run_detection, daemon=True)
            self.thread.start()
        self._emit("status", dict(self.status))

    def stop(self):
        self.stop_event.set()
        with self.lock:
            self.status["state"] = "stopping"
        self._emit("status", dict(self.status))

    def trigger(self):
        with self.lock:
            alive = bool(self.thread and self.thread.is_alive())
            if not alive:
                return False, "检测线程未运行"
            if self.status.get("processing"):
                return False, "正在识别中"
            self.last_result = None
            self.status["last_result_at"] = None
            self.status["trigger_count"] = int(self.status.get("trigger_count", 0)) + 1
            self.status["processing"] = True
        self.trigger_event.set()
        self._emit("status", self.snapshot_status())
        return True, "已触发识别"

    def set_trigger_mode(self, mode):
        mode = str(mode).lower()
        if mode not in ("manual", "auto"):
            return
        with self.lock:
            self.requested_trigger_mode = mode
            self.status["trigger_mode"] = mode
            status = dict(self.status)
        self._emit("status", status)

    def get_trigger_mode(self):
        with self.lock:
            return self.requested_trigger_mode

    def snapshot_status(self):
        with self.lock:
            data = dict(self.status)
            data["thread_alive"] = bool(self.thread and self.thread.is_alive())
            data["result_available"] = self.last_result is not None
            return data

    def update_status(self, data):
        auto_trigger_started = False
        with self.lock:
            data = dict(data)
            data["trigger_mode"] = self.requested_trigger_mode
            is_auto_trigger = data.get("trigger") == "basket_auto" and bool(data.get("processing"))
            if is_auto_trigger and not self._auto_trigger_active:
                self.status["trigger_count"] = int(self.status.get("trigger_count", 0)) + 1
                auto_trigger_started = True
            self._auto_trigger_active = is_auto_trigger
            self.status.update(jsonable(data))
            status = dict(self.status)
            status["thread_alive"] = bool(self.thread and self.thread.is_alive())
        self._emit("status", status)
        if auto_trigger_started:
            self._emit("auto_trigger", {
                "trigger_count": status.get("trigger_count", 0),
                "basket_area_ratio": status.get("basket_area_ratio", 0.0),
                "basket_sharpness": status.get("basket_sharpness", 0.0),
            })

    def update_result(self, payload):
        bottle_frame = build_bottle_visualization(payload.get("bottles") or [])
        payload = jsonable(payload)
        with self.lock:
            self.last_result = payload
            self.latest_bottle_frame = bottle_frame
            self.status["last_result_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.status["processing"] = False
            status = dict(self.status)
        self._emit("result", payload)
        self._emit("status", status)

    def update_detection_frame(self, frame):
        if frame is None:
            return
        with self.lock:
            self.latest_detection_frame = frame.copy()

    def snapshot_detection_frame(self):
        with self.lock:
            return self.latest_detection_frame

    def snapshot_bottle_frame(self):
        with self.lock:
            return self.latest_bottle_frame

    def _emit(self, event_type, data):
        self.event_queue.put((event_type, data))

    def _run_detection(self):
        conn = None
        classifier = None
        ocr_recognizers = []
        try:
            self.update_status({"state": "loading_models", "processing": False})
            self._set_default_env()
            from run_realtime_detection_yolo_new_3 import run_realtime_detection
            from src.identification.DrugMatcher import DrugMatcher
            from src.identification.Recog import PharmaceuticalBottleClassifier
            from src.identification.rknn_ocr_adapter import RknnOCRRecognizer

            with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            runtime_config = config.get("runtime", {}) or {}
            self.set_trigger_mode(runtime_config.get("trigger_mode", "manual"))
            recognition_workers = int(os.environ.get(
                "YILIAO_RECOGNITION_WORKERS",
                runtime_config.get("recognition_workers", 1),
            ))
            quiet_ocr = env_bool("YILIAO_QUIET_OCR", bool(runtime_config.get("quiet_ocr", True)))
            ocr_instance_count = int(os.environ.get(
                "YILIAO_OCR_INSTANCES",
                runtime_config.get("ocr_instances", 1),
            ))

            init_log_stream = io.StringIO() if quiet_ocr else None
            with contextlib.redirect_stdout(init_log_stream) if init_log_stream is not None else contextlib.nullcontext():
                classifier = PharmaceuticalBottleClassifier(db_conn=None, device="npu")
            drug_names = classifier.get_cached_names()
            conn = init_db(config.get("db_config", {}))
            tables = config["table_config"]
            drug_matcher = create_local_bottle_db_matcher(
                DrugMatcher,
                conn,
                drug_names,
                drug_table=tables["drug_table"],
                drug_column=tables["drug_column"],
                patient_table=tables["patient_table"],
                patient_column=tables["patient_column"],
            )

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
                yolo_model_path=config["model"].get("yolo_rknn_path", "./model_yolo_0602.rknn"),
                yolo_input_size=int(config["model"].get("yolo_input_size", 640)),
                basket_model_path=config["model"].get("basket_rknn_path", "./model_bask_0609_n.rknn"),
                basket_input_size=int(config["model"].get("basket_input_size", 640)),
                basket_area_threshold=float(runtime_config.get("basket_area_threshold", 0.40)),
                basket_sharpness_threshold=float(runtime_config.get("basket_sharpness_threshold", 55.0)),
                basket_capture_timeout=float(runtime_config.get("basket_capture_timeout", 10.0)),
                basket_resume_delay=float(runtime_config.get("basket_resume_delay", 1.0)),
                trigger_interval=999999.0,
                recognition_workers=recognition_workers,
                classifier_thread_safe=False,
                quiet_ocr=quiet_ocr,
                headless=True,
                trigger_mode=runtime_config.get("trigger_mode", "manual"),
                trigger_mode_getter=self.get_trigger_mode,
                manual_trigger_event=self.trigger_event,
                stop_event=self.stop_event,
                status_callback=self.update_status,
                result_callback=self.update_result,
                frame_callback=self.update_detection_frame,
            )
        except Exception as exc:
            self.update_status({
                "state": "error",
                "processing": False,
                "last_error": repr(exc),
            })
            self._emit("error", traceback.format_exc())
        finally:
            release_models(classifier, ocr_recognizers)
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            with self.lock:
                if self.status.get("state") != "error":
                    self.status["state"] = "stopped"
                self.status["processing"] = False
                status = dict(self.status)
            self._emit("status", status)

    @staticmethod
    def _set_default_env():
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        os.environ.setdefault("YILIAO_MIXED_OCR_BATCH", "1")
        os.environ.setdefault("YILIAO_FEATURE_CACHE", "./single_image_feature_cache.pkl")
        os.environ.setdefault("YILIAO_FEATURE_ROOT", os.path.join(PROJECT_ROOT, "src", "identification", "feat_data"))
        os.environ.setdefault("YILIAO_VERBOSE_RUNTIME", "0")
        os.environ.setdefault("YILIAO_RUNTIME_LOGS", "0")
        os.environ.setdefault("YILIAO_COLLECT_TIMING", "0")
        os.environ.setdefault("YILIAO_QUIET_OCR", "0")
        os.environ.setdefault("YILIAO_CLS_MODEL", "./model_cls_bs32.rknn")
        os.environ.setdefault("YILIAO_CLS_BATCH_SIZE", "32")


class NativeRecognitionApp:
    def __init__(self, autostart=True):
        self.window_name = "YiLiao Native Recognition"
        self.detection_window_name = "Basket Realtime Detection"
        self.trigger_popup_window_name = "Automatic Trigger"
        self.bottle_window_name = "Bottle OCR Text Regions"
        self.events = queue.Queue()
        self.runtime = NativeRuntime(self.events)
        self.status = dict(self.runtime.status)
        self.sidebar_rect = (18, 76, 306, UI_HEIGHT - 22)
        self.result_rect = (324, 76, UI_WIDTH - 18, UI_HEIGHT - 22)
        self.manual_mode_rect = (38, 116, 158, 154)
        self.auto_mode_rect = (166, 116, 286, 154)
        self.button_rect = (38, 166, 286, 222)
        self.result_text = ""
        self.result_lines = []
        self.scroll = 0
        self.cached_frame = None
        self.dirty = True
        self.running = True
        self.trigger_popup_until = 0.0
        self.set_result_text("检测线程正在启动...\n等待状态变为运行中后点击触发识别。")
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, UI_WIDTH, UI_HEIGHT)
        cv2.namedWindow(self.detection_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.detection_window_name, UI_WIDTH, UI_HEIGHT)
        cv2.namedWindow(self.bottle_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.bottle_window_name, UI_WIDTH, UI_HEIGHT)
        cv2.imshow(self.window_name, self.render())
        cv2.imshow(self.detection_window_name, np.zeros((UI_HEIGHT, UI_WIDTH, 3), dtype=np.uint8))
        cv2.imshow(self.bottle_window_name, np.zeros((UI_HEIGHT, UI_WIDTH, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        if autostart:
            self.runtime.start()

    def run(self):
        while self.running:
            self.drain_events()
            cv2.imshow(self.window_name, self.render())
            detection_frame = self.runtime.snapshot_detection_frame()
            if detection_frame is not None:
                cv2.imshow(self.detection_window_name, detection_frame)
            bottle_frame = self.runtime.snapshot_bottle_frame()
            if bottle_frame is not None:
                cv2.imshow(self.bottle_window_name, bottle_frame)
            if self.trigger_popup_until and time.time() >= self.trigger_popup_until:
                try:
                    cv2.destroyWindow(self.trigger_popup_window_name)
                except cv2.error:
                    pass
                self.trigger_popup_until = 0.0
            key = cv2.waitKey(35) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (13, 32):
                self.trigger()
            elif key in (ord("j"), 82):
                self.set_scroll(self.scroll - 1)
            elif key in (ord("k"), 84):
                self.set_scroll(self.scroll + 1)
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break
        self.runtime.stop()
        cv2.destroyAllWindows()

    def trigger(self):
        if self.runtime.get_trigger_mode() != "manual":
            self.set_result_text("Auto mode is monitoring the basket.")
            return
        ok, message = self.runtime.trigger()
        self.set_result_text("已触发识别，正在等待最终匹配结果..." if ok else message)

    def on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.point_in_rect(x, y, self.manual_mode_rect):
                self.runtime.set_trigger_mode("manual")
            elif self.point_in_rect(x, y, self.auto_mode_rect):
                self.runtime.set_trigger_mode("auto")
            elif self.point_in_button(x, y):
                self.trigger()
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.set_scroll(self.scroll - 2 if flags > 0 else self.scroll + 2)

    def point_in_button(self, x, y):
        x1, y1, x2, y2 = self.button_rect
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def point_in_rect(x, y, rect):
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def drain_events(self):
        try:
            while True:
                event_type, data = self.events.get_nowait()
                if event_type == "status" and data != self.status:
                    self.status = data
                    self.dirty = True
                elif event_type == "result":
                    self.set_result_text(format_result(data))
                elif event_type == "auto_trigger":
                    self.show_auto_trigger_popup(data)
                elif event_type == "error":
                    self.set_result_text(data)
        except queue.Empty:
            pass

    def show_auto_trigger_popup(self, data):
        canvas = np.full((220, 520, 3), (248, 250, 252), dtype=np.uint8)
        image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((12, 12, 508, 208), radius=10, fill=(255, 255, 255), outline=(37, 99, 235), width=3)
        draw.text((34, 35), "自动触发识别", font=get_font(28), fill=(15, 23, 42))
        draw.text((34, 88), "篮子触发后已等待五秒，正在执行识别匹配。", font=get_font(17), fill=(51, 65, 85))
        detail = (
            f"第 {int(data.get('trigger_count', 0))} 次触发  |  "
            f"面积 {float(data.get('basket_area_ratio', 0.0)):.1%}  |  "
            f"清晰度 {float(data.get('basket_sharpness', 0.0)):.0f}  |  "
            "等待完成后开始识别"
        )
        draw.text((34, 142), detail, font=get_font(16), fill=(37, 99, 235))
        popup = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        cv2.namedWindow(self.trigger_popup_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.trigger_popup_window_name, 520, 220)
        cv2.imshow(self.trigger_popup_window_name, popup)
        self.trigger_popup_until = time.time() + 2.5

    def set_result_text(self, text):
        self.result_text = str(text or "")
        self.scroll = 0
        self.result_lines = self.wrap_result_text()
        self.dirty = True

    def wrap_result_text(self):
        x1, _y1, x2, _y2 = self.result_rect
        max_width = x2 - x1 - 42
        font = get_font(15)
        wrapped = []
        for line in self.result_text.splitlines():
            wrapped.extend(wrap_paragraph(line, font, max_width) if line else [""])
        return wrapped or ["暂无识别结果"]

    def visible_line_count(self):
        _x1, y1, _x2, y2 = self.result_rect
        return max(1, (y2 - y1 - 84) // 22)

    def max_scroll(self):
        return max(0, len(self.result_lines) - self.visible_line_count())

    def set_scroll(self, value):
        next_scroll = min(max(0, int(value)), self.max_scroll())
        if next_scroll != self.scroll:
            self.scroll = next_scroll
            self.dirty = True

    def render(self):
        if self.cached_frame is not None and not self.dirty:
            return self.cached_frame
        canvas = np.full((UI_HEIGHT, UI_WIDTH, 3), (241, 245, 249), dtype=np.uint8)
        image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        self.draw_header(draw)
        self.draw_sidebar(draw)
        self.draw_mode_control(draw)
        self.draw_button(draw)
        self.draw_result(draw)
        self.cached_frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        self.dirty = False
        return self.cached_frame

    def draw_header(self, draw):
        draw.rectangle((0, 0, UI_WIDTH, 58), fill=(15, 23, 42))
        draw.text((24, 13), "药品识别控制台", font=get_font(24), fill=(248, 250, 252))
        draw.text((250, 20), "原生本地模式", font=get_font(15), fill=(148, 163, 184))

    def draw_sidebar(self, draw):
        x1, y1, x2, y2 = self.sidebar_rect
        draw.rounded_rectangle((x1, y1, x2, y2), radius=10, fill=(255, 255, 255), outline=(226, 232, 240))
        draw.text((x1 + 20, y1 + 20), "操作", font=get_font(20), fill=(15, 23, 42))
        self.draw_status_cards(draw)

    def draw_status_cards(self, draw):
        state = self.status.get("state", "idle")
        processing = "是" if self.status.get("processing") else "否"
        items = [
            ("Mode", self.status.get("trigger_mode", "manual"), (37, 99, 235)),
            ("Basket area", f"{float(self.status.get('basket_area_ratio') or 0.0) * 100:.1f}%", (30, 64, 175)),
            ("Sharpness", f"{float(self.status.get('basket_sharpness') or 0.0):.0f}", (30, 64, 175)),
            ("Capture", str(int(self.status.get("basket_capture_count") or 0)), (30, 64, 175)),
            ("状态", self.state_label(state), self.state_color(state)),
            ("处理中", processing, (217, 119, 6) if processing == "是" else (22, 163, 74)),
            ("帧数", str(self.status.get("frame_count", 0)), (30, 64, 175)),
            ("FPS", f"{float(self.status.get('fps') or 0.0):.1f}", (30, 64, 175)),
            ("触发次数", str(self.status.get("trigger_count", 0)), (30, 64, 175)),
            ("最近结果", self.status.get("last_result_at") or "暂无", (71, 85, 105)),
        ]
        x = self.sidebar_rect[0] + 20
        y = 210
        width = self.sidebar_rect[2] - self.sidebar_rect[0] - 40
        for label, value, color in items[:7]:
            self.draw_metric(draw, x, y, width, label, value, color)
            y += 48
        last_error = self.status.get("last_error")
        if last_error:
            draw.rounded_rectangle((x, y, x + width, y + 86), radius=7, fill=(254, 242, 242), outline=(254, 202, 202))
            draw.text((x + 12, y + 10), "错误", font=get_font(14), fill=(153, 27, 27))
            for idx, line in enumerate(wrap_paragraph(last_error, get_font(13), width - 24)[:2]):
                draw.text((x + 12, y + 34 + idx * 20), line, font=get_font(13), fill=(127, 29, 29))

    def draw_metric(self, draw, x, y, width, label, value, color):
        draw.rounded_rectangle((x, y, x + width, y + 38), radius=7, fill=(248, 250, 252), outline=(226, 232, 240))
        draw.text((x + 12, y + 9), label, font=get_font(14), fill=(100, 116, 139))
        value_font = get_font(14)
        value_text = str(value)
        max_value_width = width - 102
        while text_width(value_text, value_font) > max_value_width and len(value_text) > 3:
            value_text = value_text[:-2] + "…"
        draw.text((x + width - text_width(value_text, value_font) - 12, y + 9), value_text, font=value_font, fill=color)

    @staticmethod
    def state_label(state):
        return {
            "idle": "空闲",
            "starting": "启动中",
            "loading_models": "加载模型",
            "running": "运行中",
            "stopping": "停止中",
            "stopped": "已停止",
            "error": "异常",
        }.get(state, str(state))

    @staticmethod
    def state_color(state):
        return {
            "running": (22, 163, 74),
            "loading_models": (217, 119, 6),
            "starting": (37, 99, 235),
            "error": (220, 38, 38),
            "stopped": (100, 116, 139),
            "idle": (100, 116, 139),
        }.get(state, (71, 85, 105))

    def draw_button(self, draw):
        state = self.status.get("state", "idle")
        processing = bool(self.status.get("processing"))
        enabled = (
            state == "running"
            and not processing
            and self.status.get("trigger_mode", "manual") == "manual"
        )
        color = (37, 99, 235) if enabled else (148, 163, 184)
        x1, y1, x2, y2 = self.button_rect
        draw.rounded_rectangle((x1, y1, x2, y2), radius=10, fill=color)
        label = "触发识别" if enabled else ("识别中" if processing else "加载中")
        font = get_font(25)
        draw.text(((x1 + x2 - text_width(label, font)) / 2, y1 + 17), label, font=font, fill=(255, 255, 255))

    def draw_mode_control(self, draw):
        mode = self.status.get("trigger_mode", "manual")
        for value, rect, label in (
            ("manual", self.manual_mode_rect, "Manual"),
            ("auto", self.auto_mode_rect, "Auto"),
        ):
            selected = mode == value
            draw.rounded_rectangle(
                rect,
                radius=7,
                fill=(37, 99, 235) if selected else (226, 232, 240),
                outline=(37, 99, 235) if selected else (203, 213, 225),
            )
            font = get_font(15)
            x1, y1, x2, y2 = rect
            draw.text(
                ((x1 + x2 - text_width(label, font)) / 2, y1 + 9),
                label,
                font=font,
                fill=(255, 255, 255) if selected else (51, 65, 85),
            )

    def draw_result(self, draw):
        x1, y1, x2, y2 = self.result_rect
        draw.rounded_rectangle((x1, y1, x2, y2), radius=10, fill=(255, 255, 255), outline=(226, 232, 240))
        draw.text((x1 + 18, y1 + 16), "最终匹配与识别输出", font=get_font(20), fill=(15, 23, 42))
        draw.line((x1 + 18, y1 + 52, x2 - 18, y1 + 52), fill=(226, 232, 240), width=1)
        content_x = x1 + 18
        content_y = y1 + 66
        line_height = 22
        visible_count = self.visible_line_count()
        self.scroll = min(max(0, self.scroll), self.max_scroll())
        visible = self.result_lines[self.scroll:self.scroll + visible_count]
        for idx, line in enumerate(visible):
            draw.text((content_x, content_y + idx * line_height), line, font=get_font(15), fill=(17, 24, 39))
        max_scroll = self.max_scroll()
        if max_scroll > 0:
            track_x = x2 - 14
            track_y1 = content_y
            track_y2 = y2 - 18
            draw.rounded_rectangle((track_x, track_y1, track_x + 4, track_y2), radius=2, fill=(226, 232, 240))
            thumb_h = max(28, int((visible_count / max(visible_count, len(self.result_lines))) * (track_y2 - track_y1)))
            thumb_y = track_y1 + int((self.scroll / max_scroll) * max(1, track_y2 - track_y1 - thumb_h))
            draw.rounded_rectangle((track_x, thumb_y, track_x + 4, thumb_y + thumb_h), radius=2, fill=(100, 116, 139))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-autostart", action="store_true", help="open UI without starting detection thread")
    args = parser.parse_args()
    app = NativeRecognitionApp(autostart=not args.no_autostart)
    app.run()


if __name__ == "__main__":
    main()
