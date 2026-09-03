# -*- coding: utf-8 -*-
import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

import cv2
import numpy as np
import yaml
from rknnlite.api import RKNNLite


DEFAULT_PROJECT_ROOT = (
    "/home/forlinx/Models/AnotherYiliao/shibie/"
    "YiLiaoShiBie_0521/YiLiaoShiBie"
)
DEFAULT_REC_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "model_ocr_bs16.rknn",
)
NPU_LOAD_PATH = "/sys/class/devfreq/fdab0000.npu/load"
NPU_FREQ_PATH = "/sys/class/devfreq/fdab0000.npu/cur_freq"
RKNPU_DEBUG_LOAD_CMD = "echo forlinx | sudo -S cat /sys/kernel/debug/rknpu/load"
RKNPU_DEBUG_LOAD_STREAM_CMD = (
    "printf 'forlinx\\n' | sudo -S sh -c "
    "'while true; do "
    "printf \"TS=%s \" \"$(date +%s.%N)\"; "
    "cat /sys/kernel/debug/rknpu/load; "
    "sleep {interval}; "
    "done'"
)


class NpuSampler:
    def __init__(self, csv_path, interval=0.05):
        self.csv_path = csv_path
        self.interval = float(interval)
        self.stage = "idle"
        self._stage_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._proc = None

    def set_stage(self, stage):
        with self._stage_lock:
            self.stage = str(stage)

    def _read_text(self, path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
        except Exception:
            return ""

    def _read_debug_load(self):
        try:
            completed = subprocess.run(
                RKNPU_DEBUG_LOAD_CMD,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=0.5,
            )
            return (completed.stdout or "").strip()
        except Exception:
            return ""

    def _open_debug_load_stream(self):
        interval = max(0.005, min(self.interval, 1.0))
        try:
            return subprocess.Popen(
                RKNPU_DEBUG_LOAD_STREAM_CMD.format(interval=f"{interval:.3f}"),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception:
            return None

    @staticmethod
    def _parse_load(raw):
        if not raw:
            return None
        match = re.search(r"(-?\d+(?:\.\d+)?)", raw)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _parse_core_loads(raw):
        loads = {}
        for core, value in re.findall(r"Core\s*(\d+)\s*:\s*(\d+(?:\.\d+)?)\s*%", raw or ""):
            loads[int(core)] = float(value)
        values = [loads.get(i) for i in range(3)]
        valid = [v for v in values if v is not None]
        avg_load = sum(valid) / len(valid) if valid else None
        max_load = max(valid) if valid else None
        return values, avg_load, max_load

    def _run(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_sec",
                "wall_time",
                "stage",
                "rknpu_load_raw",
                "core0_pct",
                "core1_pct",
                "core2_pct",
                "core_avg_pct",
                "core_max_pct",
                "devfreq_load_raw",
                "devfreq_load_pct",
                "npu_cur_freq_hz",
            ])
            t0 = time.perf_counter()
            stream_rows = 0
            self._proc = self._open_debug_load_stream()
            if self._proc is not None and self._proc.stdout is not None:
                while not self._stop.is_set():
                    debug_load_raw = self._proc.stdout.readline()
                    if not debug_load_raw:
                        if self._proc.poll() is not None:
                            break
                        continue
                    self._write_sample_row(writer, t0, debug_load_raw.strip())
                    f.flush()
                    stream_rows += 1

            if stream_rows == 0:
                while not self._stop.is_set():
                    debug_load_raw = self._read_debug_load()
                    self._write_sample_row(writer, t0, debug_load_raw)
                    f.flush()
                    self._stop.wait(self.interval)

            if self._proc is not None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                self._proc = None

    def _write_sample_row(self, writer, t0, debug_load_raw):
        core_loads, core_avg, core_max = self._parse_core_loads(debug_load_raw)
        devfreq_load_raw = self._read_text(NPU_LOAD_PATH)
        freq_raw = self._read_text(NPU_FREQ_PATH)
        devfreq_load_pct = self._parse_load(devfreq_load_raw)
        with self._stage_lock:
            stage = self.stage
        writer.writerow([
            f"{time.perf_counter() - t0:.6f}",
            time.strftime("%Y-%m-%d %H:%M:%S"),
            stage,
            debug_load_raw,
            "" if core_loads[0] is None else f"{core_loads[0]:.3f}",
            "" if core_loads[1] is None else f"{core_loads[1]:.3f}",
            "" if core_loads[2] is None else f"{core_loads[2]:.3f}",
            "" if core_avg is None else f"{core_avg:.3f}",
            "" if core_max is None else f"{core_max:.3f}",
            devfreq_load_raw,
            "" if devfreq_load_pct is None else f"{devfreq_load_pct:.3f}",
            freq_raw,
        ])

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)


@contextmanager
def stage(sampler, name, timings):
    sampler.set_stage(name)
    start = time.perf_counter()
    print(f"[stage] {name} start", flush=True)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings.append((name, elapsed))
        print(f"[stage] {name} done: {elapsed:.3f}s", flush=True)
        sampler.set_stage("idle")
        time.sleep(0.15)


def load_config(project_root):
    config_path = os.path.join(project_root, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def capture_frame(project_root, timeout_sec=6.0):
    config = load_config(project_root)
    rtsp = config.get("RTSP", {})
    url = (
        f"rtsp://{rtsp.get('username')}:{rtsp.get('password')}"
        f"@{rtsp.get('ip_address')}:{rtsp.get('port')}"
        f"/Streaming/Channels/{rtsp.get('channel')}"
    )
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(url)
    deadline = time.time() + timeout_sec
    frame = None
    while time.time() < deadline:
        ok, img = cap.read()
        if ok and img is not None and img.size > 0:
            frame = img
            break
        time.sleep(0.05)
    cap.release()
    if frame is not None:
        return frame

    frame = np.full((1080, 1920, 3), 225, dtype=np.uint8)
    cv2.rectangle(frame, (280, 190), (1640, 900), (190, 190, 190), 8)
    cv2.putText(frame, "NPU LOAD TEST", (420, 440), cv2.FONT_HERSHEY_SIMPLEX, 3, (30, 30, 30), 7)
    cv2.putText(frame, "Glucose Injection 10% 100ml", (420, 560), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
    return frame


def make_crops(frame, predictions, count=32):
    crops = []
    h, w = frame.shape[:2]
    for pred in predictions or []:
        if len(pred) < 4:
            continue
        x1, y1, x2, y2 = [int(max(0, v)) for v in pred[:4]]
        x1, x2 = min(x1, w - 1), min(x2, w)
        y1, y2 = min(y1, h - 1), min(y2, h)
        if x2 > x1 + 8 and y2 > y1 + 8:
            crops.append(frame[y1:y2, x1:x2].copy())
    if not crops:
        specs = [
            (0.08, 0.12, 0.46, 0.50),
            (0.52, 0.12, 0.92, 0.50),
            (0.14, 0.54, 0.46, 0.88),
            (0.52, 0.54, 0.88, 0.88),
        ]
        for x1r, y1r, x2r, y2r in specs:
            x1, y1 = int(w * x1r), int(h * y1r)
            x2, y2 = int(w * x2r), int(h * y2r)
            crops.append(frame[y1:y2, x1:x2].copy())
    base = [c for c in crops if c is not None and c.size > 0]
    while len(crops) < count and base:
        crops.append(base[len(crops) % len(base)].copy())
    return crops[:count]


def make_rec_regions(count=32):
    regions = []
    for idx in range(count):
        img = np.full((48, 320, 3), 255, dtype=np.uint8)
        text = f"TEST OCR {idx:02d} 10% 100ml"
        cv2.putText(img, text, (8, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        regions.append(img)
    return regions


def summarize_csv(csv_path):
    values_by_stage = {}
    per_core_by_stage = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stage_name = row.get("stage") or ""
            value = row.get("core_avg_pct") or ""
            try:
                load = float(value)
            except ValueError:
                continue
            values_by_stage.setdefault(stage_name, []).append(load)
            per_core = []
            for key in ("core0_pct", "core1_pct", "core2_pct"):
                try:
                    per_core.append(float(row.get(key) or ""))
                except ValueError:
                    per_core.append(None)
            per_core_by_stage.setdefault(stage_name, []).append(per_core)
    lines = []
    for stage_name, values in sorted(values_by_stage.items()):
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float32)
        core_texts = []
        core_rows = per_core_by_stage.get(stage_name, [])
        for idx in range(3):
            core_values = [row[idx] for row in core_rows if row[idx] is not None]
            if core_values:
                core_arr = np.asarray(core_values, dtype=np.float32)
                core_texts.append(f"core{idx}_avg={core_arr.mean():.2f}, core{idx}_max={core_arr.max():.2f}")
        lines.append(
            f"{stage_name}: samples={len(values)}, "
            f"core_avg={arr.mean():.2f}, core_avg_max={arr.max():.2f}, "
            f"core_avg_min={arr.min():.2f}"
            + (", " + ", ".join(core_texts) if core_texts else "")
        )
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output-dir", default="/home/forlinx/Models/AnotherYiliao/npu_load_test_0627/results")
    parser.add_argument("--rec-model", default=DEFAULT_REC_MODEL)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--sample-interval", type=float, default=0.10)
    parser.add_argument("--crop-count", type=int, default=32)
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma separated stages: all,yolo,ocr_cls,ocr_det,ocr_rec,full_ocr_task",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    sys.path.insert(0, project_root)
    os.chdir(project_root)
    os.environ.setdefault("YILIAO_RUNTIME_LOGS", "0")
    os.environ.setdefault("YILIAO_VERBOSE_RUNTIME", "0")
    os.environ.setdefault("YILIAO_COLLECT_TIMING", "0")

    from src.segmentation.segmenter_yolo import YOLOTileProcessor
    from src.identification.rknn_ocr_adapter import (
        BOTTLE_LINE_PADDING,
        RknnOCRRecognizer,
        tight_ocr,
    )

    config = load_config(project_root)
    runtime_tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, runtime_tag)
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "npu_load.csv")
    summary_path = os.path.join(run_dir, "summary.txt")

    sampler = NpuSampler(csv_path, interval=args.sample_interval)
    timings = []
    sampler.start()

    yolo = None
    ocr = None
    predictions = []
    try:
        frame = capture_frame(project_root)
        cv2.imwrite(os.path.join(run_dir, "input_frame.jpg"), frame)

        model_cfg = config.get("model", {})
        requested = {item.strip() for item in str(args.stages).split(",") if item.strip()}
        if not requested or "all" in requested:
            requested = {"yolo", "ocr_cls", "ocr_det", "ocr_rec", "full_ocr_task"}
        need_yolo = "yolo" in requested
        need_ocr = bool({"ocr_cls", "ocr_det", "ocr_rec", "full_ocr_task"} & requested)
        with stage(sampler, "model_init", timings):
            if need_yolo:
                yolo = YOLOTileProcessor(
                    model_path=model_cfg.get("yolo_rknn_path", "./model_yolo_0615.rknn"),
                    device="npu",
                    tile_size=int(model_cfg.get("yolo_input_size", 640)),
                    input_size=int(model_cfg.get("yolo_input_size", 640)),
                    overlap=64,
                    conf_thres=0.45,
                    iou_thres=0.85,
                    batch_size=1,
                    contain_thres=0.92,
                    class_conf_thres={0: 0.35, 1: 0.25, 2: 0.32},
                    class_iou_thres={0: 0.65, 1: 0.65, 2: 0.45},
                )
            if need_ocr:
                ocr = RknnOCRRecognizer(
                    det_model_path=os.path.join(project_root, "model_det_bs16.rknn"),
                    rec_model_path=os.path.abspath(args.rec_model),
                    cls_model_path=os.path.join(project_root, "model_cls_bs32.rknn"),
                    det_input_size=448,
                    det_batch_size=16,
                    rec_batch_size=16,
                    rec_parallel_cores=(
                        RKNNLite.NPU_CORE_0,
                        RKNNLite.NPU_CORE_1,
                        RKNNLite.NPU_CORE_2,
                    ),
                    cls_batch_size=32,
                )

        if "yolo" in requested:
            with stage(sampler, "yolo", timings):
                for _ in range(max(1, args.loops)):
                    _frames, preds = yolo.process_frames_batch([frame], output_type="raw")
                    predictions = preds[0] if preds else []

        crops = make_crops(frame, predictions, count=args.crop_count)
        regions = make_rec_regions(count=args.crop_count)

        if "ocr_cls" in requested:
            with stage(sampler, "ocr_cls", timings):
                for _ in range(max(1, args.loops)):
                    ocr._run_cls_batch(crops)

        if "ocr_det" in requested:
            with stage(sampler, "ocr_det", timings):
                for _ in range(max(1, args.loops)):
                    ocr._detect_regions_batch(crops, tight_ocr, BOTTLE_LINE_PADDING)

        if "ocr_rec" in requested:
            with stage(sampler, "ocr_rec", timings):
                modules = [tight_ocr] * len(regions)
                for _ in range(max(1, args.loops)):
                    ocr._recognize_mixed_regions(modules, regions)

        if "full_ocr_task" in requested:
            with stage(sampler, "full_ocr_task", timings):
                task_group = []
                for idx, crop in enumerate(crops[: min(9, len(crops))]):
                    task_type = ("bottle", "bag", "shuye")[idx % 3]
                    task_group.append((task_type, idx, crop))
                for _ in range(max(1, min(args.loops, 3))):
                    ocr.recognize_task_batch(task_group)

    finally:
        sampler.stop()
        if yolo is not None:
            yolo.release()
        if ocr is not None:
            ocr.release()

    summary_lines = [
        f"project_root={project_root}",
        f"csv={csv_path}",
        f"loops={args.loops}",
        f"sample_interval={args.sample_interval}",
        "",
        "[stage elapsed]",
    ]
    summary_lines.extend([f"{name}: {elapsed:.6f}s" for name, elapsed in timings])
    summary_lines.extend(["", "[npu load summary]"])
    summary_lines.extend(summarize_csv(csv_path))
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines), flush=True)
    print(f"[done] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
