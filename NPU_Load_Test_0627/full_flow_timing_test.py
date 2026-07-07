# -*- coding: utf-8 -*-
import argparse
import contextlib
import csv
import io
import json
import os
import sys
import time
import traceback

import cv2
import yaml
from rknnlite.api import RKNNLite

from run_npu_load_test import NpuSampler


DEFAULT_PROJECT_ROOT = (
    "/home/forlinx/Models/AnotherYiliao/shibie/"
    "YiLiaoShiBie_0521/YiLiaoShiBie"
)
DEFAULT_REC_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "model_ocr_bs16.rknn",
)


def current_stage(sampler):
    with sampler._stage_lock:
        return sampler.stage


def wrap_stage_method(sampler, cls, method_name, stage_name):
    original = getattr(cls, method_name)

    def wrapped(self, *args, **kwargs):
        previous = current_stage(sampler)
        sampler.set_stage(stage_name)
        try:
            return original(self, *args, **kwargs)
        finally:
            sampler.set_stage(previous)

    setattr(cls, method_name, wrapped)
    return original


def install_stage_wrappers(sampler):
    from src.segmentation.segmenter_yolo import YOLOTileProcessor
    from src.identification.rknn_ocr_adapter import RknnOCRRecognizer

    originals = [
        (YOLOTileProcessor, "process_frames_batch", wrap_stage_method(
            sampler, YOLOTileProcessor, "process_frames_batch", "yolo"
        )),
        (RknnOCRRecognizer, "_run_cls_batch", wrap_stage_method(
            sampler, RknnOCRRecognizer, "_run_cls_batch", "ocr_cls"
        )),
        (RknnOCRRecognizer, "_detect_regions_batch", wrap_stage_method(
            sampler, RknnOCRRecognizer, "_detect_regions_batch", "ocr_det"
        )),
        (RknnOCRRecognizer, "_recognize_mixed_candidates_batch", wrap_stage_method(
            sampler, RknnOCRRecognizer, "_recognize_mixed_candidates_batch", "ocr_det"
        )),
        (RknnOCRRecognizer, "_recognize_mixed_regions", wrap_stage_method(
            sampler, RknnOCRRecognizer, "_recognize_mixed_regions", "ocr_rec"
        )),
    ]
    return originals


def restore_stage_wrappers(originals):
    for cls, method_name, original in originals:
        setattr(cls, method_name, original)


def capture_rtsp_frame(config, timeout_sec=8.0):
    rtsp = config["RTSP"]
    url = (
        f"rtsp://{rtsp['username']}:{rtsp['password']}"
        f"@{rtsp['ip_address']}:{rtsp['port']}"
        f"/Streaming/Channels/{rtsp['channel']}"
    )
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(url)
    deadline = time.time() + timeout_sec
    best_frame = None
    best_score = -1.0
    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            time.sleep(0.05)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if score > best_score:
            best_score = score
            best_frame = frame.copy()
        if best_score >= 80.0:
            break
    cap.release()
    if best_frame is None:
        raise RuntimeError("failed to capture RTSP frame")
    return best_frame, best_score


def jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def compact_result(result):
    background = result.get("background") or {}
    compact = {
        "image": result.get("image"),
        "runtime_sec_excluding_model_init": result.get("runtime_sec_excluding_model_init"),
        "detection_timing_sec": result.get("detection_timing_sec"),
        "timing_sec": background.get("timing_sec"),
        "counts": background.get("counts"),
        "recognized_medicines": background.get("recognized_medicines"),
        "patient_name": background.get("patient_name"),
        "database_match": background.get("database_match"),
        "bottles": [
            {
                "index": item.get("index"),
                "ocr_text": item.get("ocr_text"),
                "final_medicine": item.get("final_medicine"),
                "classification_method": item.get("classification_method"),
                "confidence": item.get("confidence"),
                "det_region_count": item.get("det_region_count"),
                "rec_nonempty_count": item.get("rec_nonempty_count"),
            }
            for item in background.get("bottles") or []
        ],
        "bags": background.get("bags"),
        "infusions": background.get("infusions"),
    }
    return jsonable(compact)


def format_timing_lines(compact):
    lines = []
    lines.append("[full flow timing]")
    lines.append(f"runtime_sec_excluding_model_init={compact.get('runtime_sec_excluding_model_init')}")
    det_timing = compact.get("detection_timing_sec") or {}
    for key in sorted(det_timing):
        lines.append(f"detection.{key}={det_timing[key]}")
    timing = compact.get("timing_sec") or {}
    for key in sorted(timing):
        lines.append(f"background.{key}={timing[key]}")
    lines.append("")
    lines.append(f"counts={compact.get('counts')}")
    lines.append(f"patient={compact.get('patient_name')}")
    lines.append(f"recognized_medicines={compact.get('recognized_medicines')}")
    match = compact.get("database_match") or {}
    lines.append(f"database_status={match.get('status')}, message={match.get('message')}")
    return lines


def summarize_npu_stage_load(csv_path):
    by_stage = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stage = row.get("stage") or ""
            if stage == "idle":
                continue
            values = []
            for key in ("core0_pct", "core1_pct", "core2_pct"):
                try:
                    values.append(float(row.get(key) or ""))
                except ValueError:
                    values.append(None)
            if any(value is not None for value in values):
                by_stage.setdefault(stage, []).append(values)

    lines = ["", "[npu stage load]"]
    for stage in sorted(by_stage):
        rows = by_stage[stage]
        parts = [f"{stage}: samples={len(rows)}"]
        for idx in range(3):
            vals = [row[idx] for row in rows if row[idx] is not None]
            if vals:
                parts.append(f"core{idx}_max={max(vals):.2f}")
                parts.append(f"core{idx}_avg={sum(vals) / len(vals):.2f}")
            else:
                parts.append(f"core{idx}_max=")
                parts.append(f"core{idx}_avg=")
        lines.append(", ".join(parts))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output-dir", default="/home/forlinx/Models/AnotherYiliao/npu_load_test_0627/full_flow_results")
    parser.add_argument("--rec-model", default=DEFAULT_REC_MODEL)
    parser.add_argument("--capture-timeout", type=float, default=8.0)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    sys.path.insert(0, project_root)
    os.chdir(project_root)

    os.environ["YILIAO_VERBOSE_RUNTIME"] = "1"
    os.environ["YILIAO_RUNTIME_LOGS"] = "1"
    os.environ["YILIAO_COLLECT_TIMING"] = "1"
    os.environ.setdefault("YILIAO_QUIET_OCR", "0")

    from native_app import (
        create_local_bottle_db_matcher,
        init_db,
        release_models,
    )
    from run_realtime_detection_yolo_new_3 import run_realtime_detection
    from src.identification.DrugMatcher import DrugMatcher
    from src.identification.Recog import PharmaceuticalBottleClassifier
    from src.identification.rknn_ocr_adapter import RknnOCRRecognizer

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    frame_path = os.path.join(run_dir, "realtime_frame.jpg")
    full_json_path = os.path.join(run_dir, "full_result.json")
    compact_json_path = os.path.join(run_dir, "compact_result.json")
    summary_path = os.path.join(run_dir, "timing_summary.txt")
    stdout_path = os.path.join(run_dir, "stdout.log")
    npu_csv_path = os.path.join(run_dir, "npu_stage_load.csv")

    conn = None
    classifier = None
    ocr_recognizers = []
    result = None
    sampler = NpuSampler(npu_csv_path, interval=args.sample_interval)
    sampler.start()
    wrappers = []
    try:
        with open(os.path.join(project_root, "config.yaml"), "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        sampler.set_stage("capture")
        t_capture = time.perf_counter()
        frame, sharpness = capture_rtsp_frame(config, timeout_sec=args.capture_timeout)
        capture_elapsed = time.perf_counter() - t_capture
        cv2.imwrite(frame_path, frame)

        sampler.set_stage("model_init")
        t_init = time.perf_counter()
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
        ocr_recognizers = [
            RknnOCRRecognizer(
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
        ]
        init_elapsed = time.perf_counter() - t_init
        sampler.set_stage("idle")

        wrappers = install_stage_wrappers(sampler)

        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer):
            result = run_realtime_detection(
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
                batch_frames=1,
                output_type=config["display"]["output_type"],
                overlay_alpha=config["display"]["overlay_alpha"],
                display_scale=config["display"]["display_scale"],
                save_video=False,
                output_path=config["saving"]["output_path"],
                save_fps=config["saving"]["save_fps"],
                device="npu",
                yolo_model_path=config["model"].get("yolo_rknn_path", "./model_yolo_0615.rknn"),
                yolo_input_size=int(config["model"].get("yolo_input_size", 640)),
                basket_model_path=config["model"].get("basket_rknn_path", "./model_bask_0609_n.rknn"),
                basket_input_size=int(config["model"].get("basket_input_size", 640)),
                trigger_interval=999999.0,
                recognition_workers=1,
                classifier_thread_safe=False,
                quiet_ocr=False,
                headless=True,
                single_image_path=frame_path,
            )
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(stdout_buffer.getvalue())

        compact = compact_result(result or {})
        compact["capture_sec"] = round(capture_elapsed, 6)
        compact["capture_sharpness"] = round(sharpness, 3)
        compact["model_init_sec"] = round(init_elapsed, 6)
        compact["npu_stage_load_csv"] = npu_csv_path
        with open(full_json_path, "w", encoding="utf-8") as f:
            json.dump(jsonable(result), f, ensure_ascii=False, indent=2)
        with open(compact_json_path, "w", encoding="utf-8") as f:
            json.dump(compact, f, ensure_ascii=False, indent=2)

        lines = [
            f"run_dir={run_dir}",
            f"frame={frame_path}",
            f"capture_sec={compact['capture_sec']}",
            f"capture_sharpness={compact['capture_sharpness']}",
            f"model_init_sec={compact['model_init_sec']}",
            "",
        ]
        lines.extend(format_timing_lines(compact))
        lines.extend(summarize_npu_stage_load(npu_csv_path))
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
        print(f"[done] {run_dir}", flush=True)
    except Exception:
        with open(os.path.join(run_dir, "error.txt"), "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        raise
    finally:
        sampler.stop()
        restore_stage_wrappers(wrappers)
        release_models(classifier, ocr_recognizers)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
