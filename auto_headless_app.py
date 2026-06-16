# -*- coding: utf-8 -*-
import argparse
import contextlib
import io
import os
import threading
import time
import traceback

import yaml

from native_app import (
    PROJECT_ROOT,
    create_local_bottle_db_matcher,
    env_bool,
    init_db,
    release_models,
)


def _set_default_env():
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    os.environ.setdefault("YILIAO_HEADLESS", "1")
    os.environ.setdefault("YILIAO_RUNTIME_LOGS", "0")
    os.environ.setdefault("YILIAO_VERBOSE_RUNTIME", "0")
    os.environ.setdefault("YILIAO_QUIET_OCR", "1")
    os.environ.setdefault("YILIAO_COLLECT_TIMING", "0")


def _append_json_like_log(prefix, payload):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {prefix}: {payload}", flush=True)


def run_auto_headless(check_only=False):
    _set_default_env()

    from run_realtime_detection_yolo_new_3 import run_realtime_detection
    from src.identification.DrugMatcher import DrugMatcher
    from src.identification.Recog import PharmaceuticalBottleClassifier
    from src.identification.rknn_ocr_adapter import RknnOCRRecognizer

    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
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

    required_paths = [
        config["model"].get("yolo_rknn_path", "./model_yolo_0615.rknn"),
        config["model"].get("basket_rknn_path", "./model_bask_0609_n.rknn"),
        "model_cls_bs32.rknn",
        "model_ocr_0526.rknn",
        os.path.join("src", "identification", "Det_bs32.rknn"),
    ]
    for path in required_paths:
        full_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(full_path)

    if check_only:
        _append_json_like_log("check", {
            "project": PROJECT_ROOT,
            "trigger_mode": "auto",
            "headless": True,
            "yolo": config["model"].get("yolo_rknn_path"),
            "yolo_input_size": config["model"].get("yolo_input_size"),
            "basket_stable_frames": runtime_config.get("basket_stable_frames"),
        })
        return

    conn = None
    classifier = None
    ocr_recognizers = []
    stop_event = threading.Event()

    def status_callback(data):
        state = data.get("state")
        trigger = data.get("trigger")
        processing = data.get("processing")
        frame_count = data.get("frame_count")
        if trigger == "basket_auto" or state in ("loading_models", "running", "error"):
            _append_json_like_log("status", {
                "state": state,
                "trigger": trigger,
                "processing": processing,
                "frame_count": frame_count,
                "stable": data.get("basket_stable_frames"),
                "stable_required": data.get("basket_stable_required"),
                "error": data.get("last_error"),
            })

    def result_callback(payload):
        match = payload.get("database_match") or {}
        _append_json_like_log("result", {
            "patient": payload.get("patient_name"),
            "medicines": payload.get("recognized_medicines"),
            "counts": payload.get("counts"),
            "match_status": match.get("status"),
            "match_message": match.get("message"),
            "timing": payload.get("timing_sec"),
        })

    try:
        _append_json_like_log("start", {
            "project": PROJECT_ROOT,
            "mode": "auto_headless",
        })

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
            yolo_model_path=config["model"].get("yolo_rknn_path", "./model_yolo_0615.rknn"),
            yolo_input_size=int(config["model"].get("yolo_input_size", 640)),
            basket_model_path=config["model"].get("basket_rknn_path", "./model_bask_0609_n.rknn"),
            basket_input_size=int(config["model"].get("basket_input_size", 640)),
            basket_area_threshold=float(runtime_config.get("basket_area_threshold", 0.40)),
            basket_sharpness_threshold=float(runtime_config.get("basket_sharpness_threshold", 55.0)),
            basket_stable_frames=int(runtime_config.get("basket_stable_frames", 6)),
            basket_capture_timeout=float(runtime_config.get("basket_capture_timeout", 10.0)),
            basket_resume_delay=float(runtime_config.get("basket_resume_delay", 1.0)),
            trigger_interval=999999.0,
            recognition_workers=recognition_workers,
            classifier_thread_safe=False,
            quiet_ocr=quiet_ocr,
            headless=True,
            trigger_mode="auto",
            trigger_mode_getter=lambda: "auto",
            manual_trigger_event=None,
            stop_event=stop_event,
            status_callback=status_callback,
            result_callback=result_callback,
            frame_callback=None,
        )
    except KeyboardInterrupt:
        _append_json_like_log("stop", "keyboard_interrupt")
    except Exception:
        _append_json_like_log("error", traceback.format_exc())
        raise
    finally:
        stop_event.set()
        release_models(classifier, ocr_recognizers)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate paths and config without starting RTSP loop")
    args = parser.parse_args()
    run_auto_headless(check_only=args.check)


if __name__ == "__main__":
    main()
