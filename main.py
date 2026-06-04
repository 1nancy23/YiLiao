import os
import sys
import traceback
import contextlib
import io
import builtins

import pymysql
import yaml

from run_realtime_detection_yolo_new_3 import run_realtime_detection
from src.identification.DrugMatcher import DrugMatcher
from src.identification.Recog import PharmaceuticalBottleClassifier
from src.identification.rknn_ocr_adapter import RknnOCRRecognizer


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


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no")


def main():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    os.environ.setdefault("YILIAO_MIXED_OCR_BATCH", "1")
    os.environ.setdefault("YILIAO_FEATURE_CACHE", "./single_image_feature_cache.pkl")
    os.environ.setdefault("YILIAO_VERBOSE_RUNTIME", "1")
    os.environ.setdefault("YILIAO_RUNTIME_LOGS", "1")
    os.environ.setdefault("YILIAO_COLLECT_TIMING", "1")
    os.environ.setdefault("YILIAO_QUIET_OCR", "0")
    os.environ.setdefault("YILIAO_CLS_MODEL", "./model_cls_bs32.rknn")
    os.environ.setdefault("YILIAO_CLS_BATCH_SIZE", "32")

    runtime_logs = env_bool("YILIAO_RUNTIME_LOGS", False)

    def print(*args, **kwargs):
        if runtime_logs:
            builtins.print(*args, **kwargs)

    with open("./config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    runtime_config = config.get("runtime", {}) or {}
    recognition_workers = int(os.environ.get(
        "YILIAO_RECOGNITION_WORKERS",
        runtime_config.get("recognition_workers", 1),
    ))
    trigger_interval = float(os.environ.get(
        "YILIAO_TRIGGER_INTERVAL",
        runtime_config.get("trigger_interval", 15),
    ))
    max_frames_env = os.environ.get("YILIAO_MAX_FRAMES", "")
    max_frames_value = runtime_config.get("max_frames")
    max_frames = int(max_frames_env or max_frames_value) if (max_frames_env or max_frames_value) else None
    quiet_ocr = env_bool("YILIAO_QUIET_OCR", bool(runtime_config.get("quiet_ocr", True)))
    headless = env_bool("YILIAO_HEADLESS", runtime_config.get("headless", None))
    ocr_instance_count = int(os.environ.get(
        "YILIAO_OCR_INSTANCES",
        runtime_config.get("ocr_instances", 1),
    ))

    conn = init_db(config.get("db_config", {}))
    print("数据库连接成功")

    tables = config["table_config"]
    if quiet_ocr:
        init_log_stream = io.StringIO()
    else:
        init_log_stream = None

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

    ocr_count = max(1, ocr_instance_count)
    print("初始化 RKNN OCR 实例池...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    det_model_path = os.path.join(project_root, "src", "identification", "Det_bs32.rknn")
    rec_model_path = os.path.join(project_root, "model_ocr_0526.rknn")
    cls_model_path = os.path.join(project_root, "model_cls_bs32.rknn")
    ocr_recognizers = [
        RknnOCRRecognizer(
            det_model_path=det_model_path,
            rec_model_path=rec_model_path,
            cls_model_path=cls_model_path,
            det_batch_size=32,
            rec_batch_size=16,
            cls_batch_size=32,
        )
        for _ in range(ocr_count)
    ]
    print(f"RKNN OCR 初始化完成: {len(ocr_recognizers)} 个实例")

    print("实时运行参数:")
    print(f"  recognition_workers: {recognition_workers}")
    print(f"  trigger_interval: {trigger_interval}")
    print(f"  max_frames: {max_frames}")
    print(f"  quiet_ocr: {quiet_ocr}")
    print(f"  headless: {headless}")
    print(f"  ocr_instances: {ocr_count}")
    print(f"  mixed_ocr_batch: {os.environ.get('YILIAO_MIXED_OCR_BATCH')}")
    print(f"  ocr_det_model: {det_model_path} (batch=32)")
    print(f"  ocr_rec_model: {rec_model_path} (batch=16)")
    print(f"  ocr_cls_model: {cls_model_path} (batch=32)")

    print("\n" + "=" * 60)
    print("启动实时视频识别: YOLO + RKNN OCR + SIFT")
    print("=" * 60)
    print("RTSP 参数:")
    print(config["RTSP"])

    try:
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
            save_video=config["saving"]["save_video"],
            output_path=config["saving"]["output_path"],
            save_fps=config["saving"]["save_fps"],
            device="npu",
            trigger_interval=trigger_interval,
            recognition_workers=recognition_workers,
            classifier_thread_safe=False,
            quiet_ocr=quiet_ocr,
            headless=headless,
            max_frames=max_frames,
        )
    finally:
        conn.close()
        print("数据库已断开连接")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
