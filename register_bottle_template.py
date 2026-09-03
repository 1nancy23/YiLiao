#!/usr/bin/env python3
"""Capture and register six bottle SIFT templates from the live camera."""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import pymysql
except ModuleNotFoundError:
    pymysql = None

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "src" / "identification" / "feat_data"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "single_image_feature_cache.pkl"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
REQUIRED_TEMPLATE_COUNT = 6


@dataclass
class BottleCandidate:
    image: np.ndarray
    captured_at: float
    sharpness: float
    appearance: np.ndarray


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="使用当前摄像头为一个药瓶录入或更新六张 SIFT 模板。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--medicine-name", "-n", required=True, help="需要录入的药品名称")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="项目配置文件")
    parser.add_argument("--capture-seconds", type=float, default=8.0, help="单药瓶触发后的采集时长")
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=0.0,
        help="等待单药瓶画面的最长秒数；0 表示一直等待",
    )
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT), help="药瓶模板根目录")
    parser.add_argument("--cache-path", default=str(DEFAULT_CACHE_PATH), help="SIFT 缓存文件")
    parser.add_argument("--no-preview", action="store_true", help="不显示实时检测窗口")
    parser.add_argument("--allow-main-running", action="store_true", help="允许与主识别程序同时运行")
    parser.add_argument("--db-host", help="覆盖数据库地址")
    parser.add_argument("--db-port", type=int, help="覆盖数据库端口")
    parser.add_argument("--db-user", help="覆盖数据库用户名")
    parser.add_argument("--db-password", help="覆盖数据库密码")
    parser.add_argument("--db-name", help="覆盖数据库名称")
    return parser.parse_args(argv)


def load_config(config_path):
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config, path


def require_runtime_dependencies():
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if np is None:
        missing.append("numpy")
    if pymysql is None:
        missing.append("pymysql")
    if yaml is None:
        missing.append("pyyaml")
    if missing:
        raise RuntimeError(f"缺少运行依赖: {', '.join(missing)}")


def validate_medicine_name(value):
    name = str(value or "").strip()
    if not name:
        raise ValueError("药品名称不能为空")
    if name in {".", ".."} or any(char in name for char in ("/", "\\", "\0")):
        raise ValueError("药品名称不能包含路径分隔符")
    if len(name) > 255:
        raise ValueError("药品名称不能超过 255 个字符")
    return name


def resolve_project_path(value):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def ensure_template_path_is_safe(feature_root, medicine_name):
    feature_root = feature_root.resolve()
    target = (feature_root / medicine_name).resolve()
    if target.parent != feature_root:
        raise ValueError("药品模板目录越出了模板根目录")
    return target


def find_running_main_processes():
    if os.name != "posix" or not Path("/proc").is_dir():
        return []
    matches = []
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
        except (OSError, PermissionError):
            continue
        if "native_app.py" in command or "auto_headless_app.py" in command:
            matches.append((int(entry.name), command.strip()))
    return matches


def build_rtsp_url(config):
    rtsp = config.get("RTSP", {}) or {}
    required = ("username", "password", "ip_address", "port", "channel")
    missing = [key for key in required if str(rtsp.get(key, "")).strip() == ""]
    if missing:
        raise ValueError(f"RTSP 配置缺少字段: {', '.join(missing)}")
    return (
        f"rtsp://{rtsp['username']}:{rtsp['password']}@{rtsp['ip_address']}:"
        f"{rtsp['port']}/Streaming/Channels/{rtsp['channel']}"
    )


def enhance_trigger_frame_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)


def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def build_appearance_descriptor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32)
    normalized -= float(normalized.mean())
    deviation = float(normalized.std())
    if deviation > 1e-6:
        normalized /= deviation
    dct = cv2.dct(normalized)
    descriptor = dct[:16, :16].reshape(-1)
    descriptor = descriptor[1:]
    norm = float(np.linalg.norm(descriptor))
    if norm > 1e-6:
        descriptor /= norm
    return descriptor.astype(np.float32, copy=False)


def appearance_distance(first, second):
    similarity = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return 1.0 - similarity


def has_valid_sift(image, sift):
    _keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors is not None and descriptors.ndim == 2 and descriptors.shape[0] >= 2


def select_template_candidates(candidates, required=REQUIRED_TEMPLATE_COUNT):
    if len(candidates) < required:
        raise ValueError(f"有效药瓶 crop 只有 {len(candidates)} 张，少于要求的 {required} 张")

    sift = cv2.SIFT_create(nfeatures=500, contrastThreshold=0.03, edgeThreshold=10, sigma=1.6)
    sift_candidates = [candidate for candidate in candidates if has_valid_sift(candidate.image, sift)]
    if len(sift_candidates) < required:
        raise ValueError(
            f"可生成 SIFT 特征的 crop 只有 {len(sift_candidates)} 张，少于要求的 {required} 张"
        )

    sharpness_values = np.asarray([item.sharpness for item in sift_candidates], dtype=np.float32)
    sharpness_floor = float(np.percentile(sharpness_values, 35.0))
    sharp_pool = [item for item in sift_candidates if item.sharpness >= sharpness_floor]
    if len(sharp_pool) < required:
        sharp_pool = sorted(sift_candidates, key=lambda item: item.sharpness, reverse=True)[:required]

    maximum_sharpness = max(item.sharpness for item in sharp_pool)
    selected = [max(sharp_pool, key=lambda item: item.sharpness)]
    remaining = [item for item in sharp_pool if item is not selected[0]]

    while len(selected) < required:
        def selection_score(candidate):
            minimum_distance = min(
                appearance_distance(candidate.appearance, chosen.appearance)
                for chosen in selected
            )
            quality = candidate.sharpness / max(maximum_sharpness, 1e-6)
            return 0.75 * minimum_distance + 0.25 * quality

        chosen_index = max(range(len(remaining)), key=lambda index: selection_score(remaining[index]))
        chosen = remaining.pop(chosen_index)
        selected.append(chosen)

    return sorted(selected, key=lambda item: item.captured_at)


def resize_for_preview(image, max_width=1024, max_height=600):
    height, width = image.shape[:2]
    scale = min(max_width / float(width), max_height / float(height), 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def show_preview(processor, frame, detections, message, enabled):
    if not enabled:
        return False
    canvas = processor.draw_detections(frame, detections, copy_image=True)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 52), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        message,
        (16, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Bottle Template Registration", resize_for_preview(canvas))
    key = cv2.waitKey(1) & 0xFF
    return key in (27, ord("q"))


def extract_single_bottle(processor, frame):
    enhanced = enhance_trigger_frame_contrast(frame)
    _result_frames, prediction_batches = processor.process_frames_batch([enhanced], output_type="raw")
    detections = prediction_batches[0] if prediction_batches else []
    bottle_detections = [item for item in detections if int(item[5]) == 0]
    if len(bottle_detections) != 1:
        return enhanced, detections, None
    _bags, bottles, _infusions = processor.crop_by_class(
        enhanced,
        detections,
        bag_class_id=1,
        bottle_class_id=0,
        shuye_class_id=2,
    )
    if len(bottles) != 1 or bottles[0].size == 0:
        return enhanced, detections, None
    return enhanced, detections, bottles[0]


def capture_candidates(config, capture_seconds, wait_timeout, preview_enabled):
    from src.segmentation.segmenter_yolo import YOLOTileProcessor
    from src.stream.video_stream import VideoStream

    model_config = config.get("model", {}) or {}
    segmentor_config = config.get("segmentor", {}) or {}
    yolo_model_path = resolve_project_path(model_config.get("yolo_rknn_path", "./model_yolo_0615.rknn"))
    yolo_input_size = int(model_config.get("yolo_input_size", 640))
    target_fps = float(segmentor_config.get("target_fps", 6))
    if not yolo_model_path.is_file():
        raise FileNotFoundError(f"YOLO RKNN 模型不存在: {yolo_model_path}")

    processor = None
    video_stream = None
    capture_started = None
    waiting_started = time.monotonic()
    candidates = []
    try:
        processor = YOLOTileProcessor(
            model=None,
            model_path=str(yolo_model_path),
            device="npu",
            tile_size=yolo_input_size,
            input_size=yolo_input_size,
            overlap=64,
            conf_thres=0.45,
            iou_thres=0.85,
            batch_size=1,
            contain_thres=0.92,
            class_conf_thres={0: 0.35, 1: 0.25, 2: 0.32},
            class_iou_thres={0: 0.65, 1: 0.65, 2: 0.45},
        )
        video_stream = VideoStream(build_rtsp_url(config), target_fps=target_fps, max_buffer_size=3)
        video_stream.start()
        print("[等待] 请让画面中保持一个药瓶；篮子和其它类别目标可以存在。按 Ctrl+C 可取消。")

        while True:
            if capture_started is None and wait_timeout > 0:
                if time.monotonic() - waiting_started >= wait_timeout:
                    raise TimeoutError(f"等待单药瓶画面超过 {wait_timeout:.1f} 秒")

            frames = video_stream.get_batch(1)
            if not frames:
                time.sleep(0.01)
                continue

            enhanced, detections, bottle_crop = extract_single_bottle(processor, frames[-1])
            now = time.monotonic()
            if capture_started is None:
                if bottle_crop is not None:
                    capture_started = now
                    video_stream.frame_buffer.clear()
                    print(f"[触发] 已检测到唯一药瓶，开始采集后续 {capture_seconds:.1f} 秒。")
                    message = f"CAPTURE 0.0/{capture_seconds:.1f}s"
                else:
                    bottle_count = sum(1 for item in detections if int(item[5]) == 0)
                    message = f"WAIT: targets={len(detections)} bottles={bottle_count}"
            else:
                elapsed = now - capture_started
                if elapsed >= capture_seconds:
                    break
                if bottle_crop is not None:
                    candidates.append(
                        BottleCandidate(
                            image=bottle_crop,
                            captured_at=elapsed,
                            sharpness=calculate_sharpness(bottle_crop),
                            appearance=build_appearance_descriptor(bottle_crop),
                        )
                    )
                message = f"CAPTURE {elapsed:.1f}/{capture_seconds:.1f}s valid={len(candidates)}"

            if show_preview(processor, enhanced, detections, message, preview_enabled):
                raise KeyboardInterrupt

        print(f"[采集完成] {capture_seconds:.1f} 秒内获得 {len(candidates)} 张有效单药瓶 crop。")
        return candidates
    finally:
        if video_stream is not None:
            video_stream.stop()
            stream_reader = getattr(video_stream, "stream_reader", None)
            if stream_reader is not None:
                stream_reader.stop()
                stream_reader.join(timeout=10.0)
                if stream_reader.is_alive():
                    print("[警告] RTSP 读取线程未在 10 秒内结束。", file=sys.stderr)
        if processor is not None:
            processor.release()
        if preview_enabled:
            try:
                cv2.destroyWindow("Bottle Template Registration")
            except cv2.error:
                pass


def connect_database(config, args):
    db_config = config.get("db_config", {}) or {}
    return pymysql.connect(
        host=args.db_host or os.environ.get("YILIAO_DB_HOST") or db_config.get("host", "127.0.0.1"),
        port=int(args.db_port or os.environ.get("YILIAO_DB_PORT") or db_config.get("port", 3306)),
        user=args.db_user or os.environ.get("YILIAO_DB_USER") or db_config.get("user", "root"),
        password=(
            args.db_password
            if args.db_password is not None
            else os.environ.get("YILIAO_DB_PASSWORD", db_config.get("password", ""))
        ),
        database=args.db_name or os.environ.get("YILIAO_DB_NAME") or db_config.get("database", "medicine_db2"),
        charset=os.environ.get("YILIAO_DB_CHARSET", db_config.get("charset", "utf8")),
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def write_image(path, image):
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 96])
    if not success:
        raise OSError(f"模板图像编码失败: {path}")
    encoded.tofile(str(path))


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"模板图像读取失败: {path}")
    return image


def atomic_write_cache(cache_path, payload):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        os.replace(str(temporary), str(cache_path))
    finally:
        if temporary.exists():
            temporary.unlink()


def restore_registration_state(target_folder, previous_folder, cache_path, cache_backup, cache_existed):
    if target_folder.exists():
        shutil.rmtree(target_folder)
    if previous_folder.exists():
        os.replace(str(previous_folder), str(target_folder))

    if cache_existed and cache_backup.exists():
        os.replace(str(cache_backup), str(cache_path))
    elif not cache_existed and cache_path.exists():
        cache_path.unlink()


def restore_cache_state(cache_path, cache_backup, cache_existed):
    if cache_existed and cache_backup.exists():
        os.replace(str(cache_backup), str(cache_path))
    elif not cache_existed and cache_path.exists():
        cache_path.unlink()


def register_templates(config, args, medicine_name, selected, feature_root, cache_path):
    from src.identification.Recog import PharmaceuticalBottleClassifier, _feature_root_signature

    feature_root.mkdir(parents=True, exist_ok=True)
    target_folder = ensure_template_path_is_safe(feature_root, medicine_name)
    if target_folder.exists() and not target_folder.is_dir():
        raise ValueError(f"药品模板路径已存在但不是文件夹: {target_folder}")
    connection = None
    classifier = None
    committed = False
    folder_changed = False

    os.environ["YILIAO_FEATURE_ROOT"] = str(feature_root)
    os.environ["YILIAO_FEATURE_CACHE"] = str(cache_path)
    os.environ["YILIAO_EAGER_BOTTLE_CLS"] = "0"

    with tempfile.TemporaryDirectory(prefix="bottle_template_registration_", dir=str(PROJECT_ROOT)) as work_dir_name:
        work_dir = Path(work_dir_name)
        staged_folder = work_dir / "new_templates"
        previous_folder = work_dir / "previous_templates"
        cache_backup = work_dir / "previous_cache.pkl"
        staged_folder.mkdir()

        for index, candidate in enumerate(selected, start=1):
            write_image(staged_folder / f"template_{index:02d}.jpg", candidate.image)

        cache_existed = cache_path.is_file()
        if cache_existed:
            shutil.copy2(cache_path, cache_backup)

        try:
            connection = connect_database(config, args)
            classifier = PharmaceuticalBottleClassifier(connection, device="cpu")

            sift_blobs = []
            template_entries = []
            for image_path in sorted(staged_folder.iterdir()):
                image = read_image(image_path)
                _keypoints, descriptors = classifier._extract_features_from_image(image)
                if descriptors is None or descriptors.ndim != 2 or descriptors.shape[0] < 2:
                    raise ValueError(f"无法从模板生成有效 SIFT 特征: {image_path.name}")
                descriptors = np.asarray(descriptors, dtype=np.float32)
                sift_blobs.append(pickle.dumps(descriptors, protocol=pickle.HIGHEST_PROTOCOL))
                template_entries.append({"desc_sift": descriptors})

            if len(sift_blobs) != REQUIRED_TEMPLATE_COUNT:
                raise ValueError(f"SIFT 特征数量不是 {REQUIRED_TEMPLATE_COUNT}")

            table_config = config.get("table_config", {}) or {}
            drug_table = str(table_config.get("drug_table", "drugs"))
            drug_column = str(table_config.get("drug_column", "medicine_name"))
            if not drug_table.replace("_", "").isalnum() or not drug_column.replace("_", "").isalnum():
                raise ValueError("数据库药品表名或名称列名不合法")

            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT 1 FROM `{drug_table}` WHERE `{drug_column}` = %s LIMIT 1",
                    (medicine_name,),
                )
                database_exists = cursor.fetchone() is not None

            folder_exists = target_folder.is_dir() and any(
                path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                for path in target_folder.iterdir()
            )
            operation = "更新" if folder_exists or database_exists else "新增"
            print(
                f"[写入模式] {operation}: 模板目录={'存在' if folder_exists else '不存在'}, "
                f"数据库记录={'存在' if database_exists else '不存在'}"
            )

            connection.begin()
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO `{drug_table}`
                        (`{drug_column}`, sift1, sift2, sift3, sift4, sift5, sift6)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        sift1 = VALUES(sift1),
                        sift2 = VALUES(sift2),
                        sift3 = VALUES(sift3),
                        sift4 = VALUES(sift4),
                        sift5 = VALUES(sift5),
                        sift6 = VALUES(sift6)
                    """,
                    (medicine_name, *sift_blobs),
                )

            if target_folder.exists():
                os.replace(str(target_folder), str(previous_folder))
                folder_changed = True
            os.replace(str(staged_folder), str(target_folder))
            folder_changed = True

            templates = dict(classifier._templates_cache)
            deep_average = dict(classifier._deep_avg_cache)
            templates[medicine_name] = template_entries
            deep_average[medicine_name] = None
            atomic_write_cache(
                cache_path,
                {
                    "templates": templates,
                    "deep_avg": deep_average,
                    "feature_root_signature": _feature_root_signature(str(feature_root)),
                },
            )

            connection.commit()
            committed = True
            if previous_folder.exists():
                shutil.rmtree(previous_folder)

            print(f"[完成] {operation}药品模板: {medicine_name}")
            print(f"[模板目录] {target_folder}")
            print(f"[SIFT 缓存] {cache_path}")
            print(f"[数据库] {drug_table}.{drug_column} 已写入 6 组 SIFT 特征")
        except Exception:
            if connection is not None and not committed:
                connection.rollback()
            if folder_changed:
                restore_registration_state(
                    target_folder,
                    previous_folder,
                    cache_path,
                    cache_backup,
                    cache_existed,
                )
            else:
                restore_cache_state(cache_path, cache_backup, cache_existed)
            raise
        finally:
            if classifier is not None:
                classifier.release()
            if connection is not None:
                connection.close()


def main(argv=None):
    args = parse_args(argv)
    try:
        require_runtime_dependencies()
        medicine_name = validate_medicine_name(args.medicine_name)
        if args.capture_seconds <= 0:
            raise ValueError("--capture-seconds 必须大于 0")
        if args.wait_timeout < 0:
            raise ValueError("--wait-timeout 不能小于 0")

        config, config_path = load_config(args.config)
        feature_root = resolve_project_path(args.feature_root)
        cache_path = resolve_project_path(args.cache_path)
        ensure_template_path_is_safe(feature_root, medicine_name)

        running = find_running_main_processes()
        if running and not args.allow_main_running:
            details = "; ".join(f"PID={pid} {command}" for pid, command in running)
            raise RuntimeError(
                "主识别程序仍在运行。请先暂停主程序，避免争用摄像头/NPU和写入旧缓存。"
                f" 当前进程: {details}"
            )

        preview_enabled = not args.no_preview and bool(os.environ.get("DISPLAY"))
        print(f"[配置] {config_path}")
        print(f"[药品名称] {medicine_name}")
        print(f"[模板根目录] {feature_root}")
        print(f"[采集时长] {args.capture_seconds:.1f} 秒")

        candidates = capture_candidates(
            config,
            capture_seconds=float(args.capture_seconds),
            wait_timeout=float(args.wait_timeout),
            preview_enabled=preview_enabled,
        )
        selected = select_template_candidates(candidates)
        print("[模板筛选] 已选出六张清晰且互异的 crop:")
        for index, candidate in enumerate(selected, start=1):
            print(
                f"  {index}: t={candidate.captured_at:.2f}s, "
                f"sharpness={candidate.sharpness:.2f}, size={candidate.image.shape[1]}x{candidate.image.shape[0]}"
            )

        register_templates(
            config,
            args,
            medicine_name,
            selected,
            feature_root,
            cache_path,
        )
        return 0
    except KeyboardInterrupt:
        print("\n[取消] 用户中止了模板录入。", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        print(f"[失败] {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"[失败] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
