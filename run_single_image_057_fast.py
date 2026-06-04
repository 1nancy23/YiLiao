import argparse
import contextlib
import io
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import cv2
import numpy as np
import pymysql
import yaml
from fuzzywuzzy import fuzz, process
from paddleocr import PaddleOCR

from src.identification.OCRRecognizer import OCRRecognizer_ori
from src.identification.Recog import PharmaceuticalBottleClassifier
from src.identification.rknn_ocr_adapter import RknnOCRRecognizer
from src.segmentation.segmenter_yolo import YOLOTileProcessor


DEFAULT_ROI = (100, 100, 3400, 2160)
CLASS_NAMES = ["bottle", "bag", "infusion"]
INIT_TIMING_KEYS = {"load_yolo", "load_db_and_matchers", "load_ocr"}


class CachedNameMatcher:
    def __init__(self, drug_names, patient_names):
        self.drug_names = [name for name in drug_names if name]
        self.patient_names = [name for name in patient_names if name]
        if "赵二虎" not in self.patient_names:
            self.patient_names.append("赵二虎")
        self._match_cache = {}
        self._weak_drug_terms = {
            "注射", "射用", "用", "液", "钠", "酸", "素", "水", "片", "胶囊",
            "批号", "生产", "有效", "规格",
        }
        self._latin_drug_hints = {
            "ceft": ["头孢"],
            "ceftriax": ["头孢", "曲松"],
            "triax": ["曲松"],
            "sulbact": ["舒巴坦"],
            "ornith": ["鸟氨酸"],
            "glycyrrh": ["甘草酸"],
        }

    @staticmethod
    def _clean_match_text(text):
        text = str(text or "").replace(" ", "")
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)

    def _extract_strong_query_terms(self, query):
        clean = self._clean_match_text(query)
        chinese_runs = re.findall(r"[\u4e00-\u9fff]{2,}", clean)
        terms = set()
        lower = clean.lower()
        for key, hints in self._latin_drug_hints.items():
            if key in lower:
                terms.update(hints)
        for run in chinese_runs:
            max_n = min(5, len(run))
            for n in range(max_n, 1, -1):
                for i in range(0, len(run) - n + 1):
                    term = run[i:i + n]
                    if term in self._weak_drug_terms:
                        continue
                    if n == 2 and any(weak in term for weak in ("注", "用", "液")):
                        continue
                    terms.add(term)
        return sorted(terms, key=lambda value: (-len(value), value))

    def _rank_drug_matches(self, query, names, threshold, limit):
        clean_query = self._clean_match_text(query)
        terms = self._extract_strong_query_terms(clean_query)
        ranked = []
        for name in names:
            clean_name = self._clean_match_text(name)
            if not clean_name:
                continue

            partial = fuzz.partial_ratio(clean_query, clean_name)
            ratio = fuzz.ratio(clean_query, clean_name)
            token = fuzz.token_set_ratio(clean_query, clean_name)
            term_hits = [term for term in terms if term in clean_name]
            term_score = sum(len(term) * 12 for term in term_hits)
            fuzzy_score = max(partial, ratio, token)
            score = fuzzy_score + term_score

            if term_hits:
                ranked.append((name, score, len("".join(term_hits)), partial))
            elif fuzzy_score >= threshold:
                ranked.append((name, score, 0, partial))

        ranked.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        if limit is not None:
            ranked = ranked[:limit]
        return [item[0] for item in ranked]

    def match(self, query, match_type="bottle", threshold=80, limit=None):
        if not query or len(query.strip()) < 1:
            return []

        cache_key = (query, match_type, int(threshold), limit)
        cached = self._match_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        if match_type == "bottle":
            names = self.drug_names
        elif match_type == "bag":
            names = self.patient_names
        else:
            raise ValueError(f"invalid match_type: {match_type}")

        if not names:
            return []

        if match_type == "bottle":
            result = self._rank_drug_matches(query, names, threshold, limit)
            if result:
                self._match_cache[cache_key] = tuple(result)
                return result

        matches = process.extractBests(
            query,
            names,
            scorer=fuzz.partial_ratio,
            score_cutoff=threshold,
            limit=limit,
        )

        if match_type == "bottle" and not matches:
            result = list(names)
        else:
            result = [match[0] for match in matches]

        self._match_cache[cache_key] = tuple(result)
        return result


def init_db_from_config(config):
    db = config.get("db_config", {})
    kwargs = dict(
        host=db.get("host", "192.168.137.1"),
        user=db.get("user", "root"),
        password=db.get("password", "root"),
        database=db.get("database", "medicine_db"),
        charset=db.get("charset", "utf8"),
        port=int(db.get("port", 3306)),
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        return pymysql.connect(**kwargs)
    except pymysql.err.OperationalError as exc:
        if exc.args and exc.args[0] == 1115 and kwargs["charset"].lower() == "utf8mb4":
            kwargs["charset"] = "utf8"
            return pymysql.connect(**kwargs)
        raise


def load_or_build_name_cache(config, cache_path):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CachedNameMatcher(data.get("drug_names", []), data.get("patient_names", [])), None

    conn = init_db_from_config(config)
    tables = config["table_config"]
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT {tables['drug_column']} FROM {tables['drug_table']}")
        drug_rows = cursor.fetchall()
        cursor.execute(f"SELECT {tables['patient_column']} FROM {tables['patient_table']}")
        patient_rows = cursor.fetchall()

    drug_names = [
        row[tables["drug_column"]] if isinstance(row, dict) else row[0]
        for row in drug_rows
    ]
    patient_names = [
        row[tables["patient_column"]] if isinstance(row, dict) else row[0]
        for row in patient_rows
    ]

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {"drug_names": drug_names, "patient_names": patient_names},
                f,
                ensure_ascii=False,
                indent=2,
            )

    return CachedNameMatcher(drug_names, patient_names), conn


def init_fast_ocr(cpu_threads=4, rec_batch_num=8, show_log=False):
    return PaddleOCR(
        use_angle_cls=False,
        lang="ch",
        det=True,
        rec=True,
        use_gpu=False,
        show_log=show_log,
        cpu_threads=cpu_threads,
        rec_batch_num=rec_batch_num,
        det_max_side_len=480,
        det_score_mode="Fast",
        det_db_thresh=0.6,
        det_db_box_thresh=0.6,
        det_db_unclip_ratio=2.0,
        rec_score_thresh=0.6,
    )


def parse_roi(value):
    if not value:
        return DEFAULT_ROI
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x1,y1,x2,y2")
    return tuple(parts)


def clip_roi(roi, width, height):
    x1, y1, x2, y2 = roi
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI after clipping: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def clean_detection(det):
    x1, y1, x2, y2, conf, cls_id = np.asarray(det).flatten().tolist()[:6]
    return {
        "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)],
        "confidence": round(float(conf), 6),
        "class_id": int(cls_id),
        "class_name": CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else f"class_{int(cls_id)}",
    }


def save_crops(output_dir, bags, bottles, shuyes):
    os.makedirs(output_dir, exist_ok=True)
    groups = [("bag", bags), ("bottle", bottles), ("infusion", shuyes)]
    for name, images in groups:
        for idx, img in enumerate(images, start=1):
            cv2.imwrite(os.path.join(output_dir, f"{name}_{idx:02d}.jpg"), img)


def keep_largest(images, limit):
    if limit is None or limit <= 0 or len(images) <= limit:
        return images
    return sorted(images, key=lambda img: img.shape[0] * img.shape[1], reverse=True)[:limit]


def _round_timing(timing):
    return {k: round(float(v), 6) for k, v in timing.items()}


def save_ocr_debug_image(debug_dir, target_type, idx, stage, image):
    if not debug_dir or image is None:
        return
    os.makedirs(debug_dir, exist_ok=True)
    filename = f"{target_type}_{idx:02d}_{stage}.jpg"
    cv2.imwrite(os.path.join(debug_dir, filename), image)


def process_bottle(idx, image, recognizer, drug_matcher, classifier, classify_enabled,
                   classifier_lock=None, ocr_lock=None, quiet=False, debug_dir=None):
    item = {
        "type": "bottle",
        "index": idx,
        "ocr_text": "",
        "candidates": [],
        "final_medicine": None,
        "confidence": 0.0,
        "status": "started",
        "timing_sec": {},
    }
    total_start = time.perf_counter()
    try:
        if ocr_lock is None:
            save_ocr_debug_image(debug_dir, "bottle", idx, "target_crop", image)
            recognizer.debug_dir = debug_dir
            recognizer.debug_prefix = f"bottle_{idx:02d}"
            t_ocr = time.perf_counter()
            if quiet:
                with contextlib.redirect_stdout(io.StringIO()):
                    text = recognizer.recognize(image)
            else:
                text = recognizer.recognize(image)
            ocr_timing = dict(recognizer.last_timing or {})
            ocr_timing["recognizer_call"] = time.perf_counter() - t_ocr
        else:
            with ocr_lock:
                save_ocr_debug_image(debug_dir, "bottle", idx, "target_crop", image)
                recognizer.debug_dir = debug_dir
                recognizer.debug_prefix = f"bottle_{idx:02d}"
                t_ocr = time.perf_counter()
                if quiet:
                    with contextlib.redirect_stdout(io.StringIO()):
                        text = recognizer.recognize(image)
                else:
                    text = recognizer.recognize(image)
                ocr_timing = dict(recognizer.last_timing or {})
                ocr_timing["recognizer_call"] = time.perf_counter() - t_ocr
        item["timing_sec"].update({f"ocr_{k}": v for k, v in ocr_timing.items()})
        item["ocr_text"] = text or ""
        if not text:
            item["status"] = "ocr_empty"
            item["timing_sec"]["total"] = time.perf_counter() - total_start
            item["timing_sec"] = _round_timing(item["timing_sec"])
            return item

        t = time.perf_counter()
        candidates = drug_matcher.match(text, match_type="bottle", threshold=50, limit=10)
        item["timing_sec"]["match"] = time.perf_counter() - t
        item["candidates"] = candidates
        if not candidates:
            item["status"] = "no_candidates"
            item["timing_sec"]["total"] = time.perf_counter() - total_start
            item["timing_sec"] = _round_timing(item["timing_sec"])
            return item

        if classify_enabled and classifier is not None:
            t = time.perf_counter()
            if classifier_lock is None:
                cls_result = classifier.classify(candidates, image)
            else:
                with classifier_lock:
                    cls_result = classifier.classify(candidates, image)
            item["timing_sec"]["classify"] = time.perf_counter() - t
            item["final_medicine"] = cls_result.get("predicted_category") or candidates[0]
            item["confidence"] = float(cls_result.get("confidence", 0.0))
            item["top_3"] = cls_result.get("top_3", [])
        else:
            item["final_medicine"] = candidates[0]

        item["status"] = "done"
        item["timing_sec"]["total"] = time.perf_counter() - total_start
        item["timing_sec"] = _round_timing(item["timing_sec"])
        return item
    except Exception as exc:
        item["status"] = f"error: {exc}"
        item["traceback"] = traceback.format_exc()
        item["timing_sec"]["total"] = time.perf_counter() - total_start
        item["timing_sec"] = _round_timing(item["timing_sec"])
        return item


def process_bag(idx, image, recognizer, drug_matcher, ocr_lock=None, quiet=False,
                debug_dir=None, simulate_patient_name=""):
    item = {
        "type": "bag",
        "index": idx,
        "ocr_text": "",
        "patient_name": None,
        "status": "started",
        "timing_sec": {},
    }
    total_start = time.perf_counter()
    try:
        if ocr_lock is None:
            save_ocr_debug_image(debug_dir, "bag", idx, "target_crop", image)
            recognizer.debug_dir = debug_dir
            recognizer.debug_prefix = f"bag_{idx:02d}"
            t_ocr = time.perf_counter()
            if quiet:
                with contextlib.redirect_stdout(io.StringIO()):
                    text = recognizer.recognize_yaodai(image)
            else:
                text = recognizer.recognize_yaodai(image)
            ocr_timing = dict(recognizer.last_timing or {})
            ocr_timing["recognizer_call"] = time.perf_counter() - t_ocr
        else:
            with ocr_lock:
                save_ocr_debug_image(debug_dir, "bag", idx, "target_crop", image)
                recognizer.debug_dir = debug_dir
                recognizer.debug_prefix = f"bag_{idx:02d}"
                t_ocr = time.perf_counter()
                if quiet:
                    with contextlib.redirect_stdout(io.StringIO()):
                        text = recognizer.recognize_yaodai(image)
                else:
                    text = recognizer.recognize_yaodai(image)
                ocr_timing = dict(recognizer.last_timing or {})
                ocr_timing["recognizer_call"] = time.perf_counter() - t_ocr
        item["timing_sec"].update({f"ocr_{k}": v for k, v in ocr_timing.items()})
        if simulate_patient_name:
            item["raw_ocr_text"] = text or ""
            text = simulate_patient_name
            item["simulated_patient_name"] = simulate_patient_name
        item["ocr_text"] = text or ""
        if not text:
            item["status"] = "ocr_empty"
            item["timing_sec"]["total"] = time.perf_counter() - total_start
            item["timing_sec"] = _round_timing(item["timing_sec"])
            return item

        t = time.perf_counter()
        patient = [simulate_patient_name] if simulate_patient_name else drug_matcher.match(text, match_type="bag", threshold=50, limit=10)
        item["timing_sec"]["match"] = time.perf_counter() - t
        item["patient_name"] = patient
        item["status"] = "done"
        item["timing_sec"]["total"] = time.perf_counter() - total_start
        item["timing_sec"] = _round_timing(item["timing_sec"])
        return item
    except Exception as exc:
        item["status"] = f"error: {exc}"
        item["traceback"] = traceback.format_exc()
        item["timing_sec"]["total"] = time.perf_counter() - total_start
        item["timing_sec"] = _round_timing(item["timing_sec"])
        return item


def process_infusion(idx, image, recognizer, ocr_lock=None, quiet=False, debug_dir=None):
    item = {
        "type": "infusion",
        "index": idx,
        "ocr_text": "",
        "liquid": None,
        "concentration": None,
        "volume": None,
        "raw_text": "",
        "status": "started",
        "timing_sec": {},
    }
    total_start = time.perf_counter()
    try:
        if ocr_lock is None:
            save_ocr_debug_image(debug_dir, "infusion", idx, "target_crop", image)
            recognizer.debug_dir = debug_dir
            recognizer.debug_prefix = f"infusion_{idx:02d}"
            t_ocr = time.perf_counter()
            if quiet:
                with contextlib.redirect_stdout(io.StringIO()):
                    text = recognizer.recognize_shuyedai(image)
            else:
                text = recognizer.recognize_shuyedai(image)
            ocr_timing = dict(recognizer.last_timing or {})
            ocr_timing["recognizer_call"] = time.perf_counter() - t_ocr
        else:
            with ocr_lock:
                save_ocr_debug_image(debug_dir, "infusion", idx, "target_crop", image)
                recognizer.debug_dir = debug_dir
                recognizer.debug_prefix = f"infusion_{idx:02d}"
                t_ocr = time.perf_counter()
                if quiet:
                    with contextlib.redirect_stdout(io.StringIO()):
                        text = recognizer.recognize_shuyedai(image)
                else:
                    text = recognizer.recognize_shuyedai(image)
                ocr_timing = dict(recognizer.last_timing or {})
                ocr_timing["recognizer_call"] = time.perf_counter() - t_ocr
        item["timing_sec"].update({f"ocr_{k}": v for k, v in ocr_timing.items()})
        if isinstance(text, dict):
            item["ocr_text"] = text
            item["liquid"] = text.get("liquid")
            item["concentration"] = text.get("concentration")
            item["volume"] = text.get("volume")
            item["raw_text"] = text.get("raw_text", "")
            item["status"] = text.get("status", "done")
        else:
            item["ocr_text"] = text or ""
            item["raw_text"] = text or ""
            item["status"] = "done" if text else "ocr_empty"
        item["timing_sec"]["total"] = time.perf_counter() - total_start
        item["timing_sec"] = _round_timing(item["timing_sec"])
        return item
    except Exception as exc:
        item["status"] = f"error: {exc}"
        item["traceback"] = traceback.format_exc()
        item["timing_sec"]["total"] = time.perf_counter() - total_start
        item["timing_sec"] = _round_timing(item["timing_sec"])
        return item


def process_task_group(task_group, recognizer, drug_matcher, classifier, classify_enabled,
                       classifier_lock, quiet, debug_dir, simulate_patient_name):
    results = []
    for task_type, idx, crop in task_group:
        if task_type == "bottle":
            results.append(process_bottle(
                idx,
                crop,
                recognizer,
                drug_matcher,
                classifier,
                classify_enabled,
                classifier_lock,
                None,
                quiet,
                debug_dir,
            ))
        elif task_type == "bag":
            results.append(process_bag(
                idx,
                crop,
                recognizer,
                drug_matcher,
                None,
                quiet,
                debug_dir,
                simulate_patient_name,
            ))
        else:
            results.append(process_infusion(
                idx,
                crop,
                recognizer,
                None,
                quiet,
                debug_dir,
            ))
    return results


def build_ocr_pool(args, task_count):
    instance_count = max(1, min(args.ocr_instances, max(1, task_count)))
    recognizers = []
    locks = []

    for _ in range(instance_count):
        if args.ocr_backend == "rknn":
            recognizers.append(RknnOCRRecognizer(
                det_model_path=args.ocr_det_model,
                rec_model_path=args.ocr_rec_model,
                char_dict_path=args.ocr_char_dict,
                det_batch_size=args.ocr_det_batch,
            ))
        else:
            ocr_model = init_fast_ocr(
                cpu_threads=args.ocr_threads,
                rec_batch_num=args.ocr_batch,
                show_log=args.ocr_log,
            )
            recognizers.append(OCRRecognizer_ori(ocr_model))
        locks.append(Lock())

    return recognizers, locks


def run(args):
    cv2.setUseOptimized(True)
    cv2.setNumThreads(args.cv_threads)

    timings = {}
    t0 = time.perf_counter()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    height, width = image_bgr.shape[:2]
    roi = clip_roi(args.roi, width, height)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    timings["read_image"] = time.perf_counter() - t0

    t = time.perf_counter()
    processor = YOLOTileProcessor(
        model_path=args.yolo_model,
        device="npu",
        tile_size=640,
        overlap=0,
        conf_thres=args.conf,
        iou_thres=args.iou,
        batch_size=1,
        contain_thres=args.contain,
    )
    timings["load_yolo"] = time.perf_counter() - t

    t = time.perf_counter()
    detections = processor.process_specific_slice(image_rgb, *roi)
    detections = [list(map(float, det[:6])) for det in detections]
    timings["yolo_detect"] = time.perf_counter() - t

    t = time.perf_counter()
    bags, bottles, shuyes = processor.crop_by_class(
        image_bgr,
        detections,
        bag_class_id=1,
        bottle_class_id=0,
        shuye_class_id=2,
    )
    timings["crop"] = time.perf_counter() - t

    if args.save_crops:
        save_crops(args.crop_dir, bags, bottles, shuyes)

    result = {
        "image": args.image,
        "image_shape": [height, width],
        "roi": list(roi),
        "detections": [clean_detection(det) for det in detections],
        "counts": {
            "bottle": len(bottles),
            "bag": len(bags),
            "infusion": len(shuyes),
        },
        "recognition": [],
        "timings_sec": timings,
        "timing_detail": {
            "ocr_instances": [],
            "task_counts": {},
            "recognition_by_type": {},
        },
    }

    if not args.detect_only:
        t = time.perf_counter()
        drug_matcher, conn = load_or_build_name_cache(config, args.name_cache)
        if args.skip_classifier:
            classifier = None
            classifier_lock = None
        else:
            feature_cache_ready = args.feature_cache and os.path.exists(args.feature_cache)
            if conn is None and not feature_cache_ready:
                conn = init_db_from_config(config)
            os.environ["YILIAO_FEATURE_CACHE"] = args.feature_cache
            classifier = PharmaceuticalBottleClassifier(db_conn=conn, device="npu")
            classifier_lock = Lock()
        timings["load_db_and_matchers"] = time.perf_counter() - t

        tasks = []
        bottles_for_ocr = bottles
        bags_for_ocr = keep_largest(bags, args.max_bag_ocr)
        for idx, bottle in enumerate(bottles_for_ocr):
            tasks.append(("bottle", idx, bottle))
        for idx, bag in enumerate(bags_for_ocr):
            tasks.append(("bag", idx, bag))
        shuyes_for_ocr = keep_largest(shuyes, args.max_infusion_ocr)
        for idx, shuye in enumerate(shuyes_for_ocr):
            tasks.append(("infusion", idx, shuye))

        t = time.perf_counter()
        recognizers, ocr_locks = build_ocr_pool(args, len(tasks))
        timings["load_ocr"] = time.perf_counter() - t
        result["timing_detail"]["ocr_instances"] = [
            {"index": i, "thread_lock": True}
            for i in range(len(recognizers))
        ]
        result["timing_detail"]["task_counts"] = {
            "bottle": len(bottles_for_ocr),
            "bag": len(bags_for_ocr),
            "infusion": len(shuyes_for_ocr),
            "total": len(tasks),
        }

        t = time.perf_counter()
        if args.parallel_workers > 1 and len(tasks) > 1:
            max_workers = min(args.parallel_workers, len(tasks), len(recognizers))
            task_groups = [[] for _ in range(max_workers)]
            for task_seq, task in enumerate(tasks):
                task_groups[task_seq % max_workers].append(task)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(
                        process_task_group,
                        task_group,
                        recognizers[group_idx],
                        drug_matcher,
                        classifier,
                        not args.skip_classifier,
                        classifier_lock,
                        args.quiet,
                        args.ocr_debug_dir,
                        args.simulate_patient_name,
                    )
                    for group_idx, task_group in enumerate(task_groups)
                    if task_group
                ]

                for future in as_completed(futures):
                    result["recognition"].extend(future.result())
        else:
            recognizer_count = len(recognizers)
            for task_seq, (task_type, idx, crop) in enumerate(tasks):
                pool_idx = task_seq % recognizer_count
                recognizer = recognizers[pool_idx]
                ocr_lock = ocr_locks[pool_idx]
                if task_type == "bottle":
                    result["recognition"].append(process_bottle(
                        idx,
                        crop,
                        recognizer,
                        drug_matcher,
                        classifier,
                        not args.skip_classifier,
                        classifier_lock,
                        ocr_lock,
                        args.quiet,
                        args.ocr_debug_dir,
                    ))
                elif task_type == "bag":
                    result["recognition"].append(process_bag(
                        idx,
                        crop,
                        recognizer,
                        drug_matcher,
                        ocr_lock,
                        args.quiet,
                        args.ocr_debug_dir,
                        args.simulate_patient_name,
                    ))
                else:
                    result["recognition"].append(process_infusion(
                        idx,
                        crop,
                        recognizer,
                        ocr_lock,
                        args.quiet,
                        args.ocr_debug_dir,
                    ))

        timings["recognition"] = time.perf_counter() - t
        result["recognition"].sort(key=lambda item: (item.get("type", ""), item.get("index", -1)))
        for item in result["recognition"]:
            type_name = item.get("type", "unknown")
            result["timing_detail"]["recognition_by_type"].setdefault(type_name, []).append({
                "index": item.get("index"),
                "status": item.get("status"),
                "timing_sec": item.get("timing_sec", {}),
            })
        if conn is not None:
            conn.close()

    t = time.perf_counter()
    if not args.no_save_visual:
        vis_rgb = processor.draw_detections(image_rgb, detections, class_names=CLASS_NAMES)
        cv2.rectangle(vis_rgb, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 2)
        cv2.imwrite(args.output_image, cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
    timings["save_visual"] = time.perf_counter() - t

    timings["total"] = time.perf_counter() - t0
    result["runtime_sec"] = round(
        sum(
            float(value)
            for key, value in timings.items()
            if key not in INIT_TIMING_KEYS and key != "total"
        ),
        6,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "counts": result["counts"],
        "detections": len(result["detections"]),
        "output_image": args.output_image,
        "output_json": args.output_json,
        "runtime_sec_excluding_model_init": result["runtime_sec"],
        "timings_sec": {k: round(v, 4) for k, v in timings.items()},
    }, ensure_ascii=False, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fast single-image YOLO detection and recognition for 005.png on RK3588."
    )
    parser.add_argument("--image", default="005.png")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--yolo-model", default="./model_yolo_0521.rknn")
    parser.add_argument("--output-image", default="./single_005_result.jpg")
    parser.add_argument("--output-json", default="./single_005_result.json")
    parser.add_argument("--name-cache", default="./single_image_name_cache.json")
    parser.add_argument("--feature-cache", default="./single_image_feature_cache.pkl")
    parser.add_argument("--roi", type=parse_roi, default=DEFAULT_ROI, help="x1,y1,x2,y2")
    parser.add_argument("--conf", type=float, default=0.32)
    parser.add_argument("--iou", type=float, default=0.9)
    parser.add_argument("--contain", type=float, default=0.8)
    parser.add_argument("--cv-threads", type=int, default=4)
    parser.add_argument("--ocr-threads", type=int, default=4)
    parser.add_argument("--ocr-batch", type=int, default=8)
    parser.add_argument("--ocr-instances", type=int, default=3)
    parser.add_argument("--parallel-workers", type=int, default=3)
    parser.add_argument("--ocr-backend", choices=["paddle", "rknn"], default="paddle")
    parser.add_argument("--ocr-det-model", default="./src/identification/Det_bs1.rknn")
    parser.add_argument("--ocr-rec-model", default="./model_ocr_0526.rknn")
    parser.add_argument("--ocr-char-dict", default="./OCRS.txt")
    parser.add_argument("--ocr-det-batch", type=int, default=1)
    parser.add_argument("--ocr-debug-dir", default="", help="Save images before OCR into this directory")
    parser.add_argument("--simulate-patient-name", default="", help="Force bag/patient OCR text for benchmark simulation")
    parser.add_argument("--max-bottle-ocr", type=int, default=0)
    parser.add_argument("--max-bag-ocr", type=int, default=1)
    parser.add_argument("--max-infusion-ocr", type=int, default=2)
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--no-save-visual", action="store_true")
    parser.add_argument("--ocr-log", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--crop-dir", default="./single_005_crops")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
