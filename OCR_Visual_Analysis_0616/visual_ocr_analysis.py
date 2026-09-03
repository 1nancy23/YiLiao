#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline OCR/matching visual analysis for bottle, bag and infusion targets.

This script intentionally lives outside the main runtime path. It imports the
existing model wrappers, runs them on supplied images, and writes all diagnostic
artifacts under its own output directory.
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


CLASS_NAMES = {
    0: "bottle",
    1: "bag",
    2: "infusion",
}


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


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def imwrite(path, image):
    if image is None or not hasattr(image, "size") or image.size == 0:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return str(path)


def resolve_project_path(path_value):
    path = Path(str(path_value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def enable_visual_debug(recognizer, debug_dir):
    """Patch debug save hooks on one recognizer instance.

    The main code disables these hooks for speed. For this independent analysis
    runner we enable them per instance without touching source files.
    """
    debug_dir = ensure_dir(debug_dir)
    recognizer.debug_dir = str(debug_dir)
    recognizer._debug_seq = 0

    def next_path(name):
        seq = int(getattr(recognizer, "_debug_seq", 0))
        recognizer._debug_seq = seq + 1
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))
        return debug_dir / f"{seq:04d}_{safe}.jpg"

    def save_debug_image(name, image):
        imwrite(next_path(name), image)

    def save_predet_image(name, image):
        imwrite(debug_dir / "predet" / f"{name}.jpg", image)

    def save_detvis_image(name, image, boxes):
        vis = recognizer._draw_det_boxes(image, boxes)
        imwrite(debug_dir / "detvis" / f"{name}.jpg", vis)

    def save_region_strip(name, regions):
        valid = [r for r in regions or [] if r is not None and r.size > 0]
        if not valid:
            return
        target_h = max(24, max(r.shape[0] for r in valid))
        resized = []
        for region in valid:
            scale = target_h / float(max(1, region.shape[0]))
            resized.append(cv2.resize(region, (max(1, int(region.shape[1] * scale)), target_h)))
        strip = cv2.hconcat(resized) if len(resized) > 1 else resized[0]
        imwrite(debug_dir / "strips" / f"{name}.jpg", strip)

    recognizer._save_debug_image = save_debug_image
    recognizer._save_predet_image = save_predet_image
    recognizer._save_detvis_image = save_detvis_image
    recognizer._save_region_strip = save_region_strip


def crop_with_boxes(image, detections, case_dir, raw_image=None):
    crops = {"bottle": [], "bag": [], "shuye": []}
    crop_dir = ensure_dir(case_dir / "crops")
    h, w = image.shape[:2]
    for det_index, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls_id = det[:6]
        cls_id = int(cls_id)
        x1 = max(0, min(w, int(round(x1))))
        y1 = max(0, min(h, int(round(y1))))
        x2 = max(0, min(w, int(round(x2))))
        y2 = max(0, min(h, int(round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        key = {0: "bottle", 1: "bag", 2: "shuye"}.get(cls_id)
        if key is None:
            continue
        sx1, sy1, sx2, sy2 = x1, y1, x2, y2
        if key == "shuye":
            box_w = x2 - x1
            box_h = y2 - y1
            pad_x = max(8, int(box_w * 0.05))
            pad_y = max(8, int(box_h * 0.08))
            sx1 = max(0, x1 - pad_x)
            sy1 = max(0, y1 - pad_y)
            sx2 = min(w, x2 + pad_x)
            sy2 = min(h, y2 + pad_y)
        crop = image[sy1:sy2, sx1:sx2].copy()
        raw_crop = (
            raw_image[sy1:sy2, sx1:sx2].copy()
            if raw_image is not None else crop
        )
        item = {
            "det_index": det_index,
            "class_id": cls_id,
            "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
            "confidence": float(conf),
            "box": [x1, y1, x2, y2],
            "crop_box": [sx1, sy1, sx2, sy2],
            "image": crop,
            "raw_image": raw_crop,
            "path": str(crop_dir / f"{key}_{len(crops[key]) + 1:02d}.jpg"),
        }
        imwrite(item["path"], crop)
        crops[key].append(item)
    return crops


def run_yolo(image, config, args):
    from run_realtime_detection_yolo_new_3 import enhance_trigger_frame_contrast
    from src.segmentation.segmenter_yolo import YOLOTileProcessor

    model_cfg = config.get("model", {})
    normalized = enhance_trigger_frame_contrast(image) if not args.no_normalize else image
    processor = YOLOTileProcessor(
        model=None,
        model_path=str(resolve_project_path(args.yolo_model or model_cfg.get("yolo_rknn_path", "./model_yolo_0615.rknn"))),
        device="npu",
        tile_size=int(args.yolo_size or model_cfg.get("yolo_input_size", 640)),
        input_size=int(args.yolo_size or model_cfg.get("yolo_input_size", 640)),
        overlap=64,
        conf_thres=float(args.conf),
        iou_thres=0.85,
        batch_size=1,
        contain_thres=float(args.contain_thres),
        class_conf_thres={0: args.bottle_conf, 1: args.bag_conf, 2: args.infusion_conf},
        class_iou_thres={0: 0.65, 1: 0.65, 2: 0.45},
    )
    try:
        t0 = time.perf_counter()
        detections = processor.process_whole_image_once(normalized)
        yolo_sec = time.perf_counter() - t0
        overlay = processor.draw_detections(normalized.copy(), detections, class_names=["bottle", "bag", "infusion"])
    finally:
        processor.release()
    return normalized, detections, overlay, yolo_sec


def init_models(config, args):
    from native_app import create_local_bottle_db_matcher, init_db
    from src.identification.DrugMatcher import DrugMatcher
    from src.identification.Recog import PharmaceuticalBottleClassifier
    from src.identification.rknn_ocr_adapter import RknnOCRRecognizer

    os.environ.setdefault("YILIAO_COLLECT_TIMING", "1")
    os.environ.setdefault("YILIAO_RUNTIME_LOGS", "0")
    os.environ.setdefault("YILIAO_VERBOSE_RUNTIME", "0")
    os.environ.setdefault("YILIAO_FEATURE_CACHE", str(PROJECT_ROOT / "single_image_feature_cache.pkl"))
    os.environ.setdefault("YILIAO_FEATURE_ROOT", str(PROJECT_ROOT / "src" / "identification" / "feat_data"))

    classifier = PharmaceuticalBottleClassifier(db_conn=None, device="npu")
    recognizer = RknnOCRRecognizer(
        det_model_path=str(PROJECT_ROOT / "model_det_bs16.rknn"),
        rec_model_path=str(PROJECT_ROOT / "model_ocr_bs16.rknn"),
        cls_model_path=str(PROJECT_ROOT / "model_cls_bs32.rknn"),
        det_batch_size=16,
        rec_batch_size=16,
        cls_batch_size=32,
    )

    matcher = None
    conn = None
    if not args.no_db:
        try:
            conn = init_db(config.get("db_config", {}))
            tables = config.get("table_config", {})
            matcher = create_local_bottle_db_matcher(
                DrugMatcher,
                conn,
                classifier.get_cached_names(),
                drug_table=tables.get("drug_table", "drugs"),
                drug_column=tables.get("drug_column", "medicine_name"),
                patient_table=tables.get("patient_table", "patients"),
                patient_column=tables.get("patient_column", "name"),
            )
        except Exception as exc:
            if not args.allow_db_fail:
                raise
            print(f"[WARN] DB disabled after connection failure: {exc}")
    return classifier, recognizer, matcher, conn


def close_models(classifier, recognizer, conn):
    for obj in (recognizer, classifier):
        release = getattr(obj, "release", None)
        if callable(release):
            release()
    if conn is not None:
        conn.close()


def process_ocr_and_match(crops, classifier, recognizer, matcher):
    from src.identification.realtime_recognition import process_task_group

    tasks = []
    for key, task_type in (("bottle", "bottle"), ("bag", "bag"), ("shuye", "shuye")):
        for idx, item in enumerate(crops[key]):
            tasks.append((task_type, idx, item["image"], item.get("raw_image", item["image"])))

    t0 = time.perf_counter()
    flow_results = process_task_group(
        tasks,
        recognizer,
        matcher,
        classifier=classifier,
        classify_enabled=classifier is not None,
        classifier_thread_safe=True,
        quiet=True,
    ) if tasks else []
    ocr_sec = time.perf_counter() - t0
    result_map = {
        (item.get("type"), int(item.get("index", 0))): item
        for item in flow_results
    }

    results = {"bottles": [], "bags": [], "infusions": [], "timing_sec": {"ocr_batch": ocr_sec}}

    for idx, item in enumerate(crops["bottle"]):
        value = dict(result_map.get(("bottle", idx), {}))
        value.pop("det_visualization", None)
        row = {k: v for k, v in item.items() if k not in ("image", "raw_image")}
        row.update(value)
        results["bottles"].append(jsonable(row))

    for idx, item in enumerate(crops["bag"]):
        value = dict(result_map.get(("bag", idx), {}))
        row = {k: v for k, v in item.items() if k not in ("image", "raw_image")}
        row.update(value)
        row["patient_match"] = value.get("patient_name")
        results["bags"].append(jsonable(row))

    for idx, item in enumerate(crops["shuye"]):
        value = dict(result_map.get(("shuye", idx), {}))
        row = {k: v for k, v in item.items() if k not in ("image", "raw_image")}
        row.update(value)
        row["ocr_text"] = {
            "liquid": value.get("liquid"),
            "concentration": value.get("concentration"),
            "volume": value.get("volume"),
            "raw_text": value.get("raw_text", ""),
            "status": value.get("status", "ocr_empty"),
        }
        results["infusions"].append(jsonable(row))
    return results


def summarize_case(case_result):
    bottles = case_result.get("bottles", [])
    bags = case_result.get("bags", [])
    infusions = case_result.get("infusions", [])
    return {
        "image": case_result.get("image"),
        "detections": len(case_result.get("detections", [])),
        "bottles": len(bottles),
        "bags": len(bags),
        "infusions": len(infusions),
        "empty_bottle_ocr": sum(1 for item in bottles if not item.get("ocr_text")),
        "empty_bag_ocr": sum(1 for item in bags if not item.get("ocr_text")),
        "empty_infusion_ocr": sum(1 for item in infusions if not (item.get("ocr_text") or {}).get("raw_text")),
        "yolo_sec": case_result.get("timing_sec", {}).get("yolo", 0.0),
        "ocr_batch_sec": case_result.get("timing_sec", {}).get("ocr_batch", 0.0),
        "final_medicines": "|".join(item.get("final_medicine") or "" for item in bottles),
        "patients": "|".join(str(item.get("patient_match") or "") for item in bags),
        "infusion_status": "|".join(str((item.get("ocr_text") or {}).get("status", "")) for item in infusions),
    }


def write_case_report(case_dir, case_result):
    summary = summarize_case(case_result)
    lines = [
        f"# OCR Visual Analysis - {Path(str(summary.get('image'))).name}",
        "",
        "## Summary",
        "",
        f"- detections: {summary['detections']}",
        f"- bottles/bags/infusions: {summary['bottles']}/{summary['bags']}/{summary['infusions']}",
        f"- empty OCR: bottle={summary['empty_bottle_ocr']}, bag={summary['empty_bag_ocr']}, infusion={summary['empty_infusion_ocr']}",
        f"- time: yolo={summary['yolo_sec']:.3f}s, ocr_batch={summary['ocr_batch_sec']:.3f}s, total={case_result.get('timing_sec', {}).get('total', 0.0):.3f}s",
        "",
        "## Bottle OCR And Matching",
        "",
    ]
    for item in case_result.get("bottles", []):
        idx = item.get("det_index")
        text = item.get("ocr_text") or ""
        final = item.get("final_medicine") or ""
        decision = item.get("decision_reason") or ""
        lines.extend([
            f"### Bottle det#{idx}",
            "",
            f"- box: {item.get('box')}, conf={float(item.get('confidence') or 0.0):.4f}",
            f"- ocr: {text if text else '[EMPTY]'}",
            f"- det_regions={item.get('det_region_count', 0)}, rec_nonempty={item.get('rec_nonempty_count', 0)}, cls_angle={item.get('classify_angle_pred')}",
            f"- final: {final if final else '[NONE]'}",
            f"- decision: {decision}",
            "",
        ])

    lines.extend(["## Bag OCR And Patient Matching", ""])
    for item in case_result.get("bags", []):
        idx = item.get("det_index")
        text = item.get("ocr_text") or ""
        lines.extend([
            f"### Bag det#{idx}",
            "",
            f"- box: {item.get('box')}, conf={float(item.get('confidence') or 0.0):.4f}",
            f"- ocr: {text if text else '[EMPTY]'}",
            f"- name_roi_count={item.get('name_roi_count', 0)}, cls_angle={item.get('classify_angle_pred')}",
            f"- patient_match: {item.get('patient_match')}",
            "",
        ])

    lines.extend(["## Infusion OCR", ""])
    for item in case_result.get("infusions", []):
        idx = item.get("det_index")
        text = item.get("ocr_text") or {}
        lines.extend([
            f"### Infusion det#{idx}",
            "",
            f"- box: {item.get('box')}, conf={float(item.get('confidence') or 0.0):.4f}",
            f"- roi_count={item.get('candidate_roi_count', 0)}",
            f"- liquid={text.get('liquid')}, concentration={text.get('concentration')}, volume={text.get('volume')}, status={text.get('status')}",
            f"- raw_text: {text.get('raw_text') or '[EMPTY]'}",
            "",
        ])

    lines.extend([
        "## Quick Diagnosis",
        "",
    ])
    if summary["empty_bottle_ocr"]:
        lines.append("- Bottle empty OCR exists. First inspect `ocr_debug/detvis` and crop images for missing or too small text boxes.")
    if summary["empty_bag_ocr"]:
        lines.append("- Bag empty OCR exists. Inspect label card/name ROI debug images.")
    if summary["empty_infusion_ocr"]:
        lines.append("- Infusion empty OCR exists. Inspect blue text ROI extraction debug images.")
    if summary["yolo_sec"] > 0:
        lines.append(f"- YOLO share of measured yolo+ocr time: {summary['yolo_sec'] / max(1e-6, summary['yolo_sec'] + summary['ocr_batch_sec']) * 100.0:.1f}%.")
    if not any(line.startswith("- ") for line in lines[-5:]):
        lines.append("- No immediate empty OCR issue detected in this case.")
    lines.append("")

    with open(Path(case_dir) / "analysis_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_global_report(out_root, summaries):
    if not summaries:
        return
    total = len(summaries)
    yolo_avg = sum(float(s.get("yolo_sec") or 0.0) for s in summaries) / total
    ocr_avg = sum(float(s.get("ocr_batch_sec") or 0.0) for s in summaries) / total
    empty_bottle = sum(int(s.get("empty_bottle_ocr") or 0) for s in summaries)
    empty_bag = sum(int(s.get("empty_bag_ocr") or 0) for s in summaries)
    empty_infusion = sum(int(s.get("empty_infusion_ocr") or 0) for s in summaries)
    lines = [
        "# OCR Visual Analysis Summary",
        "",
        f"- cases: {total}",
        f"- average yolo: {yolo_avg:.3f}s",
        f"- average ocr_batch: {ocr_avg:.3f}s",
        f"- empty OCR totals: bottle={empty_bottle}, bag={empty_bag}, infusion={empty_infusion}",
        "",
        "## Cases",
        "",
    ]
    for idx, item in enumerate(summaries, start=1):
        lines.append(
            f"- {idx:03d}: det={item['detections']}, bottle/bag/infusion={item['bottles']}/{item['bags']}/{item['infusions']}, "
            f"empty={item['empty_bottle_ocr']}/{item['empty_bag_ocr']}/{item['empty_infusion_ocr']}, "
            f"yolo={item['yolo_sec']:.3f}s, ocr={item['ocr_batch_sec']:.3f}s, image={item['image']}"
        )
    lines.extend([
        "",
        "## Optimization Direction",
        "",
        "- If empty OCR appears with det_regions=0, optimize detection crop quality or text-det preprocessing before REC.",
        "- If det_regions is nonzero but rec_nonempty is low, inspect `ocr_debug/strips` and increase text-line padding in an experiment branch.",
        "- If YOLO boxes differ between manual and auto frames, compare `source.jpg`, `normalized.jpg`, and `yolo_overlay.jpg` first.",
        "- If OCR batch time grows with extra false targets, tune YOLO class thresholds in this independent script before touching the main flow.",
        "",
    ])
    with open(Path(out_root) / "analysis_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def expand_images(patterns):
    files = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            files.extend(matches)
        elif os.path.exists(pattern):
            files.append(pattern)
    return sorted(dict.fromkeys(files))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True, help="image paths or glob patterns")
    parser.add_argument("--output", default=str(ANALYSIS_DIR / "outputs"))
    parser.add_argument("--no-db", action="store_true", help="skip DB patient/drug matching")
    parser.add_argument("--allow-db-fail", action="store_true", help="continue if DB is unavailable")
    parser.add_argument("--no-normalize", action="store_true", help="skip camera exposure/white balance normalization")
    parser.add_argument("--yolo-model", default="")
    parser.add_argument("--yolo-size", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--bottle-conf", type=float, default=0.35)
    parser.add_argument("--bag-conf", type=float, default=0.25)
    parser.add_argument("--infusion-conf", type=float, default=0.32)
    parser.add_argument("--contain-thres", type=float, default=0.92)
    args = parser.parse_args()

    if cv2 is None or np is None or yaml is None:
        raise SystemExit("cv2/numpy/yaml is required. Run this script with the board Python environment.")

    image_paths = expand_images(args.images)
    if not image_paths:
        raise SystemExit("No input images found.")

    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_root = ensure_dir(args.output)
    all_summaries = []
    all_results = []
    classifier = recognizer = matcher = conn = None

    try:
        classifier, recognizer, matcher, conn = init_models(config, args)
        for case_index, image_path in enumerate(image_paths, start=1):
            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARN] skip unreadable image: {image_path}")
                continue
            stem = Path(image_path).stem
            case_dir = ensure_dir(out_root / f"case_{case_index:03d}_{stem}")
            print(f"[CASE] {image_path}")

            normalized, detections, overlay, yolo_sec = run_yolo(image, config, args)
            imwrite(case_dir / "source.jpg", image)
            imwrite(case_dir / "normalized.jpg", normalized)
            imwrite(case_dir / "yolo_overlay.jpg", overlay)
            crops = crop_with_boxes(normalized, detections, case_dir, raw_image=image)

            enable_visual_debug(recognizer, case_dir / "ocr_debug")
            result = process_ocr_and_match(crops, classifier, recognizer, matcher)

            result.update({
                "image": image_path,
                "case_dir": str(case_dir),
                "detections": jsonable(detections),
                "counts": {key: len(value) for key, value in crops.items()},
            })
            result.setdefault("timing_sec", {})
            result["timing_sec"]["yolo"] = yolo_sec
            result["timing_sec"]["total"] = yolo_sec + result["timing_sec"].get("ocr_batch", 0.0)

            with open(case_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(jsonable(result), f, ensure_ascii=False, indent=2)
            write_case_report(case_dir, result)

            summary = summarize_case(result)
            all_summaries.append(summary)
            all_results.append(result)
            print(f"  detections={summary['detections']} bottles={summary['bottles']} bags={summary['bags']} infusions={summary['infusions']} yolo={summary['yolo_sec']:.3f}s ocr={summary['ocr_batch_sec']:.3f}s")
    finally:
        close_models(classifier, recognizer, conn)

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"cases": jsonable(all_results), "summary": jsonable(all_summaries)}, f, ensure_ascii=False, indent=2)

    if all_summaries:
        with open(out_root / "summary.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(all_summaries)
        write_global_report(out_root, all_summaries)

    print(f"[DONE] outputs: {out_root}")


if __name__ == "__main__":
    main()
