import argparse
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pymysql

from src.identification.DrugMatcher import DrugMatcher
from src.identification.OCRBatchStitcher import recognize_stitched_images_as_batch
from src.identification.OCRRecognizer import OCRRecognizer, OCRRecognizer_ori
from src.identification.Recog import PharmaceuticalBottleClassifier
from src.segmentation.segmenter_yolo import YOLOTileProcessor


ROOT_DIR = Path(__file__).resolve().parent


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"cannot encode image: {path}")
    encoded.tofile(str(path))


def parse_roi(roi_text, image_shape):
    h, w = image_shape[:2]
    if not roi_text:
        return 0, 0, w, h
    values = [int(v.strip()) for v in roi_text.split(",")]
    if len(values) != 4:
        raise ValueError("--roi must be x1,y1,x2,y2")
    x1, y1, x2, y2 = values
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def init_db(args):
    return pymysql.connect(
        host=args.db_host,
        user=args.db_user,
        password=args.db_password,
        database=args.db_name,
        charset=args.db_charset,
        port=args.db_port,
        cursorclass=pymysql.cursors.DictCursor,
    )


def init_paddle_ocr(args):
    import logging

    for level_name in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        logging._nameToLevel.setdefault(level_name, getattr(logging, level_name))

    original_check_level = logging._checkLevel

    def check_level_compat(level):
        if isinstance(level, str):
            mapped = logging._nameToLevel.get(level.upper())
            if mapped is not None:
                return mapped
        return original_check_level(level)

    logging._checkLevel = check_level_compat

    import paddle
    from paddleocr import PaddleOCR

    use_gpu = args.ocr_use_gpu
    if use_gpu is None:
        use_gpu = paddle.is_compiled_with_cuda()

    return PaddleOCR(
        use_angle_cls=True,
        det_algorithm="DB",
        det_max_side_len=args.paddle_det_max_side_len,
        det_score_mode="fast",
        det_db_thresh=args.paddle_det_db_thresh,
        det_db_box_thresh=args.paddle_det_db_box_thresh,
        lang="ch",
        rec_batch_num=args.paddle_rec_batch_num,
        use_gpu=use_gpu,
        det=True,
        rec=True,
        rec_score_thresh=args.paddle_rec_score_thresh,
        cpu_threads=args.paddle_cpu_threads,
        gpu_mem=1024,
        use_fp16=True,
        show_log=False,
    )


def crop_with_meta(image, detections, target_class, class_name):
    h, w = image.shape[:2]
    items = []
    for det_index, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls_id = np.asarray(det, dtype=np.float32).reshape(-1)[:6]
        if int(cls_id) != target_class:
            continue
        ix1 = int(max(0, min(w, x1)))
        iy1 = int(max(0, min(h, y1)))
        ix2 = int(max(0, min(w, x2)))
        iy2 = int(max(0, min(h, y2)))
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        items.append(
            {
                "index": len(items),
                "det_index": det_index,
                "class_id": int(cls_id),
                "class_name": class_name,
                "confidence": float(conf),
                "box": [ix1, iy1, ix2, iy2],
                "image": image[iy1:iy2, ix1:ix2].copy(),
            }
        )
    return items


def resize_for_ocr(image, max_side):
    if max_side <= 0:
        return image
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return image
    scale = max_side / float(long_side)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def prepare_ocr_images(items, max_side):
    for item in items:
        item["ocr_image"] = resize_for_ocr(item["image"], max_side)
    return items


def recognize_bottles(ocr, drug_matcher, classifier, bottle_items, ocr_batch_size=True):
    if not bottle_items:
        return []

    images = [item.get("ocr_image", item["image"]) for item in bottle_items]
    ocr_text_by_index = {}

    if ocr_batch_size and len(images) > 1:
        ocr_results = recognize_stitched_images_as_batch(
            ocr,
            images,
            line_gap_threshold=20,
            separator_height=20,
        )
        ocr_text_by_index = {
            item.get("image_index"): item.get("ocr_text", "")
            for item in ocr_results
        }
    else:
        for i, image in enumerate(images):
            ocr_text_by_index[i] = ocr.recognize(image, line_gap_threshold=20)

    results = []
    for item in bottle_items:
        idx = item["index"]
        ocr_text = ocr_text_by_index.get(idx, "") or ""
        candidates = []
        final_medicine = None
        cls_confidence = 0.0
        status = "ok"

        if drug_matcher is not None and ocr_text:
            candidates = drug_matcher.match(
                ocr_text,
                match_type="bottle",
                threshold=50,
                limit=10,
            )

        if classifier is not None and candidates:
            try:
                cls_result = classifier.classify(candidates, item["image"])
                final_medicine = cls_result.get("predicted_category") or candidates[0]
                cls_confidence = float(cls_result.get("confidence", 0.0))
            except Exception as exc:
                status = f"classify_failed: {exc}"
                final_medicine = candidates[0]
        elif candidates:
            final_medicine = candidates[0]
            status = "matched_without_classifier"
        elif not ocr_text:
            status = "empty_ocr"
        else:
            status = "no_match"

        results.append(
            {
                "index": idx,
                "det_index": item["det_index"],
                "box": item["box"],
                "det_confidence": item["confidence"],
                "ocr_text": ocr_text,
                "candidates": candidates,
                "final_medicine": final_medicine,
                "classification_confidence": cls_confidence,
                "status": status,
            }
        )
    return results


def recognize_bags(ocr, drug_matcher, bag_items, ocr_batch_size=True):
    if not bag_items:
        return []

    images = [item.get("ocr_image", item["image"]) for item in bag_items]
    ocr_text_by_index = {}

    if ocr_batch_size and len(images) > 1 and hasattr(ocr, "recognize_with_blocks"):
        ocr_results = recognize_stitched_images_as_batch(
            ocr,
            images,
            line_gap_threshold=20,
            separator_height=20,
        )
        ocr_text_by_index = {
            item.get("image_index"): item.get("ocr_text", "")
            for item in ocr_results
        }
    else:
        for i, image in enumerate(images):
            if hasattr(ocr, "recognize_yaodai"):
                ocr_text_by_index[i] = ocr.recognize_yaodai(image)
            else:
                ocr_text_by_index[i] = ocr.recognize(image)

    results = []
    for item in bag_items:
        text = ocr_text_by_index.get(item["index"], "") or ""
        matches = []
        if drug_matcher is not None and text:
            matches = drug_matcher.match(text, match_type="bag", threshold=50, limit=10)
        results.append(
            {
                "index": item["index"],
                "det_index": item["det_index"],
                "box": item["box"],
                "det_confidence": item["confidence"],
                "ocr_text": text,
                "patient_matches": matches,
            }
        )
    return results


def draw_results(image, detections, bottle_results, bag_results):
    vis = image.copy()
    class_colors = {0: (0, 180, 0), 1: (255, 80, 0), 2: (0, 0, 255)}
    label_by_det = {}
    for result in bottle_results:
        label_by_det[result["det_index"]] = result.get("final_medicine") or result.get("ocr_text") or "bottle"
    for result in bag_results:
        names = result.get("patient_matches") or []
        label_by_det[result["det_index"]] = names[0] if names else (result.get("ocr_text") or "bag")

    for det_index, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls_id = np.asarray(det, dtype=np.float32).reshape(-1)[:6]
        color = class_colors.get(int(cls_id), (200, 200, 200))
        p1 = int(x1), int(y1)
        p2 = int(x2), int(y2)
        cv2.rectangle(vis, p1, p2, color, 2)
        label = label_by_det.get(det_index, f"class_{int(cls_id)} {conf:.2f}")
        label = f"{label} {conf:.2f}"
        cv2.putText(
            vis,
            label[:40],
            (p1[0], max(20, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return vis


def run(args):
    t0 = time.time()
    image_path = Path(args.image)
    image = read_image(image_path)

    roi = None if args.full_image else parse_roi(args.roi, image.shape)

    processor = YOLOTileProcessor(
        model_path=args.yolo_model,
        device="npu",
        tile_size=640,
        overlap=0,
        conf_thres=args.conf,
        iou_thres=args.iou,
        batch_size=1,
    )

    yolo_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if roi is None:
        detections = processor.process_image(yolo_input)
    else:
        detections = processor.process_specific_slice(yolo_input, *roi)

    bottle_items = crop_with_meta(image, detections, target_class=0, class_name="bottle")
    bag_items = crop_with_meta(image, detections, target_class=1, class_name="bag")
    prepare_ocr_images(bottle_items, args.ocr_max_crop_side)
    prepare_ocr_images(bag_items, args.ocr_max_crop_side)

    conn = None
    drug_matcher = None
    classifier = None
    ocr = None
    try:
        if not args.no_db:
            conn = init_db(args)
            drug_matcher = DrugMatcher(
                conn,
                drug_table=args.drug_table,
                drug_column=args.drug_column,
                patient_table=args.patient_table,
                patient_column=args.patient_column,
                cache_drugs=True,
            )
            if not args.no_classify:
                classifier = PharmaceuticalBottleClassifier(db_conn=conn, device="cpu")

        if not args.no_ocr:
            if args.ocr_engine == "paddle":
                ocr = OCRRecognizer_ori(init_paddle_ocr(args))
            else:
                ocr = OCRRecognizer(
                    det_model_path=args.ocr_det_model,
                    rec_model_path=args.ocr_rec_model,
                    char_dict_path=args.ocr_dict,
                )
            bottle_results = recognize_bottles(
                ocr,
                drug_matcher,
                classifier,
                bottle_items,
                ocr_batch_size=not args.no_batch_ocr,
            )
            bag_results = recognize_bags(
                ocr,
                drug_matcher,
                bag_items,
                ocr_batch_size=not args.no_batch_ocr,
            )
        else:
            bottle_results = []
            bag_results = []
    finally:
        if conn is not None:
            conn.close()

    elapsed = time.time() - t0
    output = {
        "image": str(image_path),
        "elapsed_sec": elapsed,
        "roi": None if roi is None else list(roi),
        "detections": [
            {
                "box": [float(v) for v in det[:4]],
                "confidence": float(det[4]),
                "class_id": int(det[5]),
            }
            for det in detections
        ],
        "bottles": bottle_results,
        "bags": bag_results,
        "summary": {
            "detection_count": len(detections),
            "bottle_count": len(bottle_items),
            "bag_count": len(bag_items),
            "medicines": [
                item["final_medicine"]
                for item in bottle_results
                if item.get("final_medicine")
            ],
            "patients": [
                item["patient_matches"][0]
                for item in bag_results
                if item.get("patient_matches")
            ],
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.output_image:
        vis = draw_results(image, detections, bottle_results, bag_results)
        write_image(args.output_image, vis)

    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"elapsed: {elapsed:.3f}s")
    print(f"json: {args.output_json}")
    if args.output_image:
        print(f"image: {args.output_image}")


def build_parser():
    parser = argparse.ArgumentParser(description="Run offline YOLO + OCR recognition on one image.")
    parser.add_argument("image", help="input image path")
    parser.add_argument("--yolo-model", default=str(ROOT_DIR / "model_yolo_0510.rknn"))
    parser.add_argument("--ocr-det-model", default=str(ROOT_DIR / "src" / "identification" / "Det_bs1.rknn"))
    parser.add_argument("--ocr-rec-model", default=str(ROOT_DIR / "src" / "identification" / "Rec_bs16.rknn"))
    parser.add_argument("--ocr-dict", default=str(ROOT_DIR / "OCRS.txt"))
    parser.add_argument("--roi", default="100,100,3400,2160", help="x1,y1,x2,y2; default follows realtime code")
    parser.add_argument("--full-image", action="store_true", help="scan the whole image instead of the fast ROI")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.85)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--ocr-engine", choices=("paddle", "rknn"), default="paddle")
    parser.add_argument("--ocr-use-gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--paddle-det-max-side-len", type=int, default=32)
    parser.add_argument("--paddle-det-db-thresh", type=float, default=0.2)
    parser.add_argument("--paddle-det-db-box-thresh", type=float, default=0.2)
    parser.add_argument("--paddle-rec-batch-num", type=int, default=8)
    parser.add_argument("--paddle-rec-score-thresh", type=float, default=0.3)
    parser.add_argument("--paddle-cpu-threads", type=int, default=4)
    parser.add_argument("--ocr-max-crop-side", type=int, default=420, help="resize each crop before OCR; 0 disables")
    parser.add_argument("--no-db", action="store_true")
    parser.add_argument("--no-classify", action="store_true")
    parser.add_argument("--no-batch-ocr", action="store_true")
    parser.add_argument("--output-json", type=Path, default=ROOT_DIR / "offline_result.json")
    parser.add_argument("--output-image", type=Path, default=ROOT_DIR / "offline_result.jpg")
    parser.add_argument("--db-host", default="192.168.137.1")
    parser.add_argument("--db-port", type=int, default=3306)
    parser.add_argument("--db-user", default="root")
    parser.add_argument("--db-password", default="root")
    parser.add_argument("--db-name", default="medicine_db")
    parser.add_argument("--db-charset", default="utf8")
    parser.add_argument("--drug-table", default="drugs")
    parser.add_argument("--drug-column", default="medicine_name")
    parser.add_argument("--patient-table", default="patients")
    parser.add_argument("--patient-column", default="name")
    return parser


if __name__ == "__main__":
    try:
        run(build_parser().parse_args())
    except Exception as exc:
        print(f"failed: {exc}")
        traceback.print_exc()
        raise
