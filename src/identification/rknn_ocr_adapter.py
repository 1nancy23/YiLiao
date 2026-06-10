import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from rknnlite.api import RKNNLite

import Test_OCR as infusion_ocr
import Test_OCR_tight_lines as tight_ocr
from src.identification.OCRRecognizer import parse_required_fields
from src.identification.rknn_runtime_lock import get_rknn_lock


LABEL_LIST = [0, 90, 180, 270]
CLS_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
CLS_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
TIGHT_LINE_PADDING = {"box_padding_ratio": 0.04, "min_box_padding": 2, "max_box_padding": 8}
BOTTLE_LINE_PADDING = {
    "box_padding_ratio": 0.04,
    "min_box_padding": 2,
    "max_box_padding": 8,
    "final_short_side_padding_ratio": 0.18,
    "final_min_short_side_padding": 4,
    "final_max_short_side_padding": 12,
}
INFUSION_PADDING = {"box_padding_ratio": 0.04, "min_box_padding": 1, "max_box_padding": 4}
INFUSION_MERGE_KWARGS = {
    "center_y_ratio": 0.30,
    "overlap_ratio": 0.55,
    "max_gap_ratio": 0.85,
    "min_gap": 18,
    "crop_padding_ratio": 0.08,
    "min_crop_padding": 6,
    "max_crop_padding": 28,
}
NAME_ROI_PADDING = {"box_padding_ratio": 0.18, "min_box_padding": 4, "max_box_padding": 12}


def _collect_timing_enabled():
    return os.environ.get("YILIAO_COLLECT_TIMING", "0").lower() in ("1", "true", "yes", "on")


class _NoopTiming(dict):
    def __setitem__(self, key, value):
        return None

    def update(self, *args, **kwargs):
        return None


def _squeeze_det_map(output):
    output = np.asarray(output)
    while output.ndim > 2:
        output = output[0]
    return output


def _first_det_map(det_outputs):
    return _squeeze_det_map(det_outputs[0])


def _batch_det_maps(det_outputs, expected_count):
    output = np.asarray(det_outputs[0])
    if output.ndim >= 3 and output.shape[0] >= expected_count:
        return [_squeeze_det_map(output[i]) for i in range(expected_count)]
    if output.ndim >= 4 and output.shape[0] == 1 and output.shape[1] >= expected_count:
        return [_squeeze_det_map(output[0, i]) for i in range(expected_count)]
    first = _squeeze_det_map(output)
    return [first for _ in range(expected_count)]


def _batch_size_from_path(path, default=1):
    match = re.search(r"bs(\d+)", os.path.basename(str(path)).lower())
    if match:
        return int(match.group(1))
    return default


def _first_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def _batch_cls_preds(cls_outputs, expected_count):
    output = np.asarray(cls_outputs[0])
    while output.ndim > 2 and output.shape[0] == 1:
        output = output[0]

    if output.ndim == 1:
        logits = output.reshape(1, -1)
    elif output.ndim == 2 and output.shape[0] == len(LABEL_LIST) and output.shape[1] >= expected_count:
        logits = output[:, :expected_count].T
    elif output.ndim >= 2 and output.shape[0] >= expected_count:
        logits = output[:expected_count].reshape(expected_count, -1)
    else:
        logits = output.reshape(1, -1)

    preds = np.argmax(logits, axis=1).astype(np.int32).tolist()
    if len(preds) < expected_count:
        preds.extend([preds[-1] if preds else 0] * (expected_count - len(preds)))
    return [int(pred) for pred in preds[:expected_count]]


def _enhance_cls_image(image):
    """Clarify low-contrast and reflective label text for orientation CLS."""
    if image is None or image.size == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(6, 6)).apply(l_channel)
    enhanced = cv2.cvtColor(
        cv2.merge((l_channel, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)


def _preprocess_cls_image(image, layout="NCHW"):
    image = _enhance_cls_image(image)
    if layout.upper() != "NCHW":
        return cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        resized = np.zeros((224, 224, 3), dtype=np.uint8)
    else:
        scale = 256.0 / float(min(height, width))
        resized_w = max(224, int(round(width * scale)))
        resized_h = max(224, int(round(height * scale)))
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    y1 = max(0, (resized.shape[0] - 224) // 2)
    x1 = max(0, (resized.shape[1] - 224) // 2)
    cropped = resized[y1:y1 + 224, x1:x1 + 224]
    if cropped.shape[0] != 224 or cropped.shape[1] != 224:
        cropped = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - CLS_MEAN) / CLS_STD
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32)


def _reverse_rotate_with_label(image, pred):
    angle = LABEL_LIST[int(pred)]
    reverse_angle = (360 - angle) % 360
    if reverse_angle == 0:
        return image
    if reverse_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if reverse_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if reverse_angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


class RknnOCRRecognizer:
    """OCR replacement based on Test_OCR_tight_lines.py and Test_OCR.py."""

    def __init__(
        self,
        det_model_path=None,
        rec_model_path=None,
        char_dict_path=None,
        cls_model_path=None,
        det_batch_size=None,
        rec_batch_size=None,
        cls_batch_size=None,
        det_core=RKNNLite.NPU_CORE_1,
        rec_core=RKNNLite.NPU_CORE_2,
        cls_core=RKNNLite.NPU_CORE_1,
    ):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        det_bs32_path = os.path.join(root, "src", "identification", "Det_bs32.rknn")
        det_bs8_path = os.path.join(root, "src", "identification", "Det_bs8.rknn")
        det_bs1_path = os.path.join(root, "src", "identification", "Det_bs1.rknn")
        self.det_model_path = det_model_path or _first_existing_path([
            det_bs32_path,
            det_bs8_path,
            det_bs1_path,
        ])
        inferred_det_batch_size = _batch_size_from_path(self.det_model_path, default=1)

        rec_bs16_path = os.path.join(root, "model_ocr_0526.rknn")
        rec_bs64_path = os.path.join(root, "model_ocr_0602_bs64.rknn")
        self.rec_model_path = rec_model_path or _first_existing_path([
            rec_bs16_path,
            rec_bs64_path,
        ])
        inferred_rec_batch_size = _batch_size_from_path(
            self.rec_model_path,
            default=16,
        )
        self.rec_input_size = (320, 48)
        self.char_dict_path = char_dict_path or os.path.join(root, "OCRS.txt")
        cls_bs32_path = os.path.join(root, "model_cls_bs32.rknn")
        cls_bs1_path = os.path.join(root, "model_cls.rknn")
        self.cls_model_path = cls_model_path or _first_existing_path([
            cls_bs32_path,
            cls_bs1_path,
        ])
        self.cls_input_layout = "NHWC"
        inferred_cls_batch_size = _batch_size_from_path(self.cls_model_path, default=1)
        if det_batch_size is None:
            det_batch_size = inferred_det_batch_size
        if rec_batch_size is None:
            rec_batch_size = inferred_rec_batch_size
        if cls_batch_size is None:
            cls_batch_size = inferred_cls_batch_size
        self.det_batch_size = max(1, int(det_batch_size))
        self.rec_batch_size = max(1, int(rec_batch_size))
        self.cls_batch_size = max(1, int(cls_batch_size), int(inferred_cls_batch_size))
        self._det_batch_supported = self.det_batch_size > 1
        default_workers = 1
        self.cpu_workers = max(1, int(os.environ.get("YILIAO_OCR_CPU_WORKERS", default_workers)))
        self.det_lock = get_rknn_lock(det_core, secondary_domain=True)
        self.rec_lock = get_rknn_lock(rec_core, secondary_domain=True)
        self.cls_lock = get_rknn_lock(cls_core, secondary_domain=True)

        self.det_rknn = RKNNLite()
        ret = self.det_rknn.load_rknn(self.det_model_path)
        if ret != 0:
            raise RuntimeError(f"Load OCR det model failed: {self.det_model_path}")
        ret = self.det_rknn.init_runtime(core_mask=det_core)
        if ret != 0:
            raise RuntimeError("Init OCR det runtime failed")

        self.rec_rknn = RKNNLite()
        ret = self.rec_rknn.load_rknn(self.rec_model_path)
        if ret != 0:
            raise RuntimeError(f"Load OCR rec model failed: {self.rec_model_path}")
        ret = self.rec_rknn.init_runtime(core_mask=rec_core)
        if ret != 0:
            raise RuntimeError("Init OCR rec runtime failed")

        self.cls_rknn = RKNNLite()
        ret = self.cls_rknn.load_rknn(self.cls_model_path)
        if ret != 0:
            raise RuntimeError(f"Load OCR cls model failed: {self.cls_model_path}")
        ret = self.cls_rknn.init_runtime(core_mask=cls_core)
        if ret != 0:
            raise RuntimeError("Init OCR cls runtime failed")

        self.characters = tight_ocr.load_ctc_character_list(self.char_dict_path)
        self.last_timing = {}
        self.debug_dir = ""
        self.debug_prefix = "ocr"
        self._debug_seq = 0
        self.predet_save_dir = os.environ.get("YILIAO_PREDET_SAVE_DIR", "").strip()
        self.detvis_save_dir = os.environ.get("YILIAO_DETVIS_SAVE_DIR", "").strip()

    def release(self):
        for attr in ("det_rknn", "rec_rknn", "cls_rknn"):
            model = getattr(self, attr, None)
            if model is None:
                continue
            try:
                model.release()
            except Exception:
                pass
            setattr(self, attr, None)

    def _save_debug_image(self, name, image):
        return

    def _save_predet_image(self, name, image):
        return

    def _save_detvis_image(self, name, image, boxes):
        return

    @staticmethod
    def _draw_det_boxes(image, boxes):
        vis = image.copy()
        for box in boxes or []:
            pts = np.asarray(box, dtype=np.int32).reshape(-1, 2)
            if len(pts) >= 4:
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        return vis

    def _save_region_strip(self, name, regions):
        return

    def _add_infusion_percent_regions(self, image, regions, boxes):
        if image is None or image.size == 0:
            return regions, boxes

        height, width = image.shape[:2]
        if height < 80 or width < 120:
            return regions, boxes

        def append_proposed_region(rect):
            x1, y1, x2, y2 = rect
            x1 = max(0, min(width - 1, int(x1)))
            y1 = max(0, min(height - 1, int(y1)))
            x2 = max(x1 + 1, min(width, int(x2)))
            y2 = max(y1 + 1, min(height, int(y2)))
            if x2 - x1 < 30 or y2 - y1 < 18:
                return None

            for box in boxes or []:
                pts = np.asarray(box, dtype=np.int32).reshape(-1, 2)
                bx1, by1 = np.min(pts, axis=0)
                bx2, by2 = np.max(pts, axis=0)
                inter_w = max(0, min(x2, bx2) - max(x1, bx1))
                inter_h = max(0, min(y2, by2) - max(y1, by1))
                inter = inter_w * inter_h
                proposed_area = max(1, (x2 - x1) * (y2 - y1))
                if inter / float(proposed_area) > 0.55:
                    return None

            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                return None

            percent_box = np.array(
                [[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]],
                dtype=np.int32,
            )
            return list(regions) + [crop], list(boxes) + [percent_box]

        det_rects = []
        for box in boxes or []:
            pts = np.asarray(box, dtype=np.int32).reshape(-1, 2)
            x1, y1 = np.min(pts, axis=0)
            x2, y2 = np.max(pts, axis=0)
            rect_w = max(1, int(x2 - x1))
            rect_h = max(1, int(y2 - y1))
            cx = (x1 + x2) / 2.0
            det_rects.append((int(x1), int(y1), int(x2), int(y2), rect_w, rect_h, cx))

        left_wide = [
            rect for rect in det_rects
            if rect[4] >= width * 0.20 and rect[6] <= width * 0.68
        ]
        lower_candidates = [
            rect for rect in left_wide
            if rect[1] >= height * 0.45 and rect[5] >= height * 0.06
        ]
        if lower_candidates:
            lower = max(lower_candidates, key=lambda value: (value[1], value[4]))
            lx1, ly1, lx2, ly2, _, _, _ = lower
            upper_candidates = []
            for rect in left_wide:
                ux1, uy1, ux2, uy2, _, _, _ = rect
                gap = ly1 - uy2
                if gap < max(18, int(height * 0.045)) or gap > height * 0.42:
                    continue
                overlap = max(0, min(ux2, lx2) - max(ux1, lx1))
                min_w = max(1, min(ux2 - ux1, lx2 - lx1))
                if overlap / float(min_w) < 0.18:
                    continue
                upper_candidates.append(rect)

            if upper_candidates:
                upper = max(upper_candidates, key=lambda value: value[3])
                ux1, uy1, ux2, uy2, _, _, _ = upper
                lower_w = max(1, lx2 - lx1)
                gap = max(1, ly1 - uy2)
                pad_x = max(6, int(width * 0.018))
                y_top_pad = max(3, int(gap * 0.10))
                y_bottom_pad = max(2, int(gap * 0.05))
                proposed = append_proposed_region((
                    lx1 + int(lower_w * 0.15) - pad_x,
                    uy2 + y_top_pad,
                    lx1 + int(lower_w * 0.92) + pad_x,
                    ly1 - y_bottom_pad,
                ))
                if proposed is not None:
                    return proposed

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, np.array([90, 45, 25]), np.array([145, 255, 210]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        min_area = max(180, int(width * height * 0.008))
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < min_area or w < width * 0.18 or h < height * 0.05:
                continue
            rects.append((x, y, x + w, y + h, area))

        if len(rects) < 2:
            return regions, boxes

        best = None
        best_score = -1.0
        for upper in rects:
            ux1, uy1, ux2, uy2, uarea = upper
            for lower in rects:
                lx1, ly1, lx2, ly2, larea = lower
                gap = ly1 - uy2
                if gap <= max(12, int(height * 0.05)) or gap >= height * 0.45:
                    continue
                overlap = max(0, min(ux2, lx2) - max(ux1, lx1))
                min_w = max(1, min(ux2 - ux1, lx2 - lx1))
                if overlap / float(min_w) < 0.20:
                    continue
                score = overlap + min(uarea, larea) * 0.002 - gap * 0.2
                if score > best_score:
                    best_score = score
                    best = (upper, lower)

        if best is None:
            return regions, boxes

        upper, lower = best
        ux1, uy1, ux2, uy2, _ = upper
        lx1, ly1, lx2, ly2, _ = lower
        pad_x = max(6, int(width * 0.015))
        pad_y = max(6, int(height * 0.018))
        x1 = max(0, min(ux1, lx1) - pad_x)
        y1 = max(0, uy2 - pad_y)
        x2 = min(width, max(ux2, lx2) + pad_x)
        y2 = min(height, ly1 + pad_y)

        proposed = append_proposed_region((x1, y1, x2, y2))
        if proposed is not None:
            return proposed
        return regions, boxes

    def _detect_regions(self, image, module, padding):
        regions_list, boxes_list, _ = self._detect_regions_batch([image], module, padding)
        return regions_list[0], boxes_list[0]

    def _parallel_map(self, func, items):
        if not items:
            return []
        if self.cpu_workers <= 1 or len(items) <= 1:
            return [func(item) for item in items]
        with ThreadPoolExecutor(max_workers=min(self.cpu_workers, len(items))) as executor:
            return list(executor.map(func, items))

    def _postprocess_det_candidate(self, args):
        item, det_map = args
        padding = item["padding"]
        module = item["module"]
        extract_kwargs = {}
        if item.get("merge_boxes") is not None:
            extract_kwargs["merge_boxes"] = item.get("merge_boxes")
        elif module is infusion_ocr:
            extract_kwargs["merge_boxes"] = True
        if item.get("filter_edge_boxes") is not None:
            extract_kwargs["filter_edge_boxes"] = item.get("filter_edge_boxes")
        elif module is infusion_ocr:
            extract_kwargs["filter_edge_boxes"] = True
        if item.get("merge_kwargs") is not None:
            extract_kwargs["merge_kwargs"] = item.get("merge_kwargs")
        elif module is infusion_ocr:
            extract_kwargs["merge_kwargs"] = INFUSION_MERGE_KWARGS
        if item.get("preserve_rotated_boxes") and module is tight_ocr:
            extract_kwargs["preserve_rotated_boxes"] = True
        if item.get("short_side_padding_ratio") is not None and module is tight_ocr:
            extract_kwargs["short_side_padding_ratio"] = item["short_side_padding_ratio"]
            extract_kwargs["min_short_side_padding"] = item.get("min_short_side_padding")
            extract_kwargs["max_short_side_padding"] = item.get("max_short_side_padding")
        if item.get("final_short_side_padding_ratio") is not None and module is tight_ocr:
            extract_kwargs["final_short_side_padding_ratio"] = item["final_short_side_padding_ratio"]
            extract_kwargs["final_min_short_side_padding"] = item.get("final_min_short_side_padding")
            extract_kwargs["final_max_short_side_padding"] = item.get("final_max_short_side_padding")

        regions, boxes = module.extract_text_regions(
            item["image"],
            det_map,
            scale_factor=1,
            threshold=0.3,
            box_padding_ratio=padding["box_padding_ratio"],
            min_box_padding=padding["min_box_padding"],
            max_box_padding=padding["max_box_padding"],
            **extract_kwargs,
        )
        owner = item.get("owner")
        if isinstance(owner, tuple) and owner and owner[0] == "shuye":
            regions, boxes = self._add_infusion_percent_regions(item["image"], regions, boxes)
            owner_name = "_".join(str(part) for part in owner)
            self._save_detvis_image(f"{owner_name}_det_boxes_before_rec", item["image"], boxes)
        elif isinstance(owner, tuple) and owner and owner[0] == "bottle":
            owner_name = "_".join(str(part) for part in owner)
            self._save_detvis_image(f"{owner_name}_det_boxes_before_rec", item["image"], boxes)
            item["det_visualization"] = self._draw_det_boxes(item["image"], boxes)
            item["det_region_count"] = len(regions)

        regions = [region for region in regions if region is not None and region.size > 0]
        split_rows = item.get("split_rows")
        if split_rows == "infusion":
            regions = self._split_infusion_regions(regions)
            if isinstance(owner, tuple) and owner and owner[0] == "shuye":
                owner_name = "_".join(str(part) for part in owner)
                self._save_region_strip(f"{owner_name}_after_split", regions)
        elif split_rows == "label":
            regions = self._split_label_regions(regions)
        elif split_rows:
            regions = self._split_multiline_regions(regions)
        if isinstance(owner, tuple) and owner and owner[0] == "bottle":
            owner_name = "_".join(str(part) for part in owner)
            self._save_region_strip(f"{owner_name}_final", regions)
            item["det_region_count"] = len(regions)
        return item, regions

    def _preprocess_mixed_rec_item(self, args):
        module, region = args
        preprocess = getattr(module, "preprocess_rec_image", tight_ocr.preprocess_rec_image)
        return preprocess(region, self.rec_input_size)

    def _decode_mixed_rec_item(self, args):
        index, module, region, rec_item = args
        decode = getattr(module, "ctc_greedy_decode_one", tight_ocr.ctc_greedy_decode_one)
        raw_text, pred_indices = decode(rec_item, self.characters)
        correct_percent = getattr(module, "correct_percent_confusion", tight_ocr.correct_percent_confusion)
        text = correct_percent(raw_text, region)
        return {
            "index": index,
            "text": text,
            "raw_text": raw_text,
            "pred_indices": pred_indices,
        }

    def _decode_indices_to_text(self, pred_indices, blank_idx=0):
        decoded = []
        prev_idx = -1
        character_count = len(self.characters)
        for idx in pred_indices:
            idx = int(idx)
            if idx == blank_idx:
                prev_idx = -1
                continue
            if idx != prev_idx and idx < character_count:
                decoded.append(self.characters[idx])
            prev_idx = idx
        return "".join(decoded)

    def _decode_mixed_rec_batch(self, start, modules, regions, rec_output, valid_count):
        pred_indices_batch = np.argmax(rec_output[:valid_count], axis=-1)
        results = []
        for offset in range(valid_count):
            module = modules[offset]
            raw_text = self._decode_indices_to_text(pred_indices_batch[offset])
            correct_percent = getattr(module, "correct_percent_confusion", tight_ocr.correct_percent_confusion)
            text = correct_percent(raw_text, regions[offset])
            results.append({
                "index": start + offset,
                "text": text,
                "raw_text": raw_text,
                "pred_indices": pred_indices_batch[offset],
            })
        return results

    def _detect_regions_batch(self, images, module, padding):
        all_regions = []
        all_boxes = []
        total_infer = 0.0
        batch_size = max(1, int(self.det_batch_size)) if self._det_batch_supported else 1

        for start in range(0, len(images), batch_size):
            batch_images = images[start:start + batch_size]
            det_batch = np.empty((batch_size, 448, 448, 3), dtype=np.uint8)
            if len(batch_images) < batch_size:
                det_batch[len(batch_images):] = 0
            for offset, image in enumerate(batch_images):
                det_input = cv2.resize(image, (448, 448))
                self._save_debug_image("det_input_448", det_input)
                det_batch[offset] = det_input

            t = time.perf_counter()
            try:
                with self.det_lock:
                    det_outputs = self.det_rknn.inference(inputs=[det_batch])
                if det_outputs is None:
                    raise RuntimeError("OCR det RKNN inference returned None")
            except Exception:
                if batch_size > 1:
                    self._det_batch_supported = False
                    fallback_regions, fallback_boxes, fallback_infer = self._detect_regions_batch(
                        images[start:],
                        module,
                        padding,
                    )
                    all_regions.extend(fallback_regions)
                    all_boxes.extend(fallback_boxes)
                    total_infer += fallback_infer
                    break
                raise
            total_infer += time.perf_counter() - t
            det_maps = _batch_det_maps(det_outputs, len(batch_images))

            for image, det_map in zip(batch_images, det_maps):
                extract_kwargs = {}
                if module is infusion_ocr:
                    extract_kwargs["merge_boxes"] = True
                    extract_kwargs["filter_edge_boxes"] = True
                    extract_kwargs["merge_kwargs"] = INFUSION_MERGE_KWARGS
                regions, boxes = module.extract_text_regions(
                    image,
                    det_map,
                    scale_factor=1,
                    threshold=0.3,
                    box_padding_ratio=padding["box_padding_ratio"],
                    min_box_padding=padding["min_box_padding"],
                    max_box_padding=padding["max_box_padding"],
                    **extract_kwargs,
                )
                if module is infusion_ocr:
                    regions, boxes = self._add_infusion_percent_regions(image, regions, boxes)
                all_regions.append(regions)
                all_boxes.append(boxes)

        return all_regions, all_boxes, total_infer

    def _recognize_regions(self, module, regions):
        for idx, region in enumerate(regions):
            self._save_debug_image(f"rec_region_{idx:02d}", region)
        with self.rec_lock:
            raw_results = module.recognize_text_regions(
                self.rec_rknn,
                regions,
                self.characters,
                batch_size=self.rec_batch_size,
                rec_input_size=self.rec_input_size,
            )
        aligned_results = [{"index": idx, "text": "", "raw_text": ""} for idx in range(len(regions))]
        sequential_idx = 0
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            result_idx = item.get("index")
            if result_idx is None:
                while sequential_idx < len(aligned_results) and aligned_results[sequential_idx].get("text"):
                    sequential_idx += 1
                result_idx = sequential_idx
                sequential_idx += 1
            try:
                result_idx = int(result_idx)
            except (TypeError, ValueError):
                continue
            if 0 <= result_idx < len(aligned_results):
                aligned_results[result_idx] = item
        return aligned_results

    def _recognize_mixed_regions(self, modules, regions):
        for idx, region in enumerate(regions):
            self._save_debug_image(f"mixed_rec_region_{idx:02d}", region)

        aligned_results = [{"index": idx, "text": "", "raw_text": ""} for idx in range(len(regions))]
        if not regions:
            return aligned_results

        batch_size = max(1, int(self.rec_batch_size))
        blank_image = np.zeros((self.rec_input_size[1], self.rec_input_size[0], 3), dtype=np.uint8)
        self._last_mixed_rec_preprocess = 0.0
        self._last_mixed_rec_infer = 0.0
        self._last_mixed_rec_decode = 0.0
        self._last_mixed_rec_pad_inputs = 0

        for start in range(0, len(regions), batch_size):
            batch_regions = regions[start:start + batch_size]
            batch_modules = modules[start:start + batch_size]
            valid_count = len(batch_regions)
            self._last_mixed_rec_pad_inputs += batch_size - valid_count
            t_pre = time.perf_counter()
            batch_images = self._parallel_map(
                self._preprocess_mixed_rec_item,
                list(zip(batch_modules, batch_regions)),
            )
            self._last_mixed_rec_preprocess += time.perf_counter() - t_pre

            first_image = batch_images[0]
            batch_input = np.empty((batch_size,) + first_image.shape, dtype=first_image.dtype)
            for offset, batch_image in enumerate(batch_images):
                batch_input[offset] = batch_image
            if valid_count < batch_size:
                batch_input[valid_count:] = blank_image
            t_infer = time.perf_counter()
            with self.rec_lock:
                rec_outputs = self.rec_rknn.inference(inputs=[batch_input])
            self._last_mixed_rec_infer += time.perf_counter() - t_infer
            if rec_outputs is None:
                raise RuntimeError("OCR rec RKNN inference returned None")
            rec_output = tight_ocr.normalize_rec_output(rec_outputs[0], batch_size)

            t_decode = time.perf_counter()
            for result in self._decode_mixed_rec_batch(start, batch_modules, batch_regions, rec_output, valid_count):
                aligned_results[result["index"]] = result
            self._last_mixed_rec_decode += time.perf_counter() - t_decode

        return aligned_results

    def _split_multiline_region(self, region):
        """Split a merged OCR crop into separate text rows when clear row gaps exist."""
        if region is None or region.size == 0:
            return []
        height, width = region.shape[:2]
        if height < 18 or width < 18:
            return [region]

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        row_splits = self._split_region_by_projection(
            region,
            mask,
            axis=0,
            min_size=max(7, int(height * 0.16)),
            active_threshold=max(2, int(width * 0.025)),
            merge_gap=3,
            pad=3,
        )
        if len(row_splits) > 1:
            result = []
            for split in row_splits:
                result.extend(self._split_side_by_side_columns(split))
            return result or row_splits

        return self._split_side_by_side_columns(region)

    def _split_side_by_side_columns(self, region):
        if region is None or region.size == 0:
            return []
        height, width = region.shape[:2]
        if height < 18 or width < 18:
            return [region]

        # Some bottle labels are returned as a tall, narrow crop containing
        # two text lines side by side after rotation. Row projection cannot
        # separate those, so use a conservative column split only for such
        # high-aspect crops.
        if height >= width * 1.35:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            col_splits = self._split_region_by_projection(
                region,
                mask,
                axis=1,
                min_size=max(4, int(width * 0.18)),
                active_threshold=max(2, int(height * 0.025)),
                merge_gap=2,
                pad=2,
            )
            if len(col_splits) > 1:
                return col_splits

        return [region]

    def _split_region_by_projection(self, region, mask, axis, min_size, active_threshold, merge_gap, pad):
        counts = np.count_nonzero(mask, axis=1 if axis == 0 else 0)
        active = counts > active_threshold

        runs = []
        start = None
        for pos, is_active in enumerate(active):
            if is_active and start is None:
                start = pos
            elif not is_active and start is not None:
                runs.append([start, pos - 1])
                start = None
        if start is not None:
            runs.append([start, len(active) - 1])

        if len(runs) <= 1:
            return [region]

        merged = []
        for run in runs:
            if not merged or run[0] - merged[-1][1] > merge_gap:
                merged.append(run)
            else:
                merged[-1][1] = run[1]

        valid = [run for run in merged if run[1] - run[0] + 1 >= min_size]
        if len(valid) <= 1:
            return [region]

        height, width = region.shape[:2]
        result = []
        for start, end in valid:
            left_or_top = max(0, start - pad)
            right_or_bottom = end + pad + 1
            if axis == 0:
                crop = region[left_or_top:min(height, right_or_bottom), :].copy()
            else:
                crop = region[:, left_or_top:min(width, right_or_bottom)].copy()
            if crop.size > 0:
                result.append(crop)
        return result or [region]

    def _split_multiline_regions(self, regions):
        split = []
        for region in regions:
            split.extend(self._split_multiline_region(region))
        return split

    def _split_infusion_text_rows(self, region):
        if region is None or region.size == 0:
            return []
        height, width = region.shape[:2]
        if height < 18 or width < 30:
            return [region]

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        # Infusion labels are usually white text on a saturated blue field.
        # Use bright, low-saturation pixels as text ink; this avoids treating
        # the blue background itself as one giant text row.
        bright_text = ((hsv[:, :, 1] < 105) & (hsv[:, :, 2] > 118)).astype(np.uint8) * 255
        if cv2.countNonZero(bright_text) < max(20, int(width * height * 0.006)):
            _, bright_text = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        mask = cv2.morphologyEx(bright_text, cv2.MORPH_CLOSE, kernel, iterations=1)
        row_counts = np.count_nonzero(mask, axis=1)
        active = row_counts > max(2, int(width * 0.018))

        runs = []
        start = None
        for pos, is_active in enumerate(active):
            if is_active and start is None:
                start = pos
            elif not is_active and start is not None:
                runs.append([start, pos - 1])
                start = None
        if start is not None:
            runs.append([start, len(active) - 1])

        merged = []
        for run in runs:
            if not merged or run[0] - merged[-1][1] > 4:
                merged.append(run)
            else:
                merged[-1][1] = run[1]

        rows = []
        min_row_h = max(7, int(height * 0.08))
        for y1, y2 in merged:
            if y2 - y1 + 1 < min_row_h:
                continue
            y1p = max(0, y1 - 3)
            y2p = min(height, y2 + 4)
            row_mask = mask[y1p:y2p, :]
            col_counts = np.count_nonzero(row_mask, axis=0)
            cols = np.where(col_counts > max(1, int((y2p - y1p) * 0.08)))[0]
            if len(cols) == 0:
                continue
            x1p = max(0, int(cols[0]) - 5)
            x2p = min(width, int(cols[-1]) + 6)
            crop = region[y1p:y2p, x1p:x2p].copy()
            if crop.size > 0 and crop.shape[1] >= 12:
                rows.append(crop)

        return rows if rows else [region]

    def _split_infusion_regions(self, regions):
        split = []
        for region in regions:
            split.extend(self._split_infusion_text_rows(region))
        return split

    def _split_label_region_rows(self, region):
        if region is None or region.size == 0:
            return []
        height, width = region.shape[:2]
        if height < 26 or width < 50:
            return [region]

        row_split = self._split_multiline_region(region)
        if len(row_split) > 1:
            return row_split

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        row_counts = np.count_nonzero(mask, axis=1).astype(np.float32)
        if row_counts.max() <= 0:
            return [region]

        kernel = np.ones(5, dtype=np.float32) / 5.0
        smooth = np.convolve(row_counts, kernel, mode="same")
        lo = max(5, int(height * 0.30))
        hi = min(height - 5, int(height * 0.72))
        if hi <= lo:
            return [region]

        cut = int(lo + np.argmin(smooth[lo:hi]))
        top_ink = float(np.sum(row_counts[:cut]))
        bottom_ink = float(np.sum(row_counts[cut:]))
        total_ink = top_ink + bottom_ink
        valley_ratio = smooth[cut] / max(1.0, float(np.max(smooth)))
        if total_ink <= 0:
            return [region]
        if top_ink / total_ink < 0.12 or bottom_ink / total_ink < 0.12:
            return [region]
        if valley_ratio > 0.72:
            return [region]

        pad = 2
        top = region[:min(height, cut + pad), :].copy()
        bottom = region[max(0, cut - pad):, :].copy()
        return [crop for crop in (top, bottom) if crop.size > 0] or [region]

    def _split_label_regions(self, regions):
        split = []
        for region in regions:
            split.extend(self._split_label_region_rows(region))
        return split

    def _score_medicine_texts(self, texts):
        text = "".join(texts or [])
        if not text:
            return 0.0
        chinese_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        alnum_count = sum(1 for ch in text if ch.isalnum())
        score = chinese_count * 2.0 + min(alnum_count, 16) * 0.25
        medicine_hints = (
            "注射", "射用", "头孢", "舒巴坦", "钠", "液", "酸", "素",
            "甘草", "炎琥宁", "鸟氨酸", "多索", "维生素", "葡萄糖",
        )
        score += sum(4.0 for hint in medicine_hints if hint in text)
        batch_only_hints = ("批号", "有效", "生产", "2027", "2028")
        score -= sum(1.5 for hint in batch_only_hints if hint in text)
        return score

    def _recognize_candidate(self, image, module, padding, split_rows, timing_prefix=""):
        t = time.perf_counter()
        regions, boxes = self._detect_regions(image, module, padding)
        regions = [region for region in regions if region is not None and region.size > 0]
        if split_rows == "label":
            regions = self._split_label_regions(regions)
        elif split_rows:
            regions = self._split_multiline_regions(regions)
        self.last_timing[f"{timing_prefix}rknn_det"] = time.perf_counter() - t
        self.last_timing[f"{timing_prefix}text_regions"] = len(regions)

        if not regions:
            self.last_timing[f"{timing_prefix}rknn_rec"] = 0.0
            return []

        t = time.perf_counter()
        rec_results = self._recognize_regions(module, regions)
        self.last_timing[f"{timing_prefix}rknn_rec"] = time.perf_counter() - t
        return [item.get("text", "") for item in rec_results if item.get("text", "").strip()]

    def _recognize_candidates_batch(self, images, module, padding, split_rows, timing_prefix=""):
        t = time.perf_counter()
        regions_list, _boxes_list, det_infer = self._detect_regions_batch(images, module, padding)
        split_regions_list = []
        flat_regions = []
        owners = []

        for image_idx, regions in enumerate(regions_list):
            regions = [region for region in regions if region is not None and region.size > 0]
            if split_rows == "label":
                regions = self._split_label_regions(regions)
            elif split_rows:
                regions = self._split_multiline_regions(regions)
            split_regions_list.append(regions)
            for region in regions:
                owners.append(image_idx)
                flat_regions.append(region)

        self.last_timing[f"{timing_prefix}rknn_det"] = time.perf_counter() - t
        self.last_timing[f"{timing_prefix}rknn_det_infer"] = det_infer
        self.last_timing[f"{timing_prefix}text_regions"] = len(flat_regions)

        texts_list = [[] for _ in images]
        if not flat_regions:
            self.last_timing[f"{timing_prefix}rknn_rec"] = 0.0
            return texts_list

        t = time.perf_counter()
        rec_results = self._recognize_regions(module, flat_regions)
        self.last_timing[f"{timing_prefix}rknn_rec"] = time.perf_counter() - t
        for owner, item in zip(owners, rec_results):
            text = item.get("text", "").strip()
            if text:
                texts_list[owner].append(text)
        return texts_list

    def _recognize_mixed_candidates_batch(self, candidates, timing_prefix="mixed_"):
        """Run one shared det/rec batch for bottle, bag and infusion OCR crops."""
        texts_by_owner = {}
        if not candidates:
            self.last_timing[f"{timing_prefix}rknn_det"] = 0.0
            self.last_timing[f"{timing_prefix}rknn_rec"] = 0.0
            self.last_timing[f"{timing_prefix}text_regions"] = 0
            return texts_by_owner

        t_det = time.perf_counter()
        images = [item["image"] for item in candidates]
        batch_size = max(1, int(self.det_batch_size)) if self._det_batch_supported else 1
        flat_regions = []
        owners = []
        region_modules = []
        total_infer = 0.0
        total_preprocess = 0.0
        total_postprocess = 0.0
        total_det_pad_inputs = 0

        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start:start + batch_size]
            total_det_pad_inputs += batch_size - len(chunk)
            t_pre = time.perf_counter()
            det_batch = np.empty((batch_size, 448, 448, 3), dtype=np.uint8)
            if len(chunk) < batch_size:
                det_batch[len(chunk):] = 0
            for offset, item in enumerate(chunk):
                det_input = cv2.resize(item["image"], (448, 448))
                self._save_debug_image("det_input_448", det_input)
                det_batch[offset] = det_input

            total_preprocess += time.perf_counter() - t_pre
            t = time.perf_counter()
            try:
                with self.det_lock:
                    det_outputs = self.det_rknn.inference(inputs=[det_batch])
                if det_outputs is None:
                    raise RuntimeError("OCR det RKNN inference returned None")
            except Exception:
                if batch_size > 1:
                    self._det_batch_supported = False
                    return self._recognize_mixed_candidates_batch(candidates, timing_prefix)
                raise
            total_infer += time.perf_counter() - t
            det_maps = _batch_det_maps(det_outputs, len(chunk))

            t_post = time.perf_counter()
            postprocessed = self._parallel_map(
                self._postprocess_det_candidate,
                list(zip(chunk, det_maps)),
            )
            total_postprocess += time.perf_counter() - t_post
            for item, regions in postprocessed:
                for region in regions:
                    owners.append(item["owner"])
                    region_modules.append(item["module"])
                    flat_regions.append(region)

        self.last_timing[f"{timing_prefix}rknn_det"] = time.perf_counter() - t_det
        self.last_timing[f"{timing_prefix}rknn_det_preprocess"] = total_preprocess
        self.last_timing[f"{timing_prefix}rknn_det_infer"] = total_infer
        self.last_timing[f"{timing_prefix}rknn_det_postprocess"] = total_postprocess
        self.last_timing[f"{timing_prefix}det_inputs"] = len(candidates)
        self.last_timing[f"{timing_prefix}det_pad_inputs"] = total_det_pad_inputs
        self.last_timing[f"{timing_prefix}det_batches"] = int(np.ceil(len(candidates) / float(max(1, batch_size))))
        self.last_timing[f"{timing_prefix}text_regions"] = len(flat_regions)

        for item in candidates:
            texts_by_owner.setdefault(item["owner"], [])

        if not flat_regions:
            self.last_timing[f"{timing_prefix}rknn_rec"] = 0.0
            self.last_timing[f"{timing_prefix}rec_inputs"] = 0
            self.last_timing[f"{timing_prefix}rec_batches"] = 0
            return texts_by_owner

        t_rec = time.perf_counter()
        rec_results = self._recognize_mixed_regions(region_modules, flat_regions)
        self.last_timing[f"{timing_prefix}rknn_rec"] = time.perf_counter() - t_rec
        self.last_timing[f"{timing_prefix}rknn_rec_preprocess"] = getattr(self, "_last_mixed_rec_preprocess", 0.0)
        self.last_timing[f"{timing_prefix}rknn_rec_infer"] = getattr(self, "_last_mixed_rec_infer", 0.0)
        self.last_timing[f"{timing_prefix}rknn_rec_decode"] = getattr(self, "_last_mixed_rec_decode", 0.0)
        self.last_timing[f"{timing_prefix}rec_inputs"] = len(flat_regions)
        self.last_timing[f"{timing_prefix}rec_pad_inputs"] = getattr(self, "_last_mixed_rec_pad_inputs", 0)
        self.last_timing[f"{timing_prefix}rec_batches"] = int(np.ceil(len(flat_regions) / float(max(1, self.rec_batch_size))))
        for owner, item in zip(owners, rec_results):
            if item is None:
                continue
            text = item.get("text", "").strip()
            if text:
                texts_by_owner.setdefault(owner, []).append(text)
        return texts_by_owner

    def _extract_label_card(self, image):
        if image is None or image.size == 0:
            return image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # The medication label is a low-saturation, high-value card. Extracting
        # it keeps green bag text and the surrounding background away from OCR.
        mask = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 95, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        height, width = image.shape[:2]
        min_area = width * height * 0.10
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area >= min_area and w > width * 0.25 and h > height * 0.20:
                candidates.append((area, x, y, w, h))
        if not candidates:
            return image

        _, x, y, w, h = max(candidates, key=lambda item: item[0])
        pad = max(4, int(min(width, height) * 0.015))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + w + pad)
        y2 = min(height, y + h + pad)
        return image[y1:y2, x1:x2].copy()

    def _prepare_name_roi(self, roi):
        if roi is None or roi.size == 0:
            return roi
        height, width = roi.shape[:2]
        scale = 2.0 if max(height, width) < 360 else 1.5
        enlarged = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = cv2.merge((clahe.apply(l), a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return cv2.GaussianBlur(enhanced, (0, 0), 0.6)

    def _recognize_label_name_roi(self, image):
        start = time.perf_counter()
        t_angle = time.perf_counter()
        rotated, pred = self._classify_and_rotate(image)
        self.last_timing["angle_cls"] = time.perf_counter() - t_angle
        self.last_timing["angle_pred"] = float(pred)
        self._save_debug_image(f"label_angle_rotated_pred_{pred}", rotated)

        t_roi = time.perf_counter()
        card = self._extract_label_card(rotated)
        self._save_debug_image("label_card", card)
        height, width = card.shape[:2]
        roi_specs = (
            (0.00, 0.00, 0.56, 0.45),
            (0.00, 0.08, 0.60, 0.55),
            (0.02, 0.00, 0.48, 0.35),
        )
        self.last_timing["name_roi_extract"] = time.perf_counter() - t_roi

        all_texts = []
        for idx, (x1r, y1r, x2r, y2r) in enumerate(roi_specs):
            x1 = max(0, min(width - 1, int(width * x1r)))
            y1 = max(0, min(height - 1, int(height * y1r)))
            x2 = max(x1 + 1, min(width, int(width * x2r)))
            y2 = max(y1 + 1, min(height, int(height * y2r)))
            roi = self._prepare_name_roi(card[y1:y2, x1:x2].copy())
            self._save_debug_image(f"label_name_roi_{idx}", roi)
            texts = self._recognize_candidate(
                roi,
                tight_ocr,
                NAME_ROI_PADDING,
                split_rows=True,
                timing_prefix=f"name_roi_{idx}_",
            )
            all_texts.extend(texts)

        cleaned = []
        for text in all_texts:
            text = "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")
            # Patient names are short and appear before bed/date fields; trim
            # noisy OCR output while keeping common 2-4 character names intact.
            if 2 <= len(text) <= 6:
                cleaned.append(text[:4])
        self.last_timing["name_roi_total"] = time.perf_counter() - start
        return cleaned[0] if cleaned else ""

    def _classify_and_rotate(self, image):
        preds, _elapsed = self._run_cls_batch([image])
        pred = preds[0] if preds else 0
        return _reverse_rotate_with_label(image, pred), pred

    def _run_cls_batch(self, images):
        if not images:
            return [], 0.0

        preds = []
        total = 0.0
        pad_inputs = 0
        batch_size = max(
            1,
            int(self.cls_batch_size),
            _batch_size_from_path(self.cls_model_path, default=1),
        )

        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            pad_inputs += batch_size - len(chunk)
            first_input = _preprocess_cls_image(chunk[0], self.cls_input_layout)
            batch_data = np.empty((batch_size,) + first_input.shape, dtype=first_input.dtype)
            batch_data[0] = first_input
            for offset, image in enumerate(chunk[1:], start=1):
                batch_data[offset] = _preprocess_cls_image(image, self.cls_input_layout)
            if len(chunk) < batch_size:
                batch_data[len(chunk):] = 0
            t = time.perf_counter()
            with self.cls_lock:
                outputs = self.cls_rknn.inference(inputs=[batch_data])
            if outputs is None:
                raise RuntimeError("OCR cls RKNN inference returned None")
            total += time.perf_counter() - t
            preds.extend(_batch_cls_preds(outputs, len(chunk)))

        self._last_cls_real_inputs = len(images)
        self._last_cls_pad_inputs = pad_inputs
        return preds, total

    def _classify_and_rotate_batch(self, images, timing_prefix=""):
        preds, total = self._run_cls_batch(images)
        rotated_images = [
            _reverse_rotate_with_label(image, pred)
            for image, pred in zip(images, preds)
        ]
        self.last_timing[f"{timing_prefix}angle_cls"] = total
        self.last_timing[f"{timing_prefix}cls_real_inputs"] = getattr(self, "_last_cls_real_inputs", len(images))
        self.last_timing[f"{timing_prefix}cls_pad_inputs"] = getattr(self, "_last_cls_pad_inputs", 0)
        return rotated_images, preds

    def _extract_infusion_blue_roi(self, image, wide=False):
        """Extract the blue label area for infusion bags using the dedicated pipeline."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 40])
        upper_blue = np.array([145, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16))
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        img_h, img_w = image.shape[:2]
        contour_area = cv2.contourArea(c)
        image_area = max(1, img_w * img_h)
        if (
            contour_area < image_area * 0.012
            or w < img_w * 0.14
            or h < img_h * 0.035
        ):
            return None
        pad_x = max(14, int(w * 0.10), int(img_w * 0.018))
        pad_top = max(10, int(h * 0.16), int(img_h * 0.018))
        # The volume line may sit below the blue percentage badge, especially
        # on 5%/50ml bags. Keep the blue contour as the anchor, but extend the
        # lower crop enough to include that non-blue volume text.
        pad_bottom = max(24, int(h * 0.78), int(img_h * 0.16))
        if wide:
            pad_x = max(pad_x, int(w * 0.20), int(img_w * 0.035))
            pad_top = max(pad_top, int(h * 0.32), int(img_h * 0.035))
            pad_bottom = max(pad_bottom, int(h * 1.05), int(img_h * 0.22))
        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_top, 0)
        x2 = min(x + w + pad_x, img_w)
        y2 = min(y + h + pad_bottom, img_h)
        roi = image[y1:y2, x1:x2]
        return roi if roi.size > 0 else None

    def _crop_ratio(self, image, spec):
        if image is None or image.size == 0:
            return None
        height, width = image.shape[:2]
        x1r, y1r, x2r, y2r = spec
        x1 = max(0, min(width - 1, int(width * x1r)))
        y1 = max(0, min(height - 1, int(height * y1r)))
        x2 = max(x1 + 1, min(width, int(width * x2r)))
        y2 = max(y1 + 1, min(height, int(height * y2r)))
        roi = image[y1:y2, x1:x2].copy()
        return roi if roi.size > 0 else None

    def _append_infusion_candidate(self, candidates, name, roi):
        if roi is None or roi.size <= 0:
            return
        height, width = roi.shape[:2]
        if height < 24 or width < 40:
            return
        for _existing_name, existing_roi in candidates:
            if existing_roi is None or existing_roi.size <= 0:
                continue
            eh, ew = existing_roi.shape[:2]
            if abs(eh - height) <= 3 and abs(ew - width) <= 3:
                return
        candidates.append((name, roi))

    def _extract_infusion_candidate_rois(self, image):
        """Return infusion OCR ROI candidates from an already orientation-corrected crop."""
        candidates = []
        if image is None or image.size == 0:
            return candidates

        blue_wide = self._extract_infusion_blue_roi(image, wide=True)
        if blue_wide is not None and blue_wide.size > 0:
            self._append_infusion_candidate(candidates, "blue_wide", blue_wide)
            self._save_debug_image("infusion_blue_wide_roi", blue_wide)

        blue = self._extract_infusion_blue_roi(image, wide=False)
        if blue is not None and blue.size > 0:
            self._append_infusion_candidate(candidates, "blue", blue)
            self._save_debug_image("infusion_blue_roi", blue)
        return candidates

    def _fake_ocr_result(self, texts):
        return [[
            [[[0, 0], [1, 0], [1, 1], [0, 1]], [text, 1.0]]
            for text in texts
            if str(text).strip()
        ]]

    def _parse_infusion_texts(self, texts):
        texts = [str(text).strip() for text in texts or [] if str(text).strip()]
        if not texts:
            return parse_required_fields(None)
        return parse_required_fields(self._fake_ocr_result(texts))

    def _score_infusion_result(self, item):
        if not isinstance(item, dict):
            return -10.0
        score = 0.0
        for field in ("liquid", "concentration", "volume"):
            if item.get(field):
                score += 3.0
        raw_text = str(item.get("raw_text") or "")
        if any(token in raw_text for token in ("葡萄糖", "氯化钠", "Glucose", "Sodium", "Injection")):
            score += 2.0
        if "%" in raw_text:
            score += 1.0
        if "ml" in raw_text.lower():
            score += 1.0
        if item.get("status") == "ocr_empty":
            score -= 2.0
        score -= 0.25 * len(item.get("missing") or [])
        return score

    def _merge_infusion_field_results(self, parsed_results):
        valid_results = [item for item in parsed_results or [] if isinstance(item, dict)]
        if not valid_results:
            return parse_required_fields(None)
        valid_results = sorted(valid_results, key=self._score_infusion_result, reverse=True)

        merged = {"liquid": None, "concentration": None, "volume": None}
        raw_parts = []
        for item in valid_results:
            raw_text = str(item.get("raw_text") or "").strip()
            if raw_text and raw_text not in raw_parts:
                raw_parts.append(raw_text)
            for field in ("liquid", "concentration", "volume"):
                value = item.get(field)
                if value and merged[field] is None:
                    merged[field] = value

        missing = [field for field in ("liquid", "concentration", "volume") if merged[field] is None]
        merged["raw_text"] = " | ".join(raw_parts)
        merged["missing"] = missing
        merged["status"] = "done" if not missing else ("ocr_empty" if not raw_parts else "partial")
        return merged

    def _recognize_profile(
        self,
        image,
        module,
        padding,
        join_mode="concat",
        reset_timing=True,
        split_rows=False,
        return_texts=False,
        angle_cls=False,
    ):
        if image is None:
            return ""

        start = time.perf_counter()
        if reset_timing:
            self.last_timing = {}

        original_image = image
        angle_pred = 0
        if angle_cls:
            t = time.perf_counter()
            rotated, pred = self._classify_and_rotate(image)
            self.last_timing["angle_cls"] = time.perf_counter() - t
            self.last_timing["angle_pred"] = float(pred)
            self._save_debug_image(f"angle_rotated_pred_{pred}", rotated)
            image = rotated
            angle_pred = pred

        texts = self._recognize_candidate(image, module, padding, split_rows)
        if angle_cls and angle_pred != 0:
            rotated_score = self._score_medicine_texts(texts)
            self.last_timing["angle_rotated_score"] = rotated_score
            if rotated_score < 12.0:
                self._save_debug_image("angle_fallback_original", original_image)
                fallback_texts = self._recognize_candidate(
                    original_image,
                    module,
                    padding,
                    split_rows,
                    timing_prefix="fallback_",
                )
                fallback_score = self._score_medicine_texts(fallback_texts)
                self.last_timing["angle_fallback_score"] = fallback_score
                if fallback_score > rotated_score + 1.0:
                    texts = fallback_texts
                    self.last_timing["angle_used_fallback"] = 1.0
                else:
                    self.last_timing["angle_used_fallback"] = 0.0
            else:
                self.last_timing["angle_used_fallback"] = 0.0

        if not texts:
            self.last_timing["total"] = time.perf_counter() - start
            return [] if return_texts else ""
        self.last_timing["total"] = time.perf_counter() - start

        if return_texts:
            return texts
        if join_mode == "first":
            return texts[0] if texts else ""
        return "".join(texts)

    def recognize(self, image, line_gap_threshold=20):
        values = self.recognize_bottles_batch([image])
        return values[0] if values else ""

    def recognize_bottles_batch(self, images):
        tasks = [
            ("bottle", index, image)
            for index, image in enumerate(images)
            if image is not None and image.size > 0
        ]
        results = self.recognize_task_batch(tasks)
        return [
            results.get(("bottle", index), {}).get("text", "")
            for index in range(len(images))
        ]

    def recognize_yaodai(self, image, line_gap_threshold=20):
        # Class 1: bag/label, use Test_OCR_tight_lines.py tuned tight-line logic only.
        self.last_timing = {}
        name_text = self._recognize_label_name_roi(image)
        if name_text:
            self.last_timing["total"] = self.last_timing.get("name_roi_total", 0.0)
            return name_text

        fallback_timing = dict(self.last_timing)
        text = self._recognize_profile(
            image,
            tight_ocr,
            TIGHT_LINE_PADDING,
            join_mode="first",
            split_rows="label",
            angle_cls=True,
        )
        for key, value in fallback_timing.items():
            self.last_timing[f"name_{key}"] = value
        return text

    def recognize_shuyedai(self, image, line_gap_threshold=20):
        # Class 2: infusion bag. Correct the full YOLO crop with CLS first,
        # then extract the label candidates from the corrected crop.
        start = time.perf_counter()
        self.last_timing = {}
        if image is None:
            return parse_required_fields(None)

        t_cls = time.perf_counter()
        rotated_images, preds = self._classify_and_rotate_batch([image], timing_prefix="shuye_")
        rotated_image = rotated_images[0]
        self.last_timing["shuye_angle_cls"] = time.perf_counter() - t_cls
        self.last_timing["shuye_angle_pred"] = float(sum(preds))

        t_pre = time.perf_counter()
        candidates = self._extract_infusion_candidate_rois(rotated_image)
        self.last_timing["preprocess"] = time.perf_counter() - t_pre
        valid_candidates = [
            (name, roi)
            for name, roi in candidates
            if roi is not None and roi.size > 0
        ]
        if not valid_candidates:
            self.last_timing["total"] = time.perf_counter() - start
            return parse_required_fields(None)

        texts_list = self._recognize_candidates_batch(
            [roi for _name, roi in valid_candidates],
            infusion_ocr,
            INFUSION_PADDING,
            split_rows=False,
        )

        parsed_results = []
        for cand_idx, ((name, _roi), texts) in enumerate(zip(valid_candidates, texts_list)):
            self.last_timing[f"candidate_{cand_idx}_{name}_texts"] = len(texts)
            self.last_timing[f"candidate_{cand_idx}_texts"] = len(texts)
            parsed_results.append(self._parse_infusion_texts(texts))
        self.last_timing["total"] = time.perf_counter() - start
        return self._merge_infusion_field_results(parsed_results)

    def recognize_shuyedai_batch(self, images):
        start = time.perf_counter()
        self.last_timing = {}
        if not images:
            return []

        valid_image_pairs = [(idx, image) for idx, image in enumerate(images) if image is not None and image.size > 0]
        if not valid_image_pairs:
            return [parse_required_fields(None) for _ in images]

        t_cls = time.perf_counter()
        rotated_images, preds = self._classify_and_rotate_batch(
            [image for _idx, image in valid_image_pairs],
            timing_prefix="shuye_",
        )
        self.last_timing["shuye_angle_cls"] = time.perf_counter() - t_cls
        self.last_timing["shuye_angle_pred"] = float(sum(preds))

        t_pre = time.perf_counter()
        candidate_entries = []
        for (idx, _image), rotated_image in zip(valid_image_pairs, rotated_images):
            for name, roi in self._extract_infusion_candidate_rois(rotated_image):
                candidate_entries.append((idx, name, roi))
        self.last_timing["preprocess"] = time.perf_counter() - t_pre

        valid_pairs = [(idx, name, roi) for idx, name, roi in candidate_entries if roi is not None and roi.size > 0]
        if not valid_pairs:
            return [parse_required_fields(None) for _ in images]

        texts_list = self._recognize_candidates_batch(
            [roi for _idx, _name, roi in valid_pairs],
            infusion_ocr,
            INFUSION_PADDING,
            split_rows=False,
        )

        parsed_by_index = {idx: [] for idx, _image in enumerate(images)}
        for (idx, _name, _roi), texts in zip(valid_pairs, texts_list):
            parsed_by_index.setdefault(idx, []).append(self._parse_infusion_texts(texts))

        results = [parse_required_fields(None) for _ in images]
        for idx, parsed_results in parsed_by_index.items():
            results[idx] = self._merge_infusion_field_results(parsed_results)
        self.last_timing["total"] = time.perf_counter() - start
        return results

    def recognize_task_batch(self, task_group):
        collect_timing = _collect_timing_enabled()
        start = time.perf_counter() if collect_timing else 0.0
        self.last_timing = {} if collect_timing else _NoopTiming()
        results = {}
        bottle_tasks = [(task_type, idx, crop) for task_type, idx, crop in task_group if task_type == "bottle"]
        bag_tasks = [(task_type, idx, crop) for task_type, idx, crop in task_group if task_type == "bag"]
        shuye_tasks = [(task_type, idx, crop) for task_type, idx, crop in task_group if task_type == "shuye"]
        mixed_candidates = []

        cls_entries = []
        for _, idx, crop in bottle_tasks:
            if crop is not None and crop.size > 0:
                cls_entries.append({"owner": ("bottle", idx), "image": crop})

        for _, idx, crop in bag_tasks:
            if crop is not None and crop.size > 0:
                cls_entries.append({"owner": ("bag", idx), "image": crop})

        for _, idx, crop in shuye_tasks:
            if crop is not None and crop.size > 0:
                cls_entries.append({"owner": ("shuye", idx), "image": crop})

        cls_by_owner = {}
        if collect_timing:
            self.last_timing["shared_cls_inputs"] = len(cls_entries)
            self.last_timing["shared_cls_batches"] = int(np.ceil(len(cls_entries) / float(max(1, self.cls_batch_size)))) if cls_entries else 0
        if cls_entries:
            t_cls = time.perf_counter() if collect_timing else 0.0
            cls_images = [entry["image"] for entry in cls_entries]
            rotated_images, preds = self._classify_and_rotate_batch(cls_images, timing_prefix="shared_")
            cls_elapsed = time.perf_counter() - t_cls if collect_timing else 0.0
            for entry, rotated, pred in zip(cls_entries, rotated_images, preds):
                cls_by_owner[entry["owner"]] = {
                    "image": entry["image"],
                    "rotated": rotated,
                    "pred": pred,
                }

            bottle_preds = [value["pred"] for owner, value in cls_by_owner.items() if owner[0] == "bottle"]
            bag_preds = [value["pred"] for owner, value in cls_by_owner.items() if owner[0] == "bag"]
            shuye_preds = [value["pred"] for owner, value in cls_by_owner.items() if owner[0] == "shuye"]
            if collect_timing:
                self.last_timing["shared_angle_cls"] = cls_elapsed
                self.last_timing["shared_angle_pred"] = float(sum(preds))
                if bottle_tasks:
                    self.last_timing["bottle_angle_cls"] = cls_elapsed
                    self.last_timing["bottle_angle_pred"] = float(sum(bottle_preds))
                if bag_tasks:
                    self.last_timing["bag_angle_cls"] = cls_elapsed
                    self.last_timing["bag_angle_pred"] = float(sum(bag_preds))
                if shuye_tasks:
                    self.last_timing["shuye_angle_cls"] = cls_elapsed
                    self.last_timing["shuye_angle_pred"] = float(sum(shuye_preds))

        bottle_candidates = {}
        for _, idx, crop in bottle_tasks:
            cls_item = cls_by_owner.get(("bottle", idx))
            if cls_item is None:
                continue
            self._save_predet_image(f"bottle_{idx + 1}_after_cls", cls_item["rotated"])
            candidate = {
                "owner": ("bottle", idx),
                "image": cls_item["rotated"],
                "module": tight_ocr,
                "padding": BOTTLE_LINE_PADDING,
                "split_rows": False,
                "preserve_rotated_boxes": True,
                "final_short_side_padding_ratio": BOTTLE_LINE_PADDING["final_short_side_padding_ratio"],
                "final_min_short_side_padding": BOTTLE_LINE_PADDING["final_min_short_side_padding"],
                "final_max_short_side_padding": BOTTLE_LINE_PADDING["final_max_short_side_padding"],
            }
            bottle_candidates[idx] = candidate
            mixed_candidates.append(candidate)

        bag_name_owners = {}
        if bag_tasks:
            t_roi = time.perf_counter() if collect_timing else 0.0
            card_roi_specs = (
                (0.00, 0.00, 0.56, 0.45),
                (0.00, 0.08, 0.60, 0.55),
                (0.02, 0.00, 0.48, 0.35),
            )
            rotated_roi_specs = (
                (0.00, 0.00, 0.68, 0.42),
                (0.00, 0.00, 0.82, 0.58),
            )
            for _, idx, _crop in bag_tasks:
                cls_item = cls_by_owner.get(("bag", idx))
                if cls_item is None:
                    continue
                rotated = cls_item["rotated"]
                card = self._extract_label_card(rotated)
                self._save_debug_image("label_card", card)

                roi_idx = 0

                def append_name_roi(raw_roi, debug_name, split_rows=True):
                    nonlocal roi_idx
                    if raw_roi is None or raw_roi.size <= 0:
                        return
                    roi = self._prepare_name_roi(raw_roi.copy())
                    if roi is None or roi.size <= 0:
                        return
                    owner = ("bag_name", idx, roi_idx)
                    roi_idx += 1
                    bag_name_owners.setdefault(idx, []).append(owner)
                    self._save_debug_image(debug_name, roi)
                    mixed_candidates.append({
                        "owner": owner,
                        "image": roi,
                        "module": tight_ocr,
                        "padding": NAME_ROI_PADDING,
                        "split_rows": split_rows,
                    })

                height, width = card.shape[:2]
                for x1r, y1r, x2r, y2r in card_roi_specs:
                    x1 = max(0, min(width - 1, int(width * x1r)))
                    y1 = max(0, min(height - 1, int(height * y1r)))
                    x2 = max(x1 + 1, min(width, int(width * x2r)))
                    y2 = max(y1 + 1, min(height, int(height * y2r)))
                    append_name_roi(card[y1:y2, x1:x2], f"label_name_roi_{idx}_{roi_idx}")
                append_name_roi(card, f"label_full_card_name_roi_{idx}_{roi_idx}")

                rotated_h, rotated_w = rotated.shape[:2]
                for x1r, y1r, x2r, y2r in rotated_roi_specs:
                    x1 = max(0, min(rotated_w - 1, int(rotated_w * x1r)))
                    y1 = max(0, min(rotated_h - 1, int(rotated_h * y1r)))
                    x2 = max(x1 + 1, min(rotated_w, int(rotated_w * x2r)))
                    y2 = max(y1 + 1, min(rotated_h, int(rotated_h * y2r)))
                    append_name_roi(rotated[y1:y2, x1:x2], f"label_rotated_name_roi_{idx}_{roi_idx}")
                mixed_candidates.append({
                    "owner": ("bag_fallback", idx),
                    "image": rotated,
                    "module": tight_ocr,
                    "padding": TIGHT_LINE_PADDING,
                    "split_rows": "label",
                })
            if collect_timing:
                self.last_timing["bag_name_roi_extract"] = time.perf_counter() - t_roi

        shuye_roi_owners = {}
        shuye_candidate_count = 0
        if shuye_tasks:
            t_pre = time.perf_counter() if collect_timing else 0.0
            for _, idx, _crop in shuye_tasks:
                cls_item = cls_by_owner.get(("shuye", idx))
                if cls_item is None:
                    continue
                rotated_crop = cls_item["rotated"]
                for roi_name, roi in self._extract_infusion_candidate_rois(rotated_crop):
                    if roi is None or roi.size <= 0:
                        continue
                    owner_name = f"{roi_name}_{len(shuye_roi_owners.get(idx, []))}"
                    owner = ("shuye", idx, owner_name)
                    shuye_roi_owners.setdefault(idx, []).append(owner)
                    shuye_candidate_count += 1
                    self._save_predet_image(f"shuye_{idx + 1}_{owner_name}_before_det", roi)
                    mixed_candidates.append({
                        "owner": owner,
                        "image": roi,
                        "module": infusion_ocr,
                        "padding": INFUSION_PADDING,
                        "split_rows": False,
                        "merge_boxes": True,
                        "filter_edge_boxes": True,
                        "merge_kwargs": INFUSION_MERGE_KWARGS,
                    })
            if collect_timing:
                self.last_timing["shuye_preprocess"] = time.perf_counter() - t_pre
                self.last_timing["shuye_candidate_rois"] = shuye_candidate_count

        if collect_timing:
            self.last_timing["shared_det_inputs"] = len(mixed_candidates)
            self.last_timing["shared_det_batches"] = int(np.ceil(len(mixed_candidates) / float(max(1, self.det_batch_size)))) if mixed_candidates else 0
        texts_by_owner = self._recognize_mixed_candidates_batch(mixed_candidates)
        shared_timing = dict(self.last_timing) if collect_timing else {}

        for _, idx, _crop in bottle_tasks:
            texts = texts_by_owner.get(("bottle", idx), [])
            cls_item = cls_by_owner.get(("bottle", idx))
            candidate = bottle_candidates.get(idx, {})
            result = {
                "text": "".join(texts),
                "timing": shared_timing,
                "det_region_count": int(candidate.get("det_region_count", 0)),
                "rec_nonempty_count": len(texts),
                "det_visualization": candidate.get("det_visualization"),
            }
            if cls_item is not None:
                result["classify_image"] = cls_item["rotated"]
                result["classify_image_already_rotated"] = True
                result["classify_angle_pred"] = cls_item["pred"]
            results[("bottle", idx)] = result

        for _, idx, _crop in bag_tasks:
            cleaned = []
            for owner in bag_name_owners.get(idx, []):
                for text in texts_by_owner.get(owner, []):
                    text = "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")
                    if 2 <= len(text) <= 6:
                        cleaned.append(text[:4])
            fallback = texts_by_owner.get(("bag_fallback", idx), [])
            text = cleaned[0] if cleaned else (fallback[0] if fallback else "")
            result = {
                "text": text,
                "timing": shared_timing,
                "name_roi_count": len(bag_name_owners.get(idx, [])),
            }
            cls_item = cls_by_owner.get(("bag", idx))
            if cls_item is not None:
                result["classify_angle_pred"] = cls_item["pred"]
            results[("bag", idx)] = result

        for _, idx, _crop in shuye_tasks:
            parsed_results = []
            for owner in shuye_roi_owners.get(idx, []):
                texts = texts_by_owner.get(owner, [])
                parsed_results.append(self._parse_infusion_texts(texts))
            value = self._merge_infusion_field_results(parsed_results)
            results[("shuye", idx)] = {
                "text": value,
                "timing": shared_timing,
                "candidate_roi_count": len(shuye_roi_owners.get(idx, [])),
            }

        self.last_timing = {"batch_total": time.perf_counter() - start} if collect_timing else {}
        return results
