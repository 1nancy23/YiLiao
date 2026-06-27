import cv2
import numpy as np
from rknnlite.api import RKNNLite

from src.identification.rknn_runtime_lock import get_rknn_lock


class BasketAutoTrigger:
    def __init__(
        self,
        model_path,
        input_size=640,
        conf_threshold=0.45,
        iou_threshold=0.45,
        area_threshold=0.40,
        sharpness_threshold=55.0,
        stable_frames_required=6,
    ):
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.area_threshold = float(area_threshold)
        self.sharpness_threshold = float(sharpness_threshold)
        self.stable_frames_required = max(1, int(stable_frames_required))
        self.core = RKNNLite.NPU_CORE_1
        self.lock = get_rknn_lock(self.core, secondary_domain=True)
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(model_path) != 0:
            raise RuntimeError(f"Load basket RKNN failed: {model_path}")
        if self.rknn.init_runtime(core_mask=self.core) != 0:
            raise RuntimeError("Init basket RKNN runtime failed")
        self.armed = True
        self.candidate_active = False
        self.invalid_moving_frames = 0
        self.previous_valid_box = None
        self.previous_static_thumb = None
        self.stable_frames = 0

    @staticmethod
    def _box_iou(box_a, box_b):
        if box_a is None or box_b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _box_center_shift_ratio(box_a, box_b, frame):
        if box_a is None or box_b is None:
            return 1.0
        height, width = frame.shape[:2]
        diag = max(1.0, float((width * width + height * height) ** 0.5))
        acx = (box_a[0] + box_a[2]) * 0.5
        acy = (box_a[1] + box_a[3]) * 0.5
        bcx = (box_b[0] + box_b[2]) * 0.5
        bcy = (box_b[1] + box_b[3]) * 0.5
        return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / diag)

    def _is_box_static(self, box, frame):
        if self.previous_valid_box is None:
            return True
        iou = self._box_iou(box, self.previous_valid_box)
        shift_ratio = self._box_center_shift_ratio(box, self.previous_valid_box, frame)
        return iou >= 0.88 and shift_ratio <= 0.03

    def _is_frame_static(self, frame):
        if frame is None or frame.size == 0:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        if self.previous_static_thumb is None:
            self.previous_static_thumb = thumb
            return True
        diff = cv2.absdiff(thumb, self.previous_static_thumb)
        mean_diff = float(np.mean(diff))
        self.previous_static_thumb = thumb
        return mean_diff <= 4.5

    def _detect_largest(self, frame):
        height, width = frame.shape[:2]
        # Keep this preprocessing and postprocessing aligned with YOLOTileProcessor.
        resized = cv2.resize(frame, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        batch = np.expand_dims(rgb, 0).astype(np.float32)
        with self.lock:
            outputs = self.rknn.inference(inputs=[batch])
        predictions = np.asarray(outputs[0])[0].T
        class_scores = predictions[:, 4:]
        scores = np.max(class_scores, axis=1)
        boxes = predictions[scores >= self.conf_threshold, :4]
        scores = scores[scores >= self.conf_threshold]
        if not len(boxes):
            return None, 0.0

        scale_x = width / float(self.input_size)
        scale_y = height / float(self.input_size)
        cx, cy, box_w, box_h = boxes.T
        xyxy = np.column_stack((
            (cx - box_w / 2) * scale_x,
            (cy - box_h / 2) * scale_y,
            (cx + box_w / 2) * scale_x,
            (cy + box_h / 2) * scale_y,
        ))
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height)
        boxes_xywh = xyxy.copy()
        boxes_xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        boxes_xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        keep = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )
        if not len(keep):
            return None, 0.0
        keep = np.asarray(keep, dtype=np.int64).reshape(-1)
        xyxy = xyxy[keep]
        scores = scores[keep]
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        index = int(np.argmax(areas))
        return xyxy[index].tolist(), float(scores[index])

    def update(self, frame):
        box, confidence = self._detect_largest(frame)
        frame_area = float(max(1, frame.shape[0] * frame.shape[1]))
        area_ratio = 0.0
        sharpness = 0.0

        if box is not None:
            x1, y1, x2, y2 = [int(value) for value in box]
            area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / frame_area
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        valid = box is not None and area_ratio >= self.area_threshold and sharpness >= self.sharpness_threshold
        if valid:
            self.candidate_active = True
            self.invalid_moving_frames = 0
            if self._is_frame_static(frame):
                self.stable_frames += 1
            else:
                self.stable_frames = 1
            self.previous_valid_box = box
        elif self.candidate_active:
            # Once a basket candidate is established, do not let a single
            # unstable YOLO frame reset the static wait. Staticness is measured
            # from the raw frame, while YOLO only decides candidate entry/removal.
            if self._is_frame_static(frame):
                self.stable_frames += 1
                self.invalid_moving_frames = 0
            else:
                self.stable_frames = 1
                self.invalid_moving_frames += 1
                if self.invalid_moving_frames >= 3:
                    self.candidate_active = False
                    self.previous_valid_box = None
                    self.previous_static_thumb = None
                    self.stable_frames = 0
                    if area_ratio < self.area_threshold:
                        self.armed = True
        else:
            self.previous_valid_box = None
            self.previous_static_thumb = None
            self.stable_frames = 0
            self.invalid_moving_frames = 0
            if area_ratio < self.area_threshold:
                self.armed = True

        stable_enough = self.stable_frames >= self.stable_frames_required
        triggered = bool(self.armed and self.candidate_active and stable_enough)
        if triggered:
            self.armed = False
            self.candidate_active = False
            self.invalid_moving_frames = 0
        return {
            "triggered": triggered,
            "box": box,
            "confidence": confidence,
            "area_ratio": area_ratio,
            "sharpness": sharpness,
            "valid": valid,
            "candidate_active": self.candidate_active,
            "invalid_moving_frames": self.invalid_moving_frames,
            "stable_frames": self.stable_frames,
            "stable_required": self.stable_frames_required,
            "stable_enough": stable_enough,
            "armed": self.armed,
        }

    def draw_result(self, frame, result, monitoring=True, processing=False):
        # Draw on a resized copy to avoid duplicating a full 4K frame
        max_w, max_h = 1024, 600
        h, w = frame.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            visual = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            visual = frame.copy()
        box = result.get("box")
        area_ratio = float(result.get("area_ratio", 0.0))
        sharpness = float(result.get("sharpness", 0.0))
        valid = (
            area_ratio >= self.area_threshold
            and sharpness >= self.sharpness_threshold
        )
        color = (0, 210, 0) if valid else (0, 180, 255)
        if box is not None:
            x1, y1, x2, y2 = [int(value * scale) for value in box]
            cv2.rectangle(visual, (x1, y1), (x2, y2), color, 8)
        state = "PROCESSING" if processing else ("MONITORING" if monitoring else "PAUSED")
        info = (
            f"Basket conf={float(result.get('confidence', 0.0)):.3f} "
            f"area={area_ratio:.1%} sharp={sharpness:.0f} "
            f"stable={int(result.get('stable_frames', 0))}/{self.stable_frames_required} "
            f"{state}"
        )
        cv2.putText(visual, info, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.15, color, 3, cv2.LINE_AA)
        return visual

    def reset_tracking(self):
        self.candidate_active = False
        self.invalid_moving_frames = 0
        self.previous_valid_box = None
        self.previous_static_thumb = None
        self.stable_frames = 0

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None
