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
    ):
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.area_threshold = float(area_threshold)
        self.sharpness_threshold = float(sharpness_threshold)
        self.core = RKNNLite.NPU_CORE_1
        self.lock = get_rknn_lock(self.core, secondary_domain=True)
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(model_path) != 0:
            raise RuntimeError(f"Load basket RKNN failed: {model_path}")
        if self.rknn.init_runtime(core_mask=self.core) != 0:
            raise RuntimeError("Init basket RKNN runtime failed")
        self.armed = True

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
        if not valid:
            if area_ratio < self.area_threshold:
                self.armed = True

        triggered = bool(self.armed and valid)
        if triggered:
            self.armed = False
        return {
            "triggered": triggered,
            "box": box,
            "confidence": confidence,
            "area_ratio": area_ratio,
            "sharpness": sharpness,
            "armed": self.armed,
        }

    def draw_result(self, frame, result, monitoring=True, processing=False):
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
            x1, y1, x2, y2 = [int(value) for value in box]
            cv2.rectangle(visual, (x1, y1), (x2, y2), color, 8)
        state = "PROCESSING" if processing else ("MONITORING" if monitoring else "PAUSED")
        info = (
            f"Basket conf={float(result.get('confidence', 0.0)):.3f} "
            f"area={area_ratio:.1%} sharp={sharpness:.0f} "
            f"{state}"
        )
        cv2.putText(visual, info, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.15, color, 3, cv2.LINE_AA)
        return visual

    def reset_tracking(self):
        pass

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None
