import cv2
import numpy as np
# import torch
from typing import List, Tuple, Optional
# from ultralytics import YOLO
from rknnlite.api import RKNNLite


def postprocess_rknn_output(output, original_image, conf_threshold=0.5, iou_threshold=0.5):
    """
    处理 RKNN 输出的 (1, 7, 8400) 张量，并在图像上绘制检测框

    参数:
        output: RKNN 模型的推理输出，形状为 (1, 7, 8400)
        original_image: 原始输入图像 (numpy array)
        conf_threshold: 置信度阈值 (默认 0.5)
        iou_threshold: NMS 的 IoU 阈值

    返回:
        处理后的图像
    """
    # --- 1. 数据预处理 ---
    # output 形状: (1, 7, 8400)
    # 转置为 (8400, 7)，每一行代表一个检测框
    predictions = output[0].T  # shape: (8400, 7)

    # --- 2. 提取坐标和类别分数 ---
    # 假设模型是 3 分类 (因为 4 + 3 = 7)
    # 如果是 80 分类模型这里会是 84，说明你的模型确实是 3 分类
    num_classes = predictions.shape[1] - 4
    class_scores = predictions[:, 4:]  # 取出所有类别的分数 (8400, num_classes)

    # 找到每个框分数最高的类别及其索引
    max_class_scores = np.max(class_scores, axis=1)  # (8400,)
    class_ids = np.argmax(class_scores, axis=1)  # (8400,)

    # --- 3. 置信度过滤 (关键步骤) ---
    # 这里使用你要求的 0.5 阈值
    conf_mask = max_class_scores >= conf_threshold
    filtered_boxes = predictions[conf_mask]
    filtered_scores = max_class_scores[conf_mask]
    filtered_class_ids = class_ids[conf_mask]

    print(f"找到 {len(filtered_boxes)} 个高于 {conf_threshold} 阈值的检测框")

    # 如果没有检测到任何东西，直接返回原图
    if len(filtered_boxes) == 0:
        return original_image

    # --- 4. 坐标转换 (归一化 -> 像素) ---
    # 获取图像尺寸
    img_h, img_w = original_image.shape[:2]

    # 提取 cx, cy, w, h (注意：此时还是归一化的值 0~1)
    boxes = filtered_boxes[:, :4]

    # 将归一化坐标转换为像素坐标
    # 转换公式: pixel = normalized * image_size
    # 然后将 (cx, cy, w, h) 转换为 (x1, y1, x2, y2) 用于绘制
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 计算左上角和右下角坐标
    # x1 = x_c - (w / 2)
    # y1 = y_c - (h / 2)
    # x2 = x_c + (w / 2)
    # y2 = y_c + (h / 2)

    # 合并为 NMS 需要的格式
    detection_boxes = np.stack([x1, y1, x2, y2], axis=1)  # shape: (N, 4)

    # --- 5. 非极大值抑制 (NMS) ---
    # 这一步是为了去掉重叠的框，只保留分数最高的那个
    indices = cv2.dnn.NMSBoxes(detection_boxes.tolist(), filtered_scores.tolist(), conf_threshold, iou_threshold)

    # --- 6. 绘制结果 ---
    result_image = original_image.copy()

    # 颜色定义 (BGR格式)
    colors = {
        0: (0, 255, 0),  # 绿色
        1: (255, 0, 0),  # 蓝色
        2: (0, 0, 255),  # 红色
        # 可以根据类别数量扩展颜色
    }

    # 遍历保留下来的框
    for i in indices:
        # OpenCV 4.x 与 5.x 的 NMSBoxes 返回值类型不同，这里做兼容处理
        idx = i if isinstance(i, np.integer) else i[0]

        box = detection_boxes[idx]
        score = filtered_scores[idx]
        class_id = filtered_class_ids[idx]

        x1, y1, x2, y2 = map(int, box)  # 转为整数

        # 获取颜色
        color = colors.get(class_id, (255, 255, 255))  # 默认白色

        # 绘制矩形框
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

        # 准备标签文本
        label = f"Class {class_id}: {score:.2f}"

        # 绘制背景矩形和文本
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result_image, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
        cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        print(f"绘制框: 类别={class_id}, 置信度={score:.2f}, 坐标=({x1},{y1},{x2},{y2})")

    return result_image


class YOLOTileProcessor:
    """
    YOLO 大图切片处理器
    支持将超大分辨率图像切分为 640x640 瓦片进行推理，并合并检测结果
    """

    def __init__(
            self,
            model=None,
            model_path: str = './model_yolo_0510.rknn',
            device: str = 'cuda',
            tile_size: int = 640,
            overlap: int = 128,
            conf_thres: float = 0.2,
            iou_thres: float = 0.90,
            batch_size: int = 1,
    ):
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.batch_size = batch_size
        self.model_path = model_path
        
        self.rknn_lite = RKNNLite()
        self.model = self.rknn_lite.load_rknn(self.model_path)
        self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        print(self.model)
            # self.model = YOLO(model_path)
        # else:
            # raise ValueError("必须提供 model 或 model_path 之一")

    #     self._warmup()

    # def _warmup(self):
    #     print("🔥 GPU 预热中...")
    #     dummy = torch.randn(1, 3, self.tile_size, self.tile_size).to(self.device)
    #     self.model(dummy, verbose=False)
    #     print("✅ 预热完成")

    def _extract_tiles(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        h, w = image.shape[:2]
        stride = self.tile_size - self.overlap
        tiles = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x1, y1 = x, y
                x2, y2 = min(x + self.tile_size, w), min(y + self.tile_size, h)
                tile = image[y1:y2, x1:x2]
                if tile.shape[0] != self.tile_size or tile.shape[1] != self.tile_size:
                    pad_bottom = self.tile_size - tile.shape[0]
                    pad_right = self.tile_size - tile.shape[1]
                    tile = cv2.copyMakeBorder(
                        tile, 0, pad_bottom, 0, pad_right,
                        cv2.BORDER_CONSTANT, value=(114, 114, 114)
                    )
                tiles.append((tile, (x1, y1, x2, y2)))
        return tiles

    def _map_boxes_to_original(self, results, tile_offset: Tuple[int, int],
                               tile_size: Tuple[int, int]) -> List[List[float]]:
        print("results.shape:", results.shape)
        boxes = results.transpose(1, 0)
        # print(boxes[0:2])
        if boxes is None:
            return []
        mapped = []
        offset_x, offset_y = tile_offset
        valid_w, valid_h = tile_size
        for box in boxes:
            x_c, y_c, w, h = box[0:4].tolist()
            cls_id = int(box[4:].argmax())
            conf = box[4:][cls_id]
            if conf < self.conf_thres:
                continue
            # print('conf',conf,'cls_id',cls_id)
            x1 = x_c - (w / 2)
            y1 = y_c - (h / 2)
            x2 = x_c + (w / 2)
            y2 = y_c + (h / 2)
            x1 = min(x1, valid_w)
            y1 = min(y1, valid_h)
            x2 = min(x2, valid_w)
            y2 = min(y2, valid_h)
            x1 += offset_x
            y1 += offset_y
            x2 += offset_x
            y2 += offset_y
            if x2 > x1 and y2 > y1:
                mapped.append([x1, y1, x2, y2, conf, cls_id])
        return mapped

    def _global_nms(self, detections: List[List[float]]) -> List[List[float]]:
        print("detections:", detections)
        if not detections:
            return []
        detections = np.array(detections)
        boxes = detections[:, :4]
        scores = detections[:, 4]
        class_ids = detections[:, 5]
        keep_indices = []
        for cls in np.unique(class_ids):
            idxs = np.where(class_ids == cls)[0]
            cls_boxes = boxes[idxs]
            cls_scores = scores[idxs]
            nms_idx = cv2.dnn.NMSBoxes(
                cls_boxes.tolist(), cls_scores.tolist(),
                self.conf_thres, self.iou_thres
            )
            if len(nms_idx) > 0:
                keep_indices.extend(idxs[nms_idx.flatten()])
        return detections[keep_indices].tolist()

    def process_image(self, image: np.ndarray) -> List[List[float]]:
        tiles = self._extract_tiles(image)
        if not tiles:
            return []

        all_detections = []
        for i in range(0, len(tiles), self.batch_size):
            batch = tiles[i:i + self.batch_size]

            batch_imgs = [t[0] for t in batch]
            batch_offsets = [t[1][:2] for t in batch]
            batch_sizes = [(t[1][2] - t[1][0], t[1][3] - t[1][1]) for t in batch]
            batch_imgs = np.stack(batch_imgs, axis=0)

            print(len(batch_imgs), batch_imgs.shape)
            # print(batch_imgs)
            results = self.rknn_lite.inference(inputs=[batch_imgs])
            print(batch_offsets)
            for r, offset, size in zip(results[0], batch_offsets, batch_sizes):
                all_detections.extend(self._map_boxes_to_original(r, offset, size))

        return self._global_nms(all_detections)
        # return all_detections

    def draw_detections(
            self, image: np.ndarray, detections: List[List[float]],
            class_names: Optional[List[str]] = None, color_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        img = image.copy()
        if color_map is None:
            np.random.seed(42)
            color_map = np.random.randint(0, 255, (100, 3), dtype=np.uint8)

        for det in detections:
            # 鲁棒解包：强制扁平化，防止嵌套列表
            flat = np.array(det).flatten().tolist()
            x1, y1, x2, y2, conf, cls_id = flat[:6]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = color_map[int(cls_id) % len(color_map)].tolist()
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{class_names[int(cls_id)] if class_names else f'Class {int(cls_id)}'}: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img

    # def crop_by_class(
    #         self, image: np.ndarray, detections: List[List[float]],
    #         bag_class_id: int = 1, bottle_class_id: int = 0
    # ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    #     """
    #     根据类别 ID 裁剪药袋和药瓶区域
    #     Returns:
    #         (cropped_bags, cropped_bottles)
    #     """
    #     bags, bottles = [], []
    #     for det in detections:
    #         # 鲁棒解包：强制扁平化
    #         flat = np.array(det).flatten().tolist()
    #         x1, y1, x2, y2, conf, cls_id = flat[:6]
    #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #         x1, y1 = max(0, x1), max(0, y1)
    #         x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    #         if x2 <= x1 or y2 <= y1:
    #             continue
    #         crop = image[y1:y2, x1:x2].copy()
    #         if int(cls_id) == bag_class_id:
    #             bags.append(crop)
    #         elif int(cls_id) == bottle_class_id:
    #             bottles.append(crop)
    #     return bags, bottles

    def crop_by_class(
            self, image: np.ndarray, detections: List[List[float]],
            bag_class_id: int = 1, bottle_class_id: int = 0, shuye_class_id: int = 2
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        向量化版本：预处理 O(1)，仅裁剪 O(n)
        """
        if not detections:
            return [], []

        h, w = image.shape[:2]

        # ★ 一次性转为矩阵，替代逐条 flatten + tolist
        det_array = np.asarray(detections, dtype=np.float32).reshape(-1, 6)

        # ★ 向量化坐标裁剪（单次 numpy 调用）
        coords = det_array[:, :4].astype(np.int32)
        np.clip(coords[:, 0], 0, w, out=coords[:, 0])
        np.clip(coords[:, 1], 0, h, out=coords[:, 1])
        np.clip(coords[:, 2], 0, w, out=coords[:, 2])
        np.clip(coords[:, 3], 0, h, out=coords[:, 3])

        # ★ 向量化有效性过滤 + 分类
        valid = (coords[:, 2] > coords[:, 0]) & (coords[:, 3] > coords[:, 1])
        cls_ids = det_array[:, 5].astype(np.int32)

        bag_mask = valid & (cls_ids == bag_class_id)
        bottle_mask = valid & (cls_ids == bottle_class_id)
        shuye_mask = valid & (cls_ids == shuye_class_id)
        # ★ 只对命中的框做裁剪（不可避免的最小循环）
        bags = [image[y1:y2, x1:x2].copy() for x1, y1, x2, y2 in coords[bag_mask]]
        bottles = [image[y1:y2, x1:x2].copy() for x1, y1, x2, y2 in coords[bottle_mask]]
        shuyes = [image[y1:y2, x1:x2].copy() for x1, y1, x2, y2 in coords[shuye_mask]]
        return bags, bottles, shuyes
    
    def process_frames_batch(self, frames, output_type='overlay', alpha=0.5):
        result_frames, predictions = [], []
        for frame in frames:
            # dets = self.process_image(frame)
            # print("原始大小", frame.shape)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = self.process_specific_slice(frame, 100, 100, 3400, 2160)
            # 确保 dets 是纯 Python 列表，避免后续序列化问题
            dets_clean = [list(map(float, d[:6])) for d in dets]
            predictions.append(dets_clean)

            if output_type in ('overlay', 'color'):
                vis = self.draw_detections(frame, dets_clean)
                result_frames.append(vis)
            elif output_type == 'side_by_side':
                vis = self.draw_detections(frame, dets_clean)
                combined = np.concatenate([frame, vis], axis=1)
                result_frames.append(combined)
            elif output_type == 'gray':
                vis = self.draw_detections(frame, dets_clean)
                gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
                result_frames.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            else:
                result_frames.append(frame)
        return result_frames, predictions

    def process_specific_slice(self,
                               image: np.ndarray,
                               x1: int, y1: int, x2: int, y2: int) -> List[List[float]]:
        """
        处理原图中指定的一个切片区域，并将结果映射回原图坐标

        Args:
            image: 原始大图
            x1, y1: 指定切片的左上角坐标
            x2, y2: 指定切片的右下角坐标

        Returns:
            detections: 检测结果列表，坐标为原图的绝对坐标
        """
        # 1. 提取切片 (ROI)
        # 确保坐标不越界
        h, w = image.shape[:2]
        # print(f"image shape: {h}, {w}")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = image[y1:y2, x1:x2]

        # 记录原始切片的尺寸和偏移量，用于后续映射
        original_roi_h, original_roi_w = roi.shape[:2]
        offset_x, offset_y = x1, y1

        # 2. Resize 到 640x640
        # 注意：这里需要记录缩放比例，因为 resize 会改变长宽比，检测框需要反向缩放
        target_size = 640
        resized_roi = cv2.resize(roi, (target_size, target_size))

        # 计算缩放比例 (用于将 640x640 的检测框映射回原始 ROI 大小)
        scale_x = original_roi_w / target_size
        scale_y = original_roi_h / target_size

        # 3. 推理
        # 将图像格式调整为模型输入格式 (B, C, H, W)
        # 假设模型输入是 RGB，且需要归一化 (代码中 RKNN 通常在模型内处理，这里保持和原代码一致)
        input_data = np.expand_dims(resized_roi, axis=0).astype(np.float32)
        # 注意：如果 RKNN 模型要求输入是归一化的 (0-1)，则需要除以 255.0
        # input_data = input_data / 255.0
        # print("AAA",input_data.shape)
        # 执行推理
        # self.model = self.rknn_lite.load_rknn(self.model_path)
        # self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        # self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        results = self.rknn_lite.inference(inputs=[input_data])
        # print(len(results))
        # print(results[0].shape)
        # 4. 处理输出并映射回原图
        # results[0] 形状通常是 (1, 7, 8400) 或类似结构
        # 这里复用原代码中的逻辑，但需要调整映射方式

        # --- 数据预处理 ---
        predictions = results[0][0].T  # 转置为 (8400, 7)
        num_classes = predictions.shape[1] - 4
        class_scores = predictions[:, 4:]
        max_class_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # --- 置信度过滤 ---
        conf_mask = max_class_scores >= self.conf_thres
        filtered_boxes = predictions[conf_mask]
        filtered_scores = max_class_scores[conf_mask]
        filtered_class_ids = class_ids[conf_mask]

        if len(filtered_boxes) == 0:
            return []

        # --- 坐标转换 (从 640x640 归一化坐标 -> 原始 ROI 坐标 -> 原图绝对坐标) ---
        boxes = filtered_boxes[:, :4]  # cx, cy, w, h (在 640x640 图像中的归一化值 0~1)

        # 将归一化坐标转换为 640x640 图像的像素坐标
        boxes_pixel_640 = boxes * target_size

        # 将 640x640 的像素坐标转换回原始 ROI 的像素坐标 (利用缩放比例)
        cx_roi = boxes_pixel_640[:, 0] / 640 * scale_x
        cy_roi = boxes_pixel_640[:, 1] / 640 * scale_y
        w_roi = boxes_pixel_640[:, 2] / 640 * scale_x
        h_roi = boxes_pixel_640[:, 3] / 640 * scale_y

        # 将原始 ROI 坐标转换为原图的绝对坐标
        cx_abs = cx_roi + offset_x
        cy_abs = cy_roi + offset_y
        w_abs = w_roi
        h_abs = h_roi

        # 转换为 (x1, y1, x2, y2) 格式
        x1_abs = cx_abs - (w_abs / 2)
        y1_abs = cy_abs - (h_abs / 2)
        x2_abs = cx_abs + (w_abs / 2)
        y2_abs = cy_abs + (h_abs / 2)

        # --- 非极大值抑制 (NMS) ---
        # 在映射后的原图坐标上进行 NMS
        detections_for_nms = np.stack([x1_abs, y1_abs, x2_abs, y2_abs], axis=1)
        keep_indices = cv2.dnn.NMSBoxes(detections_for_nms.tolist(),
                                        filtered_scores.tolist(),
                                        self.conf_thres,
                                        self.iou_thres)

        # --- 组装最终结果 ---
        final_detections = []
        for i in keep_indices:
            idx = i if isinstance(i, np.integer) else i[0]
            final_detections.append([
                float(x1_abs[idx]), float(y1_abs[idx]),
                float(x2_abs[idx]), float(y2_abs[idx]),
                float(filtered_scores[idx]),
                int(filtered_class_ids[idx])
            ])

        return final_detections


if __name__ == '__main__':
    processor = YOLOTileProcessor(
        model_path='./model_yolo.rknn',
        device='cuda',
        tile_size=4000, overlap=0, batch_size=1
    )
    img = cv2.imread('./result_001203.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Ax = [img, img, img]
    dets = processor.process_frames_batch(Ax)[1][0]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(dets)
    print(f"检测到 {len(dets)} 个目标")
    bags, bottles = processor.crop_by_class(img, dets)
    print(f"药袋: {len(bags)} 个, 药瓶: {len(bottles)} 个")
    result = processor.draw_detections(img, dets, class_names=[0, 1, 2])
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output.jpg', result)

    # # 读取图像
    # img = cv2.imread('./result_001203.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # # 定义你要检测的切片位置 (左上 x1, y1; 右下 x2, y2)
    # # 请替换为你实际需要的坐标
    # target_x1, target_y1 = 300,  300
    # target_x2, target_y2 = 2900, 2160

    # # 执行特定切片检测
    # dets = processor.process_specific_slice(img, target_x1, target_y1, target_x2, target_y2)

    # print(f"在指定切片中检测到 {len(dets)} 个目标")
    # print(dets)
    # # 绘制结果 (需要确保 draw_detections 支持传入原图)
    # result = processor.draw_detections(img, dets, class_names=[0,1,2])
    # cv2.imwrite('output_specific_slice.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
