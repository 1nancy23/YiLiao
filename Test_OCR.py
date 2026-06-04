import platform
# import paddle
import os
import numpy as np
import cv2
import datetime
from rknnlite.api import RKNNLite
# from paddleocr import PaddleOCR
def create_dummy_images(image_dir, num_samples=100, img_size=1024):
    batch_files = []
    for i in range(num_samples):
        # 创建一个batch的数据: (12, 1024, 1024, 3)
        batch_data = np.random.randint(0, 255, (4,img_size, img_size,3), dtype=np.uint8)
        os.makedirs(image_dir, exist_ok=True)
        # 保存为.npy文件
        batch_file = os.path.join(image_dir, f'batch_{i:04d}.npy')
        np.save(batch_file, batch_data)
        batch_files.append(batch_file)
        print(f'✅ 创建校准batch: {batch_file}')

    # 创建数据集文件
    dataset_file = 'batch_dataset.txt'
    with open(dataset_file, 'w') as f:
        for batch_file in batch_files:
            f.write(f'{batch_file}\n')

    print(f'✅ 数据集文件创建完成: {dataset_file}')
    return dataset_file
# ocr = PaddleOCR(
#         use_angle_cls=True,
#         # det_algorithm="DB",  # 使用DB算法处理小文字
#         # det_max_side_len=960,                    # 增大最大边长以捕捉小文字
#         # det_score_mode="slow",  # 提高检测精度
#         det_db_thresh=0.5,     # 降低检测阈值（针对模糊文字）
#         det_db_box_thresh=0.5, # 降低检测框阈值
#         lang='ch',
#         rec_algorithm="CRNN",
#         rec_batch_num=4,       # 减小批处理数量提高小图精度
#         # rec_image_shape="3, 32, 320",  # 针对小文字调整输入尺寸
#         use_gpu=False,
#         det=True,
#         rec=True,
#         rec_score_thresh=0.3,
#         gpu_mem=1024,
#         use_tensorrt=True,
#         use_fp16=True,
#         use_gpu_threads=4,
#         show_log=False
#     )
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
    class_scores = predictions[:, 4:] # 取出所有类别的分数 (8400, num_classes)
    
    # 找到每个框分数最高的类别及其索引
    max_class_scores = np.max(class_scores, axis=1) # (8400,)
    class_ids = np.argmax(class_scores, axis=1)      # (8400,)
    
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
    x_c = boxes[:, 0] 
    y_c = boxes[:, 1]
    w = boxes[:, 2] 
    h = boxes[:, 3]
    
    # 计算左上角和右下角坐标
    x1 = x_c - (w / 2)
    y1 = y_c - (h / 2)
    x2 = x_c + (w / 2)
    y2 = y_c + (h / 2)
    
    # 合并为 NMS 需要的格式
    detection_boxes = np.stack([x1, y1, x2, y2], axis=1) # shape: (N, 4)
    
    # --- 5. 非极大值抑制 (NMS) ---
    # 这一步是为了去掉重叠的框，只保留分数最高的那个
    indices = cv2.dnn.NMSBoxes(detection_boxes.tolist(), filtered_scores.tolist(), conf_threshold, iou_threshold)
    
    # --- 6. 绘制结果 ---
    result_image = original_image.copy()
    
    # 颜色定义 (BGR格式)
    colors = {
        0: (0, 255, 0),    # 绿色
        1: (255, 0, 0),    # 蓝色
        2: (0, 0, 255),    # 红色
        # 可以根据类别数量扩展颜色
    }
    
    # 遍历保留下来的框
    for i in indices:
        # OpenCV 4.x 与 5.x 的 NMSBoxes 返回值类型不同，这里做兼容处理
        idx = i if isinstance(i, np.integer) else i[0]
        
        box = detection_boxes[idx]
        score = filtered_scores[idx]
        class_id = filtered_class_ids[idx]
        
        x1, y1, x2, y2 = map(int, box) # 转为整数
        
        # 获取颜色
        color = colors.get(class_id, (255, 255, 255)) # 默认白色
        
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

import cv2

label_list = [0, 90, 180, 270]
def reverse_rotate_with_label(img, pred, label_list):
    angle = label_list[int( pred )]

    # 反向旋转角度
    reverse_angle = (360 - angle) % 360

    if reverse_angle == 0:
        return img
    elif reverse_angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif reverse_angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif reverse_angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(f"不支持的角度: {reverse_angle}")

def _clamp_padding(size, padding_ratio, min_padding, max_padding):
    padding = max(float(min_padding), float(size) * float(padding_ratio))
    if max_padding is not None:
        padding = min(float(max_padding), padding)
    return padding


def expand_box_points(points, image_width, image_height, padding_ratio=0.10,
                      min_padding=4, max_padding=18):
    """Expand a rotated rectangle slightly while keeping it inside the image."""
    points = np.asarray(points, dtype=np.float32)
    rect = cv2.minAreaRect(points)
    (center_x, center_y), (box_w, box_h), angle = rect

    if box_w <= 0 or box_h <= 0:
        return points.astype(np.int32)

    pad_w = _clamp_padding(box_w, padding_ratio, min_padding, max_padding)
    pad_h = _clamp_padding(box_h, padding_ratio, min_padding, max_padding)
    expanded_rect = (
        (center_x, center_y),
        (box_w + pad_w * 2, box_h + pad_h * 2),
        angle
    )
    expanded_points = cv2.boxPoints(expanded_rect)
    expanded_points[:, 0] = np.clip(expanded_points[:, 0], 0, image_width - 1)
    expanded_points[:, 1] = np.clip(expanded_points[:, 1], 0, image_height - 1)
    return expanded_points.astype(np.int32)


def sort_text_regions_reading_order(text_regions, text_boxes):
    """Sort text regions from top to bottom, then left to right within each row."""
    if not text_regions or not text_boxes:
        return text_regions, text_boxes

    items = []
    for region, box in zip(text_regions, text_boxes):
        box = np.asarray(box, dtype=np.int32)
        x_min, y_min = np.min(box, axis=0)
        x_max, y_max = np.max(box, axis=0)
        items.append({
            'region': region,
            'box': box,
            'x_min': int(x_min),
            'center_y': (float(y_min) + float(y_max)) / 2.0,
            'height': max(1, int(y_max) - int(y_min))
        })

    median_height = float(np.median([item['height'] for item in items]))
    row_threshold = max(10.0, median_height * 0.6)
    rows = []

    for item in sorted(items, key=lambda value: value['center_y']):
        for row in rows:
            if abs(item['center_y'] - row['center_y']) <= row_threshold:
                row['items'].append(item)
                row['center_y'] = float(np.mean([v['center_y'] for v in row['items']]))
                break
        else:
            rows.append({'center_y': item['center_y'], 'items': [item]})

    sorted_regions = []
    sorted_boxes = []
    for row in sorted(rows, key=lambda value: value['center_y']):
        for item in sorted(row['items'], key=lambda value: value['x_min']):
            sorted_regions.append(item['region'])
            sorted_boxes.append(item['box'])

    return sorted_regions, sorted_boxes


def _points_to_rect(points):
    points = np.asarray(points, dtype=np.int32)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def _rect_to_points(rect, image_width, image_height):
    x1, y1, x2, y2 = rect
    x1 = max(0, min(image_width - 1, int(x1)))
    y1 = max(0, min(image_height - 1, int(y1)))
    x2 = max(0, min(image_width - 1, int(x2)))
    y2 = max(0, min(image_height - 1, int(y2)))
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)


def filter_edge_noise_boxes(text_boxes, image_width, image_height,
                            margin_ratio=0.035, min_margin=8):
    """Drop partial text boxes from ROI borders before line merging."""
    if not text_boxes:
        return []

    margin_x = max(int(image_width * margin_ratio), int(min_margin))
    margin_y = max(int(image_height * margin_ratio), int(min_margin))
    filtered = []
    for box in text_boxes:
        x1, y1, x2, y2 = _points_to_rect(box)
        touches_vertical_edge = x1 <= margin_x or x2 >= image_width - 1 - margin_x
        touches_horizontal_edge = y1 <= margin_y or y2 >= image_height - 1 - margin_y
        if touches_vertical_edge or touches_horizontal_edge:
            continue
        filtered.append(box)
    return filtered


def merge_same_line_boxes(text_boxes, image_width, image_height,
                          max_gap_ratio=0.85, min_gap=18,
                          center_y_ratio=0.65, overlap_ratio=0.15,
                          crop_padding_ratio=0.08, min_crop_padding=6,
                          max_crop_padding=28):
    """Merge fragmented contour boxes that belong to the same text line."""
    if not text_boxes:
        return []

    items = []
    for box in text_boxes:
        rect = _points_to_rect(box)
        x1, y1, x2, y2 = rect
        items.append({
            'rect': rect,
            'cx': (x1 + x2) / 2.0,
            'cy': (y1 + y2) / 2.0,
            'w': max(1, x2 - x1),
            'h': max(1, y2 - y1)
        })

    merged = True
    while merged:
        merged = False
        items = sorted(items, key=lambda value: (value['cy'], value['rect'][0]))
        used = [False] * len(items)
        next_items = []

        for i, item in enumerate(items):
            if used[i]:
                continue
            rect = item['rect'][:]
            cy = item['cy']
            h = item['h']
            used[i] = True

            changed = True
            while changed:
                changed = False
                x1, y1, x2, y2 = rect
                rect_h = max(1, y2 - y1)
                rect_cy = (y1 + y2) / 2.0
                for j, other in enumerate(items):
                    if used[j]:
                        continue
                    ox1, oy1, ox2, oy2 = other['rect']
                    other_h = max(1, oy2 - oy1)
                    vertical_overlap = max(0, min(y2, oy2) - max(y1, oy1))
                    overlap = vertical_overlap / float(min(rect_h, other_h))
                    center_close = abs(rect_cy - other['cy']) <= max(rect_h, other_h) * center_y_ratio
                    horizontal_gap = max(0, max(ox1 - x2, x1 - ox2))
                    gap_limit = max(min_gap, max(rect_h, other_h) * max_gap_ratio)

                    if (overlap >= overlap_ratio or center_close) and horizontal_gap <= gap_limit:
                        rect = [
                            min(x1, ox1),
                            min(y1, oy1),
                            max(x2, ox2),
                            max(y2, oy2)
                        ]
                        used[j] = True
                        changed = True
                        merged = True
                        break

            x1, y1, x2, y2 = rect
            rect_w = max(1, x2 - x1)
            rect_h = max(1, y2 - y1)
            pad_x = _clamp_padding(rect_w, crop_padding_ratio, min_crop_padding, max_crop_padding)
            pad_y = _clamp_padding(rect_h, crop_padding_ratio * 1.5, min_crop_padding, max_crop_padding)
            rect = [
                max(0, int(x1 - pad_x)),
                max(0, int(y1 - pad_y)),
                min(image_width - 1, int(x2 + pad_x)),
                min(image_height - 1, int(y2 + pad_y))
            ]
            next_items.append({
                'rect': rect,
                'cx': (rect[0] + rect[2]) / 2.0,
                'cy': (rect[1] + rect[3]) / 2.0,
                'w': max(1, rect[2] - rect[0]),
                'h': max(1, rect[3] - rect[1])
            })

        items = next_items

    return [_rect_to_points(item['rect'], image_width, image_height) for item in items]


def extract_text_regions(original_image, output, scale_factor=1, threshold=0.3,
                         box_padding_ratio=0.10, min_box_padding=4,
                         max_box_padding=18, merge_boxes=True,
                         merge_kwargs=None, filter_edge_boxes=False,
                         edge_margin_ratio=0.035, edge_min_margin=8):
    """
    从原图中切分出所有检测到的文本区域
    
    参数:
    original_image: 原始输入图像 (H, W, C)
    output: 模型输出，形状为 [1, 1, H_out, W_out]
    scale_factor: 输出特征图到原图的缩放因子，默认为4
    threshold: 二值化阈值，默认为0.3
    
    返回:
    text_regions: 包含所有切分文本区域的列表
    text_boxes: 包含所有检测框坐标的列表（原图坐标系）
    """
    
    # 1. 获取概率图
    prob_map = cv2.resize(output,(original_image.shape[1],original_image.shape[0]))  # 形状: [H_out, W_out]
    
    # 2. 阈值分割
    binary_map = (prob_map > threshold).astype(np.uint8)
    
    # 3. 轮廓提取
    contours, _ = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. 生成文本检测框（在特征图坐标系）
    text_boxes_feature = []
    for contour in contours:
        if len(contour) < 3:  # 至少需要3个点才能形成多边形
            continue
        box = cv2.minAreaRect(contour)
        text_boxes_feature.append(box)
    
    # 5. 坐标转换：从特征图坐标系转换到原图坐标系
    text_boxes_original = []
    for box in text_boxes_feature:
        # 获取旋转矩形的4个顶点
        points = cv2.boxPoints(box)
        # 缩放回原图尺寸
        points_original = points * scale_factor
        text_boxes_original.append(points_original)
    
    # 6. 从原图切分文本区域
    text_regions = []
    height, width = original_image.shape[:2]
    candidate_text_boxes = []
    
    for i, points in enumerate(text_boxes_original):
        # 获取4个顶点的坐标
        points = np.array(points, dtype=np.float32)
        
        # 确保所有点都在图像范围内
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        points = expand_box_points(
            points,
            width,
            height,
            padding_ratio=box_padding_ratio,
            min_padding=min_box_padding,
            max_padding=max_box_padding
        )
        candidate_text_boxes.append(points)
        continue
        
        # 计算文本区域的边界
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        
        # 扩大边界以包含整个文本（可选）
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(width, int(x_max) + 1)
        y_max = min(height, int(y_max) + 1)
        
        # 如果是旋转文本，需要更复杂的处理
        if is_rotated_text(points):
            # 方法1：使用仿射变换校正旋转
            rotated_region = extract_rotated_text(original_image, points)
            if rotated_region is not None:
                text_regions.append(rotated_region)
                expanded_text_boxes.append(points)
        else:
            # 方法2：直接裁剪矩形区域
            text_region = original_image[y_min:y_max, x_min:x_max].copy()
            if text_region.size > 0:  # 确保区域有效
                text_regions.append(text_region)
                expanded_text_boxes.append(points)
    
    valid_text_boxes = []
    if filter_edge_boxes:
        before_filter = len(candidate_text_boxes)
        candidate_text_boxes = filter_edge_noise_boxes(
            candidate_text_boxes,
            width,
            height,
            margin_ratio=edge_margin_ratio,
            min_margin=edge_min_margin
        )
        print(f"edge filter: boxes {before_filter} -> {len(candidate_text_boxes)}")

    if merge_boxes:
        merge_kwargs = merge_kwargs or {}
        merged_text_boxes = merge_same_line_boxes(
            candidate_text_boxes,
            width,
            height,
            **merge_kwargs
        )
    else:
        merged_text_boxes = candidate_text_boxes

    for points in merged_text_boxes:
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(width, int(x_max) + 1)
        y_max = min(height, int(y_max) + 1)

        if is_rotated_text(points):
            rotated_region = extract_rotated_text(original_image, points)
            if rotated_region is not None:
                text_regions.append(rotated_region)
                valid_text_boxes.append(points)
        else:
            text_region = original_image[y_min:y_max, x_min:x_max].copy()
            if text_region.size > 0:
                text_regions.append(text_region)
                valid_text_boxes.append(points)

    return sort_text_regions_reading_order(text_regions, valid_text_boxes)

def is_rotated_text(points):
    """判断文本是否旋转"""
    # 计算各边的长度
    edges = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        edge_length = np.sqrt(np.sum((p1 - p2) ** 2))
        edges.append(edge_length)
    
    # 检查相邻边是否垂直（通过点积）
    vectors = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        vectors.append(p2 - p1)
    
    # 检查角度是否接近90度
    angles = []
    for i in range(4):
        v1 = vectors[i]
        v2 = vectors[(i + 1) % 4]
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product > 0:
            cos_angle = dot_product / norm_product
            angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
            angles.append(angle)
    
    # 如果角度偏离90度较多，认为是旋转文本
    for angle in angles:
        if abs(angle - 90) > 15:  # 允许15度的误差
            return True
    
    return False

def extract_rotated_text(image, points):
    """提取旋转文本区域并校正"""
    # 获取4个顶点
    pts = points.astype(np.float32)
    
    # 计算文本区域的宽度和高度
    width_top = np.linalg.norm(pts[0] - pts[1])
    width_bottom = np.linalg.norm(pts[2] - pts[3])
    width = max(int(width_top), int(width_bottom))
    
    height_left = np.linalg.norm(pts[0] - pts[3])
    height_right = np.linalg.norm(pts[1] - pts[2])
    height = max(int(height_left), int(height_right))
    
    if width <= 0 or height <= 0:
        return None
    
    # 目标点（校正后的矩形）
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    
    # 应用透视变换
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped

def load_ctc_character_list(char_dict_path):
    """Load PaddleOCR-style character dictionary and add the CTC blank token."""
    if not os.path.exists(char_dict_path):
        candidates = [
            os.path.basename(char_dict_path),
            os.path.join('./Qiege_test', os.path.basename(char_dict_path)),
            './OCRS.txt',
            './Qiege_test/OCRS.txt'
        ]
        for fallback_path in candidates:
            if os.path.exists(fallback_path):
                char_dict_path = fallback_path
                break
        else:
            raise FileNotFoundError(f'Character dictionary not found: {char_dict_path}')

    character_list = []
    with open(char_dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            ch = line.rstrip('\r\n').strip()
            if not ch:
                continue
            if len(ch) >= 2 and ch[0] == ch[-1] and ch[0] in ("'", '"'):
                ch = ch[1:-1]
            character_list.append(ch)

    return ['blank'] + character_list


def ctc_greedy_decode_one(sequence_output, character_list, blank_idx=0):
    """Greedy CTC decode for one recognition output: [time_steps, num_classes]."""
    pred_indices = np.argmax(sequence_output, axis=-1)
    decoded_text = []
    prev_idx = -1

    for idx in pred_indices:
        idx = int(idx)
        if idx == blank_idx:
            prev_idx = -1
            continue
        if idx != prev_idx and idx < len(character_list):
            decoded_text.append(character_list[idx])
        prev_idx = idx

    return ''.join(decoded_text), pred_indices


def preprocess_rec_image(text_region, rec_input_size=(320, 48)):
    """Resize one cropped text region without distortion and pad with zeros."""
    rec_w, rec_h = rec_input_size
    if text_region.ndim == 2:
        text_region = cv2.cvtColor(text_region, cv2.COLOR_GRAY2BGR)
    elif text_region.shape[2] == 4:
        text_region = cv2.cvtColor(text_region, cv2.COLOR_BGRA2BGR)

    height, width = text_region.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((rec_h, rec_w, 3), dtype=np.uint8)

    scale = min(rec_w / float(width), rec_h / float(height))
    resized_w = max(1, min(rec_w, int(round(width * scale))))
    resized_h = max(1, min(rec_h, int(round(height * scale))))
    resized = cv2.resize(text_region, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    padded = np.zeros((rec_h, rec_w, 3), dtype=np.uint8)
    padded[:resized_h, :resized_w, :] = resized
    return padded.astype(np.uint8)


def normalize_rec_output(output, batch_size):
    """Normalize common RKNN rec output layouts to [batch, time_steps, classes]."""
    output = np.asarray(output)
    if output.ndim == 4 and output.shape[0] == 1 and output.shape[1] == batch_size:
        output = output[0]
    if output.ndim == 4 and output.shape[1] == 1:
        output = output[:, 0]
    if output.ndim == 2:
        output = np.expand_dims(output, axis=0)
    if output.ndim != 3:
        raise ValueError(f'Unsupported recognition output shape: {output.shape}')
    return output


def _to_bgr_image(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def has_percent_like_mark(text_region):
    """Detect a small percent-like mark on the right side of a cropped text line."""
    image = _to_bgr_image(text_region)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if height <= 0 or width <= 0:
        return False

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_pixels = gray[mask > 0]
    bg_pixels = gray[mask == 0]
    if fg_pixels.size > 0 and bg_pixels.size > 0 and np.mean(fg_pixels) < np.mean(bg_pixels):
        mask = cv2.bitwise_not(mask)

    right_start = int(width * 0.58)
    right_mask = mask[:, right_start:]
    if right_mask.size == 0:
        return False

    contours, _ = cv2.findContours(right_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_parts = 0
    full_height_parts = 0
    xs = []
    ys = []

    min_area = max(3, int(height * width * 0.0004))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        xs.extend([x, x + w])
        ys.extend([y, y + h])
        if h >= height * 0.62 and w >= height * 0.18:
            full_height_parts += 1
        if height * 0.05 <= h <= height * 0.55 and w <= height * 0.45:
            small_parts += 1

    if not xs or full_height_parts > 0:
        return False

    mark_height = max(ys) - min(ys)
    mark_width = max(xs) - min(xs)
    compact_mark = mark_height <= height * 0.72 and mark_width <= width * 0.34
    return small_parts >= 2 or (small_parts >= 1 and compact_mark)


def correct_percent_confusion(text, text_region):
    """Correct common OCR confusion where a trailing percent sign is decoded as 0."""
    if len(text) >= 2 and text.isdigit() and text.endswith('0') and has_percent_like_mark(text_region):
        return text[:-1] + '%'
    return text


def recognize_text_regions(rec_rknn, text_regions, character_list,
                           batch_size=16, rec_input_size=(320, 48)):
    """Run rec RKNN by fixed-size batches and CTC-decode valid text regions."""
    results = []
    if not text_regions:
        return results

    blank_image = np.zeros((rec_input_size[1], rec_input_size[0], 3), dtype=np.uint8)

    for start in range(0, len(text_regions), batch_size):
        batch_regions = text_regions[start:start + batch_size]
        valid_count = len(batch_regions)
        batch_images = [preprocess_rec_image(region, rec_input_size) for region in batch_regions]

        while len(batch_images) < batch_size:
            batch_images.append(blank_image.copy())

        batch_input = np.stack(batch_images, axis=0)
        rec_outputs = rec_rknn.inference(inputs=[batch_input])
        rec_output = normalize_rec_output(rec_outputs[0], batch_size)

        for offset in range(valid_count):
            raw_text, pred_indices = ctc_greedy_decode_one(
                rec_output[offset],
                character_list
            )
            text = correct_percent_confusion(raw_text, batch_regions[offset])
            results.append({
                'index': start + offset,
                'text': text,
                'raw_text': raw_text,
                'pred_indices': pred_indices
            })

    return results


def save_ocr_results(output_dir, original_image, text_boxes, rec_results):
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, 'ocr_results.txt')
    debug_path = os.path.join(output_dir, 'ocr_debug.txt')
    vis_path = os.path.join(output_dir, 'ocr_result.jpg')

    vis_image = original_image.copy()
    with open(txt_path, 'w', encoding='utf-8') as f:
        for result in rec_results:
            idx = result['index']
            text = result['text']
            box = np.asarray(text_boxes[idx], dtype=np.int32)
            box_flat = box.reshape(-1).tolist()
            f.write(f'{idx}\t{text}\t{box_flat}\n')

            cv2.polylines(vis_image, [box], True, (0, 255, 0), 2)
            x_min, y_min = np.min(box, axis=0)
            cv2.putText(vis_image, str(idx), (int(x_min), max(0, int(y_min) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    with open(debug_path, 'w', encoding='utf-8') as f:
        for result in rec_results:
            pred_indices = [int(idx) for idx in result['pred_indices']]
            raw_text = result.get('raw_text', result['text'])
            f.write(f"{result['index']}\t{result['text']}\traw={raw_text}\t{pred_indices}\n")

    cv2.imwrite(vis_path, vis_image)
    return txt_path, vis_path, debug_path


def save_det_diagnostics(output_dir, original_image, det_map, thresholds):
    os.makedirs(output_dir, exist_ok=True)
    prob_map = cv2.resize(det_map, (original_image.shape[1], original_image.shape[0]))
    prob_min = float(np.min(prob_map))
    prob_max = float(np.max(prob_map))
    percentiles = np.percentile(prob_map, [50, 75, 90, 95, 98, 99])
    print(
        "DET prob stats: "
        f"min={prob_min:.4f}, max={prob_max:.4f}, "
        f"p50={percentiles[0]:.4f}, p75={percentiles[1]:.4f}, "
        f"p90={percentiles[2]:.4f}, p95={percentiles[3]:.4f}, "
        f"p98={percentiles[4]:.4f}, p99={percentiles[5]:.4f}"
    )

    norm = cv2.normalize(prob_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "det_prob_heatmap.jpg"), cv2.applyColorMap(norm, cv2.COLORMAP_JET))

    image_area = float(max(1, original_image.shape[0] * original_image.shape[1]))
    for threshold in thresholds:
        binary = (prob_map > threshold).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(output_dir, f"det_binary_t{threshold:.2f}.jpg"), binary)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for contour in contours:
            if len(contour) < 3:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area <= 0:
                continue
            rects.append((x, y, w, h, area))
        rects.sort(key=lambda item: item[4], reverse=True)
        largest_ratio = (rects[0][4] / image_area) if rects else 0.0
        print(
            f"DET threshold={threshold:.2f}: contours={len(rects)}, "
            f"largest_area_ratio={largest_ratio:.3f}, "
            f"largest_rect={rects[0][:4] if rects else None}"
        )


def save_box_visualization(path, image, boxes, label):
    vis = image.copy()
    for idx, box in enumerate(boxes):
        pts = np.asarray(box, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        x_min, y_min = np.min(pts, axis=0)
        cv2.putText(vis, str(idx), (int(x_min), max(0, int(y_min) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(vis, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(path, vis)


def legacy_det_only_main():
    # 定义常量
    import gc

    gc.collect()  # 手动触发垃圾回收
    QUANTIZE_ON = True  # 设为True进行量化，False则不量化
    IMG_DIR = './dummy_images'
    DATASET = 'image_list.txt'
    RKNN_MODEL = './Det.rknn'
    IMG_PATH = 'frame_000215.png'  # 测试图像路径
    # IMG_SIZE = 640  # 模型输入尺寸
    
    # # 创建模拟图像
    # print('--> Creating dummy images')
    # create_dummy_images(IMG_DIR, num_samples=500, img_size=IMG_SIZE)
    # print('done')
    
    # # 创建数据集文件
    # print('--> Creating dataset file')
    # create_dataset_file(IMG_DIR, DATASET, num_samples=500)
    # print('done')
    
    # 创建RKNN对象
    rknn = RKNNLite()

    
    rknn.load_rknn(RKNN_MODEL)
    
    # 初始化运行时环境
    print('--> Init runtime environment')
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    # print('done')

    # 设置输出，用于模型推理
    import cv2
    import numpy as np
    
    img = cv2.imread('./corrected_4.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # text = ocr.ocr(img,det=False)
    # print("识别到的文本:", text)
    if img is not None:
        # img_ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # input_nchw = np.load('./analysis_input.npy')
        # print(input_nchw.shape)
        input_nchw = cv2.resize(img, (448, 448))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img.transpose(2, 0, 1)  
        batch_data = np.zeros((8,) + input_nchw.shape, dtype=input_nchw.dtype)
        batch_data[0] = input_nchw
        # 进行推理
        print('--> Running model')
        
        # while True:
        outputs = rknn.inference(inputs=[batch_data])
        outputs = outputs[0][0]
        #     print("总长度",len(outputs))
        #     print("结果:",np.argmax(outputs[0], axis=1))
        #     del outputs
        
        print(outputs)
        text_regions, text_boxes = extract_text_regions(
                img, 
                outputs[0], 
                scale_factor=1, 
                threshold=0.3,
                box_padding_ratio=0.1,
                min_box_padding=4,
                max_box_padding=36
            )
        all_text_images = []
        for i, text_region in enumerate(text_regions):
            if text_region is not None and text_region.size > 0:
                all_text_images.append(text_region)
                
                # 可选：保存到文件
                cv2.imwrite(f'./Qiege_test/text_region_{i}.jpg', text_region)
        
        print(f"成功切分出 {len(all_text_images)} 个文本区域")
        
        
        # pred_indices = np.argmax(outputs, axis=1)  # 形状 (20,)
        # print(pred_indices)
        # img = cv2.imread("test.jpg")

        # pred = pred_indices[0]
        # corrected = reverse_rotate_with_label(img, pred, label_list)

        # cv2.imwrite("corrected_4.jpg", corrected)
        # print(np.max(outputs[0][4:,:]))
        # print('done')
        # if outputs[0].dtype == np.int8:
        # print('Converting int8 output to float32...')
        # outputs = outputs.astype(np.float32)
        # result_img = postprocess_rknn_output(
        # output=outputs, 
        # original_image=img,  # 注意：这里使用原始未缩放的图像
        # conf_threshold=0.25
        # )
        # 如果需要，可以进行反量化（根据你的量化参数调整）
        # output = output * scale_factor  # scale_factor 需要根据量化参数确定
        # cv2.imwrite('./result_frame_000215.png', result_img)
        # print('结果已保存到 ./result_000215.png')
    # 释放资源
    rknn.release()


if __name__ == '__main__':
    import gc

    gc.collect()
    DET_RKNN_MODEL = './src/identification/Det_bs1.rknn'
    REC_RKNN_MODEL = './model_ocr_0526.rknn'
    CHAR_DICT_PATH = './OCRS.txt'
    OUTPUT_DIR = './Qiege_test'
    IMAGE_PATH = './artifacts/predet_shuye_roi_then_cls_exact2/shuye_1_blue_before_det.jpg'
    DET_BATCH_SIZE = 1
    REC_BATCH_SIZE = 16
    REC_INPUT_SIZE = (320, 48)
    DET_THRESHOLD = 0.30
    MERGE_TEXT_BOXES = True
    FILTER_EDGE_BOXES = True
    BOX_PADDING_RATIO = 0.04
    MIN_BOX_PADDING = 1
    MAX_BOX_PADDING = 4
    MERGE_KWARGS = {
        "center_y_ratio": 0.30,
        "overlap_ratio": 0.55,
        "max_gap_ratio": 0.85,
        "min_gap": 18,
        "crop_padding_ratio": 0.08,
        "min_crop_padding": 6,
        "max_crop_padding": 28,
    }

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f'Image not found: {IMAGE_PATH}')

    det_rknn = RKNNLite()
    rec_rknn = RKNNLite()

    try:
        print('--> Load det model')
        ret = det_rknn.load_rknn(DET_RKNN_MODEL)
        if ret != 0:
            raise RuntimeError('Load det model failed!')

        print('--> Init det runtime')
        ret = det_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            raise RuntimeError('Init det runtime failed!')

        print('--> Run det model')
        det_input = cv2.resize(img, (448, 448))
        det_batch = np.zeros((DET_BATCH_SIZE,) + det_input.shape, dtype=det_input.dtype)
        det_batch[0] = det_input
        det_outputs = det_rknn.inference(inputs=[det_batch])
        det_output = det_outputs[0][0]

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        det_map = det_output[0]
        diagnostic_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
        save_det_diagnostics(OUTPUT_DIR, img, det_map, diagnostic_thresholds)

        for threshold in diagnostic_thresholds:
            sweep_regions, sweep_boxes = extract_text_regions(
                img,
                det_map,
                scale_factor=1,
                threshold=threshold,
                box_padding_ratio=0.04,
                min_box_padding=1,
                max_box_padding=4,
                merge_boxes=False
            )
            save_box_visualization(
                os.path.join(OUTPUT_DIR, f"det_boxes_no_merge_t{threshold:.2f}.jpg"),
                img,
                sweep_boxes,
                f"no merge t={threshold:.2f}, boxes={len(sweep_boxes)}"
            )
            merged_regions, merged_boxes = extract_text_regions(
                img,
                det_map,
                scale_factor=1,
                threshold=threshold,
                box_padding_ratio=0.04,
                min_box_padding=1,
                max_box_padding=4,
                merge_boxes=True,
                filter_edge_boxes=True,
                merge_kwargs=MERGE_KWARGS
            )
            save_box_visualization(
                os.path.join(OUTPUT_DIR, f"det_boxes_edge_filtered_merged_t{threshold:.2f}.jpg"),
                img,
                merged_boxes,
                f"edge+merged t={threshold:.2f}, boxes={len(merged_boxes)}"
            )
            print(
                f"postprocess sweep threshold={threshold:.2f}: "
                f"no_merge_boxes={len(sweep_boxes)}, merged_boxes={len(merged_boxes)}"
            )

        text_regions, text_boxes = extract_text_regions(
            img,
            det_map,
            scale_factor=1,
            threshold=DET_THRESHOLD,
            box_padding_ratio=BOX_PADDING_RATIO,
            min_box_padding=MIN_BOX_PADDING,
            max_box_padding=MAX_BOX_PADDING,
            merge_boxes=MERGE_TEXT_BOXES,
            filter_edge_boxes=FILTER_EDGE_BOXES,
            merge_kwargs=MERGE_KWARGS
        )

        valid_text_regions = []
        valid_text_boxes = []
        for i, text_region in enumerate(text_regions):
            if text_region is not None and text_region.size > 0:
                valid_text_regions.append(text_region)
                valid_text_boxes.append(text_boxes[i])
                crop_path = os.path.join(
                    OUTPUT_DIR,
                    f'text_region_{len(valid_text_regions) - 1}.jpg'
                )
                cv2.imwrite(crop_path, text_region)

        print(f'成功切分出 {len(valid_text_regions)} 个文本区域')

        if valid_text_regions:
            print('--> Load rec model')
            ret = rec_rknn.load_rknn(REC_RKNN_MODEL)
            if ret != 0:
                raise RuntimeError('Load rec model failed!')

            print('--> Init rec runtime')
            ret = rec_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
            if ret != 0:
                raise RuntimeError('Init rec runtime failed!')

            character_list = load_ctc_character_list(CHAR_DICT_PATH)

            print('--> Run rec model')
            rec_results = recognize_text_regions(
                rec_rknn,
                valid_text_regions,
                character_list,
                batch_size=REC_BATCH_SIZE,
                rec_input_size=REC_INPUT_SIZE
            )

            for result in rec_results:
                print(f"[{result['index']}] {result['text']}")

            txt_path, vis_path, debug_path = save_ocr_results(
                OUTPUT_DIR,
                img,
                valid_text_boxes,
                rec_results
            )
            print(f'识别结果已保存: {txt_path}')
            print(f'可视化结果已保存: {vis_path}')
            print(f'调试结果已保存: {debug_path}')
        else:
            print('未检测到可识别的文本区域')

    finally:
        det_rknn.release()
        rec_rknn.release()
