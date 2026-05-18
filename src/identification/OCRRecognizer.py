import cv2
import re
import time
import numpy as np
from typing import Optional, Union
from src.utils.img_utils import ImageProcessor
from rknnlite.api import RKNNLite
# import paddle
# from paddleocr import PaddleOCR
import time
import cv2
import numpy as np
import re


# ==================== 字符字典加载 ====================

def load_char_dict(dict_path='ppocr_keys_v1.txt'):
    """
    加载PaddleOCR字符字典
    字典文件可从PaddleOCR仓库获取:
    <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/ppocr/utils/ppocr_keys_v1.txt" title="https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/ppocr/utils/ppocr_keys_v1.txt" target="_blank"><img src="/images/ext/file.png" alt="" style="width: 32px; height: 32px; vertical-align: middle;"></a>

    Returns:
        chars: 字符列表, index 0 为CTC blank token
    """
    chars = []  # index 0 = CTC blank token
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            ch = line.strip('\n').strip('\r\n')
            if ch:
                chars.append(ch)
    chars.append(' ')  # 末尾添加空格字符
    chars = ['blank'] + chars
    print(f"加载字符字典, 共 {len(chars)} 个字符")
    print(chars[0],"blank token")
    return chars


# ==================== 检测模型后处理 ====================

def _box_score_fast(pred, box):
    """
    快速计算文本框区域内的平均概率值作为置信度分数
    Args:
        pred: 概率图 (H, W), 值域 [0, 1]
        box: 文本框4个顶点坐标, shape (4, 2)
    Returns:
        score: 框内平均概率值
    """
    h, w = pred.shape
    box_ = box.copy()

    xmin = max(0, int(np.floor(box_[:, 0].min())))
    xmax = min(w - 1, int(np.ceil(box_[:, 0].max())))
    ymin = max(0, int(np.floor(box_[:, 1].min())))
    ymax = min(h - 1, int(np.ceil(box_[:, 1].max())))

    if xmax <= xmin or ymax <= ymin:
        return 0.0

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box_shifted = box_.copy()
    box_shifted[:, 0] -= xmin
    box_shifted[:, 1] -= ymin
    cv2.fillPoly(mask, [box_shifted.astype(np.int32)], 1)

    region = pred[ymin:ymax + 1, xmin:xmax + 1]
    return float(cv2.mean(region, mask)[0])


def _unclip(box, unclip_ratio):
    """
    使用Vatti裁剪算法扩展多边形区域（DB算法核心步骤）
    Args:
        box: 多边形顶点坐标, shape (4, 2)
        unclip_ratio: 扩展比例
    Returns:
        expanded: 扩展后的多边形顶点坐标, 或 None
    """
    try:
        import pyclipper
        from shapely.geometry import Polygon

        poly = Polygon(box)
        if poly.area < 1:
            return None
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(
            box.astype(np.int32).tolist(),
            pyclipper.JT_ROUND,
            pyclipper.ET_CLOSEDPOLYGON
        )
        expanded = offset.Execute(distance)
        if not expanded:
            return None
        return np.array(expanded[0], dtype=np.float32)
    except ImportError:
        # 回退方案：简单几何缩放扩展
        center = box.mean(axis=0)
        expanded = center + (box - center) * (1 + unclip_ratio * 0.3)
        return expanded.astype(np.float32)


def _order_points(pts):
    """
    将4个点按顺序排列: 左上、右上、右下、左下
    Args:
        pts: 4个点坐标, shape (4, 2)
    Returns:
        ordered: 排序后的4个点坐标
    """
    sorted_pts = sorted(pts.tolist(), key=lambda p: p[0])
    left = sorted(sorted_pts[:2], key=lambda p: p[1])
    right = sorted(sorted_pts[2:], key=lambda p: p[1])
    # 顺序: 左上, 右上, 右下, 左下
    return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)


def _get_mini_boxes(contour):
    """
    获取轮廓的最小外接矩形并排序顶点
    Args:
        contour: OpenCV轮廓
    Returns:
        box: 排序后的4个顶点坐标
        min_side: 最短边长
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = _order_points(box)
    return box, min(rect[1])


def db_postprocess(pred_map, orig_h, orig_w,
                   thresh=0.3, box_thresh=0.5,
                   unclip_ratio=1.5, min_size=3, max_candidates=1000):
    """
    DB检测模型后处理：从概率图中提取文本框坐标

    Args:
        pred_map: 检测模型输出 (可能为多维数组)
        orig_h, orig_w: 原始图像的高和宽
        thresh: 二值化阈值
        box_thresh: 文本框置信度过滤阈值
        unclip_ratio: 文本框扩展比例
        min_size: 最小文本框边长(像素)
        max_candidates: 最大候选框数量

    Returns:
        boxes: 文本框列表, 每个框为 shape (4, 2) 的坐标数组
        scores: 每个框的置信度列表
    """
    # 处理不同的输出维度: [1,1,H,W] / [1,H,W,1] / [1,H,W] / [H,W]
    pred = np.squeeze(pred_map)
    if len(pred.shape) > 2:
        pred = pred[0]  # 取第一个通道
    pred = pred.astype(np.float32)

    det_h, det_w = pred.shape

    # 二值化
    binary = (pred > thresh).astype(np.uint8)

    # 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    scores = []

    num_contours = min(len(contours), max_candidates)

    for i in range(num_contours):
        contour = contours[i]
        if len(contour) < 4:
            continue

        # 获取最小外接矩形
        box, min_side = _get_mini_boxes(contour)
        if min_side < min_size:
            continue

        # 计算框内平均概率
        score = _box_score_fast(pred, box)
        if score < box_thresh:
            continue

        # 扩展文本框 (unclip)
        expanded_box = _unclip(box, unclip_ratio)
        if expanded_box is None:
            continue

        # 对扩展后的点重新计算最小外接矩形
        expanded_contour = expanded_box.reshape(-1, 1, 2).astype(np.int32)
        box, min_side = _get_mini_boxes(expanded_contour)
        if min_side < min_size + 2:
            continue

        # 映射回原图坐标
        box[:, 0] = np.clip(box[:, 0] * orig_w / det_w, 0, orig_w - 1)
        box[:, 1] = np.clip(box[:, 1] * orig_h / det_h, 0, orig_h - 1)

        boxes.append(box.astype(np.float32))
        scores.append(score)

    return boxes, scores


# ==================== 文本区域裁剪 ====================

def get_rotate_crop_image(img, points):
    """
    根据4个顶点坐标从图像中裁剪并透视矫正文本区域

    Args:
        img: 原始图像 (H, W, C)
        points: 4个顶点坐标 shape (4, 2), 顺序: 左上、右上、右下、左下

    Returns:
        warped: 矫正后的文本区域图像, 或 None
    """
    points = np.array(points, dtype=np.float32)
    img_h, img_w = img.shape[:2]

    # 计算目标宽度和高度
    width = int(max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3])
    ))
    height = int(max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2])
    ))

    if width <= 0 or height <= 0:
        return None

    # 目标矩形坐标
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # 透视变换
    M = cv2.getPerspectiveTransform(points, dst_pts)
    warped = cv2.warpPerspective(
        img, M, (width, height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC
    )

    # 如果高度远大于宽度，说明文字是竖排，旋转90度
    if float(height) >= float(width) * 1.5:
        warped = np.rot90(warped)
        
        
        
    
    # cv2.imwrite("warped.png", warped)
    # print(f"透视变换后的图像尺寸: {warped.shape}")
    return warped


# ==================== 识别模型前处理 ====================

def resize_for_rec(crop_img, target_h=48, target_w=160):
    """
    将裁剪的文本图像缩放到识别模型输入尺寸，保持宽高比并右侧补零

    Args:
        crop_img: 裁剪的文本区域图像 (h, w, c)
        target_h: 目标高度 (默认48)
        target_w: 目标宽度 (默认160)

    Returns:
        padded: 缩放并填充后的图像 (target_h, target_w, c)
    """
    h, w, c = crop_img.shape
    ratio = float(w) / float(h)
    resized_w = min(int(target_h * ratio), target_w)
    resized_w = max(1, resized_w)

    resized = cv2.resize(crop_img, (resized_w, target_h))

    # 右侧补零（黑色填充）
    padded = np.zeros((target_h, target_w, c), dtype=resized.dtype)
    padded[:, :resized_w, :] = resized
    cv2.imwrite("padded.png", padded)
    # print(f"填充后的图像尺寸: {padded.shape}")
    return padded


# ==================== 识别模型后处理(CTC解码) ====================

def ctc_decode_single(pred, characters):
    """
    对单个序列进行CTC贪心解码

    Args:
        pred: 模型输出 shape (seq_len, num_classes)
        characters: 字符字典列表, index 0 为 blank

    Returns:
        text: 解码后的文本字符串
        confidence: 置信度 (各字符概率的均值)
    """
    # 如果输出是logits（非概率），先做softmax
    if pred.max() > 1.0 or pred.min() < 0.0:
        exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
        prob = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
    else:
        prob = pred

    # 取每个时间步的最大概率索引
    
    pred_indices = np.argmax(prob, axis=1)   # (seq_len,)
    # print(pred_indices.shape,"最大概率索引")
    pred_probs = np.max(prob, axis=1)        # (seq_len,)

    # CTC解码: 去除blank(index=0)和连续重复
    char_list = []
    conf_list = []
    prev_idx = -1
    # print(pred_indices.shape, pred_probs.shape,"时间步映射")
    for t in range(len(pred_indices)):
        idx = int(pred_indices[t])
        # 跳过blank
        if idx == 0:
            prev_idx = idx
            continue
        # 跳过连续重复
        if idx == prev_idx:
            prev_idx = idx
            continue
        # 映射到字符
        if idx < len(characters):
            char_list.append(characters[idx])
            conf_list.append(float(pred_probs[t]))
        prev_idx = idx

    text = ''.join(char_list)
    confidence = float(np.mean(conf_list)) if conf_list else 0.0

    return text, confidence


def ctc_decode_batch(preds, characters):
    """
    对一批序列进行CTC解码

    Args:
        preds: 模型输出 shape (batch, seq_len, num_classes)
        characters: 字符字典列表

    Returns:
        results: [(text, confidence), ...] 列表
    """
    results = []
    for i in range(len(preds)):
        # print(preds[i].shape,"模型输出尺寸")
        text, conf = ctc_decode_single(preds[i], characters)
        results.append((text, conf))
    return results


# ==================== 完整 OCR 管线 ====================

def recognize_text_rknn(det_model, rec_model, characters,
                        corrected_img,
                        det_input_size=(448, 448),
                        rec_input_h=48,
                        rec_input_w=320,
                        rec_batch_num=16,
                        det_db_thresh=0.5,
                        det_db_box_thresh=0.2,
                        det_db_unclip_ratio=1.6,
                        rec_score_thresh=0.1,
                        line_gap_threshold=15):
    """
    使用RKNN检测+识别模型完成完整OCR流程，功能等价于PaddleOCR的ocr()方法

    Args:
        det_model: RKNN检测模型实例
        rec_model: RKNN识别模型实例
        characters: 字符字典列表 (由load_char_dict加载)
        corrected_img: 输入图像 (BGR, uint8)
        det_input_size: 检测模型输入尺寸 (W, H)
        rec_input_h: 识别模型输入高度
        rec_input_w: 识别模型输入宽度
        rec_batch_num: 识别模型批处理大小
        det_db_thresh: DB二值化阈值
        det_db_box_thresh: 文本框置信度阈值
        det_db_unclip_ratio: 文本框扩展比例
        rec_score_thresh: 识别置信度阈值
        line_gap_threshold: 同行判定的Y坐标差异阈值

    Returns:
        combined_text: 合并后的识别文本字符串（仅保留汉字）
    """
    orig_h, orig_w = corrected_img.shape[:2]
    print(f"输入图像尺寸: {orig_h}x{orig_w}")
    
    
    # ================================================================
    # 第1步: 文本检测 — 获取文本框位置
    # ================================================================
    det_input = cv2.resize(corrected_img, det_input_size)
    batch_det = np.stack([det_input], axis=0)  # shape: (1, H, W, 3)
    A=str(time.time())
    # cv2.imwrite(f"./Temp_results/crop_{A}.png", corrected_img)
    det_outputs = det_model.inference(inputs=[batch_det])
    # det_outputs[0] 为概率图, 典型shape: (1, 1, 448, 448)
    # cv2.imwrite(f"./Temp_results_mask/crop_{A}.png", det_outputs[0][0][0]*255)
    
    boxes, det_scores = db_postprocess(
        pred_map=det_outputs[0],
        orig_h=orig_h,
        orig_w=orig_w,
        thresh=det_db_thresh,
        box_thresh=det_db_box_thresh,
        unclip_ratio=det_db_unclip_ratio
    )
    print(f"检测到{len(boxes)}个文本框")
    # print(f"检测到的文本框坐标: {boxes}")
    if not boxes:
        return ""

    # ================================================================
    # 第2步: 裁剪并预处理每个文本区域
    # ================================================================
    crop_imgs = []
    valid_boxes = []
    
    for box in boxes:
        crop = get_rotate_crop_image(corrected_img, box)
        if crop is None or crop.size == 0:
            continue
        # 缩放到识别模型输入尺寸 (保持宽高比, 右侧padding)
        processed = resize_for_rec(crop, target_h=rec_input_h, target_w=rec_input_w)
        crop_imgs.append(processed)
        valid_boxes.append(box)
        
    if not crop_imgs:
        return ""

    # ================================================================
    # 第3步: 批量文本识别
    # ================================================================
    text_blocks = []

    for i in range(0, len(crop_imgs), rec_batch_num):
        batch_crops = crop_imgs[i:i + rec_batch_num]
        batch_boxes = valid_boxes[i:i + rec_batch_num]
        actual_batch_size = len(batch_crops)

        # 补齐batch到rec_batch_num（RKNN定长推理需要）
        if actual_batch_size < rec_batch_num:
            padding = [np.zeros_like(batch_crops[0])] * (rec_batch_num - actual_batch_size)
            batch_crops = batch_crops + padding

        batch_data = np.stack(batch_crops, axis=0)  # shape: (rec_batch_num, 48, 160, 3)

        rec_outputs = rec_model.inference(inputs=[batch_data])
        rec_preds = rec_outputs[0] # shape: (rec_batch_num, seq_len, num_classes)
    
        # 只解码有效图像（忽略padding部分）
        # print(characters,"blank token")
        decoded = ctc_decode_batch(rec_preds[:actual_batch_size], characters)
        print(f"识别到{actual_batch_size}个文本框的识别结果: {decoded}")
        for j, (text, confidence) in enumerate(decoded):
            # 只保留汉字
            # text = re.sub(r'[^\u3000-\u9fff]', '', text)
            # if not text:
            #     continue
            # 置信度过滤
            # if confidence < rec_score_thresh:
                # continue

            box = batch_boxes[j]
            center_y = float(np.mean(box[:, 1]))
            center_x = float(np.mean(box[:, 0]))
            text_blocks.append({
                'text': text,
                'center_y': center_y,
                'center_x': center_x
            })
    print("得到识别结果")
    if not text_blocks:
        print("未检测到文本框")
        return ""
    print("得到识别结果2")
    # ================================================================
    # 第4步: 按Y坐标排序，合并同行文本
    # ================================================================
    text_blocks.sort(key=lambda x: x['center_y'])

    lines = []
    current_line = [text_blocks[0]]
    prev_y = text_blocks[0]['center_y']

    for i in range(1, len(text_blocks)):
        y = text_blocks[i]['center_y']
        if abs(y - prev_y) > line_gap_threshold:
            # 同一行内按X坐标从左到右排序
            current_line.sort(key=lambda x: x['center_x'])
            lines.append(''.join([b['text'] for b in current_line]))
            current_line = [text_blocks[i]]
        else:
            current_line.append(text_blocks[i])
        prev_y = y

    # 处理最后一行
    current_line.sort(key=lambda x: x['center_x'])
    lines.append(''.join([b['text'] for b in current_line]))
    print(lines,"识别结果")
    return ''.join(lines)

label_list = [0, 90, 180, 270]
def reverse_rotate_with_label(img, pred):
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

# ==================== 集成到类中的示例 ====================
Image_processor=ImageProcessor()
class OCRRecognizer:
    """
    示例：如何将上述函数集成到您现有的类中
    """

    def __init__(self):
        """
        :param ocr_model: 已初始化的 PaddleOCR 实例（或任何实现了 ocr() 方法的对象）
        :param image_processor: 图像处理器实例，默认使用 ImageProcessor()
        """
        print("初始化OCR模型...")
        self.det_model = RKNNLite()
        self.det_model.load_rknn("/home/forlinx/Models/AnotherYiliao/shibie/YiLiaoShiBie/src/identification/Det_bs1.rknn")
        ret_det = self.det_model.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
        
        self.rec_model = RKNNLite()
        self.rec_model.load_rknn("Model_Test_0428_3.rknn")
        ret_rec = self.rec_model.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
        
        self.char_dict = load_char_dict('OCRS.txt')
        self.image_processor = ImageProcessor()

    def recognize(self, corrected_img, line_gap_threshold=20):
        """
        识别单张图像中的文本 — 替代原来的PaddleOCR调用

        原代码:
            result = self.rec_model.ocr(corrected_img, det=True, rec=True)
            ... (后续处理)

        现在:
            使用RKNN模型实现相同功能
        """
        print("开始识别...药瓶或药袋")
        # corrected_img=Image_processor.image_enhance(corrected_img)
        return recognize_text_rknn(
            det_model=self.det_model,
            rec_model=self.rec_model,
            characters=self.char_dict,
            corrected_img=corrected_img,
            det_input_size=(448, 448),    # 与您的检测模型输入尺寸一致
            rec_input_h=48,               # 与您的识别模型输入高度一致
            rec_input_w=320,              # 与您的识别模型输入宽度一致
            rec_batch_num=16,             # 与您的识别模型batch大小一致
            det_db_thresh=0.5,            # DB二值化阈值（可调）
            det_db_box_thresh=0.4,        # 与原始PaddleOCR配置一致
            det_db_unclip_ratio=1.2,      # 文本框扩展比例
            rec_score_thresh=0.5,         # 与原始PaddleOCR配置一致
            line_gap_threshold=line_gap_threshold
        )
def _normalize_ocr_unit_text(text):
    text = text.replace(" ", "")
    text = text.replace("％", "%")
    text = text.replace("Ｍ", "M").replace("ｍ", "m")
    text = text.replace("Ｌ", "L").replace("ｌ", "l")
    return text.lower()


def _format_number_text(number_text):
    if "." in number_text:
        number_text = number_text.rstrip("0").rstrip(".")
    return number_text


def _normalize_concentration_value(number_text):
    value_text = number_text.strip()
    if not value_text:
        return None

    try:
        value = float(value_text)
    except ValueError:
        return None

    # Known infusion-bag concentrations in this project. Restricting fallback
    # inference to these values avoids turning unrelated numbers into percents.
    allowed = [
        (0.9, "0.9%"),
        (5.0, "5%"),
        (10.0, "10%"),
    ]

    for allowed_value, label in allowed:
        if abs(value - allowed_value) < 1e-6:
            return label

    # 0.9% is easy to OCR as "09" or "9" when the decimal point is missed.
    if value_text in ("09", "9"):
        return "0.9%"

    return None


def parse_required_fields(result):
    drug_name = None
    concentration = None
    volume = None
    normalized_texts = []

    if result is None or len(result) == 0 or result[0] is None:
        raise ValueError("OCR没有检测到任何文本")

    for line in result[0]:
        text = line[1][0]
        text = _normalize_ocr_unit_text(text)
        normalized_texts.append(text)

        m = re.search(r'[\u4e00-\u9fa5]+液', text)
        if m:
            drug_name = m.group()

        m = re.search(r'(\d+(?:\.\d+)?)%', text)
        if m:
            concentration = _normalize_concentration_value(m.group(1))
            if concentration is None:
                concentration = f"{_format_number_text(m.group(1))}%"

        m = re.search(r'(\d+(?:\.\d+)?)(?:ml|m1|毫升)', text)
        if m:
            volume = f"{_format_number_text(m.group(1))}ml"

    if concentration is None:
        # The percent sign is very small and often missed. If OCR gets a known
        # concentration number such as "0.9", "5", "10", or "09" while volume
        # is on a separate "100ml" line, infer the missing percent sign.
        for text in normalized_texts:
            if re.search(r'(?:ml|m1|毫升)', text):
                continue
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            if len(numbers) != 1:
                continue

            inferred = _normalize_concentration_value(numbers[0])
            if inferred is not None:
                concentration = inferred
                break

    if drug_name is None:
        raise ValueError("未检测到完整的 XXX液")

    if concentration is None:
        raise ValueError("未检测到完整的 xx%")

    if volume is None:
        raise ValueError("未检测到完整的 xxml")

    return drug_name, concentration, volume

# ==================== 调试工具函数 ====================

def visualize_detection(img, boxes, save_path='det_result.jpg'):
    """
    可视化检测结果，用于调试

    Args:
        img: 原始图像
        boxes: 检测到的文本框列表
        save_path: 保存路径
    """
    vis_img = img.copy()
    for i, box in enumerate(boxes):
        box_int = box.astype(np.int32)
        cv2.polylines(vis_img, [box_int], True, (0, 255, 0), 2)
        # 标注序号
        cv2.putText(vis_img, str(i), tuple(box_int[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(save_path, vis_img)
    print(f"检测可视化结果已保存到: {save_path}")


def debug_single_image(det_model, rec_model, char_dict, img_path):
    """
    调试单张图像的完整OCR流程

    Args:
        det_model: RKNN检测模型
        rec_model: RKNN识别模型
        char_dict: 字符字典
        img_path: 图像路径
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return

    print(f"图像尺寸: {img.shape}")
    orig_h, orig_w = img.shape[:2]

    # Step 1: 检测
    
    det_input = cv2.resize(img, (448, 448))
    batch_det = np.stack([det_input], axis=0)
    det_outputs = det_model.inference(inputs=[batch_det])
    print(f"检测模型输出shape: {det_outputs[0].shape}")
    print(f"检测模型输出范围: [{det_outputs[0].min():.4f}, {det_outputs[0].max():.4f}]")

    boxes, scores = db_postprocess(det_outputs[0], orig_h, orig_w)
    print(f"检测到 {len(boxes)} 个文本框")

    # 可视化检测结果
    visualize_detection(img, boxes)

    # Step 2: 逐框识别
    for i, box in enumerate(boxes):
        crop = get_rotate_crop_image(img, box)
        if crop is None:
            continue
        processed = resize_for_rec(crop, 48, 160)
        batch_data = np.stack([processed] * 16, axis=0)  # padding to batch

        rec_outputs = rec_model.inference(inputs=[batch_data])
        print(f"识别模型输出shape: {rec_outputs[0].shape}")

        text, conf = ctc_decode_single(rec_outputs[0][i], char_dict)
        print(f"  框{i}: '{text}' (置信度: {conf:.3f})")

    # 完整流程
    result = recognize_text_rknn(det_model, rec_model, char_dict, img)
    print(f"\n最终合并结果: '{result}'")


class OCRRecognizer_ori:
    """
    图像 OCR 识别器。
    职责：图像校正、OCR 调用、文本提取与合并。
    """
    def __init__(self, ocr_model=None, image_processor=None):
        """
        :param ocr_model: 已初始化的 PaddleOCR 实例（或任何实现了 ocr() 方法的对象）
        :param image_processor: 图像处理器实例，默认使用 ImageProcessor()
        """
        print("初始化OCR模型...")
        # self.det_model = RKNNLite()
        # self.det_model.load_rknn("/home/forlinx/Models/AnotherYiliao/shibie/YiLiaoShiBie/src/identification/Det_bs1.rknn")
        # ret_det = self.det_model.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
        
        # self.rec_model = RKNNLite()
        # self.rec_model.load_rknn("/home/forlinx/Models/AnotherYiliao/shibie/YiLiaoShiBie/src/identification/Rec_bs16.rknn")
        # ret_rec = self.rec_model.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
        
        self.rec_model = ocr_model
        # self.char_dict = load_char_dict('ppocr_keys_v1.txt')
        # self.image_processor = image_processor or ImageProcessor()

        self.cls_model = RKNNLite()
        self.cls_model.load_rknn("/home/forlinx/Models/AnotherYiliao/shibie/YiLiaoShiBie/model_cls.rknn")
        CLS = self.cls_model.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    def recognize(self, image, line_gap_threshold=20):
        """
        识别图像中的文字，返回过滤后的汉字字符串。
        :param image: 图像数组 (numpy) 或文件路径
        :param line_gap_threshold: 行间距阈值，用于合并同一行文字
        :return: 识别出的纯汉字字符串，失败返回空字符串
        """
        # 自动旋转校正
        # corrected_img = self.image_processor.image_enhance(image)
        if image is None:
            return ""

        # try:
        #     corrected_img = cv2.resize(corrected_img, (448, 448))
        #     batch_data = np.stack([corrected_img] * 1, axis=0)
        #     outputs1 = self.det_model.inference(inputs=[batch_data])
        #     print("ocr定位掩码结果:", outputs1[0].shape)
            
            
            
        #     ###应该由实际outputs1得到检测框裁剪得到图像
        #     X_1=cv2.resize(corrected_img, (160, 48))
        #     batch_X_1 = np.stack([X_1] * 16, axis=0)
        #     outputs2 = self.rec_model.inference(inputs=[batch_X_1])
        #     print("ocr测试识别结果:", outputs2[0].shape)
            
            
        # except Exception as e:
        #     # 可保留异常打印（调试用），此处注释掉
        #     # print(f"⚠️ OCR 识别失败：{e}")
        #     return ""
        # corrected_img = np.ascontiguousarray(image)
        # corrected_img = np.array(image, dtype=np.uint8, order='C').copy()
        # corrected_img = paddle.Tensor(corrected_img)
        # corrected_img = corrected_img.mutable_data(paddle.CPUPlace()).numpy()
        # print(corrected_img.shape)
        start = time.time()
        print("开始角度分类")
        input_nchw = cv2.resize(image, (224, 224))
        batch_data = np.stack([input_nchw]*1, axis=0)
        outputs = self.cls_model.inference(inputs=[batch_data])
        outputs = outputs[0]
        pred_indices = np.argmax(outputs, axis=1)  # 形状 (20,)
        pred = pred_indices[0]
        image=reverse_rotate_with_label(image, pred)
        print("完成角度旋转")
        result = self.rec_model.ocr(image, det=True, rec=True)
        print(f'识别用时:{time.time() - start:.3f}s')
        # del corrected_img
        del image
        ##原始利用paddleocr模型识别文本
        if not result or not result[0]:
            return ""

        # 提取文本块
        text_blocks = []
        for line in result[0]:
            if line and len(line) > 1:
                bbox = line[0]
                text = line[1][0]
                # 只保留汉字
                text = re.sub(r'[^\u4e00-\u9fff]', '', text)
                if not text:
                    continue
                y_coords = [point[1] for point in bbox]
                center_y = sum(y_coords) / 4
                text_blocks.append({'text': text, 'center_y': center_y})

        if not text_blocks:
            return ""

        # 按 Y 坐标排序并合并行
        text_blocks.sort(key=lambda x: x['center_y'])
        lines = []
        current_line = [text_blocks[0]['text']]
        prev_y = text_blocks[0]['center_y']

        for i in range(1, len(text_blocks)):
            y = text_blocks[i]['center_y']
            if abs(y - prev_y) > line_gap_threshold:
                lines.append(''.join(current_line))
                current_line = [text_blocks[i]['text']]
            else:
                current_line.append(text_blocks[i]['text'])
            prev_y = y
        lines.append(''.join(current_line))

        # 移除耗时打印
        # print(f'识别用时:{time.time() - start:.3f}s')
        return ''.join(lines)
    def recognize_yaodai(self, image, line_gap_threshold=20):
        
        input_nchw = cv2.resize(image, (224, 224))
        batch_data = np.stack([input_nchw]*1, axis=0)
        outputs = self.cls_model.inference(inputs=[batch_data])
        outputs = outputs[0]
        pred_indices = np.argmax(outputs, axis=1)  # 形状 (20,)
        pred = pred_indices[0]
        image=reverse_rotate_with_label(image, pred)
        result = self.rec_model.ocr(image, det=True, rec=True)
        # del corrected_img
        del image
        ##原始利用paddleocr模型识别文本
        if not result or not result[0]:
            return ""

        # 提取文本块
        text_blocks = []
        for line in result[0]:
            if line and len(line) > 1:
                bbox = line[0]
                text = line[1][0]
                # 只保留汉字
                text = re.sub(r'[^\u4e00-\u9fff]', '', text)
                if not text:
                    continue
                y_coords = [point[1] for point in bbox]
                center_y = sum(y_coords) / 4
                text_blocks.append({'text': text, 'center_y': center_y})

        if not text_blocks:
            return ""

        # 按 Y 坐标排序并合并行
        text_blocks.sort(key=lambda x: x['center_y'])
        lines = []
        current_line = [text_blocks[0]['text']]
        prev_y = text_blocks[0]['center_y']

        for i in range(1, len(text_blocks)):
            y = text_blocks[i]['center_y']
            if abs(y - prev_y) > line_gap_threshold:
                lines.append(''.join(current_line))
                current_line = [text_blocks[i]['text']]
            else:
                current_line.append(text_blocks[i]['text'])
            prev_y = y
        lines.append(''.join(current_line))
        return lines[0]
    
        # return "-----"
    def recognize_shuyedai(self, image, line_gap_threshold=20):
        # 转 HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 蓝色范围，需要根据实际图片微调
        lower_blue = np.array([90, 50, 40])
        upper_blue = np.array([145, 255, 255])

        # 提取蓝色区域 mask
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # 形态学操作，去噪并连接蓝色区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12,12))
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=1)

        # 查找蓝色区域轮廓
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找最大蓝色区域，通常就是蓝底文字区域
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            # 适当扩大边界
            pad = 10
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, image.shape[1])
            y2 = min(y + h + pad, image.shape[0])

            image = image[y1:y2, x1:x2]
        else:
            image = image.copy()
        
        ###完成新药袋的ocr检测
        input_nchw = cv2.resize(image, (224, 224))
        batch_data = np.stack([input_nchw]*1, axis=0)
        outputs = self.cls_model.inference(inputs=[batch_data])
        outputs = outputs[0]
        pred_indices = np.argmax(outputs, axis=1)  # 形状 (20,)
        pred = pred_indices[0]
        
        image = cv2.resize(image, (448, 448))
        image=reverse_rotate_with_label(image, pred)
        
        result=self.rec_model.ocr(image, det=True, rec=True)
        print("初始检测结果")
        print(result)
        if result==None:
            return
        cv2.imwrite("image.png", image)
        result=parse_required_fields(result)
        print(result)
        # del corrected_img
    def recognize_batch(self, image_list, line_gap_threshold=20):
        """
        批量识别图像列表中的文字。
        :param image_list: 图像列表（numpy 数组或文件路径）
        :param line_gap_threshold: 行间距阈值
        :return: 识别结果列表，每个元素为字典 {"image": 原图, "ocr_text": 文本}
        """
        results = []
        for img in image_list:
            text = self.recognize(img, line_gap_threshold)
            results.append({
                "image": img,
                "ocr_text": text
            })
        return results
