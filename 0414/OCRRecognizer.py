import cv2
import re
import time
import numpy as np
from typing import Optional, Union
from src.utils.img_utils import ImageProcessor
# from src.utils.init_utils import init_ocr_model

class OCRRecognizer:
    """
    图像 OCR 识别器。
    职责：图像校正、OCR 调用、文本提取与合并。
    """
    def __init__(self, ocr_model, image_processor=None):
        self.ocr_model = ocr_model
        self.image_processor = image_processor or ImageProcessor()

    def recognize(self, image, line_gap_threshold=20):
        """
        识别图像中的文字，返回过滤后的汉字字符串。
        :param image: 图像数组或文件路径
        :param line_gap_threshold: 行间距阈值，用于合并同一行文字
        :return: (识别出的纯汉字字符串, 平均置信度)，失败返回 (None, 0)
        """
        start = time.time()
        # 自动旋转校正（使用图像处理类）
        corrected_img = self.image_processor.image_enhance(image)
        if corrected_img is None:
            return None, 0

        try:
            result = self.ocr_model.ocr(corrected_img)
        except Exception as e:
            print(f"⚠️ OCR识别失败: {e}")
            return None, 0

        if not result or not result[0]:
            return None, 0

        # 提取文本块
        text_blocks = []
        confidences = []
        for line in result[0]:
            if line and len(line) > 1:
                bbox = line[0]
                text = line[1][0]
                confidence = line[1][1] if len(line[1]) > 1 else 0
                # 只保留汉字
                text = re.sub(r'[^\u4e00-\u9fff]', '', text)
                if not text:
                    continue
                y_coords = [point[1] for point in bbox]
                center_y = sum(y_coords) / 4
                text_blocks.append({'text': text, 'center_y': center_y})
                confidences.append(confidence)

        if not text_blocks:
            return None, 0

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

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        print(f'识别用时:{time.time() - start:.3f}s, 平均置信度:{avg_confidence:.3f}')
        return ''.join(lines), avg_confidence

    def recognize_batch(self, image_list, line_gap_threshold=20):
        results = []
        for img in image_list:
            text, confidence = self.recognize(img, line_gap_threshold)
            results.append({"image": img, "ocr_text": text, "confidence": confidence})
        return results


    def recognize_batch(self, image_list, line_gap_threshold=20):
        results = []
        for img in image_list:
            text = self.recognize(img, line_gap_threshold)
            results.append({"image": img, "ocr_text": text})
        return results




