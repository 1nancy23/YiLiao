import cv2
import numpy as np
import os
from typing import List, Any, Tuple

class ImageProcessor:
    """图像处理工具类，提供旋转、清晰度评估、最清晰图像选择等功能"""

    @staticmethod
    def image_enhance(image):
        """
        图像增强：基于瓶子边缘方向进行倾斜校正，然后统一应用增强（放大+CLAHE）。
        支持传入图像数组或文件路径。
        :param image: 图像数组或文件路径
        :return: 增强后的图像数组，失败返回 None
        """
        # 1. 加载图像
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                print(f"⚠️ 无法读取图片：{image}")
                return None
        else:
            img = image.copy()  # 避免修改原始数组

        # 内部增强函数
        def apply_enhancement(img_to_enhance):
            enlarged = cv2.resize(img_to_enhance, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return enhanced

        # ------------------- 倾斜校正（基于瓶子边缘）-------------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 轻度模糊以减少噪声干扰
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Canny 边缘检测，提取高频区域（瓶子边缘）
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        angle = 0.0
        valid_contours = []
        # 筛选可能是瓶子的轮廓：面积较大，且形状不是极端细长或点状
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # 忽略小噪点
                continue
            # 可进一步用长宽比过滤（例如瓶子通常在 0.3~3 之间）
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            aspect_ratio = max(w, h) / min(w, h)
            if 1.2 < aspect_ratio < 6:  # 瓶子通常有一定长度比例
                valid_contours.append(cnt)

        if valid_contours:
            # 将所有符合条件的轮廓点合并计算总体的外接矩形方向
            all_pts = np.vstack(valid_contours)
            rect = cv2.minAreaRect(all_pts)
            angle = rect[2]  # OpenCV 中 minAreaRect 返回的角度范围是 [-90, 0)
            # 角度转换：使水平或竖直方向为 0°
            if angle < -45:
                angle = 90 + angle
            # 校正角度（我们需要的旋转角度）
            rot_angle = angle
        else:
            rot_angle = 0.0

        # 若倾斜角度超过阈值，进行旋转校正
        if abs(rot_angle) > 3:  # 阈值可调
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            # 计算旋转矩阵
            M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
            # 计算新画布大小以避免裁剪
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            img = cv2.warpAffine(img, M, (new_w, new_h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

        # ------------------- 统一增强处理 -------------------
        img = apply_enhancement(img)
        cv2.imwrite('enhanced.jpg', img)
        return img



    @staticmethod
    def estimate_sharpness(image):
        """
        估计图像清晰度（使用拉普拉斯方差）。
        :param image: 图像数组
        :return: 方差值，越大表示越清晰
        """
        if image is None:
            return 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def select_sharpest_image(batches: List[List[Any]], use_roi: bool = False, roi_ratio: float = 0.8) -> Tuple[Any, Tuple[int, int]]:
        """
        遍历二维图像列表，找到最清晰的一张图像及其位置（已修复尺寸偏差问题）。

        参数:
            batches: 二维列表，例如 [[path1, path2], [path3, path4]]
                     元素可以是文件路径 (str) 或 numpy 数组。
            use_roi: 是否启用感兴趣区域(ROI)聚焦（推荐用于标签类图像）
            roi_ratio: ROI区域占图像的比例（0.0-1.0），默认0.8（80%的图像区域）

        返回:
            tuple: (最清晰的图像对象/路径, (行索引, 列索引))
                   如果列表为空或所有图像无法读取，返回 (None, (-1, -1))
        """
        best_image = None
        best_position = (-1, -1)
        max_score = -1.0
        found_valid = False

        for i, row in enumerate(batches):
            for j, item in enumerate(row):
                current_score = 0.0
                gray_img = None

                # --- 数据加载与预处理 ---
                try:
                    if isinstance(item, str):
                        # 情况 A: 传入的是文件路径
                        if not os.path.exists(item):
                            continue  # 跳过不存在的路径

                        img = cv2.imread(item)
                        if img is None:
                            continue  # 跳过无法读取的图片

                        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    elif hasattr(item, 'shape'):
                        # 情况 B: 传入的是 numpy 数组 (OpenCV/PIL 转成的数组)
                        # 假设 item 已经是 BGR 或 RGB 格式
                        if len(item.shape) == 3:
                            gray_img = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
                        elif len(item.shape) == 2:
                            gray_img = item  # 已经是灰度图
                        else:
                            continue

                    else:
                        # 不支持的类型
                        continue

                    # --- 计算归一化清晰度分数 ---
                    h, w = gray_img.shape[:2]
                    total_area = h * w

                    # 如果启用ROI，聚焦标签区域
                    if use_roi:
                        roi_h = int(h * roi_ratio)
                        roi_w = int(w * roi_ratio)
                        start_h = (h - roi_h) // 2
                        start_w = (w - roi_w) // 2
                        roi = gray_img[start_h:start_h + roi_h, start_w:start_w + roi_w]
                        lap = cv2.Laplacian(roi, cv2.CV_64F)
                    else:
                        lap = cv2.Laplacian(gray_img, cv2.CV_64F)

                    # 归一化：方差除以面积，避免大图因尺寸大而得分高
                    score = lap.var() / max(1, total_area)

                    # --- 更新最大值 ---
                    if not found_valid or score > max_score:
                        max_score = score
                        best_image = item  # 返回原始传入的对象（路径或数组）
                        best_position = (i, j)
                        found_valid = True

                except Exception as e:
                    print(f"警告：处理位置 [{i}][{j}] 的图像时出错: {e}")
                    continue

        if not found_valid:
            return None, (-1, -1)

        return best_image, best_position


# 简单测试
if __name__ == "__main__":
    # 测试自动旋转
    img_path = "../identification/img_3.png"  # 请替换为实际图片路径
    rotated = ImageProcessor.image_enhance(img_path)
    if rotated is not None:
        cv2.imshow("Rotated", rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # # 测试清晰度
    # img = cv2.imread(img_path)
    # if img is not None:
    #     sharp = ImageProcessor.estimate_sharpness(img)
    #     print(f"清晰度: {sharp}")
    #
    # # 测试 select_sharpest_image
    # batches = [
    #     ["img1.jpg", "img2.jpg"],
    #     ["img3.jpg", "img4.jpg"]
    # ]
    # best_img, pos = ImageProcessor.select_sharpest_image(batches, use_roi=True)
    # if best_img is not None:
    #     print(f"最清晰图像位于: {pos}, 路径: {best_img}")
    # else:
    #     print("未找到有效图像")