import cv2
import numpy as np
import os
from typing import List, Any, Tuple

class ImageProcessor:
    """图像处理工具类，提供旋转、清晰度评估、最清晰图像选择等功能"""

    @staticmethod
    def image_enhance(image):
        """
        图像增强：先进行倾斜校正（基于原始图像），然后统一应用增强（放大+CLAHE）。
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
            img = image

        # 内部增强函数
        def apply_enhancement(img_to_enhance):
            # 放大 2 倍 (提升低分辨率文字清晰度)
            enlarged = cv2.resize(img_to_enhance, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            # CLAHE 对比度增强 (在 LAB 空间对亮度通道处理)
            lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return enhanced

        # 2. 倾斜校正 (基于原始图像)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        if lines is not None and len(lines) > 5:
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = np.degrees(theta) - 90
                angles.append(angle)
            angle = np.median(angles)
            if abs(angle) > 5:  # 大于5度才旋转
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                angle_rad = np.radians(angle)
                sin_angle = np.abs(np.sin(angle_rad))
                cos_angle = np.abs(np.cos(angle_rad))
                new_w = int(w * cos_angle + h * sin_angle)
                new_h = int(h * cos_angle + w * sin_angle)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(
                    img, M, (new_w, new_h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )

        # 3. 无论是否旋转，都执行增强
        # enhanced_img = apply_enhancement(img)
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
    img_path = "../identification/11.jpg"  # 请替换为实际图片路径
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