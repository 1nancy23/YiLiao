import cv2
import numpy as np
import os
from typing import List, Any, Tuple

class ImageProcessor:
    """图像处理工具类，提供旋转、清晰度评估、最清晰图像选择等功能"""

    @staticmethod
    def _load_image(image):
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                print(f"⚠️ 无法读取图片：{image}")
            return img
        if image is None:
            return None
        return image.copy()

    @staticmethod
    def _border_median_color(img):
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return (255, 255, 255)
        strip = max(1, min(h, w) // 20)
        border = np.concatenate((
            img[:strip, :, :].reshape(-1, 3),
            img[h - strip:, :, :].reshape(-1, 3),
            img[:, :strip, :].reshape(-1, 3),
            img[:, w - strip:, :].reshape(-1, 3),
        ), axis=0)
        color = np.median(border, axis=0)
        return tuple(int(v) for v in color)

    @staticmethod
    def _weighted_median(values, weights):
        values = np.asarray(values, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        if values.size == 0:
            return None
        order = np.argsort(values)
        values = values[order]
        weights = weights[order]
        midpoint = np.sum(weights) * 0.5
        return float(values[np.searchsorted(np.cumsum(weights), midpoint)])

    @staticmethod
    def _dominant_angle_cluster(angles, weights, cluster_width=12.0):
        """Choose one coherent direction before taking its weighted median."""
        angles = np.asarray(angles, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        if angles.size == 0:
            return None

        # Line directions repeat every 180 degrees. Compare them in doubled-
        # angle space so -89 and +89 degrees are treated as neighbours.
        doubled = np.deg2rad(angles * 2.0)
        best_score = -1.0
        best_mask = None
        for center in angles:
            delta = np.abs(((angles - center + 90.0) % 180.0) - 90.0)
            mask = delta <= float(cluster_width)
            score = float(np.sum(weights[mask]))
            if score > best_score:
                best_score = score
                best_mask = mask

        if best_mask is None or not np.any(best_mask):
            return None

        # Circular mean provides a stable centre; the median suppresses the
        # remaining character strokes or isolated background lines.
        cluster_angles = angles[best_mask]
        cluster_weights = weights[best_mask]
        vector = np.sum(cluster_weights * np.exp(1j * doubled[best_mask]))
        center = np.degrees(np.angle(vector)) * 0.5
        unwrapped = center + ((cluster_angles - center + 90.0) % 180.0) - 90.0
        result = ImageProcessor._weighted_median(unwrapped, cluster_weights)
        return float(((result + 90.0) % 180.0) - 90.0)

    @staticmethod
    def _rotate_bound(img, angle, border_value):
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        matrix[0, 2] += new_w / 2.0 - center[0]
        matrix[1, 2] += new_h / 2.0 - center[1]
        return cv2.warpAffine(
            img,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

    @staticmethod
    def _rotation_to_axis(axis_angle, target_axis):
        axis = str(target_axis or "vertical").lower()
        if axis == "horizontal":
            rotate_angle = axis_angle
        elif axis == "nearest":
            if abs(axis_angle) <= 45.0:
                target = 0.0
            else:
                target = 90.0 if axis_angle >= 0 else -90.0
            rotate_angle = axis_angle - target
        else:
            target = 90.0 if axis_angle >= 0 else -90.0
            rotate_angle = axis_angle - target

        while rotate_angle <= -90.0:
            rotate_angle += 180.0
        while rotate_angle > 90.0:
            rotate_angle -= 180.0
        return rotate_angle

    @staticmethod
    def _foreground_axis_angle(img):
        h, w = img.shape[:2]
        if min(h, w) < 30:
            return None

        margin_x = max(2, int(w * 0.04))
        margin_y = max(2, int(h * 0.04))
        rect = (
            margin_x,
            margin_y,
            max(1, w - 2 * margin_x),
            max(1, h - 2 * margin_y),
        )
        mask = np.zeros((h, w), np.uint8)
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(img, mask, rect, bg_model, fg_model, 1, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return None

        fg_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < max(80.0, h * w * 0.08):
            return None

        (_cx, _cy), (rw, rh), angle = cv2.minAreaRect(contour)
        if rw < 5 or rh < 5:
            return None

        axis_angle = angle if rw >= rh else angle + 90.0
        axis_angle = ((axis_angle + 90.0) % 180.0) - 90.0
        return float(axis_angle)

    @staticmethod
    def _text_axis_angle(img, ignore_border_ratio=0.20, return_debug=False):
        """Estimate printed text/rule direction from the central image area."""
        h, w = img.shape[:2]
        margin = float(np.clip(ignore_border_ratio, 0.0, 0.40))
        x1, x2 = int(w * margin), int(w * (1.0 - margin))
        y1, y2 = int(h * margin), int(h * (1.0 - margin))
        roi = img[y1:y2, x1:x2]
        if roi.size == 0 or min(roi.shape[:2]) < 20:
            return (None, None) if return_debug else None

        if roi.ndim == 2:
            gray = roi
        elif roi.shape[2] == 1:
            gray = roi[:, :, 0]
        elif roi.shape[2] == 4:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 35, 110, apertureSize=3)
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        rh, rw = roi.shape[:2]
        short_side = max(1, min(rh, rw))
        long_side = max(rh, rw)
        min_length = max(10, int(long_side * 0.10))
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360,
            threshold=max(8, int(short_side * 0.045)),
            minLineLength=min_length,
            maxLineGap=max(5, int(short_side * 0.08)),
        )

        angles, weights = [], []
        debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if return_debug else None
        if lines is not None:
            for line in lines[:, 0, :]:
                lx1, ly1, lx2, ly2 = line
                dx, dy = float(lx2 - lx1), float(ly2 - ly1)
                length = float(np.hypot(dx, dy))
                if length < min_length:
                    continue
                angle = np.degrees(np.arctan2(dy, dx))
                angle = ((angle + 90.0) % 180.0) - 90.0
                angles.append(angle)
                weights.append(length * length)
                if return_debug:
                    cv2.line(debug, (lx1, ly1), (lx2, ly2), (0, 0, 255), 1)

        dominant = ImageProcessor._dominant_angle_cluster(
            angles,
            weights,
            cluster_width=10.0,
        )
        if return_debug:
            return dominant, (edges, debug, (x1, y1, x2, y2))
        return dominant

    @staticmethod
    def rotate_text_image(
        image,
        target_axis="horizontal",
        min_rotate_degrees=2.0,
        ignore_border_ratio=0.20,
        return_debug=False,
    ):
        """Deskew a bag label from text/rule directions in its central 60%."""
        img = ImageProcessor._load_image(image)
        if img is None or img.size == 0:
            return (None, None) if return_debug else None

        estimate = ImageProcessor._text_axis_angle(
            img,
            ignore_border_ratio=ignore_border_ratio,
            return_debug=return_debug,
        )
        if return_debug:
            dominant_angle, debug = estimate
        else:
            dominant_angle, debug = estimate, None

        if dominant_angle is None:
            return (img, debug) if return_debug else img
        rotate_angle = ImageProcessor._rotation_to_axis(dominant_angle, target_axis)
        if abs(rotate_angle) < float(min_rotate_degrees):
            return (img, debug) if return_debug else img

        result = ImageProcessor._rotate_bound(
            img,
            rotate_angle,
            ImageProcessor._border_median_color(img),
        )
        return (result, debug) if return_debug else result

    @staticmethod
    def auto_rotate_image(
        image,
        target_axis="vertical",
        min_rotate_degrees=3.0,
        min_line_ratio=0.35,
        max_detect_side=480,
        use_foreground=False,
        hough_threshold_ratio=0.10,
        max_line_gap_ratio=0.16,
        edge_close_kernel=5,
    ):
        """
        Fast bottle-axis correction for YOLO crops.

        By default it uses only long edge segments with Canny + HoughLinesP for
        speed. Set use_foreground=True to run a slower foreground-contour
        estimate first, useful when strong background edges dominate the crop.
        If no stable axis is found, the original crop is returned unchanged.
        edge_close_kernel and max_line_gap_ratio let slightly curved/broken
        label edges be treated as one dominant straight segment.
        """
        img = ImageProcessor._load_image(image)
        if img is None or img.size == 0:
            return None

        h, w = img.shape[:2]
        if min(h, w) < 20:
            return img

        if use_foreground:
            foreground_angle = ImageProcessor._foreground_axis_angle(img)
            if foreground_angle is not None:
                rotate_angle = ImageProcessor._rotation_to_axis(foreground_angle, target_axis)
                if abs(rotate_angle) < float(min_rotate_degrees):
                    return img
                border_value = ImageProcessor._border_median_color(img)
                return ImageProcessor._rotate_bound(img, rotate_angle, border_value)

        detect_img = img
        scale = 1.0
        max_side = max(h, w)
        if max_detect_side and max_side > max_detect_side:
            scale = max_detect_side / float(max_side)
            detect_img = cv2.resize(
                img,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        if edge_close_kernel and edge_close_kernel >= 3:
            k = int(edge_close_kernel)
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        dh, dw = detect_img.shape[:2]
        short_side = max(1, min(dh, dw))
        long_side = max(dh, dw)
        min_line_length = max(18, int(long_side * float(min_line_ratio)))
        max_line_gap = max(8, int(short_side * float(max_line_gap_ratio)))
        threshold = max(14, int(short_side * float(hough_threshold_ratio)))

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        if lines is None:
            return img

        angles = []
        lengths = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = line
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = float(np.hypot(dx, dy))
            if length < min_line_length:
                continue
            angle = np.degrees(np.arctan2(dy, dx))
            angle = ((angle + 90.0) % 180.0) - 90.0
            angles.append(angle)
            midpoint_x = (x1 + x2) * 0.5 / max(1.0, float(dw))
            midpoint_y = (y1 + y2) * 0.5 / max(1.0, float(dh))
            center_distance = min(1.0, np.hypot(midpoint_x - 0.5, midpoint_y - 0.5) / 0.707)
            center_weight = 1.0 - 0.65 * center_distance
            lengths.append(length * length * center_weight)

        dominant_angle = ImageProcessor._dominant_angle_cluster(
            angles,
            lengths,
            cluster_width=12.0,
        )
        if dominant_angle is None:
            return img

        rotate_angle = ImageProcessor._rotation_to_axis(dominant_angle, target_axis)

        if abs(rotate_angle) < float(min_rotate_degrees):
            return img

        border_value = ImageProcessor._border_median_color(img)
        return ImageProcessor._rotate_bound(img, rotate_angle, border_value)
    
    @staticmethod
    def image_enhance(image):
        """
        图像增强：先进行倾斜校正（基于原始图像），然后统一应用增强（放大+CLAHE）。
        支持传入图像数组或文件路径。
        :param image: 图像数组或文件路径
        :return: 增强后的图像数组，失败返回 None
        """
        img = ImageProcessor._load_image(image)
        if img is None:
            return None

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
        img = ImageProcessor.auto_rotate_image(img)
        if img is None:
            return None

        # 3. 无论是否旋转，都执行增强
        enhanced_img = apply_enhancement(img)
        return enhanced_img


    @staticmethod
    def reduce_glare(image, strength=0.9, inpaint_radius=5, return_mask=False):
        """
        Reduce plastic-film glare with OpenCV only.

        This is designed for recognition preprocessing:
        1. correct slow illumination changes on LAB-L,
        2. detect low-saturation saturated highlights,
        3. avoid strong text/edge pixels as much as possible,
        4. inpaint the remaining specular mask,
        5. restore local contrast with CLAHE and mild sharpening.
        """
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                print(f"Failed to read image: {image}")
                return (None, None) if return_mask else None
        else:
            img = image.copy()

        h, w = img.shape[:2]
        img_area = max(1, h * w)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # Step 1: repair compact saturated highlights.
        v_thr = max(210, int(np.percentile(v, 96.5)))
        bright_low_sat = cv2.inRange(v, v_thr, 255) & cv2.inRange(s, 0, 120)

        bgr_min = np.min(img, axis=2)
        bgr_max = np.max(img, axis=2)
        neutral_white = ((bgr_min > 205) & ((bgr_max - bgr_min) < 65)).astype(np.uint8) * 255
        mask = cv2.bitwise_or(bright_low_sat, neutral_white)

        # Keep likely text strokes and hard edges out of the inpaint mask.
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        grad = cv2.convertScaleAbs(cv2.addWeighted(
            cv2.convertScaleAbs(grad_x), 0.5,
            cv2.convertScaleAbs(grad_y), 0.5, 0
        ))
        edge_keep = cv2.dilate(cv2.inRange(grad, 60, 255), np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(edge_keep))

        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel7, iterations=2)

        # Drop tiny components; these are usually characters or sensor noise.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean_mask = np.zeros_like(mask)
        min_area = max(30, int(img_area * 0.00035))
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            if area >= min_area and bw >= 6 and bh >= 6:
                clean_mask[labels == i] = 255
        clean_mask = cv2.dilate(clean_mask, kernel3, iterations=2)

        inpainted = cv2.inpaint(img, clean_mask, inpaint_radius, cv2.INPAINT_TELEA)
        mask_soft = cv2.GaussianBlur(clean_mask.astype(np.float32) / 255.0, (0, 0), 2.0)
        mask_soft = np.clip(mask_soft * strength, 0.0, 1.0)[:, :, None]
        base = (img.astype(np.float32) * (1.0 - mask_soft) +
                inpainted.astype(np.float32) * mask_soft).astype(np.uint8)

        # Step 2: reduce broad plastic-film glare as an additive specular layer.
        dark = np.min(base, axis=2)
        bg_kernel = max(31, (min(h, w) // 14) | 1)
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_kernel, bg_kernel))
        base_dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, bg_kernel)
        specular = cv2.subtract(dark, base_dark)
        specular = cv2.GaussianBlur(specular, (0, 0), 5)

        base_hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
        base_v = base_hsv[:, :, 2]
        broad_mask = ((specular > 12) & (base_v > 145)).astype(np.uint8) * 255
        broad_mask = cv2.morphologyEx(broad_mask, cv2.MORPH_OPEN, kernel5, iterations=1)
        broad_mask = cv2.morphologyEx(broad_mask, cv2.MORPH_CLOSE, kernel7, iterations=1)
        broad_mask = cv2.bitwise_and(broad_mask, cv2.bitwise_not(edge_keep))
        broad_soft = cv2.GaussianBlur(broad_mask.astype(np.float32) / 255.0, (0, 0), 3.0)
        specular_sub = (specular.astype(np.float32) * 0.85 * broad_soft)[:, :, None]
        corrected = np.clip(base.astype(np.float32) - specular_sub, 0, 255).astype(np.uint8)

        # Step 3: mild local contrast restoration without boosting glare back.
        base_lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        base_l, base_a, base_b = cv2.split(base_lab)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(base_l)
        enhanced_l = cv2.addWeighted(enhanced_l, 0.45, base_l, 0.55, 0)
        corrected = cv2.cvtColor(cv2.merge((enhanced_l, base_a, base_b)), cv2.COLOR_LAB2BGR)

        blur = cv2.GaussianBlur(corrected, (0, 0), 1.0)
        corrected = cv2.addWeighted(corrected, 1.15, blur, -0.15, 0)

        if return_mask:
            return corrected, cv2.bitwise_or(clean_mask, broad_mask)
        return corrected



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
