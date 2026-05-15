"切割模型识别后的图像裁剪函数"
import numpy as np
import cv2
import time
import os



def crop_target_regions(result_frame, prediction,bag_min_area,bottle_min_area):
    """
    裁剪药袋(类别1)和药瓶(类别2)的连续区域（调用核心函数）

    :param result_frame: 分割后的可视化图像 (batch, H, W, 3)
    :param prediction: 原始分割结果 (batch, H, W)
    :param min_area: 最小面积阈值
    :return: (crop_bag, crop_bottle)
    一次裁剪约用时：0.05230093002319336
    """
    crop_bag = crop_single_class(result_frame, prediction, 1, bag_min_area)
    crop_bottle = crop_single_class(result_frame, prediction, 2, bottle_min_area)
    return crop_bag, crop_bottle


def crop_single_class(result_frame, prediction, class_id, min_area=1000):
    """
    裁剪指定类别中所有面积 >= min_area 的连通区域，并调整至接近5:4比例
    :param result_frame: 分割后的可视化图像 (H, W, 3) 或 (batch, H, W, 3) —— 实际使用时应传入单张
    :param prediction: 原始分割结果 (batch, H, W)
    :param class_id: 目标类别ID (1=药袋, 2=药瓶)
    :param min_area: 最小面积阈值（像素数）
    :return: list of cropped images [(H_crop, W_crop, 3), ...]
    """
    # 若传入batch维，取第一张（可根据实际调整）
    if len(result_frame.shape) == 4:
        frame = result_frame[0]
    else:
        frame = result_frame
    pred = prediction[0]                     # (H, W)

    mask = (pred == class_id).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cropped_images = []
    target_ratio = 4.0 / 5.0                  # 目标宽高比 5:4

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        print(area)
        if area >= min_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # 计算原区域中心
            cx = x + w / 2.0
            cy = y + h / 2.0
            current_ratio = w / h if h != 0 else float('inf')

            # 根据比例计算目标宽高（保持中心不变）
            if current_ratio >= target_ratio:
                new_w = w
                new_h = w / target_ratio
            else:
                new_w = h * target_ratio
                new_h = h

            # 缩放至图像边界内（若超出则等比例缩小）
            frame_h, frame_w = frame.shape[:2]
            scale = min(frame_w / new_w, frame_h / new_h, 1.0)
            new_w = int(new_w * scale)
            new_h = int(new_h * scale)

            # 重新计算左上角坐标并确保在图像内
            new_x = int(cx - new_w / 2.0)
            new_y = int(cy - new_h / 2.0)
            new_x = max(0, min(new_x, frame_w - new_w))
            new_y = max(0, min(new_y, frame_h - new_h))

            # 裁剪
            cropped = frame[new_y:new_y + new_h, new_x:new_x + new_w]
            cropped_images.append(cropped)

    return cropped_images


# 使用示例（直接调用）
if __name__ == "__main__":
    # 模拟数据（实际使用时替换为你的模型输出）
    result_frame = np.random.randint(0, 255, (1, 1000, 1280, 3), dtype=np.uint8)
    prediction = np.random.randint(0, 3, (1, 1000, 1280), dtype=np.uint8)

    # 裁剪
    crop_bag, crop_bottle = crop_target_regions(result_frame, prediction, min_area=300)

    # 保存结果
    if crop_bag is not None:
        os.makedirs("train/bag", exist_ok=True)
        cv2.imwrite(f"train/bag/bag_{int(time.time())}.jpg", crop_bag)
        print(f"✅ 药袋裁剪成功！尺寸: {crop_bag.shape}")

    if crop_bottle is not None:
        os.makedirs("train/bottle", exist_ok=True)
        cv2.imwrite(f"train/bottle/bottle_{int(time.time())}.jpg", crop_bottle)
        print(f"✅ 药瓶裁剪成功！尺寸: {crop_bottle.shape}")