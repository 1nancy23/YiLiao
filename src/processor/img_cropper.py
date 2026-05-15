"摄像头区域选择函数"
import numpy as np
import cv2

def crop_image_by_coordinates(frame, cropper):
    """按照坐标裁剪图像，保留原始通道维度（3通道）
    参数:
        frame: 原始图像，形状为 (1, H, W, 3) 或 (H, W, 3)
        cropper: 裁剪配置，包含 x_min, y_min, x_max, y_max
    返回:
        cropped_frame: 裁剪后的图像，形状为 (1, new_H, new_W, 3) 或 (new_H, new_W, 3)
    """
    if not cropper:
        return frame


    x_min = cropper['x_min']
    y_min = cropper['y_min']
    x_max = cropper['x_max']
    y_max = cropper['y_max']

    # 检查裁剪区域是否有效
    if x_min >= x_max or y_min >= y_max:
        print(f"⚠️ 裁剪区域无效！x_min={x_min} >= x_max={x_max} 或 y_min={y_min} >= y_max={y_max}")
        return frame  # 返回原始帧避免空图像

    # 检查输入维度
    if frame.ndim == 4 and frame.shape[0] == 1:
        img = frame[0]  # 转换为 (H, W, 3)
    elif frame.ndim == 3:
        img = frame
    else:
        raise ValueError("输入图像维度必须是 (1, H, W, 3) 或 (H, W, 3)")

    H, W, _ = img.shape
    # 确保坐标在有效范围内
    x_min = max(0, min(x_min, W - 1))
    y_min = max(0, min(y_min, H - 1))
    x_max = max(0, min(x_max, W - 1))
    y_max = max(0, min(y_max, H - 1))

    # 再次检查裁剪区域是否有效
    if x_min >= x_max or y_min >= y_max:
        print(f"⚠️ 有效范围内裁剪区域无效！x_min={x_min} >= x_max={x_max} 或 y_min={y_min} >= y_max={y_max}")
        return frame

    # 执行裁剪
    cropped_img = img[y_min:y_max, x_min:x_max, :]

    # 关键修复：检查裁剪后图像是否为空
    if cropped_img.size == 0:
        print(f"⚠️ 裁剪后图像为空！裁剪区域: [{x_min}, {y_min}, {x_max}, {y_max}]")
        print(f"原始图像尺寸: {H}x{W}")
        # 返回原始图像避免空帧
        return frame

    # 保留原始维度结构
    if frame.ndim == 4:
        cropped_frame = np.expand_dims(cropped_img, axis=0)
    else:
        cropped_frame = cropped_img

    return cropped_frame

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 模拟原始帧 (1, 2160, 3840, 3)
    original_frame = np.random.randint(0, 255, (1, 2160, 3840, 3), dtype=np.uint8)

    # 配置护士操作区域坐标 (示例值)
    cropper = {
        'x_min': 500,  # 左边界
        'y_min': 300,  # 上边界
        'x_max': 1500,  # 右边界
        'y_max': 900  # 下边界
    }

    # 执行裁剪
    cropped_frame = crop_image_by_coordinates(
        original_frame,
        cropper
    )

    # 验证结果
    print("原始帧形状:", original_frame.shape)  # (1, 2160, 3840, 3)
    print("裁剪后形状:", cropped_frame.shape)  # (1, 600, 1000, 3)
    print("裁剪区域尺寸:", (cropper['x_max'] - cropper['x_min'], cropper['y_max'] - cropper['y_min']))  # (1000, 600)

    # 保存结果用于验证
    cv2.imwrite("cropped_frame.jpg", cropped_frame[0])  # 从batch中取出