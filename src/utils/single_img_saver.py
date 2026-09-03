"数据集构建函数"
import os
import time
import cv2
import numpy as np

def safe_save_image(img, folder, prefix):
    """安全保存单张图像到指定文件夹"""
    if img is None or img.size == 0:
        return False
    # 转为 uint8
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    timestamp = int(time.time() * 1000)  # 毫秒时间戳
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(folder, filename)
    try:
        success = cv2.imwrite(filepath, img)
        return success
    except Exception as e:
        print(f"⚠️ 保存失败: {filepath}, 错误: {e}")
        return False

def save_cropped_images(cropped_bag, cropped_bottle):
    """统一保存药袋和药瓶图像"""
    bag_dir = "./data/data_base/bag_data"
    bottle_dir = "./data/data_base/bottle_data"
    os.makedirs(bag_dir, exist_ok=True)
    os.makedirs(bottle_dir, exist_ok=True)

    saved_bag = sum(
        safe_save_image(bag, bag_dir, "bag") for bag in cropped_bag
    )
    saved_bottle = sum(
        safe_save_image(bottle, bottle_dir, "bottle") for bottle in cropped_bottle
    )

    print(f"💾 成功保存 {saved_bag} 张药袋图像到 {bag_dir}")
    print(f"💾 成功保存 {saved_bottle} 张药瓶图像到 {bottle_dir}")

def img_shower(cropped_bag, cropped_bottle, save=False, show=False):
    # 如果都没有检测到，直接返回
    if len(cropped_bag) == 0 and len(cropped_bottle) == 0:
        print("⚠️ 未检测到药袋或药瓶，跳过显示/保存")
        return

    # 情况1：只显示，不保存
    if show and not save:
        for i, bag in enumerate(cropped_bag):
            cv2.imshow(f'Cropped bag {i + 1}', bag)
        for i, bottle in enumerate(cropped_bottle):
            cv2.imshow(f'Cropped bottle {i + 1}', bottle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 情况2：只保存，不显示（非交互式，一般用于后台）
    if save and not show:
        save_cropped_images(cropped_bag, cropped_bottle)
        return

    # 情况3：既显示又保存 → 进入交互模式
    if show and save:
        # 先显示所有图像
        for i, bag in enumerate(cropped_bag):
            cv2.imshow(f'Cropped bag {i + 1}', bag)
        for i, bottle in enumerate(cropped_bottle):
            cv2.imshow(f'Cropped bottle {i + 1}', bottle)

        print("⌨️ 按空格键保存图像，按任意其他键退出（不保存）")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == ord(' '):  # 空格键
            print("✅ 用户选择保存图像")
            save_cropped_images(cropped_bag, cropped_bottle)
        else:
            print("❌ 用户未按空格，跳过保存")

    # 其他情况（如 save=False, show=False）直接静默返回