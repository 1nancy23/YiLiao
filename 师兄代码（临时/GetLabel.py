import napari
import skimage.io
import os
import numpy as np
from glob import glob

# ================= 配置路径 =================
# 您的图像文件夹
IMG_DIR = f'A:\项目\医药物品检测\output_dataset_1222_1\images'
# 您的标签文件夹
LBL_DIR = f'A:\项目\医药物品检测\output_dataset_1222_1\labels'
# 图像后缀 (根据实际情况修改，如 .jpg, .png)
EXT = '*.png' 
# ===========================================

def get_files():
    # 获取所有图片路径并排序，确保一一对应
    img_files = sorted(glob(os.path.join(IMG_DIR, EXT)))
    lbl_files = sorted(glob(os.path.join(LBL_DIR, EXT)))
    
    # 简单检查数量是否一致
    if len(img_files) != len(lbl_files):
        print(f"警告: 图像({len(img_files)}) 与 标签({len(lbl_files)}) 数量不一致！")
        print("请确保文件名是可以对应排序的。")
    
    return img_files, lbl_files

# 初始化文件列表
img_files, lbl_files = get_files()
current_idx = 0

# 启动 Napari
viewer = napari.Viewer(title="遥感图像标签修正工具")

def load_current_sample():
    """加载当前索引的图像和标签"""
    global current_idx
    
    if len(img_files) == 0:
        print("未找到文件")
        return

    # 1. 清除旧图层 (防止图层堆叠)
    viewer.layers.clear()

    # 2. 读取文件
    img_path = img_files[current_idx]
    lbl_path = lbl_files[current_idx]
    
    print(f"正在处理 [{current_idx+1}/{len(img_files)}]: {os.path.basename(img_path)}")

    image = skimage.io.imread(img_path)
    label = skimage.io.imread(lbl_path)

    # 3. 添加图层
    # 添加底图 (Image layer)
    viewer.add_image(image, name='image')
    
    # 添加标签 (Labels layer) -这是关键！必须用 add_labels 才能修改
    # 强制转为 int 类型，防止因读取为 float 而无法编辑
    lbl_layer = viewer.add_labels(label.astype(int), name='mask')
    
    # 4. 自动选中标签层并开启画笔模式
    viewer.layers.selection.active = lbl_layer
    lbl_layer.mode = 'paint' 
    # 设置画笔大小
    lbl_layer.brush_size = 10 

def save_current_mask():
    """保存当前的标签修改"""
    if 'mask' in viewer.layers:
        # 获取当前标签数据
        modified_label = viewer.layers['mask'].data
        # 获取保存路径 (覆盖原文件)
        save_path = lbl_files[current_idx]
        
        # 保存 (注意保持数据类型，通常 mask 是 uint8)
        skimage.io.imsave(save_path, modified_label.astype(np.uint8), check_contrast=False)
        print(f"已保存修改: {os.path.basename(save_path)}")
    else:
        print("当前没有标签层可保存")

# ================= 绑定快捷键 =================

@viewer.bind_key('n')
def next_sample(viewer):
    """按 'n' 保存当前并在切换到下一张"""
    global current_idx
    save_current_mask() # 自动保存
    
    if current_idx < len(img_files) - 1:
        current_idx += 1
        load_current_sample()
    else:
        print("已经是最后一张了！")

@viewer.bind_key('p')
def prev_sample(viewer):
    """按 'p' 切换到上一张 (不自动保存，防止误操作)"""
    global current_idx
    if current_idx > 0:
        current_idx -= 1
        load_current_sample()
    else:
        print("已经是第一张了！")

@viewer.bind_key('s')
def save_only(viewer):
    """按 's' 仅保存"""
    save_current_mask()

# ================= 开始运行 =================
if __name__ == '__main__':
    if len(img_files) > 0:
        load_current_sample()
        napari.run()
    else:
        print("文件夹为空或路径错误。")