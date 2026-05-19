from ultralytics import YOLO
import os

def convert_yolo_to_onnx(model_path, output_dir=None, imgsz=640):
    """
    将YOLO模型转换为ONNX格式

    Args:
        model_path: YOLO模型路径(.pt文件)
        output_dir: ONNX输出目录，默认为模型所在目录
        imgsz: 输入图像尺寸，默认640
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    if output_dir is None:
        output_dir = os.path.dirname(model_path)

    os.makedirs(output_dir, exist_ok=True)

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    print(f"开始转换为ONNX格式...")
    print(f"输出路径: {onnx_path}")

    model.export(
        format='onnx',
        imgsz=imgsz,
        dynamic=False,
        simplify=True,
        opset=12
    )

    actual_onnx_path = os.path.join(os.path.dirname(model_path), f"{model_name}.onnx")

    if os.path.exists(actual_onnx_path):
        file_size = os.path.getsize(actual_onnx_path) / (1024 * 1024)
        print(f"✓ 转换成功！")
        print(f"  文件: {actual_onnx_path}")
        print(f"  大小: {file_size:.2f} MB")
        return actual_onnx_path
    else:
        print("✗ 转换失败")
        return None

if __name__ == '__main__':
    convert_yolo_to_onnx("../data/runs/detect/train6/weights/best.pt")
