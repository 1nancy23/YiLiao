"数据库初始化，模型初始化函数"
import paddle
import pymysql
import time
import cv2
from paddleocr import PaddleOCR

# ==================== 独立数据库函数 ====================
def init_db(host="192.168.137.1", user="root", password="root", database="medicine_db", charset='utf8'):
    """
    初始化数据库连接，返回连接对象（使用 DictCursor）。
    调用者需确保在使用完毕后调用 close_db() 关闭连接。
    """
    try:
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            charset=charset,
            port=3306,
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except pymysql.MySQLError as e:
        raise Exception(f"数据库连接失败: {e}")

def close_db(connection):
    """关闭数据库连接"""
    if connection:
        connection.close()

def init_ocr_model(use_gpu=None, show_log=False):
    """
    初始化 PaddleOCR 模型。
    参数:
        use_gpu: 是否使用 GPU，若为 None 则自动检测。
        show_log: 是否显示日志。
    返回:
        PaddleOCR 实例
    """
    print("🚀 正在初始化 OCR 模型...")
    if use_gpu is None:
        use_gpu = paddle.is_compiled_with_cuda()
    ocr = PaddleOCR(
        use_angle_cls=False,
        det_algorithm="DB",  # 使用DB算法处理小文字
        det_max_side_len=480,                    # 增大最大边长以捕捉小文字
        det_score_mode="Fast",  # 提高检测精度
        det_db_thresh=0.6,     # 降低检测阈值（针对模糊文字）
        det_db_box_thresh=0.6, # 降低检测框阈值
        lang='ch',
        # rec_algorithm="CRNN",
        rec_batch_num=2,       # 减小批处理数量提高小图精度
        # rec_image_shape="3, 48, 360",  # 针对小文字调整输入尺寸
        use_gpu=use_gpu,
        det=True,
        rec=True,
        rec_score_thresh=0.6,
        cpu_threads=4,
        # rec_batch_num=2,
        gpu_mem=1024,
        use_tensorrt=True,
        use_fp16=True,
        # enable_mkldnn=True,
        # use_gpu_threads=4,
        show_log=show_log,
        # rec_batch_num=16,
    )
    return ocr
def init_ocr_model_2(use_gpu=None, show_log=False):
    """
    初始化 PaddleOCR 模型。
    参数:
        use_gpu: 是否使用 GPU，若为 None 则自动检测。
        show_log: 是否显示日志。
    返回:
        PaddleOCR 实例
    """
    print("🚀 正在初始化 OCR 模型...")
    if use_gpu is None:
        use_gpu = paddle.is_compiled_with_cuda()
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='ch',
        det_db_thresh=0.05,
        det_db_box_thresh=0.05,
        det_db_unclip_ratio=2.0,
        det_limit_side_len=1920,
        det_limit_type='max',
        show_log=False,
        # cls_thresh=0.5,
    )
    
    
    # 关键参数调整
    # ocr = PaddleOCR(
    #     # 使用针对小文字优化的模型
    #     det_model_dir='ch_PP-OCRv4_det_infer',  # 推荐使用v4检测模型
    #     rec_model_dir='ch_PP-OCRv4_rec_infer',  # 推荐使用v4识别模型
        
    #     # 检测参数优化
    #     det_algorithm="DB",  # 使用DB算法处理小文字
    #     det_max_side_len=960,  # 增大最大边长以捕捉小文字
    #     det_score_mode="slow",  # 提高检测精度
    #     det_db_thresh=0.3,     # 降低检测阈值（针对模糊文字）
    #     det_db_box_thresh=0.5, # 降低检测框阈值
        
    #     # 识别参数优化
    #     rec_algorithm="CRNN",
    #     rec_char_type="ch",    # 中文识别
    #     rec_char_dict_path="ppocr/utils/ppocr_keys_v1.txt",  # 使用完整字典
    #     rec_batch_num=3,       # 减小批处理数量提高小图精度
    #     rec_image_shape="3, 32, 320",  # 针对小文字调整输入尺寸
        
    #     # 通用设置
    #     use_angle_cls=True,
    #     lang='ch',
    #     use_gpu=False,
    #     show_log=False,
    #     use_tensorrt=True,
    #     use_fp16=True,
    #     gpu_mem=1024
    # )
    
    print("✅ 医疗专用OCR模型初始化完成")
    print(f"✅ OCR 模型初始化完成，使用{'GPU' if use_gpu else 'CPU'}")
    if not use_gpu:
        print("⚠️ 当前使用 CPU 模式，如需 GPU 请安装 paddlepaddle-gpu")
    return ocr

A=init_db() 
print("数据库连接成功")
B_1=init_ocr_model()
B_2=init_ocr_model()
B_3=init_ocr_model()
B_4=init_ocr_model()
B_5=init_ocr_model()
B_6=init_ocr_model()
B_7=init_ocr_model_2()
B_8=init_ocr_model_2()
print("OCR模型初始化成功")
# xxx=[]
# print(xxx[1])
# close_db()
# print("数据库连接已关闭")


import os
import sys
import time
# import torch
import yaml
print("成功")
# from model.UNetFormer_0329 import UNetFormer
# from ultralytics import YOLO
# from run_realtime_detection import run_realtime_detection
from run_realtime_detection_yolo_new_3 import run_realtime_detection
from src.identification.Recog import PharmaceuticalBottleClassifier
# from src.utils.init_utils import A,B
from src.identification.OCRRecognizer import OCRRecognizer,OCRRecognizer_ori
from src.identification.DrugMatcher import DrugMatcher
from src.utils.single_img_saver import img_shower

sys.path.append("D:/ZJF/种植作物分类/ZhongzhiCodes/XiuGai1")

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


if __name__ == "__main__":
    # 读取配置文件参数

    with open("./config.yaml",'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ============ 设备 ============
    device = 'cpu'
    # print(f"🚀 使用设备: {device}")

    # if device.type == 'cuda':
    #     print(f"   GPU: {torch.cuda.get_device_name(0)}")
    #     print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    #     # 优化CUDA设置
    #     torch.backends.cudnn.benchmark = True
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True

    # ============ 创建模型 ============
    print("\n🔧 创建模型...")

    # Unet模型
    # model = UNetFormer(num_classes=config['model']['num_classes'], pretrained=False)
    # print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # yolo模型
    model = None

    # ============ 连接数据库 ============
    # 连接数据库
    db_config = config['db_config']
    conn = None
    try:
        conn = A # 传递 host, user, password, database
        print("数据库连接成功")

    except Exception as e:
        print(f"数据库连接发生错误: {e}")

    # 初始化需要的组件类
    # ocr_model = 
    ocr_recognizer_1 = OCRRecognizer_ori(B_1)
    ocr_recognizer_2 = OCRRecognizer_ori(B_2)
    ocr_recognizer_3 = OCRRecognizer_ori(B_3)
    ocr_recognizer_4 = OCRRecognizer_ori(B_4)
    ocr_recognizer_5 = OCRRecognizer_ori(B_5)
    ocr_recognizer_6 = OCRRecognizer_ori(B_6)
    ocr_recognizer_7 = OCRRecognizer_ori(B_7)
    ocr_recognizer_8 = OCRRecognizer_ori(B_8)
    drug_matcher = DrugMatcher(conn, drug_table=config['table_config']['drug_table'], drug_column=config['table_config']['drug_column'],
                               patient_table=config['table_config']['patient_table'], patient_column=config['table_config']['patient_column'], cache_drugs=True)
    classifier = PharmaceuticalBottleClassifier(db_conn=conn, device=device)



    # ============ 运行 ============
    print("\n" + "=" * 60)
    print("🚀 启动实时视频分割 (深度GPU批量加速)")
    print("=" * 60)
    # RTSP 参数
    print("RTSP 参数:")
    print(config['RTSP'])
    # 主运行函数
    run_realtime_detection(
        
        username=config['RTSP']['username'],
        password=config['RTSP']['password'],
        ip_address=config['RTSP']['ip_address'],
        port=config['RTSP']['port'],
        channel=config['RTSP']['channel'],

        # 切割模型参数
        model=model,
        checkpoint_path=config['model']['checkpoint_path'],
        num_classes=config['model']['num_classes'],

        # 模型组件
        ocr_recognizer=[ocr_recognizer_1,ocr_recognizer_2,ocr_recognizer_3,ocr_recognizer_4,ocr_recognizer_5,ocr_recognizer_6,ocr_recognizer_7,ocr_recognizer_8],
        drug_matcher=drug_matcher,
        classifier = classifier,
        length = 3,

        # 分割器参数
        tile_size=config['segmentor']['tile_size'],
        overlap=config['segmentor']['overlap'],
        target_fps=config['segmentor']['target_fps'],
        batch_frames=config['segmentor']['batch_frames'],

        # 显示参数
        output_type=config['display']['output_type'],
        overlay_alpha=config['display']['overlay_alpha'],
        display_scale=config['display']['display_scale'],

        # 保存参数
        save_video=config['saving']['save_video'],
        output_path=config['saving']['output_path'],
        save_fps=config['saving']['save_fps'],

        # 设备
        device=device,
        # 裁剪参数
        # crop_region=config['crop_region']
    )
    # close_db(conn)
    print("数据库已断开连接√")