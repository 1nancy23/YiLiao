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
        use_angle_cls=True,
        lang='ch',
        use_gpu=use_gpu,
        det=True,
        rec=True,
        rec_score_thresh=0.7,
        gpu_mem=1024,
        use_tensorrt=True,
        use_fp16=True,
        use_gpu_threads=4,
        show_log=show_log
    )
    print(f"✅ OCR 模型初始化完成，使用{'GPU' if use_gpu else 'CPU'}")
    if not use_gpu:
        print("⚠️ 当前使用 CPU 模式，如需 GPU 请安装 paddlepaddle-gpu")
    return ocr

A=init_db() 
print("数据库连接成功")
B=init_ocr_model()
# print("OCR模型初始化成功")
# close_db()
# print("数据库连接已关闭")
