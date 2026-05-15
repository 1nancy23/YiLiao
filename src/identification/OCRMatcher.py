import cv2
import numpy as np
import re
import time
import pymysql
from fuzzywuzzy import fuzz, process
from src.utils.init_utils import close_db, init_db, init_ocr_model
# 导入图像处理类（请根据实际路径调整）
from src.utils.img_utils import ImageProcessor

class OCRMatcher:
    """
    图像 OCR 识别 + 药品名称模糊匹配类。
    主要方法：
        - recognize_text(image)              # 单张图像文字识别
        - fuzzy_match(conn, query, ...)      # 对单个查询进行模糊匹配
        - process_bottle_recognition(...)    # 批量处理流水线
    """
    def __init__(self, ocr_model):
        """
        初始化匹配器。
        :param ocr_model: 已初始化的 PaddleOCR 实例
        """
        self.ocr_model = ocr_model
        self.image_processor = ImageProcessor()  # 实例化图像处理器

    # ---------- 公有方法：单张图像文字识别 ----------
    def recognize_text(self, image, line_gap_threshold=20):
        """
        识别图像中的文字，返回过滤后的汉字字符串。
        :param image: 图像数组或文件路径
        :param line_gap_threshold: 行间距阈值，用于合并同一行文字
        :return: 识别出的纯汉字字符串，失败返回 None
        """
        start = time.time()
        # 自动旋转校正（使用图像处理类）
        corrected_img = self.image_processor.auto_rotate_image(image)
        if corrected_img is None:
            return None

        try:
            result = self.ocr_model.ocr(corrected_img, det=True, rec=True)
        except Exception as e:
            print(f"⚠️ OCR识别失败: {e}")
            return None

        if not result or not result[0]:
            return None

        # 提取文本块
        text_blocks = []
        for line in result[0]:
            if line and len(line) > 1:
                bbox = line[0]
                text = line[1][0]
                # 只保留汉字
                text = re.sub(r'[^\u4e00-\u9fff]', '', text)
                if not text:
                    continue
                y_coords = [point[1] for point in bbox]
                center_y = sum(y_coords) / 4
                text_blocks.append({'text': text, 'center_y': center_y})

        if not text_blocks:
            return None

        # 按 Y 坐标排序并合并行
        text_blocks.sort(key=lambda x: x['center_y'])
        lines = []
        current_line = [text_blocks[0]['text']]
        prev_y = text_blocks[0]['center_y']

        for i in range(1, len(text_blocks)):
            y = text_blocks[i]['center_y']
            if abs(y - prev_y) > line_gap_threshold:
                lines.append(''.join(current_line))
                current_line = [text_blocks[i]['text']]
            else:
                current_line.append(text_blocks[i]['text'])
            prev_y = y
        lines.append(''.join(current_line))

        print(f'识别用时:{time.time() - start:.3f}s')
        return ''.join(lines)

    # ---------- 公有方法：模糊匹配 ----------
    def fuzzy_match(self, conn, query, table='drugs', column='medicine_name',
                    threshold=80, limit=None):
        """
        对单个查询进行模糊匹配，返回匹配到的药品名称列表。
        :param conn: 数据库连接（应为 DictCursor 连接）
        :param query: 查询字符串
        :param table: 表名
        :param column: 药品名称列名
        :param threshold: 匹配阈值 (0-100)
        :param limit: 返回最大数量
        :return: 匹配到的名称列表
        """
        # 从数据库读取所有药品名称
        drug_names = []
        try:
            with conn.cursor() as cursor:
                sql = f"SELECT {column} FROM {table}"
                cursor.execute(sql)
                results = cursor.fetchall()
                # 兼容 DictCursor 和普通 Cursor
                for row in results:
                    if isinstance(row, dict):
                        drug_names.append(row[column])
                    else:
                        # 假设是元组，取第一个元素
                        drug_names.append(row[0] if row else None)
        except pymysql.MySQLError as e:
            raise Exception(f"数据库查询失败: {e}")

        if not drug_names:
            return []

        # 模糊匹配
        matches = process.extractBests(
            query,
            drug_names,
            scorer=fuzz.partial_ratio,
            score_cutoff=threshold,
            limit=limit
        )
        return [match[0] for match in matches]

    # ---------- 公有方法：批量处理流水线 ----------
    def recognize_single_bottle(self, image, conn, threshold=50, limit=10,
                                table='drugs', column='medicine_name'):
        """
        识别单个药瓶图像：OCR识别 + 模糊匹配。
        :param image: 图像（numpy数组或文件路径）
        :param conn: 数据库连接
        :param threshold: 模糊匹配阈值
        :param limit: 最大返回匹配数
        :param table: 药品表名
        :param column: 药品名称列名
        :return: 字典 {
            "image": 原始图像（如果传入的是路径则为路径，否则保留原对象）,
            "ocr_text": 识别出的文字,
            "matches": [匹配到的药品名称]
        }
        """
        # OCR 识别
        query = self.recognize_text(image)
        if not query or len(query.strip()) < 3:
            print("⚠️ OCR 结果为空或过短，跳过匹配")
            return {
                "image": image,
                "ocr_text": "",
                "matches": []
            }

        print(f'OCR 识别结果：{query}')
        # 模糊匹配
        matches = self.fuzzy_match(
            conn, query,
            table=table, column=column,
            threshold=threshold, limit=limit
        )
        if matches:
            print(f"匹配结果数量: {len(matches)}")
            print(matches)
        else:
            print("匹配失败，请重新拍摄")

        return {
            "image": image,
            "ocr_text": query,
            "matches": matches
        }

    def process_bottle_recognition(self, cropped_img_list, conn,
                                   threshold=50, limit=10,
                                   table='drugs', column='medicine_name'):
        """
        批量处理药瓶图像列表：循环调用 recognize_single_bottle。
        :param cropped_img_list: 图像列表（每个元素为 numpy 数组或文件路径）
        :param conn: 数据库连接
        :param threshold: 模糊匹配阈值
        :param limit: 最大返回匹配数
        :param table: 药品表名
        :param column: 药品名称列名
        :return: 列表，每个元素为 recognize_single_bottle 的返回结果
        """
        all_results = []
        total = len(cropped_img_list)

        for idx, img in enumerate(cropped_img_list):
            print(f"--------- 处理第 {idx + 1}/{total} 号图像 ---------")
            result = self.recognize_single_bottle(
                img, conn,
                threshold=threshold, limit=limit,
                table=table, column=column
            )
            all_results.append(result)

        return all_results

    def check_patient_batch_medicines(self,conn, patient_name, batch_id, expected_medicine_names):
        """
        验证指定病人、指定批次所需的药品是否与预期列表一致。

        :param conn: 数据库连接对象
        :param patient_name: 病人姓名
        :param batch_id: 批次ID
        :param expected_medicine_names: 预期的药品名称列表（列表 of str）
        :return: dict {
            'matched': bool,           # 是否完全匹配
            'actual': list,            # 数据库中实际查询到的药品名称列表
            'missing': list,           # 预期有但实际没有的药品
            'extra': list,             # 实际有但预期没有的药品
            'patient_id': int or None, # 查询到的病人ID（验证用）
            'batch_exists': bool       # 批次是否存在且属于该病人
        }
        """
        result = {
            'matched': False,
            'actual': [],
            'missing': [],
            'extra': [],
            'patient_id': None,
            'batch_exists': False
        }

        with conn.cursor() as cursor:
            # 1. 根据病人名称查询 patient_id
            cursor.execute("SELECT patient_id FROM patients WHERE name = %s", (patient_name,))
            row = cursor.fetchone()
            if not row:
                print(f"错误：未找到名为 '{patient_name}' 的病人")
                return result

            # 兼容元组和字典
            patient_id = row[0] if isinstance(row, (tuple, list)) else row['patient_id']
            result['patient_id'] = patient_id

            # 2. 验证批次是否存在且属于该病人
            cursor.execute("SELECT batch_id FROM batches WHERE batch_id = %s AND patient_id = %s",
                           (batch_id, patient_id))
            if not cursor.fetchone():
                print(f"错误：批次 {batch_id} 不存在或不属于病人 {patient_name}")
                return result
            result['batch_exists'] = True

            # 3. 查询该批次的所有药品名称
            sql = """
                SELECT d.medicine_name
                FROM batch_medicines bm
                JOIN drugs d ON bm.medicine_id = d.id
                WHERE bm.batch_id = %s
            """
            cursor.execute(sql, (batch_id,))
            rows = cursor.fetchall()
            # 提取药品名称列表
            actual_names = [row[0] if isinstance(row, (tuple, list)) else row['medicine_name'] for row in rows]
            result['actual'] = actual_names

            # 4. 比对（忽略顺序，视为集合）
            expected_set = set(expected_medicine_names)
            actual_set = set(actual_names)

            result['missing'] = list(expected_set - actual_set)
            result['extra'] = list(actual_set - expected_set)
            result['matched'] = (expected_set == actual_set)

            return result

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 1. 初始化数据库连接
    db_config = {
        'host': '127.0.0.1',
        'user': 'root',
        'password': 'root',
        'database': 'medicine_db',
        'charset': 'utf8mb4',
    }
    conn = init_db(**db_config)

    # 2. 初始化 OCR 模型
    ocr_model = init_ocr_model()

    # 3. 创建匹配器实例
    matcher = OCRMatcher(ocr_model)

    # 4. 单张图像识别测试
    query = matcher.recognize_text("./data/img_7.png")
    print(query)
    matcher.recognize_single_bottle(cv2.imread("./data/img_7.png"),conn)

    # 5. 批量处理示例（假设有图像列表）
    # img_list = [cv2.imread(f) for f in ["img1.jpg", "img2.jpg"]]
    # results = matcher.process_bottle_recognition(img_list, conn, threshold=50, limit=10)
    # for res in results:
    #     print(f"OCR: {res['ocr_text']}, Matches: {res['matches']}")

    # print(matcher.check_patient_batch_medicines(conn=conn,patient_name='魏理想',batch_id=1,expected_medicine_names=["注射用艾司奥拉美拉唑钠"]))


    # 关闭连接
    close_db(conn)