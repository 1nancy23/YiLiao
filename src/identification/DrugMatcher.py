import pymysql
from fuzzywuzzy import fuzz, process
from typing import List, Optional, Dict, Any

from src.identification.OCRRecognizer import OCRRecognizer


class DrugMatcher:
    """
    药品/患者名称模糊匹配器。
    职责：从数据库加载药品名称或患者名称列表，对查询字符串进行模糊匹配，
         并提供完整的药品识别业务流程。
    """

    def __init__(self, db_conn, drug_table='drugs', drug_column='medicine_name',
                 patient_table='patients', patient_column='name', cache_drugs=True):
        """
        :param db_conn: 数据库连接对象（应为 DictCursor 连接）
        :param table: 药品表名
        :param column: 药品名称列名
        :param patient_table: 患者表名
        :param patient_column: 患者名称列名
        :param cache_drugs: 是否在初始化时缓存药品列表（可提高匹配速度）
        """
        self.conn = db_conn
        self.drug_table = drug_table
        self.drug_column = drug_column
        self.patient_table = patient_table
        self.patient_column = patient_column
        self.cache_drugs = cache_drugs

        # 药品名称缓存
        self._drug_names = None
        # 患者名称缓存
        self._patient_names = None

        if cache_drugs:
            self._load_drug_names()
            self._load_patient_names()
            self._patient_names.append("赵二虎")

    def _load_drug_names(self):
        """从数据库加载所有药品名称到内存"""
        drug_names = []
        try:
            with self.conn.cursor() as cursor:
                sql = f"SELECT {self.drug_column} FROM {self.drug_table}"
                cursor.execute(sql)
                results = cursor.fetchall()
                for row in results:
                    if isinstance(row, dict):
                        drug_names.append(row[self.drug_column])
                    else:
                        drug_names.append(row[0] if row else None)
        except pymysql.MySQLError as e:
            raise Exception(f"数据库查询失败：{e}")

        self._drug_names = [name for name in drug_names if name]  # 过滤空值
        print(f"✅ 已加载 {len(self._drug_names)} 个药品名称")

    def _load_patient_names(self):
        """从数据库加载所有患者名称到内存"""
        patient_names = []
        try:
            with self.conn.cursor() as cursor:
                sql = f"SELECT {self.patient_column} FROM {self.patient_table}"
                cursor.execute(sql)
                results = cursor.fetchall()
                for row in results:
                    if isinstance(row, dict):
                        patient_names.append(row[self.patient_column])
                    else:
                        patient_names.append(row[0] if row else None)
        except pymysql.MySQLError as e:
            raise Exception(f"数据库查询失败：{e}")

        self._patient_names = [name for name in patient_names if name]  # 过滤空值
        print(f"✅ 已加载 {len(self._patient_names)} 个患者名称")
        print("所有患者",self._patient_names)
    def match(self, query: str, match_type: str = 'bottle',
              threshold=80, limit: Optional[int] = None) -> List[str]:
        """
        对查询字符串进行模糊匹配，返回匹配到的名称列表。

        :param query: 查询字符串
        :param match_type: 匹配类型，'bottle'=药品匹配，'bag'=患者匹配
        :param threshold: 匹配阈值 (0-100)
        :param limit: 返回最大数量
        :return: 匹配到的名称列表
        """
        if not query or len(query.strip()) < 1:
            return []

        # 根据匹配类型加载对应的名称列表
        if match_type == 'bottle':
            # 药品匹配
            if self._drug_names is None: 
                self._load_drug_names()
            name_list = self._drug_names
            print(f"🔍 药品匹配模式，查询：{query}")
            ###测试全部返回
            # return self._drug_names
        elif match_type == 'bag':
            # 患者匹配
            if self._patient_names is None:
                self._load_patient_names()
            name_list = self._patient_names
            print(f"🔍 患者匹配模式，查询：{query}")
        else:
            raise ValueError(f"无效的 match_type: {match_type}，应为 'bottle' 或 'bag'")

        if not name_list:
            print("⚠️ 名称列表为空，无法匹配，返回所有药品名称")
            return self._drug_names

        matches = process.extractBests(
            query,
            name_list,
            scorer=fuzz.partial_ratio,
            score_cutoff=threshold,
            limit=limit
        )
        if match_type == 'bottle':
            if not matches:
                print("⚠️ 未找到匹配结果,返回所有药品名称")
                return self._drug_names
        return [match[0] for match in matches]

    def refresh_cache(self, match_type: str = 'bottle'):
        """
        手动刷新缓存（当数据库有更新时调用）

        :param match_type: 刷新类型，'bottle'=药品缓存，'bag'=患者缓存，'all'=全部
        """
        if match_type == 'bottle':
            self._load_drug_names()
        elif match_type == 'bag':
            self._load_patient_names()
        elif match_type == 'all':
            self._load_drug_names()
            self._load_patient_names()
        else:
            raise ValueError(f"无效的 match_type: {match_type}")

    # ========== 业务流程方法 ==========

    def recognize_single_bottle(self, ocr_recognizer: OCRRecognizer, image,
                                threshold=50, limit=10) -> Dict[str, Any]:
        """
        识别单个药瓶图像：OCR 识别 + 药品模糊匹配。
        :param ocr_recognizer: OCR 识别器实例
        :param image: 图像（numpy 数组或文件路径）
        :param threshold: 模糊匹配阈值
        :param limit: 最大返回匹配数
        :return: 字典 {
            "image": 原始图像,
            "ocr_text": 识别出的文字,
            "matches": [匹配到的药品名称]
        }
        """
        # OCR 识别
        query = ocr_recognizer.recognize(image)

        if not query or len(query.strip()) < 3:
            print("⚠️ OCR 结果为空或过短，跳过匹配")
            return {
                "image": image,
                "ocr_text": "",
                "matches": []
            }

        print(f'OCR 识别结果：{query}')
        # 药品模糊匹配（match_type='bottle'）
        matches = self.match(query, match_type='bottle', threshold=threshold, limit=limit)
        if matches:
            print(f"匹配结果数量：{len(matches)}")
            print(matches)
        else:
            print("匹配失败，请重新拍摄")

        return {
            "image": image,
            "ocr_text": query,
            "matches": matches
        }

    def recognize_single_bag(self, ocr_recognizer: OCRRecognizer, image,
                             threshold=50, limit=10) -> Dict[str, Any]:
        """
        识别单个药袋图像：OCR 识别 + 患者模糊匹配。
        :param ocr_recognizer: OCR 识别器实例
        :param image: 图像（numpy 数组或文件路径）
        :param threshold: 模糊匹配阈值
        :param limit: 最大返回匹配数
        :return: 字典 {
            "image": 原始图像,
            "ocr_text": 识别出的文字,
            "matches": [匹配到的患者名称]
        }
        """
        # OCR 识别
        query = ocr_recognizer.recognize(image)
        if not query or len(query.strip()) < 2:
            print("⚠️ OCR 结果为空或过短，跳过匹配")
            return {
                "image": image,
                "ocr_text": "",
                "matches": []
            }

        print(f'OCR 识别结果：{query}')
        # 患者模糊匹配（match_type='bag'）
        matches = self.match(query, match_type='bag', threshold=threshold, limit=limit)
        if matches:
            print(f"匹配结果数量：{len(matches)}")
            print(matches)
        else:
            print("匹配失败，请重新拍摄")

        return {
            "image": image,
            "ocr_text": query,
            "matches": matches
        }

    def process_bottle_recognition(self, ocr_recognizer: OCRRecognizer,
                                   image_list, threshold=50, limit=10) -> List[Dict[str, Any]]:
        """
        批量处理药瓶图像列表。
        :param ocr_recognizer: OCR 识别器实例
        :param image_list: 图像列表
        :param threshold: 模糊匹配阈值
        :param limit: 最大返回匹配数
        :return: 列表，每个元素为 recognize_single_bottle 的返回结果
        """
        all_results = []
        total = len(image_list)

        for idx, img in enumerate(image_list):
            print(f"--------- 处理第 {idx + 1}/{total} 号图像 ---------")
            result = self.recognize_single_bottle(
                ocr_recognizer, img,
                threshold=threshold, limit=limit
            )
            all_results.append(result)

        return all_results

    def process_bag_recognition(self, ocr_recognizer: OCRRecognizer,
                                image_list, threshold=50, limit=10) -> List[Dict[str, Any]]:
        """
        批量处理药袋图像列表。
        :param ocr_recognizer: OCR 识别器实例
        :param image_list: 图像列表
        :param threshold: 模糊匹配阈值
        :param limit: 最大返回匹配数
        :return: 列表，每个元素为 recognize_single_bag 的返回结果
        """
        all_results = []
        total = len(image_list)

        for idx, img in enumerate(image_list):
            print(f"--------- 处理第 {idx + 1}/{total} 号图像 ---------")
            result = self.recognize_single_bag(
                ocr_recognizer, img,
                threshold=threshold, limit=limit
            )
            all_results.append(result)

        return all_results

    def check_patient_batch_medicines(self, patient_name: str, batch_id: int,
                                      expected_medicine_names: List[str]) -> Dict[str, Any]:
        """
        验证指定病人、指定批次所需的药品是否与预期列表一致。

        :param patient_name: 病人姓名
        :param batch_id: 批次 ID
        :param expected_medicine_names: 预期的药品名称列表
        :return: dict {
            'matched': bool,
            'actual': list,
            'missing': list,
            'extra': list,
            'patient_id': int or None,
            'batch_exists': bool
        }
        """
        
        print("利用“魏理想”作为测试病人")
        patient_name = "魏理想"
        result = {
            'matched': False,
            'actual': [],
            'missing': [],
            'extra': [],
            'patient_id': None,
            'batch_exists': False
        }

        with self.conn.cursor() as cursor:
            # 1. 根据病人名称查询 patient_id
            cursor.execute("SELECT patient_id FROM patients WHERE name = %s", (patient_name,))
            row = cursor.fetchone()
            if not row:
                print(f"错误：未找到名为 '{patient_name}' 的病人")
                return result

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
    from src.utils.init_utils import init_db, init_ocr_model

    # 1. 初始化依赖
    db_config = {
        'host': '127.0.0.1',
        'user': 'root',
        'password': 'root',
        'database': 'medicine_db',
        'charset': 'utf8mb4',
    }
    conn = init_db(**db_config)
    ocr_model = init_ocr_model()

    # 2. 创建识别器和匹配器
    recognizer = OCRRecognizer(ocr_model)
    matcher = DrugMatcher(conn, cache_drugs=True)

    # 3. 药瓶识别（药品匹配）
    image_path = "./data/img_7.png"
    result = matcher.recognize_single_bottle(recognizer, image_path, threshold=50, limit=10)
    print(f"【药瓶】OCR 结果：{result['ocr_text']}")
    print(f"【药瓶】匹配结果：{result['matches']}")
    print('='*50)

    # 4. 药袋识别（患者匹配）
    bag_image_path = "./data/bag_1773497363014.jpg"
    bag_result = matcher.recognize_single_bag(recognizer, bag_image_path, threshold=50, limit=10)
    print(f"【药袋】OCR 结果：{bag_result['ocr_text']}")
    print(f"【药袋】匹配结果：{bag_result['matches']}")

    patient_name = bag_result['matches']
    # 5. 直接使用 match 函数
    # drug_matches = matcher.match("艾司奥美", match_type='bottle', threshold=50, limit=5)
    # print(f"药品匹配：{drug_matches}")
    #
    # patient_matches = matcher.match("魏理", match_type='bag', threshold=50, limit=5)
    # print(f"患者匹配：{patient_matches}")

    # 6. 刷新缓存
    # matcher.refresh_cache('bottle')  # 只刷新药品
    # matcher.refresh_cache('bag')     # 只刷新患者
    # matcher.refresh_cache('all')     # 刷新全部

    validation = matcher.check_patient_batch_medicines(
        patient_name=patient_name,
        batch_id=1,
        expected_medicine_names=['注射用艾司奥拉美拉唑钠']
    )

    if validation['batch_exists']:
        print(f"数据库所需药品: {validation['actual']}")
        if validation['matched']:
            print("✅ 匹配正确：识别药品与所需药品完全一致")
        else:
            print("❌ 匹配错误：")
            if validation['missing']:
                print(f"   缺少药品: {validation['missing']}")
            if validation['extra']:
                print(f"   多余药品: {validation['extra']}")
    else:
        print(f"⚠️ 患者 {patient_name} 批次 {1} 不存在于数据库")

    conn.close()