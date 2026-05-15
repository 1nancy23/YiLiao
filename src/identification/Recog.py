import cv2
import numpy as np
from PIL import Image
import os
import json
import pickle
import pymysql
import time

class PharmaceuticalBottleClassifier:
    """
    药瓶分类器，支持从MySQL数据库存储和加载特征。
    初始化时一次性加载全部特征到内存，分类时零数据库查询。

    主要接口：
        - save_features_to_db(folder_path, medicine_name=None)   # 单个药品录入
        - save_batch_features_to_db(root_folder)                 # 批量录入
        - classify(medicine_names, image)                        # 单张图片分类
        - reload_features()                                      # 手动刷新内存缓存
    """

    def __init__(self, db_conn, device='cuda'):
        self.conn = db_conn

        # self.sift = cv2.SIFT_create(nfeatures=320,contrastThreshold=0.02,nOctaveLayers=3,)
        self.sift = cv2.ORB_create(nfeatures=160)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
        search_params = dict(checks=20)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        # self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._init_db_table()

        # ========== 核心优化：初始化时一次性加载全部特征到内存 ==========
        self._templates_cache = {}   # {medicine_name: [{'desc_sift': np.array}, ...]}
        self._deep_avg_cache = {}    # {medicine_name: np.array or None}
        self._load_all_features()
        print(f"[初始化完成] 已加载 {len(self._templates_cache)} 种药品特征到内存")

    # ---------- 私有辅助方法 ----------
    def _preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _extract_sift_features(self, image):
        # print(image.shape)
        # image = cv2.resize(image, (224, 224))
        # print(image.shape)
        return self.sift.detectAndCompute(image, None)

    def _extract_features_from_image(self, image):
        print("image",image.shape)
        kp, desc = self._extract_sift_features(image)
        return kp[:320], desc[:320]

    def _match_sift_features(self, desc1, desc2):
        # print("desc1",desc1.shape)
        # print("desc2",desc2.shape)
        
        if desc1 is None or desc2 is None:
            return 0
        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good = 0
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good += 1
            return good
        except Exception:
            return 0

    def _init_db_table(self):
        with self.conn.cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drugs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    medicine_name VARCHAR(255) NOT NULL UNIQUE,
                    sift1 MEDIUMBLOB,
                    sift2 MEDIUMBLOB,
                    sift3 MEDIUMBLOB,
                    sift4 MEDIUMBLOB,
                    sift5 MEDIUMBLOB,
                    sift6 MEDIUMBLOB,
                    deep_avg MEDIUMBLOB
                )
            ''')
        self.conn.commit()

    # ========== 核心优化：一次性加载全部特征 ==========
    def _load_all_features(self):
        """从数据库一次性加载全部药品特征到内存"""
        query = '''
            SELECT medicine_name, sift1, sift2, sift3, sift4, sift5, sift6, deep_avg
            FROM drugs
        '''
        with self.conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

        self._templates_cache.clear()
        self._deep_avg_cache.clear()

        for row in rows:
            # 兼容 DictCursor 和普通 Cursor
            if isinstance(row, dict):
                name = row['medicine_name']
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                sift_blobs = [row[f'sift{i}'] for i in range(1, 7)]
                deep_blob = row['deep_avg']
            else:
                name = row[0]
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                sift_blobs = list(row[1:7])
                deep_blob = row[7]

            # 解析 SIFT 特征
            sift_list = []
            for blob in sift_blobs:
                if blob is not None:
                    try:
                        desc = pickle.loads(blob)
                        sift_list.append({'desc_sift': desc})
                    except Exception:
                        pass
            self._templates_cache[name] = sift_list

            # 解析深度特征
            if deep_blob is not None:
                try:
                    self._deep_avg_cache[name] = pickle.loads(deep_blob)
                except Exception:
                    self._deep_avg_cache[name] = None
            else:
                self._deep_avg_cache[name] = None

    def reload_features(self):
        """公共接口：手动刷新内存缓存（数据库有外部变更时调用）"""
        self._load_all_features()
        print(f"[缓存刷新] 已重新加载 {len(self._templates_cache)} 种药品特征")

    # ---------- 公共接口：特征录入 ----------
    def save_features_to_db(self, folder_path, medicine_name=None):
        if medicine_name is None:
            medicine_name = os.path.basename(folder_path.rstrip('/\\'))

        image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])[:6]

        if not image_files:
            return

        sift_list = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imdecode(
                np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if img is None:
                continue
            kp, desc = self._extract_features_from_image(img)
            sift_list.append(pickle.dumps(desc))

        while len(sift_list) < 6:
            sift_list.append(None)

        with self.conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO drugs 
                    (medicine_name, sift1, sift2, sift3, sift4, sift5, sift6, deep_avg)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    sift1 = VALUES(sift1),
                    sift2 = VALUES(sift2),
                    sift3 = VALUES(sift3),
                    sift4 = VALUES(sift4),
                    sift5 = VALUES(sift5),
                    sift6 = VALUES(sift6),
                    deep_avg = VALUES(deep_avg)
            ''', (medicine_name, *sift_list, None))
        self.conn.commit()

        # ========== 录入后自动刷新缓存 ==========
        self._update_single_cache(medicine_name, sift_list)

    def save_batch_features_to_db(self, root_folder):
        if not os.path.isdir(root_folder):
            return

        for item in os.listdir(root_folder):
            sub_path = os.path.join(root_folder, item)
            if os.path.isdir(sub_path):
                self.save_features_to_db(sub_path, medicine_name=item)

        print(f"[批量录入完成] 当前缓存共 {len(self._templates_cache)} 种药品")

    def _update_single_cache(self, medicine_name, sift_blob_list):
        """录入单个药品后，直接更新内存缓存（避免全量重载）"""
        sift_templates = []
        for blob in sift_blob_list:
            if blob is not None:
                try:
                    desc = pickle.loads(blob)
                    sift_templates.append({'desc_sift': desc})
                except Exception:
                    pass
        self._templates_cache[medicine_name] = sift_templates
        self._deep_avg_cache[medicine_name] = None

    # ---------- 公共接口：分类（零数据库查询） ----------
    def classify(self, medicine_names, image):
        """
        对单张图片进行分类，仅考虑指定的药品名称列表。
        直接从内存缓存读取特征，不再查询数据库。

        :param medicine_names: 待匹配的药品名称列表
        :param image: OpenCV图像（BGR格式）
        :return: 字典，包含预测类别、置信度、所有得分等
        """
        start_time = time.time()
        # ========== 直接从内存缓存过滤，无DB查询 ==========
        templates_dict = {}
        missing = []
        for name in medicine_names:
            if name in self._templates_cache and self._templates_cache[name]:
                templates_dict[name] = self._templates_cache[name]
            else:
                missing.append(name)

        if missing:
            print(f"[警告] 以下药品未在缓存中找到: {missing}")

        if not templates_dict:
            raise ValueError("未找到任何指定药品的特征，无法分类")
        start_time2 = time.time()
        _, desc_query = self._extract_sift_features(image)
        end_time2 = time.time()
        print(f"[特征提取耗时] {end_time2 - start_time2:.4f} 秒")
        scores = {}
        details = {}
        
        for name, templates in templates_dict.items():
            sift_scores = [
                self._match_sift_features(t['desc_sift'], desc_query)
                for t in templates
            ]
            max_sift = max(sift_scores) if sift_scores else 0
            sift_conf = min(1.0, max_sift / 15.0)

            final = sift_conf * 0.6
            scores[name] = final
            details[name] = {
                'max_sift_score': max_sift,
                'sift_confidence': sift_conf,
            }

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(f"[分类耗时] {time.time() - start_time:.4f} 秒")
        return {
            'predicted_category': sorted_items[0][0],
            'confidence': sorted_items[0][1],
            'all_scores': dict(sorted_items),
            'details': details,
            'top_3': sorted_items[:3],
        }

    def get_cached_names(self):
        """返回当前内存缓存中所有药品名称列表"""
        return list(self._templates_cache.keys())


# ==================== 使用示例 ====================
if __name__ == "__main__":
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        database='medicine_db',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.Cursor
    )

    # 创建分类器 → 自动加载全部特征到内存
    classifier = PharmaceuticalBottleClassifier(db_conn=conn, device='cuda')

    # 查看已缓存的药品
    print(f"已缓存药品: {classifier.get_cached_names()}")

    # 批量录入（录入后自动更新缓存）
    # classifier.save_batch_features_to_db(os.path.abspath("./feat_data"))

    # 分类（直接走内存，零DB查询）
    candidate_names = [
        "人免疫球蛋白",
        "注射用人干扰素a2a",
        "注射用人干扰素a2b",
        "注射用哌拉西林钠他唑巴坦钠",
        "注射用头孢他啶",
        "注射用头孢哌酮钠舒巴坦钠（2：1）",
        "注射用头孢唑啉钠",
        "注射用头孢曲松钠",
        "注射用泮托拉唑钠",
        "注射用炎琥宁",
        "注射用艾司奥拉美拉唑钠",
        "破伤风人免疫球蛋白",
        "重组人血小板生成素注射液",
    ]

    img = cv2.imread("./data/bottle_1773990086441.jpg")
    if img is not None:
        img = cv2.resize(img, (224, 224))
        feat_result = classifier.classify(candidate_names, img)
        best_match = feat_result['predicted_category']
        confidence = feat_result['confidence']
        print(f"  特征匹配最佳: {best_match} (置信度: {confidence:.4f})")

    conn.close()


# import cv2
# import numpy as np
# import os
# import pickle
# import pymysql
# import time
# import threading
# from collections import defaultdict


# class PharmaceuticalBottleClassifier:
#     """
#     药瓶分类器，支持从 MySQL 数据库存储和加载 SIFT 特征。

#     优化版特点：
#         1. 初始化时一次性加载全部药品 SIFT 特征到内存；
#         2. 分类时零数据库查询；
#         3. 分类时不再逐个模板循环匹配；
#         4. 将候选药品的所有模板 SIFT 特征一次性加入 FLANN；
#         5. 当前实时图片只执行一次 knnMatch；
#         6. 根据 good matches 投票判断最像的模板和类别。

#     主要接口：
#         - save_features_to_db(folder_path, medicine_name=None)
#         - save_batch_features_to_db(root_folder)
#         - classify(medicine_names, image)
#         - classify_fast(medicine_names, image)
#         - reload_features()
#         - get_cached_names()
#     """

#     def __init__(self, db_conn, device='cuda'):
#         self.conn = db_conn
#         self.device = device

#         # --------------------------------------------------
#         # SIFT 初始化
#         # --------------------------------------------------
#         # nfeatures 限制最多保留的关键点数量，可以明显提升速度。
#         # 如果你更追求速度，可以改成 200 或 300。
#         # 如果你更追求准确率，可以改成 500 或 800。
#         self.sift = cv2.SIFT_create(
#             nfeatures=80,
#             contrastThreshold=0.04,
#             edgeThreshold=10,
#             sigma=1.6
#         )

#         # --------------------------------------------------
#         # FLANN 参数
#         # --------------------------------------------------
#         self.FLANN_INDEX_KDTREE = 1

#         self.index_params = dict(
#             algorithm=self.FLANN_INDEX_KDTREE,
#             trees=5
#         )

#         # checks 越小越快，但精度略低。
#         # 16 快，32 折中，50 较稳。
#         self.search_params = dict(
#             checks=32
#         )

#         # --------------------------------------------------
#         # 缓存和锁
#         # --------------------------------------------------
#         self._templates_cache = {}
#         # 格式:
#         # {
#         #   medicine_name: [
#         #       {
#         #           'template_index': 0,
#         #           'desc_sift': np.ndarray, shape=(N,128)
#         #       },
#         #       ...
#         #   ]
#         # }

#         self._deep_avg_cache = {}

#         self._cache_lock = threading.Lock()

#         # --------------------------------------------------
#         # 初始化数据库表并加载缓存
#         # --------------------------------------------------
#         self._init_db_table()
#         self._load_all_features()

#         print(f"[初始化完成] 已加载 {len(self._templates_cache)} 种药品特征到内存")

#     # ======================================================
#     # 图像预处理
#     # ======================================================

#     def _preprocess_image(self, image):
#         """
#         统一预处理：
#             1. resize 到 224×224；
#             2. 转灰度；
#             3. CLAHE 对比度增强。

#         注意：
#             录入模板和实时分类都必须使用同样的预处理流程。
#         """

#         if image is None:
#             return None

#         image = cv2.resize(image, (224, 224))

#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image

#         clahe = cv2.createCLAHE(
#             clipLimit=2.0,
#             tileGridSize=(8, 8)
#         )

#         gray = clahe.apply(gray)

#         return gray

#     def _extract_sift_features(self, image):
#         """
#         提取 SIFT 特征。

#         返回:
#             kp, desc

#         desc 的 shape:
#             (关键点数量, 128)

#         注意：
#             关键点数量不是固定的；
#             128 是固定的 SIFT 描述符维度。
#         """

#         gray = self._preprocess_image(image)

#         if gray is None:
#             return [], None

#         kp, desc = self.sift.detectAndCompute(gray, None)

#         if desc is not None:
#             desc = np.asarray(desc, dtype=np.float32)

#         return kp, desc

#     def _extract_features_from_image(self, image):
#         kp, desc = self._extract_sift_features(image)
#         return kp, desc

#     # ======================================================
#     # 数据库表初始化
#     # ======================================================

#     def _init_db_table(self):
#         """
#         初始化数据库表。
#         """

#         with self.conn.cursor() as cursor:
#             cursor.execute('''
#                 CREATE TABLE IF NOT EXISTS drugs (
#                     id INT AUTO_INCREMENT PRIMARY KEY,
#                     medicine_name VARCHAR(255) NOT NULL UNIQUE,
#                     sift1 MEDIUMBLOB,
#                     sift2 MEDIUMBLOB,
#                     sift3 MEDIUMBLOB,
#                     sift4 MEDIUMBLOB,
#                     sift5 MEDIUMBLOB,
#                     sift6 MEDIUMBLOB,
#                     deep_avg MEDIUMBLOB
#                 )
#             ''')

#         self.conn.commit()

#     # ======================================================
#     # 从数据库一次性加载全部特征到内存
#     # ======================================================

#     def _load_all_features(self):
#         """
#         从数据库一次性加载全部药品特征到内存。
#         分类时不再查询数据库。
#         """

#         query = '''
#             SELECT medicine_name, sift1, sift2, sift3, sift4, sift5, sift6, deep_avg
#             FROM drugs
#         '''

#         with self.conn.cursor() as cursor:
#             cursor.execute(query)
#             rows = cursor.fetchall()

#         self._templates_cache.clear()
#         self._deep_avg_cache.clear()

#         for row in rows:

#             # --------------------------------------------------
#             # 兼容 DictCursor 和普通 Cursor
#             # --------------------------------------------------
#             if isinstance(row, dict):
#                 name = row['medicine_name']

#                 if isinstance(name, bytes):
#                     name = name.decode('utf-8')

#                 sift_blobs = [
#                     row.get(f'sift{i}', None)
#                     for i in range(1, 7)
#                 ]

#                 deep_blob = row.get('deep_avg', None)

#             else:
#                 name = row[0]

#                 if isinstance(name, bytes):
#                     name = name.decode('utf-8')

#                 sift_blobs = list(row[1:7])
#                 deep_blob = row[7]

#             # --------------------------------------------------
#             # 解析 SIFT 特征
#             # --------------------------------------------------
#             sift_list = []

#             for i, blob in enumerate(sift_blobs):
#                 if blob is None:
#                     continue

#                 try:
#                     desc = pickle.loads(blob)

#                     if desc is None:
#                         continue

#                     desc = np.asarray(desc, dtype=np.float32)

#                     if len(desc.shape) != 2:
#                         continue

#                     if desc.shape[1] != 128:
#                         continue

#                     if desc.shape[0] < 2:
#                         continue

#                     sift_list.append({
#                         'template_index': i,
#                         'desc_sift': desc
#                     })

#                 except Exception as e:
#                     print(f"[警告] 解析药品 {name} 的 sift{i + 1} 失败: {e}")

#             self._templates_cache[name] = sift_list

#             # --------------------------------------------------
#             # 解析 deep_avg，当前分类没用，但保留
#             # --------------------------------------------------
#             if deep_blob is not None:
#                 try:
#                     self._deep_avg_cache[name] = pickle.loads(deep_blob)
#                 except Exception:
#                     self._deep_avg_cache[name] = None
#             else:
#                 self._deep_avg_cache[name] = None

#     def reload_features(self):
#         """
#         手动刷新内存缓存。
#         当数据库中的特征被外部程序修改后，可调用该函数。
#         """

#         with self._cache_lock:
#             self._load_all_features()

#         print(f"[缓存刷新] 已重新加载 {len(self._templates_cache)} 种药品特征")

#     # ======================================================
#     # 特征录入
#     # ======================================================

#     def save_features_to_db(self, folder_path, medicine_name=None):
#         """
#         将单个药品文件夹中的图片提取 SIFT 特征并保存到数据库。

#         folder_path:
#             某个药品的图片文件夹

#         medicine_name:
#             药品名称。如果不传，则使用文件夹名称。
#         """

#         if medicine_name is None:
#             medicine_name = os.path.basename(folder_path.rstrip('/\\'))

#         image_files = sorted([
#             f for f in os.listdir(folder_path)
#             if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))
#         ])[:6]

#         if not image_files:
#             print(f"[警告] 文件夹中没有图片: {folder_path}")
#             return

#         sift_blob_list = []

#         for img_file in image_files:
#             img_path = os.path.join(folder_path, img_file)

#             img = cv2.imdecode(
#                 np.fromfile(img_path, dtype=np.uint8),
#                 cv2.IMREAD_COLOR
#             )

#             if img is None:
#                 print(f"[警告] 无法读取图片: {img_path}")
#                 continue

#             kp, desc = self._extract_features_from_image(img)

#             if desc is not None:
#                 desc = np.asarray(desc, dtype=np.float32)

#             print(
#                 f"[录入] 药品: {medicine_name}, 图片: {img_file}, "
#                 f"SIFT shape: {None if desc is None else desc.shape}"
#             )

#             sift_blob_list.append(pickle.dumps(desc))

#         while len(sift_blob_list) < 6:
#             sift_blob_list.append(None)

#         with self.conn.cursor() as cursor:
#             cursor.execute('''
#                 INSERT INTO drugs 
#                     (medicine_name, sift1, sift2, sift3, sift4, sift5, sift6, deep_avg)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#                 ON DUPLICATE KEY UPDATE
#                     sift1 = VALUES(sift1),
#                     sift2 = VALUES(sift2),
#                     sift3 = VALUES(sift3),
#                     sift4 = VALUES(sift4),
#                     sift5 = VALUES(sift5),
#                     sift6 = VALUES(sift6),
#                     deep_avg = VALUES(deep_avg)
#             ''', (
#                 medicine_name,
#                 sift_blob_list[0],
#                 sift_blob_list[1],
#                 sift_blob_list[2],
#                 sift_blob_list[3],
#                 sift_blob_list[4],
#                 sift_blob_list[5],
#                 None
#             ))

#         self.conn.commit()

#         # 录入后直接更新当前药品缓存，不需要全量 reload
#         self._update_single_cache(medicine_name, sift_blob_list)

#         print(f"[录入完成] {medicine_name}")

#     def save_batch_features_to_db(self, root_folder):
#         """
#         批量录入。

#         root_folder 结构示例：
#             root_folder/
#                 药品A/
#                     1.jpg
#                     2.jpg
#                 药品B/
#                     1.jpg
#                     2.jpg
#         """

#         if not os.path.isdir(root_folder):
#             print(f"[错误] 目录不存在: {root_folder}")
#             return

#         for item in os.listdir(root_folder):
#             sub_path = os.path.join(root_folder, item)

#             if os.path.isdir(sub_path):
#                 self.save_features_to_db(
#                     folder_path=sub_path,
#                     medicine_name=item
#                 )

#         print(f"[批量录入完成] 当前缓存共 {len(self._templates_cache)} 种药品")

#     def _update_single_cache(self, medicine_name, sift_blob_list):
#         """
#         单个药品录入后更新内存缓存。
#         """

#         sift_templates = []

#         for i, blob in enumerate(sift_blob_list):
#             if blob is None:
#                 continue

#             try:
#                 desc = pickle.loads(blob)

#                 if desc is None:
#                     continue

#                 desc = np.asarray(desc, dtype=np.float32)

#                 if len(desc.shape) != 2:
#                     continue

#                 if desc.shape[1] != 128:
#                     continue

#                 if desc.shape[0] < 2:
#                     continue

#                 sift_templates.append({
#                     'template_index': i,
#                     'desc_sift': desc
#                 })

#             except Exception as e:
#                 print(f"[警告] 更新缓存失败: {medicine_name}, sift{i + 1}, {e}")

#         with self._cache_lock:
#             self._templates_cache[medicine_name] = sift_templates
#             self._deep_avg_cache[medicine_name] = None

#     # ======================================================
#     # 构建候选药品 FLANN matcher
#     # ======================================================

#     def _build_candidate_matcher(self, medicine_names):
#         """
#         将候选药品的所有模板 SIFT 特征加入同一个 FLANN matcher。

#         返回:
#             matcher:
#                 cv2.FlannBasedMatcher

#             train_meta:
#                 train_meta[imgIdx] 对应 matcher 中第 imgIdx 个模板的元信息。

#         train_meta 示例:
#             [
#                 {
#                     'medicine_name': '注射用头孢曲松钠',
#                     'template_index': 0,
#                     'desc_count': 440
#                 },
#                 {
#                     'medicine_name': '注射用头孢曲松钠',
#                     'template_index': 1,
#                     'desc_count': 386
#                 },
#                 {
#                     'medicine_name': '注射用泮托拉唑钠',
#                     'template_index': 0,
#                     'desc_count': 320
#                 }
#             ]

#         OpenCV 中：
#             matcher.add([desc1, desc2, desc3])
#         匹配结果 m.imgIdx 就表示匹配到了第几个 desc。
#         """

#         matcher = cv2.FlannBasedMatcher(
#             self.index_params,
#             self.search_params
#         )

#         train_desc_list = []
#         train_meta = []

#         missing = []

#         with self._cache_lock:
#             for name in medicine_names:
#                 templates = self._templates_cache.get(name, [])

#                 if not templates:
#                     missing.append(name)
#                     continue

#                 for t in templates:
#                     desc = t.get('desc_sift', None)

#                     if desc is None:
#                         continue

#                     desc = np.asarray(desc, dtype=np.float32)

#                     if len(desc.shape) != 2:
#                         continue

#                     if desc.shape[1] != 128:
#                         continue

#                     if desc.shape[0] < 2:
#                         continue

#                     train_desc_list.append(desc)

#                     train_meta.append({
#                         'medicine_name': name,
#                         'template_index': t.get('template_index', None),
#                         'desc_count': desc.shape[0]
#                     })

#         if missing:
#             print(f"[警告] 以下候选药品未在缓存中找到或无有效特征: {missing}")

#         if not train_desc_list:
#             return None, []

#         matcher.add(train_desc_list)
#         matcher.train()

#         return matcher, train_meta

#     # ======================================================
#     # 快速分类函数
#     # ======================================================

#     def classify_fast(
#             self,
#             medicine_names,
#             image,
#             ratio_thresh=0.7,
#             min_good_matches=3,
#             decide_by_template=True,
#             use_distance_score=True
#     ):
#         """
#         快速分类。

#         核心流程：
#             1. 对当前实时药瓶图像提取一次 SIFT；
#             2. 把所有候选药品的所有模板 SIFT 放进一个 FLANN；
#             3. 当前图像只执行一次 knnMatch；
#             4. 统计每个模板 / 每个类别获得的 good matches；
#             5. 判断当前样本最像哪个模板，从而确定类别。

#         参数:
#             medicine_names:
#                 OCR 匹配得到的候选药品名称列表。

#             image:
#                 当前药瓶图像，OpenCV BGR 格式。

#             ratio_thresh:
#                 Lowe ratio test 阈值。
#                 常用 0.7、0.75。
#                 越小越严格。

#             min_good_matches:
#                 用于置信度估计的最少匹配点数量。

#             decide_by_template:
#                 True:
#                     完全根据“最像的单个模板”决定类别。
#                     这符合你的需求：当前样本与哪个模板最像。
#                 False:
#                     根据类别总投票决定类别。

#             use_distance_score:
#                 是否额外使用距离加权分数。

#         返回:
#             {
#                 predicted_category,
#                 confidence,
#                 all_scores,
#                 details,
#                 top_3,
#                 best_template,
#                 total_good_matches,
#                 query_sift_shape,
#                 time_cost
#             }
#         """

#         start_time = time.time()

#         if not medicine_names:
#             raise ValueError("medicine_names 为空，无法分类")

#         # 去重但保持原顺序
#         medicine_names = list(dict.fromkeys(medicine_names))

#         # --------------------------------------------------
#         # 1. 提取当前图片 SIFT
#         # --------------------------------------------------

#         kp_query, desc_query = self._extract_sift_features(image)

#         if desc_query is None or len(desc_query) < 2:
#             print("[警告] 当前图片未提取到足够 SIFT 特征，使用候选首位")
#             return {
#                 'predicted_category': medicine_names[0],
#                 'confidence': 0.0,
#                 'all_scores': {medicine_names[0]: 0.0},
#                 'details': {
#                     medicine_names[0]: {
#                         'good_matches': 0,
#                         'score': 0.0,
#                         'note': 'no_query_sift'
#                     }
#                 },
#                 'top_3': [(medicine_names[0], 0.0)],
#                 'best_template': None,
#                 'total_good_matches': 0,
#                 'query_sift_shape': None,
#                 'time_cost': time.time() - start_time,
#             }

#         desc_query = np.asarray(desc_query, dtype=np.float32)

#         print(f"[实时图片] SIFT shape: {desc_query.shape}")

#         # --------------------------------------------------
#         # 2. 构建候选模板 FLANN matcher
#         # --------------------------------------------------

#         matcher, train_meta = self._build_candidate_matcher(medicine_names)

#         if matcher is None or not train_meta:
#             print("[警告] 候选药品中没有可用模板特征，使用候选首位")
#             return {
#                 'predicted_category': medicine_names[0],
#                 'confidence': 0.0,
#                 'all_scores': {medicine_names[0]: 0.0},
#                 'details': {
#                     medicine_names[0]: {
#                         'good_matches': 0,
#                         'score': 0.0,
#                         'note': 'no_template_sift'
#                     }
#                 },
#                 'top_3': [(medicine_names[0], 0.0)],
#                 'best_template': None,
#                 'total_good_matches': 0,
#                 'query_sift_shape': desc_query.shape,
#                 'time_cost': time.time() - start_time,
#             }

#         # --------------------------------------------------
#         # 3. 一次性 FLANN 匹配
#         # --------------------------------------------------

#         try:
#             matches = matcher.knnMatch(
#                 desc_query,
#                 k=2
#             )
#         except Exception as e:
#             print(f"[错误] FLANN knnMatch 失败: {e}")
#             return {
#                 'predicted_category': medicine_names[0],
#                 'confidence': 0.0,
#                 'all_scores': {medicine_names[0]: 0.0},
#                 'details': {
#                     medicine_names[0]: {
#                         'good_matches': 0,
#                         'score': 0.0,
#                         'note': f'flann_error: {e}'
#                     }
#                 },
#                 'top_3': [(medicine_names[0], 0.0)],
#                 'best_template': None,
#                 'total_good_matches': 0,
#                 'query_sift_shape': desc_query.shape,
#                 'time_cost': time.time() - start_time,
#             }

#         # --------------------------------------------------
#         # 4. 统计 good matches
#         # --------------------------------------------------

#         medicine_votes = defaultdict(int)
#         medicine_distance_score = defaultdict(float)

#         template_votes = defaultdict(int)
#         template_distance_score = defaultdict(float)

#         total_good = 0

#         for item in matches:
#             if len(item) < 2:
#                 continue

#             m, n = item

#             # Lowe ratio test
#             if m.distance < ratio_thresh * n.distance:
#                 img_idx = m.imgIdx

#                 if img_idx < 0 or img_idx >= len(train_meta):
#                     continue

#                 meta = train_meta[img_idx]

#                 medicine_name = meta['medicine_name']
#                 template_index = meta['template_index']

#                 template_key = (
#                     medicine_name,
#                     template_index
#                 )

#                 medicine_votes[medicine_name] += 1
#                 template_votes[template_key] += 1

#                 if use_distance_score:
#                     dist_score = 1.0 / (m.distance + 1e-6)

#                     medicine_distance_score[medicine_name] += dist_score
#                     template_distance_score[template_key] += dist_score

#                 total_good += 1

#         # --------------------------------------------------
#         # 5. 如果没有有效匹配，则退化为候选首位
#         # --------------------------------------------------

#         if total_good == 0 or not medicine_votes:
#             elapsed = time.time() - start_time

#             print("[警告] 没有通过 ratio test 的 good matches，使用候选首位")

#             return {
#                 'predicted_category': medicine_names[0],
#                 'confidence': 0.0,
#                 'all_scores': {medicine_names[0]: 0.0},
#                 'details': {
#                     medicine_names[0]: {
#                         'good_matches': 0,
#                         'score': 0.0,
#                         'note': 'no_good_matches'
#                     }
#                 },
#                 'top_3': [(medicine_names[0], 0.0)],
#                 'best_template': None,
#                 'total_good_matches': 0,
#                 'query_sift_shape': desc_query.shape,
#                 'time_cost': elapsed,
#             }

#         # --------------------------------------------------
#         # 6. 计算类别分数
#         # --------------------------------------------------

#         raw_scores = {}

#         for name in medicine_votes.keys():
#             vote_score = medicine_votes[name]

#             if use_distance_score:
#                 dist_score = medicine_distance_score[name]
#                 score = vote_score + 0.1 * dist_score
#             else:
#                 score = vote_score

#             raw_scores[name] = score

#         sorted_class_items = sorted(
#             raw_scores.items(),
#             key=lambda x: x[1],
#             reverse=True
#         )

#         # --------------------------------------------------
#         # 7. 找出最像的具体模板
#         # --------------------------------------------------

#         template_scores = {}

#         for template_key in template_votes.keys():
#             vote_score = template_votes[template_key]

#             if use_distance_score:
#                 dist_score = template_distance_score[template_key]
#                 score = vote_score + 0.1 * dist_score
#             else:
#                 score = vote_score

#             template_scores[template_key] = score

#         sorted_template_items = sorted(
#             template_scores.items(),
#             key=lambda x: x[1],
#             reverse=True
#         )

#         best_template = None

#         if sorted_template_items:
#             best_template_key, best_template_score = sorted_template_items[0]

#             best_template = {
#                 'medicine_name': best_template_key[0],
#                 'template_index': best_template_key[1],
#                 'good_matches': template_votes[best_template_key],
#                 'score': best_template_score
#             }

#         # --------------------------------------------------
#         # 8. 决定最终类别
#         # --------------------------------------------------

#         if decide_by_template and best_template is not None:
#             # 你当前的需求：
#             # 当前样本与哪个模板最像，就属于哪个模板对应的类别。
#             best_name = best_template['medicine_name']
#             best_score = best_template['score']
#         else:
#             # 按类别总投票
#             best_name = sorted_class_items[0][0]
#             best_score = sorted_class_items[0][1]

#         # --------------------------------------------------
#         # 9. 置信度估计
#         # --------------------------------------------------

#         best_good = medicine_votes.get(best_name, 0)

#         if decide_by_template and best_template is not None:
#             best_good = best_template['good_matches']

#         vote_conf = best_good / max(1, total_good)

#         good_conf = min(
#             1.0,
#             best_good / max(1, min_good_matches * 5)
#         )

#         confidence = 0.7 * vote_conf + 0.3 * good_conf

#         # --------------------------------------------------
#         # 10. details
#         # --------------------------------------------------

#         details = {}

#         for name in medicine_names:
#             details[name] = {
#                 'good_matches': medicine_votes.get(name, 0),
#                 'distance_score': medicine_distance_score.get(name, 0.0),
#                 'score': raw_scores.get(name, 0.0),
#             }

#         elapsed = time.time() - start_time

#         print(f"[快速分类耗时] {elapsed:.4f} 秒")
#         print(f"[类别投票] {dict(medicine_votes)}")
#         print(f"[最佳模板] {best_template}")
#         print(f"[最终类别] {best_name}, confidence={confidence:.4f}")

#         return {
#             'predicted_category': best_name,
#             'confidence': confidence,
#             'all_scores': dict(sorted_class_items),
#             'details': details,
#             'top_3': sorted_class_items[:3],
#             'best_template': best_template,
#             'total_good_matches': total_good,
#             'query_sift_shape': desc_query.shape,
#             'time_cost': elapsed,
#         }

#     # ======================================================
#     # 保持原 classify 接口不变
#     # ======================================================

#     def classify(self, medicine_names, image):
#         """
#         外部仍然调用 classify。
#         内部使用快速分类逻辑。

#         当前设置：
#             decide_by_template=True

#         含义：
#             按“当前样本最像哪个模板”来决定类别。
#         """

#         return self.classify_fast(
#             medicine_names=medicine_names,
#             image=image,
#             ratio_thresh=0.7,
#             min_good_matches=3,
#             decide_by_template=True,
#             use_distance_score=True
#         )

#     def get_cached_names(self):
#         """
#         返回当前缓存中的全部药品名称。
#         """

#         return list(self._templates_cache.keys())


# # ==========================================================
# # 使用示例
# # ==========================================================

# if __name__ == "__main__":

#     # ------------------------------------------------------
#     # 1. 连接数据库
#     # ------------------------------------------------------
#     # 注意：
#     # 如果你的 MySQL 不支持 utf8mb4，请使用 utf8。
#     # 你之前报过 Unknown character set: 'utf8mb4'，
#     # 所以这里使用 utf8。
#     conn = pymysql.connect(
#         host='localhost',
#         user='root',
#         password='root',
#         database='medicine_db',
#         charset='utf8',
#         cursorclass=pymysql.cursors.Cursor
#     )

#     # ------------------------------------------------------
#     # 2. 初始化分类器
#     # ------------------------------------------------------

#     classifier = PharmaceuticalBottleClassifier(
#         db_conn=conn,
#         device='cuda'
#     )

#     print(f"已缓存药品: {classifier.get_cached_names()}")

#     # ------------------------------------------------------
#     # 3. 如需批量录入模板特征，取消下面注释
#     # ------------------------------------------------------

#     # classifier.save_batch_features_to_db(
#     #     os.path.abspath("./feat_data")
#     # )

#     # ------------------------------------------------------
#     # 4. 测试分类
#     # ------------------------------------------------------

#     candidate_names = [
#         "人免疫球蛋白",
#         "注射用人干扰素a2a",
#         "注射用人干扰素a2b",
#         "注射用哌拉西林钠他唑巴坦钠",
#         "注射用头孢他啶",
#         "注射用头孢哌酮钠舒巴坦钠（2：1）",
#         "注射用头孢唑啉钠",
#         "注射用头孢曲松钠",
#         "注射用泮托拉唑钠",
#         "注射用炎琥宁",
#         "注射用艾司奥拉美拉唑钠",
#         "破伤风人免疫球蛋白",
#         "重组人血小板生成素注射液",
#     ]

#     img = cv2.imread("./data/bottle_1773990086441.jpg")

#     if img is not None:
#         result = classifier.classify(
#             candidate_names,
#             img
#         )

#         best_match = result['predicted_category']
#         confidence = result['confidence']

#         print("\n==============================")
#         print("分类结果")
#         print("==============================")
#         print(f"最佳类别: {best_match}")
#         print(f"置信度: {confidence:.4f}")
#         print(f"Top3: {result['top_3']}")
#         print(f"最佳模板: {result['best_template']}")
#         print(f"总 good matches: {result['total_good_matches']}")
#         print(f"耗时: {result['time_cost']:.4f} 秒")

#     else:
#         print("图片读取失败")

#     conn.close()

