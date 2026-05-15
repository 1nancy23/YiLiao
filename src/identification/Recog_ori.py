import cv2
import numpy as np

from PIL import Image
import os
# from scipy.spatial.distance import cosine
import json
import pickle
import pymysql


class PharmaceuticalBottleClassifier:
    """
    药瓶分类器，支持从MySQL数据库存储和加载特征。
    主要接口：
        - save_features_to_db(folder_path, medicine_name=None)   # 单个药品录入
        - save_batch_features_to_db(root_folder)                 # 批量录入
        - classify(medicine_names, image)                        # 单张图片分类
    """

    def __init__(self, db_conn, device='cuda'):
        self.conn = db_conn
        # self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.sift = cv2.SIFT_create()

        # self.model = models.mobilenet_v3_small(pretrained=True)
        # self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        # self.model.eval()
        # self.model = self.model.to(self.device)

        # self.preprocess = transforms.Compose([
        #     transforms.Resize(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225]),
        # ])

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self._init_db_table()

    # ---------- 私有辅助方法 ----------
    def _preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _extract_sift_features(self, image):
        # processed = self._preprocess_image(image)
        return self.sift.detectAndCompute(image, None)

    def _extract_deep_features(self, image):
        
    

    def _extract_features_from_image(self, image):
        kp, desc = self._extract_sift_features(image)
        deep_feat = self._extract_deep_features(image)
        return kp, desc, deep_feat

    def _match_sift_features(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return 0
        try:
            matches = self.flann.knnMatch(desc1, desc2, k=2)
            good = 0
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good += 1
            return good
        except:
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

    # ---------- 公共接口：特征录入 ----------
    def save_features_to_db(self, folder_path, medicine_name=None):
        if medicine_name is None:
            medicine_name = os.path.basename(folder_path.rstrip('/\\'))

        image_files = sorted([f for f in os.listdir(folder_path)
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))])[:6]
        if not image_files:
            return

        sift_list = []
        deep_list = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            kp, desc, deep_feat = self._extract_features_from_image(img)
            sift_list.append(pickle.dumps(desc))
            deep_list.append(deep_feat)

        while len(sift_list) < 6:
            sift_list.append(None)

        if deep_list:
            deep_avg = np.mean(deep_list, axis=0)
            deep_avg_blob = pickle.dumps(deep_avg)
        else:
            deep_avg_blob = None

        with self.conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO drugs (medicine_name, sift1, sift2, sift3, sift4, sift5, sift6, deep_avg)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    sift1 = VALUES(sift1),
                    sift2 = VALUES(sift2),
                    sift3 = VALUES(sift3),
                    sift4 = VALUES(sift4),
                    sift5 = VALUES(sift5),
                    sift6 = VALUES(sift6),
                    deep_avg = VALUES(deep_avg)
            ''', (medicine_name, *sift_list, deep_avg_blob))
        self.conn.commit()

    def save_batch_features_to_db(self, root_folder):
        if not os.path.isdir(root_folder):
            return

        for item in os.listdir(root_folder):
            sub_path = os.path.join(root_folder, item)
            if os.path.isdir(sub_path):
                self.save_features_to_db(sub_path, medicine_name=item)

    # ---------- 公共接口：分类 ----------
    def classify(self, medicine_names, image):
        """
        对单张图片进行分类，仅考虑指定的药品名称列表。
        :param medicine_names: 待匹配的药品名称列表
        :param image: OpenCV图像（BGR格式）
        :return: 字典，包含预测类别、置信度、所有得分等
        """
        templates_dict, deep_avg_dict = self._load_features_from_db(medicine_names)
        if not templates_dict:
            raise ValueError("未找到任何指定药品的特征，无法分类")

        _, desc_query = self._extract_sift_features(image)
        # deep_query = self._extract_deep_features(image)

        scores = {}
        details = {}
        for name, templates in templates_dict.items():
            sift_scores = [self._match_sift_features(t['desc_sift'], desc_query) for t in templates]
            max_sift = max(sift_scores) if sift_scores else 0
            sift_conf = min(1.0, max_sift / 15.0)

            # deep_avg = deep_avg_dict.get(name)
            # if deep_avg is not None:
            #     deep_sim = 1 - cosine(deep_avg, deep_query)
            #     deep_conf = max(0, deep_sim)
            # else:
            #     deep_conf = 0

            final = sift_conf * 0.6 
            scores[name] = final
            details[name] = {
                'max_sift_score': max_sift,
                'sift_confidence': sift_conf,
            
            }

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result = {
            'predicted_category': sorted_items[0][0],
            'confidence': sorted_items[0][1],
            'all_scores': dict(sorted_items),
            'details': details,
            'top_3': sorted_items[:3]
        }
        return result

    def _load_features_from_db(self, medicine_names):
        placeholders = ','.join(['%s'] * len(medicine_names))
        query = f'''
            SELECT medicine_name, sift1, sift2, sift3, sift4, sift5, sift6, deep_avg
            FROM drugs
            WHERE medicine_name IN ({placeholders})
        '''
        with self.conn.cursor() as cursor:
            cursor.execute(query, medicine_names)
            rows = cursor.fetchall()

        if not rows:
            return {}, {}

        templates = {}
        deep_avgs = {}
        for row in rows:
            sift_list = []
            # 兼容 DictCursor 和普通 Cursor
            if isinstance(row, dict):
                name = row['medicine_name'].decode('utf-8') if isinstance(row['medicine_name'], bytes) else row[
                    'medicine_name']
                s1 = row['sift1']
                s2 = row['sift2']
                s3 = row['sift3']
                s4 = row['sift4']
                s5 = row['sift5']
                s6 = row['sift6']
                deep_blob = row['deep_avg']
            else:
                name = row[0].decode('utf-8') if isinstance(row[0], bytes) else row[0]
                s1, s2, s3, s4, s5, s6, deep_blob = row[1:]

            for i, blob in enumerate([s1, s2, s3, s4, s5, s6]):
                if blob is not None:
                    desc = pickle.loads(blob)
                    sift_list.append({'desc_sift': desc})
            templates[name] = sift_list

            if deep_blob is not None:
                deep_avgs[name] = pickle.loads(deep_blob)
            else:
                deep_avgs[name] = None

        return templates, deep_avgs

    # 移除所有打印相关方法（如 save_json_results, print_classification_report）
    # 如果外部需要保存结果，可自行实现

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 建立数据库连接（请根据实际配置修改）
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        database='medicine_db',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.Cursor
    )
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建分类器实例时传入连接
    classifier = PharmaceuticalBottleClassifier(db_conn=conn, device='cuda')

    # 1. 批量录入特征（遍历子文件夹）
    # classifier.save_batch_features_to_db(os.path.abspath("./feat_data"))

    # 2. 单个录入
    # classifier.save_features_to_db("A:/Temp_dataset/Blured/药品A")

    # 3. 分类
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
        "重组人血小板生成素注射液"
        ]
    img = cv2.imread("./data/bottle_1773990086441.jpg")
    if img is not None:
        img = cv2.resize(img, (224, 224))
        feat_result = classifier.classify(candidate_names, img)
        best_match = feat_result['predicted_category']
        confidence = feat_result['confidence']
        print(f"  特征匹配最佳: {best_match} (置信度: {confidence:.4f})")


    # 使用完毕后关闭连接（由调用者负责）
    conn.close()