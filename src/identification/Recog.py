import cv2
import numpy as np
from PIL import Image
import os
import json
import pickle
import pymysql
import time
import hashlib
from src.identification.rknn_runtime_lock import get_rknn_lock

try:
    from rknnlite.api import RKNNLite
except ModuleNotFoundError:
    class RKNNLite:
        NPU_CORE_2 = 0

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("rknnlite is required for NPU rotation classification")


def _runtime_logs_enabled():
    return os.environ.get("YILIAO_RUNTIME_LOGS", "0").lower() in ("1", "true", "yes", "on")


def _runtime_log(*args, **kwargs):
    if _runtime_logs_enabled():

        print(*args, **kwargs)


label_list = [0, 90, 180, 270]
CLS_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
CLS_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
SIFT_CACHE_PIPELINE = "original_resolution_bidirectional_bf_v1"


def _default_cls_model_path():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    bs32_path = os.path.join(root, "model_cls_bs32.rknn")
    if os.path.exists(bs32_path):
        return bs32_path
    return os.path.abspath(
        os.path.join(root, "model_cls.rknn")
    )


def _default_feature_cache_path():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "single_image_feature_cache.pkl")
    )


def _default_feature_root_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "feat_data"))


def _batch_size_from_path(path, default=1):
    name = os.path.basename(str(path)).lower()
    marker = "bs"
    pos = name.find(marker)
    if pos < 0:
        return default
    pos += len(marker)
    digits = []
    while pos < len(name) and name[pos].isdigit():
        digits.append(name[pos])
        pos += 1
    return int("".join(digits)) if digits else default


def _feature_root_signature(root_folder):
    entries = []
    for medicine_name in sorted(os.listdir(root_folder)):
        medicine_dir = os.path.join(root_folder, medicine_name)
        if not os.path.isdir(medicine_dir):
            continue
        for file_name in sorted(os.listdir(medicine_dir)):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(medicine_dir, file_name)
            stat = os.stat(path)
            rel = os.path.relpath(path, root_folder).replace("\\", "/")
            entries.append(f"{rel}|{stat.st_size}|{stat.st_mtime_ns}")
    digest = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
    return {
        "root": os.path.abspath(root_folder),
        "digest": digest,
        "entries": len(entries),
        "sift_pipeline": SIFT_CACHE_PIPELINE,
    }


def reverse_rotate_with_label(img, pred):
    angle = label_list[int( pred )]

    # 反向旋转角度
    reverse_angle = (360 - angle) % 360

    if reverse_angle == 0:
        return img
    elif reverse_angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif reverse_angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif reverse_angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(f"不支持的角度: {reverse_angle}")
def preprocess_cls_image(image, layout="NCHW"):
    if layout.upper() != "NCHW":
        return cv2.resize(image, (224, 224))

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        resized = np.zeros((224, 224, 3), dtype=np.uint8)
    else:
        scale = 256.0 / float(min(height, width))
        resized_w = max(224, int(round(width * scale)))
        resized_h = max(224, int(round(height * scale)))
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    y1 = max(0, (resized.shape[0] - 224) // 2)
    x1 = max(0, (resized.shape[1] - 224) // 2)
    cropped = resized[y1:y1 + 224, x1:x1 + 224]
    if cropped.shape[0] != 224 or cropped.shape[1] != 224:
        cropped = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - CLS_MEAN) / CLS_STD
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32)


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
        # self.sift = cv2.ORB_create(nfeatures=160)
        self.sift = cv2.SIFT_create(
            nfeatures=500,
            contrastThreshold=0.03,
            edgeThreshold=10,
            sigma=1.6
        )
        FLANN_INDEX_KDTREE = 1
        self.cls_model_path = os.environ.get("YILIAO_CLS_MODEL", _default_cls_model_path())
        self.cls_batch_size = int(os.environ.get(
            "YILIAO_CLS_BATCH_SIZE",
            32 if "bs32" in os.path.basename(self.cls_model_path).lower() else 1,
        ))
        self.cls_batch_size = max(
            1,
            int(self.cls_batch_size),
            _batch_size_from_path(self.cls_model_path, default=1),
        )
        self.cls_input_layout = "NHWC"
        self.cls_core = RKNNLite.NPU_CORE_2
        self.cls_lock = get_rknn_lock(self.cls_core, secondary_domain=True)
        self.cls_model = None
        if os.environ.get("YILIAO_EAGER_BOTTLE_CLS", "0").lower() in ("1", "true", "yes", "on"):
            self._ensure_cls_model()
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
        search_params = dict(checks=20)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        # self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.matcher = cv2.BFMatcher()
        # cv2.NORM_HAMMING, crossCheck=False

        # ========== 核心优化：初始化时一次性加载全部特征到内存 ==========
        self._templates_cache = {}   # {medicine_name: [{'desc_sift': np.array}, ...]}
        self._deep_avg_cache = {}    # {medicine_name: np.array or None}
        self._sift_candidate_cache = {}
        self._sift_candidate_cache_limit = int(os.environ.get("YILIAO_SIFT_CANDIDATE_CACHE_LIMIT", "16"))
        if self.conn is not None:
            self._init_db_table()
        self._load_all_features()
        
        _runtime_log(f"[初始化完成] 已加载 {len(self._templates_cache)} 种药品特征到内存")

    # ---------- 私有辅助方法 ----------
    def _ensure_cls_model(self):
        if self.cls_model is not None:
            return
        self.cls_model = RKNNLite()
        self.cls_model.load_rknn(self.cls_model_path)
        self.cls_model.init_runtime(core_mask=self.cls_core)

    def release(self):
        if self.cls_model is not None:
            try:
                self.cls_model.release()
            except Exception:
                pass
            self.cls_model = None

    def _preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _extract_sift_features(self, image):
        # SIFT templates and live bottle crops must retain their source resolution.
        return self.sift.detectAndCompute(image, None)

    def _extract_features_from_image(self, image):
        _runtime_log("image",image.shape)
        kp, desc = self._extract_sift_features(image)
        if desc is None:
            return kp[:320], None
        return kp[:320], desc[:320]

    def _match_sift_features(self, desc1, desc2):
        desc1 = self._valid_sift_desc(desc1)
        desc2 = self._valid_sift_desc(desc2)
        if desc1 is None or desc2 is None:
            return 0, 0.0
        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good = 0
            quality = 0.0
            for item in matches:
                if len(item) < 2:
                    continue
                m, n = item
                if m.distance < 0.72 * n.distance:
                    good += 1
                    quality += 1.0 - (m.distance / max(n.distance, 1e-6))
            reverse_matches = self.matcher.knnMatch(desc2, desc1, k=2)
            reverse_good = 0
            for item in reverse_matches:
                if len(item) < 2:
                    continue
                m, n = item
                if m.distance < 0.72 * n.distance:
                    reverse_good += 1
            stable_good = min(good, reverse_good)
            avg_quality = quality / max(1, good)
            denom = max(1.0, float(np.sqrt(len(desc1) * len(desc2))))
            score = (stable_good / denom) * (0.75 + 0.25 * avg_quality)
            return good, score
        except Exception:
            return 0, 0.0

    def _valid_sift_desc(self, desc):
        if desc is None:
            return None
        desc = np.asarray(desc, dtype=np.float32)
        if desc.ndim != 2 or desc.shape[1] != 128 or desc.shape[0] < 2:
            return None
        return desc

    def _build_candidate_descriptors(self, templates_dict):
        cache_key = tuple(templates_dict.keys())
        cached = self._sift_candidate_cache.get(cache_key)
        if cached is not None:
            return cached

        train_descs = []
        train_meta = []
        for name, templates in templates_dict.items():
            for template_index, template in enumerate(templates):
                desc = self._valid_sift_desc(template.get("desc_sift"))
                if desc is None:
                    continue
                train_meta.append({
                    "medicine_name": name,
                    "template_index": template_index,
                    "desc_count": int(desc.shape[0]),
                })
                train_descs.append(desc)
        result = (train_descs, train_meta)
        if self._sift_candidate_cache_limit > 0:
            if len(self._sift_candidate_cache) >= self._sift_candidate_cache_limit:
                self._sift_candidate_cache.pop(next(iter(self._sift_candidate_cache)))
            self._sift_candidate_cache[cache_key] = result
        return result

    def _competitive_sift_scores(self, templates_dict, desc_query, ratio_thresh=0.72):
        desc_query = self._valid_sift_desc(desc_query)
        if desc_query is None:
            return {}, {}, None, 0

        train_descs, train_meta = self._build_candidate_descriptors(templates_dict)
        if not train_descs:
            return {}, {}, None, 0

        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matcher.add(train_descs)
        matcher.train()

        template_stats = {}
        medicine_stats = {
            name: {"good": 0, "score_sum": 0.0, "templates": []}
            for name in templates_dict.keys()
        }
        total_good = 0

        try:
            matches = matcher.knnMatch(desc_query, k=2)
        except Exception:
            return {}, {}, None, 0

        for item in matches:
            if len(item) < 2:
                continue
            m, n = item
            if m.imgIdx < 0 or m.imgIdx >= len(train_meta):
                continue
            if m.distance >= ratio_thresh * n.distance:
                continue

            meta = train_meta[m.imgIdx]
            key = (meta["medicine_name"], meta["template_index"])
            quality = 1.0 - (m.distance / max(n.distance, 1e-6))
            stat = template_stats.setdefault(key, {
                "medicine_name": meta["medicine_name"],
                "template_index": meta["template_index"],
                "good_matches": 0,
                "quality_sum": 0.0,
                "desc_template_len": meta["desc_count"],
                "desc_query_len": int(desc_query.shape[0]),
            })
            stat["good_matches"] += 1
            stat["quality_sum"] += quality
            medicine_stats[meta["medicine_name"]]["good"] += 1
            total_good += 1

        if total_good == 0:
            return {}, {}, None, 0

        template_scores = {}
        for key, stat in template_stats.items():
            avg_quality = stat["quality_sum"] / max(1, stat["good_matches"])
            denom = max(
                1.0,
                float(np.sqrt(stat["desc_template_len"] * stat["desc_query_len"]))
            )
            score = (stat["good_matches"] / denom) * (0.75 + 0.25 * avg_quality)
            scored = {
                "medicine_name": stat["medicine_name"],
                "template_index": stat["template_index"],
                "good_matches": int(stat["good_matches"]),
                "match_score": float(score),
                "quality": float(avg_quality),
                "desc_template_len": int(stat["desc_template_len"]),
                "desc_query_len": int(stat["desc_query_len"]),
            }
            template_scores[key] = scored
            medicine_stats[stat["medicine_name"]]["score_sum"] += score
            medicine_stats[stat["medicine_name"]]["templates"].append(scored)

        medicine_scores = {}
        details = {}
        for name, stat in medicine_stats.items():
            templates = sorted(
                stat["templates"],
                key=lambda item: item["match_score"],
                reverse=True,
            )
            if not templates:
                medicine_scores[name] = 0.0
                details[name] = {
                    "max_sift_score": 0,
                    "sift_confidence": 0.0,
                    "best_template": {
                        "good_matches": 0,
                        "match_score": 0.0,
                        "desc_template_len": 0,
                        "desc_query_len": int(desc_query.shape[0]),
                    },
                    "template_scores": [],
                    "total_good_matches": 0,
                }
                continue

            best_template = templates[0]
            top2_sum = sum(item["match_score"] for item in templates[:2])
            aggregate = stat["score_sum"] / max(1, min(6, len(templates_dict.get(name, []))))
            vote_share = stat["good"] / max(1, total_good)
            score = (
                0.68 * best_template["match_score"] +
                0.20 * top2_sum +
                0.07 * aggregate +
                0.05 * vote_share
            )
            medicine_scores[name] = float(score)
            details[name] = {
                "max_sift_score": int(best_template["good_matches"]),
                "sift_confidence": float(score),
                "best_template": best_template,
                "template_scores": templates[:6],
                "total_good_matches": int(stat["good"]),
            }

        best_template = max(
            template_scores.values(),
            key=lambda item: item["match_score"],
            default=None,
        )
        return medicine_scores, details, best_template, total_good

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
                    sift6 MEDIUMBLOB
                )
            ''')
        self.conn.commit()

    # ========== 核心优化：一次性加载全部特征 ==========
    def _load_all_features(self):
        """从数据库一次性加载全部药品特征到内存"""
        self._sift_candidate_cache.clear()
        cache_path = os.environ.get("YILIAO_FEATURE_CACHE", _default_feature_cache_path())
        feature_root = os.environ.get("YILIAO_FEATURE_ROOT", "").strip()
        if feature_root:
            feature_root = os.path.abspath(feature_root)
            if not os.path.isdir(feature_root):
                raise FileNotFoundError(f"feature template folder not found: {feature_root}")

            signature = _feature_root_signature(feature_root)
            if signature["entries"] <= 0:
                raise ValueError(f"feature template folder has no images: {feature_root}")

            if cache_path and os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        cache = pickle.load(f)
                    if cache.get("feature_root_signature") == signature:
                        self._templates_cache = cache.get("templates", {})
                        self._deep_avg_cache = cache.get("deep_avg", {})
                        if not self._templates_cache:
                            raise ValueError("feature cache has no templates")
                        _runtime_log(
                            f"[特征缓存] 已从 {cache_path} 加载 {len(self._templates_cache)} 种药品特征"
                        )
                        return
                    _runtime_log("[特征缓存] 模板目录已变化，重新构建缓存")
                except Exception as e:
                    _runtime_log(f"[特征缓存] 读取失败，重新构建缓存: {e}")

            self._load_features_from_template_folder(feature_root)
            if not self._templates_cache:
                raise ValueError(f"no feature templates loaded from: {feature_root}")
            if cache_path:
                with open(cache_path, "wb") as f:
                    pickle.dump(
                        {
                            "templates": self._templates_cache,
                            "deep_avg": self._deep_avg_cache,
                            "feature_root_signature": signature,
                        },
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                _runtime_log(f"[特征缓存] 已根据模板目录写入 {cache_path}")
            return

        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                self._templates_cache = cache.get("templates", {})
                self._deep_avg_cache = cache.get("deep_avg", {})
                _runtime_log(f"[特征缓存] 已从 {cache_path} 加载 {len(self._templates_cache)} 种药品特征")
                return
            except Exception as e:
                _runtime_log(f"[特征缓存] 读取失败，回退到数据库: {e}")

        if self.conn is None:
            raise ValueError("db_conn is required when feature cache is missing or invalid")

        query = '''
            SELECT medicine_name, sift1, sift2, sift3, sift4, sift5, sift6
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
            else:
                name = row[0]
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                sift_blobs = list(row[1:7])

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
            self._deep_avg_cache[name] = None

        if cache_path:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(
                        {
                            "templates": self._templates_cache,
                            "deep_avg": self._deep_avg_cache,
                        },
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                _runtime_log(f"[特征缓存] 已写入 {cache_path}")
            except Exception as e:
                _runtime_log(f"[特征缓存] 写入失败: {e}")

    def _load_features_from_template_folder(self, root_folder):
        self._sift_candidate_cache.clear()
        self._templates_cache.clear()
        self._deep_avg_cache.clear()
        for medicine_name in sorted(os.listdir(root_folder)):
            medicine_dir = os.path.join(root_folder, medicine_name)
            if not os.path.isdir(medicine_dir):
                continue
            image_files = sorted([
                name for name in os.listdir(medicine_dir)
                if name.lower().endswith((".jpg", ".jpeg", ".png"))
            ])[:6]
            sift_templates = []
            for image_file in image_files:
                image_path = os.path.join(medicine_dir, image_file)
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                _kp, desc = self._extract_features_from_image(image)
                if desc is not None:
                    sift_templates.append({"desc_sift": desc})
            if sift_templates:
                self._templates_cache[medicine_name] = sift_templates
                self._deep_avg_cache[medicine_name] = None

    def reload_features(self):
        """公共接口：手动刷新内存缓存（数据库有外部变更时调用）"""
        self._sift_candidate_cache.clear()
        self._load_all_features()
        _runtime_log(f"[缓存刷新] 已重新加载 {len(self._templates_cache)} 种药品特征")

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
            if desc is not None:
                sift_list.append(pickle.dumps(desc))

        while len(sift_list) < 6:
            sift_list.append(None)

        with self.conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO drugs 
                    (medicine_name, sift1, sift2, sift3, sift4, sift5, sift6)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    sift1 = VALUES(sift1),
                    sift2 = VALUES(sift2),
                    sift3 = VALUES(sift3),
                    sift4 = VALUES(sift4),
                    sift5 = VALUES(sift5),
                    sift6 = VALUES(sift6)
            ''', (medicine_name, *sift_list))
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

        _runtime_log(f"[批量录入完成] 当前缓存共 {len(self._templates_cache)} 种药品")

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

    def _classify_already_rotated(self, medicine_names, image):
        log_timing = _runtime_logs_enabled()
        start_time = time.time() if log_timing else 0.0
        medicine_names = list(dict.fromkeys(medicine_names))
        templates_dict = {}
        missing = []
        for name in medicine_names:
            if name in self._templates_cache and self._templates_cache[name]:
                templates_dict[name] = self._templates_cache[name]
            else:
                missing.append(name)

        if missing:
            _runtime_log(f"[warning] medicines missing in feature cache: {missing}")

        if not templates_dict:
            raise ValueError("no cached features found for candidate medicines")

        start_time2 = time.time() if log_timing else 0.0
        _, desc_query = self._extract_sift_features(image)
        if log_timing:
            _runtime_log(f"[feature extract time] {time.time() - start_time2:.4f} s")

        scores, details, best_global_template, total_good = self._competitive_sift_scores(
            templates_dict,
            desc_query,
            ratio_thresh=0.72,
        )

        if not scores:
            scores = {}
            details = {}
            for name, templates in templates_dict.items():
                template_scores = []
                for template_index, template in enumerate(templates):
                    good_matches, match_score = self._match_sift_features(
                        template["desc_sift"],
                        desc_query,
                    )
                    template_scores.append({
                        "medicine_name": name,
                        "template_index": template_index,
                        "good_matches": int(good_matches),
                        "match_score": float(match_score),
                        "desc_template_len": 0 if template["desc_sift"] is None else len(template["desc_sift"]),
                        "desc_query_len": 0 if desc_query is None else len(desc_query),
                    })

                best_template = max(
                    template_scores,
                    key=lambda item: item["match_score"],
                    default={
                        "medicine_name": name,
                        "template_index": None,
                        "good_matches": 0,
                        "match_score": 0.0,
                        "desc_template_len": 0,
                        "desc_query_len": 0,
                    },
                )
                scores[name] = float(best_template["match_score"])
                details[name] = {
                    "max_sift_score": int(best_template["good_matches"]),
                    "sift_confidence": float(best_template["match_score"]),
                    "best_template": best_template,
                    "template_scores": sorted(
                        template_scores,
                        key=lambda item: item["match_score"],
                        reverse=True,
                    )[:6],
                    "total_good_matches": int(best_template["good_matches"]),
                }
            best_global_template = None
            total_good = 0

        for rank, name in enumerate(medicine_names):
            if name in scores:
                scores[name] += 0.002 / (rank + 1)

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if log_timing:
            _runtime_log(f"[classify time] {time.time() - start_time:.4f} s")
        return {
            "predicted_category": sorted_items[0][0],
            "confidence": sorted_items[0][1],
            "all_scores": dict(sorted_items),
            "details": details,
            "top_3": sorted_items[:3],
            "best_template": best_global_template,
            "total_good_matches": int(total_good),
            "cls_skipped": True,
            "cls_pred": None,
        }

    # ---------- 公共接口：分类（零数据库查询） ----------
    def classify(self, medicine_names, image, image_already_rotated=False):
        """
        对单张图片进行分类，仅考虑指定的药品名称列表。
        直接从内存缓存读取特征，不再查询数据库。

        :param medicine_names: 待匹配的药品名称列表
        :param image: OpenCV图像（BGR格式）
        :return: 字典，包含预测类别、置信度、所有得分等
        """
        if image_already_rotated:
            return self._classify_already_rotated(medicine_names, image)

        log_timing = _runtime_logs_enabled()
        start_time = time.time() if log_timing else 0.0
        medicine_names = list(dict.fromkeys(medicine_names))
        # ========== 直接从内存缓存过滤，无DB查询 ==========
        templates_dict = {}
        missing = []
        for name in medicine_names:
            if name in self._templates_cache and self._templates_cache[name]:
                templates_dict[name] = self._templates_cache[name]
            else:
                missing.append(name)

        if missing:
            _runtime_log(f"[警告] 以下药品未在缓存中找到: {missing}")

        if not templates_dict:
            raise ValueError("未找到任何指定药品的特征，无法分类")
        start_time2 = time.time() if log_timing else 0.0
        
        self._ensure_cls_model()
        input_nchw = preprocess_cls_image(image, self.cls_input_layout)
        batch_size = max(1, int(self.cls_batch_size))
        batch_data = np.zeros((batch_size,) + input_nchw.shape, dtype=input_nchw.dtype)
        batch_data[0] = input_nchw
        with self.cls_lock:
            outputs = self.cls_model.inference(inputs=[batch_data])
        outputs = np.asarray(outputs[0])
        while outputs.ndim > 2 and outputs.shape[0] == 1:
            outputs = outputs[0]
        pred_indices = np.argmax(outputs, axis=1)  # 形状 (20,)
        pred = pred_indices[0]
        if outputs.ndim == 1:
            pred = int(np.argmax(outputs))
        elif outputs.ndim == 2 and outputs.shape[0] == len(label_list) and outputs.shape[1] >= 1:
            pred = int(np.argmax(outputs[:, 0]))
        else:
            pred = int(np.argmax(outputs[0]))
        _runtime_log("药瓶旋转")
        image=reverse_rotate_with_label(image, pred)
        
        
        _, desc_query = self._extract_sift_features(image)
        if log_timing:
            _runtime_log(f"[特征提取耗时] {time.time() - start_time2:.4f} 秒")

        scores, details, best_global_template, total_good = self._competitive_sift_scores(
            templates_dict,
            desc_query,
            ratio_thresh=0.72,
        )

        if not scores:
            scores = {}
            details = {}
            for name, templates in templates_dict.items():
                template_scores = []
                for template_index, t in enumerate(templates):
                    good_matches, match_score = self._match_sift_features(
                        t['desc_sift'],
                        desc_query
                    )
                    template_scores.append({
                        'medicine_name': name,
                        'template_index': template_index,
                        'good_matches': int(good_matches),
                        'match_score': float(match_score),
                        'desc_template_len': 0 if t['desc_sift'] is None else len(t['desc_sift']),
                        'desc_query_len': 0 if desc_query is None else len(desc_query),
                    })

                best_template = max(
                    template_scores,
                    key=lambda item: item['match_score'],
                    default={
                        'medicine_name': name,
                        'template_index': None,
                        'good_matches': 0,
                        'match_score': 0.0,
                        'desc_template_len': 0,
                        'desc_query_len': 0,
                    }
                )
                scores[name] = float(best_template['match_score'])
                details[name] = {
                    'max_sift_score': int(best_template['good_matches']),
                    'sift_confidence': float(best_template['match_score']),
                    'best_template': best_template,
                    'template_scores': sorted(
                        template_scores,
                        key=lambda item: item["match_score"],
                        reverse=True,
                    )[:6],
                    'total_good_matches': int(best_template['good_matches']),
                }
            best_global_template = None
            total_good = 0

        # OCR 候选顺序已经包含文本语义信息。SIFT 分数非常接近或整体偏弱时，
        # 给靠前候选一个极小先验，避免低置信模板偶然匹配压过明显 OCR 语义。
        for rank, name in enumerate(medicine_names):
            if name in scores:
                scores[name] += 0.002 / (rank + 1)

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if log_timing:
            _runtime_log(f"[分类耗时] {time.time() - start_time:.4f} 秒")
        return {
            'predicted_category': sorted_items[0][0],
            'confidence': sorted_items[0][1],
            'all_scores': dict(sorted_items),
            'details': details,
            'top_3': sorted_items[:3],
            'best_template': best_global_template,
            'total_good_matches': int(total_good),
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
        database='medicine_db2',
        charset='utf8',
        cursorclass=pymysql.cursors.Cursor
    )

    # 创建分类器 → 自动加载全部特征到内存
    classifier = PharmaceuticalBottleClassifier(db_conn=conn, device='cuda')

    # 查看已缓存的药品
    _runtime_log(f"已缓存药品: {classifier.get_cached_names()}")

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
        feat_result = classifier.classify(candidate_names, img)
        best_match = feat_result['predicted_category']
        confidence = feat_result['confidence']
        _runtime_log(f"  特征匹配最佳: {best_match} (置信度: {confidence:.4f})")

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
#     conn = pymysql.connect(
#         host='localhost',
#         user='root',
#         password='root',
#         database='medicine_db2',
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
