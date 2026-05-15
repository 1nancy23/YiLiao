import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from scipy.spatial.distance import cosine
import json
from tqdm import tqdm
import pickle

class PharmaceuticalBottleClassifier:
    """
    药瓶多类别分类器 - 优化版本
    
    使用 SIFT 特征和深度学习特征的混合方法进行药瓶分类。
    支持特征缓存，避免重复计算。
    """

    def __init__(self, device='cuda', cache_dir='./feature_cache'):
        """
        初始化分类器
        
        参数:
            device: 计算设备，'cuda' 或 'cpu'
            cache_dir: 特征缓存目录路径
        """
        # 设置计算设备（GPU 或 CPU）
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        
        # 创建缓存目录（如果不存在）
        os.makedirs(cache_dir, exist_ok=True)
        
        # ========== SIFT 特征提取器 ==========
        # SIFT (Scale-Invariant Feature Transform) 用于提取图像的关键点和描述符
        # 对旋转、缩放、光照变化具有鲁棒性
        self.sift = cv2.SIFT_create()
        
        # ========== 深度学习特征提取器 ==========
        # 使用预训练的 MobileNetV3-Small 模型
        # MobileNetV3-Small 是轻量级模型，速度快，精度足够
        self.model = models.mobilenet_v3_small(pretrained=True)
        
        # 移除最后的分类层，只保留特征提取部分
        # 这样可以获得 1024 维的特征向量
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        # 设置为评估模式（不进行梯度计算）
        self.model.eval()
        
        # 将模型移到指定设备
        self.model = self.model.to(self.device)
        
        # ========== 图像预处理流程 ==========
        # 将输入图像标准化为模型期望的格式
        self.preprocess = transforms.Compose([
            transforms.Resize(224),  # 调整为 224x224
            transforms.ToTensor(),   # 转换为张量 [0, 1]
            transforms.Normalize(    # 使用 ImageNet 标准化参数
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # ========== FLANN 特征匹配器 ==========
        # FLANN (Fast Library for Approximate Nearest Neighbors)
        # 用于快速匹配 SIFT 描述符
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # ========== 存储模板特征 ==========
        # category_templates: {类别名: [特征字典列表]}
        # 每个特征字典包含: filename, path, kp_sift, desc_sift, deep_features
        self.category_templates = {}
        
        # category_deep_features: {类别名: 平均深度特征向量}
        # 存储每个类别的深度特征均值，用于快速相似度计算
        self.category_deep_features = {}

    def _get_cache_path(self, base_folder):
        """
        根据模板文件夹路径生成唯一的缓存文件路径
        
        参数:
            base_folder: 模板文件夹的绝对路径
            
        返回:
            缓存文件的完整路径
        """
        # 使用文件夹路径的哈希值作为缓存文件名的一部分
        # 这样不同的模板文件夹会有不同的缓存文件
        folder_hash = str(hash(os.path.abspath(base_folder)))
        cache_file = os.path.join(self.cache_dir, f'features_{folder_hash}.pkl')
        return cache_file

    def _save_features_cache(self, base_folder):
        """
        将提取的特征保存到本地缓存文件
        
        参数:
            base_folder: 模板文件夹路径
            
        说明:
            - SIFT keypoints 需要转换为字典格式才能序列化
            - 使用 pickle 格式保存，便于快速加载
        """
        cache_path = self._get_cache_path(base_folder)
        
        # 准备可序列化的缓存数据
        cache_data = {}
        
        # 遍历所有类别的模板
        for category, templates in self.category_templates.items():
            cache_data[category] = []
            
            for template in templates:
                # ========== 转换 SIFT keypoints ==========
                # cv2.KeyPoint 对象不能直接序列化，
                kp_data = []
                if template['kp_sift']:
                    for kp in template['kp_sift']:
                        kp_data.append({
                            'pt': kp.pt,              # 关键点坐标 (x, y)
                            'size': kp.size,          # 关键点的大小
                            'angle': kp.angle,        # 关键点的方向角
                            'response': kp.response,  # 关键点的响应强度
                            'octave': kp.octave,      # 关键点所在的金字塔层级
                            'class_id': kp.class_id   # 关键点的类别 ID
                        })
                
                # 构建可序列化的模板信息
                cache_item = {
                    'filename': template['filename'],
                    'path': template['path'],
                    'kp_sift': kp_data,
                    'desc_sift': template['desc_sift'],           # SIFT 描述符
                    'deep_features': template['deep_features']   # 深度学习特征
                }
                cache_data[category].append(cache_item)
        
        # 使用 pickle 保存到文件
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"特征已保存到: {cache_path}\n")

    def _load_features_cache(self, base_folder):
        """
        从缓存文件加载已提取的特征
        
        参数:
            base_folder: 模板文件夹路径
            
        返回:
            True 如果成功加载，False 如果缓存不存在或加载失败
        """
        cache_path = self._get_cache_path(base_folder)
        
        # 检查缓存文件是否存在
        if not os.path.exists(cache_path):
            return False
        
        try:
            # 从 pickle 文件加载缓存数据
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 恢复模板数据
            for category, templates in cache_data.items():
                self.category_templates[category] = []
                
                for template in templates:
                    # ========== 恢复 SIFT keypoints ==========
                    # 将字典转换回 cv2.KeyPoint 对象
                    kp_list = []
                    for kp_data in template['kp_sift']:
                        kp = cv2.KeyPoint(
                            x=kp_data['pt'][0],
                            y=kp_data['pt'][1],
                            size=kp_data['size'],
                            angle=kp_data['angle'],
                            response=kp_data['response'],
                            octave=kp_data['octave'],
                            class_id=kp_data['class_id']
                        )
                        kp_list.append(kp)
                    
                    # 构建模板信息字典
                    template_info = {
                        'filename': template['filename'],
                        'path': template['path'],
                        'kp_sift': kp_list,
                        'desc_sift': template['desc_sift'],
                        'deep_features': template['deep_features']
                    }
                    self.category_templates[category].append(template_info)
            
            # 计算每个类别的深度特征均值
            self._compute_category_deep_features()
            
            print(f"从缓存加载特征: {cache_path}")
            print(f"加载了 {len(self.category_templates)} 个类别\n")
            return True
        
        except Exception as e:
            print(f"加载缓存失败: {e}")
            return False

    def _compute_category_deep_features(self):
        """
        计算每个类别的深度特征均值
        
        说明:
            - 对每个类别的所有模板的深度特征求平均
            - 这样在分类时，只需与类别均值比较，而不是与每个模板比较
            - 大大加快了分类速度
        """
        for category, templates in self.category_templates.items():
            if templates:
                # 收集该类别所有模板的深度特征
                deep_feats = np.array([t['deep_features'] for t in templates])
                # 计算平均特征向量
                self.category_deep_features[category] = np.mean(deep_feats, axis=0)

    def preprocess_image(self, image):
        """
        预处理图像以提高特征提取效果
        
        参数:
            image: BGR 格式的 OpenCV 图像
            
        返回:
            预处理后的灰度图像
            
        处理步骤:
            1. 转换为灰度图
            2. 应用 CLAHE (对比度受限自适应直方图均衡化)
               - 增强局部对比度
               - 防止过度增强
        """
        # 转换为灰度图（SIFT 需要灰度图）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用 CLAHE 增强对比度
        # clipLimit: 对比度限制，值越大增强越强
        # tileGridSize: 分块大小，(8,8) 表示分成 8x8 的网格
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced

    def extract_sift_features(self, image):
        """
        提取图像的 SIFT 特征
        
        参数:
            image: BGR 格式的 OpenCV 图像
            
        返回:
            kp: 关键点列表 (cv2.KeyPoint 对象)
            desc: 描述符矩阵 (N x 128)，N 是关键点数量
        """
        # 预处理图像
        processed = self.preprocess_image(image)
        
        # 检测关键点并计算描述符
        # detectAndCompute 返回关键点和对应的描述符
        kp, desc = self.sift.detectAndCompute(processed, None)
        
        return kp, desc

    def extract_deep_features(self, image):
        """
        使用深度学习模型提取图像特征
        
        参数:
            image: BGR 格式的 OpenCV 图像
            
        返回:
            features: 1024 维的特征向量 (numpy 数组)
        """
        # 转换 BGR 到 RGB（PIL 需要 RGB 格式）
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 应用预处理（缩放、标准化等）
        img_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        
        # 前向传播，提取特征
        with torch.no_grad():
            features = self.model(img_tensor)
            # 移除批次维度并转换为 numpy 数组
            features = features.squeeze().cpu().numpy()
        
        return features

    def match_sift_features(self, desc1, desc2):
        """
        使用 FLANN 匹配两个 SIFT 描述符集合
        
        参数:
            desc1: 第一个描述符矩阵 (N1 x 128)
            desc2: 第二个描述符矩阵 (N2 x 128)
            
        返回:
            good_count: 好的匹配数量
            
        说明:
            - 使用 Lowe's ratio test 过滤匹配
            - 只保留距离比小于 0.7 的匹配（高质量匹配）
        """
        # 检查描述符是否为空
        if desc1 is None or desc2 is None:
            return 0
        
        try:
            # 使用 KNN 匹配，k=2 返回最近的两个匹配
            matches = self.flann.knnMatch(desc1, desc2, k=2)
            good_count = 0
            
            # 应用 Lowe's ratio test
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    # 如果最近邻的距离远小于次近邻，则认为是好的匹配
                    if m.distance < 0.7 * n.distance:
                        good_count += 1
            
            return good_count
        except:
            # 如果匹配失败，返回 0
            return 0

    def load_category_templates(self, base_folder, force_recompute=False):
        """
        加载所有类别的模板图像并提取特征
        
        参数:
            base_folder: 模板文件夹路径
                        结构: base_folder/
                             ├── category_1/
                             │   ├── img1.jpg
                             │   ├── img2.jpg
                             │   └── ...
                             ├── category_2/
                             │   └── ...
                             └── ...
            force_recompute: 是否强制重新计算特征（忽略缓存）
                           False: 优先从缓存加载
                           True: 强制重新计算并更新缓存
        """
        # 尝试从缓存加载特征
        if not force_recompute and self._load_features_cache(base_folder):
            return
        
        print(f"正在加载模板图像...")
        
        # 获取所有类别文件夹
        category_folders = [d for d in os.listdir(base_folder) 
                           if os.path.isdir(os.path.join(base_folder, d))]
        
        # 遍历每个类别
        for category in category_folders:
            category_path = os.path.join(base_folder, category)
            
            # 获取该类别下的所有图像文件
            image_files = [f for f in os.listdir(category_path) 
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            self.category_templates[category] = []
            
            # 处理每个模板图像
            for img_file in tqdm(image_files, desc=f"处理 {category}", leave=False):
                img_path = os.path.join(category_path, img_file)
                img = cv2.imread(img_path)
                
                # 跳过无法读取的图像
                if img is None:
                    continue
                
                # 提取 SIFT 特征
                kp_sift, desc_sift = self.extract_sift_features(img)
                
                # 提取深度学习特征
                deep_feat = self.extract_deep_features(img)
                
                # 构建模板信息字典
                template_info = {
                    'filename': img_file,
                    'path': img_path,
                    'kp_sift': kp_sift,
                    'desc_sift': desc_sift,
                    'deep_features': deep_feat
                }
                
                self.category_templates[category].append(template_info)
            
            print(f"  类别 '{category}': 加载 {len(image_files)} 张模板")
        
        print(f"总共加载 {len(category_folders)} 个类别")
        
        # 计算每个类别的深度特征均值
        self._compute_category_deep_features()
        
        # 保存特征到缓存
        self._save_features_cache(base_folder)

    def classify_single_image(self, query_img):
        """
        对单个查询图像进行分类
        
        参数:
            query_img: BGR 格式的查询图像
            
        返回:
            result: 分类结果字典，包含:
                - predicted_category: 预测的类别
                - confidence: 置信度 [0, 1]
                - all_scores: 所有类别的得分
                - details: 详细信息
                - top_3: 前三个预测结果
        """
        # 提取查询图像的特征
        kp_query, desc_query = self.extract_sift_features(query_img)
        deep_feat_query = self.extract_deep_features(query_img)
        
        category_scores = {}
        category_details = {}
        
        # 对每个类别进行匹配和评分
        for category, templates in self.category_templates.items():
            sift_scores = []
            
            # ========== SIFT 特征匹配 ==========
            # 与该类别的所有模板进行匹配
            for template in templates:
                score = self.match_sift_features(template['desc_sift'], desc_query)
                sift_scores.append(score)
            
            # ========== 深度学习特征相似度 ==========
            # 与该类别的平均特征进行比较
            category_mean_feat = self.category_deep_features[category]
            # 使用余弦相似度：1 - cosine_distance
            deep_sim = 1 - cosine(category_mean_feat, deep_feat_query)
            
            # 获取该类别的最高 SIFT 匹配分数
            max_sift_score = np.max(sift_scores) if sift_scores else 0
            
            # ========== 特征融合 ==========
            # 将 SIFT 和深度学习特征进行加权融合
            
            # 归一化 SIFT 分数到 [0,