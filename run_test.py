import paddle
import time
import threading
import traceback
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

from paddleocr import PaddleOCR
import cv2
import numpy as np
import yaml
import pymysql

from src.segmentation.segmenter_yolo import YOLOTileProcessor
from src.processor.img_cropper import crop_image_by_coordinates

from src.identification.OCRRecognizer import OCRRecognizer_ori
from src.identification.Recog import PharmaceuticalBottleClassifier
from src.identification.DrugMatcher import DrugMatcher


# ==========================================================
# 数据库初始化
# ==========================================================

def init_db(
        host="192.168.137.1",
        user="root",
        password="root",
        database="medicine_db",
        charset="utf8",
        port=3306
):
    """
    初始化数据库连接。

    重要：
    你的 MySQL 报错：
        Unknown character set: 'utf8mb4'

    说明当前 MySQL 版本不支持 utf8mb4。
    因此这里默认使用 utf8。
    """

    try:
        print(
            f"正在连接数据库: host={host}, database={database}, "
            f"charset={charset}, port={port}"
        )

        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            charset=charset,
            port=port,
            cursorclass=pymysql.cursors.DictCursor
        )

        print(f"✅ 数据库连接成功，字符集: {charset}")
        return connection

    except pymysql.err.OperationalError as e:
        err_msg = str(e)

        if "Unknown character set" in err_msg and "utf8mb4" in err_msg:
            print("⚠️ 当前 MySQL 不支持 utf8mb4，自动降级为 utf8 重新连接...")

            try:
                connection = pymysql.connect(
                    host=host,
                    user=user,
                    password=password,
                    database=database,
                    charset="utf8",
                    port=port,
                    cursorclass=pymysql.cursors.DictCursor
                )

                print("✅ 数据库连接成功，字符集已降级为 utf8")
                return connection

            except pymysql.MySQLError as e2:
                raise Exception(f"数据库使用 utf8 重新连接仍然失败: {e2}")

        raise Exception(f"数据库连接失败: {e}")

    except pymysql.MySQLError as e:
        raise Exception(f"数据库连接失败: {e}")


# ==========================================================
# OCR 初始化
# ==========================================================

def init_ocr_model(use_gpu=None, show_log=False):
    """
    初始化 PaddleOCR 模型。

    注意：
    如果没有 GPU，不建议开启 TensorRT / FP16。
    如果你初始化多个 OCR 模型，cpu_threads 不宜太大。
    """

    print("🚀 正在初始化 OCR 模型...")

    if use_gpu is None:
        use_gpu = paddle.is_compiled_with_cuda()

    ocr = PaddleOCR(
        # 如果药瓶/药袋文字没有明显旋转，False 会更快
        use_angle_cls=True,

        det_algorithm="DB",

        # 旧版本 PaddleOCR 支持 det_max_side_len
        det_max_side_len=480,

        det_score_mode="fast",
        det_db_thresh=0.6,
        det_db_box_thresh=0.6,

        lang='ch',

        # 多 OCR 实例时不要太大
        rec_batch_num=4,

        use_gpu=use_gpu,

        det=True,
        rec=True,

        # 多实例时建议 2 或 4，避免 8 个 OCR × 8 线程导致 CPU 过载
        cpu_threads=4,

        # CPU 模式下可开启
        # enable_mkldnn=True if not use_gpu else False,

        rec_score_thresh=0.6,

        # 没有 GPU 时不要打开这些
        # gpu_mem=1024,
        # use_tensorrt=True,
        # use_fp16=True,

        show_log=show_log,
    )

    print("✅ OCR 模型初始化完成")
    print(f"✅ 当前使用: {'GPU' if use_gpu else 'CPU'}")

    if not use_gpu:
        print("⚠️ 当前使用 CPU 模式，如需 GPU 请安装 paddlepaddle-gpu")

    return ocr


# ==========================================================
# 单张图片完整识别流程
# ==========================================================

def process_single_image(
        image,
        model=None,
        device=None,
        ocr_recognizer=None,
        drug_matcher=None,
        classifier=None,
        crop_region=None,
        tile_size=640,
        overlap=64,
        conf_thres=0.5,
        iou_thres=0.9,
        save_vis_path=None,

        # 分类器是否允许多线程并发
        classifier_thread_safe=True,

        # drug_matcher 是否线程安全
        # 如果 drug_matcher 使用同一个 pymysql 连接，建议 False
        drug_matcher_thread_safe=True,

        # 药袋使用几个 OCR 模型同步识别
        bag_ocr_num=2,

        # 超时
        bottle_timeout=120,
        bag_timeout=60,
):
    """
    对单张图像执行完整识别流程：

        YOLO检测
        → 裁剪药袋 / 药瓶
        → 多线程同步处理：
              药瓶：OCR → 药品匹配 → 分类
              药袋：两个 OCR 模型同步识别 → 患者姓名匹配
        → 融合药袋识别结果
        → 数据库比对

    返回:
        {
            patient_name,
            final_medicines,
            validation,
            vis_image,
            bottle_results,
            bag_results
        }
    """

    t_start = time.time()

    # ======================================================
    # 1. 读取图像
    # ======================================================

    if isinstance(image, str):
        frame = cv2.imread(image)
        if frame is None:
            raise FileNotFoundError(f"无法读取图像: {image}")
    else:
        frame = image.copy()

    if crop_region:
        frame = crop_image_by_coordinates(
            np.array([frame]),
            crop_region
        )[0]

    # ======================================================
    # 2. 解析 OCR 识别器
    # ======================================================

    if not ocr_recognizer:
        raise ValueError("必须提供 ocr_recognizer")

    if not isinstance(ocr_recognizer, (list, tuple)):
        ocr_list = [ocr_recognizer]
    else:
        ocr_list = list(ocr_recognizer)

    if len(ocr_list) == 0:
        raise ValueError("ocr_recognizer 不能为空")

    # 每个 OCR 实例对应一个锁，避免同一个 OCR 对象被多个线程同时调用
    ocr_locks = [
        threading.Lock()
        for _ in range(len(ocr_list))
    ]

    classifier_lock = threading.Lock()
    drug_matcher_lock = threading.Lock()

    # ======================================================
    # 3. YOLO 推理
    # ======================================================

    processor = YOLOTileProcessor(
        model=model,
        device=str(device),
        tile_size=tile_size,
        overlap=overlap,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        batch_size=1
    )

    result_frames, predictions = processor.process_frames_batch(
        [frame],
        output_type='overlay'
    )

    vis_image = result_frames[0]
    detections = predictions[0]

    if save_vis_path:
        cv2.imwrite(save_vis_path, vis_image)
        print(f"✅ 可视化结果已保存: {save_vis_path}")

    # ======================================================
    # 4. 裁剪药袋 / 药瓶
    # ======================================================

    cropped_bags, cropped_bottles = processor.crop_by_class(
        frame,
        detections,
        bag_class_id=1,
        bottle_class_id=0
    )

    print(f"📦 药袋: {len(cropped_bags)} 个 | 💊 药瓶: {len(cropped_bottles)} 个")

    if not cropped_bottles:
        print("⚠️ 未检测到药瓶")

    if not cropped_bags:
        print("⚠️ 未检测到药袋")

    # ======================================================
    # 5. drug_matcher 安全调用
    # ======================================================

    def safe_drug_match(text, match_type, threshold=50, limit=10):
        """
        根据 drug_matcher_thread_safe 决定是否加锁。

        注意：
        pymysql 的同一个 connection 不建议多线程同时使用，
        所以默认 drug_matcher_thread_safe=False。
        """

        if drug_matcher is None:
            return None

        if drug_matcher_thread_safe:
            return drug_matcher.match(
                text,
                match_type=match_type,
                threshold=threshold,
                limit=limit
            )
        else:
            with drug_matcher_lock:
                return drug_matcher.match(
                    text,
                    match_type=match_type,
                    threshold=threshold,
                    limit=limit
                )

    # ======================================================
    # 6. 患者姓名结果标准化与融合
    # ======================================================

    def normalize_patient_result(x):
        """
        将 drug_matcher.match 返回结果规范化为字符串。

        兼容：
            str
            list
            tuple
            dict
        """

        if x is None:
            return None

        if isinstance(x, str):
            return x

        if isinstance(x, dict):
            for key in ['patient_name', 'name', '姓名', 'patient']:
                if key in x:
                    return str(x[key])
            return str(x)

        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return None

            first = x[0]

            if isinstance(first, str):
                return first

            if isinstance(first, dict):
                for key in ['patient_name', 'name', '姓名', 'patient']:
                    if key in first:
                        return str(first[key])
                return str(first)

            if isinstance(first, (list, tuple)) and len(first) > 0:
                return str(first[0])

            return str(first)

        return str(x)

    def fuse_bag_patient_results(bag_results):
        """
        融合多个药袋 OCR 模型得到的患者识别结果。

        策略：
            1. 收集非空 patient_name
            2. 投票选择出现次数最多的
            3. 如果只有一个有效结果，直接使用
        """

        valid_names = []

        for r in bag_results:
            if r is None:
                continue

            patient_raw = r.get('patient_name', None)
            patient_norm = normalize_patient_result(patient_raw)

            if patient_norm:
                valid_names.append(patient_norm)

        if not valid_names:
            return None

        counter = {}
        for name in valid_names:
            counter[name] = counter.get(name, 0) + 1

        sorted_items = sorted(
            counter.items(),
            key=lambda x: x[1],
            reverse=True
        )

        best_name = sorted_items[0][0]

        print(f"👤 药袋患者候选投票: {counter}")
        print(f"👤 融合后患者姓名: {best_name}")

        return best_name

    # ======================================================
    # 7. 单个药瓶处理：OCR → 药品匹配 → 分类
    # ======================================================

    def process_bottle_single(args):
        """
        单个药瓶完整处理：
            OCR → 药品匹配 → 分类
        """

        idx, bottle, recognizer, ocr_lock = args

        result = {
            'type': 'bottle',
            'index': idx,
            'bottle_img': bottle,
            'ocr_text': None,
            'candidates': None,
            'final_medicine': None,
            'confidence': None,
            'status': ''
        }

        print(f"\n--- [药瓶线程] 药瓶 {idx + 1} 开始处理 ---")

        # -----------------------------
        # OCR
        # -----------------------------

        if recognizer is None:
            result['status'] = 'OCR识别器未提供'
            print(f"  药瓶 {idx + 1}: OCR识别器未提供")
            return result

        try:
            with ocr_lock:
                ocr_text = recognizer.recognize(bottle)

        except Exception as e:
            result['status'] = f'OCR异常: {e}'
            print(f"  药瓶 {idx + 1}: OCR异常: {e}")
            traceback.print_exc()
            return result

        if not ocr_text:
            result['status'] = 'OCR无结果'
            print(f"  药瓶 {idx + 1}: OCR无结果")
            return result

        result['ocr_text'] = ocr_text
        print(f"  药瓶 {idx + 1} OCR: {ocr_text}")

        # -----------------------------
        # 药品匹配
        # -----------------------------

        if drug_matcher is None:
            result['status'] = '匹配器未提供'
            print(f"  药瓶 {idx + 1}: 匹配器未提供")
            return result

        try:
            candidates = safe_drug_match(
                ocr_text,
                match_type='bottle',
                threshold=50,
                limit=10
            )

        except Exception as e:
            result['status'] = f'药品匹配异常: {e}'
            print(f"  药瓶 {idx + 1}: 药品匹配异常: {e}")
            traceback.print_exc()
            return result

        if not candidates:
            result['status'] = '无候选药品'
            print(f"  药瓶 {idx + 1}: 无候选药品")
            return result

        result['candidates'] = candidates
        print(f"  药瓶 {idx + 1} 候选: {candidates[:3]}...")

        # -----------------------------
        # 分类
        # -----------------------------

        if classifier is not None:
            try:
                if classifier_thread_safe:
                    cls_res = classifier.classify(
                        candidates,
                        bottle
                    )
                else:
                    with classifier_lock:
                        cls_res = classifier.classify(
                            candidates,
                            bottle
                        )

                best = cls_res.get('predicted_category', None)
                conf = cls_res.get('confidence', 0.0)

                if best is None:
                    best = candidates[0]
                    conf = 0.0
                    result['status'] = '分类结果为空，使用候选首位'
                else:
                    result['status'] = '完成'

                result['final_medicine'] = best
                result['confidence'] = conf

                print(
                    f"  药瓶 {idx + 1} 最终分类: {best} "
                    f"(conf={conf:.4f})"
                )

            except Exception as e:
                print(
                    f"  药瓶 {idx + 1}: 分类失败: {e}，"
                    f"使用第一个候选"
                )
                traceback.print_exc()

                result['final_medicine'] = candidates[0]
                result['confidence'] = 0.0
                result['status'] = f'分类失败，使用候选首位: {e}'

        else:
            result['final_medicine'] = candidates[0]
            result['confidence'] = 0.0
            result['status'] = '无分类器，使用候选首位'

            print(
                f"  药瓶 {idx + 1}: 无分类器，"
                f"使用候选首位 {candidates[0]}"
            )

        return result

    # ======================================================
    # 8. 单个药袋 OCR 处理
    # ======================================================

    def process_bag_single(args):
        """
        单个药袋 OCR 模型处理：
            recognize_yaodai → patient match
        """

        idx, bag_img, recognizer, ocr_lock = args

        result = {
            'type': 'bag',
            'index': idx,
            'bag_ocr_text': None,
            'patient_name': None,
            'status': ''
        }

        print(f"\n--- [药袋线程] 药袋 OCR 模型 {idx + 1} 开始处理 ---")

        if recognizer is None:
            result['status'] = '药袋OCR识别器未提供'
            print(f"  药袋 OCR 模型 {idx + 1}: 识别器未提供")
            return result

        try:
            with ocr_lock:
                bag_ocr = recognizer.recognize_yaodai(bag_img)

        except Exception as e:
            result['status'] = f'药袋OCR异常: {e}'
            print(f"  药袋 OCR 模型 {idx + 1}: OCR异常: {e}")
            traceback.print_exc()
            return result

        if not bag_ocr:
            result['status'] = '药袋OCR无结果'
            print(f"  药袋 OCR 模型 {idx + 1}: 无OCR结果")
            return result

        result['bag_ocr_text'] = bag_ocr
        print(f"  药袋 OCR 模型 {idx + 1} OCR: {bag_ocr}")

        if drug_matcher is None:
            result['status'] = '匹配器未提供'
            print(f"  药袋 OCR 模型 {idx + 1}: 匹配器未提供")
            return result

        try:
            patient_name = safe_drug_match(
                bag_ocr,
                match_type='bag',
                threshold=50,
                limit=10
            )

        except Exception as e:
            result['status'] = f'患者匹配异常: {e}'
            print(f"  药袋 OCR 模型 {idx + 1}: 患者匹配异常: {e}")
            traceback.print_exc()
            return result

        if not patient_name:
            result['status'] = '患者匹配无结果'
            print(f"  药袋 OCR 模型 {idx + 1}: 患者匹配无结果")
            return result

        result['patient_name'] = patient_name
        result['status'] = '完成'

        print(f"  药袋 OCR 模型 {idx + 1} 患者候选: {patient_name}")

        return result

    # ======================================================
    # 9. 构建药袋任务和药瓶任务
    # ======================================================

    bag_tasks = []
    bottle_tasks = []

    # -----------------------------
    # 药袋任务：默认使用前 2 个 OCR 模型
    # -----------------------------

    if cropped_bags and bag_ocr_num > 0:
        real_bag_ocr_num = min(
            bag_ocr_num,
            len(ocr_list)
        )

        bag_img = cropped_bags[0]

        for i in range(real_bag_ocr_num):
            bag_tasks.append(
                (
                    i,
                    bag_img,
                    ocr_list[i],
                    ocr_locks[i]
                )
            )

    # -----------------------------
    # 药瓶任务：优先使用剩余 OCR 模型
    # 例如 8 个 OCR：
    #   药袋用 0,1
    #   药瓶用 2,3,4,5,6,7
    # -----------------------------

    if cropped_bottles:
        if len(ocr_list) > bag_ocr_num:
            bottle_ocr_pool = ocr_list[bag_ocr_num:]
            bottle_lock_pool = ocr_locks[bag_ocr_num:]
        else:
            bottle_ocr_pool = ocr_list
            bottle_lock_pool = ocr_locks

        for i, bottle in enumerate(cropped_bottles):
            ocr_idx = i % len(bottle_ocr_pool)

            bottle_tasks.append(
                (
                    i,
                    bottle,
                    bottle_ocr_pool[ocr_idx],
                    bottle_lock_pool[ocr_idx]
                )
            )

    print(f"\n{'=' * 50}")
    print("[并行任务启动]")
    print(f"  药瓶任务数: {len(bottle_tasks)}")
    print(f"  药袋 OCR 任务数: {len(bag_tasks)}")
    print(f"  OCR 实例总数: {len(ocr_list)}")
    print(f"{'=' * 50}")

    # ======================================================
    # 10. 同一个线程池并行执行药瓶和药袋任务
    # ======================================================

    t_parallel_start = time.time()

    total_tasks = len(bottle_tasks) + len(bag_tasks)

    bottle_results_tmp = [None] * len(bottle_tasks)
    bag_results_tmp = [None] * len(bag_tasks)

    if total_tasks == 0:
        print("⚠️ 没有任何药瓶或药袋任务")
        return {
            'patient_name': None,
            'final_medicines': [],
            'validation': None,
            'vis_image': vis_image,
            'bottle_results': [],
            'bag_results': []
        }

    # 线程数量：
    # 不超过任务数，也不超过 OCR 数量
    max_workers = min(
        total_tasks,
        len(ocr_list)
    )
    max_workers = max(1, max_workers)

    all_futures = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:

        # 提交药瓶任务
        for task in bottle_tasks:
            future = pool.submit(process_bottle_single, task)
            all_futures[future] = ('bottle', task[0])

        # 提交药袋任务
        for task in bag_tasks:
            future = pool.submit(process_bag_single, task)
            all_futures[future] = ('bag', task[0])

        for future in as_completed(all_futures):
            task_type, idx = all_futures[future]

            try:
                if task_type == 'bottle':
                    r = future.result(timeout=bottle_timeout)
                    bottle_results_tmp[r['index']] = r

                elif task_type == 'bag':
                    r = future.result(timeout=bag_timeout)
                    bag_results_tmp[r['index']] = r

            except Exception as e:
                print(f"  {task_type} 任务 {idx + 1} 线程异常: {e}")
                traceback.print_exc()

    t_parallel_done = time.time()

    bottle_results = [
        r for r in bottle_results_tmp
        if r is not None
    ]

    bag_results = [
        r for r in bag_results_tmp
        if r is not None
    ]

    print(
        f"\n[并行任务完成] 总耗时: "
        f"{t_parallel_done - t_parallel_start:.2f}s"
    )

    # ======================================================
    # 11. 汇总药瓶结果
    # ======================================================

    valid_bottle_count = sum(
        1 for r in bottle_results
        if r.get('final_medicine') is not None
    )

    print(
        f"[药瓶结果] 有效 {valid_bottle_count}/{len(bottle_results)}"
    )

    final_medicines = [
        r['final_medicine']
        for r in bottle_results
        if r.get('final_medicine') is not None
    ]

    # 去重
    final_medicines = list(set(final_medicines))

    print(f"\n🏥 识别药品: {final_medicines}")

    # ======================================================
    # 12. 融合药袋识别结果
    # ======================================================

    patient_name = None

    if bag_results:
        patient_name = fuse_bag_patient_results(bag_results)
    else:
        print("⚠️ 药袋 OCR 没有有效结果")

    # ======================================================
    # 13. 数据库比对
    # ======================================================

    validation = None

    if patient_name and final_medicines and drug_matcher:
        try:
            if drug_matcher_thread_safe:
                validation = drug_matcher.check_patient_batch_medicines(
                    patient_name=patient_name,
                    batch_id=1,
                    expected_medicine_names=final_medicines
                )
            else:
                with drug_matcher_lock:
                    validation = drug_matcher.check_patient_batch_medicines(
                        patient_name=patient_name,
                        batch_id=1,
                        expected_medicine_names=final_medicines
                    )

            print(f"\n{'=' * 40}")
            print("【匹配结果】")
            print(f"  患者: {patient_name}")
            print(f"  识别药品: {final_medicines}")

            if validation['batch_exists']:
                print(f"  数据库药品: {validation['actual']}")

                if validation['matched']:
                    print("  ✅ 匹配正确")
                else:
                    print(f"  ❌ 缺少: {validation['missing']}")
                    print(f"     多余: {validation['extra']}")
            else:
                print("  ⚠️ 患者批次不存在")

            print(f"{'=' * 40}")

        except Exception as e:
            print(f"数据库比对异常: {e}")
            traceback.print_exc()

    else:
        print("⚠️ 无法比对：缺少患者姓名或药品")

    print(
        f"\n⏱ 总耗时: {time.time() - t_start:.2f}s "
        f"(药瓶+药袋并行处理: "
        f"{t_parallel_done - t_parallel_start:.2f}s)"
    )

    gc.collect()

    return {
        'patient_name': patient_name,
        'final_medicines': final_medicines,
        'validation': validation,
        'vis_image': vis_image,
        'bottle_results': bottle_results,
        'bag_results': bag_results,
    }


# ==========================================================
# 主程序
# ==========================================================

if __name__ == "__main__":

    # ======================================================
    # 1. 读取配置文件
    # ======================================================

    with open("./config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    db_config = config.get('db_config', {})

    # ======================================================
    # 2. 初始化数据库
    # ======================================================

    conn = init_db(
        host=db_config.get('host', '192.168.137.1'),
        user=db_config.get('user', 'root'),
        password=db_config.get('password', 'root'),
        database=db_config.get('database', 'medicine_db'),

        # 关键修改：
        # 你的 MySQL 不支持 utf8mb4，因此这里强制使用 utf8
        charset='utf8',

        port=db_config.get('port', 3306)
    )

    print("✅ 数据库连接成功")

    # ======================================================
    # 3. 初始化药品匹配器
    # ======================================================

    drug_matcher = DrugMatcher(
        conn,
        drug_table=config['table_config']['drug_table'],
        drug_column=config['table_config']['drug_column'],
        patient_table=config['table_config']['patient_table'],
        patient_column=config['table_config']['patient_column'],
        cache_drugs=True
    )

    # ======================================================
    # 4. 初始化分类器
    # ======================================================

    classifier = PharmaceuticalBottleClassifier(
        db_conn=conn,
        device='cpu'
    )

    # ======================================================
    # 5. 初始化多个 OCR 模型
    # ======================================================

    B_1 = init_ocr_model()
    B_2 = init_ocr_model()
    B_3 = init_ocr_model()
    B_4 = init_ocr_model()
    B_5 = init_ocr_model()
    B_6 = init_ocr_model()
    B_7 = init_ocr_model()
    B_8 = init_ocr_model()

    ocr_recognizer_1 = OCRRecognizer_ori(B_1)
    ocr_recognizer_2 = OCRRecognizer_ori(B_2)
    ocr_recognizer_3 = OCRRecognizer_ori(B_3)
    ocr_recognizer_4 = OCRRecognizer_ori(B_4)
    ocr_recognizer_5 = OCRRecognizer_ori(B_5)
    ocr_recognizer_6 = OCRRecognizer_ori(B_6)
    ocr_recognizer_7 = OCRRecognizer_ori(B_7)
    ocr_recognizer_8 = OCRRecognizer_ori(B_8)

    ocr_recognizers = (
        ocr_recognizer_1,
        ocr_recognizer_2,
        ocr_recognizer_3,
        ocr_recognizer_4,
        ocr_recognizer_5,
        ocr_recognizer_6,
        ocr_recognizer_7,
        ocr_recognizer_8,
    )

    # ======================================================
    # 6. 单图测试
    # ======================================================

    result = process_single_image(
        image="014.png",

        model="SSSS",

        # 如果 YOLO 模型使用 GPU，可以用 cuda:0
        # 如果没有 GPU，改成 cpu
        device="cuda:0",

        ocr_recognizer=ocr_recognizers,
        drug_matcher=drug_matcher,
        classifier=classifier,

        crop_region=None,

        tile_size=640,
        overlap=64,
        conf_thres=0.5,
        iou_thres=0.9,

        save_vis_path="./vis_result.jpg",

        # 药袋使用两个 OCR 模型同步识别
        bag_ocr_num=2,

        # 如果分类器多线程不稳定，改成 False
        classifier_thread_safe=True,

        # 重要：
        # 因为 drug_matcher 使用同一个 pymysql 连接，
        # 所以建议 False，内部会自动加锁，避免数据库连接并发错误
        drug_matcher_thread_safe=False,

        bottle_timeout=120,
        bag_timeout=60,
    )

    print("\n==============================")
    print("最终结果")
    print("==============================")
    print("患者姓名:", result['patient_name'])
    print("识别药品:", result['final_medicines'])
    # print("药袋结果:", result['bag_results'])
    # print("药瓶结果:", result['bottle_results'])
