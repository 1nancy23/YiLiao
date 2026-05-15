import time
import threading
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from src.segmentation.segmenter_yolo import YOLOTileProcessor
from src.processor.img_cropper import crop_image_by_coordinates
from src.stream.video_stream import VideoStream
from src.utils.img_utils import ImageProcessor


def run_realtime_detection(
        username="admin", password="a1b2c3d4",
        ip_address="192.168.1.168", port="554", channel="201",
        model=None, checkpoint_path=None, num_classes=2,
        ocr_recognizer='paddleocr', drug_matcher=None,
        classifier=None, length=None,
        tile_size=1024, overlap=64, target_fps=25, batch_frames=2,
        crop_region=None,
        output_type='overlay', overlay_alpha=0.5, display_scale=0.5,
        save_video=False, output_path="./realtime_output.mp4", save_fps=15,
        max_frames=None, device=None, trigger_interval=15,
):
    import gc

    ocr_recognizer_1, ocr_recognizer_2, ocr_recognizer_3, ocr_recognizer_4 = ocr_recognizer

    # ============================================================
    #  RTSP 连接
    # ============================================================
    rtsp_url = (f"rtsp://{username}:{password}@{ip_address}:{port}"
                f"/Streaming/Channels/{channel}")
    print(f"📡 连接: rtsp://{username}:****@{ip_address}:{port}"
          f"/Streaming/Channels/{channel}")

    video_stream = VideoStream(rtsp_url, target_fps=target_fps)
    video_stream.start()

    print("⏳ 正在连接RTSP流...")
    timeout = 10
    start_time = time.time()
    while not video_stream.is_connected() and time.time() - start_time < timeout:
        time.sleep(0.1)
    if not video_stream.is_connected():
        print("❌ 连接超时！")
        video_stream.stop()
        return

    # ============================================================
    #  YOLO 处理器
    # ============================================================
    processor = YOLOTileProcessor(
        model=model, device=str(device),
        tile_size=640, overlap=64,
        conf_thres=0.5, iou_thres=0.9, batch_size=1
    )

    # ============================================================
    #  视频录制器
    # ============================================================
    video_writer = None
    if save_video:
        out_w = (video_stream.stream_reader.frame_width * 2
                 if output_type == 'side_by_side'
                 else video_stream.stream_reader.frame_width)
        out_h = video_stream.stream_reader.frame_height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, save_fps, (out_w, out_h))

    # ============================================================
    #  显示 & 统计
    # ============================================================
    output_types = ['overlay', 'color', 'side_by_side', 'gray']
    current_type_idx = (output_types.index(output_type)
                        if output_type in output_types else 0)
    fps_counter = deque(maxlen=3)
    frame_count = 0
    last_trigger_time = time.time()

    print("\n" + "=" * 60)
    print("🚀 实时目标检测已启动 (YOLOv8 + 切片处理)")
    print(f"  批量帧数: {batch_frames}")
    print(f"  自动触发间隔: {trigger_interval}秒")
    print("  按键: 'q'-退出  's'-截图  'r'-切换模式")
    print("=" * 60 + "\n")

    # 缓存最近批次
    max_batches_kept = 2
    n_batch_frames = deque(maxlen=max_batches_kept)
    n_batch_predictions = deque(maxlen=max_batches_kept)

    # ============================================================
    # ★ 线程安全控制
    # ============================================================
    is_processing = threading.Event()
    processing_done = threading.Event()   # ← 新增：用于 finally 等待完成
    processing_done.set()                 # 初始状态：已完成（无任务在跑）
    ocr_locks = [threading.Lock() for _ in range(len(ocr_recognizer))]
    classifier_lock = threading.Lock()    # ← 新增：分类器独占锁

    # ----------------------------------------------------------
    # ★ 阶段1：单个药瓶 OCR + 匹配（多线程并行，不做分类）
    # ----------------------------------------------------------
    def ocr_and_match_single(args):
        """
        只做 OCR 识别 + 药品候选匹配，不做 classifier 分类
        返回: {index, bottle_img, ocr_text, candidates, status}
        """
        idx, bottle, recognizer, lock = args
        result = {
            'index': idx,
            'bottle_img': bottle,       # 保留图像，供阶段2分类使用
            'ocr_text': None,
            'candidates': None,
            'status': ''
        }

        with lock:                      # 同一识别器同时只有一个线程使用
            print(f"\n--- [阶段1] 药瓶 {idx + 1} OCR识别 ---")

            if recognizer is None:
                result['status'] = 'OCR识别器未提供'
                return result

            # OCR 识别
            try:
                ocr_text = recognizer.recognize(bottle)
            except Exception as e:
                print(f"  OCR异常: {e}")
                result['status'] = f'OCR异常: {e}'
                return result

            if not ocr_text:
                result['status'] = 'OCR无结果'
                print(f"  OCR 无结果")
                return result

            result['ocr_text'] = ocr_text
            # print(f"  OCR: {ocr_text}")

            # 药品匹配
            if drug_matcher is None:
                result['status'] = '匹配器未提供'
                return result

            candidates = drug_matcher.match(
                ocr_text, match_type='bottle', threshold=50, limit=10
            )
            if not candidates:
                result['status'] = '无候选'
                print(f"  无候选")
                return result

            result['candidates'] = candidates
            result['status'] = '待分类'
            print(f"  候选: {candidates[:3]}...")

        return result

    # ----------------------------------------------------------
    # ★ 阶段2：顺序分类（单线程，在后台线程中顺序执行）
    # ----------------------------------------------------------
    def classify_all_bottles(ocr_results):
        """
        对所有有候选结果的药瓶，逐个调用 classifier.classify
        单线程顺序执行，避免 classifier 并发问题
        """
        final_medicines = []

        # 过滤出有候选结果的
        valid_results = [r for r in ocr_results if r['candidates'] is not None]

        if not valid_results:
            print("[阶段2] 没有需要分类的药瓶")
            return final_medicines

        print(f"\n{'='*40}")
        print(f"[阶段2] 开始特征分类，共 {len(valid_results)} 个药瓶")
        print(f"{'='*40}")

        with classifier_lock:           # 分类器独占，防止其他地方并发调用
            for r in valid_results:
                idx = r['index']
                bottle_img = r['bottle_img']
                candidates = r['candidates']

                if classifier is not None:
                    try:
                        res = classifier.classify(candidates, bottle_img)
                        best = res['predicted_category']
                        conf = res['confidence']
                        print(f"  药瓶 {idx + 1}: {best} (置信度 {conf:.4f})")
                        final_medicines.append(best)
                    except Exception as e:
                        print(f"  药瓶 {idx + 1} 分类失败: {e}，使用第一个候选")
                        final_medicines.append(candidates[0])
                else:
                    # 无分类器，直接用第一个候选
                    print(f"  药瓶 {idx + 1}: 无分类器，使用候选首位 {candidates[0]}")
                    final_medicines.append(candidates[0])

        return final_medicines

    # ----------------------------------------------------------
    # ★ 后台处理主函数
    # ----------------------------------------------------------
    def process_batch_in_background(snapshot_frames, snapshot_predictions):
        """
        后台线程：阶段1(多线程OCR) → 阶段2(单线程分类) → 数据库比对
        """
        try:
            if not snapshot_frames:
                return

            t_start = time.time()

            # ========== 找最清晰帧 ==========
            best_frame, frame_pos = ImageProcessor.select_sharpest_image(
                snapshot_frames
            )
            batch_idx, frame_idx = frame_pos
            print(f"最清晰帧: 批次 {batch_idx + 1}/{len(snapshot_frames)} "
                  f"中的第 {frame_idx + 1} 张")

            detections = snapshot_predictions[batch_idx][frame_idx]

            # ========== 裁剪药袋和药瓶 ==========
            cropped_bags, cropped_bottles = processor.crop_by_class(
                best_frame, detections, bag_class_id=1, bottle_class_id=0
            )
            print(f"药袋: {len(cropped_bags)} 个, 药瓶: {len(cropped_bottles)} 个")

            if not (cropped_bags and cropped_bottles):
                print("未同时检测到药袋和药瓶")
                return

            # =====================================================
            # ★★★ 阶段1：多线程并行 OCR + 匹配 ★★★
            # =====================================================
            print(f"\n{'='*40}")
            print(f"[阶段1] 多线程OCR识别，{len(cropped_bottles)} 个药瓶，"
                  f"{len(ocr_recognizer)} 个识别器")
            print(f"{'='*40}")

            tasks = [
                (i, bottle,
                 ocr_recognizer[i % len(ocr_recognizer)],
                 ocr_locks[i % len(ocr_recognizer)])
                for i, bottle in enumerate(cropped_bottles)
            ]

            ocr_results = [None] * len(cropped_bottles)

            with ThreadPoolExecutor(max_workers=len(ocr_recognizer)) as pool:
                futures = {
                    pool.submit(ocr_and_match_single, task): task[0]
                    for task in tasks
                }
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        ocr_results[result['index']] = result
                    except Exception as e:
                        idx = futures[future]
                        print(f"  药瓶 {idx + 1} 线程异常: {e}")

            # 移除 None（线程异常的）
            ocr_results = [r for r in ocr_results if r is not None]

            t_ocr_done = time.time()
            print(f"\n[阶段1完成] OCR耗时: {t_ocr_done - t_start:.2f}s, "
                  f"有效结果: {sum(1 for r in ocr_results if r['candidates'])}"
                  f"/{len(ocr_results)}")

            # =====================================================
            # ★★★ 阶段2：单线程顺序分类 ★★★
            # =====================================================
            final_medicines = classify_all_bottles(ocr_results)
            # del cropped_bottles
            # del ocr_results
            t_cls_done = time.time()
            print(f"\n[阶段2完成] 分类耗时: {t_cls_done - t_ocr_done:.2f}s")

            # 去重
            final_medicines = list(set(final_medicines))
            print(f"\n🏥 识别药品: {final_medicines}")

            # =====================================================
            # 药袋 OCR（单线程，用第一个识别器）
            # =====================================================
            patient_name = None
            if cropped_bags and ocr_recognizer_1 and drug_matcher:
                with ocr_locks[0]:      # 锁住第一个识别器
                    bag_ocr = ocr_recognizer_1.recognize_yaodai(cropped_bags[0])
                if bag_ocr:
                    patient_name = drug_matcher.match(
                        bag_ocr, match_type='bag', threshold=50, limit=10
                    )
                    print(f"👤 患者姓名: {patient_name}")

            # =====================================================
            # 数据库比对
            # =====================================================
            if patient_name and final_medicines and drug_matcher:
                validation = drug_matcher.check_patient_batch_medicines(
                    patient_name=patient_name,
                    batch_id=1,
                    expected_medicine_names=final_medicines
                )
                print(f"\n{'='*40}")
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
                print(f"{'='*40}")
            else:
                print("无法比对：缺少患者姓名或药品")

            print(f"\n⏱ 总耗时: {time.time() - t_start:.2f}s "
                  f"(OCR: {t_ocr_done - t_start:.2f}s + "
                  f"分类: {t_cls_done - t_ocr_done:.2f}s)")

            # 显式释放大图
            del cropped_bottles, cropped_bags, ocr_results
            gc.collect()

        except Exception as e:
            print(f"❌ 后台处理异常: {e}")
            traceback.print_exc()
        finally:
            is_processing.clear()
            processing_done.set()       # ← 通知 finally 块：处理已完成
            print("🔓 后台处理完成")

    # ============================================================
    #  主循环
    # ============================================================
    try:
        while video_stream.running:
            frames = video_stream.get_batch(batch_frames)
            if not frames:
                continue

            if crop_region:
                frames = list(
                    crop_image_by_coordinates(np.array(frames), crop_region)
                )

            # YOLO推理
            process_start = time.time()
            result_frames, predictions = processor.process_frames_batch(
                frames, output_type=output_types[current_type_idx]
            )
            process_time = time.time() - process_start

            n_batch_frames.append(frames)
            n_batch_predictions.append(predictions)

            # ★ 非阻塞触发
            current_time = time.time()
            if current_time - last_trigger_time >= trigger_interval:
                if not is_processing.is_set():
                    is_processing.set()
                    processing_done.clear()

                    # 快照（深拷贝解耦）
                    snapshot_frames = [list(batch) for batch in n_batch_frames]
                    snapshot_preds = [list(batch) for batch in n_batch_predictions]

                    bg_thread = threading.Thread(
                        target=process_batch_in_background,
                        args=(snapshot_frames, snapshot_preds),
                        daemon=True
                    )
                    bg_thread.start()
                    print(f"\n⏰ 定时触发 → 后台线程已启动")
                    last_trigger_time = current_time
                else:
                    print(f"\n⏰ 上一轮仍在处理，跳过")
                    last_trigger_time = current_time

            # FPS
            batch_fps = len(frames) / max(process_time, 0.001)
            fps_counter.append(batch_fps)
            current_fps = np.mean(list(fps_counter))

            # 显示
            for result_frame in result_frames:
                if video_writer:
                    video_writer.write(result_frame)

                if display_scale != 1.0:
                    h, w = result_frame.shape[:2]
                    display_frame = cv2.resize(
                        result_frame,
                        (int(w * display_scale), int(h * display_scale))
                    )
                else:
                    display_frame = result_frame.copy()

                status = "🔄处理中" if is_processing.is_set() else "⏳等待"
                info = (f"FPS: {current_fps:.1f} | "
                        f"Mode: {output_types[current_type_idx]} | "
                        f"Frame: {frame_count} | {status}")
                cv2.putText(display_frame, info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow('YOLO Realtime Detection', display_frame)
                frame_count += 1

            if max_frames and frame_count >= max_frames:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"screenshot_{int(time.time())}.png",
                            result_frames[-1])
            elif key == ord('r'):
                current_type_idx = (current_type_idx + 1) % len(output_types)

            del frames

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    finally:
        # ★ 正确等待后台线程完成
        if not processing_done.is_set():
            print("⏳ 等待后台处理完成（最多20秒）...")
            processing_done.wait(timeout=20)
            if not processing_done.is_set():
                print("⚠️ 后台线程超时，强制退出")

        video_stream.stop()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print(f"✅ 完成！总帧数: {frame_count}")
