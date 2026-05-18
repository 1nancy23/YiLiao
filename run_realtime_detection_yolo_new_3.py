# import time
# import threading
# import traceback
# from collections import deque
# from concurrent.futures import ThreadPoolExecutor, as_completed

# import cv2
# import numpy as np

# from src.segmentation.segmenter_yolo import YOLOTileProcessor
# from src.processor.img_cropper import crop_image_by_coordinates
# from src.stream.video_stream import VideoStream
# from src.utils.img_utils import ImageProcessor


# def run_realtime_detection(
#         username="admin",
#         password="a1b2c3d4",
#         ip_address="192.168.1.168",
#         port="554",
#         channel="201",
#         model=None,
#         checkpoint_path=None,
#         num_classes=2,
#         ocr_recognizer='paddleocr',
#         drug_matcher=None,
#         classifier=None,
#         length=None,
#         tile_size=1024,
#         overlap=64,
#         target_fps=25,
#         batch_frames=2,
#         crop_region=None,
#         output_type='overlay',
#         overlay_alpha=0.5,
#         display_scale=0.5,
#         save_video=False,
#         output_path="./realtime_output.mp4",
#         save_fps=15,
#         max_frames=None,
#         device=None,
#         trigger_interval=15,

#         # 新增参数：
#         # 如果 classifier.classify 支持多线程，设为 True
#         # 如果不支持多线程，设为 False，会自动加锁
#         classifier_thread_safe=True,
# ):
#     import gc

#     # ============================================================
#     # OCR 识别器处理
#     # ============================================================

#     if not isinstance(ocr_recognizer, (list, tuple)):
#         ocr_recognizer = [ocr_recognizer]

#     if len(ocr_recognizer) == 0:
#         raise ValueError("ocr_recognizer 不能为空，请至少提供一个 OCR 识别器实例。")

#     # 为了兼容你原来使用 ocr_recognizer_1 的逻辑
#     ocr_recognizer_1 = ocr_recognizer[0]

#     # ============================================================
#     # RTSP 连接
#     # ============================================================

#     rtsp_url = (
#         f"rtsp://{username}:{password}@{ip_address}:{port}"
#         f"/Streaming/Channels/{channel}"
#     )

#     print(
#         f"📡 连接: rtsp://{username}:****@{ip_address}:{port}"
#         f"/Streaming/Channels/{channel}"
#     )

#     video_stream = VideoStream(rtsp_url, target_fps=target_fps)
#     video_stream.start()

#     print("⏳ 正在连接RTSP流...")

#     timeout = 10
#     start_time = time.time()

#     while not video_stream.is_connected() and time.time() - start_time < timeout:
#         time.sleep(0.1)

#     if not video_stream.is_connected():
#         print("❌ 连接超时！")
#         video_stream.stop()
#         return

#     # ============================================================
#     # YOLO 处理器
#     # ============================================================

#     processor = YOLOTileProcessor(
#         model=model,
#         device=str(device),
#         tile_size=640,
#         overlap=64,
#         conf_thres=0.5,
#         iou_thres=0.85,
#         batch_size=1
#     )

#     # ============================================================
#     # 视频录制器
#     # ============================================================

#     video_writer = None

#     if save_video:
#         out_w = (
#             video_stream.stream_reader.frame_width * 2
#             if output_type == 'side_by_side'
#             else video_stream.stream_reader.frame_width
#         )

#         out_h = video_stream.stream_reader.frame_height

#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#         video_writer = cv2.VideoWriter(
#             output_path,
#             fourcc,
#             save_fps,
#             (out_w, out_h)
#         )

#     # ============================================================
#     # 显示与统计
#     # ============================================================

#     output_types = ['overlay', 'color', 'side_by_side', 'gray']

#     current_type_idx = (
#         output_types.index(output_type)
#         if output_type in output_types
#         else 0
#     )

#     fps_counter = deque(maxlen=3)
#     frame_count = 0
#     last_trigger_time = time.time()

#     print("\n" + "=" * 60)
#     print("🚀 实时目标检测已启动 YOLOv8 + 切片处理")
#     print(f"  批量帧数: {batch_frames}")
#     print(f"  自动触发间隔: {trigger_interval} 秒")
#     print("  按键: 'q'-退出  's'-截图  'r'-切换模式")
#     print("=" * 60 + "\n")

#     # 缓存最近批次
#     max_batches_kept = 2
#     n_batch_frames = deque(maxlen=max_batches_kept)
#     n_batch_predictions = deque(maxlen=max_batches_kept)

#     # ============================================================
#     # 线程安全控制
#     # ============================================================

#     is_processing = threading.Event()

#     processing_done = threading.Event()
#     processing_done.set()

#     # 每个 OCR 实例对应一个锁
#     # 防止同一个 OCR 实例被多个线程同时调用
#     ocr_locks = [
#         threading.Lock()
#         for _ in range(len(ocr_recognizer))
#     ]

#     # 分类器锁
#     classifier_lock = threading.Lock()

#     # ============================================================
#     # 单个药瓶完整处理函数：
#     # OCR → 药品匹配 → 分类
#     # ============================================================

#     def process_bottle_single(args):
#         """
#         单个药瓶完整处理流程：

#         1. OCR 识别
#         2. 药品候选匹配 drug_matcher.match
#         3. 分类确认 classifier.classify

#         返回:
#             {
#                 'index': idx,
#                 'ocr_text': ocr_text,
#                 'candidates': candidates,
#                 'final_medicine': best,
#                 'confidence': conf,
#                 'status': status
#             }
#         """

#         idx, bottle, recognizer, ocr_lock = args

#         result = {
#             'index': idx,
#             'ocr_text': None,
#             'candidates': None,
#             'final_medicine': None,
#             'confidence': None,
#             'status': ''
#         }

#         print(f"\n--- [药瓶线程] 药瓶 {idx + 1} 开始处理 ---")

#         # ========================================================
#         # 1. OCR 识别
#         # ========================================================

#         if recognizer is None:
#             result['status'] = 'OCR识别器未提供'
#             print(f"  药瓶 {idx + 1}: OCR识别器未提供")
#             return result

#         try:
#             with ocr_lock:
#                 ocr_text = recognizer.recognize(bottle)

#         except Exception as e:
#             result['status'] = f'OCR异常: {e}'
#             print(f"  药瓶 {idx + 1}: OCR异常: {e}")
#             traceback.print_exc()
#             return result

#         if not ocr_text:
#             result['status'] = 'OCR无结果'
#             print(f"  药瓶 {idx + 1}: OCR无结果")
#             return result

#         result['ocr_text'] = ocr_text
#         print(f"  药瓶 {idx + 1} OCR: {ocr_text}")

#         # ========================================================
#         # 2. 药品候选匹配
#         # ========================================================

#         if drug_matcher is None:
#             result['status'] = '药品匹配器未提供'
#             print(f"  药瓶 {idx + 1}: 药品匹配器未提供")
#             return result

#         try:
#             candidates = drug_matcher.match(
#                 ocr_text,
#                 match_type='bottle',
#                 threshold=50,
#                 limit=10
#             )

#         except Exception as e:
#             result['status'] = f'药品匹配异常: {e}'
#             print(f"  药瓶 {idx + 1}: 药品匹配异常: {e}")
#             traceback.print_exc()
#             return result

#         if not candidates:
#             result['status'] = '无候选药品'
#             print(f"  药瓶 {idx + 1}: 无候选药品")
#             return result

#         result['candidates'] = candidates
#         print(f"  药瓶 {idx + 1} 候选: {candidates[:3]}...")

#         # ========================================================
#         # 3. 分类器确认
#         # ========================================================

#         if classifier is not None:
#             try:
#                 if classifier_thread_safe:
#                     cls_res = classifier.classify(candidates, bottle)
#                 else:
#                     with classifier_lock:
#                         cls_res = classifier.classify(candidates, bottle)

#                 best = cls_res.get('predicted_category', None)
#                 conf = cls_res.get('confidence', 0.0)

#                 if best is None:
#                     best = candidates[0]
#                     conf = 0.0
#                     result['status'] = '分类结果为空，使用候选首位'
#                 else:
#                     result['status'] = '完成'

#                 result['final_medicine'] = best
#                 result['confidence'] = conf

#                 print(
#                     f"  药瓶 {idx + 1} 最终分类: {best}, "
#                     f"置信度: {conf:.4f}"
#                 )

#             except Exception as e:
#                 print(
#                     f"  药瓶 {idx + 1}: 分类失败: {e}，"
#                     f"使用第一个候选"
#                 )
#                 traceback.print_exc()

#                 result['final_medicine'] = candidates[0]
#                 result['confidence'] = 0.0
#                 result['status'] = f'分类失败，使用候选首位: {e}'

#         else:
#             result['final_medicine'] = candidates[0]
#             result['confidence'] = 0.0
#             result['status'] = '无分类器，使用候选首位'

#             print(
#                 f"  药瓶 {idx + 1}: 无分类器，"
#                 f"使用候选首位 {candidates[0]}"
#             )

#         return result

#     # ============================================================
#     # 后台处理主函数
#     # ============================================================

#     def process_batch_in_background(snapshot_frames, snapshot_predictions):
#         """
#         后台线程处理流程：

#         1. 从缓存帧中选择最清晰帧
#         2. 根据 YOLO 检测结果裁剪药袋和药瓶
#         3. 多线程处理每个药瓶：
#            OCR → 药品匹配 → 分类
#         4. 药袋 OCR 识别患者信息
#         5. 数据库比对
#         """

#         try:
#             if not snapshot_frames:
#                 return

#             t_start = time.time()

#             # ====================================================
#             # 选择最清晰帧
#             # ====================================================

#             best_frame, frame_pos = ImageProcessor.select_sharpest_image(
#                 snapshot_frames
#             )

#             batch_idx, frame_idx = frame_pos

#             print(
#                 f"最清晰帧: 批次 {batch_idx + 1}/{len(snapshot_frames)} "
#                 f"中的第 {frame_idx + 1} 张"
#             )

#             detections = snapshot_predictions[batch_idx][frame_idx]

#             # ====================================================
#             # 裁剪药袋和药瓶
#             # ====================================================

#             cropped_bags, cropped_bottles, cropped_shuyes = processor.crop_by_class(
#                 best_frame,
#                 detections,
#                 bag_class_id=1,
#                 bottle_class_id=0,
#                 shuye_class_id=2,
#             )

#             print(
#                 f"药袋: {len(cropped_bags)} 个, "
#                 f"药瓶: {len(cropped_bottles)} 个"
#                 f"输液袋:{len(cropped_shuyes)} 个"
#             )

#             if not cropped_bags or not cropped_bottles:
#                 print("未同时检测到药袋和药瓶")
#                 return

#             # ====================================================
#             # 多线程处理药瓶：OCR + 匹配 + 分类
#             # ====================================================

#             print(f"\n{'=' * 40}")
#             print(
#                 f"[多线程药瓶处理] 共 {len(cropped_bottles)} 个药瓶，"
#                 f"{len(ocr_recognizer)} 个 OCR 识别器"
#             )
#             print(f"{'=' * 40}")

#             bottle_tasks = [
#                 (
#                     i,
#                     bottle,
#                     ocr_recognizer[i % len(ocr_recognizer)],
#                     ocr_locks[i % len(ocr_recognizer)]
#                 )
#                 for i, bottle in enumerate(cropped_bottles)
#             ]

#             bottle_results = [None] * len(cropped_bottles)

#             # 线程数不要超过药瓶数和 OCR 实例数
#             max_workers = min(
#                 len(cropped_bottles),
#                 len(ocr_recognizer)
#             )

#             # 如果只有一个药瓶，仍然可以正常运行
#             max_workers = max(1, max_workers)

#             with ThreadPoolExecutor(max_workers=max_workers) as pool:
#                 futures = {
#                     pool.submit(process_bottle_single, task): task[0]
#                     for task in bottle_tasks
#                 }

#                 for future in as_completed(futures):
#                     idx = futures[future]

#                     try:
#                         result = future.result(timeout=90)
#                         bottle_results[result['index']] = result

#                     except Exception as e:
#                         print(f"  药瓶 {idx + 1} 线程异常: {e}")
#                         traceback.print_exc()

#             bottle_results = [
#                 r for r in bottle_results
#                 if r is not None
#             ]

#             t_bottle_done = time.time()

#             valid_count = sum(
#                 1 for r in bottle_results
#                 if r.get('final_medicine') is not None
#             )

#             print(
#                 f"\n[药瓶多线程处理完成] "
#                 f"耗时: {t_bottle_done - t_start:.2f}s, "
#                 f"有效结果: {valid_count}/{len(bottle_results)}"
#             )

#             final_medicines = [
#                 r['final_medicine']
#                 for r in bottle_results
#                 if r.get('final_medicine') is not None
#             ]

#             # 去重
#             final_medicines = list(set(final_medicines))

#             print(f"\n🏥 识别药品: {final_medicines}")

#             # ====================================================
#             # 药袋 OCR 识别患者姓名
#             # ====================================================

#             patient_name = None

#             if cropped_bags and ocr_recognizer_1 and drug_matcher:
#                 try:
#                     print("\n--- [药袋 OCR] 开始识别药袋 ---")

#                     with ocr_locks[0]:
#                         bag_ocr = ocr_recognizer_1.recognize_yaodai(
#                             cropped_bags[0]
#                         )

#                     if bag_ocr:
#                         patient_name = drug_matcher.match(
#                             bag_ocr,
#                             match_type='bag',
#                             threshold=50,
#                             limit=10
#                         )

#                         print(f"👤 患者姓名: {patient_name}")
#                     else:
#                         print("药袋 OCR 无结果")

#                 except Exception as e:
#                     print(f"药袋 OCR 或患者匹配异常: {e}")
#                     traceback.print_exc()

#             # ====================================================
#             # 数据库比对
#             # ====================================================

#             if patient_name and final_medicines and drug_matcher:
#                 try:
#                     validation = drug_matcher.check_patient_batch_medicines(
#                         patient_name=patient_name,
#                         batch_id=1,
#                         expected_medicine_names=final_medicines
#                     )

#                     print(f"\n{'=' * 40}")
#                     print("【匹配结果】")
#                     print(f"  患者: {patient_name}")
#                     print(f"  识别药品: {final_medicines}")

#                     if validation['batch_exists']:
#                         print(f"  数据库药品: {validation['actual']}")

#                         if validation['matched']:
#                             print("  ✅ 匹配正确")
#                         else:
#                             print(f"  ❌ 缺少: {validation['missing']}")
#                             print(f"     多余: {validation['extra']}")
#                     else:
#                         print("  ⚠️ 患者批次不存在")

#                     print(f"{'=' * 40}")

#                 except Exception as e:
#                     print(f"数据库比对异常: {e}")
#                     traceback.print_exc()

#             else:
#                 print("无法比对：缺少患者姓名或药品")

#             print(
#                 f"\n⏱ 总耗时: {time.time() - t_start:.2f}s "
#                 f"(药瓶 OCR + 匹配 + 分类: "
#                 f"{t_bottle_done - t_start:.2f}s)"
#             )

#             # 释放大对象
#             del cropped_bottles
#             del cropped_bags
#             del bottle_results
#             gc.collect()

#         except Exception as e:
#             print(f"❌ 后台处理异常: {e}")
#             traceback.print_exc()

#         finally:
#             is_processing.clear()
#             processing_done.set()
#             print("🔓 后台处理完成")

#     # ============================================================
#     # 主循环
#     # ============================================================

#     try:
#         while video_stream.running:
#             frames = video_stream.get_batch(batch_frames)

#             if not frames:
#                 continue

#             # ====================================================
#             # 可选 ROI 裁剪
#             # ====================================================

#             if crop_region:
#                 frames = list(
#                     crop_image_by_coordinates(
#                         np.array(frames),
#                         crop_region
#                     )
#                 )

#             # ====================================================
#             # YOLO 推理
#             # ====================================================

#             process_start = time.time()

#             result_frames, predictions = processor.process_frames_batch(
#                 frames,
#                 output_type=output_types[current_type_idx]
#             )

#             process_time = time.time() - process_start

#             # 缓存最近批次
#             n_batch_frames.append(frames)
#             n_batch_predictions.append(predictions)

#             # ====================================================
#             # 定时触发后台 OCR + 匹配 + 分类
#             # ====================================================

#             current_time = time.time()

#             if current_time - last_trigger_time >= trigger_interval:
#                 if not is_processing.is_set():
#                     is_processing.set()
#                     processing_done.clear()

#                     # 快照，避免后台线程和主线程共享 deque
#                     snapshot_frames = [
#                         list(batch)
#                         for batch in n_batch_frames
#                     ]

#                     snapshot_preds = [
#                         list(batch)
#                         for batch in n_batch_predictions
#                     ]

#                     bg_thread = threading.Thread(
#                         target=process_batch_in_background,
#                         args=(snapshot_frames, snapshot_preds),
#                         daemon=True
#                     )

#                     bg_thread.start()

#                     print("\n⏰ 定时触发 → 后台线程已启动")

#                     last_trigger_time = current_time

#                 else:
#                     print("\n⏰ 上一轮仍在处理，跳过")
#                     last_trigger_time = current_time

#             # ====================================================
#             # FPS 统计
#             # ====================================================

#             batch_fps = len(frames) / max(process_time, 0.001)
#             fps_counter.append(batch_fps)

#             current_fps = np.mean(list(fps_counter))

#             # ====================================================
#             # 显示与保存
#             # ====================================================

#             for result_frame in result_frames:
#                 if video_writer:
#                     video_writer.write(result_frame)

#                 if display_scale != 1.0:
#                     h, w = result_frame.shape[:2]

#                     display_frame = cv2.resize(
#                         result_frame,
#                         (
#                             int(w * display_scale),
#                             int(h * display_scale)
#                         )
#                     )
#                 else:
#                     display_frame = result_frame.copy()

#                 status = "处理中" if is_processing.is_set() else "等待"

#                 info = (
#                     f"FPS: {current_fps:.1f} | "
#                     f"Mode: {output_types[current_type_idx]} | "
#                     f"Frame: {frame_count} | "
#                     f"{status}"
#                 )

#                 cv2.putText(
#                     display_frame,
#                     info,
#                     (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     (0, 255, 0),
#                     2
#                 )

#                 cv2.imshow(
#                     'YOLO Realtime Detection',
#                     display_frame
#                 )

#                 frame_count += 1

#             if max_frames and frame_count >= max_frames:
#                 break

#             key = cv2.waitKey(1) & 0xFF

#             if key == ord('q'):
#                 break

#             elif key == ord('s'):
#                 cv2.imwrite(
#                     f"screenshot_{int(time.time())}.png",
#                     result_frames[-1]
#                 )

#             elif key == ord('r'):
#                 current_type_idx = (
#                     current_type_idx + 1
#                 ) % len(output_types)

#             del frames

#     except KeyboardInterrupt:
#         print("\n⚠️ 用户中断")

#     finally:
#         # ========================================================
#         # 等待后台线程完成
#         # ========================================================

#         if not processing_done.is_set():
#             print("⏳ 等待后台处理完成，最多 20 秒...")

#             processing_done.wait(timeout=20)

#             if not processing_done.is_set():
#                 print("⚠️ 后台线程超时，强制退出")

#         video_stream.stop()

#         if video_writer:
#             video_writer.release()

#         cv2.destroyAllWindows()

#         print(f"✅ 完成！总帧数: {frame_count}")



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
        username="admin",
        password="a1b2c3d4",
        ip_address="192.168.1.168",
        port="554",
        channel="201",
        model=None,
        checkpoint_path=None,
        num_classes=2,
        ocr_recognizer='paddleocr',
        drug_matcher=None,
        classifier=None,
        length=None,
        tile_size=1024,
        overlap=64,
        target_fps=25,
        batch_frames=2,
        crop_region=None,
        output_type='overlay',
        overlay_alpha=0.5,
        display_scale=0.5,
        save_video=False,
        output_path="./realtime_output.mp4",
        save_fps=15,
        max_frames=None,
        device=None,
        trigger_interval=15,
        classifier_thread_safe=True,
):
    import gc

    # ============================================================
    # OCR 识别器处理
    # ============================================================

    if not isinstance(ocr_recognizer, (list, tuple)):
        ocr_recognizer = [ocr_recognizer]

    if len(ocr_recognizer) < 8:
        raise ValueError(f"ocr_recognizer 需要至少 8 个模型，当前只有 {len(ocr_recognizer)} 个")

    # 分配 OCR 模型
    bottle_ocr_models = ocr_recognizer[0:4]  # 前 4 个用于药瓶
    bag_ocr_models = ocr_recognizer[4:6]     # 中间 2 个用于药袋
    shuye_ocr_models = ocr_recognizer[6:8]   # 最后 2 个用于输液袋

    print(f"📋 OCR 模型分配:")
    print(f"  药瓶: {len(bottle_ocr_models)} 个模型")
    print(f"  药袋: {len(bag_ocr_models)} 个模型")
    print(f"  输液袋: {len(shuye_ocr_models)} 个模型")

    # ============================================================
    # RTSP 连接
    # ============================================================

    rtsp_url = (
        f"rtsp://{username}:{password}@{ip_address}:{port}"
        f"/Streaming/Channels/{channel}"
    )

    print(
        f"📡 连接: rtsp://{username}:****@{ip_address}:{port}"
        f"/Streaming/Channels/{channel}"
    )

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
    # YOLO 处理器
    # ============================================================

    processor = YOLOTileProcessor(
        model=model,
        device=str(device),
        tile_size=640,
        overlap=64,
        conf_thres=0.32,
        iou_thres=0.9,
        batch_size=1
    )

    # ============================================================
    # 视频录制器
    # ============================================================

    video_writer = None

    if save_video:
        out_w = (
            video_stream.stream_reader.frame_width * 2
            if output_type == 'side_by_side'
            else video_stream.stream_reader.frame_width
        )

        out_h = video_stream.stream_reader.frame_height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            save_fps,
            (out_w, out_h)
        )

    # ============================================================
    # 显示与统计
    # ============================================================

    output_types = ['overlay', 'color', 'side_by_side', 'gray']

    current_type_idx = (
        output_types.index(output_type)
        if output_type in output_types
        else 0
    )

    fps_counter = deque(maxlen=3)
    frame_count = 0
    last_trigger_time = time.time()

    print("\n" + "=" * 60)
    print("🚀 实时目标检测已启动 YOLOv8 + 切片处理")
    print(f"  批量帧数: {batch_frames}")
    print(f"  自动触发间隔: {trigger_interval} 秒")
    print("  按键: 'q'-退出  's'-截图  'r'-切换模式")
    print("=" * 60 + "\n")

    # 缓存最近批次
    max_batches_kept = 2
    n_batch_frames = deque(maxlen=max_batches_kept)
    n_batch_predictions = deque(maxlen=max_batches_kept)

    # ============================================================
    # 线程安全控制
    # ============================================================

    is_processing = threading.Event()
    processing_done = threading.Event()
    processing_done.set()

    # 每个 OCR 实例对应一个锁
    bottle_ocr_locks = [threading.Lock() for _ in bottle_ocr_models]
    bag_ocr_locks = [threading.Lock() for _ in bag_ocr_models]
    shuye_ocr_locks = [threading.Lock() for _ in shuye_ocr_models]

    # 分类器锁
    classifier_lock = threading.Lock()

    # ============================================================
    # 单个药瓶完整处理函数
    # ============================================================

    def process_bottle_single(args):
        """
        单个药瓶完整处理流程：OCR → 药品匹配 → 分类
        """
        idx, bottle, recognizer, ocr_lock = args

        result = {
            'type': 'bottle',
            'index': idx,
            'ocr_text': None,
            'candidates': None,
            'final_medicine': None,
            'confidence': None,
            'status': ''
        }

        print(f"\n--- [药瓶线程 {idx + 1}] 开始处理 ---")

        # OCR 识别
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

        # 药品候选匹配
        if drug_matcher is None:
            result['status'] = '药品匹配器未提供'
            print(f"  药瓶 {idx + 1}: 药品匹配器未提供")
            return result

        try:
            candidates = drug_matcher.match(
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

        # 分类器确认
        if classifier is not None:
            try:
                if classifier_thread_safe:
                    cls_res = classifier.classify(candidates, bottle)
                else:
                    with classifier_lock:
                        cls_res = classifier.classify(candidates, bottle)

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

                print(f"  药瓶 {idx + 1} 最终分类: {best}, 置信度: {conf:.4f}")

            except Exception as e:
                print(f"  药瓶 {idx + 1}: 分类失败: {e}，使用第一个候选")
                traceback.print_exc()

                result['final_medicine'] = candidates[0]
                result['confidence'] = 0.0
                result['status'] = f'分类失败，使用候选首位: {e}'

        else:
            result['final_medicine'] = candidates[0]
            result['confidence'] = 0.0
            result['status'] = '无分类器，使用候选首位'

            print(f"  药瓶 {idx + 1}: 无分类器，使用候选首位 {candidates[0]}")

        return result

    # ============================================================
    # 单个药袋处理函数
    # ============================================================

    def process_bag_single(args):
        """
        单个药袋处理流程：OCR → 患者姓名匹配
        """
        idx, bag, recognizer, ocr_lock = args

        result = {
            'type': 'bag',
            'index': idx,
            'ocr_text': None,
            'patient_name': None,
            'status': ''
        }

        print(f"\n--- [药袋线程 {idx + 1}] 开始处理 ---")

        if recognizer is None:
            result['status'] = 'OCR识别器未提供'
            print(f"  药袋 {idx + 1}: OCR识别器未提供")
            return result

        try:
            with ocr_lock:
                ocr_text = recognizer.recognize_yaodai(bag)

        except Exception as e:
            result['status'] = f'OCR异常: {e}'
            print(f"  药袋 {idx + 1}: OCR异常: {e}")
            traceback.print_exc()
            return result

        if not ocr_text:
            result['status'] = 'OCR无结果'
            print(f"  药袋 {idx + 1}: OCR无结果")
            return result

        result['ocr_text'] = ocr_text
        print(f"  药袋 {idx + 1} OCR: {ocr_text}")

        # 患者姓名匹配
        if drug_matcher is None:
            result['status'] = '药品匹配器未提供'
            print(f"  药袋 {idx + 1}: 药品匹配器未提供")
            return result

        try:
            patient_name = drug_matcher.match(
                ocr_text,
                match_type='bag',
                threshold=50,
                limit=10
            )

            result['patient_name'] = patient_name
            result['status'] = '完成'

            print(f"  药袋 {idx + 1} 患者姓名: {patient_name}")

        except Exception as e:
            result['status'] = f'患者匹配异常: {e}'
            print(f"  药袋 {idx + 1}: 患者匹配异常: {e}")
            traceback.print_exc()

        return result

    # ============================================================
    # 单个输液袋处理函数
    # ============================================================

    def process_shuye_single(args):
        """
        单个输液袋处理流程：OCR识别
        """
        idx, shuye, recognizer, ocr_lock = args

        result = {
            'type': 'shuye',
            'index': idx,
            'ocr_text': None,
            'status': ''
        }

        print(f"\n--- [输液袋线程 {idx + 1}] 开始处理 ---")

        if recognizer is None:
            result['status'] = 'OCR识别器未提供'
            print(f"  输液袋 {idx + 1}: OCR识别器未提供")
            return result

        try:
            with ocr_lock:
                ocr_text = recognizer.recognize_shuyedai(shuye)

        except Exception as e:
            result['status'] = f'OCR异常: {e}'
            print(f"  输液袋 {idx + 1}: OCR异常: {e}")
            traceback.print_exc()
            return result

        if not ocr_text:
            result['status'] = 'OCR无结果'
            print(f"  输液袋 {idx + 1}: OCR无结果")
            return result

        result['ocr_text'] = ocr_text
        result['status'] = '完成'

        print(f"  输液袋 {idx + 1} OCR: {ocr_text}")

        return result

    # ============================================================
    # 后台处理主函数
    # ============================================================

    def process_batch_in_background(snapshot_frames, snapshot_predictions):
        """
        后台线程处理流程：
        1. 选择最清晰帧
        2. 裁剪药袋、药瓶、输液袋
        3. 三类同步多线程处理
        4. 统一匹配结果
        """
        try:
            if not snapshot_frames:
                return

            t_start = time.time()

            # ====================================================
            # 选择最清晰帧
            # ====================================================

            best_frame, frame_pos = ImageProcessor.select_sharpest_image(
                snapshot_frames
            )

            batch_idx, frame_idx = frame_pos

            print(
                f"最清晰帧: 批次 {batch_idx + 1}/{len(snapshot_frames)} "
                f"中的第 {frame_idx + 1} 张"
            )

            detections = snapshot_predictions[batch_idx][frame_idx]

            # ====================================================
            # 裁剪药袋、药瓶、输液袋
            # ====================================================

            cropped_bags, cropped_bottles, cropped_shuyes = processor.crop_by_class(
                best_frame,
                detections,
                bag_class_id=1,
                bottle_class_id=0,
                shuye_class_id=2,
            )

            print(
                f"检测结果: 药袋 {len(cropped_bags)} 个, "
                f"药瓶 {len(cropped_bottles)} 个, "
                f"输液袋 {len(cropped_shuyes)} 个"
            )

            # ====================================================
            # 准备所有任务
            # ====================================================

            all_tasks = []
            task_handlers = []

            # 药瓶任务
            for i, bottle in enumerate(cropped_bottles):
                task = (
                    i,
                    bottle,
                    bottle_ocr_models[i % len(bottle_ocr_models)],
                    bottle_ocr_locks[i % len(bottle_ocr_locks)]
                )
                all_tasks.append(task)
                task_handlers.append(('bottle', process_bottle_single))

            # 药袋任务
            for i, bag in enumerate(cropped_bags):
                task = (
                    i,
                    bag,
                    bag_ocr_models[i % len(bag_ocr_models)],
                    bag_ocr_locks[i % len(bag_ocr_locks)]
                )
                all_tasks.append(task)
                task_handlers.append(('bag', process_bag_single))

            # 输液袋任务
            for i, shuye in enumerate(cropped_shuyes):
                task = (
                    i,
                    shuye,
                    shuye_ocr_models[i % len(shuye_ocr_models)],
                    shuye_ocr_locks[i % len(shuye_ocr_locks)]
                )
                all_tasks.append(task)
                task_handlers.append(('shuye', process_shuye_single))

            if not all_tasks:
                print("未检测到任何目标")
                return

            print(f"\n{'=' * 60}")
            print(
                f"[多线程同步处理] 共 {len(all_tasks)} 个任务: "             
                f"药瓶 {len(cropped_bottles)}, "
                f"药袋 {len(cropped_bags)}, "
                f"输液袋 {len(cropped_shuyes)}"
            )
            print(f"{'=' * 60}")

            # ====================================================
            # 多线程同步执行所有任务
            # ====================================================

            all_results = []

            # 计算最大线程数：不超过总任务数和所有 OCR 模型数
            max_workers = min(
                len(all_tasks),
                len(bottle_ocr_models) + len(bag_ocr_models) + len(shuye_ocr_models)
            )
            max_workers = max(1, max_workers)

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {}

                for i, (task, (task_type, handler)) in enumerate(zip(all_tasks, task_handlers)):
                    future = pool.submit(handler, task)
                    futures[future] = (i, task_type)

                for future in as_completed(futures):
                    task_idx, task_type = futures[future]

                    try:
                        result = future.result(timeout=90)
                        all_results.append(result)

                    except Exception as e:
                        print(f"  任务 {task_idx + 1} ({task_type}) 线程异常: {e}")
                        traceback.print_exc()

            t_parallel_done = time.time()

            print(
                f"\n[多线程同步处理完成] "
                f"耗时: {t_parallel_done - t_start:.2f}s, "
                f"完成任务: {len(all_results)}/{len(all_tasks)}"
            )

            # ====================================================
            # 分类整理结果
            # ====================================================

            bottle_results = [r for r in all_results if r.get('type') == 'bottle']
            bag_results = [r for r in all_results if r.get('type') == 'bag']
            shuye_results = [r for r in all_results if r.get('type') == 'shuye']

            print(f"\n{'=' * 60}")
            print("【处理结果汇总】")
            print(f"  药瓶: {len(bottle_results)} 个")
            print(f"  药袋: {len(bag_results)} 个")
            print(f"  输液袋: {len(shuye_results)} 个")
            print(f"{'=' * 60}")

            # ====================================================
            # 提取药瓶识别的药品
            # ====================================================

            final_medicines = []

            for r in bottle_results:
                if r.get('final_medicine'):
                    final_medicines.append(r['final_medicine'])
                    print(
                        f"  药瓶 {r['index'] + 1}: {r['final_medicine']} "
                        f"(置信度: {r.get('confidence', 0.0):.4f})"
                    )

            # 去重
            final_medicines = list(set(final_medicines))

            print(f"\n🏥 识别药品（去重后）: {final_medicines}")

            # ====================================================
            # 提取药袋识别的患者姓名
            # ====================================================

            patient_names = []

            for r in bag_results:
                if r.get('patient_name'):
                    patient_names.append(r['patient_name'])
                    print(f"  药袋 {r['index'] + 1}: 患者 {r['patient_name']}")

            # 取第一个有效患者姓名
            patient_name = patient_names[0] if patient_names else None

            if patient_name:
                print(f"\n👤 患者姓名: {patient_name}")
            else:
                print("\n⚠️ 未识别到患者姓名")

            # ====================================================
            # 提取输液袋识别结果
            # ====================================================

            shuye_texts = []

            for r in shuye_results:
                if r.get('ocr_text'):
                    shuye_texts.append(r['ocr_text'])
                    print(f"  输液袋 {r['index'] + 1}: {r['ocr_text']}")

            if shuye_texts:
                print(f"\n💧 输液袋信息: {shuye_texts}")
            else:
                print("\n⚠️ 未识别到输液袋信息")

            # ====================================================
            # 数据库比对
            # ====================================================

            if patient_name and final_medicines and drug_matcher:
                try:
                    validation = drug_matcher.check_patient_batch_medicines(
                        patient_name=patient_name,
                        batch_id=1,
                        expected_medicine_names=final_medicines
                    )

                    print(f"\n{'=' * 60}")
                    print("【数据库匹配结果】")
                    print(f"  患者: {patient_name}")
                    print(f"  识别药品: {final_medicines}")

                    if shuye_texts:
                        print(f"  输液袋信息: {shuye_texts}")

                    if validation['batch_exists']:
                        print(f"  数据库药品: {validation['actual']}")

                        if validation['matched']:
                            print("  ✅ 匹配正确")
                        else:
                            print(f"  ❌ 缺少: {validation['missing']}")
                            print(f"     多余: {validation['extra']}")
                    else:
                        print("  ⚠️ 患者批次不存在")

                    print(f"{'=' * 60}")

                except Exception as e:
                    print(f"数据库比对异常: {e}")
                    traceback.print_exc()

            else:
                print("\n⚠️ 无法进行数据库比对：缺少患者姓名或药品信息")

            print(
                f"\n⏱ 总耗时: {time.time() - t_start:.2f}s "
                f"(多线程处理: {t_parallel_done - t_start:.2f}s)"
            )

            # 释放大对象
            del cropped_bottles
            del cropped_bags
            del cropped_shuyes
            del all_results
            del bottle_results
            del bag_results
            del shuye_results
            gc.collect()

        except Exception as e:
            print(f"❌ 后台处理异常: {e}")
            traceback.print_exc()

        finally:
            is_processing.clear()
            processing_done.set()
            print("🔓 后台处理完成")

    # ============================================================
    # 主循环
    # ============================================================

    try:
        while video_stream.running:
            frames = video_stream.get_batch(batch_frames)

            if not frames:
                continue

            # ====================================================
            # 可选 ROI 裁剪
            # ====================================================

            if crop_region:
                frames = list(
                    crop_image_by_coordinates(
                        np.array(frames),
                        crop_region
                    )
                )

            # ====================================================
            # YOLO 推理
            # ====================================================

            process_start = time.time()

            result_frames, predictions = processor.process_frames_batch(
                frames,
                output_type=output_types[current_type_idx]
            )

            process_time = time.time() - process_start

            # 缓存最近批次
            n_batch_frames.append(frames)
            n_batch_predictions.append(predictions)

            # ====================================================
            # 定时触发后台 OCR + 匹配 + 分类
            # ====================================================

            current_time = time.time()

            if current_time - last_trigger_time >= trigger_interval:
                if not is_processing.is_set():
                    is_processing.set()
                    processing_done.clear()

                    # 快照，避免后台线程和主线程共享 deque
                    snapshot_frames = [
                        list(batch)
                        for batch in n_batch_frames
                    ]

                    snapshot_preds = [
                        list(batch)
                        for batch in n_batch_predictions
                    ]

                    bg_thread = threading.Thread(
                        target=process_batch_in_background,
                        args=(snapshot_frames, snapshot_preds),
                        daemon=True
                    )

                    bg_thread.start()

                    print("\n⏰ 定时触发 → 后台线程已启动")

                    last_trigger_time = current_time

                else:
                    print("\n⏰ 上一轮仍在处理，跳过")
                    last_trigger_time = current_time

            # ====================================================
            # FPS 统计
            # ====================================================

            batch_fps = len(frames) / max(process_time, 0.001)
            fps_counter.append(batch_fps)

            current_fps = np.mean(list(fps_counter))

            # ====================================================
            # 显示与保存
            # ====================================================

            for result_frame in result_frames:
                if video_writer:
                    video_writer.write(result_frame)

                if display_scale != 1.0:
                    h, w = result_frame.shape[:2]

                    display_frame = cv2.resize(
                        result_frame,
                        (
                            int(w * display_scale),
                            int(h * display_scale)
                        )
                    )
                else:
                    display_frame = result_frame.copy()

                status = "处理中" if is_processing.is_set() else "等待"

                info = (
                    f"FPS: {current_fps:.1f} | "
                    f"Mode: {output_types[current_type_idx]} | "
                    f"Frame: {frame_count} | "
                    f"{status}"
                )

                cv2.putText(
                    display_frame,
                    info,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(
                    'YOLO Realtime Detection',
                    display_frame
                )

                frame_count += 1

            if max_frames and frame_count >= max_frames:
                break

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('s'):
                cv2.imwrite(
                    f"screenshot_{int(time.time())}.png",
                    result_frames[-1]
                )

            elif key == ord('r'):
                current_type_idx = (
                    current_type_idx + 1
                ) % len(output_types)

            del frames

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")

    finally:
        # ========================================================
        # 等待后台线程完成
        # ========================================================

        if not processing_done.is_set():
            print("⏳ 等待后台处理完成，最多 20 秒...")

            processing_done.wait(timeout=20)

            if not processing_done.is_set():
                print("⚠️ 后台线程超时，强制退出")

        video_stream.stop()

        if video_writer:
            video_writer.release()

        cv2.destroyAllWindows()

        print(f"✅ 完成！总帧数: {frame_count}")
