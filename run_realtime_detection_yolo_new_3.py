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

#     def process_batch_in_background(snapshot_frames, snapshot_predictions, keep_payload=False):
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
import os
import re
from collections import deque

import cv2
import numpy as np

from src.segmentation.segmenter_yolo import YOLOTileProcessor
from src.processor.img_cropper import crop_image_by_coordinates
from src.stream.video_stream import VideoStream
from src.utils.img_utils import ImageProcessor
from src.identification.realtime_recognition import process_task_group


DISPLAY_MAX_WIDTH = 1024
DISPLAY_MAX_HEIGHT = 600


def _first_text(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            text = _first_text(item)
            if text:
                return text
        return None
    text = str(value).strip()
    return text or None


def _unique_texts(values):
    result = []
    seen = set()
    for value in values:
        text = _first_text(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _resolve_latest_patient_batch_id(drug_matcher, patient_name):
    conn = getattr(drug_matcher, "conn", None)
    if conn is None or not patient_name:
        return None
    if hasattr(conn, "ping"):
        conn.ping(reconnect=True)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT p.patient_id, b.batch_id
            FROM patients p
            JOIN batches b ON b.patient_id = p.patient_id
            WHERE p.name = %s
            ORDER BY b.batch_id DESC
            LIMIT 1
            """,
            (patient_name,),
        )
        row = cursor.fetchone()
    if not row:
        return None
    return row["batch_id"] if isinstance(row, dict) else row[1]


def _resize_for_display(frame, display_scale=1.0, max_width=DISPLAY_MAX_WIDTH, max_height=DISPLAY_MAX_HEIGHT):
    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    scale = float(display_scale) if display_scale else 1.0
    if scale <= 0:
        scale = 1.0

    target_w = max(1, int(round(w * scale)))
    target_h = max(1, int(round(h * scale)))

    fit_scale = min(1.0, float(max_width) / target_w, float(max_height) / target_h)
    target_w = max(1, int(round(target_w * fit_scale)))
    target_h = max(1, int(round(target_h * fit_scale)))

    if target_w == w and target_h == h:
        return frame.copy()
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


class TimingWindow:
    def __init__(self, maxlen=20):
        self.items = deque(maxlen=maxlen)

    def add(self, timing):
        self.items.append(dict(timing))

    def summary(self):
        if not self.items:
            return {}

        keys = sorted({key for item in self.items for key in item.keys()})
        result = {}

        for key in keys:
            values = sorted(float(item[key]) for item in self.items if key in item)
            if not values:
                continue
            p95_idx = min(len(values) - 1, int(round((len(values) - 1) * 0.95)))
            result[key] = {
                "avg": sum(values) / len(values),
                "max": values[-1],
                "p95": values[p95_idx],
            }

        return result


def _fmt_timing_summary(summary):
    parts = []
    for key, stat in summary.items():
        if key.endswith(("_count", "_inputs", "_batches", "_regions", "_pred")) or "text_regions" in key:
            parts.append(
                f"{key}:avg={stat['avg']:.1f},max={stat['max']:.0f},p95={stat['p95']:.0f}"
            )
            continue
        parts.append(
            f"{key}:avg={stat['avg']:.2f}s,max={stat['max']:.2f}s,p95={stat['p95']:.2f}s"
        )
    return " | ".join(parts)


def _fmt_timing_items(timing, max_items=None):
    if not timing:
        return ""
    items = sorted((key, float(value)) for key, value in timing.items())
    if max_items is not None:
        items = items[:max_items]
    parts = []
    for key, value in items:
        if key.endswith(("_count", "_inputs", "_batches", "_regions", "_pred")) or "text_regions" in key:
            parts.append(f"{key}={value:.0f}")
        else:
            parts.append(f"{key}={value:.4f}s")
    return " | ".join(parts)


def _finish_timed_result(result, start_time):
    result.setdefault("timing_sec", {})
    result["timing_sec"]["total"] = time.time() - start_time
    result["timing_sec"] = {
        key: round(float(value), 6)
        for key, value in result["timing_sec"].items()
    }
    return result


def _looks_like_bag_ocr_text(text):
    text = str(text or "").strip()
    if not text:
        return False

    compact = re.sub(r"\s+", "", text)
    bag_terms = (
        "床", "病区", "住院", "患者", "姓名", "年龄", "性别",
        "输液", "静脉", "滴注", "滴速", "用法", "用量", "护士",
        "二维码", "处方", "门诊", "医嘱"
    )
    if any(term in compact for term in bag_terms):
        return True

    if re.search(r"\d+\s*床", compact):
        return True
    if re.search(r"(qd|bid|tid|qid|q\d+h|ivgtt|iv)", compact, flags=re.IGNORECASE):
        return True
    return False


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
        single_image_path=None,
        single_image_output_json="./realtime_single_005_result.json",
        recognition_workers=1,
        simulate_patient_name="",
        quiet_ocr=True,
        headless=None,
        trigger_mode="auto",
        manual_trigger_event=None,
        stop_event=None,
        status_callback=None,
        result_callback=None,
        frame_callback=None,
):
    import gc
    import builtins

    runtime_logs = os.environ.get("YILIAO_RUNTIME_LOGS", "0").lower() in ("1", "true", "yes", "on")

    def print(*args, **kwargs):
        if runtime_logs:
            builtins.print(*args, **kwargs)

    trigger_mode = str(trigger_mode or "auto").lower()
    if trigger_mode not in ("auto", "manual"):
        trigger_mode = "auto"

    def emit_status(**kwargs):
        if status_callback is None:
            return
        try:
            status_callback(dict(kwargs))
        except Exception:
            pass

    def emit_result(payload):
        if result_callback is None:
            return
        try:
            result_callback(payload)
        except Exception:
            pass

    def emit_frame(frame):
        if frame_callback is None:
            return
        try:
            frame_callback(frame)
        except Exception:
            pass

    # ============================================================
    # OCR 识别器处理
    # ============================================================

    if not isinstance(ocr_recognizer, (list, tuple)):
        ocr_recognizer = [ocr_recognizer]

    if len(ocr_recognizer) == 0:
        raise ValueError("ocr_recognizer 不能为空，请至少提供一个 OCR 识别器实例。")

    # One recognizer owns the RKNN contexts. OCR is batched across all current
    # targets, then matching/SIFT post-processing runs in order.
    source_ocr_models = list(ocr_recognizer)
    recognition_ocr_models = [source_ocr_models[0]]
    print("  OCR/SIFT backend: multi-target OCR batch + ordered postprocess")
    print("📋 OCR 模型分配:")
    print("  OCR/SIFT 后台识别: 单线程顺序处理")
    print(f"  实际 OCR 实例: {len(source_ocr_models)} 个")
    print("  使用第 1 个 OCR 实例处理所有药瓶/药袋/输液袋任务")

    if headless is None:
        headless_env = os.environ.get("YILIAO_HEADLESS", "")
        if headless_env:
            headless = headless_env.lower() not in ("0", "false", "no")
        else:
            headless = not bool(os.environ.get("DISPLAY"))
    print(f"  headless显示模式: {headless}")
    verbose_runtime = runtime_logs and os.environ.get("YILIAO_VERBOSE_RUNTIME", "0").lower() not in ("0", "false", "no")

    # ============================================================
    # RTSP 连接
    # ============================================================

    video_stream = None
    processor = None
    if not single_image_path:
        rtsp_url = (
            f"rtsp://{username}:{password}@{ip_address}:{port}"
            f"/Streaming/Channels/{channel}"
        )

        print(
            f"RTSP: rtsp://{username}:****@{ip_address}:{port}"
            f"/Streaming/Channels/{channel}"
        )

        video_stream = VideoStream(rtsp_url, target_fps=target_fps)
        video_stream.start()

        print("Connecting RTSP stream...")

        timeout = 10
        start_time = time.time()

        while not video_stream.is_connected() and time.time() - start_time < timeout:
            time.sleep(0.1)

        if not video_stream.is_connected():
            print("RTSP connection timeout")
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
        conf_thres=0.45,
        iou_thres=0.85,
        batch_size=1,
        contain_thres=0.92,
        class_conf_thres={
            0: 0.45,
            1: 0.38,
            2: 0.32,
        },
        class_iou_thres={
            0: 0.65,
            1: 0.65,
            2: 0.45,
        },
    )

    # ============================================================
    # 视频录制器
    # ============================================================

    video_writer = None

    if save_video and video_stream is not None:
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
    display_window_ready = False
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
    bg_thread = None

    # 分类器锁
    classifier_lock = threading.Lock()
    realtime_timing_window = TimingWindow(maxlen=20)
    detection_timing_window = TimingWindow(maxlen=30)

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
            'status': '',
            'timing_sec': {}
        }
        t_total = time.time()

        print(f"\n--- [药瓶线程 {idx + 1}] 开始处理 ---")

        # OCR 识别
        if recognizer is None:
            result['status'] = 'OCR识别器未提供'
            print(f"  药瓶 {idx + 1}: OCR识别器未提供")
            return result

        try:
            t = time.time()
            with ocr_lock:
                ocr_text = recognizer.recognize(bottle)
            result['timing_sec']['ocr'] = time.time() - t
            result['timing_sec'].update({
                f"ocr_{key}": value
                for key, value in getattr(recognizer, 'last_timing', {}).items()
            })

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

        if _looks_like_bag_ocr_text(ocr_text):
            result['type'] = 'bag_like_bottle'
            result['status'] = 'bag-like OCR, skipped bottle recognition'
            result['confidence'] = 0.0
            print(f"  bottle {idx + 1}: bag-like OCR, skip drug matching")
            return _finish_timed_result(result, t_total)

        # 药品候选匹配
        if drug_matcher is None:
            result['status'] = '药品匹配器未提供'
            print(f"  药瓶 {idx + 1}: 药品匹配器未提供")
            return result

        try:
            t = time.time()
            candidates = drug_matcher.match(
                ocr_text,
                match_type='bottle',
                threshold=50,
                limit=10
            )
            result['timing_sec']['match'] = time.time() - t

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
                t = time.time()
                if classifier_thread_safe:
                    cls_res = classifier.classify(candidates, bottle)
                else:
                    with classifier_lock:
                        cls_res = classifier.classify(candidates, bottle)
                result['timing_sec']['classify'] = time.time() - t

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

        return _finish_timed_result(result, t_total)

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
            'status': '',
            'timing_sec': {}
        }
        t_total = time.time()

        print(f"\n--- [药袋线程 {idx + 1}] 开始处理 ---")

        if recognizer is None:
            result['status'] = 'OCR识别器未提供'
            print(f"  药袋 {idx + 1}: OCR识别器未提供")
            return result

        try:
            t = time.time()
            with ocr_lock:
                ocr_text = recognizer.recognize_yaodai(bag)
            result['timing_sec']['ocr'] = time.time() - t
            result['timing_sec'].update({
                f"ocr_{key}": value
                for key, value in getattr(recognizer, 'last_timing', {}).items()
            })

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
            t = time.time()
            patient_name = drug_matcher.match(
                ocr_text,
                match_type='bag',
                threshold=50,
                limit=10
            )
            result['timing_sec']['match'] = time.time() - t

            result['patient_name'] = patient_name
            result['status'] = '完成'

            print(f"  药袋 {idx + 1} 患者姓名: {patient_name}")

        except Exception as e:
            result['status'] = f'患者匹配异常: {e}'
            print(f"  药袋 {idx + 1}: 患者匹配异常: {e}")
            traceback.print_exc()

        return _finish_timed_result(result, t_total)

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
            'liquid': None,
            'concentration': None,
            'volume': None,
            'raw_text': '',
            'status': '',
            'timing_sec': {}
        }
        t_total = time.time()

        print(f"\n--- [输液袋线程 {idx + 1}] 开始处理 ---")

        if recognizer is None:
            result['status'] = 'OCR识别器未提供'
            print(f"  输液袋 {idx + 1}: OCR识别器未提供")
            return result

        try:
            t = time.time()
            with ocr_lock:
                ocr_text = recognizer.recognize_shuyedai(shuye)
            result['timing_sec']['ocr'] = time.time() - t
            result['timing_sec'].update({
                f"ocr_{key}": value
                for key, value in getattr(recognizer, 'last_timing', {}).items()
            })

        except Exception as e:
            result['status'] = f'OCR异常: {e}'
            print(f"  输液袋 {idx + 1}: OCR异常: {e}")
            traceback.print_exc()
            return result

        if not ocr_text:
            result['status'] = 'OCR无结果'
            print(f"  输液袋 {idx + 1}: OCR无结果")
            return result

        if isinstance(ocr_text, dict):
            result['ocr_text'] = ocr_text
            result['liquid'] = ocr_text.get('liquid')
            result['concentration'] = ocr_text.get('concentration')
            result['volume'] = ocr_text.get('volume')
            result['raw_text'] = ocr_text.get('raw_text', '')
            result['status'] = ocr_text.get('status', '完成')
        else:
            result['ocr_text'] = ocr_text
            result['raw_text'] = ocr_text or ''
            result['status'] = '完成'

        print(f"  输液袋 {idx + 1} OCR: {ocr_text}")

        return _finish_timed_result(result, t_total)

    # ============================================================
    # 后台处理主函数
    # ============================================================

    def process_batch_in_background(snapshot_frames, snapshot_predictions, keep_payload=False):
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
            t_select_start = time.time()

            # ====================================================
            # 选择最清晰帧
            # ====================================================

            if len(snapshot_frames) == 1 and len(snapshot_frames[0]) == 1:
                best_frame = snapshot_frames[0][0]
                frame_pos = (0, 0)
            else:
                best_frame, frame_pos = ImageProcessor.select_sharpest_image(
                    snapshot_frames
                )
            t_select_done = time.time()

            batch_idx, frame_idx = frame_pos

            print(
                f"最清晰帧: 批次 {batch_idx + 1}/{len(snapshot_frames)} "
                f"中的第 {frame_idx + 1} 张"
            )

            detections = snapshot_predictions[batch_idx][frame_idx]

            # ====================================================
            # 裁剪药袋、药瓶、输液袋
            # ====================================================

            t_crop_start = time.time()
            cropped_bags, cropped_bottles, cropped_shuyes = processor.crop_by_class(
                best_frame,
                detections,
                bag_class_id=1,
                bottle_class_id=0,
                shuye_class_id=2,
            )
            t_crop_done = time.time()

            print(
                f"检测结果: 药袋 {len(cropped_bags)} 个, "
                f"药瓶 {len(cropped_bottles)} 个, "
                f"输液袋 {len(cropped_shuyes)} 个"
            )

            # ====================================================
            # 准备所有任务
            # ====================================================

            all_tasks = []

            # 药瓶任务
            for i, bottle in enumerate(cropped_bottles):
                all_tasks.append((
                    'bottle',
                    i,
                    bottle,
                ))

            # 药袋任务
            for i, bag in enumerate(cropped_bags):
                all_tasks.append((
                    'bag',
                    i,
                    bag,
                ))

            # 输液袋任务
            for i, shuye in enumerate(cropped_shuyes):
                all_tasks.append((
                    'shuye',
                    i,
                    shuye,
                ))

            if not all_tasks:
                print("未检测到任何目标")
                return

            print(f"\n{'=' * 60}")
            print(
                f"[Multi-target OCR batch] tasks={len(all_tasks)}, "
                f"bottles={len(cropped_bottles)}, bags={len(cropped_bags)}, "
                f"infusions={len(cropped_shuyes)}"
            )
            print(
                f"[单线程顺序处理] 共 {len(all_tasks)} 个任务: "
                f"药瓶 {len(cropped_bottles)}, "
                f"药袋 {len(cropped_bags)}, "
                f"输液袋 {len(cropped_shuyes)}"
            )
            print(f"{'=' * 60}")

            # ====================================================
            # 单线程顺序执行所有 OCR/SIFT 任务
            # ====================================================

            all_results = process_task_group(
                all_tasks,
                recognition_ocr_models[0],
                drug_matcher,
                classifier,
                classifier is not None,
                classifier_lock,
                classifier_thread_safe,
                quiet_ocr,
                simulate_patient_name,
            )

            t_parallel_done = time.time()

            batch_timing = {}

            if verbose_runtime:
                batch_timing = {
                    "background_total": t_parallel_done - t_start,
                    "select_sharpest": t_select_done - t_select_start,
                    "crop": t_crop_done - t_crop_start,
                    "prepare_and_recognition": t_parallel_done - t_crop_done,
                    "recognition_tasks": t_parallel_done - t_start,
                    "runtime_excluding_model_init": t_parallel_done - t_start,
                    "target_count": len(all_tasks),
                    "bottle_count": len(cropped_bottles),
                    "bag_count": len(cropped_bags),
                    "shuye_count": len(cropped_shuyes),
                }
                for item in all_results:
                    prefix = item.get('type', 'unknown')
                    for key, value in (item.get('timing_sec', {}) or {}).items():
                        timing_key = f"{prefix}_{key}"
                        if key.startswith((
                            "ocr_mixed_",
                            "ocr_bottle_angle_",
                            "ocr_bag_angle_",
                            "ocr_bag_name_roi_extract",
                            "ocr_shuye_angle_",
                            "ocr_shuye_roi_angle_",
                            "ocr_shared_",
                            "ocr_shuye_preprocess",
                        )):
                            batch_timing[timing_key] = max(
                                batch_timing.get(timing_key, 0.0),
                                float(value),
                            )
                        else:
                            batch_timing[timing_key] = batch_timing.get(timing_key, 0.0) + float(value)

                realtime_timing_window.add(batch_timing)
                summary_text = _fmt_timing_summary(realtime_timing_window.summary())
                if summary_text:
                    print(f"\n[Timing Summary last {len(realtime_timing_window.items)}] {summary_text}")

                print("\n[Step Timing Detail]")
                print(f"  batch: {_fmt_timing_items(batch_timing)}")
            for item in (sorted(all_results, key=lambda value: (value.get("type", ""), value.get("index", 0))) if verbose_runtime else []):
                label = f"{item.get('type', 'unknown')}#{int(item.get('index', 0)) + 1}"
                timing_text = _fmt_timing_items(item.get("timing_sec", {}))
                status = item.get("status", "")
                final_name = item.get("final_medicine") or item.get("patient_name") or item.get("liquid") or ""
                decision = item.get("decision_reason", "")
                print(f"  {label}: status={status}, result={final_name}, timing: {timing_text}")
                if decision:
                    print(f"    decision: {decision}")

            print(
                f"\n[单线程顺序处理完成] "
                f"耗时: {t_parallel_done - t_start:.2f}s, "
                f"完成任务: {len(all_results)}/{len(all_tasks)}"
            )

            # ====================================================
            # 分类整理结果
            # ====================================================

            print(
                f"[Multi-target OCR batch done] elapsed={t_parallel_done - t_start:.2f}s, "
                f"finished={len(all_results)}/{len(all_tasks)}"
            )

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
            bottle_outputs = []

            for r in bottle_results:
                index = int(r.get('index', 0)) + 1
                candidates = r.get('candidates') or []
                bottle_outputs.append({
                    "index": index,
                    "ocr_text": r.get("ocr_text", ""),
                    "candidates": candidates[:5],
                    "final_medicine": r.get("final_medicine"),
                    "confidence": r.get("confidence"),
                    "classification_method": r.get("classification_method"),
                    "top_3": r.get("top_3") or [],
                    "decision_reason": r.get("decision_reason", ""),
                    "status": r.get("status", ""),
                })
                print(f"  药瓶 {index} OCR: {r.get('ocr_text', '')}")
                print(f"    匹配候选: {candidates[:5]}")
                if r.get('ocr_match_weak'):
                    source = r.get('classification_candidate_source', '')
                    count = r.get('classification_candidate_count', 0)
                    print(f"    OCR弱匹配: 是, SIFT候选来源={source}, 数量={count}")
                if r.get('top_3'):
                    print(f"    分类Top3: {r.get('top_3')}")
                if r.get('final_medicine'):
                    final_medicines.append(r['final_medicine'])
                    method = r.get('classification_method', 'unknown')
                    best_template = r.get('sift_best_template') or {}
                    sift_text = ""
                    if method == "sift_template":
                        sift_text = (
                            f", SIFT good={best_template.get('good_matches', 0)}, "
                            f"score={best_template.get('match_score', 0.0):.4f}"
                        )
                    zero_reason = ""
                    if float(r.get('confidence', 0.0) or 0.0) <= 0.0:
                        zero_reason = f", zero_reason={r.get('classification_zero_reason', r.get('decision_reason', 'unknown'))}"
                    print(
                        f"    最终分类: {r['final_medicine']} "
                        f"(方式: {method}, 置信度: {r.get('confidence', 0.0):.4f}"
                        f"{sift_text}{zero_reason})"
                    )
                    if r.get('decision_reason'):
                        print(f"    分类决策: {r.get('decision_reason')}")
                else:
                    print(
                        f"    最终分类: 无 "
                        f"(status={r.get('status', '')}, reason={r.get('classification_zero_reason', '')})"
                    )

            # 去重
            final_medicines = list(set(final_medicines))

            print(f"\n🏥 识别药品（去重后）: {final_medicines}")

            # ====================================================
            # 提取药袋识别的患者姓名
            # ====================================================

            patient_names = []
            bag_outputs = []

            for r in bag_results:
                matched_patient = _first_text(r.get("patient_name"))
                bag_outputs.append({
                    "index": int(r.get("index", 0)) + 1,
                    "ocr_text": r.get("ocr_text", ""),
                    "patient_name": matched_patient,
                    "patient_candidates": _unique_texts([r.get("patient_name")]),
                    "status": r.get("status", ""),
                })
                print(f"  药袋 {r.get('index', 0) + 1} OCR: {r.get('ocr_text', '')}")
                if matched_patient:
                    patient_names.append(matched_patient)
                    print(f"  药袋 {r['index'] + 1}: 患者匹配 {r['patient_name']}")

            # 取第一个有效患者姓名
            patient_names = _unique_texts(patient_names)
            patient_name = patient_names[0] if patient_names else None

            if patient_name:
                print(f"\n👤 患者姓名: {patient_name}")
            else:
                print("\n⚠️ 未识别到患者姓名")

            # ====================================================
            # 提取输液袋识别结果
            # ====================================================

            shuye_texts = []
            structured_infusions = []

            for r in shuye_results:
                structured_item = {
                    "index": r.get("index"),
                    "liquid": r.get("liquid"),
                    "concentration": r.get("concentration"),
                    "volume": r.get("volume"),
                    "raw_text": r.get("raw_text", ""),
                    "status": r.get("status", ""),
                }
                structured_infusions.append(structured_item)
                print(f"  输液袋 {r.get('index', 0) + 1} OCR: {r.get('ocr_text', '')}")
                print(
                    f"  输液袋 {r.get('index', 0) + 1}: "
                    f"液体={structured_item['liquid']}, "
                    f"浓度={structured_item['concentration']}, "
                    f"容量={structured_item['volume']}, "
                    f"状态={structured_item['status']}"
                )
                if r.get('ocr_text'):
                    shuye_texts.append(r['ocr_text'])

            if shuye_texts:
                print(f"\n💧 输液袋信息: {shuye_texts}")
                print(f"💧 输液袋结构化信息: {structured_infusions}")
            else:
                print("\n⚠️ 未识别到输液袋信息")

            # ====================================================
            # 数据库比对
            # ====================================================

            validation_result = None
            validation_status = "skipped"
            validation_message = "missing patient name, medicine result, or database matcher"
            validation_batch_id = None

            if patient_name and final_medicines and drug_matcher and hasattr(drug_matcher, "check_patient_batch_medicines"):
                try:
                    validation_batch_id = _resolve_latest_patient_batch_id(drug_matcher, patient_name)
                    if validation_batch_id is None:
                        validation = {
                            "matched": False,
                            "actual": [],
                            "missing": [],
                            "extra": [],
                            "patient_id": None,
                            "batch_exists": False,
                        }
                    else:
                        validation = drug_matcher.check_patient_batch_medicines(
                            patient_name=patient_name,
                            batch_id=validation_batch_id,
                            expected_medicine_names=final_medicines
                        )
                        validation["batch_id"] = validation_batch_id
                    validation_result = validation
                    if validation.get("batch_exists"):
                        validation_status = "matched" if validation.get("matched") else "mismatch"
                        validation_message = (
                            f"matched batch {validation_batch_id}"
                            if validation.get("matched")
                            else f"medicine mismatch in batch {validation_batch_id}"
                        )
                    else:
                        validation_status = "batch_not_found"
                        validation_message = "patient batch not found"

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
                    validation_status = "error"
                    validation_message = str(e)
                    print(f"数据库比对异常: {e}")
                    traceback.print_exc()

            else:
                print("\n⚠️ 跳过数据库比对：缺少患者姓名/药品信息，或当前匹配器不支持数据库校验")

            print(
                f"\n⏱ 总耗时: {time.time() - t_start:.2f}s "
                f"(多线程处理: {t_parallel_done - t_start:.2f}s)"
            )

            background_payload = {
                "timing_sec": {
                    key: round(float(value), 6)
                    for key, value in batch_timing.items()
                },
                "bottles": bottle_outputs,
                "bags": bag_outputs,
                "infusions": structured_infusions,
                "recognized_medicines": final_medicines,
                "patient_name": patient_name,
                "patient_names": patient_names,
                "database_match": {
                    "status": validation_status,
                    "message": validation_message,
                    "batch_id": validation_batch_id,
                    "validation": validation_result,
                },
                "structured_infusions": structured_infusions,
                "counts": {
                    "bottle": len(cropped_bottles),
                    "bag": len(cropped_bags),
                    "shuye": len(cropped_shuyes),
                    "total": len(all_tasks),
                },
            }

            if keep_payload:
                background_payload["results"] = all_results
            emit_result(background_payload)

            # 释放大对象
            del all_tasks
            del best_frame
            del detections
            del snapshot_frames
            del snapshot_predictions
            del cropped_bottles
            del cropped_bags
            del cropped_shuyes
            del all_results
            del bottle_results
            del bag_results
            del shuye_results
            gc.collect()
            return background_payload

        except Exception as e:
            print(f"❌ 后台处理异常: {e}")
            traceback.print_exc()

        finally:
            is_processing.clear()
            processing_done.set()
            emit_status(state="running", processing=False, frame_count=frame_count)
            print("🔓 后台处理完成")

    # ============================================================
    # 主循环
    # ============================================================

    if single_image_path:
        import json

        single_total_start = time.time()
        frame = cv2.imread(single_image_path)
        if frame is None:
            raise FileNotFoundError(single_image_path)

        if crop_region:
            frame = crop_image_by_coordinates(np.array([frame]), crop_region)[0]

        detect_start = time.time()
        result_frames, predictions = processor.process_frames_batch(
            [frame],
            output_type=output_types[current_type_idx]
        )
        detect_elapsed = time.time() - detect_start
        detection_timing = {
            "detect_batch": detect_elapsed,
            "detect_runtime_excluding_model_init": detect_elapsed,
            "frames": 1,
            "fps": 1 / max(detect_elapsed, 0.001),
        }
        detection_timing_window.add(detection_timing)

        payload = process_batch_in_background([[frame]], [predictions], keep_payload=True)
        result = {
            "image": single_image_path,
            "detection_timing_sec": {
                key: round(float(value), 6)
                for key, value in detection_timing.items()
            },
            "background": payload,
            "runtime_sec_excluding_model_init": round(time.time() - single_total_start, 6),
        }
        if single_image_output_json:
            with open(single_image_output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[Realtime Single Image] result saved: {single_image_output_json}")
        return result

    try:
        while video_stream.running and not (stop_event is not None and stop_event.is_set()):
            frames = video_stream.get_batch(batch_frames)

            if not frames:
                continue


            # ====================================================
            # 可选 ROI 裁剪
            # ====================================================

            if crop_region:
                frames = [
                    crop_image_by_coordinates(frame, crop_region)
                    for frame in frames
                ]

            # ====================================================
            # YOLO 推理
            # ====================================================

            process_start = time.time()

            result_frames, predictions = processor.process_frames_batch(
                frames,
                output_type=output_types[current_type_idx]
            )

            process_time = time.time() - process_start
            if verbose_runtime:
                detection_timing_window.add({
                    "detect_batch": process_time,
                    "detect_runtime_excluding_model_init": process_time,
                    "frames": len(frames),
                    "fps": len(frames) / max(process_time, 0.001),
                })

            # 缓存最近批次
            n_batch_frames.append(frames)
            n_batch_predictions.append(predictions)

            # ====================================================
            # 定时触发后台 OCR + 匹配 + 分类
            # ====================================================

            current_time = time.time()
            should_trigger = False
            trigger_label = "timed"
            if trigger_mode == "manual":
                if manual_trigger_event is not None and manual_trigger_event.is_set():
                    manual_trigger_event.clear()
                    should_trigger = True
                    trigger_label = "manual"
            elif current_time - last_trigger_time >= trigger_interval:
                should_trigger = True

            if should_trigger:
                if not is_processing.is_set():
                    is_processing.set()
                    processing_done.clear()
                    emit_status(state="running", processing=True, frame_count=frame_count, trigger=trigger_label)

                    # 快照当前批次，避免多批 4K 帧在主线程和后台线程重复滞留
                    snapshot_frames = [frames]
                    snapshot_preds = [predictions]

                    bg_thread = threading.Thread(
                        target=process_batch_in_background,
                        args=(snapshot_frames, snapshot_preds),
                        daemon=False
                    )

                    bg_thread.start()
                    n_batch_frames.clear()
                    n_batch_predictions.clear()
                    gc.collect()

                    print("\n⏰ 定时触发 → 后台线程已启动")

                    last_trigger_time = current_time

                else:
                    print("\n⏰ 上一轮仍在处理，跳过")
                    n_batch_frames.clear()
                    n_batch_predictions.clear()
                    last_trigger_time = current_time

            # ====================================================
            # FPS 统计
            # ====================================================

            batch_fps = len(frames) / max(process_time, 0.001)
            fps_counter.append(batch_fps)

            current_fps = np.mean(list(fps_counter))
            if verbose_runtime and frame_count > 0 and frame_count % 30 == 0:
                detect_summary = _fmt_timing_summary(detection_timing_window.summary())
                if detect_summary:
                    print(f"[Detection Timing last {len(detection_timing_window.items)}] {detect_summary}")

            # ====================================================
            # 显示与保存
            # ====================================================

            if not headless and not display_window_ready:
                cv2.namedWindow('YOLO Realtime Detection', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('YOLO Realtime Detection', DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT)
                display_window_ready = True

            for result_frame in result_frames:
                if video_writer:
                    video_writer.write(result_frame)

                display_frame = _resize_for_display(result_frame, display_scale)

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

                emit_frame(display_frame)

                if not headless:
                    cv2.imshow(
                        'YOLO Realtime Detection',
                        display_frame
                    )

                frame_count += 1

            if max_frames and frame_count >= max_frames:
                break

            key = 255 if headless else (cv2.waitKey(1) & 0xFF)

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

            emit_status(state="running", processing=is_processing.is_set(), frame_count=frame_count, fps=float(current_fps))
            if frame_count % 60 == 0:
                gc.collect()
            del frames
            del result_frames
            del predictions

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")

    finally:
        # ========================================================
        # 等待后台线程完成
        # ========================================================

        if not processing_done.is_set():
            print("⏳ 等待后台处理完成，最多 120 秒...")

            processing_done.wait(timeout=120)

            if not processing_done.is_set():
                print("⚠️ 后台线程仍未结束，将继续等待识别线程池安全退出")

        if bg_thread is not None and bg_thread.is_alive():
            print("⏳ join 后台处理线程...")
            bg_thread.join(timeout=120)
            if bg_thread.is_alive():
                print("⚠️ 后台处理线程仍未结束")

        if video_stream is not None:
            video_stream.stop()

        if video_writer:
            video_writer.release()

        if processor is not None:
            release = getattr(processor, "release", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    pass

        if not headless:
            cv2.destroyAllWindows()

        gc.collect()
        print(f"✅ 完成！总帧数: {frame_count}")
