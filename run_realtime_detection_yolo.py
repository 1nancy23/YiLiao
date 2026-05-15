import time
from collections import deque
import cv2
import numpy as np
# import torch
import threading
from src.segmentation.segmenter_yolo import YOLOTileProcessor
from src.processor.img_cropper import crop_image_by_coordinates
from src.stream.video_stream import VideoStream
from src.utils.img_utils import ImageProcessor


def run_realtime_detection(
        username="admin", password="a1b2c3d4", ip_address="192.168.1.168", port="554", channel="201",
        model=None, checkpoint_path=None, num_classes=2,
        ocr_recognizer='paddleocr', drug_matcher=None, classifier=None, length=None,
        tile_size=1024, overlap=64, target_fps=15, batch_frames=2,
        crop_region=None,
        output_type='overlay', overlay_alpha=0.5, display_scale=0.5,
        save_video=False, output_path="./realtime_output.mp4", save_fps=15,
        max_frames=None, device=None, trigger_interval=15,
):
    import gc
    # if device is None:
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"🚀 使用设备: {device}")
    ocr_recognizer_1, ocr_recognizer_2, ocr_recognizer_3, ocr_recognizer_4 = ocr_recognizer
    # 构建RTSP地址
    rtsp_url = f"rtsp://{username}:{password}@{ip_address}:{port}/Streaming/Channels/{channel}"
    print(f"📡 连接: rtsp://{username}:****@{ip_address}:{port}/Streaming/Channels/{channel}")

    # 启动视频流
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

    # 初始化YOLO切片处理器
    processor = YOLOTileProcessor(
        model=model,
        device=str(device),
        tile_size=640,          # 切片尺寸
        overlap=64,             # 重叠像素
        conf_thres=0.25,
        iou_thres=0.95,
        batch_size=1            # 切片推理批量
    )

    # 视频录制器
    video_writer = None
    if save_video:
        out_w = video_stream.stream_reader.frame_width * 2 if output_type == 'side_by_side' else video_stream.stream_reader.frame_width
        out_h = video_stream.stream_reader.frame_height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, save_fps, (out_w, out_h))
        print(f"📹 录制到: {output_path}")

    # 显示模式
    output_types = ['overlay', 'color', 'side_by_side', 'gray']
    current_type_idx = output_types.index(output_type) if output_type in output_types else 0

    # 性能统计
    fps_counter = deque(maxlen=3)
    frame_count = 0
    last_trigger_time = time.time()

    print("\n" + "=" * 60)
    print("🚀 实时目标检测已启动 (YOLOv8 + 切片处理)")
    print(f" 批量帧数: {batch_frames}")
    print(f" 自动触发间隔: {trigger_interval}秒")
    print(" 按键: 'q'-退出 's'-截图 'r'-切换模式")
    print("=" * 60 + "\n")

    # 缓存最近批次（用于触发时回溯）
    max_batches_kept = 2
    n_batch_frames = deque(maxlen=max_batches_kept)        # 每个元素是一批帧的列表
    n_batch_predictions = deque(maxlen=max_batches_kept)   # 每个元素是一批检测结果的列表

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 四个识别器列表
    recognizers = ocr_recognizer
    ocr_locks = [threading.Lock() for _ in range(len(ocr_recognizer))]
    def process_single_bottle(args):
        idx, bottle, recognizer, lock = args
        result = {'index': idx, 'medicine': None, 'status': ''}

        with lock:  # ★ 同一个识别器同时只有一个线程使用
            print(f"\n--- 药瓶 {idx+1} ---")
            if recognizer is None:
                result['status'] = 'OCR识别器未提供'
                return result

            try:
                ocr_text = recognizer.recognize(bottle)
            except Exception as e:
                print(f"  OCR异常: {e}")
                result['status'] = f'OCR异常: {e}'
                return result

            if not ocr_text:
                result['status'] = 'OCR无结果'
                return result
            print(f"  OCR: {ocr_text}")

            if drug_matcher is None:
                result['status'] = '匹配器未提供'
                return result

            candidates = drug_matcher.match(ocr_text, match_type='bottle', threshold=50, limit=10)
            if not candidates:
                result['status'] = '无候选'
                return result
            print(f"  候选: {candidates[:3]}...")

            if classifier is not None:
                try:
                    print("特征分类开始")
                    res = classifier.classify(candidates, bottle)
                    result['medicine'] = res['predicted_category']
                except Exception as e:
                    print(f"  分类失败: {e}")
                    result['medicine'] = candidates[0]
            else:
                result['medicine'] = candidates[0]

            result['status'] = '成功'
        return result

        
    
    def process_batch_if_triggered():
        """从缓存中找出最清晰帧并执行OCR+匹配"""
        if not n_batch_frames:
            return

        t_start = time.time()

        # 找出最清晰的帧 (返回 best_frame, (batch_idx, frame_idx))
        best_frame, frame_pos = ImageProcessor.select_sharpest_image(n_batch_frames)
        batch_idx, frame_idx = frame_pos
        print(f"最清晰帧: 批次 {batch_idx+1}/{len(n_batch_frames)} 中的第 {frame_idx+1} 张")

        # 关键修复：正确索引出对应帧的检测结果
        batch_predictions = n_batch_predictions[batch_idx]   # 该批次所有帧的预测列表
        detections = batch_predictions[frame_idx]            # 具体那一帧的检测框

        # 裁剪药袋和药瓶（类别ID根据你的模型设定，这里假设 1=bag, 0=bottle）
        cropped_bags, cropped_bottles = processor.crop_by_class(best_frame, detections, bag_class_id=1, bottle_class_id=0)
        print(f"药袋: {len(cropped_bags)} 个, 药瓶: {len(cropped_bottles)} 个")

        # 可选：显示裁剪结果（调试用）
        # img_shower(cropped_bags, cropped_bottles, save=False, show=True)

        if cropped_bags and cropped_bottles:
            # ---------- 处理药瓶 ----------
            # final_medicines =  pipeline_process_bottles(
            #             cropped_bottles, ocr_recognizer, drug_matcher, classifier
            #         )
            # final_medicines=[]
            # for i, bottle in enumerate(cropped_bottles):
            #     print(f"\n--- 药瓶 {i+1} ---")
            #     if ocr_recognizer_1 is None:
            #         print("  OCR 识别器未提供，跳过")
            #         continue
            #     ocr_text = ocr_recognizer_1.recognize(bottle)
            #     if not ocr_text:
            #         print("  OCR 无结果，跳过")
            #         continue
            #     print(f"  OCR: {ocr_text}")

            #     if drug_matcher is None:
            #         print("  匹配器未提供，跳过")
            #         continue
            #     candidates = drug_matcher.match(ocr_text, match_type='bottle', threshold=50, limit=10)
            #     if not candidates:
            #         print("  无候选，跳过")
            #         continue
            #     print(f"  候选: {candidates[:3]}...")

            #     if classifier is not None:
            #         try:
            #             res = classifier.classify(candidates, bottle)
            #             best = res['predicted_category']
            #             conf = res['confidence']
            #             print(f"  特征匹配: {best} ({conf:.4f})")
            #             final_medicines.append(best)
            #         except Exception as e:
            #             print(f"  特征匹配失败: {e}，用第一个候选")
            #             final_medicines.append(candidates[0])
            #     else:
            #         final_medicines.append(candidates[0])

            # 构建任务列表：按 index % 4 轮询分配识别器
            # tasks = [
            #     (i, bottle, ocr_recognizer[i % len(ocr_recognizer)])
            #     for i, bottle in enumerate(cropped_bottles)
            # ]
            tasks = [
                (i, bottle, 
                ocr_recognizer[i % len(ocr_recognizer)],
                ocr_locks[i % len(ocr_recognizer)])       # ★ 传入对应的锁
                for i, bottle in enumerate(cropped_bottles)
                ]
            # 多线程并行执行
            final_medicines = [None] * len(cropped_bottles)

            with ThreadPoolExecutor(max_workers=len(ocr_recognizer)) as executor:
                futures = {executor.submit(process_single_bottle, task): task[0] for task in tasks}

                for future in as_completed(futures):
                    result = future.result()
                    idx = result['index']
                    if result['medicine'] is not None:
                        final_medicines[idx] = result['medicine']

            # 移除 None（无结果的药瓶）
            final_medicines = [m for m in final_medicines if m is not None]
            """激进清理"""
            gc.collect()
            gc.collect()  # 运行两次

            # # 清理 Python 对象缓存
            # import ctypes
            # ctypes.CDLL("libc.so.6").malloc_trim(0)
            
            final_medicines = list(set(final_medicines))
            print(f"\n识别药品: {final_medicines}")

            # ---------- 处理药袋 ----------
            patient_name = None
            if cropped_bags and ocr_recognizer_1 and drug_matcher:
                bag_ocr = ocr_recognizer_1.recognize(cropped_bags[0])
                if bag_ocr:
                    patient_name = drug_matcher.match(bag_ocr, match_type='bag', threshold=50, limit=10)
                    print(f"患者姓名: {patient_name}")

            # ---------- 数据库比对 ----------
            if patient_name and final_medicines and drug_matcher:
                validation = drug_matcher.check_patient_batch_medicines(
                    patient_name=patient_name, batch_id=1, expected_medicine_names=final_medicines
                )
                print("\n【匹配结果】")
                print(f"患者: {patient_name}")
                print(f"识别药品: {final_medicines}")
                if validation['batch_exists']:
                    print(f"数据库药品: {validation['actual']}")
                    if validation['matched']:
                        print("✅ 匹配正确")
                    else:
                        print(f"❌ 缺少: {validation['missing']}, 多余: {validation['extra']}")
                else:
                    print("⚠️ 患者批次不存在")
            else:
                print("无法比对：缺少患者姓名或药品")
        else:
            print("未同时检测到药袋和药瓶")
            
        print(f"识别耗时: {time.time() - t_start:.2f}s")
        del cropped_bottles
        del cropped_bags
        del final_medicines
        del patient_name
        
    # 主循环
    try:
        while video_stream.running:
            # 获取一批帧（例如2帧）
            frames = video_stream.get_batch(batch_frames)
            if not frames:
                continue
            print("frames",frames[0].shape)
            # 可选：对每帧进行ROI裁剪
            if crop_region:
                frames = list(crop_image_by_coordinates(np.array(frames), crop_region))

            # YOLO推理（切片+检测）
            process_start = time.time()
            result_frames, predictions = processor.process_frames_batch(
                frames, output_type=output_types[current_type_idx]
            )
            process_time = time.time() - process_start

            # 存入缓存（注意：frames是列表，predictions也是列表，长度相等）
            n_batch_frames.append(frames)
            n_batch_predictions.append(predictions)

            # 检查是否到达触发时间
            current_time = time.time()
            if current_time - last_trigger_time >= trigger_interval:
                print(f"\n⏰ 定时触发 (间隔 {trigger_interval}s)")
                process_batch_if_triggered()
                last_trigger_time = current_time

            # FPS计算
            batch_fps = len(frames) / max(process_time, 0.001)
            fps_counter.append(batch_fps)
            current_fps = np.mean(list(fps_counter))

            # 显示结果帧
            for result_frame in result_frames:
                if video_writer:
                    video_writer.write(result_frame)

                # 缩放显示
                if display_scale != 1.0:
                    h, w = result_frame.shape[:2]
                    display_frame = cv2.resize(result_frame, (int(w * display_scale), int(h * display_scale)))
                else:
                    display_frame = result_frame.copy()

                # 叠加信息
                info = f"FPS: {current_fps:.1f} | Mode: {output_types[current_type_idx]} | Frame: {frame_count}"
                cv2.putText(display_frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if save_video:
                    cv2.putText(display_frame, "REC", (display_frame.shape[1] - 60, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('YOLO Realtime Detection', display_frame)
                frame_count += 1

            if max_frames and frame_count >= max_frames:
                break

            # 键盘控制（可选，如果有显示窗口）
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"screenshot_{int(time.time())}.png", result_frames[-1])
                print("📸 截图已保存")
            elif key == ord('r'):
                current_type_idx = (current_type_idx + 1) % len(output_types)
                print(f"🔄 切换显示模式: {output_types[current_type_idx]}")
            del frames

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    finally:
        video_stream.stop()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print(f"✅ 完成！总帧数: {frame_count}")
