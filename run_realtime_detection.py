import time
from collections import deque
import cv2
import numpy as np
import torch
import threading
from pynput import keyboard
from sympy.codegen.ast import continue_

from src.segmentation.segmenter import RealtimeSegmentationProcessor
from src.segmentation.cropDetectedArea import crop_target_regions
from src.processor.img_cropper import crop_image_by_coordinates
from src.stream.video_stream import VideoStream
from src.utils.img_utils import ImageProcessor
from src.utils.single_img_saver import img_shower


def run_realtime_detection(
        # RTSP配置
        username="admin",
        password="a1b2c3d4",
        ip_address="192.168.1.168",
        port="554",
        channel="201",
        # 模型配置
        model=None,
        checkpoint_path=None,
        num_classes=2,
        # 组件模型
        ocr_recognizer=None,
        drug_matcher=None,
        classifier=None,
        length = None,
        # 处理配置
        tile_size=1024,
        overlap=64,
        target_fps=15,
        batch_frames=2,
        # 区域界定
        crop_region=None,
        # 显示配置
        output_type='overlay',
        overlay_alpha=0.5,
        display_scale=0.5,
        # 保存配置
        save_video=False,
        output_path="./realtime_output.mp4",
        save_fps=15,
        # 其他
        max_frames=None,
        device=None,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")

    # 构建RTSP URL
    rtsp_url = f"rtsp://{username}:{password}@{ip_address}:{port}/Streaming/Channels/{channel}?rtsp://admin:a1b2c3d4@192.168.1.168:554/Streaming/Channels/101?transportmode=unicast&profile=media2"
    print(f"📡 连接: rtsp://{username}:****@{ip_address}:{port}/Streaming/Channels/{channel}")

    # 创建视频流对象
    video_stream = VideoStream(rtsp_url, target_fps=target_fps)
    video_stream.start()

    # 等待连接
    print("⏳ 正在连接RTSP流...")
    timeout = 10
    start_time = time.time()
    while not video_stream.is_connected() and time.time() - start_time < timeout:
        time.sleep(0.1)
    if not video_stream.is_connected():
        print("❌ 连接超时！")
        video_stream.stop()
        return

    # 创建分割处理器
    processor = RealtimeSegmentationProcessor(
        model=model,
        device=device,
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        frame_height=video_stream.stream_reader.frame_height,
        frame_width=video_stream.stream_reader.frame_width,
        tile_size=tile_size,
        overlap=overlap,
        batch_frames=batch_frames,
    )

    # 视频写入器
    video_writer = None
    if save_video:
        out_width = video_stream.stream_reader.frame_width * 2 if output_type == 'side_by_side' else video_stream.stream_reader.frame_width
        out_height = video_stream.stream_reader.frame_height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, save_fps, (out_width, out_height))
        print(f"📹 录制到: {output_path}")

    # 显示模式
    output_types = ['overlay', 'color', 'side_by_side', 'gray']
    current_type_idx = output_types.index(output_type) if output_type in output_types else 0

    # 性能统计
    fps_counter = deque(maxlen=30)
    frame_count = 0
    batch_count = 0

    print("\n" + "=" * 60)
    print("🚀 实时分割已启动 (深度GPU批量加速)")
    print(f" 批量大小: {batch_frames} 帧")
    print(" 按键: 'q'-退出 's'-截图 'r'-切换模式 空格-触发识别")
    print("=" * 60 + "\n")

    # ========== 键盘监听（线程安全） ==========
    trigger_event = threading.Event()  # 触发事件

    def on_press(key):
        try:
            if key == keyboard.Key.space:
                trigger_event.set()
                print("✅ 人工触发：已按下空格键，开始处理当前批次")
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # ========== 存储最近批次的图像和预测结果（自动淘汰） ==========
    max_batches_kept = 15  # 最多保留30个批次
    n_batch_frames = deque(maxlen=max_batches_kept)
    n_batch_predictions = deque(maxlen=max_batches_kept)

    
    import threading
    from queue import Queue

    
    def process_batch_if_triggered():
        """处理当前批次的药瓶，仅在键盘触发时执行"""
        if trigger_event.is_set() and n_batch_frames:
            a = time.time()
            # 找到最清晰图像
            best_frame, frame_position = ImageProcessor.select_sharpest_image(n_batch_frames)
            print(f"检测到第{frame_position[0] + 1}张图片最清晰，后续处理采用该张图片")
            pre_result = n_batch_predictions[frame_position[0]]

            # 裁剪预测结果为药瓶药袋
            cropped_bag, cropped_bottle = crop_target_regions(best_frame, pre_result, 30000, 15000)

            # 批量保存图像使用：
            # img_shower(cropped_bag, cropped_bottle, save=True, show=True)
            # return None


            if cropped_bag and cropped_bottle:
                print("\n" + "=" * 60)
                print(f"共识别到药瓶 {len(cropped_bottle)} 个，药袋 {len(cropped_bag)} 个")
                print("=" * 60)

                # ---------- 1. 处理所有药瓶 ----------
                final_bottle_medicines = []  # 最终识别的药品列表（去重后）

                # 对每个药瓶进行 OCR + 模糊匹配 + 特征匹配
                for idx, bottle_img in enumerate(cropped_bottle):
                    print(f"\n--- 药瓶 {idx + 1} 处理 ---")

                    # OCR 识别
                    ocr_text = ocr_recognizer.recognize(bottle_img)
                    if not ocr_text:
                        print("  OCR 无有效文本，跳过")
                        continue
                    print(f"  OCR 结果: {ocr_text}")

                    # 模糊匹配得到候选列表
                    candidate_names = drug_matcher.match(ocr_text, match_type='bottle',threshold=50, limit=10)
                    if not candidate_names:
                        print("  模糊匹配无候选，跳过")
                        continue
                    print(
                        f"  模糊匹配候选 ({len(candidate_names)}): {candidate_names[:3]}{'...' if len(candidate_names) > 3 else ''}")

                    # 特征精细匹配
                    if classifier is not None:
                        try:
                            feat_result = classifier.classify(candidate_names, bottle_img)
                            best_match = feat_result['predicted_category']
                            confidence = feat_result['confidence']
                            print(f"  特征匹配最佳: {best_match} (置信度: {confidence:.4f})")
                            final_bottle_medicines.append(best_match)
                        except Exception as e:
                            print(f"  特征匹配失败: {e}，使用模糊匹配第一个")
                            final_bottle_medicines.append(candidate_names[0])
                    else:
                        # 无分类器，直接取第一个
                        final_bottle_medicines.append(candidate_names[0])
                        print(f"  无分类器，使用模糊匹配第一个: {candidate_names[0]}")

                # 去重
                final_bottle_medicines = list(set(final_bottle_medicines))
                print(f"\n最终识别到的药品（去重后）: {final_bottle_medicines}")
                # print(type(candidate_names), repr(candidate_names[0]))

                # ---------- 2. 处理药袋获取患者姓名 ----------
                # OCR 识别

                print(f"  OCR 结果: {ocr_text}")

                # 模糊匹配得到候选列表

                if cropped_bag:
                    # 取第一个药袋进行识别
                    bag_ocr = ocr_recognizer.recognize(cropped_bag[0])
                    if bag_ocr:
                        patient_name = drug_matcher.match(bag_ocr, match_type='bag', threshold=50, limit=10)

                        print(f"\n药袋OCR识别为患者姓名: {patient_name}")
                    else:
                        patient_name = None
                        print("\n药袋识别失败，无法获取患者姓名")
                else:
                    print("\n未检测到药袋")

                # ---------- 3. 与数据库比对 ----------
                if patient_name and final_bottle_medicines:
                    batch_id = 1  # 批次固定为1
                    validation = drug_matcher.check_patient_batch_medicines(
                        patient_name=patient_name,
                        batch_id=batch_id,
                        expected_medicine_names=final_bottle_medicines
                    )

                    # 打印比对结果
                    print("\n" + "-" * 60)
                    print("【匹配结果汇总】")
                    print(f"患者姓名: {patient_name} (批次: {batch_id})")
                    print(f"识别到的药品: {final_bottle_medicines}")
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
                        print(f"⚠️ 患者 {patient_name} 批次 {batch_id} 不存在于数据库")
                    print("-" * 60)
                else:
                    print("\n无法完成比对：缺少患者姓名或识别药品")

                print(f"\n一次识别执行完毕，用时{time.time() - a:.2f}s")
            else:
                print("未识别到有效药瓶和药袋，请重新拍摄")

            # 清除触发标志，等待下一次空格
            trigger_event.clear()

    # 开始运行
    try:
        while video_stream.running:
            # 获取一批帧
            frames = video_stream.get_batch(batch_frames)
            if not frames:
                time.sleep(0.001)
                continue

            # 执行裁剪,保留坐标内的图像进行后续处理
            if crop_region:
                frames = np.array(frames)
                cropped_frame = crop_image_by_coordinates(frames, crop_region)
                frames = list(cropped_frame)

            # 分割模型进行预测
            process_start = time.time()
            result_frames, predictions = processor.process_frames_batch(
                frames, output_type=output_types[current_type_idx], alpha=overlay_alpha
            )
            process_time = time.time() - process_start

            # 将识别结果存入队列（自动淘汰旧数据）
            n_batch_frames.append(frames)
            n_batch_predictions.append(predictions)

            # 检查并处理触发事件
            process_batch_if_triggered()

            # FPS统计
            batch_fps = len(frames) / max(process_time, 0.001)
            fps_counter.append(batch_fps)
            current_fps = np.mean(list(fps_counter))

            # 处理每一帧的输出
            for result_frame in result_frames:
                if video_writer:
                    video_writer.write(result_frame)
                if display_scale != 1.0:
                    display_h = int(result_frame.shape[0] * display_scale)
                    display_w = int(result_frame.shape[1] * display_scale)
                    display_frame = cv2.resize(result_frame, (display_w, display_h))
                else:
                    display_frame = result_frame.copy()

                info = f"FPS: {current_fps:.1f} | Batch: {len(frames)} | Mode: {output_types[current_type_idx]} | Frame: {frame_count}"
                cv2.putText(display_frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if save_video:
                    cv2.putText(display_frame, "REC", (display_frame.shape[1] - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)
                cv2.imshow('Realtime Segmentation [Deep GPU Accelerated]', display_frame)
                frame_count += 1
                batch_count += 1

            if max_frames and frame_count >= max_frames:
                print(f"✅ 达到最大帧数 {max_frames}")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 用户退出")
                break
            elif key == ord('s'):
                cv2.imwrite(f"screenshot_{int(time.time())}.png", result_frames[-1])
                print("📸 截图已保存")
            elif key == ord('r'):
                current_type_idx = (current_type_idx + 1) % len(output_types)
                print(f"🔄 模式: {output_types[current_type_idx]}")

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    finally:
        video_stream.stop()
        if video_writer:
            video_writer.release()
            print(f"✅ 视频已保存: {output_path}")
        cv2.destroyAllWindows()
        print(f"\n✅ 完成！总帧数: {frame_count}, 总批次: {batch_count}")
        print(f" 平均FPS: {frame_count / (time.time() - start_time):.1f}")

