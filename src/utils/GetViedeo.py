"""
RTSP视频流录制 - 指定时长版本
"""

import os
import time
import threading
import queue
from datetime import datetime

import cv2
import numpy as np

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class FrameBuffer:
    """线程安全的帧缓冲区"""

    def __init__(self, maxsize=30):
        self.queue = queue.Queue(maxsize=maxsize)
        self.latest_frame = None
        self.lock = threading.Lock()

    def put(self, frame):
        """添加帧到缓冲区"""
        with self.lock:
            self.latest_frame = frame.copy()
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(frame)
            except:
                pass

    def get(self, timeout=1.0):
        """获取帧"""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest(self):
        """获取最新帧"""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def clear(self):
        """清空缓冲区"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break


class RTSPStreamReader(threading.Thread):
    """RTSP流读取线程"""

    def __init__(self, rtsp_url, frame_buffer, target_fps=None):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.frame_buffer = frame_buffer
        self.target_fps = target_fps
        self.running = False
        self.connected = False
        self.frame_count = 0
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0
        self.error_count = 0
        self.max_errors = 10

    def run(self):
        """线程主循环"""
        self.running = True

        cap = cv2.VideoCapture(self.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        if not cap.isOpened():
            print("❌ RTSP连接失败！")
            self.running = False
            return

        self.connected = True
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25

        print(f"✅ RTSP连接成功！")
        print(f"   分辨率: {self.frame_width}x{self.frame_height}")
        print(f"   帧率: {self.fps:.1f} FPS")

        frame_interval = 1.0 / self.target_fps if self.target_fps else 0
        last_frame_time = 0

        while self.running:
            ret, frame = cap.read()

            if not ret:
                self.error_count += 1
                print(f"⚠️ 读取帧失败 ({self.error_count}/{self.max_errors})")

                if self.error_count >= self.max_errors:
                    print("❌ 错误次数过多，停止录制")
                    break

                print("⚠️ 视频流中断，尝试重连...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(self.rtsp_url)
                continue

            self.error_count = 0  # 重置错误计数

            # 帧率控制
            current_time = time.time()
            if frame_interval > 0 and (current_time - last_frame_time) < frame_interval:
                continue
            last_frame_time = current_time

            self.frame_buffer.put(frame)
            self.frame_count += 1
        cap.release()
        print("📹 RTSP流读取线程已停止")

    def stop(self):
        """停止线程"""
        self.running = False


def record_rtsp_video(
        username="admin",
        password="a1b2c3d4",
        ip_address="192.168.1.168",
        port="554",
        channel="201",
        duration=60,
        output_path=None,
        target_fps=25,
        codec='mp4v',
        show_preview=True,
        preview_scale=0.5
):
    """
    录制RTSP视频流到本地文件

    参数:
        username: RTSP用户名
        password: RTSP密码
        ip_address: 摄像头IP地址
        port: RTSP端口
        channel: 通道号
        duration: 录制时长（秒），None表示手动停止
        output_path: 输出文件路径，None则自动生成
        target_fps: 目标帧率
        codec: 视频编码格式 ('mp4v', 'XVID', 'H264', 'avc1')
        show_preview: 是否显示预览窗口
        preview_scale: 预览窗口缩放比例

    返回:
        录制的视频文件路径
    """

    # 构建RTSP URL
    rtsp_url = f"rtsp://{username}:{password}@{ip_address}:{port}/Streaming/Channels/{channel}"
    print(f"\n{'=' * 60}")
    print(f"RTSP视频录制")
    print(f"{'=' * 60}")
    print(f"RTSP URL: rtsp://{username}:****@{ip_address}:{port}/Streaming/Channels/{channel}")
    # 生成输出文件名
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"rtsp_record_{timestamp}.mp4"

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"输出文件: {output_path}")
    print(f"录制时长: {duration}秒" if duration else "录制时长: 手动停止")
    print(f"目标帧率: {target_fps} FPS")

    # 创建帧缓冲区
    frame_buffer = FrameBuffer(maxsize=30)

    # 启动RTSP流读取线程
    stream_reader = RTSPStreamReader(rtsp_url, frame_buffer, target_fps=target_fps)
    stream_reader.start()

    # 等待连接
    print("\n正在连接RTSP流...")
    timeout = 10
    start_time = time.time()
    while not stream_reader.connected and time.time() - start_time < timeout:
        time.sleep(0.1)

    if not stream_reader.connected:
        print("❌ 连接超时！")
        stream_reader.stop()
        return None

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        target_fps,
        (stream_reader.frame_width, stream_reader.frame_height)
    )

    if not video_writer.isOpened():
        print(f"❌ 无法创建视频写入器！尝试使用的编码: {codec}")
        stream_reader.stop()
        return None

    print(f"\n✅ 开始录制...")
    print(f"   分辨率: {stream_reader.frame_width}x{stream_reader.frame_height}")
    print(f"   编码格式: {codec}")

    if show_preview:
        print("\n按键说明:")
        print("  'q' - 停止录制")
        print("  's' - 截图")

    # 录制循环
    record_start_time = time.time()
    frame_count = 0
    screenshot_count = 0
    last_print_time = time.time()

    try:
        while stream_reader.running:
            # 检查是否达到指定时长
            if duration and (time.time() - record_start_time) >= duration:
                print(f"\n✅ 达到指定录制时长 {duration}秒")
                break

            # 获取最新帧
            frame = frame_buffer.get_latest()
            if frame is None:
                time.sleep(0.01)
                continue

            # 写入视频
            video_writer.write(frame)
            frame_count += 1

            # 显示预览
            if show_preview:
                if preview_scale != 1.0:
                    display_h = int(frame.shape[0] * preview_scale)
                    display_w = int(frame.shape[1] * preview_scale)
                    display_frame = cv2.resize(frame, (display_w, display_h))
                else:
                    display_frame = frame.copy()

                # 添加信息文字
                elapsed_time = time.time() - record_start_time
                remaining_time = duration - elapsed_time if duration else 0

                info_lines = [
                    f"Recording: {elapsed_time:.1f}s",
                    f"Frames: {frame_count}",
                    f"FPS: {frame_count / elapsed_time:.1f}" if elapsed_time > 0 else "FPS: --"
                ]

                if duration:
                    info_lines.append(f"Remaining: {remaining_time:.1f}s")

                y_offset = 30
                for line in info_lines:
                    cv2.putText(display_frame, line, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25

                # 录制指示
                cv2.circle(display_frame, (display_frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                cv2.imshow('RTSP Recording (Press q to stop, s for screenshot)', display_frame)

                # 处理按键
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n⏹️  用户停止录制")
                    break
                elif key == ord('s'):
                    screenshot_name = f"screenshot_{int(time.time())}_{screenshot_count}.png"
                    cv2.imwrite(screenshot_name, frame)
                    print(f"📸 截图保存: {screenshot_name}")
                    screenshot_count += 1
            # 定期打印进度（每5秒）
            if time.time() - last_print_time >= 5:
                elapsed = time.time() - record_start_time
                print(f"⏺️  录制中... {elapsed:.1f}s | 帧数: {frame_count} | 平均FPS: {frame_count / elapsed:.1f}")
                last_print_time = time.time()

    except KeyboardInterrupt:
        print("\n⏹️  用户中断录制")

    finally:
        # 清理资源
        stream_reader.stop()
        stream_reader.join(timeout=2)
        video_writer.release()

        if show_preview:
            cv2.destroyAllWindows()

        # 统计信息
        total_time = time.time() - record_start_time
        actual_fps = frame_count / total_time if total_time > 0 else 0

        print(f"\n{'=' * 60}")
        print(f"录制完成")
        print(f"{'=' * 60}")
        print(f"输出文件: {output_path}")
        print(f"录制时长: {total_time:.2f}秒")
        print(f"总帧数: {frame_count}")
        print(f"平均帧率: {actual_fps:.2f} FPS")
        print(f"文件大小: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
        print(f"{'=' * 60}\n")

        return output_path


def record_multiple_segments(
        username="admin",
        password="a1b2c3d4",
        ip_address="192.168.1.168",
        port="554",
        channel="101",
        segment_duration=60,
        num_segments=5,
        output_dir="./recordings",
        target_fps=25,
        codec='mp4v'
):
    """
    录制多个视频片段

    参数:
        segment_duration: 每段时长（秒）
        num_segments: 录制段数
        output_dir: 输出目录
        其他参数同 record_rtsp_video

    返回:
        录制的视频文件路径列表
    """

    os.makedirs(output_dir, exist_ok=True)

    recorded_files = []

    print(f"\n{'=' * 60}")
    print(f"批量录制 - {num_segments}段，每段{segment_duration}秒")
    print(f"{'=' * 60}\n")

    for i in range(num_segments):
        print(f"\n📹 录制第 {i + 1}/{num_segments} 段...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"segment_{i + 1:03d}_{timestamp}.mp4")

        result = record_rtsp_video(
            username=username,
            password=password,
            ip_address=ip_address,
            port=port,
            channel=channel,
            duration=segment_duration,
            output_path=output_path,
            target_fps=target_fps,
            codec=codec,
            show_preview=True
        )

        if result:
            recorded_files.append(result)
        else:
            print(f"❌ 第 {i + 1} 段录制失败")
            break

        # 段间休息
        if i < num_segments - 1:
            print(f"\n⏸️  休息5秒后继续...")
            time.sleep(5)

    print(f"\n✅ 批量录制完成！共录制 {len(recorded_files)} 段")
    return recorded_files


# ==================== 主程序 ====================

if __name__ == "__main__":
    # RTSP配置
    rtsp_config = {
        'username': "admin",
        'password': "a1b2c3d4",
        'ip_address': "192.168.1.168",
        'port': "554",
        'channel': "101",
    }

    # 录制配置
    record_config = {
        'duration': 100,  # 录制30秒，None表示手动停止
        'target_fps': 25,  # 目标帧率
        'codec': 'mp4v',  # 编码格式: 'mp4v', 'XVID', 'H264', 'avc1'
        'show_preview': True,  # 显示预览
        'preview_scale': 0.5,  # 预览缩放
    }

    # ========== 方式1: 录制单个视频 ==========
    print("\n开始录制...")
    output_file = record_rtsp_video(
        **rtsp_config,
        **record_config,
        output_path="./my_recording_0109_1.mp4"
    )

    if output_file:
        print(f"✅ 录制成功: {output_file}")

    # ========== 方式2: 录制多个片段 ==========
    # recorded_files = record_multiple_segments(
    #     **rtsp_config,
    #     segment_duration=60,      # 每段60秒
    #     num_segments=3,           # 录制3段
    #     output_dir="./recordings",
    #     target_fps=25,
    #     codec='mp4v'
    # )
    #
    # print(f"\n录制的文件:")
    # for f in recorded_files:
    #     print(f"  - {f}")