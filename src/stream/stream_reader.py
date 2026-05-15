import time
import threading
import os
import cv2


class RTSPStreamReader(threading.Thread):
    """RTSP流读取线程 - 修复版（解决 non-existing PPS 问题）"""

    def __init__(self, rtsp_url, frame_buffer, target_fps=None, skip_frames_on_reconnect=10):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.frame_buffer = frame_buffer
        self.target_fps = target_fps
        self.skip_frames_on_reconnect = skip_frames_on_reconnect  # 重连后跳过前N帧
        self.running = False
        self.connected = False
        self.frame_count = 0
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0

    def _create_capture(self):
        """创建新的 VideoCapture 实例"""
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        self.running = True
        cap = None
        consecutive_failures = 0
        max_failures = 5

        while self.running:
            if cap is None:
                print("🔄 正在连接 RTSP 流...")
                cap = self._create_capture()
                if not cap.isOpened():
                    print("❌ RTSP连接失败！")
                    time.sleep(3)
                    continue

                # 获取流信息
                self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = cap.get(cv2.CAP_PROP_FPS) or 25
                print(f"✅ RTSP连接成功！分辨率: {self.frame_width}x{self.frame_height}, 帧率: {self.fps:.1f}")
                self.connected = True
                consecutive_failures = 0

                # 重连后跳过前几帧（等待关键帧）
                for _ in range(self.skip_frames_on_reconnect):
                    cap.read()

            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                print(f"⚠️ 读取失败 #{consecutive_failures}")
                if consecutive_failures >= max_failures:
                    print("⚠️ 视频流中断，尝试重连...")
                    cap.release()
                    cap = None
                    time.sleep(2)
                continue

            # 成功读取，重置失败计数
            consecutive_failures = 0

            # 控制帧率
            current_time = time.time()
            if hasattr(self, '_last_frame_time'):
                frame_interval = 1.0 / self.target_fps if self.target_fps else 0
                if frame_interval > 0 and (current_time - self._last_frame_time) < frame_interval:
                    continue
            self._last_frame_time = current_time

            # 放入缓冲区
            self.frame_buffer.put(frame)
            self.frame_count += 1

        if cap is not None:
            cap.release()

    def stop(self):
        self.running = False