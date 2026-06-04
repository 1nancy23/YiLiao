# src/stream/video_stream.py
import time
import threading
from src.stream.frame_buffer import FrameBuffer
from src.stream.stream_reader import RTSPStreamReader


class VideoStream:
    """封装RTSP视频流获取逻辑"""

    def __init__(self, rtsp_url, target_fps=15, max_buffer_size=30):
        """
        初始化视频流

        Args:
            rtsp_url: RTSP流地址
            target_fps: 目标帧率
            max_buffer_size: 帧缓冲区最大大小
        """
        self.rtsp_url = rtsp_url
        self.target_fps = target_fps
        self.max_buffer_size = max_buffer_size
        self.frame_buffer = FrameBuffer(maxsize=max_buffer_size)
        self.stream_reader = RTSPStreamReader(rtsp_url, self.frame_buffer, target_fps * 2)
        self.running = False
        self._thread = None

    def start(self):
        """启动视频流读取线程"""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

        # 等待连接建立
        timeout = 60


        start_time = time.time()
        while not self.stream_reader.connected and time.time() - start_time < timeout:
            time.sleep(0.1)

        if not self.stream_reader.connected:
            raise ConnectionError(f"RTSP连接超时: {self.rtsp_url}")

    def _run(self):
        """内部线程运行函数"""
        self.stream_reader.start()
        while self.running:
            time.sleep(0.1)
        self.stream_reader.stop()

    def get_batch(self, batch_size=1):
        """
        获取一批帧

        Args:
            batch_size: 要获取的帧数

        Returns:
            list: 包含batch_size帧的列表，如果不够则返回实际数量
        """
        if not self.running:
            raise RuntimeError("视频流未启动")

        frames = self.frame_buffer.get_batch(batch_size)
        return frames

    def stop(self):
        """停止视频流"""
        if not self.running:
            return
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self.stream_reader.stop()
        self.frame_buffer.clear()

    def is_connected(self):
        """检查连接状态"""
        return self.stream_reader.connected

