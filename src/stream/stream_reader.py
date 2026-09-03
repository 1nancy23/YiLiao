import time
import threading
import os
import subprocess
import cv2


def _runtime_logs_enabled():
    return os.environ.get("YILIAO_RUNTIME_LOGS", "0").lower() in ("1", "true", "yes", "on")


def _runtime_log(*args, **kwargs):
    if _runtime_logs_enabled():
        print(*args, **kwargs)


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
                _runtime_log("🔄 正在连接 RTSP 流...")
                cap = self._create_capture()
                if not cap.isOpened():
                    _runtime_log("❌ RTSP连接失败！")
                    time.sleep(3)
                    continue

                # 获取流信息
                self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = cap.get(cv2.CAP_PROP_FPS) or 25
                _runtime_log(f"✅ RTSP连接成功！分辨率: {self.frame_width}x{self.frame_height}, 帧率: {self.fps:.1f}")
                self.connected = True
                consecutive_failures = 0

                # 重连后跳过前几帧（等待关键帧）
                for _ in range(self.skip_frames_on_reconnect):
                    cap.read()

            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                _runtime_log(f"⚠️ 读取失败 #{consecutive_failures}")
                if consecutive_failures >= max_failures:
                    _runtime_log("⚠️ 视频流中断，尝试重连...")
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


class USBStreamReader(threading.Thread):
    """Read the verified 4K MJPG USB camera into the shared frame buffer."""

    def __init__(
        self,
        device,
        frame_buffer,
        target_fps=None,
        width=3840,
        height=2160,
        capture_fps=30,
        warmup_frames=25,
        controls=None,
    ):
        super().__init__(daemon=True)
        self.device = str(device)
        self.frame_buffer = frame_buffer
        self.target_fps = float(target_fps or 0)
        self.request_width = int(width)
        self.request_height = int(height)
        self.capture_fps = float(capture_fps)
        self.warmup_frames = max(0, int(warmup_frames))
        self.controls = dict(controls or {})
        self.running = False
        self.connected = False
        self.frame_count = 0
        self.fps = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self.fourcc = ""

    @staticmethod
    def _fourcc_text(value):
        value = int(value)
        return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))

    def _apply_controls(self):
        for name, value in self.controls.items():
            result = subprocess.run(
                [
                    "v4l2-ctl",
                    "-d",
                    self.device,
                    "--set-ctrl={}={}".format(name, value),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "USB camera control {}={} failed: {}".format(
                        name,
                        value,
                        result.stdout.strip(),
                    )
                )

    def _create_capture(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise ConnectionError("Cannot open USB camera {}".format(self.device))

        # Keep the exact order used by the verified USB test scripts.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.request_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.request_height)
        cap.set(cv2.CAP_PROP_FPS, self.capture_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fourcc = self._fourcc_text(cap.get(cv2.CAP_PROP_FOURCC))
        if self.frame_width != self.request_width or self.frame_height != self.request_height:
            cap.release()
            raise RuntimeError(
                "USB 4K negotiation failed: {}x{}".format(
                    self.frame_width,
                    self.frame_height,
                )
            )
        if self.fourcc != "MJPG":
            cap.release()
            raise RuntimeError("USB MJPG negotiation failed: {}".format(self.fourcc))

        self._apply_controls()
        return cap

    def run(self):
        self.running = True
        cap = None
        consecutive_failures = 0
        last_frame_time = 0.0
        try:
            while self.running:
                if cap is None:
                    try:
                        cap = self._create_capture()
                    except Exception as exc:
                        self.connected = False
                        _runtime_log("USB camera connection failed: {}".format(exc))
                        time.sleep(2)
                        continue

                    for _ in range(self.warmup_frames):
                        if not self.running:
                            break
                        cap.read()
                    self.connected = self.running
                    consecutive_failures = 0
                    last_frame_time = 0.0
                    _runtime_log(
                        "USB camera connected: {}x{} {:.1f} FPS {}".format(
                            self.frame_width,
                            self.frame_height,
                            self.fps,
                            self.fourcc,
                        )
                    )

                ok, frame = cap.read()
                if not ok or frame is None or not frame.size:
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        self.connected = False
                        cap.release()
                        cap = None
                        time.sleep(1)
                    continue

                consecutive_failures = 0
                now = time.time()
                if self.target_fps > 0 and last_frame_time:
                    if now - last_frame_time < 1.0 / self.target_fps:
                        continue
                last_frame_time = now
                self.frame_buffer.put(frame)
                self.frame_count += 1
        finally:
            self.connected = False
            if cap is not None:
                cap.release()

    def stop(self):
        self.running = False
