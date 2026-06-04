import queue
import threading

class FrameBuffer:
    """线程安全的帧缓冲区 - 支持批量获取"""

    def __init__(self, maxsize=30):
        self.queue = queue.Queue(maxsize=maxsize)
        self.latest_frame = None
        self.lock = threading.Lock()

    def put(self, frame):
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

    def get_latest(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_batch(self, batch_size):
        """批量获取帧"""
        frames = []
        for _ in range(batch_size):
            try:
                frame = self.queue.get_nowait()
                frames.append(frame)
            except queue.Empty:
                break
        return frames

    def clear(self):
        """清空缓冲区中的所有帧"""
        with self.lock:
            self.latest_frame = None
        # 清空队列中的所有元素
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
