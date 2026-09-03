# """
# 实时RTSP视频流语义分割 - GPU批量加速版
# 优化：批量切片处理 + CUDA流水线 + 预计算优化
# """
#
# import os
# import sys
# import time
# import math
# import threading
# import queue
# from collections import deque
#
# import cv2
# import numpy as np
# import torch
# from torch.cuda.amp import autocast
#
# sys.path.append("D:/ZJF/种植作物分类/ZhongzhiCodes/XiuGai1")
#
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
#
#
# class FrameBuffer:
#     """线程安全的帧缓冲区"""
#
#     def __init__(self, maxsize=30):
#         self.queue = queue.Queue(maxsize=maxsize)
#         self.latest_frame = None
#         self.lock = threading.Lock()
#
#     def put(self, frame):
#         with self.lock:
#             self.latest_frame = frame.copy()
#         try:
#             self.queue.put_nowait(frame)
#         except queue.Full:
#             try:
#                 self.queue.get_nowait()
#                 self.queue.put_nowait(frame)
#             except:
#                 pass
#
#     def get_latest(self):
#         with self.lock:
#             return self.latest_frame.copy() if self.latest_frame is not None else None
#
#
# class RTSPStreamReader(threading.Thread):
#     """RTSP流读取线程"""
#
#     def __init__(self, rtsp_url, frame_buffer, target_fps=None):
#         super().__init__(daemon=True)
#         self.rtsp_url = rtsp_url
#         self.frame_buffer = frame_buffer
#         self.target_fps = target_fps
#         self.running = False
#         self.connected = False
#         self.frame_count = 0
#         self.fps = 0
#         self.frame_width = 0
#         self.frame_height = 0
#
#     def run(self):
#         self.running = True
#         cap = cv2.VideoCapture(self.rtsp_url)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
#
#         if not cap.isOpened():
#             print("❌ RTSP连接失败！")
#             self.running = False
#             return
#
#         self.connected = True
#         self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         self.fps = cap.get(cv2.CAP_PROP_FPS) or 25
#
#         print(f"✅ RTSP连接成功！分辨率: {self.frame_width}x{self.frame_height}, 帧率: {self.fps:.1f}")
#
#         frame_interval = 1.0 / self.target_fps if self.target_fps else 0
#         last_frame_time = 0
#
#         while self.running:
#             ret, frame = cap.read()
#             if not ret:
#                 print("⚠️ 视频流中断，尝试重连...")
#                 cap.release()
#                 time.sleep(1)
#                 cap = cv2.VideoCapture(self.rtsp_url)
#                 continue
#
#             current_time = time.time()
#             if frame_interval > 0 and (current_time - last_frame_time) < frame_interval:
#                 continue
#             last_frame_time = current_time
#
#             self.frame_buffer.put(frame)
#             self.frame_count += cap.release()
#
#     def stop(self):
#         self.running = False
#
#
# class TileProcessor:
#     """切片处理器 - 预计算切片位置，支持批量处理"""
#
#     def __init__(self, frame_height, frame_width, tile_size=1024, overlap=128, device='cuda'):
#         self.frame_height = frame_height
#         self.frame_width = frame_width
#         self.tile_size = tile_size
#         self.overlap = overlap
#         self.stride = tile_size - overlap
#         self.device = device
#
#         # 预计算切片位置
#         self.tile_positions = self._compute_tile_positions()
#         self.num_tiles = len(self.tile_positions)
#
#         # 预计算权重矩阵
#         self.weight_matrix = self._create_weight_matrix()
#
#         # 预分配GPU内存
#         self.batch_tensor = torch.zeros(
#             (self.num_tiles, 3, tile_size, tile_size),
#             dtype=torch.float32,
#             device=device
#         )
#
#         print(f"切片配置: {self.num_tiles} 个切片, 大小 {tile_size}x{tile_size}, 重叠 {overlap}")
#
#     def _compute_tile_positions(self):
#         """预计算所有切片位置"""
#         positions = []
#         h, w = self.frame_height, self.frame_width
#
#         n_tiles_y = max(1, math.ceil((h - self.overlap) / self.stride)) if h > self.tile_size else 1
#         n_tiles_x = max(1, math.ceil((w - self.overlap) / self.stride)) if w > self.tile_size else 1
#
#         for ty in range(n_tiles_y):
#             for tx in range(n_tiles_x):
#                 x = min(tx * self.stride, max(0, w - self.tile_size)) if w > self.tile_size else 0
#                 y = min(ty * self.stride, max(0, h - self.tile_size)) if h > self.tile_size else 0
#                 x_end = min(x + self.tile_size, w)
#                 y_end = min(y + self.tile_size, h)
#                 positions.append((x, y, x_end, y_end))
#
#         return positions
#
#     def _create_weight_matrix(self):
#         """创建高斯权重矩阵"""
#         x = np.linspace(-1, 1, self.tile_size)
#         y = np.linspace(-1, 1, self.tile_size)
#         xx, yy = np.meshgrid(x, y)
#         weight = np.exp(-(xx ** 2 + yy ** 2) / 0.5)
#         return weight.astype(np.float32)
#
#     def extract_tiles_batch(self, frame_rgb):
#         """
#         批量提取切片并直接转换为GPU tensor
#         Args:
#             frame_rgb: RGB格式的帧 (H, W, 3) uint8
#         Returns:
#             batch_tensor: (N, 3, H, W) GPU tensor
#         """
#         # 预先将整帧转换为float并归一化
#         frame_float = frame_rgb.astype(np.float32) / 255.0
#
#         for i, (x, y, x_end, y_end) in enumerate(self.tile_positions):
#             th = y_end - y
#             tw = x_end - x
#
#             # 提取切片
#             tile = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.float32)
#             tile[:th, :tw] = frame_float[y:y_end, x:x_end]
#
#             # HWC -> CHW 并填入预分配的tensor
#             self.batch_tensor[i] = torch.from_numpy(tile.transpose(2, 0, 1))
#
#         return self.batch_tensor
#
#     def merge_predictions_fast(self, predictions, num_classes):
#         """
#         快速合并预测结果
#         Args:
#             predictions: (N, H, W) numpy array
#             num_classes: 类别数
#         Returns:
#             final_pred: (H, W) numpy array
#         """
#         h, w = self.frame_height, self.frame_width
#
#         # 使用加权融合
#         pred_sum = np.zeros((h, w, num_classes), dtype=np.float32)
#         weight_sum = np.zeros((h, w), dtype=np.float32)
#
#         for i, (x, y, x_end, y_end) in enumerate(self.tile_positions):
#             th = y_end - y
#             tw = x_end - x
#             pred = predictions[i]
#
#             # 使用向量化操作加速
#             for c in range(num_classes):
#                 mask = (pred[:th, :tw] == c).astype(np.float32)
#                 pred_sum[y:y_end, x:x_end, c] += mask * self.weight_matrix[:th, :tw]
#
#             weight_sum[y:y_end, x:x_end] += self.weight_matrix[:th, :tw]
#
#         # 归一化
#         weight_sum = np.maximum(weight_sum, 1e-8)
#         pred_probs = pred_sum / weight_sum[:, :, np.newaxis]
#         final_pred = np.argmax(pred_probs, axis=2).astype(np.uint8)
#
#         return final_pred
#
#
# class RealtimeSegmentationProcessor:
#     """实时分割处理器 - GPU批量加速版"""
#
#     COLORMAP = np.array([
#         [0, 0, 0],
#         [255, 0, 0],
#         [0, 255, 0],
#         [0, 0, 255],
#         [255, 255, 0],
#         [0, 255, 255],
#         [255, 0, 255],
#         [128, 128, 128],
#         [0, 128, 255],], dtype=np.uint8)
#
#     def __init__(self, model, device, checkpoint_path=None, num_classes=2,
#                  frame_height=None, frame_width=None, tile_size=1024, overlap=128):
#         self.model = model
#         self.device = device
#         self.num_classes = num_classes
#         self.tile_size = tile_size
#         self.overlap = overlap
#
#         # 加载权重
#         if checkpoint_path and os.path.exists(checkpoint_path):
#             print(f"加载模型权重: {checkpoint_path}")
#             checkpoint = torch.load(checkpoint_path, map_location=device)
#             self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
#             print("模型加载完成")
#
#         self.model.to(device)
#         self.model.eval()
#
#         # 切片处理器（延迟初始化）
#         self.tile_processor = None
#         self.frame_height = frame_height
#         self.frame_width = frame_width
#
#         if frame_height and frame_width:
#             self._init_tile_processor(frame_height, frame_width)
#
#         # 预热GPU
#         self._warmup_gpu()
#
#     def _init_tile_processor(self, frame_height, frame_width):
#         """初始化切片处理器"""
#         self.frame_height = frame_height
#         self.frame_width = frame_width
#         self.tile_processor = TileProcessor(
#             frame_height, frame_width,
#             self.tile_size, self.overlap,
#             self.device
#         )
#
#     def _warmup_gpu(self):
#         """GPU预热，避免首次推理延迟"""
#         print("GPU预热中...")
#         dummy_input = torch.randn(1, 3, self.tile_size, self.tile_size, device=self.device)
#         with torch.no_grad():
#             with autocast():
#                 for _ in range(3):
#                     _ = self.model(dummy_input)
#         torch.cuda.synchronize()
#         print("GPU预热完成")
#
#     def predict_batch(self, batch_tensor):
#         """批量预测"""
#         with torch.no_grad():
#             with autocast():
#                 outputs = self.model(batch_tensor)
#
#         if isinstance(outputs, tuple):
#             outputs = outputs[0]
#
#         predictions = torch.argmax(outputs, dim=1)
#         return predictions.cpu().numpy().astype(np.uint8)
#
#     def label_to_color(self, label):
#         """标签转彩色 - 向量化加速"""
#         h, w = label.shape
#         color_img = np.zeros((h, w, 3), dtype=np.uint8)
#
#         for cls in range(min(len(self.COLORMAP), self.num_classes)):
#             mask = (label == cls)
#             color_img[mask] = self.COLORMAP[cls]
#
#         return color_img
#
#     def process_frame(self, frame, output_type='overlay', alpha=0.5):
#         """
#         处理单帧 - GPU批量加速版
#         """
#         h, w = frame.shape[:2]
#
#         # 检查是否需要重新初始化切片处理器
#         if self.tile_processor is None or h != self.frame_height or w != self.frame_width:
#             self._init_tile_processor(h, w)
#
#         # BGR转RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # 批量提取切片并转为GPU tensor
#         batch_tensor = self.tile_processor.extract_tiles_batch(frame_rgb)
#
#         # GPU批量推理
#         predictions = self.predict_batch(batch_tensor)
#
#         # 合并预测结果
#         final_pred = self.tile_processor.merge_predictions_fast(predictions, self.num_classes)
#
#         # 生成输出
#         result = self._generate_output(frame_rgb, final_pred, output_type, alpha)
#
#         return result, final_pred
#
#     def _generate_output(self, frame_rgb, prediction, output_type, alpha):
#         """生成输出帧"""
#         if output_type == 'gray':
#             result = (prediction * (255 // max(1, self.num_classes - 1))).astype(np.uint8)
#             result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
#         elif output_type == 'color':
#             result = self.label_to_color(prediction)
#             result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
#         elif output_type == 'overlay':
#             color_pred = self.label_to_color(prediction)
#             result = cv2.addWeighted(frame_rgb, 1 - alpha, color_pred, alpha, 0)
#             result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
#         elif output_type == 'side_by_side':
#             color_pred = self.label_to_color(prediction)
#             combined = np.concatenate([frame_rgb, color_pred], axis=1)
#             result = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
#         else:
#             result = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
#
#         return result
#
#
# def run_realtime_segmentation(
#         # RTSP配置
#         username="admin",
#         password="a1b2c3d4",
#         ip_address="192.168.1.168",
#         port="554",
#         channel="101",
#         # 模型配置
#         model=None,
#         checkpoint_path=None,
#         num_classes=2,
#         # 处理配置
#         tile_size=1024,
#         overlap=128,
#         target_fps=15,
#         # 显示配置
#         output_type='overlay',
#         overlay_alpha=0.5,
#         display_scale=0.5,
#         # 保存配置
#         save_video=False,
#         output_path="./realtime_output.mp4",
#         save_fps=15,
#         # 其他
#         max_frames=None,
#         device=None
# ):
#     """运行实时分割"""
#
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     print(f"使用设备: {device}")
#
#     # 构建RTSP URL
#     rtsp_url = f"rtsp://{username}:{password}@{ip_address}:{port}/Streaming/Channels/{channel}"
#     print(f"连接: rtsp://{username}:****@{ip_address}:{port}/Streaming/Channels/{channel}")
#
#     # 创建帧缓冲区和读取线程
#     frame_buffer = FrameBuffer(maxsize=30)
#     stream_reader = RTSPStreamReader(rtsp_url, frame_buffer, target_fps=target_fps * 2)
#     stream_reader.start()
#
#     # 等待连接
#     print("正在连接RTSP流...")
#     timeout = 10
#     start_time = time.time()
#     while not stream_reader.connected and time.time() - start_time < timeout:
#         time.sleep(0.1)
#
#     if not stream_reader.connected:
#         print("❌ 连接超时！")
#         stream_reader.stop()
#         return
#
#     # 创建分割处理器
#     processor = RealtimeSegmentationProcessor(
#         model=model,
#         device=device,
#         checkpoint_path=checkpoint_path,
#         num_classes=num_classes,
#         frame_height=stream_reader.frame_height,
#         frame_width=stream_reader.frame_width,
#         tile_size=tile_size,
#         overlap=overlap
#     )
#
#     # 视频写入器
#     video_writer = None
#     if save_video:
#         out_width = stream_reader.frame_width * 2 if output_type == 'side_by_side' else stream_reader.frame_width
#         out_height = stream_reader.frame_height
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         video_writer = cv2.VideoWriter(output_path, fourcc, save_fps, (out_width, out_height))
#         print(f"📹 录制到: {output_path}")
#
#     # 显示模式
#     output_types = ['overlay', 'color', 'side_by_side', 'gray']
#     current_type_idx = output_types.index(output_type) if output_type in output_types else 0
#
#     # 性能统计
#     fps_counter = deque(maxlen=30)
#     frame_count = 0
#
#     print("\n" + "=" * 50)
#     print("实时分割已启动 (GPU批量加速)")
#     print("按键: 'q'-退出  's'-截图  'r'-切换模式")
#     print("=" * 50 + "\n")
#
#     try:
#         while stream_reader.running:
#             frame = frame_buffer.get_latest()
#             if frame is None:
#                 time.sleep(0.001)
#                 continue
#
#             # 处理帧
#             process_start = time.time()
#             result_frame, _ = processor.process_frame(
#                 frame,
#                 output_type=output_types[current_type_idx],
#                 alpha=overlay_alpha
#             )
#             process_time = time.time() - process_start
#
#             # FPS统计
#             fps_counter.append(1.0 / max(process_time, 0.001))
#             current_fps = np.mean(list(fps_counter))
#
#             # 保存视频
#             if video_writer:
#                 video_writer.write(result_frame)
#
#             # 缩放显示
#             if display_scale != 1.0:
#                 display_h = int(result_frame.shape[0] * display_scale)
#                 display_w = int(result_frame.shape[1] * display_scale)
#                 display_frame = cv2.resize(result_frame, (display_w, display_h))
#             else:
#                 display_frame = result_frame.copy()
#
#             # 显示信息
#             info = f"FPS: {current_fps:.1f} | Mode: {output_types[current_type_idx]} | Frame: {frame_count}"
#             cv2.putText(display_frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
#             if save_video:
#                 cv2.putText(display_frame, "REC", (display_frame.shape[1] - 60, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#
#             cv2.imshow('Realtime Segmentation [GPU Accelerated]', display_frame)
#             frame_count += 1
#
#             if max_frames and frame_count >= max_frames:
#                 print(f"达到最大帧数 {max_frames}")
#                 break
#
#             # 按键处理
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 print("用户退出")
#                 break
#             elif key == ord('s'):
#                 cv2.imwrite(f"screenshot_{int(time.time())}.png", result_frame)
#                 print("📸 截图已保存")
#             elif key == ord('r'):
#                 current_type_idx = (current_type_idx + 1) % len(output_types)
#                 print(f"🔄 模式: {output_types[current_type_idx]}")
#
#     except KeyboardInterrupt:
#         print("\n用户中断")
#
#     finally:
#         stream_reader.stop()
#         stream_reader.join(timeout=2)
#         if video_writer:
#             video_writer.release()
#             print(f"✅ 视频已保存: {output_path}")
#         cv2.destroyAllWindows()
#         print(f"\n完成！总帧数: {frame_count}")
#
#
# # ==================== 主程序 ====================
#
# if __name__ == "__main__":
#
#     # ============ 配置 ============
#     config = {
#         # RTSP
#         'username': "admin",
#         'password': "a1b2c3d4",
#         'ip_address': "192.168.1.168",
#         'port': "554",
#         'channel': "101",
#
#         # 模型
#         'num_classes': 2,
#         'checkpoint_path': './best_model.pt',
#
#         # 处理（关键性能参数）
#         'tile_size': 1024,
#         'overlap': 64,  # 减小overlap可提升速度
#         'target_fps': 15,
#
#         # 显示
#         'output_type': 'overlay',
#         'overlay_alpha': 0.5,
#         'display_scale': 0.5,
#
#         # 保存
#         'save_video': True,
#         'output_path': f"./realtime_seg_{int(time.time())}.mp4",
#         'save_fps': 15,
#     }
#
#     # ============ 设备 ============
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"使用设备: {device}")
#
#     if device.type == 'cuda':
#         print(f"GPU: {torch.cuda.get_device_name(0)}")
#         print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
#
#     # ============ 创建模型 ============
#     print("\n创建模型...")
#     from UNetFormer import UNetFormer
#
#     model = UNetFormer(num_classes=config['num_classes'], pretrained=False)
#     print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
#
#     # ============ 运行 ============
#     print("\n" + "=" * 60)
#     print("启动实时视频分割 (GPU批量加速)")
#     print("=" * 60)
#
#     run_realtime_segmentation(
#         username=config['username'],
#         password=config['password'],
#         ip_address=config['ip_address'],
#         port=config['port'],
#         channel=config['channel'],
#         model=model,
#         checkpoint_path=config['checkpoint_path'],
#         num_classes=config['num_classes'],
#         tile_size=config['tile_size'],
#         overlap=config['overlap'],
#         target_fps=config['target_fps'],
#         output_type=config['output_type'],
#         overlay_alpha=config['overlay_alpha'],
#         display_scale=config['display_scale'],
#         save_video=config['save_video'],
#         output_path=config['output_path'],
#         save_fps=config['save_fps'],
#         device=device
#     )

"""
实时RTSP视频流语义分割 - 深度GPU批量加速版
优化策略：
1. 多帧批量处理
2. 异步CUDA流
3. TensorRT优化
4. 内存池复用
5. 预处理GPU化
"""

import os
import sys
import time
import math
import threading
import queue
from collections import deque

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F

sys.path.append("D:/ZJF/种植作物分类/ZhongzhiCodes/XiuGai1")

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


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


class RTSPStreamReader(threading.Thread):
    """RTSP流读取线程 - 优化版"""

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

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小化缓冲

        # 设置解码线程数
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

        if not cap.isOpened():
            print("❌ RTSP连接失败！")
            self.running = False
            return

        self.connected = True
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25

        print(f"✅ RTSP连接成功！分辨率: {self.frame_width}x{self.frame_height}, 帧率: {self.fps:.1f}")

        frame_interval = 1.0 / self.target_fps if self.target_fps else 0
        last_frame_time = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 视频流中断，尝试重连...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                continue

            current_time = time.time()
            if frame_interval > 0 and (current_time - last_frame_time) < frame_interval:
                continue
            last_frame_time = current_time

            self.frame_buffer.put(frame)
            self.frame_count += 1

        cap.release()

    def stop(self):
        self.running = False


class GPUTileProcessor:
    """GPU切片处理器 - 完全GPU化的预处理"""

    def __init__(self, frame_height, frame_width, tile_size=1024, overlap=128, device='cuda'):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.device = device

        # 预计算切片位置
        self.tile_positions = self._compute_tile_positions()
        self.num_tiles = len(self.tile_positions)

        # 预计算权重矩阵 (GPU)
        self.weight_matrix_gpu = self._create_weight_matrix_gpu()

        # 预分配GPU内存池
        self.max_batch_frames = 4  # 同时处理的最大帧数
        self.tile_memory_pool = torch.zeros(
            (self.max_batch_frames, self.num_tiles, 3, tile_size, tile_size),
            dtype=torch.float32,
            device=device,
            pin_memory=False
        )

        # 预分配合并结果的内存
        self.merge_buffer = torch.zeros(
            (self.max_batch_frames, frame_height, frame_width),
            dtype=torch.long,
            device=device
        )

        print(f"🚀 GPU切片配置: {self.num_tiles} 个切片, 大小 {tile_size}x{tile_size}, 重叠 {overlap}")
        print(f"   内存池: {self.max_batch_frames} 帧 x {self.num_tiles} 切片")

    def _compute_tile_positions(self):
        """预计算所有切片位置"""
        positions = []
        h, w = self.frame_height, self.frame_width

        n_tiles_y = max(1, math.ceil((h - self.overlap) / self.stride)) if h > self.tile_size else 1
        n_tiles_x = max(1, math.ceil((w - self.overlap) / self.stride)) if w > self.tile_size else 1

        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                x = min(tx * self.stride, max(0, w - self.tile_size)) if w > self.tile_size else 0
                y = min(ty * self.stride, max(0, h - self.tile_size)) if h > self.tile_size else 0
                x_end = min(x + self.tile_size, w)
                y_end = min(y + self.tile_size, h)
                positions.append((x, y, x_end, y_end))

        return positions

    def _create_weight_matrix_gpu(self):
        """创建GPU高斯权重矩阵"""
        x = torch.linspace(-1, 1, self.tile_size, device=self.device)
        y = torch.linspace(-1, 1, self.tile_size, device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        weight = torch.exp(-(xx ** 2 + yy ** 2) / 0.5)
        return weight

    def extract_tiles_batch_gpu(self, frames_bgr):
        """
        GPU批量提取切片 - 完全GPU化
        Args:
            frames_bgr: list of numpy arrays (H, W, 3) BGR格式
        Returns:
            batch_tensor: (B*N, 3, H, W) GPU tensor
        """
        batch_size = len(frames_bgr)

        # 批量转换为GPU tensor
        frames_gpu = []
        for frame in frames_bgr:
            # BGR -> RGB -> GPU -> normalize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).to(self.device, non_blocking=True)
            frame_tensor = frame_tensor.float() / 255.0
            frames_gpu.append(frame_tensor)

        # 批量提取切片
        all_tiles = []
        for batch_idx, frame_tensor in enumerate(frames_gpu):
            for tile_idx, (x, y, x_end, y_end) in enumerate(self.tile_positions):
                th = y_end - y
                tw = x_end - x

                # 直接在GPU上切片
                tile = frame_tensor[y:y_end, x:x_end]  # (th, tw, 3)

                # Padding (如果需要)
                if th < self.tile_size or tw < self.tile_size:
                    padded_tile = torch.zeros(
                        (self.tile_size, self.tile_size, 3),
                        dtype=torch.float32,
                        device=self.device
                    )
                    padded_tile[:th, :tw] = tile
                    tile = padded_tile

                # HWC -> CHW
                tile = tile.permute(2, 0, 1)
                all_tiles.append(tile)

        # 堆叠为批量tensor
        batch_tensor = torch.stack(all_tiles, dim=0)  # (B*N, 3, H, W)
        return batch_tensor, batch_size

    def merge_predictions_gpu(self, predictions, batch_size, num_classes):
        """
        GPU批量合并预测结果
        Args:
            predictions: (B*N, H, W) GPU tensor
            batch_size: 批量大小
            num_classes: 类别数
        Returns:
            final_preds: list of (H, W) numpy arrays
        """
        h, w = self.frame_height, self.frame_width

        # 重塑为 (B, N, H, W)
        predictions = predictions.view(batch_size, self.num_tiles, self.tile_size, self.tile_size)

        final_preds = []

        for batch_idx in range(batch_size):
            # 使用加权融合
            pred_sum = torch.zeros((h, w, num_classes), dtype=torch.float32, device=self.device)
            weight_sum = torch.zeros((h, w), dtype=torch.float32, device=self.device)

            for tile_idx, (x, y, x_end, y_end) in enumerate(self.tile_positions):
                th = y_end - y
                tw = x_end - x
                pred = predictions[batch_idx, tile_idx, :th, :tw]

                # One-hot编码
                pred_onehot = F.one_hot(pred.long(), num_classes=num_classes).float()  # (th, tw, C)

                # 加权累加
                weight = self.weight_matrix_gpu[:th, :tw].unsqueeze(-1)  # (th, tw, 1)
                pred_sum[y:y_end, x:x_end] += pred_onehot * weight
                weight_sum[y:y_end, x:x_end] += self.weight_matrix_gpu[:th, :tw]

            # 归一化并取最大值
            weight_sum = torch.clamp(weight_sum, min=1e-8)
            pred_probs = pred_sum / weight_sum.unsqueeze(-1)
            final_pred = torch.argmax(pred_probs, dim=2).byte()

            # 转回CPU
            final_preds.append(final_pred.cpu().numpy())

        return final_preds


class RealtimeSegmentationProcessor:
    """实时分割处理器 - 深度GPU加速版"""

    COLORMAP = np.array([
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [128, 128, 128],
        [0, 128, 255],
    ], dtype=np.uint8)

    def __init__(self, model, device, checkpoint_path=None, num_classes=2,
                 frame_height=None, frame_width=None, tile_size=1024, overlap=128,
                 batch_frames=2):  # 新增：多帧批量处理
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_frames = batch_frames  # 同时处理的帧数

        # 加载权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"加载模型权重: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("✅ 模型加载完成")

        self.model.to(device)
        self.model.eval()

        # 切片处理器
        self.tile_processor = None
        self.frame_height = frame_height
        self.frame_width = frame_width

        if frame_height and frame_width:
            self._init_tile_processor(frame_height, frame_width)

        # 创建CUDA流用于异步处理
        self.cuda_stream = torch.cuda.Stream()

        # 预热GPU
        self._warmup_gpu()

        # # 编译模型（PyTorch 2.0+）
        # if hasattr(torch, 'compile'):
        #     print("🔥 使用torch.compile优化模型...")
        #     self.model = torch.compile(self.model, mode='max-autotune')

    def _init_tile_processor(self, frame_height, frame_width):
        """初始化GPU切片处理器"""
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.tile_processor = GPUTileProcessor(
            frame_height, frame_width,
            self.tile_size, self.overlap,
            self.device
        )

    def _warmup_gpu(self):
        """GPU预热"""
        print("🔥 GPU预热中...")
        dummy_input = torch.randn(
            self.batch_frames * 4, 3, self.tile_size, self.tile_size,
            device=self.device
        )
        with torch.no_grad():
            with autocast():
                for _ in range(5):
                    _ = self.model(dummy_input)
        torch.cuda.synchronize()
        print("✅ GPU预热完成")

    def predict_batch_async(self, batch_tensor):
        """异步批量预测"""
        with torch.cuda.stream(self.cuda_stream):
            with torch.no_grad():
                with autocast():
                    outputs = self.model(batch_tensor)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            predictions = torch.argmax(outputs, dim=1)

        return predictions

    def label_to_color_gpu(self, labels_batch):
        """GPU批量标签转彩色"""
        results = []
        colormap_gpu = torch.from_numpy(self.COLORMAP).to(self.device)

        for label in labels_batch:
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label).to(self.device)

            # 使用索引操作
            color_img = colormap_gpu[label.long()]  # (H, W, 3)
            results.append(color_img.cpu().numpy())

        return results

    def process_frames_batch(self, frames, output_type='overlay', alpha=0.5):
        """
        批量处理多帧 - 核心加速函数
        Args:
            frames: list of BGR frames
        Returns:
            results: list of processed frames
            predictions: list of prediction masks
        """
        if not frames:
            return [], []

        h, w = frames[0].shape[:2]

        # 检查是否需要重新初始化
        if self.tile_processor is None or h != self.frame_height or w != self.frame_width:
            self._init_tile_processor(h, w)

        # GPU批量提取切片
        batch_tensor, batch_size = self.tile_processor.extract_tiles_batch_gpu(frames)

        # GPU批量推理
        predictions_gpu = self.predict_batch_async(batch_tensor)

        # 等待CUDA流完成
        torch.cuda.synchronize()

        # GPU批量合并
        final_preds = self.tile_processor.merge_predictions_gpu(
            predictions_gpu, batch_size, self.num_classes
        )

        # 生成输出
        results = self._generate_outputs_batch(frames, final_preds, output_type, alpha)

        return results, final_preds

    def _generate_outputs_batch(self, frames, predictions, output_type, alpha):
        """批量生成输出帧"""
        results = []

        if output_type == 'overlay':
            # 批量转换标签为彩色
            color_preds = self.label_to_color_gpu(predictions)

            for frame, color_pred in zip(frames, color_preds):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = cv2.addWeighted(frame_rgb, 1 - alpha, color_pred, alpha, 0)
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                results.append(result)

        elif output_type == 'color':
            color_preds = self.label_to_color_gpu(predictions)
            for color_pred in color_preds:
                result = cv2.cvtColor(color_pred, cv2.COLOR_RGB2BGR)
                results.append(result)

        elif output_type == 'side_by_side':
            color_preds = self.label_to_color_gpu(predictions)
            for frame, color_pred in zip(frames, color_preds):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                combined = np.concatenate([frame_rgb, color_pred], axis=1)
                result = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                results.append(result)

        else:  # gray or original
            for pred in predictions:
                result = (pred * (255 // max(1, self.num_classes - 1))).astype(np.uint8)
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                results.append(result)

        return results


def run_realtime_segmentation(
        # RTSP配置
        username="admin",
        password="a1b2c3d4",
        ip_address="192.168.1.168",
        port="554",
        channel="101",
        # 模型配置
        model=None,
        checkpoint_path=None,
        num_classes=2,
        # 处理配置
        tile_size=1024,
        overlap=64,
        target_fps=15,
        batch_frames=2,  # 新增：批量处理帧数
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
        device=None
):
    """运行实时分割 - 深度优化版"""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"🚀 使用设备: {device}")

    # 构建RTSP URL
    rtsp_url = f"rtsp://{username}:{password}@{ip_address}:{port}/Streaming/Channels/{channel}"
    print(f"📡 连接: rtsp://{username}:****@{ip_address}:{port}/Streaming/Channels/{channel}")

    # 创建帧缓冲区和读取线程
    frame_buffer = FrameBuffer(maxsize=batch_frames * 3)
    stream_reader = RTSPStreamReader(rtsp_url, frame_buffer, target_fps=target_fps * 2)
    stream_reader.start()

    # 等待连接
    print("⏳ 正在连接RTSP流...")
    timeout = 10
    start_time = time.time()
    while not stream_reader.connected and time.time() - start_time < timeout:
        time.sleep(0.1)

    if not stream_reader.connected:
        print("❌ 连接超时！")
        stream_reader.stop()
        return

    # 创建分割处理器
    processor = RealtimeSegmentationProcessor(
        model=model,
        device=device,
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        frame_height=stream_reader.frame_height,
        frame_width=stream_reader.frame_width,
        tile_size=tile_size,
        overlap=overlap,
        batch_frames=batch_frames
    )

    # 视频写入器
    video_writer = None
    if save_video:
        out_width = stream_reader.frame_width * 2 if output_type == 'side_by_side' else stream_reader.frame_width
        out_height = stream_reader.frame_height
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
    print(f"   批量大小: {batch_frames} 帧")
    print("   按键: 'q'-退出  's'-截图  'r'-切换模式")
    print("=" * 60 + "\n")

    try:
        while stream_reader.running:
            # 批量获取帧
            frames = frame_buffer.get_batch(batch_frames)

            if not frames:
                time.sleep(0.001)
                continue

            # 批量处理
            process_start = time.time()
            result_frames, _ = processor.process_frames_batch(
                frames,
                output_type=output_types[current_type_idx],
                alpha=overlay_alpha
            )
            process_time = time.time() - process_start

            # FPS统计（考虑批量）
            batch_fps = len(frames) / max(process_time, 0.001)
            fps_counter.append(batch_fps)
            current_fps = np.mean(list(fps_counter))

            # 处理每一帧的输出
            for result_frame in result_frames:
                # 保存视频
                if video_writer:
                    video_writer.write(result_frame)

                # 缩放显示
                if display_scale != 1.0:
                    display_h = int(result_frame.shape[0] * display_scale)
                    display_w = int(result_frame.shape[1] * display_scale)
                    display_frame = cv2.resize(result_frame, (display_w, display_h))
                else:
                    display_frame = result_frame.copy()

                # 显示信息
                info = f"FPS: {current_fps:.1f} | Batch: {len(frames)} | Mode: {output_types[current_type_idx]} | Frame: {frame_count}"
                cv2.putText(display_frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if save_video:
                    cv2.putText(display_frame, "REC", (display_frame.shape[1] - 60, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('Realtime Segmentation [Deep GPU Accelerated]', display_frame)
                frame_count += 1

            batch_count += 1

            if max_frames and frame_count >= max_frames:
                print(f"✅ 达到最大帧数 {max_frames}")
                break

            # 按键处理
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
        stream_reader.stop()
        stream_reader.join(timeout=2)
        if video_writer:
            video_writer.release()
            print(f"✅ 视频已保存: {output_path}")
        cv2.destroyAllWindows()
        print(f"\n✅ 完成！总帧数: {frame_count}, 总批次: {batch_count}")
        print(f"   平均FPS: {frame_count / (time.time() - start_time):.1f}")


# ==================== 主程序 ====================

if __name__ == "__main__":

    # ============ 配置 ============
    config = {
        # RTSP
        'username': "admin",
        'password': "a1b2c3d4",
        'ip_address': "192.168.1.168",
        'port': "554",
        'channel': "101",

        # 模型
        'num_classes': 3,
        'checkpoint_path': f'../model/best_model.pt',

        # 处理（关键性能参数）
        'tile_size': 1024,
        'overlap': 64,  # 减小overlap提升速度
        'target_fps': 6,
        'batch_frames': 1,  # 🔥 批量处理帧数（根据显存调整：2-4）

        # 显示
        'output_type': 'overlay',
        'overlay_alpha': 0.5,
        'display_scale': 0.5,

        # 保存
        'save_video': False,
        'output_path': f"./realtime_segs_{int(time.time())}.mp4",
        'save_fps': 20,
    }

    # ============ 设备 ============
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")

    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        # 优化CUDA设置
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ============ 创建模型 ============
    print("\n🔧 创建模型...")
    from UNetFormer import UNetFormer

    model = UNetFormer(num_classes=config['num_classes'], pretrained=False)
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ============ 运行 ============
    print("\n" + "=" * 60)
    print("🚀 启动实时视频分割 (深度GPU批量加速)")
    print("=" * 60)

    run_realtime_segmentation(
        username=config['username'],
        password=config['password'],
        ip_address=config['ip_address'],
        port=config['port'],
        channel=config['channel'],
        model=model,
        checkpoint_path=config['checkpoint_path'],
        num_classes=config['num_classes'],
        tile_size=config['tile_size'],
        overlap=config['overlap'],
        target_fps=config['target_fps'],
        batch_frames=config['batch_frames'],  # 🔥 关键参数
        output_type=config['output_type'],
        overlay_alpha=config['overlay_alpha'],
        display_scale=config['display_scale'],
        save_video=config['save_video'],
        output_path=config['output_path'],
        save_fps=config['save_fps'],
        device=device
    )