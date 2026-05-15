import os
import cv2
import numpy as np
import torch

from src.processor.tile_processor import GPUTileProcessor
from model.UNetFormer import UNetFormer



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
            checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=True)
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
            with torch.amp.autocast('cuda'):
                for _ in range(5):
                    _ = self.model(dummy_input)
        torch.cuda.synchronize()
        print("✅ GPU预热完成")

    def predict_batch_async(self, batch_tensor):
        """异步批量预测"""
        with torch.cuda.stream(self.cuda_stream):
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
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
