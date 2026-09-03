import math
import torch.nn.functional as F
import cv2
import torch


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