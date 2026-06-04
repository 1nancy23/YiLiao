import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def save_image(image, output_path):
    if output_path is None:
        return False
    output = Path(str(output_path))
    if output.parent != Path('.'):
        output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix or '.jpg'
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        print(f"??????: {output_path}")
        return False
    encoded.tofile(str(output))
    print(f"?????: {output_path}")
    return True


def normalize_image(image):
    if image is None:
        return None
    img = image.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def stitch_images_vertically_with_regions(images: Sequence[np.ndarray], separator_height=30, separator_color=(255, 255, 255)):
    normalized = []
    for image in images:
        img = normalize_image(image)
        if img is None:
            return None, []
        normalized.append(img)

    if not normalized:
        return None, []

    max_width = max(img.shape[1] for img in normalized)
    parts = []
    regions = []
    current_y = 0

    for index, img in enumerate(normalized):
        height, width = img.shape[:2]
        scale = 1.0
        if width != max_width:
            new_height = int(height * max_width / width)
            scale = max_width / width
            img = cv2.resize(img, (max_width, new_height), interpolation=cv2.INTER_AREA)
            height = new_height

        regions.append({
            'index': index,
            'image': images[index],
            'y_start': current_y,
            'y_end': current_y + height,
            'height': height,
            'width': max_width,
            'scale': scale,
        })
        parts.append(img)
        current_y += height

        if separator_height > 0 and index < len(normalized) - 1:
            separator = np.full((separator_height, max_width, 3), separator_color, dtype=np.uint8)
            parts.append(separator)
            current_y += separator_height

    return cv2.vconcat(parts), regions

import cv2
import numpy as np
from typing import Sequence, Tuple, List, Dict, Optional
import math


def normalize_image(image: np.ndarray) -> Optional[np.ndarray]:
    if image is None:
        return None
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image

import math
from typing import Sequence, Tuple, List, Dict, Optional

def stitch_images_with_optimal_ratio(
    images: Sequence[np.ndarray],
    target_ratio: float = 1.0,
    separator_size: int = 20,
    separator_color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[Optional[np.ndarray], List[Dict]]:
    """
    拼接图像，使最终长宽比接近 target_ratio，不改变源图像尺寸。

    :param images: 图像列表
    :param target_ratio: 目标长宽比 (高度/宽度)，1.0 为正方形
    :param separator_size: 图像间隔像素
    :param separator_color: 背景/间隔颜色
    :return: (拼接图, 区域信息列表)
    """
    normalized = [normalize_image(img) for img in images]
    if any(img is None for img in normalized):
        return None, []

    n = len(normalized)
    heights = [img.shape[0] for img in normalized]
    widths  = [img.shape[1] for img in normalized]

    # ----------------------------------------------------------------
    # 计算最优列数：枚举所有可能列数，选使最终比例最接近 target_ratio 的
    # ----------------------------------------------------------------
    best_cols = 1
    best_diff = float('inf')

    for n_cols in range(1, n + 1):
        n_rows = math.ceil(n / n_cols)

        # 每行宽度 = 该行所有图宽之和 + 间隔
        row_widths = []
        for r in range(n_rows):
            start, end = r * n_cols, min(r * n_cols + n_cols, n)
            row_w = sum(widths[start:end]) + separator_size * (end - start - 1)
            row_widths.append(row_w)

        # 每列高度 = 该列所有图高之和 + 间隔
        col_heights = []
        for c in range(n_cols):
            idxs = [c + r * n_cols for r in range(n_rows) if c + r * n_cols < n]
            col_h = sum(heights[i] for i in idxs) + separator_size * (len(idxs) - 1)
            col_heights.append(col_h)

        canvas_w = max(row_widths)
        canvas_h = max(col_heights)

        ratio = canvas_h / canvas_w
        diff = abs(ratio - target_ratio)
        if diff < best_diff:
            best_diff = diff
            best_cols = n_cols

    n_cols = best_cols
    n_rows = math.ceil(n / n_cols)

    # ----------------------------------------------------------------
    # 计算画布尺寸（以各行最大宽、各列最大高为准）
    # ----------------------------------------------------------------
    row_max_heights = []
    for r in range(n_rows):
        idxs = range(r * n_cols, min(r * n_cols + n_cols, n))
        row_max_heights.append(max(heights[i] for i in idxs))

    col_max_widths = []
    for c in range(n_cols):
        idxs = [c + r * n_cols for r in range(n_rows) if c + r * n_cols < n]
        col_max_widths.append(max(widths[i] for i in idxs))

    canvas_w = sum(col_max_widths) + separator_size * (n_cols - 1)
    canvas_h = sum(row_max_heights) + separator_size * (n_rows - 1)

    canvas = np.full((canvas_h, canvas_w, 3), separator_color, dtype=np.uint8)

    # ----------------------------------------------------------------
    # 放置图像（不缩放，左上角对齐到单元格）
    # ----------------------------------------------------------------
    regions = []
    col_offsets = [sum(col_max_widths[:c]) + separator_size * c for c in range(n_cols)]
    row_offsets = [sum(row_max_heights[:r]) + separator_size * r for r in range(n_rows)]

    for idx, img in enumerate(normalized):
        r, c = divmod(idx, n_cols)
        x = col_offsets[c]
        y = row_offsets[r]
        h, w = img.shape[:2]

        canvas[y:y+h, x:x+w] = img

        regions.append({
            'index': idx,
            'image': images[idx],
            'x_start': x,
            'y_start': y,
            'x_end': x + w,
            'y_end': y + h,
            'width': w,
            'height': h,
            'row': r,
            'col': c,
            'scale': 1.0,  # 源图像未缩放
        })

    print(f"拼接布局: {n_rows}行x{n_cols}列, 画布: {canvas_w}x{canvas_h}, "
          f"长宽比: {canvas_h/canvas_w:.2f} (目标: {target_ratio:.2f})")

    return canvas, regions


def merge_text_blocks(text_blocks, line_gap_threshold=20):
    if not text_blocks:
        return []

    text_blocks = sorted(text_blocks, key=lambda x: x['center_y'])
    grouped_lines = []
    current_line = [text_blocks[0]]
    prev_y = text_blocks[0]['center_y']

    for block in text_blocks[1:]:
        if abs(block['center_y'] - prev_y) > line_gap_threshold:
            grouped_lines.append(current_line)
            current_line = [block]
        else:
            current_line.append(block)
        prev_y = block['center_y']
    grouped_lines.append(current_line)

    lines = []
    for line_blocks in grouped_lines:
        line_blocks = sorted(line_blocks, key=lambda x: x['center_x'])
        lines.append(''.join(block['text'] for block in line_blocks))
    return lines


def split_blocks_by_regions(text_blocks, regions, line_gap_threshold=20):
    grouped_blocks = [[] for _ in regions]
    grouped_confidences = [[] for _ in regions]

    for block in text_blocks:
        center_x = block['center_x']
        center_y = block['center_y']
        for region in regions:
            if (
                region['x_start'] <= center_x < region['x_end']
                and region['y_start'] <= center_y < region['y_end']
            ):
                local_block = block.copy()
                local_block['center_x'] = center_x - region['x_start']
                local_block['center_y'] = center_y - region['y_start']
                bbox = block.get('bbox')
                if bbox is not None:
                    local_bbox = bbox.copy().astype(np.float32)
                    local_bbox[:, 0] -= region['x_start']
                    local_bbox[:, 1] -= region['y_start']
                    local_block['bbox'] = local_bbox
                grouped_blocks[region['index']].append(local_block)
                grouped_confidences[region['index']].append(block.get('confidence', 0))
                break

    results = []
    for region, blocks, confidences in zip(regions, grouped_blocks, grouped_confidences):
        lines = merge_text_blocks(blocks, line_gap_threshold=line_gap_threshold)
        results.append({
            'image': region['image'],
            'image_index': region['index'],
            'ocr_text': ''.join(lines),
            'confidence': sum(confidences) / len(confidences) if confidences else 0,
            'region': region,
        })
    return results


def draw_stitched_ocr_result(stitched_img, text_blocks, regions):
    vis_img = stitched_img.copy()
    for region in regions:
        x_start = int(region['x_start'])
        y_start = int(region['y_start'])
        x_end = int(region['x_end'])
        y_end = int(region['y_end'])
        cv2.rectangle(vis_img, (x_start, y_start), (x_end - 1, y_end - 1), (255, 160, 0), 2)
        cv2.putText(vis_img, f"img_{region['index']}", (x_start + 10, max(25, y_start + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 160, 0), 2)

    for block in text_blocks:
        bbox = block.get('bbox')
        if bbox is not None:
            cv2.polylines(vis_img, [bbox.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(vis_img, block.get('text', ''), (int(block['center_x']), int(block['center_y'])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis_img


def recognize_stitched_images_as_batch(
    recognizer,
    images: Sequence[np.ndarray],
    line_gap_threshold=20,
    separator_height=20,
    stitched_output_path=None,
    ocr_output_path=None,
):
    start = time.time()
    stitched_img, regions = stitch_images_with_optimal_ratio(
        images,
        separator_size=separator_height,
    )
    mid_time = time.time()
    print(f"拼接耗时:{mid_time - start:.3f}s")
    print("使用了该函数")
    if stitched_img is None:
        return []
    print("使用了该函数")
    # save_image(stitched_img, stitched_output_path)
    start_2 = time.time()
    print("开始OCR",stitched_img.shape)
    _, text_blocks = recognizer.recognize_with_blocks(stitched_img, line_gap_threshold=line_gap_threshold)
    mid_time = time.time()
    print(f"一次耗时:{mid_time - start_2:.3f}s")
    if ocr_output_path is not None:
        vis_img = draw_stitched_ocr_result(stitched_img, text_blocks, regions)
        save_image(vis_img, ocr_output_path)

    results = split_blocks_by_regions(text_blocks, regions, line_gap_threshold=line_gap_threshold)
    print(f"一次OCR耗时:{time.time() - start:.3f}s")
    for item in results:
        print(f"[{item['image_index']}] {item['ocr_text']}")
    return results


# import time
# from pathlib import Path
# from typing import Sequence

# import cv2
# import numpy as np


# def save_image(image, output_path):
#     if output_path is None:
#         return False

#     output = Path(str(output_path))
#     if output.parent != Path('.'):
#         output.parent.mkdir(parents=True, exist_ok=True)

#     suffix = output.suffix or '.jpg'
#     success, encoded = cv2.imencode(suffix, image)

#     if not success:
#         print(f"保存失败: {output_path}")
#         return False

#     encoded.tofile(str(output))
#     print(f"已保存: {output_path}")
#     return True


# def normalize_image(image):
#     if image is None:
#         return None

#     img = image.copy()

#     if len(img.shape) == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     elif img.shape[2] == 4:
#         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

#     return img


# def _calc_grid_layout(image_shapes, cols, separator_size=20):
#     """
#     计算指定列数下的网格布局尺寸。

#     image_shapes: [(h, w), ...]
#     cols: 列数
#     """
#     n = len(image_shapes)
#     rows = int(np.ceil(n / cols))

#     col_widths = [0] * cols
#     row_heights = [0] * rows

#     for index, (h, w) in enumerate(image_shapes):
#         r = index // cols
#         c = index % cols

#         col_widths[c] = max(col_widths[c], w)
#         row_heights[r] = max(row_heights[r], h)

#     total_width = sum(col_widths) + separator_size * (cols - 1)
#     total_height = sum(row_heights) + separator_size * (rows - 1)

#     return {
#         "rows": rows,
#         "cols": cols,
#         "col_widths": col_widths,
#         "row_heights": row_heights,
#         "total_width": total_width,
#         "total_height": total_height,
#     }


# def _choose_best_grid_layout(image_shapes, separator_size=20):
#     """
#     自动选择最接近正方形的网格布局。

#     目标：
#     - 拼接后的整体图像宽高尽量接近；
#     - 避免单纯垂直长图导致 OCR 输入过高；
#     - 避免单纯水平长图导致 OCR 输入过宽。
#     """
#     n = len(image_shapes)

#     if n == 1:
#         return _calc_grid_layout(image_shapes, cols=1, separator_size=separator_size)

#     best_layout = None
#     best_score = None

#     for cols in range(1, n + 1):
#         layout = _calc_grid_layout(
#             image_shapes=image_shapes,
#             cols=cols,
#             separator_size=separator_size,
#         )

#         w = layout["total_width"]
#         h = layout["total_height"]

#         if w <= 0 or h <= 0:
#             continue

#         # 越接近 1 越好
#         aspect = w / h

#         # 使用 log(aspect) 可以让宽图和高图惩罚更对称
#         aspect_score = abs(np.log(aspect))

#         # 可选：轻微惩罚过大的画布面积，避免极端情况下产生过多空白
#         area = w * h
#         score = aspect_score + area * 1e-12

#         if best_score is None or score < best_score:
#             best_score = score
#             best_layout = layout

#     return best_layout


# def stitch_images_grid_with_regions(
#     images: Sequence[np.ndarray],
#     separator_size=30,
#     separator_color=(255, 255, 255),
#     align="center",
# ):
#     """
#     将多张图片拼接为一个尽量宽高接近的网格图。

#     返回：
#     - stitched_img: 拼接后的大图
#     - regions: 每张原图在大图中的区域信息

#     region 字段说明：
#     {
#         'index': 原图索引,
#         'image': 原始输入图,
#         'x_start': 在拼接图中的左上角 x,
#         'y_start': 在拼接图中的左上角 y,
#         'x_end': 在拼接图中的右下角 x,
#         'y_end': 在拼接图中的右下角 y,
#         'height': 当前图高度,
#         'width': 当前图宽度,
#         'scale': 缩放比例，这里没有缩放，所以为 1.0,
#         'row': 所在行,
#         'col': 所在列,
#     }
#     """
#     normalized = []

#     for image in images:
#         img = normalize_image(image)

#         if img is None:
#             return None, []

#         normalized.append(img)

#     if not normalized:
#         return None, []

#     image_shapes = [(img.shape[0], img.shape[1]) for img in normalized]

#     layout = _choose_best_grid_layout(
#         image_shapes=image_shapes,
#         separator_size=separator_size,
#     )

#     rows = layout["rows"]
#     cols = layout["cols"]
#     col_widths = layout["col_widths"]
#     row_heights = layout["row_heights"]
#     total_width = layout["total_width"]
#     total_height = layout["total_height"]

#     stitched_img = np.full(
#         (total_height, total_width, 3),
#         separator_color,
#         dtype=np.uint8,
#     )

#     regions = []

#     # 每一列的 x 起点
#     col_x_starts = []
#     current_x = 0
#     for c in range(cols):
#         col_x_starts.append(current_x)
#         current_x += col_widths[c] + separator_size

#     # 每一行的 y 起点
#     row_y_starts = []
#     current_y = 0
#     for r in range(rows):
#         row_y_starts.append(current_y)
#         current_y += row_heights[r] + separator_size

#     for index, img in enumerate(normalized):
#         h, w = img.shape[:2]

#         row = index // cols
#         col = index % cols

#         cell_x = col_x_starts[col]
#         cell_y = row_y_starts[row]

#         cell_w = col_widths[col]
#         cell_h = row_heights[row]

#         if align == "center":
#             x_start = cell_x + (cell_w - w) // 2
#             y_start = cell_y + (cell_h - h) // 2
#         else:
#             x_start = cell_x
#             y_start = cell_y

#         x_end = x_start + w
#         y_end = y_start + h

#         stitched_img[y_start:y_end, x_start:x_end] = img

#         regions.append({
#             "index": index,
#             "image": images[index],
#             "x_start": x_start,
#             "y_start": y_start,
#             "x_end": x_end,
#             "y_end": y_end,
#             "height": h,
#             "width": w,
#             "scale": 1.0,
#             "row": row,
#             "col": col,
#         })

#     return stitched_img, regions


# def merge_text_blocks(text_blocks, line_gap_threshold=20):
#     if not text_blocks:
#         return []

#     text_blocks = sorted(text_blocks, key=lambda x: x["center_y"])

#     grouped_lines = []
#     current_line = [text_blocks[0]]
#     prev_y = text_blocks[0]["center_y"]

#     for block in text_blocks[1:]:
#         if abs(block["center_y"] - prev_y) > line_gap_threshold:
#             grouped_lines.append(current_line)
#             current_line = [block]
#         else:
#             current_line.append(block)

#         prev_y = block["center_y"]

#     grouped_lines.append(current_line)

#     lines = []

#     for line_blocks in grouped_lines:
#         line_blocks = sorted(line_blocks, key=lambda x: x["center_x"])
#         lines.append("".join(block["text"] for block in line_blocks))

#     return lines


# def split_blocks_by_regions(text_blocks, regions, line_gap_threshold=20):
#     """
#     将整张拼接图上的 OCR block 按 region 拆回原图。

#     修改点：
#     - 原来只判断 y 范围；
#     - 现在同时判断 x/y 范围；
#     - 将 center_x / center_y 转换成当前小图局部坐标；
#     - 如果 bbox 存在，也转换成当前小图局部坐标。
#     """
#     grouped_blocks = [[] for _ in regions]
#     grouped_confidences = [[] for _ in regions]

#     for block in text_blocks:
#         center_x = block["center_x"]
#         center_y = block["center_y"]

#         for region in regions:
#             if (
#                 region["x_start"] <= center_x < region["x_end"]
#                 and region["y_start"] <= center_y < region["y_end"]
#             ):
#                 local_block = block.copy()

#                 local_block["center_x"] = center_x - region["x_start"]
#                 local_block["center_y"] = center_y - region["y_start"]

#                 bbox = block.get("bbox")
#                 if bbox is not None:
#                     local_bbox = bbox.copy().astype(np.float32)
#                     local_bbox[:, 0] -= region["x_start"]
#                     local_bbox[:, 1] -= region["y_start"]
#                     local_block["bbox"] = local_bbox

#                 grouped_blocks[region["index"]].append(local_block)
#                 grouped_confidences[region["index"]].append(block.get("confidence", 0))

#                 break

#     results = []

#     for region, blocks, confidences in zip(regions, grouped_blocks, grouped_confidences):
#         lines = merge_text_blocks(
#             blocks,
#             line_gap_threshold=line_gap_threshold,
#         )

#         results.append({
#             "image": region["image"],
#             "image_index": region["index"],
#             "ocr_text": "".join(lines),
#             "confidence": sum(confidences) / len(confidences) if confidences else 0,
#             "region": region,
#         })

#     return results


# def draw_stitched_ocr_result(stitched_img, text_blocks, regions):
#     vis_img = stitched_img.copy()

#     for region in regions:
#         x_start = int(region["x_start"])
#         y_start = int(region["y_start"])
#         x_end = int(region["x_end"])
#         y_end = int(region["y_end"])

#         cv2.rectangle(
#             vis_img,
#             (x_start, y_start),
#             (x_end - 1, y_end - 1),
#             (255, 160, 0),
#             2,
#         )

#         cv2.putText(
#             vis_img,
#             f"img_{region['index']}",
#             (x_start + 10, max(y_start + 25, 25)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (255, 160, 0),
#             2,
#         )

#     for block in text_blocks:
#         bbox = block.get("bbox")

#         if bbox is not None:
#             cv2.polylines(
#                 vis_img,
#                 [bbox.astype(np.int32)],
#                 True,
#                 (0, 255, 0),
#                 2,
#             )

#         cv2.putText(
#             vis_img,
#             block.get("text", ""),
#             (int(block["center_x"]), int(block["center_y"])),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (0, 0, 255),
#             2,
#         )

#     return vis_img


# def recognize_stitched_images_as_batch(
#     recognizer,
#     images: Sequence[np.ndarray],
#     line_gap_threshold=20,
#     separator_size=20,
#     stitched_output_path=None,
#     ocr_output_path=None,
# ):
#     """
#     多图批量 OCR。

#     修改点：
#     - 不再垂直拼接；
#     - 改成自动选择接近正方形的网格拼接；
#     - OCR 仍然只调用一次；
#     - OCR 结果再根据每张图在拼接图中的二维区域拆分回去。
#     """
#     start = time.time()

#     stitched_img, regions = stitch_images_grid_with_regions(
#         images,
#         separator_size=separator_size,
#     )

#     mid_time = time.time()
#     print(f"拼接耗时: {mid_time - start:.3f}s")

#     if stitched_img is None:
#         return []

#     if stitched_output_path is not None:
#         save_image(stitched_img, stitched_output_path)

#     start_ocr = time.time()
#     print("开始OCR", stitched_img.shape)

#     _, text_blocks = recognizer.recognize_with_blocks(
#         stitched_img,
#         line_gap_threshold=line_gap_threshold,
#     )

#     print(f"一次OCR耗时: {time.time() - start_ocr:.3f}s")

#     if ocr_output_path is not None:
#         vis_img = draw_stitched_ocr_result(
#             stitched_img,
#             text_blocks,
#             regions,
#         )
#         save_image(vis_img, ocr_output_path)

#     results = split_blocks_by_regions(
#         text_blocks,
#         regions,
#         line_gap_threshold=line_gap_threshold,
#     )

#     print(f"总耗时: {time.time() - start:.3f}s")

#     for item in results:
#         print(f"[{item['image_index']}] {item['ocr_text']}")

#     return results

