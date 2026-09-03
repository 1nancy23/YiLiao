# OCR Visual Analysis 0616

这个目录用于独立分析三类目标的 OCR 与匹配链路，不修改主流程文件。

## 目标

- 保存 YOLO 检测框、三类 crop、CLS 旋转后图、DET 文本框、REC 文本行图。
- 输出药瓶、药袋标签、输液袋的 OCR 文本、匹配结果、分类/SIFT 结果与耗时。
- 支持小范围调整 YOLO 阈值来对比检测效果和速度。

## 在板子上运行

在项目主目录执行：

```bash
/home/forlinx/Models/Python2/bin/python3.9 OCR_Visual_Analysis_0616/visual_ocr_analysis.py \
  --images AutoTrigger_Source_Frames_board/*.jpg \
  --output OCR_Visual_Analysis_0616/outputs
```

如果只想看检测和 OCR，不连接数据库：

```bash
/home/forlinx/Models/Python2/bin/python3.9 OCR_Visual_Analysis_0616/visual_ocr_analysis.py \
  --images AutoTrigger_Source_Frames_board/*.jpg \
  --no-db
```

## 输出结构

- `summary.json`: 全部图片的汇总。
- `summary.csv`: 每张图片的计数、耗时和主要结果。
- `case_xxx/yolo_overlay.jpg`: YOLO 检测框可视化。
- `case_xxx/crops/`: 三类目标 crop。
- `case_xxx/ocr_debug/`: OCR 中间图，包括 CLS、DET、REC 阶段。
- `case_xxx/result.json`: 单张图片完整结果。

## 优化观察重点

1. YOLO 框是否漏掉类别 1 药袋标签，或被类别 2 输液袋包含后误筛。
2. 药瓶 crop 是否完整包含瓶身文字区域，CLS 后方向是否正确。
3. 药瓶 DET 文本框高度是否过低，REC 行图是否截断字符上下边缘。
4. 药袋标签 `label_card` 和 `label_name_roi` 是否覆盖姓名区域。
5. 输液袋 `infusion_blue_*` ROI 是否真正覆盖蓝色文字和浓度/容量字段。
6. `summary.csv` 中 `ocr_ms`、`det_ms`、`rec_ms`、`classify_ms` 用于定位耗时瓶颈。

## 不影响主流程的约束

本目录只新增分析脚本和输出目录；不会修改 `run_realtime_detection_yolo_new_3.py`、`native_app.py`、`src/` 下主流程逻辑。
