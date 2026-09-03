# 三类目标 OCR 与匹配优化分析备忘

本文件只记录独立实验目录下的观察方法和候选优化点，不代表已修改主流程。

## 检测效果优先排查

### 1. YOLO 阶段

重点看 `case_xxx/yolo_overlay.jpg` 和 `result.json/detections`：

- 类别 1 药袋标签是否漏检。
- 药袋标签是否被类别 2 输液袋包住但仍保留。
- 药瓶框是否明显偏移、切掉瓶身文字或包含过多背景。
- 输液袋框是否完整覆盖蓝色文字区域、浓度、容量位置。

可用参数对比：

```bash
--bottle-conf 0.30 --bag-conf 0.18 --infusion-conf 0.28
--contain-thres 0.95
```

判断标准不是目标数越多越好，而是：

- 药袋标签类别 1 必须稳定出现。
- 药瓶 crop 必须包含完整标签文字。
- 输液袋 crop 必须给后续蓝色 ROI 提取留足边界。

### 2. CLS 旋转阶段

看 `ocr_debug/predet/*after_cls*.jpg` 与 `classify_angle_pred`：

- 三类 crop 是否都已经转正。
- 药瓶是否存在单个 crop 旋转失败。
- 药袋标签姓名区是否在旋转后仍位于可提取 ROI 内。

若 CLS 输出可靠，优化方向应是改 crop 输入质量，而不是对 OCR 后结果兜底。

### 3. DET 文本框阶段

看 `ocr_debug/detvis/*det_boxes_before_rec.jpg`：

- 药瓶弧面文字框高度是否偏小，是否切掉字的上下边缘。
- 多行文字是否被合并成一个过高/过宽框，导致 REC 行图包含多行。
- 输液袋蓝色区域 DET 是否漏掉浓度/容量。
- 药袋标签姓名 ROI 是否有 DET 框覆盖。

药瓶当前最值得尝试的是增加短边 padding、提高曲面矫正后文本行高度，而不是强行修改 REC。

### 4. REC 行图阶段

看 `ocr_debug/*rec_region*.jpg`：

- 行图是否太扁。
- 字符是否贴边。
- 背景是否过暗、反光过强或对比度不够。

如果行图本身已经缺字，REC 模型无法恢复，需要回到 DET crop/padding 或预处理。

## 速度优化优先级

`summary.csv` 里重点看：

- `yolo_sec`
- `ocr_batch_sec`
- 单个 `result.json` 中的 `shared_det_*`、`shared_rec_*`、`shared_cls_*`

优先优化方向：

1. 让 YOLO 阶段只产生必要的目标，减少后续 OCR 任务数。
2. 保持三类目标共享 CLS/DET/REC batch，不退回逐个目标处理。
3. 减少输液袋候选 ROI 数，但不能牺牲 5%/50ml、10%/100ml 字段覆盖。
4. 对药袋标签只保留姓名区域相关 ROI，避免全标签反复 REC。
5. 只在本实验脚本打开图像保存，主流程保持无中间保存以减少 I/O。

## 推荐实验顺序

1. 先用默认参数跑 3-5 张自动触发保存帧。
2. 对比 `yolo_overlay.jpg`，确认检测框是否是主因。
3. 看 `ocr_debug/detvis`，确认 DET 文本框是否是主因。
4. 看 `rec_region`，确认是否需要加 padding 或调整行拆分。
5. 再做阈值小范围扫描，记录到不同输出目录。

示例：

```bash
/home/forlinx/Models/Python2/bin/python3.9 OCR_Visual_Analysis_0616/visual_ocr_analysis.py \
  --images AutoTrigger_Source_Frames_board/*.jpg \
  --output OCR_Visual_Analysis_0616/outputs_default

/home/forlinx/Models/Python2/bin/python3.9 OCR_Visual_Analysis_0616/visual_ocr_analysis.py \
  --images AutoTrigger_Source_Frames_board/*.jpg \
  --output OCR_Visual_Analysis_0616/outputs_low_bag \
  --bag-conf 0.18 --contain-thres 0.95
```
