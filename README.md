# 药品识别与配药安全核验系统

本项目运行于 RK3588/OK3588 开发板，通过 RTSP 摄像头识别篮子中的药瓶、药袋标签和输液袋，并将识别到的患者、药品和输液信息与医院数据库中的患者批次进行核验。

当前版本使用 Python 原生 OpenCV/Pillow 界面，不使用 Flask 或其他 Web 服务。系统支持手动触发和自动触发，可作为配药流程中的辅助核验工具，但不能替代人工复核和医院现有安全制度。

## 当前主要功能

- RTSP摄像头实时采集和低延迟帧缓存。
- 篮子YOLO实时监控、清晰度判断和连续静止帧自动触发。
- 手动按钮触发与自动触发共用同一套后续识别和匹配流程。
- YOLO同时检测药瓶、药袋标签和输液袋。
- 多个真实目标统一组成CLS、DET和REC批次，不使用单图重复堆叠伪造多目标Batch。
- CLS旋转矫正、DET文本行检测、基于DET概率图的文本方向矫正和REC文字识别。
- 输液袋单个最佳蓝色文字ROI提取和液体、浓度、容量结构化解析。
- 药瓶OCR候选与原始分辨率SIFT模板特征联合决策。
- 药袋标签患者姓名匹配，以及患者批次药品数据库核验。
- 原生全屏结果界面、自动触发进度提示、识别中提示和最终对/错弹窗。
- 药瓶模板动态录入、模板图片更新、本地SIFT缓存更新和数据库特征更新。

## 当前处理流程

```text
RTSP摄像头
    |
    +-- 手动触发
    |
    +-- 自动触发
          篮子YOLO -> 面积/清晰度条件 -> 原始画面连续静止帧
    |
触发后选取清晰原始帧
    |
主YOLO检测（药瓶0、药袋标签1、输液袋2）
    |
裁剪全部真实目标并进行图像增强
    |
共享多目标CLS Batch旋转矫正
    |
共享多目标DET Batch文本检测
    |
根据DET文本分布进行方向矫正并裁剪文本行
    |
共享多文本行REC Batch识别
    |
药瓶：OCR候选 + 本地SIFT模板匹配
药袋：患者姓名数据库匹配
输液袋：液体/浓度/容量结构化
    |
medicine_db2患者批次核验
    |
原生界面显示明细并弹出最终对/错结果
```

混合OCR批处理失败时，默认不启用逐目标兜底流程，避免静默改变计算路径。

## 当前RKNN模型与NPU分配

| 阶段 | 当前模型 | 输入/Batch | NPU核心 | 颜色格式 |
| --- | --- | --- | --- | --- |
| 主目标检测 | `model_yolo_0615.rknn` | 640x640，Batch 1 | Core 0 | RGB |
| 篮子检测 | `model_bask_0609_n.rknn` | 640x640，Batch 1 | Core 1 | RGB |
| 方向CLS | `model_cls_bs32.rknn` | 224x224，Batch 32 | Core 1 | 按模型预处理 |
| OCR DET | `model_det_bs16.rknn` | 448x448，Batch 16 | Core 1、Core 2 | BGR |
| OCR REC | `model_ocr_bs16.rknn` | 320x48，Batch 16 | Core 0、Core 1、Core 2 | BGR |

DET和REC采用多个RKNN上下文与线程并行执行。不同核心上的结果会按照原始任务顺序重新组装，补零项不会进入后处理结果。

根目录还保留若干历史或测试模型，但主流程只使用表格列出的模型。

## 药瓶与数据库数据来源

药瓶类别模板来自：

```text
src/identification/feat_data/<药品名称>/
```

模板SIFT特征缓存为：

```text
single_image_feature_cache.pkl
```

程序启动时检查模板目录文件名、大小和修改时间签名。模板未变化时直接加载缓存；模板变化时重新按原始分辨率生成SIFT特征。

药瓶候选名称和SIFT特征从本地模板缓存读取。患者姓名、患者批次和该批次应包含的药品仍从MySQL数据库读取。当前数据库配置为 `medicine_db2`，活动表结构使用：

- `batches.patient_name`：患者姓名。
- `batches.medicines_json`：患者批次应包含的药品JSON。
- `drugs.medicine_name`及SIFT字段：用于模板录入时同步数据库特征。

## 项目结构

```text
.
├── main.py                              # 原生界面入口
├── native_app.py                        # 界面、运行状态和模型初始化
├── run_realtime_detection_yolo_new_3.py # 自动/手动触发与完整识别匹配流程
├── config.yaml                          # RTSP、模型、触发、显示和数据库配置
├── register_bottle_template.py          # 摄像头药瓶模板录入工具
├── auto_headless_app.py                 # 无主界面的自动流程入口
├── Test_OCR.py                          # 输液袋OCR后处理
├── Test_OCR_tight_lines.py              # 药瓶/药袋紧凑文本行后处理
├── single_image_feature_cache.pkl       # 本地药瓶SIFT特征缓存
├── src
│   ├── segmentation                     # 主YOLO与篮子自动触发检测
│   ├── identification                   # RKNN OCR、SIFT和数据库匹配
│   ├── processor                        # 图像裁剪
│   ├── stream                           # RTSP读取和帧缓冲
│   └── utils                            # 图像矫正与通用工具
├── NPU_Load_Test_0627                   # NPU核心负载与阶段耗时测试
├── OCR_Visual_Analysis_0616             # 独立OCR可视化分析工具
└── Optimization_Report_0720.md          # 0720版本性能优化报告
```

历史备份和独立测试脚本不参与 `main.py` 的活动运行链路。

## 运行环境

板子验证环境：

- RK3588/OK3588。
- Python 3.9。
- RKNN Toolkit Lite2 2.3.2。
- OpenCV及`opencv-contrib`（SIFT）。
- NumPy、Pillow、PyYAML、PyMySQL、fuzzywuzzy。
- 建议安装`python-Levenshtein`以加快模糊字符串匹配。
- MySQL数据库可从板子访问。

主流程已经完全使用RKNN模型完成YOLO、CLS、DET和REC推理，不依赖GPU版PaddleOCR或PyTorch完成在线识别。

## 配置

修改 `config.yaml` 中的实际设备参数。不要把生产环境密码提交到公开仓库。

```yaml
runtime:
  trigger_mode: manual
  basket_area_threshold: 0.40
  basket_sharpness_threshold: 55
  basket_stable_frames: 12
  basket_capture_timeout: 10.0
  basket_resume_delay: 1.0

RTSP:
  username: "<camera-user>"
  password: "<camera-password>"
  ip_address: "<camera-ip>"
  port: "554"
  channel: "201"

db_config:
  host: "<mysql-host>"
  user: "<mysql-user>"
  password: "<mysql-password>"
  database: medicine_db2
  charset: utf8
```

`crop_region`用于限制识别画面范围。修改摄像头分辨率后，应同步检查该区域是否仍覆盖完整篮子。

## 启动方式

在板子项目根目录运行：

```bash
/home/forlinx/Models/Python2/bin/python3.9 main.py
```

直接启动自动触发、结果专用、全屏界面：

```bash
/home/forlinx/Models/Python2/bin/python3.9 native_app.py --auto --result-only --fullscreen
```

使用本地视频测试：

```bash
/home/forlinx/Models/Python2/bin/python3.9 native_app.py --video /path/to/test.mp4
```

无界面配置检查：

```bash
/home/forlinx/Models/Python2/bin/python3.9 auto_headless_app.py --check
```

仓库包含以下启动脚本：

- `start_yiliao_native.sh`
- `start_yiliao_auto_result.sh`
- `start_yiliao_auto_headless.sh`

部署到新路径时，先将脚本中的 `PROJECT_DIR` 修改为实际项目根目录。开机自启动桌面项应调用 `start_yiliao_auto_result.sh`，对应命令为 `native_app.py --auto --result-only --fullscreen`。

## 药瓶模板录入

运行模板录入前应先停止主识别程序，避免摄像头和NPU资源冲突。篮子可以保留，但画面中只能存在一个待录入药瓶目标。

```bash
/home/forlinx/Models/Python2/bin/python3.9 register_bottle_template.py \
  --medicine-name "注射用更昔洛韦"
```

默认流程会在检测到单个药瓶后采集8秒，选择清晰且互相差异较大的6张药瓶裁剪图，更新模板目录、SIFT缓存和数据库特征。使用以下命令查看完整参数：

```bash
/home/forlinx/Models/Python2/bin/python3.9 register_bottle_template.py --help
```

## 性能与内存优化

当前版本已包含：

- CPU预处理、DET和REC常驻线程池。
- DET双核心和REC三核心并行。
- 灰度方向分析及DET/REC冗余复制消除。
- 药瓶弧面矫正映射缓存。
- 有界SIFT matcher缓存和模糊匹配LRU缓存。
- 单次患者批次数据库查询。
- 原生界面按需重绘和结果图标缓存。
- 有界事件队列及高频状态合并。

详细改动和等价性验证见 `Optimization_Report_0720.md`。

## 可选诊断环境变量

默认关闭运行日志、详细计时和中间结果保存。排查问题时可临时设置：

```bash
export YILIAO_RUNTIME_LOGS=1
export YILIAO_VERBOSE_RUNTIME=1
export YILIAO_COLLECT_TIMING=1
```

文本方向诊断图仅在设置 `YILIAO_TEXT_ANGLE_VIS_DIR` 时保存。正常运行不要开启中间图保存，以免增加I/O和内存压力。

## 常见问题

### 模型初始化后程序退出

检查RKNN模型文件是否存在、RKNN Runtime版本是否匹配，以及是否还有另一套主流程占用摄像头或NPU。静态形状RKNN模型出现`Query dynamic range failed`警告时，只要后续没有初始化错误通常可以忽略。

### 自动触发一直等待

检查篮子面积、清晰度、连续静止帧数和摄像头曝光是否稳定。静止判断基于原始画面，不依赖篮子检测框的抖动。

### 药袋标签或输液袋未识别

先检查主YOLO是否检测到类别1和类别2，再检查CLS方向、DET文本框和输液袋蓝色ROI。不要把绘制检测框后的可视化图传入OCR流程。

### 药瓶分类错误

检查OCR候选、SIFT Top结果、模板图清晰度和模板缓存签名。模板更新后应重启主程序或调用特征重载，确保运行内存使用最新模板。

### 患者批次不存在

确认板子可以访问MySQL，`medicine_db2.batches`中存在对应`patient_name`，且`medicines_json`格式正确。药瓶模板存在不代表患者批次已存在，两者数据来源不同。

## 安全声明

本系统只提供辅助识别、配药核验和风险提示。识别结果会受到模型能力、标签角度、反光、遮挡、摄像头曝光、模板质量和数据库完整性的影响。任何临床使用都必须保留人工核对和专业人员最终确认。
