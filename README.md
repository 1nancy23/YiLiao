# 药品注射配对安全检查系统

本项目面向临床配药场景，用于护士将药品注射到药袋前，对“药品”和“药袋/患者信息”进行自动识别与配对核验，为人工操作增加一道安全检查，降低药品注射错误的风险。

系统通过摄像头或 RTSP 视频流采集现场画面，使用 YOLO/RKNN 模型定位画面中的药品、药袋等目标区域，再将裁剪后的局部图像交给 OCR 模型识别文字信息。识别结果会与数据库中的药品、患者和批次信息进行模糊匹配，同时结合药品图像特征匹配结果进行二次确认，最终输出药品与药袋是否匹配的判断结果。

> 说明：本项目用于辅助核验和风险提示，不能替代护士、药师或医院既有的人工核对流程。

## 核心功能

- 实时视频流接入：支持从 RTSP 摄像头读取画面。
- 目标检测定位：使用 YOLO/RKNN 模型定位药品、药袋等关键区域。
- 图像裁剪与预处理：对检测区域进行裁剪、旋转校正、增强等处理，提高 OCR 和特征匹配稳定性。
- OCR 文字识别：使用 PaddleOCR 识别药品标签、药袋标签中的中文文字。
- 文本模糊匹配：使用 `fuzzywuzzy` 将 OCR 结果与数据库中的药品名、患者名进行相似度匹配。
- 图像特征匹配：使用 ORB/SIFT 类特征对药品外观进行辅助匹配，降低仅依赖 OCR 带来的误判风险。
- 数据库管理：使用 MySQL 存储药品、患者、批次和药品特征数据。

## 工作流程

```text
摄像头/视频流
    ↓
读取实时帧
    ↓
YOLO/RKNN 定位药品和药袋区域
    ↓
裁剪目标区域并进行图像预处理
    ↓
PaddleOCR 识别标签文字
    ↓
数据库文本模糊匹配
    ↓
药品图像特征匹配
    ↓
综合判断药品与药袋是否匹配
```

可以把整套流程理解为两条证据链：

- 文字证据链：OCR 识别标签文字，再和数据库中的药品名、患者名做模糊匹配。
- 图像证据链：提取药瓶/药盒图像特征，再和数据库中保存的药品特征模板做相似度匹配。

两条证据链共同参与判断，可以减少 OCR 识别错误、标签遮挡、光照变化等情况造成的误判。

## 项目结构

```text
.
├── main.py                              # 主入口：加载配置、初始化数据库/OCR/匹配器并启动实时检测
├── config.yaml                          # RTSP、模型、显示、保存、数据库等配置
├── Data_base.sql                        # MySQL 数据表初始化和调试 SQL
├── requirements.txt                     # 依赖安装参考
├── model_yolo_0510.rknn                 # YOLO/RKNN 目标检测模型
├── model_cls.rknn                       # 分类模型文件
├── run_realtime_detection_yolo_new_3.py # 当前主流程使用的实时检测脚本
├── src
│   ├── segmentation                     # 目标检测、分块推理、区域定位
│   ├── identification                   # OCR、药品匹配、图像特征识别
│   ├── processor                        # 图像裁剪与处理
│   ├── stream                           # 视频流读取与帧缓存
│   └── utils                            # 初始化、图像工具、保存/显示工具
├── data
│   ├── data.yaml                        # YOLO 数据集配置
│   └── data_base                        # 药品/药袋样本数据
├── runs                                 # YOLO 训练或推理输出
├── Temp_results                         # 临时裁剪结果
└── Temp_results_mask                    # 临时 mask/处理结果
```

## 环境要求

推荐环境：

- Python 3.9
- MySQL
- OpenCV
- PaddleOCR / PaddlePaddle
- RKNNLite
- PyMySQL
- fuzzywuzzy
- python-Levenshtein

如果使用 GPU 版本 PaddlePaddle，需要根据本机 CUDA 版本安装对应版本。当前 `requirements.txt` 中保留了项目开发时的安装命令，可作为环境复现参考。

## 安装依赖

建议先创建虚拟环境：

```bash
conda create -n yiliao python=3.9
conda activate yiliao
```

安装主要依赖：

```bash
pip install paddlepaddle==2.6.2
pip install paddleocr==2.7.0.3
pip install opencv-python-headless==4.6.0.66 opencv-contrib-python==4.6.0.66
pip install numpy==1.23.5
pip install pymysql fuzzywuzzy python-Levenshtein
pip install pynput einops timm imgaug
```

如果需要安装 PyTorch，可根据显卡和 CUDA 版本选择合适命令。例如 CUDA 12.1：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 数据库配置

项目使用 MySQL 保存药品、患者、批次和药品特征信息。先创建数据库：

```sql
CREATE DATABASE medicine_db DEFAULT CHARACTER SET utf8mb4;
```

然后执行 `Data_base.sql` 中的建表语句，主要数据表包括：

- `drugs`：药品表，保存药品名称和图像特征。
- `patients`：患者表，保存患者姓名等信息。
- `batches`：批次表，表示某个患者对应的一次配药/注射批次。
- `batch_medicines`：批次药品关联表，表示某个批次应包含哪些药品。

数据库连接信息在 `config.yaml` 中配置：

```yaml
db_config:
  host: 192.168.137.1
  user: root
  password: root
  database: medicine_db
  charset: utf8mb4
```

请根据本机 MySQL 地址、用户名和密码进行修改。

## 配置说明

主要配置文件为 `config.yaml`。

### 摄像头/RTSP 配置

```yaml
RTSP:
  username: "admin"
  password: "a1b2c3d4"
  ip_address: "192.168.1.168"
  port: "554"
  channel: "201"
```

程序会根据这些参数拼接 RTSP 地址：

```text
rtsp://用户名:密码@IP:端口/Streaming/Channels/通道号
```

### 模型配置

```yaml
model:
  num_classes: 4
  checkpoint_path: "./model/best_model_0329_2.pt"
  yolo_path: "./model/best_yolo.pt"
```

当前代码中的 YOLO/RKNN 推理默认会加载根目录下的 `model_yolo_0510.rknn`。如果更换模型，需要同步检查 `src/segmentation/segmenter_yolo.py` 中的默认模型路径。

### 检测与显示配置

```yaml
segmentor:
  tile_size: 1024
  overlap: 16
  target_fps: 6
  batch_frames: 1

display:
  output_type: overlay
  overlay_alpha: 0.5
  display_scale: 0.5
```

这些参数控制分块检测大小、重叠区域、目标帧率和显示方式。

## 运行项目

确认以下内容已经准备好：

1. Python 环境和依赖已安装。
2. MySQL 已启动，`medicine_db` 数据库和相关表已创建。
3. `config.yaml` 中的数据库和 RTSP 参数已改成本机可用配置。
4. YOLO/RKNN 模型文件存在，例如 `model_yolo_0510.rknn`。
5. 摄像头视频流可以正常访问。

启动主程序：

```bash
python main.py
```

程序启动后会依次完成：

1. 读取 `config.yaml`。
2. 初始化数据库连接。
3. 初始化多个 PaddleOCR 实例。
4. 初始化药品文本匹配器 `DrugMatcher`。
5. 初始化药品图像特征分类器 `PharmaceuticalBottleClassifier`。
6. 连接 RTSP 视频流并进入实时检测流程。

## 关键模块说明

### `main.py`

项目主入口，负责组装整个系统。它不直接做复杂算法，而是把配置、数据库、OCR、药品匹配器、分类器和实时检测函数连接起来。

### `src/segmentation/segmenter_yolo.py`

负责 YOLO/RKNN 模型推理，包括：

- 加载 `.rknn` 模型。
- 将大图切成小块进行检测。
- 对检测结果做后处理和 NMS。
- 将局部检测框映射回原图坐标。

### `src/identification/OCRRecognizer.py`

封装 OCR 识别逻辑，输入裁剪后的药品或药袋图像，输出识别到的文本。

### `src/identification/DrugMatcher.py`

负责文字层面的匹配。OCR 得到的文本不一定完全准确，所以这里使用模糊匹配方法，在数据库中寻找相似的药品名或患者名。

### `src/identification/Recog.py`

负责药品图像特征匹配。它会从药品图像中提取 ORB/SIFT 类特征，并和数据库中保存的药品模板特征进行比较，用于辅助确认药品身份。

### `src/stream`

负责视频流读取和帧缓存，保证实时检测可以持续从摄像头获取画面。

## 数据和模型

项目中包含以下几类数据：

- `data/data.yaml`：YOLO 训练数据配置。
- `data/data_base/bag_data`：药袋样本图像。
- `data/data_base/bottle_data`：药品样本图像。
- `src/identification/feat_data`：药品特征模板样本。
- `runs/detect`：YOLO 训练或推理输出。
- `Temp_results`、`Temp_results_mask`：运行过程中的临时裁剪或中间结果。

模型文件：

- `model_yolo_0510.rknn`：用于目标检测。
- `model_cls.rknn`：用于分类或辅助识别。
- `runs/detect/runs/train/.../weights/best.pt`：训练过程中产生的 YOLO 权重。

## 常见问题

### 1. OCR 识别效果不稳定怎么办？

可以从以下方向排查：

- 检查摄像头清晰度和对焦。
- 减少反光、阴影和遮挡。
- 调整目标区域裁剪逻辑，保证标签文字完整进入 OCR。
- 检查 PaddleOCR 是否正确使用 GPU。
- 尝试调整 OCR 的检测阈值和图像预处理参数。

### 2. 目标检测漏检怎么办？

可以检查：

- `model_yolo_0510.rknn` 是否为最新模型。
- 训练数据中是否覆盖当前药品、药袋摆放角度。
- `conf_thres`、`iou_thres` 是否过高。
- 分块大小 `tile_size` 和重叠区域 `overlap` 是否合适。

### 3. 数据库匹配不到药品怎么办？

可以检查：

- `drugs` 表中是否存在对应药品名。
- OCR 文本是否包含过多噪声字符。
- `DrugMatcher.match()` 中的相似度阈值是否过高。
- 药品图像特征是否已经保存到数据库。

## 后续优化方向

- 增加统一的日志系统，替代大量 `print` 调试输出。
- 将临时脚本和旧版本脚本归档，保留稳定入口。
- 增加单图测试模式，方便不接摄像头时调试 OCR 和匹配逻辑。
- 增加 Web 或桌面端界面，显示药品、药袋、匹配结果和风险提示。
- 对模型文件、数据库配置、RTSP 密码等敏感内容做更规范的管理。
- 增加测试用例，覆盖 OCR 文本匹配、数据库查询、特征匹配等核心逻辑。

## 免责声明

本系统仅作为医疗配药/注射流程中的辅助安全检查工具，识别结果可能受到光照、遮挡、摄像头质量、模型精度、数据库完整性等因素影响。实际临床使用时，必须结合医院制度、人工核对和专业人员判断，不得将本系统作为唯一判断依据。
