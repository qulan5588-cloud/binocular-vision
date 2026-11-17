# 双目视觉：Astra Pro Plus 深度 + 彩色采集与联合标定

Orbbec Astra Pro Plus 是一台 RGB-D 一体相机，但厂方工具跑不出可用的同步数据。
这套脚本直接管理两路流：**深度走 OpenNI2（ctypes 调用，不依赖 GUI SDK）**，
**彩色走 ffmpeg 管道直取 MJPG 裸流**，720p 双流稳定 30/30fps、同步中位 6ms。

## 1. 换机器后怎么跑起来

### 1.1 前置

| 项目 | 要求 |
|---|---|
| 系统 | Windows 10/11 x64 |
| Python | 3.12（3.13 也测过，注意依赖要装到同一个解释器） |
| ffmpeg | 需在 PATH 里（`ffmpeg -version` 能跑）；彩色流靠它取裸流 |
| USB 驱动 | 需装 Orbbec Astra 驱动（`obdrv4.sys`），装完**必须重启一次**，否则设备完全不被识别（连未知设备都不出现）；重启后设备管理器应出现 `ORBBEC Depth Sensor`（Orbbec 类）和 `USB Camera`（彩色 UVC） |

### 1.2 装依赖

```bash
git clone <本仓库地址> && cd binocular-vision
uv venv .venv                                   # 或 python -m venv .venv
.venv/Scripts/activate                          # PowerShell: .venv\Scripts\Activate.ps1
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

依赖只有三件：`numpy`、`opencv-python==5.0.0.93`、`pillow`（界面用）。

### 1.3 OpenNI2 运行时

仓库已带精简运行时 `third_party/openni2`（约 4.4MB：OpenNI2.dll + orbbec 驱动 +
VC 运行库），拷到任何机器都能直接用。程序按以下顺序自动查找含 `OpenNI2.dll` 的目录：

1. 环境变量 `OPENNI2_LIB` 或 `OPENNI2_REDIST`
2. 本项目 `_driver/openni2/*/Win64-Release/sdk/libs`（原始 SDK 包，本机保留、不入库）
3. 本项目 `third_party/openni2`（**随仓库提交，推荐新机器用这个**）
4. `C:\Program Files[ (x86)]\OpenNI2\Redist`

都找不到时不会在 import 阶段崩掉，而是打开相机时给出带操作指引的错误。

### 1.4 启动

```bash
python capture_gui.py
```

启动日志第一行会打印版本号，例如 `界面版本：2026-11-15 手动采集版 v3`。
看不到这行说明跑的是旧进程/旧文件。

### 1.5 一键启动（`start.bat`）

不想记 python 路径 / 依赖装哪个解释器 / 语音要不要单独开窗口，**双击 `start.bat`**：

```
  双目视觉 · 启动台      内存 5.1/15.7 GB 可用
  1. 室内扫描（实时平面图 + 切片）
  2. 室内扫描 + TSDF 三维重建（导出 mesh.ply）
  3. 语音控制（自动挑麦，控制 Indoor Scan 窗口）
  4. 扫描 + 语音一起开（语音另开窗口）
  5. 功能启动台（全部卡片）
  6. 麦克风自检（录 4 秒看电平 + 试识别）
  7. 列出输入设备 + 现场电平
  0. 退出
```

也可以带参数直接进（适合做桌面快捷方式）：`start.bat scan|tsdf|voice|both|toolbox|check|list`。

启动前会做两件事：

* **内存体检**：可用内存 < 3 GB 时提醒先关 VS Code —— 16 GB 机器上 IDE + 浏览器
  能吃掉 6 GB，扫描时余量不足会丢帧、把双目同步打散（实测关掉 VS Code 从
  1.6 GB 回到 5.2 GB，帧率 30.0/29.6 fps、同步差均值 12.6 ms）。
* **确认解释器**：不在 `.venv` 里跑会提示，避免缺 opencv/onnxruntime。

## 2. 文件说明

| 文件 | 作用 |
|---|---|
| `capture_gui.py` | 图形界面（Tkinter）：预览、手动/自动抓帧、**引导式一键标定** |
| `app_align.py` | **第一个应用**：深度→彩色实时对齐 + 鼠标测距（自动取最新一份标定） |
| `capture_pair.py` | 采集内核：`DepthSource`（OpenNI2）+ `ColorSourceFFmpeg`（ffmpeg 管道） |
| `calib_capture.py` | 标定采集的底层工具：标定板检测（棋盘格 / 圆点板）、IR 转 8bit、中文路径写入 |
| `calib_solve.py` | 标定求解：彩色内参 → 深度内参 → 立体外参 → 验证叠加图（板型需与采集一致） |
| `make_chessboard.py` | 生成可打印的棋盘 PDF，支持按实测尺寸补偿打印机缩放 |
| `check_usb.ps1` | USB 链路自检（识别 HUB 层数、抓多余的设备节点） |
| `depth_probe.py` | 深度/IR 流验证：枚举、帧率、有效像素、深度统计，结果存 `_detect/<时间戳>/` |
| `probe_*.py` / `set_exposure.py` | 排查用的小工具：帧率、取流、曝光、设备枚举 |
| `yolo_seg.py` | YOLO11-seg ONNX 推理包装（DirectML/CPU），app_objects / app_people 共用 |
| `voice_cmd.py` | 本地语音控制（whisper 转写 → 关键词 → 给应用窗口发键），见 §4.12 |
| `plan_from_ply.py` | 离线重渲扫描平面图（读 cloud.ply，不碰相机） |
| `tsdf_fuse.py` | 可选 TSDF 融合后端（Open3D VoxelBlockGrid，`app_scan.py --tsdf` 用） |
| `start.bat` / `_launch.py` | 一键启动菜单（见 §1.5）：内存体检 + 常用组合，双击即用 |
| `_detect/voice_monitor.py` | 语音链路旁听器：同一套门限/切句，只打印不发键，用来查"说了没反应" |

## 3. 采集要点（踩过的坑）

- **彩色必须走 ffmpeg 后端**。OpenCV 5.0 的 MJPG 解码吞吐只有约 6.6MP/s，
  直接 `VideoCapture` 读 720p 只有 7fps、1080p 只有 3fps，与线材/HUB 无关。
- **暗光会让帧率掉到 17~20fps**：UVC 自动曝光会把曝光时间拉过 1/30s 主动降帧。
  界面里勾"锁定彩色曝光"（exp=-6，即 1/64s）即可恢复 30fps，或者开灯。
- **USB 延长线**：必须用带信号芯片的（有源）延长线。相机自带的内部 HUB 不算层，**外接一层有源延长线实测可用**（深度+彩色双流满 30/30、连跑 10 秒零丢帧、无端口复位）；原因是双流合计约 165 Mbps，远低于 USB 2.0 的 480 Mbps，多一级 HUB 只加几十微秒延迟、不吃带宽。但**不要再往下叠**——第 3 层开始分层轮询延迟会影响双流同步精度；也**不要用纯被动长线**，表现为"能枚举但偶发死锁"，最难查。
- 中文路径下 `cv2.imwrite` 会静默失败，统一走 `imwrite_u()`。

## 4. 标定流程

### 4.0 两种标定板任选

程序同时支持棋盘格和非对称圆点阵，界面"一键标定"里选类型，命令行用 `--board`：

| | 棋盘格 `--board chess` | 圆点板 `--board circles` |
|---|---|---|
| 靶标 | 打印 `chessboard_9x6_25mm.pdf` | CGB-020 铝基板（20 点，行内间距 20mm） |
| 参数（界面填） | 特征点 9 / 6，间距 25mm | 特征点 4 / 5，间距 20mm |
| 单帧特征点 | 54 | 20 |
| 物理跨度 | 200×125mm | 40×70mm |
| 工作距离 | 0.45~0.85 m | **0.35~0.60 m**（板小，必须靠得更近） |
| 180° 歧义 | 有，靠求解器双假设兜底 | **无**，转 180° 检出顺序不变 |
| 精度 | 更高 | 略低（点少、跨度小） |

圆点板要反色才能检出（`findCirclesGrid` 默认找浅底暗斑，这块板是黑底白圆），代码里已处理。
铝基板刚性好、不怕受潮翘曲，适合长期复用；需要最高精度时仍建议棋盘格。

### 4.1 采集步骤

1. 准备靶标：棋盘格用 `python make_chessboard.py --measured 95` 打印（按实测尺寸补偿
   打印机缩放，打印对话框必须选"实际大小/100%"），整面贴死在硬板上、哑光纸、照明均匀；
   圆点板直接用 CGB-020 铝基板。
2. **固定相机，手持板移动**；每次摆好后手离开板，静止 0.5 秒以上再采集。
3. 界面点"引导标定"，在配置框里选**摆放方案**（默认"中间横带"）：

   | 方案 | 怎么摆 | 适用 | 实测外参旋转误差* |
   |---|---|---|---|
   | **中间横带**（默认） | 板保持同一高度，只在左/中/右之间平移，叠加倾角和远近 | 没有升降台、桌面够宽 | 0.36° |
   | 只摆中间 | 板一直留在画面正中，只改倾角和远近 | 连左右都挪不动时 | 2.55°（不推荐） |
   | 全幅覆盖 | 中心→四边→四角→补倾斜，走满 12 格 | 有升降台/云台 | 0.14° |

   \* `_detect/pose_set_exp.py` 合成实测（圆点板、0.3px 角点噪声、20 次重复）。
   "只摆中间"角点横向跨度为 0，旋转会解到 2°级 —— 0.5m 处约 22mm 错位，
   深度贴彩色必然错边；只要能在桌面上左右挪板，就用横带。

   每步条件满足后**手动点"采集本步"**（程序不会自动抓帧）。
   引导对两种板都适配：示意图按板画（棋盘方块 / 圆点点阵）、距离提示和"板够大"
   门槛按板收紧，面板顶部会显示当前板型 —— 采集前核对一眼，别拿错板。
4. 采满自动求解，结果写到 `calib/<时间>/stereo_calib.json`，并生成 `verify_*.png`
   验证叠加图（红点应精确落在靶标特征点上）。

命令行采集同样支持：`python calib_capture.py --board circles`，
求解时板型必须一致：`python calib_solve.py --dir calib_raw --board circles`。

### 标定判据

| 指标 | 优秀 | 可用 | 必须回查 |
|---|---|---|---|
| 单目重投影 RMS | < 0.4 px | < 1.0 px | > 1.5 px |
| 立体重投影 RMS | < 0.5 px | < 1.0 px | > 1.5 px |
| 外参旋转角 | < 3° | < 15° | 解出 180° 附近必是错解 |
| 外参基线 \|T\| | 25~40 mm | 10~150 mm | 超出范围必是错解 |

### 求解器已经处理的两个陷阱

- **棋盘图案的 180° 简并**：10×7 格棋盘在"旋转 180°+黑白反转"下与自身等价，
  OpenCV 在彩色流和 IR 流里可能挑相反的起点角点，直接解会得到绕光轴 180° 的
  镜像解。求解时会对"IR 角点整串反序"再解一次，按物理合理性（旋转 < 15°、
  \|T\| 在 10~150mm）选解；两个假设都不合格就直接报错，不会静默保存垃圾结果。
- **联合精化内参**：`stereoCalibrate` 会把内参一起优化，验证投影和 JSON 里保存的
  都是精化后的 K/D，避免出现"RMS 很低但验证图乱掉"的假象。

覆盖不足（例如板没动、只覆盖 1~2 格）时内参本身就会漂，外参必然发散——
这种情况下宁可不采，也不要凑数。角点在画面上的**横向跨度**尤其关键：
全部挤在中间时旋转会解到 2°级（见 4.1 的方案对比）。

## 4.2 第一个应用：深度-彩色对齐 + 测距（`app_align.py`）

```bash
python app_align.py                    # 自动取 calib/ 下最新一份标定
python app_align.py --calib calib/20261116_154823/stereo_calib.json
```

标定做完的第一个应用：把深度逐像素反投影 → 用外参换到彩色相机系 → 用彩色
内参投影回彩色图，得到"每个彩色像素处的深度"。彩色画面上叠深度伪彩
（蓝近红远，量程可用 `--near/--far` 调），**鼠标移到哪就实时显示那点的距离**。

- 操作：鼠标=测距，左键=锁定读数，`m`=切换 叠加/并排/仅深度，`s`=存图，
  `q`/ESC=退出；
- 两个必然现象，不是 bug：彩色 **左右两侧没有深度**（彩色 FOV 66.1° 比深度
  58.4° 宽），**上下贴边被裁**（彩色 40.2° 比深度 45.5° 窄）；太近的物体
  （<0.6m，规格书量程下限）成片空洞；
- 性能：对齐在彩色半分辨率上泼溅再线性放大（全分辨率会留六成空洞且补洞慢），
  测量读数则直接查全分辨率投影结果；实测 720p 双流 ~25 fps。

## 5. 输出

```
captures/<YYYYmmdd_HHMMSS>/     普通采集：color_*.jpg, depth_*.png, captures.csv, device.txt
calib/<YYYYmmdd_HHMMSS>/        标定：calib_*_color.png / calib_*_ir.png, pairs.csv,
                                stereo_calib.json, verify_*.png, _failed/（未采纳的失败帧）
```

`stereo_calib.json` 关键字段：

| 字段 | 含义 |
|---|---|
| `color.K` / `color.D` | 彩色内参 / 畸变（已立体联合精化），尺寸 `color.size` |
| `depth.K` / `depth.D` | 深度（=IR 共镜头）内参 / 畸变 |
| `R_depth_to_color` / `T_depth_to_color` | 把点从**深度相机坐标系**变换到**彩色相机坐标系**：`X_color = R·X_depth + T`，T 单位 mm |
| `stereo_rms_px` | 立体重投影 RMS |

## 6. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'/'numpy'` | 终端里的 Python 和你装包的 Python 不是同一个。确认已激活 `.venv`，并 `python -c "import sys; print(sys.executable)"` |
| `找不到 OpenNI2 运行时` | 见 1.3，把 `third_party/openni2` 放回原位，或设 `OPENNI2_LIB` |
| 深度枚举不到设备 | 刚异常退出会导致占用，拔插一次相机；或先跑 `python depth_probe.py`（与主程序同链路，输出带排查指引） |
| 彩色帧率只有 7fps | 后端没走 ffmpeg（界面选 ffmpeg），或 OpenCV 在解码 MJPG |
| 彩色帧率 17~20fps、深度正常 | 暗光自动曝光降帧，勾"锁定彩色曝光"或开灯 |
| 标定解出 180° 旋转 / 基线离谱 | 姿态覆盖不足，用引导标定重新采（默认"中间横带"即可，别用"只摆中间"） |

### 4.3 3D 点云（`app_pointcloud.py`）

第二个应用：深度反投影 → 外参变换到彩色相机系 → 彩色采样，得到带 RGB 的点云，
自写 numpy 渲染器实时旋转/缩放（无 open3d 依赖），可导出标准 binary PLY。

```bash
python app_pointcloud.py                  # 实时点云
python app_pointcloud.py --accum 5        # 累积 5 帧：去噪、补空洞
python app_pointcloud.py --depth-color    # 按深度伪彩着色
python app_pointcloud.py --shot out.png   # 渲染一帧存图后退出
```

操作：拖动 = 旋转，滚轮 = 缩放，双击/r = 复位，1/2/3 = 正视/俯视/侧视，
c = 彩色/伪彩切换，g = 坐标轴，s = 存 PLY（captures/cloud_*.ply，
MeshLab / CloudCompare 直接打开），q = 退出。

坐标约定：彩色相机系，X 右 / Y 下 / Z 前（OpenCV 约定，米）。

### 4.4 平面尺寸测量（`app_measure.py`）

第三个应用：框选 → RANSAC 拟合平面 → 平面内最小外接矩形 → 输出长×宽(mm)、
平面距离、倾角（是否正对相机）、拟合质量（内点率/RMS）。

```bash
python app_measure.py                     # 实时拖框测量
python app_measure.py --thresh 0.004      # 平面阈值放宽（远处/噪声大）
python app_measure.py --rect 410,250,705,455 --shot out.png   # 无人值守出图
```

操作：拖框 = 测量（每 10 帧自动跟随刷新），u = 立即重算，r = 清除，
m = 叠加深度伪彩（诊断空洞），p = 存平面内的点云，q = 退出。

关键实现：
* RANSAC 只在 ≤4000 点子集上采样（300 次），内点判定与 SVD 精化用全量点；
* ★ 尺寸不用 minAreaRect（它包噪声极值，3mm 噪声让 A4 偏大 11mm），
  改成 PCA 主轴对齐 + 1%/99% 分位反推跨度，1m 处误差 <1.5mm；
* 自测：`_detect/measure_selftest.py`，6 组合成数据（倾斜/外点/旋转/噪声）。

### 4.5 近处障碍 / 安全距离提示（`app_guard.py`）

第四个应用：视场按列分区（默认 3 区），每区实时输出最近有效深度，按阈值分级
（危险 / 警告 / 安全），危险区物体染红/橙色点 + 闪红边 + 大字提示。

```bash
python app_guard.py                        # 警告 0.8m / 危险 0.5m
python app_guard.py --warn 1.2 --danger 0.7 --cols 5
python app_guard.py --shot out.png         # 出一张图后退出
```

操作：s = 存图，m = 叠加深度伪彩，[ / ] = 警告阈值 ±0.1m，q = 退出。

实现要点：每区最近距离取 **5% 分位**而不是 min——深度图边缘的飞点（遮挡边界、
多路径）会把 min 骗到 5cm 级，分位对少量离群点免疫（自测通过）。

### 4.6 物体分割 / 计数 / 粗尺寸（`app_objects.py`）

第五个应用：把视场里的物体一块块分出来，标序号、距离、宽×高×深、体积估算。
这是"货位有没有货、有几个"的最小可用版本。

```bash
python app_objects.py                  # 默认：剔背景平面 + 计数
python app_objects.py --keep-plane     # 不剔平面（整片场景分块）
python app_objects.py --jump 60        # 深度跳变阈值放宽
python app_objects.py --min-area 800   # 只关心更大的物体
python app_objects.py --zmax 1.5       # 只看 1.5m 内（操作区）
python app_objects.py --shot out.png   # 出一张图后退出
```

操作：点右上角 MIRROR（或 f）= 镜像画面，p = 平面剔除开关，d = 深度伪彩，
**y = 语义分割（YOLO）/ 几何分割 切换**，s = 存图，q = 退出。

**YOLO 语义分割模式**（默认开启，需 `models/yolo11n_seg.onnx` + `onnxruntime-directml`）：

* 模型 YOLO11n-seg（COCO 80 类，静态 640 输入，11MB）在彩色图上出实例掩码，
  认得出的类别直接标名字（bottle / suitcase / chair / person / backpack……）；
* 掩码内深度点（经标定投影查掩码）用 0.5%/99.5% 分位量长宽高 —— **识别归
  RGB、量尺寸归深度**，尺寸仍是真实测量不是猜的；
* 掩码内做**深度主簇**过滤（10cm 直方图最密峰 + 峰值 20% 连续生长）：
  RGB 掩码边缘的遮挡泄漏会让物体身后的墙混进来，不滤的话椅子量出 4.2m
  进深（实机首测）；
* 推理走 DirectML（RTX 4060 实测纯推理 8.2ms、整链 14.5ms，CPU 是 42.6ms）。
  为什么不是 onnxruntime-gpu：Windows 上要自装 CUDA+cuDNN，DirectML 一个
  pip 包直接吃任意 DX12 GPU。**注意 ultralytics 会带一个 CPU 版
  onnxruntime 把 directml 文件顶掉**（两个包共用模块目录），装完要重装
  onnxruntime-directml；
* 已知限制：透明/吸光容器（矿泉水瓶）深度是洞，量到的是身后背景的簇；
  3m 外的低置信检出尺寸不可信。这些按 `y` 切回几何分割即可。

YOLO 模型重导出（换类/换尺寸才需要）：`pip install ultralytics` 后
`YOLO('yolo11n-seg.pt').export(format='onnx', imgsz=640, opset=17,
dynamic=False)`。运行时不依赖 torch。

关键实现：
* 分割在**深度图 2D 域**做（相邻像素深度差 > jump 即断开 + 膨胀 1px 成分隔带，
  再连通域），几毫秒完成，比 3D 欧式聚类快两个数量级，也不用 open3d；
* **迭代多平面剔除**（最多 3 个）：墙剔掉后桌面还在，桌子和物体侧面又常是
  深度连续的，会把物体连成一片——所以一路剔到大平面剔完为止。但判据用
  "占整幅比例 > 15%"，不然箱子正面（也是平面）会被误剔；
* 尺寸用 0.5%/99.5% 分位反推（比 app_measure 的 1% 更松：掠射方向的可见面
  在图像上只有几十列像素，砍 1% 就等于砍掉一整列）；
* 自测：`_detect/objects_selftest.py`，真实内参光线求交渲染"背板+两箱子"，
  分块 2/2、尺寸相对误差 3.0% / 7.9%。

已知限制（几何模式）：贴在一起且深度连续的物体会算成一块；正对光轴的物体
侧面是掠射角（1px 宽），量不出深度方向尺寸——包围盒深向会偏小。（这两条
正是 YOLO 模式要解决的：实例分割不靠深度连续性分物体。）
自测：`_detect/yolo_selftest.py`——letterbox 往返、掩码->深度->分位尺寸链路
（合成刚体误差 <3.3%）、背景剔除、GPU provider、性能五组断言。

### 4.7 工具箱入口（`app_launcher.py`）

```bash
python app_launcher.py
```

点卡片（或按 1-6）启动应用，右上角实时显示相机占用状态，底部显示最新标定。
`r` 刷新状态，`q` 或窗口 X 退出（关闭工具箱时会一并结束正在运行的子应用，
避免留下孤儿进程占着相机）。界面用 Pillow 渲染中文（cv2 的 putText 不支持）。

#### 关窗技巧（`ui_common.py`）

Windows 上 cv2 的标题栏 X 只是把窗口标成不可见，`waitKey` 循环不会退出，
界面会「关不掉」。五个应用统一用 `ui_common.wait_key()` 取键，它顺手查
`WND_PROP_VISIBLE`，窗口被关掉就返回 ESC，循环自然退出。自己写新界面时
也请用它，别再直接调 `cv2.waitKey`。

#### 相机互斥锁（`camera_lock.py`）

**为什么必须存在**：Astra 只有一台，两个进程同时开深度流**不会报错**，
而是深度流大面积失效——界面 HUD 显示 depth 只有几个百分点、画面大片黑洞，
会被误判成传感器故障（2026-11-16 实际踩过，排查了半天才发现是双开）。

锁的实现：`.camera.lock`（JSON：pid/name/时间），`DepthSource.__init__` 开头
自动抢锁，抢不到直接抛 `CameraBusy` 并给出占用者 PID 与处置命令；
进程退出（atexit）自动清理；被 kill 后残留的锁按 PID 判活自动失效。
所有应用零改动即受保护；`app_launcher` 启动前还会先检查一遍并提示。

### 4.8 实时深度监视（`app_depthview.py`）

`depth_diag.py` 是命令行脚本（统计打印完存两张 PNG 就退出），不适合日常看。
这个界面把同样的统计做成实时的：

```bash
python app_depthview.py             # 左深度伪彩 + 右掩码网格 + 底部直方图
python app_depthview.py --mirror    # 翻转与相机同向（深度流本身是镜像的）
python app_depthview.py --sec 5     # 跑 5 秒打印真实帧率后退出
```

* 右侧视图 `m` 切换：8x6 有效率网格（每格标 %，<30% 红 / <60% 黄）/
  稳定度累积（亮=一直有深度，暗=一直空，灰=闪空）/ 纯掩码
* 底部直方图：9 档深度分布，近距盲区、>8m 超程一眼看出
* HUD：真实 fps（按帧时间戳去重，29.9fps 实测）、z=0 / <0.3m / >8m 占比、中位距离
* `r` 重置累积，`f` 镜像，`s` 存图，窗口 X / q 退出

### 4.9 人数统计 + 人脸追踪（`app_people.py`）

第七个应用：YuNet 人脸检测（`models/face_detection_yunet_2023mar.onnx`，227KB，
OpenCV 5 精简包不再附带 Haar 人脸模型）+ 自写多目标跟踪 + 深度融合。

```bash
python app_people.py                # 实时：人数 / ID / 距离 / 跨线计数
python app_people.py --fast         # 960x540 检测，省一半 CPU
python app_people.py --line 0.5     # 计数线位置（画面宽度比例）
python app_people.py --thresh 0.7   # 检测置信度阈值
python app_people.py --image a.png  # 单张图检测出图（不占相机）
```

* 每张脸带稳定编号 `#id` 和距离（深度图对齐后取人脸框中心 40% 区域中位深度）；
* 顶部 HUD：当前人数（大字）/ IN / OUT / PEAK / fps；中线（黑描边+橙线）跨线计
  进出，±30px 滞后带防"站在线上抖动"重复计数；
* 连续命中 3 帧才确认（单帧误检不算人），丢失 12 帧注销，未确认轨迹画灰框；
* **交叉防串**：匹配按"中心距预测位置 / 人脸尺度"贪心——两人迎面走过画面上
  重合的那几帧，只看距离必然配反，轨迹互相拖拽卡在线两侧不穿越；加匀速预测
  （速度指数平滑）后自测 9/9 通过；
* **人体兜底（P2）**：背面/低头/暗光下 YuNet 看不到脸，复用 YOLO 分割模型
  （`models/yolo11n_seg.onnx`，GPU 约 15ms/次，隔帧跑）的 person 类兜底。
  两段式匹配防重复计数：人脸优先配轨迹 → 没配上的脸开新轨迹 → 人体框再配
  （刚开的脸轨迹也在候选里，同一人的身体框被脸轨迹"吃掉"）→ 剩下的才开
  人体轨迹。人体框锚点取框顶往下 1/4（头的位置），和人脸中心语义一致；
  人体轨迹确认门槛 4 帧（人脸 3 帧）——证据弱（外套/海报可能误检），宁可
  晚计不误计。`--no-yolo` 关闭，`y` 键运行中切换；
* 操作：m 镜像，l 显隐计数线，[ / ] 调阈值，y 人体兜底开关，s 存图，q / 窗口 X 退出。

自测：`_detect/people_selftest.py`（13 例合成轨迹：单程/往返/线上抖/交叉/丢失续
接/注销/误检/并排 + 人体兜底接管/纯人体跨线/脸回来不重复/双人混合）。

### 4.10 手势交互（`app_gesture.py`）

第八个应用：MediaPipe HandLandmarker（`models/hand_landmarker.task`，7.8MB）
+ 食指虚拟光标 + 捏合点击 + 悬停自动触发。

```bash
python app_gesture.py               # 实时：手骨架 + 手势名 + 光标按钮
python app_gesture.py --mirror      # 画面镜像
python app_gesture.py --image a.png # 单张图识别出图（不占相机）
```

* **手势识别**：指尖到手腕的距离 > 该指 PIP 到手腕 × 1.08 判伸直（对掌向不敏感），
  组合出 FIST / OPEN / POINT / VICTORY / THUMB / 捏合；拇指单独用"指尖到食指根"
  判据（它 PIP 离腕近，通用判据恒假）；
* **镜像与坐标**：MediaPipe 永远吃原始相机帧（画面中途翻转跟踪器会跟丢），
  显示镜像由 `View` 统一做 相机像素→画布 映射——骨架、光标、深度叠加全在同一
  坐标系，镜像时才不会"手在右、骨架在左"；查深度用 `x_back` 反算回相机坐标
  （投影 proj 恒为相机坐标系，之前镜像时查到的是对面位置的深度，已修）；
* **按钮面板在画面外**：四个按钮放画面右侧独立面板（画布 1280+292），不压画面；
  指尖够不到，画面最右 16% 是"推出去"过渡带——手指越过缝隙光标滑进面板
  （位置连续，纵向不变），手往右一推+上下移即可选按钮；鼠标也能直接点；
* **光标**：食指尖 EMA 平滑（关键点自带 2~3px 抖，不平滑点不中按钮）；
  切换镜像时光标锚点重置，避免从旧位置飘过去；
* **点击**：捏合（拇指尖-食指尖距 < 0.30×掌心长，松开阈值 0.45 迟滞）；
  悬停按钮 0.9s 也自动触发（dwell click，带进度条），触发后冷却 1s 防连点；
* 按钮：MIRROR / DEPTH（深度伪彩叠加）/ SNAPSHOT / RESET，副行显示热键、
  触发次数与 ON/OFF 状态；面板底部有操作提示；
* 手掌中心（中指根）用全分辨率投影查精确距离，随 HUD 显示；
* 布局/镜像回归：`_detect/gesture_layout_test.py`（假关键点走真实绘制链路，
  8 项断言 + 3 张渲染图肉眼核对，不需要相机）。

依赖：mediapipe 1.2.3（`--no-deps` 装本体 + absl-py/certifi/flatbuffers/
sounddevice/matplotlib，**不装 opencv-contrib** 以免覆盖 cv2 5.0）。

### 4.11 室内扫描建模（`app_scan.py`）

第九个应用：端着相机走一圈，逐帧配准拼出**全局彩色点云**，实时渲染成
**俯视平面图**（5cm 格、按高度着色 + 轨迹 + 视锥 + 比例尺）。纯 numpy 的
KinectFusion 精简版，不需要 scipy/open3d。

```bash
python app_scan.py                    # 交互扫描，左边平面图右边实况
python app_scan.py --voxel 0.03      # 体素更粗（点少一半、更快）
python app_scan.py --tsdf            # 额外跑 Open3D TSDF，导出 mesh.ply（可选）
python app_scan.py --sec 20 --out captures/scan_demo   # 定时自测并保存
```

* **帧间配准**：深度 → 顶点/法线图（3 级金字塔，建在抽稀后的深度上，
  640x480 全图算法线要 73ms 而精度没用，省掉后 Frame 11.5ms）→
  投影法 ICP（point-to-plane，残差 = n·(T·p−q)，雅可比 [p×n, n]，金字塔
  2→1→0 各层高斯牛顿，最细层抽到 2.4 万点）；
* **退化方向阻尼**：正对白墙/大片地板时横向平移在几何上没约束，硬解会
  "顺着墙滑走"（合成场景实测侧滑 37cm）。对法方程做特征分解，把远小于
  主方向的特征值抬到 3e-3×主值 —— 等价于在这些方向上相信匀速预测。
  修复后快速运动（15cm/帧）误差从 277mm 降到 0.5mm；
* **位姿累计**：G = G·T（T 是当前帧→上一帧），匀速预测给初值，跟丢时
  清零速度模型重猜；单帧位移 >0.45m 或转角 >25° 判失配，标 LOST 不融合；
* **融合**：只在位移 >1.2cm 或转角 >0.6° 时融合；点按 2cm 体素去重
  （np.unique 结构化视图），缓冲超过 90 万点才做一次全量压实；
* **平面图**：点投到 (x,z) 平面按 5cm 格取平均高度，高度用 turbo 着色
  （y 向下，世界系 y 就是"低"）；视野按点云 1%~99% 分位自适应 + EMA 平滑；
* **地板平面 + 高度切片**（重要）：不切片时地板和天花板叠进同一格取
  均值，整张图是看不出房间轮廓的彩虹糊团（实机首测教训）；而长扫描的
  姿态漂移还会把世界系带歪，固定"水平切片"切不到地板（实机 36 分钟
  扫描实测，各高度层点量平滑分布、无峰）。所以点攒够 1.5 万后周期性
  （每 60 帧）从点云拟合**真实地板平面**：(x,z) 25cm 网格每格取最低点
  + 剔除稀疏边缘碎格 + 3 点 RANSAC 播种（法线约束在竖直 45° 内，
  防天花碎格拉歪初始平面）+ PCA 精修，43ms/次。平面图投影到地板平面
  上、"高度"= 地板上方高度、切片沿地板法线切 —— 世界系歪了也能出
  正的房间图。默认切片地板+0.10±0.20m；`x` 一键切全高度/切片（看家具
  实物用全高度），`[ ]` 上下移（10cm），
  `-` / `=` 调带宽（5cm）；保存同时出切片版 plan.png + 全高度
  plan_full.png；`r` 重扫会重新拟合地板；
* **保存**（s 键 / 定时）：`captures/scan_*` 下 cloud.ply（顶点色，
  MeshLab/CloudCompare 直接开）+ plan.png（1080 正射平面图）+ traj.csv +
  meta.json（点数、关键帧数、路径长度）；
* **离线重渲**：`plan_from_ply.py captures/scan_xxx`——不碰相机，读已保存
  的 cloud.ply 重新拟合地板、重出平面图（`--band` 切片高度 / `--width`
  带宽 / `--size` 输出边长），算法升级后不必重扫；
* **自测**：`_detect/scan_selftest.py`——合成房间（地面/天花/四壁+两家具
  盒子）光线求交渲染 + 已知轨迹喂真实 ICP 链路，三个场景全过：
  常规 5cm/帧误差 0.3mm、快速 15cm/帧 0.5mm、原地转 3°/帧 0.4mm，
  终点漂移分别 4.4/3.0/6.9mm（路程 1.95m / 1.65m / 转 69°）；
* **已知局限**：没有 IMU，平面图默认相机基本端平；没有回环检测，绕一圈
  回起点有累积漂移（走远了分段扫再拼）；反光/吸光面在点云里是洞。

#### 可选后端：Open3D TSDF（`--tsdf`，实测数据说话）

默认的 2cm 体素去重零依赖、可回溯，短板是**只有散点、没有网格**，噪声也不
做平均（薄门板扫两遍会糊成两层）。加 `--tsdf` 会额外把同一批深度帧喂给
Open3D 的 `VoxelBlockGrid`（TSDF 加权融合），保存时多出两个文件：

| 文件 | 内容 |
|---|---|
| `cloud_tsdf.ply` | 等值面点云（tsdf=0），比原始点云更薄更准 |
| `mesh.ply` | 三角网格（带顶点色，8.6MB/房间量级），可量面积体积 |

**2026-11-17 实测**（`_detect/tsdf_bench.py`，合成 6x8x2.6m 房间 + 两件家具，
20 帧 / 2cm 体素，RTX 4060 8G）：

| 方案 | 每帧耗时 | 点数 | 三角面 | 墙面厚度 | 地面平整度 |
|---|---|---|---|---|---|
| 现方案 numpy 体素 | 14.4ms | 160.9k | — | 23.0mm | 19.0mm |
| Open3D TSDF · CUDA | **1.2ms** | 125.9k | 246.5k | **15.7mm** | **14.0mm** |
| Open3D TSDF · CPU | 5.1ms | 125.9k | 246.5k | 15.7mm | 14.0mm |

（耗时只算 integrate；彩色融合再加 5.6ms/帧，因为要把彩色 remap 到深度图、
按 uint16+uint8 上传。ICP 那 11ms/帧 是两种方案都要付的。）

结论：**TSDF 确实更好，但只作为可选增强**——装完实测占 **2.6GB**（open3d
333MB + cublas 737MB / cusolver 450MB / cusparse 456MB / npp 385MB /
nvrtc 179MB，其中 cublasLt、npp 的绝大多数库根本用不到），而且换台没 N 卡的
机器就退化成 CPU 版。所以默认关，主流程保持零依赖。

装法（Windows，Python 3.12）：

```bash
# 1) 官方 release 里的 CUDA preview wheel（官网文档的 URL 是错的，
#    真正的文件名是 open3d_cuda-...，不是 open3d-...）
pip install https://github.com/isl-org/Open3D/releases/download/v0.20.0/open3d_cuda-0.20.0-cp312-cp312-win_amd64.whl
# 2) 它静态依赖的 CUDA 运行库（pip 不会自动加到 PATH，tsdf_fuse.py 里用
#    os.add_dll_directory 自己挂）
pip install nvidia-cublas-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-npp-cu12
```

* 版本号对不上（open3d 期望 12.6、pip 装到 12.9）只是 warning，同一大版本
  二进制兼容，实测可用；
* 退出时会打印一行 `CUDA runtime error: driver shutting down`——open3d 的
  静态析构与解释器退出顺序冲突，**无害**，数据已保存完；
* 自测：`_detect/tsdf_selftest.py`（6 项：可用性 / 彩色集成 / 点云非空 /
  颜色 / 墙厚 / mesh 写盘回读），当前 6/6 通过。

### 4.12 本地语音控制（`voice_cmd.py`）

第十个应用：说中文控制应用——独立进程监听麦克风 → faster-whisper 转写 →
关键词精确匹配 → 给目标窗口 `PostMessage WM_CHAR` 发键（复用已实测的远程
按键链路，目标窗口不用聚焦）。

为什么独立进程而不是塞进各 app 主循环：转写一句 0.7~3s，塞进去会把 19fps
的扫描卡成幻灯片；各 app 本来就有键盘接口，语音只是"替你按键"。

```bash
python voice_cmd.py                     # 默认控制 Indoor Scan（自动挑麦克风）
python voice_cmd.py --check             # ★ 说了没反应先跑这个：录 4 秒看电平
python voice_cmd.py --list              # 列输入设备 + 现场电平（挑设备）
python voice_cmd.py --device 2          # 指定麦克风（--list 里的编号）
python voice_cmd.py --target People     # 控制指定窗口（标题子串）
python voice_cmd.py --wav a.wav         # 用录音跑完整链路（不占麦克风）
python voice_cmd.py --text "帮我保存"    # 离线测关键词映射（不开麦）
python voice_cmd.py --model small       # 一级就上 small（更准，慢 3 倍）
python voice_cmd.py --no-fallback       # 关掉 small 复核
```

* **指令表**：开始/继续/暂停 → 空格，保存 → s，重扫/清空 → r，
  三维/预览 → v，网格 → g，全高度 → x，上一层 → `]`，下一层 → `[`，
  加宽 → `=`，变窄 → `-`，退出 → q；`t` 键切换目标窗口，`q` 退出；
* **两级识别**：一级 base int8（约 1s）；一级没匹配上（或命中多个不同
  指令）时用 small int8（约 3s）复核一遍，仍不确定就不发键。实测同一句
  很轻的输入（峰值 0.03），base 听成「暴存」、small 能听对「保存」；
* **宁可不执行，不能乱执行**：能量门限触发录音（自适应噪声底）、短于
  0.4s 丢弃、`no_speech_prob>0.55` 判幻觉丢弃、"同词重复 ≥3 次或碎片
  >8 个"判幻觉丢弃、一句里命中**多个不同指令**时交给 small 裁决、发键后
  1.5s 冷却防重复；
* **取最先说的那个词**（不是词表优先级）：实测弱信号下 base 会输出
  「保存 第1层 下一层 下一层…」，按词表顺序会命中"下一层"发出 `[`——
  用户明明说的是"保存"；
* **音频踩过的三个坑**（2026-11-17 排查"语音控制没反应"）：
  1. 系统默认输入可能指向**相机自带的远场麦克风**（本机默认 #1 Astra，
     峰值 3e-5，等于没有）；`--check` / `--list` 能看出来，`--device N` 换；
  2. **必须按设备原生采样率开流**：某些设备用 16kHz 开流不报错，但读回来
     全是 0（静音而非异常），比报错难查得多；重采样到 16k 交给 whisper；
  3. 提示词别写成整句（"以下是室内扫描软件的简短中文指令："）——输入一
     弱，whisper 会把**提示句本身**当内容吐出来；改成纯词表后不再复现；
* **自动增益**：送 whisper 前把峰值抬到 0.2 量级（最多 +18dB），弱麦克风
  也能识别；输入实在太轻（峰值 <0.0015）直接跳过，不喂给模型；
* **模型**：`models/whisper/base`（145MB）+ `models/whisper/small`（461MB），
  都不入库。换机重建：`HF_ENDPOINT=https://hf-mirror.com`，
  `snapshot_download("Systran/faster-whisper-base", local_dir="models/whisper/base")`；
* **依赖坑**：faster-whisper 会带 CPU 版 onnxruntime 顶掉 directml
  （同 ultralytics 的坑），装完必须
  `pip install --force-reinstall --no-deps onnxruntime-directml==1.24.4`；
* **自测**：
  * `_detect/voice_selftest.py`（27 例：同义说法 / 负例 / 多词取最先 /
    歧义检测）；
  * `_detect/voice_e2e.py`（系统 TTS 中文指令 → 真实转写 → 映射）；
  * `_detect/key_selftest.py`（PostMessage WM_CHAR 发键链路：开一个 cv2
    窗口自收自发，5/5 收到，含空格）；
  * `--wav` 四级音量回归（0dB/-20dB/-30dB/-36dB）：四级都正确发出 `s`，
    最弱两档由 small 复核兜住。

