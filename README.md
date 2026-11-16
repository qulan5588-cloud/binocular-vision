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
