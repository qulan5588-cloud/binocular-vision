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
| USB 驱动 | 需装 Orbbec Astra 驱动（`obdrv4.sys`），装完设备管理器能看到 Astra |

### 1.2 装依赖

```bash
git clone <本仓库地址> && cd 双目视觉
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
| `capture_pair.py` | 采集内核：`DepthSource`（OpenNI2）+ `ColorSourceFFmpeg`（ffmpeg 管道） |
| `calib_capture.py` | 标定采集的底层工具：角点检测、IR 转 8bit、中文路径写入 |
| `calib_solve.py` | 标定求解：彩色内参 → 深度内参 → 立体外参 → 验证叠加图 |
| `make_chessboard.py` | 生成可打印的棋盘 PDF，支持按实测尺寸补偿打印机缩放 |
| `check_usb.ps1` | USB 链路自检（识别 HUB 层数、抓多余的设备节点） |
| `probe_*.py` / `set_exposure.py` | 排查用的小工具：帧率、取流、曝光、设备枚举 |

## 3. 采集要点（踩过的坑）

- **彩色必须走 ffmpeg 后端**。OpenCV 5.0 的 MJPG 解码吞吐只有约 6.6MP/s，
  直接 `VideoCapture` 读 720p 只有 7fps、1080p 只有 3fps，与线材/HUB 无关。
- **暗光会让帧率掉到 17~20fps**：UVC 自动曝光会把曝光时间拉过 1/30s 主动降帧。
  界面里勾"锁定彩色曝光"（exp=-6，即 1/64s）即可恢复 30fps，或者开灯。
- **USB 延长线**：带信号芯片的延长线可用；链路里允许有一层 HUB，再多就会掉带宽。
- 中文路径下 `cv2.imwrite` 会静默失败，统一走 `imwrite_u()`。

## 4. 标定流程

1. 打印棋盘：`python make_chessboard.py --measured 95`（若打印件量出来只有 95mm，
   该参数按实测值补偿打印机自动缩放）。打印对话框必须选"实际大小/100%"。
2. 棋盘整面贴死在硬板上（喷胶满贴，别只贴四角），哑光纸，照明均匀。
3. **固定相机，手持板移动**；每次摆好后手离开板，静止 0.5 秒以上再采集。
4. 界面点"引导标定"，按 15 步引导走满画面 **12 个格子**（中心→四边→四角→补倾斜），
   每步条件满足后**手动点"采集本步"**（程序不会自动抓帧）。
5. 采满自动求解，结果写到 `calib/<时间>/stereo_calib.json`，并生成 `verify_*.png`
   验证叠加图（红点应精确落在棋盘角点上）。

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
这种情况下宁可不采，也不要凑数。

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
| 深度枚举不到设备 | 刚异常退出会导致占用，拔插一次相机；或先跑 `python depth_probe.py` |
| 彩色帧率只有 7fps | 后端没走 ffmpeg（界面选 ffmpeg），或 OpenCV 在解码 MJPG |
| 彩色帧率 17~20fps、深度正常 | 暗光自动曝光降帧，勾"锁定彩色曝光"或开灯 |
| 标定解出 180° 旋转 / 基线离谱 | 姿态覆盖不足，用引导标定重新采满 12 格 |
