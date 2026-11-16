# -*- coding: utf-8 -*-
"""
Astra Pro Plus · 双流同步采集（图形界面）
------------------------------------------------------------------
深度：OpenNI2 直读 640x480@30fps
彩色：ffmpeg 管道直取 MJPG（可满 30fps）或 OpenCV DShow（备用）

运行:
    python capture_gui.py

快捷键:
    空格    抓取一组
    A       开关自动抓帧
"""
import os
import sys
import csv
import io
import json
import queue
import time
import ctypes
import threading
import traceback
import contextlib

import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import cv2
from PIL import Image, ImageTk

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import capture_pair as cp     # noqa: E402
import calib_capture as cc    # noqa: E402
import calib_solve as cs      # noqa: E402

# 启动时打进日志，用来确认跑的是不是最新代码。改完代码没重启界面时，
# 从这一行就能一眼看出来（老进程不会打印当前版本）。
BUILD = "2026-11-15 手动采集版 v3"

# ------------------------------------------------------------------ 配色 ---
C_BG     = "#0f1218"
C_PANEL  = "#151a23"
C_CARD   = "#1b2230"
C_CARD2  = "#212a3a"
C_LINE   = "#2a3444"
C_FG     = "#e8eef8"
C_SUB    = "#8795a8"
C_DIM    = "#5d6a7d"
C_ACCENT = "#4aa8ff"
C_OK     = "#3ddc9a"
C_WARN   = "#ffc45c"
C_ERR    = "#ff6b6b"
C_CANVAS = "#07090d"

FT   = "Microsoft YaHei UI"
MONO = "Consolas"

F_TITLE = (FT, -19, "bold")
F_SUB   = (FT, -11)
F_SM    = (FT, -11)
F_MD    = (FT, -13)
F_BOLD  = (FT, -13, "bold")
F_SEC   = (FT, -11, "bold")
F_BIG   = (MONO, -26, "bold")
F_NUM   = (MONO, -13)


def _dpi_aware():
    """让 Tk 用自己的像素坐标系，避免系统缩放导致界面发虚"""
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# ------------------------------------------------------------- 联合标定 ---
class _LineWriter(io.TextIOBase):
    """把 print 输出按行转给日志回调（calib_solve.run_solve 里全是 print）"""

    def __init__(self, put):
        super().__init__()
        self._put = put
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._put(line.rstrip())
        return len(s)

    def flush(self):
        if self._buf.strip():
            self._put(self._buf.strip())
            self._buf = ""


def apply_board(kind, cols, rows, square_mm):
    """把界面上的标定板规格同步到标定模块。

    calib_capture 与 calib_solve 共用 BOARD/PATTERN：检测用 pattern()，几何用
    BOARD，OBJP 则是按 BOARD 烘焙出来的常量。三处都改，才不会出现"检测按新板、
    求解却按旧板"的错位。
    """
    cc.set_board(kind, cols, rows, float(square_mm))
    cs.refresh_objp()


# 引导式标定的固定步骤表。cell 是 4x3 覆盖网格里的 (列, 行)，行 0 在上。
# axis 是这一步建议的倾斜轴：yaw 偏航 / pitch 俯仰 / roll 滚转 / None 正对。
# 顺序刻意排成"中心 → 四条边 → 四角 → 补倾斜"，每一步只改一个变量，
# 用户手上动作最简单，也最容易看出哪一步卡住了。
CALIB_STEPS = [
    dict(cell=(1, 1), axis=None,  deg=0,  dist="mid",
         title="中心 · 正对", tip="这组定基准，决定整体尺度"),
    dict(cell=(2, 1), axis=None,  deg=0,  dist="near",
         title="中心 · 靠近", tip="近距离样本，管住内参的近端"),
    dict(cell=(1, 0), axis="pitch", deg=20, dist="mid",
         title="上边缘 · 向上仰", tip="板在上半幅并仰起，约束 fy 和主点 cy"),
    dict(cell=(2, 0), axis="pitch", deg=20, dist="mid",
         title="上边缘 · 向上仰", tip="换一列再来一次，让上边缘两侧都有样本"),
    dict(cell=(1, 2), axis="pitch", deg=-20, dist="mid",
         title="下边缘 · 向下俯", tip="与上边缘对称，把 cy 的另一侧也压住"),
    dict(cell=(2, 2), axis="pitch", deg=-20, dist="mid",
         title="下边缘 · 向下俯", tip="下边缘另一列"),
    dict(cell=(0, 1), axis="yaw", deg=30, dist="mid",
         title="左边缘 · 向左偏", tip="左侧加偏航，透视压缩最强烈，约束 fx 和 cx"),
    dict(cell=(3, 1), axis="yaw", deg=-30, dist="mid",
         title="右边缘 · 向右偏", tip="右侧加偏航，与左侧配成一对"),
    dict(cell=(0, 0), axis=None, deg=0, dist="mid",
         title="左上角 · 正对", tip="角落样本，给径向畸变系数提供边界条件"),
    dict(cell=(3, 0), axis=None, deg=0, dist="mid",
         title="右上角 · 正对", tip="角落样本"),
    dict(cell=(0, 2), axis=None, deg=0, dist="mid",
         title="左下角 · 正对", tip="角落样本"),
    dict(cell=(3, 2), axis=None, deg=0, dist="mid",
         title="右下角 · 正对", tip="角落样本，这一格最难摆，板别顶到边"),
    dict(cell=(1, 1), axis=None, deg=0, dist="far",
         title="中心 · 退远", tip="远距离样本，管住内参的远端"),
    dict(cell=(2, 1), axis="roll", deg=20, dist="mid",
         title="中心 · 顺时针转", tip="滚转约束相机安装的旋转偏差"),
    dict(cell=(1, 1), axis="roll", deg=-20, dist="mid",
         title="中心 · 逆时针转", tip="反向滚转，与上一步配成一对"),
]

AXIS_TXT = {"yaw": "偏航", "pitch": "俯仰", "roll": "滚转", None: ""}

# ---- 方案二：中间横带（推荐给"没有升降台"的场地）----
# 现场痛点：没有升降台，把板举到上边缘或四角要挪凳子、垫书、再对高度，
# 十几分钟过不了两三步，最后多半放弃标定。
# 关键观察：桌面上左右挪是不费力的，只有"上下抬"才是体力活。而外参精度
# 恰恰主要由"角点在画面上的横向跨度"决定（见下面 2026-11-16 的实测对比），
# 所以只保留中间一条横带、把板在左/中/右之间挪，就能拿回几乎全部精度。
#   _detect/pose_set_exp.py 实测（圆点板、0.3px 角点噪声、各 20 次重复，
#   三套步骤表都直接从本文件读，只换姿态集）：
#     全幅覆盖        旋转 0.14°（最差 0.48°） 平移 1.09mm
#     中间横带 3 列   旋转 0.36°（最差 0.84°） 平移 0.83mm
#     只摆中间        旋转 2.55°（最差 6.55°） 平移 1.51mm
# 也就是说"只在中间"会把旋转解到 2°级 —— 0.5m 处偏 2.5° 就是 22mm 错位，
# 深度贴彩色必然错边；横带 0.36° 在 0.5m 处约 3mm，够用。所以默认落在横带上。
BAND_U = {"L": 0.22, "M": 0.50, "R": 0.78}
BAND_HALF = 0.12        # 板心允许的横向偏差（归一化，约 ±154px @1280）

CALIB_STEPS_BAND = [
    dict(cell=None, u=BAND_U["M"], axis=None, deg=0, dist="mid",
         title="中间 · 正对", tip="基准组：板心对准画面中央，板面正对镜头"),
    dict(cell=None, u=BAND_U["L"], axis=None, deg=0, dist="mid",
         title="偏左 · 正对", tip="往左挪，高度不用变 —— 桌面上平移就行"),
    dict(cell=None, u=BAND_U["R"], axis=None, deg=0, dist="mid",
         title="偏右 · 正对", tip="再挪到右边，这一格把横向覆盖拉开"),
    dict(cell=None, u=BAND_U["M"], axis=None, deg=0, dist="near",
         title="中间 · 靠近", tip="往回中间并往前送，用远近换尺度"),
    dict(cell=None, u=BAND_U["M"], axis=None, deg=0, dist="far",
         title="中间 · 退远", tip="退到最远档，和上一组配成尺度差"),
    dict(cell=None, u=BAND_U["L"], axis="pitch", deg=20, dist="mid",
         title="左侧 · 上缘后仰 20°", tip="左边不动，只把上缘往后推"),
    dict(cell=None, u=BAND_U["R"], axis="pitch", deg=20, dist="mid",
         title="右侧 · 上缘后仰 20°", tip="右边同样后仰，两侧都留倾斜样本"),
    dict(cell=None, u=BAND_U["M"], axis="pitch", deg=25, dist="near",
         title="中间 · 后仰 25° · 近", tip="大俯仰配近距离，倾斜信息最足"),
    dict(cell=None, u=BAND_U["L"], axis="pitch", deg=-25, dist="mid",
         title="左侧 · 上缘前俯 25°", tip="反向俯仰，与后仰那组配对"),
    dict(cell=None, u=BAND_U["R"], axis="pitch", deg=-25, dist="mid",
         title="右侧 · 上缘前俯 25°", tip="右边再来一次前俯"),
    dict(cell=None, u=BAND_U["M"], axis="yaw", deg=30, dist="mid",
         title="中间 · 左偏 30°", tip="绕竖轴转，透视压缩最强烈，约束 fx"),
    dict(cell=None, u=BAND_U["M"], axis="yaw", deg=-30, dist="mid",
         title="中间 · 右偏 30°", tip="反向偏航，与上一组配成一对"),
    dict(cell=None, u=BAND_U["L"], axis="yaw", deg=-30, dist="near",
         title="左侧 · 右偏 30° · 近", tip="边上加偏航，边角样本最有价值"),
    dict(cell=None, u=BAND_U["R"], axis="yaw", deg=30, dist="near",
         title="右侧 · 左偏 30° · 近", tip="右边对称来一次"),
    dict(cell=None, u=BAND_U["M"], axis="roll", deg=20, dist="mid",
         title="中间 · 顺时针转 20°", tip="板面内旋转，约束相机安装的滚转"),
    dict(cell=None, u=BAND_U["M"], axis="roll", deg=-20, dist="mid",
         title="中间 · 逆时针转 20°", tip="反向滚转，与上一组配成一对"),
]

# ---- 方案三：只摆中间（最省力，精度明显下降，默认不选）----
# 板心一直留在画面正中，只改倾角和远近。实测外参旋转会解到 2°级（见上），
# 0.5m 处约 22mm 错位 —— 只在"连左右挪都不可能"时才用。
CALIB_STEPS_CENTER = [
    dict(cell=None, u=None, axis=None, deg=0, dist="mid",
         title="中间 · 正对", tip="基准组：板心对准画面中央，板面正对镜头"),
    dict(cell=None, u=None, axis=None, deg=0, dist="near",
         title="中间 · 靠近", tip="只往前送，别抬高度 —— 用远近换尺度"),
    dict(cell=None, u=None, axis=None, deg=0, dist="far",
         title="中间 · 退远", tip="退到最远档，和上一组配成尺度差"),
    dict(cell=None, u=None, axis="pitch", deg=20, dist="mid",
         title="中间 · 上缘后仰 20°", tip="位置别动，只把上缘往后推"),
    dict(cell=None, u=None, axis="pitch", deg=-25, dist="mid",
         title="中间 · 上缘前俯 25°", tip="反向俯仰，与上一组配对压住主点"),
    dict(cell=None, u=None, axis="pitch", deg=35, dist="near",
         title="中间 · 后仰 35° · 近", tip="大俯仰配近距离，倾斜信息最足"),
    dict(cell=None, u=None, axis="yaw", deg=30, dist="mid",
         title="中间 · 左偏 30°", tip="绕竖轴转，板心仍留在画面中间"),
    dict(cell=None, u=None, axis="yaw", deg=-30, dist="mid",
         title="中间 · 右偏 30°", tip="反向偏航，和上一组配成一对"),
    dict(cell=None, u=None, axis="yaw", deg=40, dist="near",
         title="中间 · 左偏 40° · 近", tip="大偏航配近距离"),
    dict(cell=None, u=None, axis="roll", deg=20, dist="mid",
         title="中间 · 顺时针转 20°", tip="板面内旋转，约束相机安装的滚转"),
    dict(cell=None, u=None, axis="roll", deg=-20, dist="near",
         title="中间 · 逆时针转 20°", tip="反向滚转，顺便换个距离"),
    dict(cell=None, u=None, axis="pitch", deg=20, dist="far",
         title="中间 · 后仰 20° · 远", tip="远距离再补一组倾斜，压住焦距"),
]

# 非覆盖方案的纵向范围（归一化 v）：不要求高度精确，只防"板飘到上下边顶掉"。
ZONE_V = (0.28, 0.72)
# "只摆中间"的横向范围
CENTER_ZONE = (0.30, 0.70, ZONE_V[0], ZONE_V[1])

PLANS = {"band": "中间横带", "center": "只摆中间", "cover": "全幅覆盖"}


def in_step_area(st, u, v, cell):
    """板当前位置满足这一步的目标区域吗。

    cover  ：按 4x3 网格的格号精确比对；
    band   ：纵向宽松（只防顶边），横向要落在目标列 ±BAND_HALF；
    center ：横向纵向都只要求落在中间带。
    cell 由调用方按 u/v 现算（抓取帧和预览帧各算一次，不能共用）。
    """
    if st.get("cell") is not None:
        return cell == st["cell"]
    if not ZONE_V[0] <= v <= ZONE_V[1]:
        return False
    if st.get("u") is None:
        return CENTER_ZONE[0] <= u <= CENTER_ZONE[1]
    return abs(u - st["u"]) <= BAND_HALF


def cell_name(cell):
    """4x3 网格里的格号 -> 人话。中间两格合并叫"中央"。

    原先这份映射在界面和采集线程里各写了一遍，采集线程那份漏了中间两格，
    落到 (1,1)/(2,1) 会说出"画面右边缘"这种错话 —— 提示指错方向，比不说
    更耽误事。现在只有这一份。
    """
    c, r = cell
    if (c, r) in ((1, 1), (2, 1)):
        return "画面中央"
    corner = {(0, 0): "左上角", (3, 0): "右上角",
              (0, 2): "左下角", (3, 2): "右下角"}.get((c, r))
    if corner:
        return "画面" + corner
    if r == 0:
        return "画面上边缘" + ("偏左" if c == 1 else "偏右")
    if r == 2:
        return "画面下边缘" + ("偏左" if c == 1 else "偏右")
    return "画面左边缘" if c == 0 else "画面右边缘"

# 引导步骤里的距离档位按标定板展开。圆点板特征跨度只有 40mm（棋盘 200mm），
# 必须靠得更近才占够像素，否则"板够大"这一关永远过不去。
BOARD_DIST = {
    "chess": dict(near="0.45 m", mid="0.6 m", far="0.85 m"),
    "circles": dict(near="0.35 m", mid="0.45 m", far="0.60 m"),
}

# 板宽占画面比例的合格区间。圆点板特征跨度只有 40mm，按彩色 1280 宽、
# fx≈1000 换算：0.45m 处约 6.9%、0.60m 处 5.2%、0.78m 处 4.1%。
# 下限取 0.04 对应约 0.78m —— 覆盖引导最远档（0.60m）且留 30% 余量；
# 再往下放会让板退到一两米外也算"够大"，角点精度就没保证了。
BOARD_SIZE_MIN = {"chess": 0.20, "circles": 0.04}
SIZE_MAX = 0.78


def calib_steps(kind="chess", plan="cover"):
    """按标定板类型和引导方案展开步骤表，把距离档位换成具体米数"""
    d = BOARD_DIST.get(kind, BOARD_DIST["chess"])
    table = {"center": CALIB_STEPS_CENTER,
             "band": CALIB_STEPS_BAND}.get(plan, CALIB_STEPS)
    return [dict(s, dist=d[s["dist"]]) for s in table]


def _note_text(kind="chess", plan="band"):
    """标定配置对话框底部的说明文字，随标定板类型/方案切换"""
    if plan == "band":
        head = (f"「中间横带」共 {len(CALIB_STEPS_BAND)} 步：板始终保持同一高度，\n"
                "只在左/中/右之间平移，再叠加倾角和远近 —— 不用升降台。\n"
                "画面里的竖带框的是**板心**（绿色点），不是整块板 —— 板比带宽\n"
                "很正常，对齐绿点就行。实测外参旋转误差 0.36°（全幅覆盖 0.14°）。\n")
    elif plan == "center":
        head = (f"「只摆中间」共 {len(CALIB_STEPS_CENTER)} 步：板一直留在画面正中，\n"
                "只改倾角和远近。最省力，但角点横向跨度为 0，外参旋转会解到\n"
                "2.5°级（0.5m 处约 22mm 错位）—— 只在完全没法左右挪时用。\n")
    else:
        head = (f"「全幅覆盖」共 {len(CALIB_STEPS)} 步，顺序固定：中心 → 四条边 → 四角 → 补倾斜。\n"
                "每步在画面和面板上标出该摆的位置；摆好、板静止后\n"
                "点『采集本步』（或按空格）才会保存并进入下一步。\n")
    if kind == "circles":
        return head + ("圆点板填 OpenCV 的 patternSize：CGB-020 是 4 / 5，间距 20mm。\n"
                       "板面只有棋盘的 1/5，站位要比棋盘近，按提示距离来。")
    return head + "内角点数与方格边长必须与打印件一致。"


class CalibWorker(threading.Thread):
    """引导式标定：IR（深度相机视角）+ 彩色成对采集，按步骤走完自动求解。

    与 calib_capture.py 的区别：不弹 OpenCV 窗口。取帧、角点检测、绘制全在
    这个后台线程里做，成品 BGR 图交给界面 blit；主线程只管显示，不会被
    角点检测（720p 约 20~40ms）卡住。

    引导逻辑也在这里：每帧评估一次"板在哪一格、多大、倾斜多少、稳不稳"，
    结果实时交给界面显示。采集由用户点按钮/按空格触发，不由程序自动抓，
    这样不会在手还没松开板的时候就把帧存进去。
    """

    # 判定为"可用图像"的阈值（SIZE_MIN 默认对应棋盘格，
    # 换成圆点板时在 __init__ 里按 BOARD_SIZE_MIN 覆盖）
    SIZE_MIN = 0.20        # 板宽占画面宽度的下限
    SIZE_MAX = 0.78        # 上限（太满容易顶边）
    STEADY_PX = 0.010      # 板中心连续帧抖动上限（归一化）
    TILT_MIN = 12.0        # 倾斜轴至少转这么多度才算"有倾斜"

    # "与已采姿态足够不同"的阈值。与任何一组已采姿态比，至少要有一个维度
    # 超出下面的量，才算新姿态 —— 否则板不动也会被当成新的一组采进去，
    # 白占一张名额，还让求解退化（重投影残差漂亮但内参发散）。
    NOVEL_POS = 0.055      # 板中心位移（归一化，约 70px @1280）
    NOVEL_SIZE = 0.045     # 板宽占画面比例的变化（上限，小尺寸板按比例收紧）
    NOVEL_ANG = 7.0        # 任一轴角度变化（度）

    # 引导预览的检测宽度。缩图能把每帧检测压到 10ms 级，但圆点板很小 ——
    # 彩色 720p、0.6m 处圆直径只有约 17px，缩到 760 宽后剩 10px，接近
    # findCirclesGrid 的下限。实测（合成圆阵，圆径/噪声接近实物）：
    #   ≤750mm 缩到 760 仍 6/6 全中，900mm 掉到 2/6
    # 圆点板放宽到 900，多花的几毫秒换取余量。
    DETECT_W = {"chess": 760, "circles": 900}

    def __init__(self, cfg, put):
        super().__init__(daemon=True)
        self.cfg = cfg
        self._put = put
        self.ir = None
        self.color = None
        self.running = True
        self.trigger = threading.Event()
        self.phase = "starting"      # starting/capturing/solving/done/stopped/error
        self.msg = "准备中 …"
        self.saved = 0
        self.tried = 0
        self.last = None             # (ir_vis_bgr, color_vis_bgr)
        self.error = None
        self.result = None
        self.json_path = None
        self.ir_mode = ""
        self.color_mode = ""
        self.cells = set()           # 已采集姿态落在 4x3 网格的哪些格
        self._fp = None
        self._csv = None

        # ---- 引导 ----
        kind = (self.cfg.get("board") or ("chess",))[0]
        self.plan = self.cfg.get("plan", "cover")
        self.steps = calib_steps(kind, self.plan)
        self.SIZE_MIN = BOARD_SIZE_MIN.get(kind, self.SIZE_MIN)
        self.detect_w = self.DETECT_W.get(kind, 760)
        self.step_idx = 0
        self.live = None             # 最近一次评估结果 dict
        # 采集完全由用户点按钮 / 按空格触发。这两项只用来告诉界面
        # "现在能不能采"，不参与自动抓帧 —— 自动抓会在手还没放开时
        # 就攒够静止时间，采到正在移动的板。
        self.ready = False           # 条件是否全部满足
        self.stability = 0.0         # 0~1，越接近 1 越静止
        self.stable_sec = max(0.3, float(cfg.get("hold", 0.5)))
        self._last_grab = 0.0
        self.trigger_src = ""        # 谁触发了这次采集（按钮/空格/自动）
        self.paused = False          # 用户暂停 / 跳过本步时置位
        self.skip = False
        self._hist = []              # [(t, u, v)] 板中心历史，用于判稳
        self.taken = []              # 已采纳姿态摘要，用于"必须换姿态"校验

    # ---------- 流 ----------
    def _open_streams(self):
        self.msg = "启动 IR 流 …"
        self.ir = cp.DepthSource(use_ir=True)
        self.ir.start()
        iw, ih, ifps, ifmt = self.ir.mode
        self.ir_mode = f"{iw}×{ih} {cp.PIXFMT.get(ifmt, ifmt)}"
        self._put(f"IR 流就绪：{self.ir_mode} @{ifps}fps（与深度共镜头）")

        self.msg = "启动彩色流 …"
        if self.cfg["backend"] == "ffmpeg":
            self.color = cp.ColorSourceFFmpeg(self.cfg["w"], self.cfg["h"], 30)
        else:
            idx = cp.find_orbbec_camera(verbose=False)
            if idx is None:
                raise RuntimeError("未找到可用彩色相机")
            self.color = cp.ColorSource(idx, self.cfg["w"], self.cfg["h"], 30)
        self.color.start()
        cw, ch, cf = self.color.mode
        self.color_mode = f"{cw}×{ch}"
        self._put(f"彩色流就绪：{cw}×{ch} @{cf}fps（{self.cfg['backend']}）")
        time.sleep(0.8)

    def _decode_color(self):
        c = self.color.latest() if self.color else None
        if c is None:
            return None
        data, ts = c
        if getattr(self.color, "is_jpeg", False):
            return cp.ColorSourceFFmpeg.decode(data), ts
        return data, ts

    # ---------- 预览 + 实时评估 ----------
    def _estimate_pose(self, pts):
        """由"物面坐标 -> 图像点"的投影拟合估三轴倾角（度，带正负）。

        原理：板的平面成像近似为 A ≈ s·K·[r1 r2]，对仿射部分 A 做 SVD 后
        σ2/σ1 就是透视压缩量（cos 倾角），压缩方向（V 的最小奇异向量）
        决定是绕横轴还是竖轴转。

        这么做的好处：不依赖"第几个点在第几行"，也不依赖点集凸包是
        四边形 —— 非对称圆阵的凸包是六边形，用"矩形四角找最近点"会挑
        错角。棋盘和圆点板走同一套几何。

        符号另算：SVD 只给绝对值，±20° 会读成同一个数。而"只在中间摆"
        的方案里，一对 ±倾斜 位置相同、尺寸相同，若角度也读成一样，新颖度
        校验就会判"跟已采那组太像"，人怎么扭都过不去。所以再用单应矩阵的
        分母行求"沿压缩轴往哪边变远"，给出正负：远端那一侧成像更小，方向
        就代表板的哪条边贴近/远离相机。

        精度只到几度，用于"倾斜够不够"的粗判完全够用，不作为硬门槛。
        """
        p = pts.reshape(-1, 2).astype(np.float64)
        obj = cs.OBJP[:, :2].astype(np.float64)
        if len(p) != len(obj) or len(p) < 6:
            return 0.0, 0.0, 0.0          # 物面坐标没跟上板型，不硬猜
        X = np.hstack([obj, np.ones((len(obj), 1))])
        sol, *_ = np.linalg.lstsq(X, p, rcond=None)
        A = sol[:2].T                      # 2x2, 物面(mm) -> 图像(px)
        try:
            _U, S, Vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0
        if S[0] < 1e-9 or S[1] < 1e-9:
            return 0.0, 0.0, 0.0
        tilt = float(np.degrees(np.arccos(np.clip(S[1] / S[0], 0.0, 1.0))))
        v = Vt[1]                                    # 板坐标下的压缩方向
        a = abs(np.degrees(np.arctan2(v[1], v[0])))
        sgn = self._tilt_sign(obj, p, v)
        tilt *= sgn
        yaw, pitch = (tilt, 0.0) if a < 45.0 else (0.0, tilt)
        ex = A @ np.array([1.0, 0.0])                # 板 x 轴在图像里的方向
        roll = float(np.degrees(np.arctan2(ex[1], ex[0])))
        if roll > 90.0:
            roll -= 180.0
        elif roll < -90.0:
            roll += 180.0
        return yaw, pitch, roll

    @staticmethod
    def _tilt_sign(obj, p, vdir):
        """沿压缩轴哪一头更远（+1/-1）。远端成像更小，用单应矩阵判定。

        单应 H 满足 [u,v,w]ᵀ = H·[x,y,1]ᵀ，其中 w 是相机系深度（正比），
        所以 d w/ds = (H20, H21)·vdir 的符号就说明沿 vdir 走是变远还是变近。
        H 差一个尺度，先把 w 在板心处校成正值，符号才有意义。

        vdir 来自 SVD，本身差一个正负号（LAPACK 只保证正交），必须先把
        它规整到"主分量朝正"再算 —— 否则同一块板在两个距离上会读出相反的
        符号，±倾斜又变成分不开。
        """
        vd = np.array(vdir, dtype=float)
        if abs(vd[0]) >= abs(vd[1]):
            if vd[0] < 0:
                vd = -vd
        elif vd[1] < 0:
            vd = -vd
        try:
            H, _ = cv2.findHomography(
                obj.astype(np.float32), p.astype(np.float32), 0)
        except cv2.error:
            return 1.0
        if H is None or not np.all(np.isfinite(H)):
            return 1.0
        c = obj.mean(axis=0)
        w0 = float(H[2, 0] * c[0] + H[2, 1] * c[1] + H[2, 2])
        if abs(w0) < 1e-12:
            return 1.0
        if w0 < 0:                        # 尺度不定，翻成"板心 w>0"
            H = -H
            w0 = -w0
        g = float(H[2, 0] * vd[0] + H[2, 1] * vd[1])
        return 1.0 if g >= 0 else -1.0

    def _summary(self, cpts, w, h):
        """从全分辨率角点算一组姿态摘要：位置、占宽比、三轴倾角"""
        p = cpts.reshape(-1, 2)
        yaw, pitch, roll = self._estimate_pose(cpts)
        return dict(u=float(p[:, 0].mean()) / w,
                    v=float(p[:, 1].mean()) / h,
                    size=float(p[:, 0].max() - p[:, 0].min()) / w,
                    yaw=yaw, pitch=pitch, roll=roll)

    def _novelty(self, L):
        """当前姿态与已采姿态的差异。

        返回 (是否算新姿态, 最小差距 0~1+, 差距明细)。每张已采图都在
        位置/尺寸/倾角三个维度上和当前姿态比一次，取最像的那组；任何
        一个维度达到 1 就算不同。距离阈值按"板中心挪半个网格、尺寸差
        一成、角度差 7 度"来定 —— 比这更小的变化对求解没有新信息，采
        进去只会让残差虚低、内参发散。
        """
        if not self.taken:
            return True, 9.9, {}
        best = None
        for t in self.taken:
            dp = dict(h=abs(L.get("u", 0.0) - t["u"]) / self.NOVEL_POS,
                      v=abs(L.get("v", 0.0) - t["v"]) / self.NOVEL_POS)
            ds = abs(L.get("size", 0.0) - t["size"]) / self._size_eps(
                L.get("size", 0.0), t["size"])
            da = max(abs(L.get(a, 0.0) - t[a])
                     for a in ("yaw", "pitch", "roll")) / self.NOVEL_ANG
            g = max(dp["h"], dp["v"], ds, da)
            if best is None or g < best[0]:
                best = (g, dict(pos=dp, size=ds, ang=da, ref=t))
        return best[0] >= 1.0, best[0], best[1]

    def _size_eps(self, a, b):
        """尺寸维度的"算不同"阈值。

        NOVEL_SIZE 是按棋盘（占宽 0.30~0.40）定的绝对值。圆点板只占 5%~9%，
        用同一个绝对值等于要求"尺寸变化接近一倍"，这一维就形同失效 ——
        而引导第 13 步（中心·退远）恰恰只靠距离变化，会被判成"跟已采太像"
        永远采不进去。所以尺寸阈值随板本身的大小收紧，取两者较小：
        棋盘仍走 0.045（不变），圆点板约 0.014（0.45m→0.60m 差值 0.017 生效）。

        中间带方案再收紧一档（20%→13%）：位置维度在方案里恒为 0，剩下能
        拉开差距的只有"倾角"和"远近"两项，尺寸这一维就成了主力。按 20%
        算，0.45m→0.35m 的尺寸差只到 1.1 倍阈值，手一抖就掉回"太像"，
        人明明照做了却过不去。横带同理（那里同一列上也会连着采 near/far）。
        """
        frac = {"cover": 0.20, "band": 0.15}.get(
            getattr(self, "plan", "cover"), 0.13)
        return min(self.NOVEL_SIZE, frac * max(a, b, 1e-6))

    def _evaluate(self, cok, cpts, iok, ipts, shape):
        h, w = shape[0], shape[1]
        now = time.time()
        L = dict(cok=bool(cok), iok=bool(iok), cell=None, u=0.0, v=0.0,
                 size=0.0, yaw=0.0, pitch=0.0, roll=0.0,
                 steady=False, jitter=9.9, stability=0.0,
                 novel=True, gap=9.9, gapdims={}, src="color")
        if cok:
            L.update(self._summary(cpts, w, h))
            u, v = L["u"], L["v"]
            L["cell"] = (min(3, max(0, int(u * 4))), min(2, max(0, int(v * 3))))
            self._hist.append((now, u, v))
            L["novel"], L["gap"], L["gapdims"] = self._novelty(L)
        elif iok and self.ir is not None:
            # 彩色暂时没检出时，用 IR 的位置兜底，界面照样能提示往哪边挪。
            # 注意 Astra 的 IR 流是水平镜像的（与 calib_solve 同一已知特性），
            # u 必须翻回来，否则箭头/提示会指着反方向。
            p = ipts.reshape(-1, 2)
            iw, ih = self.ir.mode[0], self.ir.mode[1]
            u = 1.0 - float(p[:, 0].mean()) / iw
            v = float(p[:, 1].mean()) / ih
            L.update(u=u, v=v, src="ir",
                     cell=(min(3, max(0, int(u * 4))), min(2, max(0, int(v * 3)))))
        # 稳定性只看最近 stable_sec 内的帧。板一旦移出画面就不再追加，
        # 窗口自然过期、判为"在动"，不会拿过期数据说"已经稳了"。
        cutoff = now - self.stable_sec
        while self._hist and self._hist[0][0] < cutoff:
            self._hist.pop(0)
        if len(self._hist) >= 3:
            span = now - self._hist[0][0]
            us = [e[1] for e in self._hist]
            vs = [e[2] for e in self._hist]
            rng = max(max(us) - min(us), max(vs) - min(vs))
            L["jitter"] = rng
            L["stability"] = max(0.0, min(1.0, 1.0 - rng / self.STEADY_PX))
            # 既要抖得小，也要真的持续了一段时间，避免"刚好停一瞬"就过
            L["steady"] = (rng < self.STEADY_PX
                           and span >= self.stable_sec * 0.7)
        self.live = L
        return L

    def _preview(self):
        got = self._decode_color()
        irf = self.ir.latest() if self.ir else None
        if got is None or irf is None:
            return
        img, _ = got
        if img is None:
            return
        ig = cc.ir_to_8bit(irf[0])
        # 引导判定只看位置/尺寸/倾斜，缩到 760 宽再检测，省一半多时间；
        # 真正抓取时 _grab_once 还是用全分辨率，精度不受影响。
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        dw = float(self.detect_w)
        sc = dw / w if w > dw else 1.0
        small = (cv2.resize(gray, None, fx=sc, fy=sc,
                            interpolation=cv2.INTER_AREA) if sc < 1.0 else gray)
        cok, cpts = cc.detect_corners(small)
        if cok and sc < 1.0:
            cpts = (cpts / sc).astype(np.float32)
        iok, ipts = cc.detect_corners(ig)
        self._evaluate(cok, cpts, iok, ipts, (h, w))
        vis_c = cc.draw_found(img, cok, cpts, "COLOR", cok)
        vis_i = cc.draw_found(ig, iok, ipts, "IR", iok)
        self._overlay(vis_c)
        self.last = (vis_i, vis_c)

    def _overlay(self, vis):
        """画面叠加：目标区域 + 板当前中心 + 指向箭头，右下角另附进度

        三种方案的"目标区域"画法不同：覆盖方案画格子角括号，横带画一条
        竖向虚线带（只约束左右），只摆中间画一个中间虚线框。
        """
        h, w = vis.shape[0], vis.shape[1]
        L = self.live or {}
        n = len(self.steps)
        if self.step_idx < n:
            st = self.steps[self.step_idx]
            col = (60, 190, 255)          # BGR 琥珀，和界面强调色一致
            ln = max(16, int(min(w, h) * 0.055))
            if st["cell"] is None:
                # 横带/只摆中间：画"板心带"——两条通高竖虚线 + 淡淡的竖条。
                # 绝不画闭合矩形：判定只看板心，而板在要求距离上往往比带宽
                # 更宽，闭合框会让人误以为"整块板要放进框里"，怎么摆都进不去。
                tu = st.get("u")
                if tu is None:
                    u0, u1 = CENTER_ZONE[0], CENTER_ZONE[1]
                else:
                    u0, u1 = tu - BAND_HALF, tu + BAND_HALF
                x1, x2 = int(u0 * w), int(u1 * w)
                ov = vis.copy()
                cv2.rectangle(ov, (x1, 0), (x2, h), col, -1)
                cv2.addWeighted(ov, 0.10, vis, 0.90, 0, vis)
                for x in (x1, x2):
                    for y in range(0, h, 34):
                        cv2.line(vis, (x, y), (x, min(y + 22, h)), col, 2)
                cv2.putText(vis, "board-center band", (x1 + 6, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1,
                            cv2.LINE_AA)
                cy = int(float(L.get("v", 0.5)) * h) if L.get("cok") else h // 2
                cx = (x1 + x2) // 2
            else:
                c, r = st["cell"]
                cx, cy = (c + 0.5) * w / 4.0, (r + 0.5) * h / 3.0
                x1, y1 = int(c * w / 4), int(r * h / 3)
                x2, y2 = int((c + 1) * w / 4) - 1, int((r + 1) * h / 3) - 1
                # 只画四个角括号：板本身不被画花，位置范围却一眼可见
                for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                                       (x1, y2, 1, -1), (x2, y2, -1, -1)):
                    cv2.line(vis, (px, py), (px + dx * ln, py), col, 2)
                    cv2.line(vis, (px, py), (px, py + dy * ln), col, 2)
                cv2.drawMarker(vis, (int(cx), int(cy)), col,
                               cv2.MARKER_CROSS, 24, 1)
            u, v = L.get("u"), L.get("v")
            if u is not None and (L.get("cok") or L.get("src") == "ir"):
                px, py = int(u * w), int(v * h)
                if px > 0 and py > 0:
                    cv2.circle(vis, (px, py), 8, (0, 0, 0), -1)
                    cv2.circle(vis, (px, py), 5, (120, 240, 120), -1)
                    if abs(px - cx) + abs(py - cy) > 45:
                        cv2.arrowedLine(vis, (px, py), (int(cx), int(cy)),
                                        (255, 255, 255), 2, tipLength=0.16)

        if self.plan != "cover":
            # 非覆盖方案没有"覆盖哪几格"可言，进度按步数点阵显示
            s, g = 9, 3
            x0 = w - n * (s + g) - 10
            y0 = h - s - 24
            for i in range(n):
                x = x0 + i * (s + g)
                if i == self.step_idx:
                    cv2.rectangle(vis, (x - 1, y0 - 1), (x + s + 1, y0 + s + 1),
                                  (60, 190, 255), 1)
                cv2.rectangle(vis, (x, y0), (x + s, y0 + s),
                              (110, 220, 140) if i < self.saved else (58, 66, 80), -1)
            cv2.putText(vis, f"step {self.saved}/{n}", (x0 - 2, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 165, 185), 1)
        else:
            # 右下角 4x3 网格：绿=已采，亮框=当前该摆的格
            s, g = 11, 3
            x0 = w - 4 * (s + g) - 10
            y0 = h - 3 * (s + g) - 22
            tgt = self.steps[self.step_idx]["cell"] if self.step_idx < n else None
            for ry in range(3):
                for cx in range(4):
                    x, y = x0 + cx * (s + g), y0 + ry * (s + g)
                    on = (cx, ry) in self.cells
                    if (cx, ry) == tgt:
                        cv2.rectangle(vis, (x - 1, y - 1), (x + s + 1, y + s + 1),
                                      (60, 190, 255), 1)
                    cv2.rectangle(vis, (x, y), (x + s, y + s),
                                  (110, 220, 140) if on else (58, 66, 80), -1)
            cv2.putText(vis, f"cover {len(self.cells)}/12", (x0 - 2, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 165, 185), 1)

    # ---------- 采一组 ----------
    def _grab_once(self, forced=False):
        if self.step_idx >= len(self.steps):
            return
        now = time.time()
        src = self.trigger_src or "手动"
        self.trigger_src = ""
        if not forced and now - self._last_grab < 0.4:
            return                        # 连点防抖：同一次尝试不重复触发
        self._last_grab = now
        got = self._decode_color()
        irf = self.ir.latest() if self.ir else None
        if got is None or irf is None or got[0] is None:
            self.msg = "数据帧未就绪，本次跳过"
            return
        img, cts = got
        ir16, its = irf[0], irf[1]
        ig = cc.ir_to_8bit(ir16)
        cok, cpts = cc.detect_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        iok, ipts = cc.detect_corners(ig)
        self.tried += 1

        vis_c = cc.draw_found(img, cok, cpts, "COLOR", cok)
        vis_i = cc.draw_found(ig, iok, ipts, "IR", iok)

        # 用本帧全分辨率角点重算一遍再决定要不要存 —— 预览帧和抓取帧
        # 相隔几十毫秒，板若还在挪，两边的判定可能不一致。
        sm = self._summary(cpts, img.shape[1], img.shape[0]) if cok else None
        why = None if forced else self._reject_reason(sm)
        if cok and iok and why:
            self.msg = why
            self._put(f"[未采集·{src}] {why}")
            self.ready = False
            self._overlay(vis_c)
            self.last = (vis_i, vis_c)
            return

        if cok and iok:
            tag = f"{self.saved + 1:03d}"
            fn_c, fn_i = f"calib_{tag}_color.png", f"calib_{tag}_ir.png"
            cp.imwrite_u(os.path.join(self.cfg["out"], fn_c), img)
            cp.imwrite_u(os.path.join(self.cfg["out"], fn_i), ir16)
            self._row([tag, fn_c, fn_i, f"{cts:.4f}", f"{its:.4f}", "1", "1"])
            self.saved += 1
            self.taken.append(sm)          # 记下来，后面几步都得跟它不一样
            self.cells.add((min(3, int(sm["u"] * 4)), min(2, int(sm["v"] * 3))))
            n = len(self.steps)
            title = self.steps[self.step_idx]["title"] if self.step_idx < n else "—"
            self.msg = f"第 {self.step_idx + 1} 步完成：{title}"
            self._put(f"[{tag}] 第 {self.step_idx + 1}/{n} 步  {title}  ✓彩色 ✓IR"
                      f"   （覆盖 {len(self.cells)}/12，触发：{src}）")
            self.step_idx += 1
            self.ready = False
            self._hist.clear()
            if self.step_idx >= len(self.steps):
                self.msg = "全部步骤完成，正在求解 …"
        else:
            why = "、".join(w for w, ok in (("彩色未检出", cok), ("IR 未检出", iok)) if not ok)
            fd = os.path.join(self.cfg["out"], "_failed")
            os.makedirs(fd, exist_ok=True)
            cp.imwrite_u(os.path.join(fd, "last_fail_color.png"), img)
            cp.imwrite_u(os.path.join(fd, "last_fail_ir.png"), ig)
            self.msg = f"{why}：调整位置或光照后再点采集"
            self._put(f"[未采集] {why}（存 _failed/ 便于排查）")

        self._overlay(vis_c)
        cv2.putText(vis_c, f"{self.saved}/{len(self.steps)}",
                    (14, vis_c.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (90, 235, 150) if (cok and iok) else (80, 120, 240), 2)
        self.last = (vis_i, vis_c)

    # ---------- 引导判定 ----------
    def _step_state(self):
        """返回 (当前步骤, 六项状态, 是否可采)。

        硬条件四个：两路都检出、板落在目标格、板大小合适、姿态与已采的
        不同。倾斜仍是加分项 —— 估计值有误差，卡太死会让人一直过不去；
        但"必须变一点"是硬的，否则同一位置重复采会把标定采废。
        """
        st = self.steps[min(self.step_idx, len(self.steps) - 1)]
        L = self.live or {}
        s = dict(color=bool(L.get("cok")), ir=bool(L.get("iok")),
                 pos=in_step_area(st, L.get("u", -1.0), L.get("v", -1.0),
                                  L.get("cell")),
                 size=False, tilt=True,
                 moved=bool(L.get("novel", not self.taken)))
        if not (s["color"] and s["ir"]):
            return st, s, False
        s["size"] = self.SIZE_MIN <= L.get("size", 0.0) <= self.SIZE_MAX
        if st["axis"]:
            s["tilt"] = abs(L.get(st["axis"], 0.0)) >= self.TILT_MIN
        return st, s, (s["color"] and s["ir"] and s["pos"] and s["size"]
                       and s["moved"])

    def _advance(self, now):
        """只更新"能不能采"的状态，绝不自动抓帧。

        自动抓的问题：手还在挪板时，只要最近几帧位移小于阈值就被判成
        静止，倒计时一攒够就采 —— 采到的正是用户说的"还没稳定"那张。
        改成由用户点按钮/按空格确认，这里只负责算状态给界面用。
        """
        if self.skip:
            self.skip = False
            self._put(f"跳过第 {self.step_idx + 1} 步")
            self.step_idx += 1
            self._hist.clear()
            self.ready = False
            return
        _st, _s, ok = self._step_state()
        steady = bool((self.live or {}).get("steady"))
        self.ready = bool(ok and steady)
        self.stability = float((self.live or {}).get("stability", 0.0))

    def _reject_reason(self, sm):
        """手动采集前把一遍关，返回拒绝原因；可以采则返回 None。

        用本帧全分辨率角点重算位置和尺寸（预览帧与抓取帧相隔几十毫秒，
        板可能已经动了），稳定性只能取最近的滑窗结果。
        """
        st = self.steps[min(self.step_idx, len(self.steps) - 1)]
        if sm is None:
            return "没检出标定板：把板移到画面中间、正对镜头"
        cell = (min(3, max(0, int(sm["u"] * 4))), min(2, max(0, int(sm["v"] * 3))))
        if not in_step_area(st, sm["u"], sm["v"], cell):
            return f"板不在{self._step_where(st)}：先按提示挪到目标位置"
        if sm["size"] < self.SIZE_MIN:
            return "板太小：靠近一些再采"
        if sm["size"] > self.SIZE_MAX:
            return "板太满：退远一些再采"
        fresh, gap, dims = self._novelty(sm)
        if not fresh:
            return self._similar_reason(st, dims, gap)
        if not bool((self.live or {}).get("steady")):
            return "画面还在动：手离开板，等它彻底静止再采"
        return None

    @staticmethod
    def _step_where(st):
        """这一步该把板摆在哪：横带按列名，只摆中间固定，覆盖按格号"""
        if st.get("cell") is not None:
            return cell_name(st["cell"])
        u = st.get("u")
        if u is None:
            return "画面中间带"
        return "画面偏左那一列" if u < 0.4 else (
            "画面偏右那一列" if u > 0.6 else "画面中间那一列")

    def _similar_reason(self, st, dims, gap=0.0):
        """与已采组太像时，指出最该改变的那个维度"""
        pos = dims.get("pos") or {}
        if self.plan == "center":
            # 中间带方案下位置本来就是固定的，位置差恒为 0；拿它当"最该改
            # 的维度"会让人把板往边上挪 —— 恰好是这一步不该做的事。
            cands = [(dims.get("size", 9.0), "size"), (dims.get("ang", 9.0), "ang")]
        else:
            cands = [(pos.get("h", 9.0), "h"), (pos.get("v", 9.0), "v"),
                     (dims.get("size", 9.0), "size"),
                     (dims.get("ang", 9.0), "ang")]
        weakest = min(cands)[1]
        if st["axis"] and weakest in ("ang", "h", "v"):
            return f"跟已采那组太像：{AXIS_TXT[st['axis']]}再转大一些"
        if weakest == "size":
            return f"跟已采那组太像：把板移到约 {st['dist']} 再采"
        if self.plan == "center":
            return f"跟已采那组太像：{AXIS_TXT[st['axis']] if st['axis'] else '倾角'}再明显一些"
        return "跟已采那组太像：把板挪开半格以上"

    def _row(self, values):
        if self._csv is None:
            path = os.path.join(self.cfg["out"], "pairs.csv")
            new = not os.path.exists(path)
            self._fp = open(path, "a", newline="", encoding="utf-8")
            self._csv = csv.writer(self._fp)
            if new:
                self._csv.writerow(["index", "color_png", "ir_png", "color_ts",
                                    "ir_ts", "color_ok", "ir_ok"])
        self._csv.writerow(values)
        self._fp.flush()

    # ---------- 求解 ----------
    def _solve(self):
        self.phase = "solving"
        self.msg = "正在求解内外参 …"
        self._close_streams()          # 先让出带宽，求解纯 CPU
        apply_board(*self.cfg["board"])
        self._put("─" * 34)
        self._put(f"开始求解：{self.saved} 组有效姿态")
        w = _LineWriter(self._put)
        try:
            with contextlib.redirect_stdout(w):
                res = cs.run_solve(cs.load_pairs(self.cfg["out"]), self.cfg["out"])
        except SystemExit as e:
            raise RuntimeError(str(e))
        finally:
            w.flush()
        self.result = res
        self.json_path = os.path.join(self.cfg["out"], "stereo_calib.json")
        self.phase = "done"
        self.msg = "标定完成"

    def _close_streams(self):
        if self.color is not None:
            try:
                self.color.close()
            except Exception:
                pass
            self.color = None
        if self.ir is not None:
            try:
                self.ir.release()
            except Exception:
                pass
            self.ir = None
        if self._fp is not None:
            try:
                self._fp.close()
            except Exception:
                pass
            self._fp = None

    # ---------- 主循环 ----------
    def run(self):
        try:
            self._open_streams()
            self.phase = "capturing"
            n = len(self.steps)
            self.msg = self.steps[0]["title"]
            self._put(f"引导标定开始：共 {n} 步，摆好板后点『采集本步』或按空格")
            self._put("程序不会自动采集 —— 每次保存都由你确认，日志会记下触发方式")
            prev = 0.0
            while self.running and self.step_idx < n:
                now = time.time()
                if now - prev > 0.12:      # 评估约 8fps，再快就只是抢 CPU
                    prev = now
                    self._preview()
                if self.trigger.is_set():
                    # 用户点『采集本步』/按空格：走一遍完整校验再存
                    self.trigger.clear()
                    self._grab_once()
                elif not self.paused:
                    self._advance(now)
                time.sleep(0.02)
            if self.running and self.step_idx >= n:
                self._solve()
            else:
                self.phase = "stopped"
                self.msg = f"已停止，采到 {self.saved}/{n} 步"
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            self.phase = "error"
            try:
                self._put("标定线程异常：")
                for ln in traceback.format_exc().strip().splitlines()[-6:]:
                    self._put("  " + ln)
            except Exception:
                pass
        finally:
            self._close_streams()


class CameraApp:
    TICK = 40          # 预览刷新周期 ms
    RESOLUTIONS = ["1280x720", "1920x1080", "640x480"]

    def __init__(self, root):
        self.root = root
        self.depth = None
        self.color = None
        self.connected = False
        self.connecting = False
        self.auto_on = False
        self.interval = 2.0
        self.saved = 0
        self.seq = 1
        self.out_dir = ""
        self._cfp = None
        self._csvw = None
        self._busy = False
        self._conn_result = None
        self._last_auto = 0.0
        self._tk = 0
        self._fps_t = time.time()
        self._fps_d = 0.0
        self._fps_c = 0.0
        self._dcount = 0
        self._ccount = 0
        self._fps_hint_shown = False
        self._connected_at = 0.0
        self._stat = {"diff": None, "pct": None, "med": None}
        self.calib = None           # CalibWorker
        self.calib_mode = False
        self._wiz_drawn = None      # 引导示意图上次绘制的 (步骤, 覆盖, 已采)
        self._logq = queue.Queue()  # 后台线程 -> 主线程的日志管道

        root.title("Astra Pro Plus · 双流同步采集")
        root.configure(bg=C_BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(1560, sw - 60), min(940, sh - 90)
        root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 2)}")
        root.minsize(1120, 720)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._init_style()
        self._build()
        self._bind_keys()

        self.log(f"界面版本：{BUILD}")
        self.log("界面就绪，正在自动连接相机 …")
        self.root.after(300, self.connect)
        self.root.after(self.TICK, self._tick)

    # ------------------------------------------------------------ 样式 ---
    def _init_style(self):
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except Exception:
            pass
        st.configure("Dark.TCombobox",
                     fieldbackground=C_CARD2, background=C_CARD2,
                     foreground=C_FG, arrowcolor=C_SUB, bordercolor=C_LINE,
                     lightcolor=C_CARD2, darkcolor=C_CARD2,
                     selectbackground=C_CARD2, selectforeground=C_FG,
                     padding=4)
        st.map("Dark.TCombobox",
               fieldbackground=[("readonly", C_CARD2)],
               foreground=[("readonly", C_FG)],
               selectbackground=[("readonly", C_CARD2)],
               selectforeground=[("readonly", C_FG)],
               bordercolor=[("focus", C_ACCENT)])
        self.root.option_add("*TCombobox*Listbox.background", C_CARD2)
        self.root.option_add("*TCombobox*Listbox.foreground", C_FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", C_ACCENT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#04121f")
        self.root.option_add("*TCombobox*Listbox.font", F_MD)

        st.configure("Calib.Horizontal.TProgressbar",
                     troughcolor=C_CARD2, background=C_ACCENT,
                     bordercolor=C_CARD2, lightcolor=C_ACCENT,
                     darkcolor=C_ACCENT, thickness=6)

    # ------------------------------------------------------------ 构件 ---
    def _btn(self, parent, text, cmd, kind="ghost", **kw):
        bg = {"primary": C_ACCENT, "ghost": C_CARD2, "danger": "#3a2330"}[kind]
        fg = "#04121f" if kind == "primary" else (C_ERR if kind == "danger" else C_FG)
        abg = {"primary": "#6cbcff", "ghost": "#2b3648", "danger": "#4a2b3a"}[kind]
        b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                      activebackground=abg, activeforeground=fg,
                      relief="flat", bd=0, highlightthickness=0,
                      cursor="hand2", padx=12, pady=6,
                      font=(FT, -13, "bold" if kind == "primary" else "normal"),
                      disabledforeground=C_DIM, **kw)
        return b

    def _section(self, title, parent=None):
        wrap = tk.Frame(parent or self.panel, bg=C_PANEL)
        wrap.pack(fill="x", padx=14, pady=(10, 0))
        tk.Label(wrap, text=title, font=F_SEC, fg=C_DIM, bg=C_PANEL,
                 anchor="w").pack(fill="x", pady=(0, 5))
        box = tk.Frame(wrap, bg=C_CARD, highlightthickness=1,
                       highlightbackground=C_LINE)
        box.pack(fill="x")
        inner = tk.Frame(box, bg=C_CARD)
        inner.pack(fill="x", padx=11, pady=10)
        if not hasattr(self, "_wraps"):
            self._wraps = {}
        self._wraps[title] = wrap
        return inner

    def _preview_card(self, parent, title):
        card = tk.Frame(parent, bg=C_CARD, highlightthickness=1,
                        highlightbackground=C_LINE)
        bar = tk.Frame(card, bg=C_CARD, height=30)
        bar.pack(fill="x", padx=10, pady=(8, 0))
        bar.pack_propagate(False)
        tk.Label(bar, text=title, font=F_BOLD, fg=C_FG,
                 bg=C_CARD).pack(side="left")
        info = tk.Label(bar, text="未连接", font=(MONO, -11), fg=C_DIM, bg=C_CARD)
        info.pack(side="right")
        cv = tk.Canvas(card, bg=C_CANVAS, highlightthickness=0, bd=0)
        cv.pack(fill="both", expand=True, padx=8, pady=(6, 8))
        return card, cv, info

    def _build(self):
        # ---------- 顶栏 ----------
        head = tk.Frame(self.root, bg=C_BG, height=62)
        head.pack(fill="x", side="top")
        head.pack_propagate(False)

        lf = tk.Frame(head, bg=C_BG)
        lf.pack(side="left", padx=18)
        tk.Label(lf, text="ASTRA PRO PLUS", font=(FT, -10, "bold"),
                 fg=C_ACCENT, bg=C_BG).pack(anchor="w", pady=(4, 0))
        tk.Label(lf, text="深度 · 彩色 双流同步采集", font=F_TITLE,
                 fg=C_FG, bg=C_BG).pack(anchor="w")

        rf = tk.Frame(head, bg=C_BG)
        rf.pack(side="right", padx=18)
        self.btn_close = self._btn(rf, "关闭相机", self.disconnect, "danger")
        self.btn_close.pack(side="right", padx=(8, 0))
        self.btn_open = self._btn(rf, "启动相机", self.connect, "primary")
        self.btn_open.pack(side="right")
        self.pill = tk.Label(rf, text="● 未连接", font=F_MD, fg=C_SUB,
                             bg=C_CARD, padx=12, pady=5)
        self.pill.pack(side="right", padx=10)

        # ---------- 底栏 ----------
        sb = tk.Frame(self.root, bg=C_PANEL, height=30)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)
        self.lbl_status = tk.Label(sb, text="就绪", font=F_SM, fg=C_SUB,
                                   bg=C_PANEL, anchor="w")
        self.lbl_status.pack(side="left", padx=14)
        self.lbl_rate = tk.Label(sb, text="深度 -- fps   彩色 -- fps   ·   已保存 0 组",
                                 font=(MONO, -11), fg=C_SUB, bg=C_PANEL)
        self.lbl_rate.pack(side="right", padx=14)

        # ---------- 主体 ----------
        body = tk.Frame(self.root, bg=C_BG)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        pv = tk.Frame(body, bg=C_BG)
        pv.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=(4, 12))
        pv.grid_columnconfigure(0, weight=3, uniform="v")
        pv.grid_columnconfigure(1, weight=2, uniform="v")
        pv.grid_rowconfigure(0, weight=1)

        card1, self.cv_color, self.info_color = self._preview_card(pv, "彩色")
        card1.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        card2, self.cv_depth, self.info_depth = self._preview_card(pv, "深度")
        card2.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        self.panel = tk.Frame(body, bg=C_PANEL, width=346)
        self.panel.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        self.panel.pack_propagate(False)
        self.panel.grid_propagate(False)
        self._build_panel()

    def _build_panel(self):
        # 常规区：标定模式下整块隐藏，把面板高度让给引导卡片
        self.box_norm = tk.Frame(self.panel, bg=C_PANEL)
        self.box_norm.pack(fill="x")

        # --- 设备 ---
        s1 = self._section("设备", self.box_norm)
        r = tk.Frame(s1, bg=C_CARD)
        r.pack(fill="x")
        tk.Label(r, text="分辨率", font=F_MD, fg=C_SUB, bg=C_CARD).pack(side="left")
        self.var_res = tk.StringVar(value=self.RESOLUTIONS[0])
        cb = ttk.Combobox(r, textvariable=self.var_res, values=self.RESOLUTIONS,
                          state="readonly", style="Dark.TCombobox", width=11)
        cb.pack(side="right")

        r = tk.Frame(s1, bg=C_CARD)
        r.pack(fill="x", pady=(8, 0))
        tk.Label(r, text="彩色后端", font=F_MD, fg=C_SUB, bg=C_CARD).pack(side="left")
        self.var_be = tk.StringVar(value="ffmpeg")
        ttk.Combobox(r, textvariable=self.var_be, values=["ffmpeg", "dshow"],
                     state="readonly", style="Dark.TCombobox",
                     width=11).pack(side="right")

        tk.Label(s1, text="ffmpeg 满 30fps；dshow 备用（约 7fps）",
                 font=(FT, -10), fg=C_DIM, bg=C_CARD, anchor="w",
                 wraplength=290, justify="left").pack(fill="x", pady=(7, 0))

        r = tk.Frame(s1, bg=C_CARD)
        r.pack(fill="x", pady=(8, 0))
        self.var_lockexp = tk.BooleanVar(value=False)
        tk.Checkbutton(r, text="锁定彩色曝光（防暗光降帧）", variable=self.var_lockexp,
                       font=(FT, -12), fg=C_SUB, bg=C_CARD, selectcolor=C_CARD2,
                       activebackground=C_CARD, activeforeground=C_FG,
                       highlightthickness=0, bd=0, cursor="hand2").pack(side="left")
        tk.Label(s1, text="勾选后固定 1/64s 曝光：恢复 30fps、亮度不漂移；暗光下偏暗",
                 font=(FT, -10), fg=C_DIM, bg=C_CARD, anchor="w",
                 wraplength=290, justify="left").pack(fill="x", pady=(7, 0))

        # --- 抓取 ---
        s2 = self._section("抓取", self.box_norm)
        self.btn_grab = self._btn(s2, "抓取一组      [空格]", self.grab, "primary")
        self.btn_grab.pack(fill="x")

        r = tk.Frame(s2, bg=C_CARD)
        r.pack(fill="x", pady=(8, 0))
        self.btn_auto = self._btn(r, "自动抓帧：关", self.toggle_auto, "ghost")
        self.btn_auto.pack(side="left")
        self.var_interval = tk.StringVar(value="2.0")
        sp = tk.Spinbox(r, from_=0.2, to=60.0, increment=0.2, width=5,
                        textvariable=self.var_interval, font=(MONO, -13),
                        bg=C_CARD2, fg=C_FG, buttonbackground=C_CARD2,
                        relief="flat", bd=0, highlightthickness=1,
                        highlightbackground=C_LINE, insertbackground=C_FG,
                        justify="center", format="%.1f")
        sp.pack(side="right")
        tk.Label(r, text="秒", font=F_SM, fg=C_SUB, bg=C_CARD).pack(side="right", padx=(0, 4))

        self.lbl_progress = tk.Label(s2, text="已保存 0 组", font=(MONO, -12),
                                     fg=C_SUB, bg=C_CARD, anchor="w")
        self.lbl_progress.pack(fill="x", pady=(8, 0))

        # --- 引导标定（标定模式下才显示，平时整块不占位）---
        self.box_wiz = tk.Frame(self.panel, bg=C_PANEL)
        wk = self._section("引导标定", self.box_wiz)

        top = tk.Frame(wk, bg=C_CARD)
        top.pack(fill="x")
        self.lbl_stepno = tk.Label(top, text="步骤 1 / 15", font=F_SEC,
                                   fg=C_ACCENT, bg=C_CARD, anchor="w")
        self.lbl_stepno.pack(side="left")
        self.lbl_cov = tk.Label(top, text="覆盖 0/12", font=(MONO, -11),
                                fg=C_DIM, bg=C_CARD, anchor="e")
        self.lbl_cov.pack(side="right")
        # 当前用的是哪种板：换板后数字会变，写在面板上避免"拿错板还在找原因"
        self.lbl_board = tk.Label(top, text="", font=(FT, -10),
                                  fg=C_SUB, bg=C_CARD)
        self.lbl_board.pack(side="left", padx=(8, 0))

        self.cv_step = tk.Canvas(wk, width=290, height=163, bg=C_CANVAS,
                                 highlightthickness=1,
                                 highlightbackground=C_LINE, bd=0)
        self.cv_step.pack(fill="x", pady=(8, 0))

        self.lbl_step_title = tk.Label(wk, text="", font=F_BOLD, fg=C_FG,
                                       bg=C_CARD, anchor="w")
        self.lbl_step_title.pack(fill="x", pady=(9, 0))
        self.lbl_step_do = tk.Label(wk, text="", font=(FT, -12), fg=C_FG,
                                    bg=C_CARD, anchor="w", justify="left",
                                    wraplength=272)
        self.lbl_step_do.pack(fill="x", pady=(3, 0))
        self.lbl_step_tip = tk.Label(wk, text="", font=(FT, -10), fg=C_DIM,
                                     bg=C_CARD, anchor="w", justify="left",
                                     wraplength=272)
        self.lbl_step_tip.pack(fill="x", pady=(3, 0))

        sr = tk.Frame(wk, bg=C_CARD)
        sr.pack(fill="x", pady=(9, 0))
        self.chips = {}
        for key, name in (("color", "彩色"), ("ir", "IR"), ("pos", "位置"),
                          ("size", "尺寸"), ("moved", "变化"), ("tilt", "倾斜")):
            lab = tk.Label(sr, text="· " + name, font=(FT, -10), fg=C_DIM,
                           bg=C_CARD2, padx=5, pady=2)
            lab.pack(side="left", padx=(0, 3))
            self.chips[key] = (lab, name)

        st_r = tk.Frame(wk, bg=C_CARD)
        st_r.pack(fill="x", pady=(9, 0))
        tk.Label(st_r, text="静止度", font=(FT, -10), fg=C_DIM,
                 bg=C_CARD).pack(side="left")
        self.lbl_stab = tk.Label(st_r, text="0%", font=(MONO, -10), fg=C_DIM,
                                 bg=C_CARD, width=4, anchor="e")
        self.lbl_stab.pack(side="right")
        self.pb_step = ttk.Progressbar(st_r, style="Calib.Horizontal.TProgressbar",
                                       mode="determinate", maximum=100)
        self.pb_step.pack(side="left", fill="x", expand=True, padx=(7, 7))

        self.lbl_wiz = tk.Label(wk, text="等待画面 …", font=(FT, -11), fg=C_SUB,
                                bg=C_CARD, anchor="w", justify="left",
                                wraplength=272)
        self.lbl_wiz.pack(fill="x", pady=(5, 0))

        wr = tk.Frame(wk, bg=C_CARD)
        wr.pack(fill="x", pady=(8, 0))
        self.btn_force = self._btn(wr, "采集本步   [空格]", self.force_step,
                                   "primary")
        self.btn_force.pack(side="left", fill="x", expand=True)
        self.btn_skip = self._btn(wr, "跳过", self.skip_step, "ghost")
        self.btn_skip.pack(side="left", padx=(6, 0))

        # --- 联合标定 ---
        s2b = self._section("联合标定")
        self.btn_calib = self._btn(s2b, "引导标定", self.start_calibration, "primary")
        self.btn_calib.pack(fill="x")
        self.pb = ttk.Progressbar(s2b, style="Calib.Horizontal.TProgressbar",
                                  mode="determinate", maximum=100)
        self.pb.pack(fill="x", pady=(8, 0))
        self.lbl_calib = tk.Label(s2b, text="15 步引导：可选棋盘格或圆点板，按提示摆位后手动确认采集",
                                  font=(FT, -10), fg=C_DIM, bg=C_CARD,
                                  anchor="w", justify="left", wraplength=290)
        self.lbl_calib.pack(fill="x", pady=(5, 0))

        # --- 读数 ---
        s3 = self._section("本组读数")
        self.lbl_stats = tk.Label(s3, text="同步 -- ms   ·   有效 -- %   ·   中位 -- mm",
                                  font=(MONO, -12), fg=C_SUB, bg=C_CARD,
                                  anchor="w", justify="left")
        self.lbl_stats.pack(fill="x")

        # --- 输出 ---
        s4 = self._section("输出目录")
        self.lbl_out = tk.Label(s4, text="（连接后自动创建）", font=(MONO, -10),
                                fg=C_SUB, bg=C_CARD, anchor="w", justify="left",
                                wraplength=296)
        self.lbl_out.pack(fill="x")
        r = tk.Frame(s4, bg=C_CARD)
        r.pack(fill="x", pady=(8, 0))
        self._btn(r, "选择目录", self.choose_dir, "ghost").pack(side="left")
        self._btn(r, "打开目录", self.open_dir, "ghost").pack(side="right")

        # --- 日志 ---
        wrap = tk.Frame(self.panel, bg=C_PANEL)
        wrap.pack(fill="both", expand=True, padx=14, pady=(14, 12))
        tk.Label(wrap, text="运行日志", font=F_SEC, fg=C_DIM, bg=C_PANEL,
                 anchor="w").pack(fill="x", pady=(0, 5))
        box = tk.Frame(wrap, bg=C_CARD, highlightthickness=1,
                       highlightbackground=C_LINE)
        box.pack(fill="both", expand=True)
        self.txt = tk.Text(box, bg=C_CARD, fg=C_SUB, font=(MONO, -11),
                           relief="flat", bd=0, highlightthickness=0,
                           wrap="word", padx=8, pady=6, height=4,
                           insertbackground=C_FG, state="disabled")
        scr = tk.Scrollbar(box, command=self.txt.yview, width=10,
                           bg=C_CARD, troughcolor=C_CARD, bd=0,
                           highlightthickness=0, relief="flat",
                           activebackground=C_LINE)
        self.txt.configure(yscrollcommand=scr.set)
        scr.pack(side="right", fill="y")
        self.txt.pack(side="left", fill="both", expand=True)

        self._sync_buttons()

    def _bind_keys(self):
        # bind_all：无论焦点在哪个控件都能收到（Combobox 拿走焦点后
        # root.bind 收不到，这是之前热键"失灵"的原因）
        self.root.bind_all("<space>", self._hotkey_grab)
        self.root.bind_all("<a>", self._hotkey_auto)
        self.root.bind_all("<A>", self._hotkey_auto)
        self.root.bind_all("<c>", self._hotkey_calib)
        self.root.bind_all("<C>", self._hotkey_calib)
        self.root.bind_all("<Escape>", lambda e: self.on_close())
        self.root.focus_set()

    def _typing(self):
        """正在文本框/下拉框里操作时不响应热键。

        空格额外排除 Combobox：readonly 下拉框会拿空格开列表，两者冲突。
        C/A 之类的键下拉框用不到，放行。
        """
        fg = self.root.focus_get()
        return isinstance(fg, (tk.Entry, tk.Spinbox, tk.Text))

    def _hotkey_grab(self, _e=None):
        if isinstance(self.root.focus_get(),
                      (tk.Entry, tk.Spinbox, tk.Text, ttk.Combobox)):
            return
        self.grab("空格")

    def _hotkey_auto(self, _e=None):
        if self._typing():
            return
        self.toggle_auto()

    def _hotkey_calib(self, _e=None):
        if self._typing():
            return
        self.start_calibration()

    # ------------------------------------------------------------ 日志 ---
    def log(self, msg):
        t = time.strftime("%H:%M:%S")
        self.txt.configure(state="normal")
        self.txt.insert("end", f"[{t}] {msg}\n")
        if int(self.txt.index("end-1c").split(".")[0]) > 400:
            self.txt.delete("1.0", "60.0")
        self.txt.see("end")
        self.txt.configure(state="disabled")
        self.lbl_status.configure(text=msg)
        # 镜像到 stdout：从终端启动时可直接诊断；无控制台（pythonw）时静默
        try:
            print(f"[{t}] {msg}", flush=True)
        except Exception:
            pass

    def _pill(self, text, color):
        self.pill.configure(text=text, fg=color)

    def _sync_buttons(self):
        self.btn_open.configure(
            state="disabled" if (self.connected or self.connecting or self.calib_mode)
            else "normal")
        self.btn_close.configure(
            state="disabled" if self.calib_mode
            else ("normal" if (self.connected or self.connecting) else "disabled"))
        self.btn_grab.configure(
            state="normal" if (self.connected or self.calib_mode) else "disabled")
        self.btn_calib.configure(state="disabled" if self.connecting else "normal")

    # ------------------------------------------------------------ 连接 ---
    def connect(self):
        if self.connected or self.connecting:
            return
        self.connecting = True
        self._pill("● 连接中 …", C_WARN)
        self._sync_buttons()
        res = self.var_res.get()
        w, h = (int(x) for x in res.split("x"))
        backend = self.var_be.get()
        lockexp = self.var_lockexp.get()
        self.log(f"正在连接：深度 OpenNI2 · 彩色 {w}x{h} ({backend}) …")
        # Tk 变量只能在主线程读，参数在这里取好再传给线程
        threading.Thread(target=self._worker_connect,
                         args=(w, h, backend, lockexp),
                         daemon=True).start()

    def _worker_connect(self, w, h, backend, lockexp):
        d = None
        try:
            if lockexp:
                ok, got = cp.set_camera_exposure(-6.0)
                self._conn_note = ("已锁定彩色曝光 1/64s" if ok
                                   else "锁曝光未生效，仍为自动")
            else:
                cp.set_camera_exposure(auto=True)   # 显式恢复自动曝光
                self._conn_note = None
            d = cp.DepthSource()
            d.start()
            if backend == "ffmpeg":
                c = cp.ColorSourceFFmpeg(w, h, 30)
            else:
                idx = cp.find_orbbec_camera(verbose=False)
                if idx is None:
                    raise RuntimeError("未找到可用彩色相机")
                c = cp.ColorSource(idx, w, h, 30)
            c.start()
            self._conn_result = ("ok", (d, c, w, h, backend))
        except BaseException as e:  # noqa: BLE001
            if d is not None:
                try:
                    d.release()
                except Exception:
                    pass
            self._conn_result = ("err", f"{type(e).__name__}: {e}")

    def _apply_connect_result(self):
        kind, payload = self._conn_result
        self._conn_result = None
        self.connecting = False
        if kind == "ok":
            self.depth, self.color, w, h, backend = payload
            self.connected = True
            dw, dh, df, fmt = self.depth.mode
            cw, ch, cf = self.color.mode
            self._prepare_out()
            self._pill("● 在线", C_OK)
            self.info_color.configure(text=f"{cw}×{ch} MJPG · {backend}", fg=C_SUB)
            self.info_depth.configure(text=f"{dw}×{dh} {cp.PIXFMT.get(fmt, fmt)}",
                                      fg=C_SUB)
            self.log(f"连接成功：深度 {dw}×{dh}@{df}，彩色 {cw}×{ch}@{cf}")
            if getattr(self, "_conn_note", None):
                self.log(self._conn_note)
            self.log(f"输出目录 {self.out_dir}")
            self._dcount = self.depth.count
            self._ccount = self.color.count
            self._fps_t = time.time()
            self._fps_hint_shown = False
            self._connected_at = time.time()
        else:
            self._pill("● 连接失败", C_ERR)
            self.log("连接失败：" + payload)
            self._ph(self.cv_color, "彩色未连接")
            self._ph(self.cv_depth, "深度未连接")
        self._sync_buttons()

    def disconnect(self):
        if self.connecting:
            return
        if self.calib is not None:
            self.calib.running = False
            try:
                self.calib.join(2.0)
            except Exception:
                pass
            self.calib = None
            self.calib_mode = False
            self.btn_calib.configure(text="一键标定", bg=C_ACCENT, fg="#04121f",
                                     activebackground="#6cbcff")
            self.pb["value"] = 0
        self._teardown_camera()
        if self._cfp is not None:
            try:
                self._cfp.close()
            except Exception:
                pass
            self._cfp = None
            self._csvw = None
        self.auto_on = False
        self.btn_auto.configure(text="自动抓帧：关")
        self._pill("● 未连接", C_SUB)
        self.info_color.configure(text="未连接", fg=C_DIM)
        self.info_depth.configure(text="未连接", fg=C_DIM)
        self.lbl_rate.configure(text="深度 -- fps   彩色 -- fps   ·   已保存 0 组")
        self._ph(self.cv_color, "彩色未连接")
        self._ph(self.cv_depth, "深度未连接")
        self.log("相机已关闭")
        self._sync_buttons()

    def _teardown_camera(self):
        """静默关闭两路流（不动界面文案），供标定切换 IR/深度时复用"""
        if self.color is not None:
            try:
                self.color.close()
            except Exception:
                pass
            self.color = None
        if self.depth is not None:
            try:
                self.depth.release()
            except Exception:
                pass
            self.depth = None
        self.connected = False

    # ------------------------------------------------------------ 输出 ---
    def _prepare_out(self):
        if not self.out_dir:
            self.out_dir = os.path.join(BASE, "captures",
                                        time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.out_dir, exist_ok=True)
        self.lbl_out.configure(text=self.out_dir, fg=C_SUB)
        if self._cfp is None:
            dv = os.path.join(self.out_dir, "device.txt")
            with open(dv, "w", encoding="utf-8") as fp:
                fp.write(f"device_name={self.depth.name}\n")
                fp.write(f"device_uri={self.depth.uri}\n")
                fp.write(f"vid_pid={self.depth.vid:04X}:{self.depth.pid:04X}\n")
                dw, dh, df, fmt = self.depth.mode
                fp.write(f"stream=DEPTH {dw}x{dh}@{df} {cp.PIXFMT.get(fmt, fmt)}\n")
                cw, ch, cf = self.color.mode
                fp.write(f"color={cw}x{ch}@{cf} backend={self.var_be.get()}\n")
                fp.write("depth_unit=mm (uint16 little endian, 0 = invalid)\n")
            csv_path = os.path.join(self.out_dir, "captures.csv")
            new = not os.path.exists(csv_path)
            self._cfp = open(csv_path, "a", newline="", encoding="utf-8")
            self._csvw = csv.writer(self._cfp)
            if new:
                self._csvw.writerow(["index", "color_png", "depth_png", "color_ts",
                                     "depth_ts", "sync_diff_ms", "depth_valid_pct",
                                     "depth_median_mm", "depth_frame_idx"])
                self._cfp.flush()

    def choose_dir(self):
        d = filedialog.askdirectory(initialdir=self.out_dir or BASE,
                                    title="选择保存目录")
        if d:
            self.out_dir = os.path.normpath(d)
            self.lbl_out.configure(text=self.out_dir, fg=C_SUB)
            if self._cfp is not None:
                self._cfp.close()
                self._cfp = None
                self._csvw = None
            self.seq = 1
            self.log(f"输出目录切换为 {self.out_dir}")

    def open_dir(self):
        if not self.out_dir or not os.path.isdir(self.out_dir):
            self.log("目录还不存在")
            return
        try:
            os.startfile(self.out_dir)  # noqa: S606
        except Exception as e:
            self.log(f"打不开目录：{e}")

    # ------------------------------------------------------------ 抓帧 ---
    def toggle_auto(self):
        if self.calib_mode:
            self.log("标定模式下不启用自动抓帧 —— 每步都由你点『采集本步』确认")
            return
        self.auto_on = not self.auto_on
        try:
            self.interval = max(0.2, float(self.var_interval.get()))
        except Exception:
            self.interval = 2.0
            self.var_interval.set("2.0")
        if self.auto_on:
            self._last_auto = 0.0
            self.btn_auto.configure(text=f"自动抓帧：开 {self.interval:.1f}s",
                                    bg="#1f4a3a", fg=C_OK, activebackground="#276050")
        else:
            self.btn_auto.configure(text="自动抓帧：关", bg=C_CARD2, fg=C_FG,
                                    activebackground="#2b3648")
        self.log("自动抓帧已开启" if self.auto_on else "自动抓帧已关闭")

    def grab(self, src="按钮"):
        if self.calib_mode:
            if self.calib is not None:
                self.calib.trigger_src = src
                self.calib.trigger.set()      # 标定模式下 = 采一组姿态
            return
        if not self.connected or self._busy:
            return
        self._busy = True
        try:
            self._save_pair()
        finally:
            self._busy = False

    def _save_pair(self):
        c = self.color.latest()
        if not c:
            self.log("彩色帧未就绪，本组跳过")
            return
        data, cts = c
        jpeg = getattr(self.color, "is_jpeg", False)
        img = None
        if not jpeg:
            img = data
            if img is None:
                return

        snaps = self.depth.snapshot() if self.depth else []
        best = cp.pick_depth_pair(snaps, cts)
        i = self.seq

        if jpeg:
            f1 = f"color_{i:06d}.jpg"
            with open(os.path.join(self.out_dir, f1), "wb") as fp:
                fp.write(data)
        else:
            f1 = f"color_{i:06d}.png"
            cp.imwrite_u(os.path.join(self.out_dir, f1), img)

        if best is None:
            self._csvw.writerow([i, f1, "", f"{cts:.4f}", "", "", "", "", ""])
            self._cfp.flush()
            self.lbl_stats.configure(text="同步 -- ms   ·   有效 -- %   ·   中位 -- mm")
            self.log(f"[{i:06d}] 仅彩色（无可用深度帧）")
        else:
            darr, dts, didx, _ = best
            diff = abs(cts - dts) * 1000.0
            f2, f3 = f"depth_{i:06d}.png", f"depth_{i:06d}_vis.png"
            cp.imwrite_u(os.path.join(self.out_dir, f2), darr)
            cp.imwrite_u(os.path.join(self.out_dir, f3), cp.depth_to_vis(darr))
            v = darr[darr > 0]
            pct = v.size / darr.size * 100.0
            med = float(np.median(v)) if v.size else 0.0
            self._csvw.writerow([i, f1, f2, f"{cts:.4f}", f"{dts:.4f}",
                                 f"{diff:.1f}", f"{pct:.1f}", f"{med:.0f}", didx])
            self._cfp.flush()
            self.lbl_stats.configure(
                text=f"同步 {diff:.1f} ms   ·   有效 {pct:.1f} %   ·   中位 {med:.0f} mm")
            self.log(f"[{i:06d}] 已保存  同步 {diff:.1f}ms  "
                     f"有效 {pct:.1f}%  中位 {med:.0f}mm")

        self.seq += 1
        self.saved += 1
        self.lbl_progress.configure(text=f"已保存 {self.saved} 组")

    # ------------------------------------------------------------ 绘制 ---
    def _ph(self, cv, text):
        cv.delete("all")
        cv.create_text(cv.winfo_width() // 2 or 200,
                       cv.winfo_height() // 2 or 150,
                       text=text, fill=C_DIM, font=(FT, -14), tags="ph")

    def _blit(self, cv, bgr):
        cw, ch = cv.winfo_width(), cv.winfo_height()
        if cw < 20 or ch < 20:
            return
        h, w = bgr.shape[:2]
        s = min(cw / w, ch / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        if (nw, nh) != (w, h):
            interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
            bgr = cv2.resize(bgr, (nw, nh), interpolation=interp)
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ph = ImageTk.PhotoImage(pil)
        cv.delete("all")
        cv.create_image(cw // 2, ch // 2, image=ph)
        cv.image = ph

    # ------------------------------------------------------------ 循环 ---
    def _tick(self):
        self._tk += 1
        try:
            self._drain_log()
            if self._conn_result is not None:
                self._apply_connect_result()

            if self.calib_mode:
                self._calib_tick()
            elif self.connected:
                now = time.time()
                if now - self._fps_t >= 0.5:
                    dt = now - self._fps_t
                    if self.depth:
                        self._fps_d = (self.depth.count - self._dcount) / dt
                        self._dcount = self.depth.count
                    if self.color:
                        self._fps_c = (self.color.count - self._ccount) / dt
                        self._ccount = self.color.count
                    self._fps_t = now
                    self.lbl_rate.configure(
                        text=f"深度 {self._fps_d:4.1f} fps   彩色 {self._fps_c:4.1f} fps"
                             f"   ·   已保存 {self.saved} 组")
                    if (not self._fps_hint_shown
                            and time.time() - self._connected_at > 12
                            and self._fps_c < 25 and self._fps_d > 27):
                        self._fps_hint_shown = True
                        self.log("提示：彩色帧率偏低而深度正常 —— 多半是暗光下自动"
                                 "曝光降帧。可勾选「锁定彩色曝光」重启相机，或开灯")
                    self.info_color.configure(
                        text=f"{self._fps_c:4.1f} fps", fg=C_OK if self._fps_c > 25 else C_WARN)
                    self.info_depth.configure(
                        text=f"{self._fps_d:4.1f} fps", fg=C_OK if self._fps_d > 25 else C_WARN)

                self._refresh_preview()

                if self.auto_on and not self._busy and not self.calib_mode:
                    if time.time() - self._last_auto >= self.interval:
                        self._last_auto = time.time()
                        self.grab("自动抓帧")
        except Exception as e:  # noqa: BLE001
            import traceback
            detail = traceback.format_exc(limit=3)
            self.log(f"界面循环异常：{type(e).__name__}: {e}")
            try:
                print(detail, flush=True)
            except Exception:
                pass
        finally:
            self.root.after(self.TICK, self._tick)

    def _refresh_preview(self):
        # 预览解码很吃 GIL（720p JPEG 约 6ms/帧），跑满 25fps 会把取流线程饿着，
        # 实测彩色帧率从 30 掉到 21。这里彩色 12.5fps、深度 6.25fps 就够看。
        if self._tk % 2 == 0 and self.color:
            c = self.color.latest()
            if c:
                data, ts = c
                if getattr(self.color, "is_jpeg", False):
                    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                else:
                    frame = data
                if frame is not None:
                    self._blit(self.cv_color, frame)

        if self._tk % 4 == 2 and self.depth:
            d = self.depth.latest()
            if d:
                self._blit(self.cv_depth, cp.depth_to_vis(d[0]))

    # ------------------------------------------------------------ 标定 ---
    def _drain_log(self):
        for _ in range(16):
            try:
                self.log(self._logq.get_nowait())
            except queue.Empty:
                return

    def start_calibration(self):
        if self.calib_mode:
            self.stop_calibration()
            return
        if not self.connected:
            self.log("相机未连接，请先点「启动相机」")
            return
        cfg = self._ask_calib_cfg()
        if not cfg:
            return

        self.auto_on = False
        self.btn_auto.configure(text="自动抓帧：关", bg=C_CARD2, fg=C_FG,
                                activebackground="#2b3648")
        self.pb["value"] = 0
        self.lbl_calib.configure(text="正在切换到 IR 模式 …", fg=C_SUB)
        self.btn_calib.configure(text="停止标定", bg="#3a2330", fg=C_ERR,
                                 activebackground="#4a2b3a")
        self.calib_mode = True
        self._pill("● 标定中", C_WARN)
        self._sync_buttons()
        self.info_color.configure(text="启动中 …", fg=C_SUB)
        self.info_depth.configure(text="启动中 …", fg=C_SUB)
        self._wizard_show(True)
        self.lbl_wiz.configure(text="正在启动 IR + 彩色流 …", fg=C_SUB)
        self.pb_step["value"] = 0

        # 采集阶段就要用对板型：detect_corners 按 BOARD["kind"] 分派，OBJP 也要
        # 跟着重烘焙（预览里的倾角估计靠它）。之前只在 _solve() 里 apply，
        # 结果选圆点板进去仍按棋盘检测，预览永远"没检到标定板"。
        apply_board(*cfg["board"])
        self.log(f"标定板：{cc.board_label()}")
        self.log(f"摆放方案：{PLANS.get(cfg.get('plan'), '全幅覆盖')}"
                 f"（{cfg.get('poses', len(CALIB_STEPS))} 步）")
        b = cc.BOARD
        self.lbl_board.configure(
            text=f"{'圆点板' if b['kind'] == 'circles' else '棋盘格'} "
                 f"{b['cols']}×{b['rows']}·{b['square_mm']:g}mm")
        self.log("进入标定模式：关闭深度流，切换为 IR + 彩色")
        self._teardown_camera()
        self.calib = CalibWorker(cfg, self._logq.put)
        self.calib.start()

    def stop_calibration(self):
        w = self.calib
        if w is not None and w.is_alive():
            w.running = False
            self.btn_calib.configure(state="disabled", text="正在停止 …")
            self.log("正在停止标定采集 …")
        else:
            self._finish_calibration()

    # ------------------------------------------------------ 引导标定 UI ---
    @staticmethod
    def _tilt_txt(axis, deg):
        """倾斜动作的人话描述，角度跟着步骤走（步骤表里有 20/25/30/35/40°）"""
        d = abs(int(deg))
        return {
            "yaw": (f"绕竖轴向左偏 {d}°", f"绕竖轴向右偏 {d}°"),
            "pitch": (f"上缘向后仰 {d}°", f"上缘向前俯 {d}°"),
            "roll": (f"顺时针转 {d}°", f"逆时针转 {d}°"),
        }[axis]

    def _step_do(self, st):
        if st["axis"]:
            a, b = self._tilt_txt(st["axis"], st["deg"])
            hint = a if st["deg"] > 0 else b
        else:
            hint = "正对镜头，不要歪"
        where = "画面中间" if st.get("cell") is None else cell_name(st["cell"])
        if st.get("cell") is None and st.get("u") is not None:
            where = "画面偏左" if st["u"] < 0.4 else (
                "画面偏右" if st["u"] > 0.6 else "画面中间")
        return f"{where}，{hint}，距离约 {st['dist']}"

    def _step_hint(self, s, st, w):
        L = w.live or {}
        if not s["color"] and not s["ir"]:
            return "没检到标定板：把板移到画面中间，正对镜头"
        if not s["color"] or not s["ir"]:
            bad = "彩色" if not s["color"] else "IR"
            return f"{bad}没检出：看看是不是板顶到边了，或光照太偏"
        if not s["pos"]:
            if st.get("cell") is not None:
                cell = L.get("cell") or (0, 1)
                dc = st["cell"][0] - cell[0]
                dr = st["cell"][1] - cell[1]
                mv = ("往左" if dc < 0 else "往右" if dc > 0 else "") + \
                     ("往上" if dr < 0 else "往下" if dr > 0 else "")
                return f"位置偏了，{mv}挪一些"
            du = L.get("u", 0.5) - st["u"] if st.get("u") is not None else \
                L.get("u", 0.5) - 0.5
            dv = L.get("v", 0.5) - sum(ZONE_V) / 2
            # v 向下为正：板比目标低（dv>0）就得往上挪
            mv = ("往左" if du > 0.02 else "往右" if du < -0.02 else "") + \
                 ("往上" if dv > 0.04 else "往下" if dv < -0.04 else "")
            if st.get("u") is None:
                return f"板心不在中间带里：{mv or '往中间'}挪一点（看绿点，板可以比带宽）"
            return f"板心不在目标竖带里，{mv}挪一些（板大没关系，对齐绿点即可）"
        if L.get("size", 0.0) < w.SIZE_MIN:
            return "板太小，靠近一点"
        if L.get("size", 1.0) > w.SIZE_MAX:
            return "板太满，退远一点"
        if not s.get("moved", True):
            return w._similar_reason(st, L.get("gapdims") or {},
                                     L.get("gap", 0.0))
        if st["axis"] and not s["tilt"]:
            return f"{AXIS_TXT[st['axis']]}还可以再转一点（可选，不阻塞）"
        if not L.get("steady"):
            return "位置对了，手离开板，等它彻底静止"
        return "把板拿稳，别晃 …"

    def _draw_diagram(self, w):
        cv = self.cv_step
        cv.delete("all")
        x0, y0, W, H = 6, 6, 278, 156
        cv.create_rectangle(x0 - 1, y0 - 1, x0 + W + 1, y0 + H + 1,
                            fill="#07090d", outline=C_LINE)
        st = w.steps[min(w.step_idx, len(w.steps) - 1)]
        L = w.live or {}
        if getattr(w, "plan", "cover") != "cover":
            self._draw_diagram_zone(cv, w, st, L, x0, y0, W, H)
            return
        cw, ch = W / 4.0, H / 3.0
        # 板当前落在哪一格：画个黄点，用户一眼能看出"我在哪、要去哪"
        now_cell = L.get("cell") if (L.get("cok") or L.get("src") == "ir") else None
        for r in range(3):
            for c in range(4):
                gx, gy = x0 + c * cw, y0 + r * ch
                done = (c, r) in w.cells
                cur = (c, r) == st["cell"]
                if done:
                    cv.create_rectangle(gx, gy, gx + cw, gy + ch,
                                        fill="#123020", outline="#1f5c39")
                elif cur:
                    cv.create_rectangle(gx, gy, gx + cw, gy + ch,
                                        fill="#14304d", outline=C_ACCENT, width=2)
                else:
                    cv.create_rectangle(gx, gy, gx + cw, gy + ch,
                                        outline="#232d3b")
                if done:
                    cv.create_text(gx + cw - 9, gy + 9, text="✓",
                                   fill="#3ddc9a", font=(FT, -9))
                if cur:
                    self._draw_board_icon(cv, gx + cw / 2, gy + ch / 2 - 8)
                    if st["axis"]:
                        cv.create_text(gx + cw / 2, gy + ch - 9,
                                       text=f"{AXIS_TXT[st['axis']]} {st['deg']:+d}°",
                                       fill=C_WARN, font=(FT, -9))
                if now_cell == (c, r):
                    # 当前板位置：右下角黄点（目标格内则画在标记块旁边）
                    px = gx + cw - 10 if not cur else gx + 12
                    py = gy + ch - 10 if not cur else gy + 12
                    cv.create_oval(px - 5, py - 5, px + 5, py + 5,
                                   fill=C_WARN, outline="#2a1c00")

    def _draw_diagram_zone(self, cv, w, st, L, x0, y0, W, H):
        """非覆盖方案的示意：整幅画面 + 板心竖带 + 当前板心 + 步骤进度。

        覆盖方案那张 4x3 网格在这里没有意义（横带方案每一步都在中间那一行，
        只摆中间方案连列都不变），换成"板心竖带 + 步骤条"。竖带画通高：
        判定只看板心，板本身常常比带宽，画闭合框会让人误以为要把整块板
        塞进框里（现场实际发生过）。绿=已采、亮框=当前，一眼看出还差几步。
        """
        n = len(w.steps)
        cv.create_rectangle(x0, y0, x0 + W, y0 + H, outline="#232d3b")
        tu = st.get("u")
        if tu is None:
            zones, cur = [(CENTER_ZONE[0], CENTER_ZONE[1])], 0
        else:
            zones = [(u - BAND_HALF, u + BAND_HALF) for u in BAND_U.values()]
            cur = min(range(len(zones)), key=lambda i: abs(
                (zones[i][0] + zones[i][1]) / 2 - tu))
        for i, (a, b) in enumerate(zones):
            xa, xb = x0 + a * W, x0 + b * W
            on = i == cur
            cv.create_rectangle(xa, y0, xb, y0 + H,
                                fill="#1a2740" if on else "",
                                outline=C_ACCENT if on else "#2a3546",
                                dash=(4, 4) if on else None,
                                width=2 if on else 1)
        v0, v1 = ZONE_V
        zx1, zx2 = x0 + zones[cur][0] * W, x0 + zones[cur][1] * W
        self._draw_board_icon(cv, (zx1 + zx2) / 2, y0 + (v0 + v1) / 2 * H)
        cv.create_text(zx1 + 3, y0 + 11, text="板心带", fill=C_ACCENT,
                       font=(FT, -9), anchor="w")
        # 板心实际落点：在目标带里=绿点，出带=黄点
        if L.get("cok") or L.get("src") == "ir":
            pu, pv = float(L.get("u", 0.5)), float(L.get("v", 0.5))
            px, py = x0 + pu * W, y0 + pv * H
            inside = zones[cur][0] <= pu <= zones[cur][1] and v0 <= pv <= v1
            cv.create_oval(px - 5, py - 5, px + 5, py + 5,
                           fill=C_OK if inside else C_WARN, outline="#2a1c00")
        if st["axis"]:
            cv.create_text(x0 + 8, y0 + 11,
                           text=f"{AXIS_TXT[st['axis']]} {st['deg']:+d}°  {st['dist']}",
                           fill=C_WARN, font=(FT, -10), anchor="w")
        else:
            cv.create_text(x0 + 8, y0 + 11, text=f"正对  {st['dist']}",
                           fill=C_SUB, font=(FT, -10), anchor="w")
        s, g = 8, 2
        by = y0 + H - s - 4
        for i in range(n):
            x = x0 + 5 + i * (s + g)
            done, is_cur = i < w.step_idx, i == w.step_idx
            fill = "#123020" if done else ("#14304d" if is_cur else "#1a2230")
            cv.create_rectangle(x, by, x + s, by + s, fill=fill,
                                outline=C_ACCENT if is_cur else "#232d3b")

    def _draw_board_icon(self, cv, bx, by):
        """在 (bx,by) 画这块标定板的样子。

        棋盘画黑白方块（示意），圆点板按物面坐标逐点打点 —— 手上有几行几列、
        是竖长条还是横条，一眼就能跟手里的板对上，避免"图案对但拿错的板"。
        长宽比取自 OBJP，用户改了特征点数也会跟着变。
        """
        try:
            o = cs.OBJP[:, :2]
            x0, x1 = float(o[:, 0].min()), float(o[:, 0].max())
            y0, y1 = float(o[:, 1].min()), float(o[:, 1].max())
            span = max(x1 - x0, y1 - y0, 1e-6)
        except Exception:
            x0, x1, y0, y1, span = 0.0, 1.0, 0.0, 1.0, 1.0
        LMAX = 30.0
        bw = max(11.0, LMAX * max(x1 - x0, 1e-6) / span)
        bh = max(11.0, LMAX * max(y1 - y0, 1e-6) / span)
        cv.create_rectangle(bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2,
                            fill="#eef3fa", outline="#8fa0b8")
        if cc.BOARD.get("kind") == "circles":
            r = max(1.3, min(bw, bh) * 0.10)
            for x, y in o:
                px = bx - bw / 2 + (x - x0) / (x1 - x0) * bw if x1 > x0 else bx
                py = by + bh / 2 - (y - y0) / (y1 - y0) * bh if y1 > y0 else by
                cv.create_oval(px - r, py - r, px + r, py + r,
                               fill="#3a4a5e", outline="")
            return
        cwn, chn = bw / 3.0, bh / 2.0
        for i in range(3):
            for j in range(2):
                if (i + j) % 2 == 0:
                    cv.create_rectangle(bx - bw / 2 + i * cwn, by - bh / 2 + j * chn,
                                        bx - bw / 2 + (i + 1) * cwn,
                                        by - bh / 2 + (j + 1) * chn,
                                        fill="#3a4a5e", outline="")

    def _calib_tick(self):
        w = self.calib
        if w is None:
            return
        if w.last is not None and self._tk % 2 == 0:
            vis_i, vis_c = w.last
            self._blit(self.cv_depth, vis_i)
            self._blit(self.cv_color, vis_c)

        n = max(1, len(w.steps))
        self.pb["value"] = min(100.0, w.saved * 100.0 / n)
        self.lbl_calib.configure(
            text=w.msg, fg=(C_ERR if w.phase == "error" else C_SUB))
        self.lbl_rate.configure(
            text=f"引导标定 {w.saved}/{n} 步   ·   重试 {w.tried} 次")
        if w.ir_mode:
            self.info_depth.configure(text="IR " + w.ir_mode, fg=C_OK)
        if w.color_mode:
            self.info_color.configure(text=w.color_mode, fg=C_OK)

        idx = min(w.step_idx, len(w.steps) - 1)
        st = w.steps[idx]
        key = (idx, len(w.cells), w.saved, w.phase,
               (w.live or {}).get("cell"))
        # 非覆盖方案不谈覆盖格数，标个方案名免得面板上写着"覆盖 0/12"让人误会
        if getattr(w, "plan", "cover") == "cover":
            cov = f"覆盖 {len(w.cells)}/12"
        else:
            cov = PLANS.get(w.plan, "")
        if self._wiz_drawn != key:
            self._wiz_drawn = key
            self._draw_diagram(w)
            if w.step_idx >= len(w.steps) or w.phase in ("solving", "done"):
                self.lbl_stepno.configure(text=f"完成 {len(w.steps)} / {len(w.steps)}")
                self.lbl_cov.configure(text=cov)
                self.lbl_step_title.configure(text="全部步骤完成")
                self.lbl_step_do.configure(
                    text="正在求解内外参，稍等片刻，结果会弹窗给出" if w.phase != "done"
                    else "求解结束")
                self.lbl_step_tip.configure(text="原始图像已存盘，可反复重算")
            else:
                self.lbl_stepno.configure(
                    text=f"步骤 {w.step_idx + 1} / {len(w.steps)}")
                self.lbl_cov.configure(text=cov)
                self.lbl_step_title.configure(text=st["title"])
                self.lbl_step_do.configure(text=self._step_do(st))
                self.lbl_step_tip.configure(text=st["tip"])

        _st, s, ok = w._step_state()
        for k, (lab, name) in self.chips.items():
            if k == "tilt" and not st["axis"]:
                lab.configure(text="· 倾斜", fg=C_DIM, bg=C_CARD2)
            elif s.get(k):
                lab.configure(text="✓ " + name, fg=C_OK, bg="#16302a")
            else:
                lab.configure(text="✗ " + name, fg=C_DIM, bg=C_CARD2)

        # 静止度条：只反映"最近窗口内板抖不抖"，与位置对错无关，
        # 所以哪怕位置还没到位，用户也能看到手是不是已经放开
        stab_pct = int(round(float(getattr(w, "stability", 0.0)) * 100))
        self.pb_step["value"] = stab_pct
        stable = bool((w.live or {}).get("steady"))
        self.lbl_stab.configure(text=f"{stab_pct}%", fg=(C_OK if stable else C_DIM))
        if w.ready:
            self.lbl_wiz.configure(
                text="条件全满足 —— 点『采集本步』或按空格", fg=C_OK)
        elif ok:
            self.lbl_wiz.configure(text="位置对了，等板彻底静止再点采集", fg=C_WARN)
        else:
            self.lbl_wiz.configure(text=self._step_hint(s, st, w), fg=C_WARN)

        if w.phase in ("done", "error", "stopped") and not w.is_alive():
            self._finish_calibration()

    def skip_step(self):
        w = self.calib
        if w is not None:
            w.skip = True

    def force_step(self):
        w = self.calib
        if w is not None:
            w.trigger.set()

    def _wizard_show(self, on):
        if on:
            self.box_norm.pack_forget()
            self.box_wiz.pack(fill="x", before=self._wraps["联合标定"])
        else:
            self.box_wiz.pack_forget()
            self.box_norm.pack(fill="x", before=self._wraps["联合标定"])
        self._wiz_drawn = None

    def _finish_calibration(self):
        w = self.calib
        self.calib = None
        self.calib_mode = False
        self._wizard_show(False)
        self.btn_calib.configure(text="引导标定", bg=C_ACCENT, fg="#04121f",
                                 activebackground="#6cbcff", state="normal")
        self.pb["value"] = 0
        self._pill("● 未连接", C_SUB)
        self.info_color.configure(text="未连接", fg=C_DIM)
        self.info_depth.configure(text="未连接", fg=C_DIM)
        self._ph(self.cv_color, "彩色未连接")
        self._ph(self.cv_depth, "深度未连接")
        self._sync_buttons()

        if w is not None:
            if w.error:
                self.lbl_calib.configure(text="标定失败：" + w.error[:64], fg=C_ERR)
                self.log("标定失败：" + w.error)
            elif w.phase == "stopped":
                self.lbl_calib.configure(text=f"已停止，保留 {w.saved} 组数据",
                                         fg=C_WARN)
                if w.saved >= 10:
                    self.log(f"数据已保留：{w.cfg['out']}（可运行 "
                             f"python calib_solve.py --dir 该目录 单独求解）")
            elif w.result:
                self.lbl_calib.configure(text=f"标定完成：{w.saved} 组姿态", fg=C_OK)

        self.log("退出标定模式，恢复深度流 …")
        self.root.after(200, self.connect)
        if w is not None and w.result:
            self.root.after(600, lambda: self._show_calib_result(w))

    def _open_path(self, path):
        if not path or not os.path.exists(path):
            self.log("路径不存在：" + str(path))
            return
        try:
            os.startfile(path)  # noqa: S606
        except Exception as e:
            self.log(f"打不开：{e}")

    def _ask_calib_cfg(self):
        """标定参数对话框（模态），返回 cfg 或 None"""
        try:
            w, h = (int(x) for x in self.var_res.get().split("x"))
        except Exception:
            w, h = 1280, 720

        dlg = tk.Toplevel(self.root)
        dlg.title("一键标定")
        dlg.configure(bg=C_BG)
        dlg.transient(self.root)
        dlg.resizable(False, False)
        dlg.grab_set()

        wrap = tk.Frame(dlg, bg=C_BG, padx=24, pady=20)
        wrap.pack(fill="both", expand=True)
        tk.Label(wrap, text="联合标定", font=(FT, -18, "bold"), fg=C_FG,
                 bg=C_BG).pack(anchor="w")
        tk.Label(wrap, text=f"彩色 {w}×{h} · IR 640×480 · 自动检测角点并求解外参",
                 font=F_SM, fg=C_SUB, bg=C_BG).pack(anchor="w", pady=(2, 14))

        form = tk.Frame(wrap, bg=C_BG)
        form.pack(fill="x")
        form.grid_columnconfigure(1, weight=1)
        var = {}

        def sbox(row, label, key, default, lo, hi, inc, tip=""):
            tk.Label(form, text=label, font=F_MD, fg=C_SUB, bg=C_BG,
                     anchor="w").grid(row=row, column=0, sticky="w", pady=4)
            v = tk.StringVar(value=default)
            var[key] = v
            tk.Spinbox(form, from_=lo, to=hi, increment=inc, width=7,
                       textvariable=v, font=(MONO, -13),
                       bg=C_CARD2, fg=C_FG, buttonbackground=C_CARD2,
                       relief="flat", bd=0, highlightthickness=1,
                       highlightbackground=C_LINE, insertbackground=C_FG,
                       justify="center").grid(row=row, column=1, sticky="e", pady=4)
            if tip:
                tk.Label(form, text=tip, font=(FT, -10), fg=C_DIM, bg=C_BG,
                         anchor="w").grid(row=row, column=2, sticky="w", padx=(8, 0))

        sbox(0, "静止判定时长", "hold", "0.5", 0.3, 3, 0.1, "秒")

        # 摆放方案：没有升降台时"全幅覆盖"根本摆不动，这里让用户直接选。
        # 默认落在"中间横带"：只用桌面左右平移，精度实测与全幅覆盖持平。
        tk.Label(form, text="摆放方案", font=F_MD, fg=C_SUB, bg=C_BG,
                 anchor="w").grid(row=1, column=0, sticky="w", pady=4)
        plan_var = tk.StringVar(value="band")
        var["plan"] = plan_var
        pf = tk.Frame(form, bg=C_BG)
        pf.grid(row=1, column=1, columnspan=2, sticky="w")
        for _p, _txt in (("band", "中间横带"), ("center", "只摆中间"),
                         ("cover", "全幅覆盖")):
            tk.Radiobutton(pf, text=_txt, value=_p, variable=plan_var,
                           font=F_MD, fg=C_FG, bg=C_BG, selectcolor=C_CARD2,
                           activebackground=C_BG, activeforeground=C_FG,
                           highlightthickness=0, bd=0,
                           ).pack(side="left", padx=(0, 12))

        tk.Label(form, text="标定板", font=F_MD, fg=C_SUB, bg=C_BG,
                 anchor="w").grid(row=2, column=0, sticky="w", pady=4)
        kind_var = tk.StringVar(value="chess")
        var["kind"] = kind_var
        bf = tk.Frame(form, bg=C_BG)
        bf.grid(row=2, column=1, columnspan=2, sticky="w")
        for _k, _txt in (("chess", "棋盘格"), ("circles", "圆点板")):
            tk.Radiobutton(bf, text=_txt, value=_k, variable=kind_var,
                           font=F_MD, fg=C_FG, bg=C_BG, selectcolor=C_CARD2,
                           activebackground=C_BG, activeforeground=C_FG,
                           highlightthickness=0, bd=0,
                           ).pack(side="left", padx=(0, 14))

        sbox(3, "特征点（横）", "cols", "9", 2, 20, 1)
        sbox(4, "特征点（纵）", "rows", "6", 2, 20, 1)
        sbox(5, "间距 / 边长", "square", "25.0", 5, 100, 0.5, "mm")

        # 两种板的默认几何：换板时把数字一并带过去，省得手填错
        DEF = {"chess": ("9", "6", "25.0"), "circles": ("4", "5", "20.0")}

        lbl_note = tk.Label(wrap, text=_note_text("chess", "band"),
                            font=(FT, -10), fg=C_DIM, bg=C_BG, anchor="w",
                            justify="left")
        lbl_note.pack(fill="x", pady=(12, 0))

        def refresh_note():
            lbl_note.configure(text=_note_text(kind_var.get(), plan_var.get()))

        def on_kind(*_):
            c, r, sq = DEF[kind_var.get()]
            var["cols"].set(c)
            var["rows"].set(r)
            var["square"].set(sq)
            refresh_note()

        # 放在 lbl_note 之后绑定，避免回调先于控件创建
        kind_var.trace_add("write", on_kind)
        plan_var.trace_add("write", lambda *_: refresh_note())

        hint = tk.Label(wrap, text="", font=(FT, -11), fg=C_ERR, bg=C_BG, anchor="w")
        hint.pack(fill="x", pady=(6, 0))

        bar = tk.Frame(wrap, bg=C_BG)
        bar.pack(fill="x", pady=(12, 0))
        result = {}

        def on_ok():
            try:
                hold = float(var["hold"].get())
                kind = var["kind"].get()
                plan = var["plan"].get()
                cols = int(float(var["cols"].get()))
                rows = int(float(var["rows"].get()))
                sq = float(var["square"].get())
            except Exception:
                hint.configure(text="有数值格式不正确")
                return
            if not 0.3 <= hold <= 3.0:
                hint.configure(text="稳定确认时长应在 0.3~3 秒")
                return
            if not (2 <= cols <= 20 and 2 <= rows <= 20):
                hint.configure(text="特征点数应在 2~20 之间")
                return
            if not 5 <= sq <= 100:
                hint.configure(text="间距 / 边长应在 5~100mm")
                return
            result.update(hold=hold, poses=len(calib_steps(kind, plan)),
                          kind=kind, plan=plan,
                          cols=cols, rows=rows, square=sq, w=w, h=h,
                          backend=self.var_be.get())
            dlg.destroy()

        self._btn(bar, "开始", on_ok, "primary").pack(side="right")
        self._btn(bar, "取消", dlg.destroy, "ghost").pack(side="right", padx=(0, 8))

        dlg.update_idletasks()
        dw, dh = 470, 512
        sw, sh = dlg.winfo_screenwidth(), dlg.winfo_screenheight()
        dlg.geometry(f"{dw}x{dh}+{max(0, (sw - dw) // 2)}+{max(0, (sh - dh) // 2)}")
        dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
        dlg.wait_window()

        if not result:
            return None
        out = os.path.join(BASE, "calib", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(os.path.join(out, "_failed"), exist_ok=True)
        result["out"] = out
        result["board"] = (result["kind"], result["cols"], result["rows"], result["square"])
        self.log(f"标定输出目录 {out}")
        return result

    def _show_calib_result(self, w):
        res = w.result
        c, d = res["color"], res["depth"]
        K_c, K_d = np.array(c["K"]), np.array(d["K"])
        D_c, D_d = np.array(c["D"]), np.array(d["D"])
        R = np.array(res["R_depth_to_color"])
        T = np.array(res["T_depth_to_color"], dtype=float).ravel()
        ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        tn = float(np.linalg.norm(T))
        rms = float(res["stereo_rms_px"])
        if rms < 1.0:
            qual, qc = "优秀", C_OK
        elif rms < 2.0:
            qual, qc = "可用", C_OK
        else:
            qual, qc = "偏高，建议补采倾斜姿态", C_WARN

        top = tk.Toplevel(self.root)
        top.title("标定结果")
        top.configure(bg=C_BG)
        top.transient(self.root)

        wrap = tk.Frame(top, bg=C_BG, padx=24, pady=20)
        wrap.pack(fill="both", expand=True)
        tk.Label(wrap, text="联合标定完成", font=(FT, -18, "bold"),
                 fg=C_FG, bg=C_BG).pack(anchor="w")
        tk.Label(wrap, text=f"{res['poses']} 组姿态   ·   立体重投影 RMS "
                            f"{rms:.3f} px（{qual}）",
                 font=F_MD, fg=qc, bg=C_BG).pack(anchor="w", pady=(2, 14))

        def card(title, badge, lines):
            box = tk.Frame(wrap, bg=C_CARD, highlightthickness=1,
                           highlightbackground=C_LINE)
            box.pack(fill="x", pady=(0, 10))
            inner = tk.Frame(box, bg=C_CARD)
            inner.pack(fill="x", padx=14, pady=11)
            head = tk.Frame(inner, bg=C_CARD)
            head.pack(fill="x")
            tk.Label(head, text=title, font=F_BOLD, fg=C_FG, bg=C_CARD).pack(side="left")
            if badge:
                tk.Label(head, text=badge, font=(MONO, -11), fg=C_SUB,
                         bg=C_CARD).pack(side="right")
            for txt in lines:
                tk.Label(inner, text=txt, font=(MONO, -12), fg=C_SUB, bg=C_CARD,
                         anchor="w", justify="left").pack(fill="x", pady=(3, 0))

        card("彩色内参", f"RMS {c['rms_px']:.3f} px", [
            f"{c['size'][0]}×{c['size'][1]}    fx {K_c[0, 0]:.2f}   fy {K_c[1, 1]:.2f}",
            f"cx {K_c[0, 2]:.2f}   cy {K_c[1, 2]:.2f}",
            f"k1 {D_c[0]:+.4f}  k2 {D_c[1]:+.4f}  p1 {D_c[2]:+.4f}  p2 {D_c[3]:+.4f}",
        ])
        card("深度（IR 共镜头）内参", f"RMS {d['rms_px']:.3f} px", [
            f"{d['size'][0]}×{d['size'][1]}    fx {K_d[0, 0]:.2f}   fy {K_d[1, 1]:.2f}",
            f"cx {K_d[0, 2]:.2f}   cy {K_d[1, 2]:.2f}",
            f"k1 {D_d[0]:+.4f}  k2 {D_d[1]:+.4f}  p1 {D_d[2]:+.4f}  p2 {D_d[3]:+.4f}",
        ])
        card("外参  （深度相机 → 彩色相机）", "", [
            f"T = ({T[0]:+.2f}, {T[1]:+.2f}, {T[2]:+.2f}) mm    |T| = {tn:.2f} mm",
            f"旋转角 {ang:.2f}°",
        ])
        card("保存位置", "", [w.json_path or ""])

        bar = tk.Frame(wrap, bg=C_BG)
        bar.pack(fill="x", pady=(6, 0))
        self._btn(bar, "打开目录", lambda: self._open_path(w.cfg["out"]),
                  "primary").pack(side="left")

        def copy_path():
            top.clipboard_clear()
            top.clipboard_append(w.json_path or "")
            self.log("JSON 路径已复制到剪贴板")

        self._btn(bar, "复制 JSON 路径", copy_path, "ghost").pack(side="left", padx=(8, 0))
        self._btn(bar, "关闭", top.destroy, "ghost").pack(side="right")

        top.update_idletasks()
        tw, th = 640, 560
        sw, sh = top.winfo_screenwidth(), top.winfo_screenheight()
        top.geometry(f"{tw}x{th}+{max(0, (sw - tw) // 2)}+{max(0, (sh - th) // 2)}")

    # ------------------------------------------------------------ 退出 ---
    def on_close(self):
        try:
            self.disconnect()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


def main():
    _dpi_aware()
    root = tk.Tk()
    CameraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
