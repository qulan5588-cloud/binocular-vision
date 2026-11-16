# -*- coding: utf-8 -*-
"""
深度+彩色 联合标定 —— 数据采集
------------------------------------------------------------------
原理：
  Astra Pro Plus 的 IR 传感器与深度传感器共镜头，像素一一对应。
  标定时对每台相机分别检测棋盘格角点：
    - 彩色图：直接检测
    - IR 图  ：预处理（16bit 归一化 + 模糊抑制结构光散斑）后检测
  求解阶段用 IR 角点标出深度相机内参（共镜头 => 即深度相机内参），
  再用彩色/IR 角点对应关系解算两相机外参。全程不需要深度数值流，
  因此带宽只需 IR(18.4MB/s) + 彩色 MJPG(~2.6MB/s)，非常从容。

用法:
    python calib_capture.py --poses 25 --auto 2 --out calib_raw
    python calib_capture.py --poses 25            # 手动模式，回车抓取
    python calib_capture.py --width 1920 --height 1080   # 彩色用 1080p

操作建议：
  - 标定板二选一（--board chess|circles）：
      chess   : 打印 chessboard_9x6_25mm.pdf（100% 缩放）贴在硬板上
      circles : CGB-020 铝基圆点板（20 点，行内间距 20mm），黑底白圆
  - 每次变换标定板的角度/位置/距离（0.4~1.2m），避免纯平移
  - 采集中脚本实时显示两侧特征点是否检出；未检出的姿态会被自动丢弃
    （原图存入 <out>/_failed/ 以便排查）
"""
import argparse, csv, os, time

import cv2
import numpy as np

from capture_pair import (DepthSource, ColorSource, ColorSourceFFmpeg,
                          find_orbbec_camera, find_ffmpeg, imwrite_u, PIXFMT,
                          FFMPEG_DEVICE)

# --------------------------------------------------------------- 标定板 ---
# kind 决定检测算法，其余字段含义随类型不同：
#   chess   : cols/rows = 内角点数(横向/纵向)，square_mm = 方格边长
#   circles : cols/rows = findCirclesGrid 的 patternSize(横向/纵向)，
#             square_mm = 同一行内相邻圆心的间距（错行位移取它的一半）
BOARDS = {
    "chess": dict(kind="chess", cols=9, rows=6, square_mm=25.0),
    # CGB-020 铝基板：20 个白圆，行内间距 20mm，8 行 3/2/3/2… 交错，
    # 物理特征跨度约 40x70mm。OpenCV 的 patternSize 是 (4, 5) 而不是 (5, 4)。
    "circles": dict(kind="circles", cols=4, rows=5, square_mm=20.0),
}

BOARD = dict(BOARDS["chess"])
PATTERN = (BOARD["cols"], BOARD["rows"])


def pattern():
    return (BOARD["cols"], BOARD["rows"])


def board_label():
    if BOARD["kind"] == "circles":
        return (f"圆点阵 {BOARD['cols']}x{BOARD['rows']} = "
                f"{BOARD['cols'] * BOARD['rows']} 点, 行内间距 {BOARD['square_mm']}mm")
    return (f"棋盘格 {BOARD['cols']}x{BOARD['rows']} 内角点, "
            f"{BOARD['square_mm']}mm 方格")


def board_span_mm():
    """标定点的物理横向跨度(mm)，用来估算"板该占画面多大" """
    if BOARD["kind"] == "circles":
        return (BOARD["rows"] - 1) * BOARD["square_mm"] / 2.0
    return (BOARD["cols"] - 1) * BOARD["square_mm"]


def set_board(kind, cols=None, rows=None, size_mm=None):
    """切换标定板类型。BOARD 原地更新，已持有引用的模块继续有效。"""
    if kind not in BOARDS:
        raise ValueError(f"未知标定板类型: {kind}（可选 {list(BOARDS)}）")
    BOARD.clear()
    BOARD.update(BOARDS[kind])
    if cols:
        BOARD["cols"] = int(cols)
    if rows:
        BOARD["rows"] = int(rows)
    if size_mm:
        BOARD["square_mm"] = float(size_mm)
    globals()["PATTERN"] = pattern()
    return BOARD



def ir_to_8bit(ir16, blur=True):
    """16bit IR -> 8bit，并抑制结构光散斑"""
    v = ir16.astype(np.float32)
    nz = v[v > 0]
    lo, hi = (np.percentile(nz, 1), np.percentile(nz, 99)) if nz.size else (0, 1)
    if hi <= lo:
        hi = lo + 1
    g = (np.clip((v - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    if blur:
        g = cv2.GaussianBlur(g, (5, 5), 0)
    return g


def detect_corners(gray, refine=True):
    """按当前标定板类型检测特征点。

    返回 (found, pts)，pts 形状 (N,1,2) float32。
    顺序必须与 calib_solve.board_points() 的物面坐标严格一一对应。
    """
    if gray.ndim == 3:                    # 彩图进来也能跑，转灰度即可
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if BOARD["kind"] == "circles":
        return _detect_circles(gray)
    return _detect_chess(gray, refine)


def _detect_chess(gray, refine=True):
    """棋盘格：SB 算法优先，失败退回自适应阈值经典算法"""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCornersSB(gray, pattern(),
                                                 flags | cv2.CALIB_CB_FILTER_QUADS)
    if not found:
        found, corners = cv2.findChessboardCorners(gray, pattern(), flags)
        if found and refine:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3))
    return found, corners


_BLOB_DET = None


def _blob_detector():
    """给铝基板用的 blob 检测器：面积+圆度+凸度约束。

    阳极氧化铝面在灯光下会有亮斑，反色后变成假"圆点"，默认检测器
    （不过滤形状）容易被它们带偏。这里用宽面积区间 + 宽松圆度，
    既能滤掉反光碎片，也不至于把斜视时的椭圆误杀。
    """
    global _BLOB_DET
    if _BLOB_DET is None:
        p = cv2.SimpleBlobDetector_Params()
        p.filterByArea = True
        p.minArea = 60.0
        p.maxArea = 500000.0
        p.filterByCircularity = True
        p.minCircularity = 0.70
        p.filterByConvexity = True
        p.minConvexity = 0.85
        p.filterByInertia = True
        p.minInertiaRatio = 0.30
        p.minDistBetweenBlobs = 8.0
        p.blobColor = 0
        _BLOB_DET = cv2.SimpleBlobDetector_create(p)
    return _BLOB_DET


def _detect_circles(gray):
    """非对称圆点阵。

    findCirclesGrid 的 blob 检测默认找"浅底上的暗斑"，而 CGB-020 是黑底白圆，
    必须先反色 —— 实测不反色时一次也检不出。反色优先、原图兜底，
    以后换浅底深点的印刷圆点板也能直接用；最后再用带形状约束的检测器
    试一次，应付铝板反光。

    实战教训（2026-11-16 14:41 现场）：板立在泡沫箱上、人站在板侧，
    彩色帧里人脸/泡沫紧挨着板缘，反相后全变成暗色连通域 —— 实测只要
    点阵旁有**任何一个**多余连通域，CLUSTERING 的点阵匹配就会失败，
    20 个标准圆也配不上。所以最后再兜一层"组件过滤"：自己挑出像圆点
    的连通域、按原样画到白底画布上，再交给 findCirclesGrid 匹配。
    """
    flags = cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
    inv = cv2.bitwise_not(gray)
    for img, det in ((inv, None), (gray, None), (inv, _blob_detector())):
        if det is None:
            ok, pts = cv2.findCirclesGrid(img, pattern(), flags=flags)
        else:
            ok, pts = cv2.findCirclesGrid(img, pattern(), flags=flags, blobDetector=det)
        if ok and pts is not None:
            return True, pts.astype(np.float32)
    return _circles_component_match(inv)


def _circles_component_match(inv):
    """现场兜底：把"像圆点"的连通域抠出来重建干净图，再做点阵匹配。

    输入是反色图（黑圆白底）。步骤：
    1. Otsu 二值出暗色区域（= 原图里的圆点 + 场景里的亮物）；
    2. 以"中小、方圆"连通域的面积中位数为圆点尺寸先验，按
       面积带 / 长宽比 / 填充率过滤 —— 长条是人脸边缘，超大面积
       是泡沫箱，填充率低的是板缝框形，都不要；
    3. 把留下的连通域按**原形状**（斜视时是椭圆，中心位置必须准）
       画到白底画布上，交给 findCirclesGrid。
    只在前三级全失败时才走，正常场景零影响。
    """
    # 反相图里圆点是"暗斑"，前景取暗色；Otsu 自动适配曝光强弱
    _t, bw = cv2.threshold(inv, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw)
    if n <= 1:
        return False, None
    # 圆点尺寸先验：只统计"不太大不太长"的连通域，圆点数量最多，
    # 中位数自然落在圆点上 —— 即使混进若干干扰块也不影响
    cand = [st[i, 4] for i in range(1, n)
            if 150 < st[i, 4] < bw.size * 0.02
            and max(st[i, 2], st[i, 3]) < min(bw.shape) // 3
            and max(st[i, 2], st[i, 3]) / max(1, min(st[i, 2], st[i, 3])) < 1.9]
    if len(cand) < 20:
        return False, None
    med = float(np.median(cand))
    canvas = np.full(bw.shape, 255, np.uint8)
    keep = 0
    for i in range(1, n):
        w, h, a = st[i, 2], st[i, 3], st[i, 4]
        if not 0.35 * med <= a <= 3.0 * med:
            continue                       # 面积不像圆点（泡沫/灯板/碎屑）
        if max(w, h) / max(1, min(w, h)) > 1.9:
            continue                       # 长条：人脸边缘、板缝
        if a < 0.55 * w * h:
            continue                       # 填充率低：L 形、框形
        canvas[lab == i] = 0
        keep += 1
    if keep < 20:
        return False, None
    ok, pts = cv2.findCirclesGrid(canvas, pattern(),
                                  flags=cv2.CALIB_CB_ASYMMETRIC_GRID
                                  | cv2.CALIB_CB_CLUSTERING)
    if ok and pts is not None:
        return True, pts.astype(np.float32)
    return False, None


def draw_found(img, found, corners, label, ok):
    vis = img if img.ndim == 2 else img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if found and corners is not None:
        if BOARD["kind"] == "circles":
            # 圆点阵的检出顺序是沿格基对角遍历的，连成折线反而看不清
            for p in corners.reshape(-1, 2).astype(int):
                cv2.circle(vis, tuple(p), 4, (0, 255, 255), -1)
        else:
            cv2.drawChessboardCorners(vis, pattern(), corners, found)
    color = (0, 220, 0) if ok else (0, 0, 255)
    cv2.putText(vis, f"{label}: {'OK' if ok else 'NO'}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return vis



def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="深度+彩色联合标定数据采集")
    ap.add_argument("--out", default=os.path.join(here, "calib_raw"))
    ap.add_argument("--poses", type=int, default=25, help="目标有效姿态数")
    ap.add_argument("--auto", type=float, default=2.0, help="自动抓取间隔(秒)，0=手动回车")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--camera", type=int, default=-1)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--color-backend", default="auto", choices=["auto", "ffmpeg", "dshow"],
                    help="彩色取流方式：ffmpeg 管道(推荐) / dshow(OpenCV，MJPG 解码很慢)")
    ap.add_argument("--device", default=FFMPEG_DEVICE,
                    help=f"ffmpeg 后端使用的 DirectShow 设备名，默认 '{FFMPEG_DEVICE}'")
    ap.add_argument("--board", default="chess", choices=list(BOARDS),
                    help="标定板类型：chess=9x6 棋盘格 / circles=CGB-020 圆点阵")
    args = ap.parse_args()

    set_board(args.board)
    fail_dir = os.path.join(args.out, "_failed")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    print("=" * 66)
    print(f" 深度+彩色 联合标定数据采集   {board_label()}")
    print("=" * 66)

    print("[1/3] 启动 IR 流（= 深度相机视角）...")
    ir = DepthSource(use_ir=True)
    iw, ih, ifps, ifmt = ir.mode
    print(f"    {ir.name}  {ir.vid:04X}:{ir.pid:04X}")
    print(f"    IR: {iw}x{ih} @{ifps}fps  {PIXFMT.get(ifmt, ifmt)}")
    ir.start()

    print("[2/3] 启动彩色流 ...")
    backend = args.color_backend
    if backend == "auto":
        backend = "ffmpeg" if find_ffmpeg() else "dshow"
    if backend == "ffmpeg":
        print(f"    后端 ffmpeg 管道  设备 '{args.device}'")
        color = ColorSourceFFmpeg(args.width, args.height, 30, args.device)
    else:
        idx = args.camera if args.camera >= 0 else find_orbbec_camera()
        if idx is None:
            ir.close()
            raise SystemExit("未找到彩色相机")
        color = ColorSource(idx, args.width, args.height, 30)
        print(f"    后端 OpenCV DSHOW  index={idx}")
    print(f"    彩色: {color.mode[0]}x{color.mode[1]} @{color.mode[2]}fps")
    color.start()

    print("[3/3] 预热 2 秒 ...")
    time.sleep(2.0)
    print("-" * 66)

    manifest = os.path.join(args.out, "pairs.csv")
    new_file = not os.path.exists(manifest)
    mfp = open(manifest, "a", newline="", encoding="utf-8")
    writer = csv.writer(mfp)
    if new_file:
        writer.writerow(["index", "color_png", "ir_png", "color_ts", "ir_ts",
                         "color_ok", "ir_ok"])

    show = not args.no_window
    saved = tried = 0
    auto_next = time.time() + (args.auto if args.auto > 0 else 1.0)
    print(f" 采集中 … 目标 {args.poses} 组有效姿态"
          + ("（预览窗口已打开，Q 退出）" if show else "（无窗口，Ctrl+C 退出）"))
    try:
        while saved < args.poses:
            now = time.time()
            if now < auto_next:
                time.sleep(0.02)
                continue
            auto_next = now + (args.auto if args.auto > 0 else 1.0)
            tried += 1

            c = color.latest()
            if c is None:
                print("    !! 彩色未就绪"); continue
            cimg, cts = c
            if getattr(color, "is_jpeg", False):
                # ffmpeg 后端交上来的是 MJPG 原始字节，标定需要 BGR
                cimg = ColorSourceFFmpeg.decode(cimg)
                if cimg is None:
                    print("    !! JPEG 解码失败"); continue
            irf = ir.latest()
            if irf is None:
                print("    !! IR 未就绪"); continue
            ir16, its, fidx, _ = irf

            cgray = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
            igray = ir_to_8bit(ir16)
            cok, cpts = detect_corners(cgray)
            iok, ipts = detect_corners(igray)

            tag = f"{saved + 1:03d}"
            if cok and iok:
                fn_c = f"calib_{tag}_color.png"
                fn_i = f"calib_{tag}_ir.png"
                imwrite_u(os.path.join(args.out, fn_c), cimg)
                imwrite_u(os.path.join(args.out, fn_i), ir16)  # 16bit 原始 IR
                writer.writerow([tag, fn_c, fn_i, f"{cts:.4f}", f"{its:.4f}",
                                 "1", "1"])
                mfp.flush()
                saved += 1
                print(f"    [{tag}] 有效姿态  彩色+IR 角点均检出  "
                      f"(累计 {saved}/{args.poses})")
            else:
                imwrite_u(os.path.join(fail_dir, f"fail_{tried:03d}_color.png"), cimg)
                imwrite_u(os.path.join(fail_dir, f"fail_{tried:03d}_ir.png"), igray)
                why = []
                if not cok: why.append("彩色未检出")
                if not iok: why.append("IR 未检出")
                print(f"    [skip] {why}  (已丢弃, 存 _failed/)")
                if not cok and not iok and not show:
                    pass

            if show:
                vis_c = draw_found(cimg, cok, cpts, "COLOR", cok)
                vis_i = draw_found(igray, iok, ipts, "IR", iok)
                hh = 360
                def rs(x):
                    return cv2.resize(x, (int(x.shape[1] * hh / x.shape[0]), hh))
                disp = np.hstack([rs(vis_c), rs(vis_i)])
                cv2.putText(disp, f"saved {saved}/{args.poses}   [Q]quit",
                            (12, disp.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("calibration capture", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("    用户退出")
                    break
    except KeyboardInterrupt:
        print("\n    用户中断")
    finally:
        mfp.close()
        ir.close()
        color.close()
        if show:
            cv2.destroyAllWindows()

    print("-" * 66)
    print(f" 结束：尝试 {tried} 次，保存 {saved} 组有效姿态 -> {args.out}")
    if saved < 10:
        print(" ⚠ 有效姿态少于 10 组，建议继续采集（标定至少需要 10~15 组）")
    if ir.error:
        print(" IR 错误:", ir.error)
    if color.error:
        print(" 彩色错误:", color.error)


if __name__ == "__main__":
    main()
