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
  - 打印 chessboard_9x6_25mm.pdf（100% 缩放）并贴在硬纸板上
  - 每次变换标定板的角度/位置/距离（0.5~1.2m），避免纯平移
  - 采集中脚本实时显示两侧角点是否检出；未检出的姿态会被自动丢弃
    （原图存入 <out>/_failed/ 以便排查）
"""
import argparse, csv, os, time

import cv2
import numpy as np

from capture_pair import (DepthSource, ColorSource, ColorSourceFFmpeg,
                          find_orbbec_camera, find_ffmpeg, imwrite_u, PIXFMT,
                          FFMPEG_DEVICE)

BOARD = dict(cols=9, rows=6, square_mm=25.0)
PATTERN = (BOARD["cols"], BOARD["rows"])


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
    """棋盘格角点检测，SB 算法优先，失败退回自适应阈值经典算法"""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCornersSB(gray, PATTERN,
                                                 flags | cv2.CALIB_CB_FILTER_QUADS)
    if not found:
        found, corners = cv2.findChessboardCorners(gray, PATTERN, flags)
        if found and refine:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3))
    return found, corners


def draw_found(img, found, corners, label, ok):
    vis = img if img.ndim == 2 else img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(vis, PATTERN, corners, found) if found else None
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
    args = ap.parse_args()

    fail_dir = os.path.join(args.out, "_failed")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    print("=" * 66)
    print(" 深度+彩色 联合标定数据采集   棋盘格: "
          f"{BOARD['cols']}x{BOARD['rows']} 内角点, {BOARD['square_mm']}mm 方格")
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
