# -*- coding: utf-8 -*-
"""Astra Pro Plus 深度 / IR 流验证（OpenNI2，与 capture_gui.py 同链路）

用法:
    python depth_probe.py                 # 采 3 秒，写 _detect/<时间戳>/
    python depth_probe.py --out D:/tmp/x
    python depth_probe.py --ir --secs 5   # 验证标定用的 IR 流
退出码: 0 通过 / 1 打不开设备 / 2 没收到帧
"""
import argparse
import os
import sys
import time

import numpy as np

from capture_pair import BASE_DIR, ONI_LIB_DIR, PIXFMT, DepthSource, depth_to_vis, imwrite_u

# 像素格式 -> 每个原始单位代表多少毫米
MM_PER_UNIT = {100: 1.0, 101: 0.1, 102: 1.0}


def stretch_u8(arr):
    """按 2%~98% 分位线性拉伸到 0~255；0 值（无效区）保持全黑"""
    v = arr[arr > 0]
    if v.size == 0:
        return np.zeros(arr.shape, np.uint8)
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    span = hi - lo if hi > lo else 1.0
    g = (np.clip((arr.astype(np.float32) - lo) / span, 0, 1) * 255).astype(np.uint8)
    g[arr == 0] = 0
    return g


def main():
    ap = argparse.ArgumentParser(description="Astra Pro Plus 深度/IR 流验证（OpenNI2）")
    ap.add_argument("--out", default="", help="输出目录，默认 _detect/<时间戳>/")
    ap.add_argument("--ir", action="store_true", help="验证 IR 流（标定那路）")
    ap.add_argument("--secs", type=float, default=3.0, help="采集时长(秒)")
    args = ap.parse_args()

    out = os.path.abspath(args.out or os.path.join(BASE_DIR, "_detect", time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(out, exist_ok=True)
    tag = "ir" if args.ir else "depth"
    label = "IR" if args.ir else "深度"

    print("=" * 60)
    print(f" Astra Pro Plus {label}流验证（OpenNI2）")
    print("=" * 60)
    print(f" 运行时 : {ONI_LIB_DIR}")
    print(f" 输出   : {out}")

    print("\n[1/4] 打开设备 ...")
    try:
        src = DepthSource(use_ir=args.ir)
    except RuntimeError as e:
        print(f"  !! {e}")
        print("  排查：插好相机（设备管理器应有 ORBBEC Depth Sensor）")
        print("       驱动装完必须重启电脑")
        print("       换 USB 口，中间不要接外接 HUB")
        print("       powershell -File check_usb.ps1 看链路")
        return 1

    w, h, fps, fmt = src.mode
    print(f"  {src.name}  {src.vid:04X}:{src.pid:04X}  {w}x{h}@{fps}fps  {PIXFMT.get(fmt, fmt)}")

    print(f"\n[2/4] 采集 {args.secs:g} 秒 ...")
    try:
        src.start()
    except RuntimeError as e:
        print(f"  !! {e}")
        src.release()
        return 2

    t0, base = time.time(), src.count
    while time.time() - t0 < args.secs:
        time.sleep(0.05)
    dt, got = time.time() - t0, src.count - base
    frame = src.latest()
    src.close()
    src.release()

    if frame is None:
        print("  !! 一帧都没收到")
        return 2

    arr = frame[0]
    measured = got / dt if dt else 0.0
    print(f"  {got} 帧 / {dt:.2f} 秒 => {measured:.1f} fps")
    if measured < fps * 0.8:
        print(f"  !! 低于标称 {fps}fps 的 80%：查暗光降帧 / HUB 层数 / 线材")

    print("\n[3/4] 数据 ...")
    valid = arr[arr > 0]
    print(f"  {arr.shape[1]}x{arr.shape[0]} {arr.dtype}  "
          f"有效 {valid.size}/{arr.size} ({valid.size / arr.size * 100:.1f}%)")
    if valid.size == 0:
        print("  !! 整帧全 0，没有有效数据")
        np.save(os.path.join(out, f"{tag}_raw.npy"), arr)
        return 0
    print(f"  原始值 {arr.min()} ~ {arr.max()}")
    if args.ir:
        print(f"  强度中位 {np.median(valid):.0f}（IR 是反射强度，不换算毫米）")
    else:
        k = MM_PER_UNIT.get(fmt, 1.0)
        mm = valid.astype(np.float64) * k
        cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
        c = arr[max(0, cy - 10):cy + 10, max(0, cx - 10):cx + 10]
        c = c[c > 0]
        print(f"  距离 {mm.min():.0f} ~ {mm.max():.0f} mm  中位 {np.median(mm):.0f} mm")
        if c.size:
            print(f"  中心20x20 中位 {np.median(c) * k:.0f} mm")

    print("\n[4/4] 写盘 ...")
    files = [os.path.join(out, f"{tag}_raw.png")]
    imwrite_u(files[0], stretch_u8(arr))
    if not args.ir:
        files.append(os.path.join(out, f"{tag}_vis.png"))
        imwrite_u(files[-1], depth_to_vis(arr))
    files.append(os.path.join(out, f"{tag}_raw.npy"))
    np.save(files[-1], arr)
    for f in files:
        print(f"  {f}")

    print(f"\n 完成，{label}流验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
