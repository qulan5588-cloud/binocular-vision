"""只测帧率，不落盘 —— 用于换线/换口后快速判断链路是否够用。

用法：
    python probe_fps.py                      # 720p 双流测 10 秒
    python probe_fps.py --width 1920 --height 1080
    python probe_fps.py --color-only         # 深度死锁时单独验彩色
    python probe_fps.py --secs 20
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capture_pair as cp


def bar(v, total=30, width=28):
    n = max(0, min(width, int(round(v / total * width))))
    return "#" * n + "." * (width - n)


def main():
    ap = argparse.ArgumentParser(description="帧率探测（不保存图像）")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--color-only", action="store_true")
    ap.add_argument("--camera", type=int, default=-1)
    args = ap.parse_args()

    print("=" * 62)
    print(f" 帧率探测  {args.width}x{args.height} @{args.fps}  时长 {args.secs:.0f}s")
    print("=" * 62)

    depth = None
    color = None
    try:
        # 开流前锁短曝光：暗光下 UVC 自动曝光会把彩色拖到 ~20fps，
        # 探测的是"软件真实可达帧率"，必须先排除这个干扰。
        try:
            cp.set_camera_exposure(exp=-6, device_index=0, auto=False)
        except Exception:
            pass

        if not args.color_only:
            print("[1/3] 深度流 ...")
            depth = cp.DepthSource(use_ir=False)
            w, h, f, fmt = depth.mode
            print(f"      {depth.name}  深度 {w}x{h}@{f}  {cp.PIXFMT.get(fmt, fmt)}")
            depth.start()
        else:
            print("[1/3] 跳过深度流")

        print("[2/3] 彩色流 ...")
        # 用 ffmpeg 裸流后端（app_scan.py 实际使用的后端，可跑满 30fps）。
        # 不用 OpenCV/dshow —— OpenCV 5.0 的 MJPG 解码只有约 7fps，
        # 会让探测结果假性不达标。
        if cp.find_ffmpeg():
            color = cp.ColorSourceFFmpeg(args.width, args.height, args.fps,
                                         device=cp.FFMPEG_DEVICE)
            cw, ch, cf = color.mode
            print(f"      ffmpeg裸流  设备 '{color.device}'  协商 {cw}x{ch}@{cf}  FOURCC=MJPG")
        else:
            idx = args.camera
            if idx < 0:
                idx = cp.find_orbbec_camera()
                if idx is None:
                    print("      !! 没有可用的彩色相机")
                    return 1
            color = cp.ColorSource(idx, args.width, args.height, args.fps)
            cw, ch, cf = color.mode
            print(f"      dshow  index={idx}  实际协商 {cw}x{ch}@{cf}  FOURCC={color.fourcc}")
            if color.fourcc != "MJPG":
                print("      !! FOURCC 不是 MJPG，帧率必然很低")
        color.start()

        print(f"[3/3] 预热 1.5s 后采样 {args.secs:.0f}s ...")
        time.sleep(1.5)

        c0 = color.count
        d0 = depth.count if depth else 0
        sync = []
        no_depth = 0
        t0 = time.time()
        while time.time() - t0 < args.secs:
            c = color.latest()
            d = depth.latest() if depth else None
            if c and d:
                sync.append(abs(c[1] - d[1]) * 1000.0)
            elif c and not d:
                no_depth += 1
            time.sleep(0.03)
        dt = time.time() - t0

        cfps = (color.count - c0) / dt
        dfps = (depth.count - d0) / dt if depth else 0.0

        print("-" * 62)
        if depth:
            print(f"  深度  {depth.count - d0:5d} 帧 / {dt:.1f}s  =  {dfps:5.1f} fps  {bar(dfps)}")
        print(f"  彩色  {color.count - c0:5d} 帧 / {dt:.1f}s  =  {cfps:5.1f} fps  {bar(cfps)}")
        if sync:
            sync_sorted = sorted(sync)
            p95 = sync_sorted[int(len(sync_sorted) * 0.95) - 1] if len(sync_sorted) > 1 else sync_sorted[0]
            print(f"  同步差 均值 {sum(sync)/len(sync):.1f} ms   中位 {sync_sorted[len(sync_sorted)//2]:.1f} ms"
                  f"   P95 {p95:.1f} ms   最大 {max(sync):.1f} ms")
        elif depth:
            print(f"  !! 采样期间几乎没拿到成对的深度/彩色帧（{no_depth} 次缺深度）")

        print("-" * 62)
        ok_c = cfps >= args.fps * 0.9
        ok_d = (dfps >= 29.0) if depth else True
        if ok_c and ok_d:
            print(f"  判定：通过  彩色 {cfps:.1f}/{args.fps}  深度 {dfps:.1f}/30")
        else:
            if not ok_c:
                pct = cfps / args.fps * 100
                print(f"  判定：彩色不达标  {cfps:.1f}/{args.fps} ({pct:.0f}%)  差 {args.fps - cfps:.1f} fps")
            if not ok_d:
                print(f"  判定：深度不达标  {dfps:.1f}/30")
            print("  建议：先跑 check_usb.ps1 看 HUB 层数，>1 层就是链路问题")
        return 0

    except KeyboardInterrupt:
        print("\n  用户中断")
        return 130
    except Exception as e:
        print(f"\n  !! 失败：{e}")
        print("  提示：深度流死锁时需要物理拔插 USB")
        return 1
    finally:
        if depth is not None:
            try:
                depth.close()
                time.sleep(0.15)
            except Exception:
                pass
        if color is not None:
            try:
                color.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
