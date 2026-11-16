"""app_depthview.py —— 实时深度监视 / 空洞诊断界面

`depth_diag.py` 是命令行脚本：打印统计、把两张 PNG 存到 _detect/ 就退出，
所以从工具箱点进去会觉得"没有界面"。这个界面把同样的统计做成实时的：

    左：原始深度伪彩（无效 = 纯黑）
    右：有效掩码（绿=有深度 / 暗红=空洞）+ 8x6 网格每格有效率
    底：深度值直方图（一眼看出近距盲区、超量程比例）
    HUD：fps / 有效率 / 中位距离 / z=0 与 <0.3m 占比

用法：
    python app_depthview.py                 # 实时
    python app_depthview.py --mirror        # 深度流是镜像的，翻过来跟相机同向
    python app_depthview.py --shot out.png  # 出一张图后退出

按键：m = 右侧视图（掩码网格 / 稳定度累积 / 纯掩码），f = 镜像，
      r = 重置累积，s = 存诊断图，q 或窗口 X = 退出。
"""
import argparse
import os
import time

import cv2
import numpy as np

import capture_pair as cp
from depth_diag import Z_MAX, Z_MIN, analyze, grid_table
from ui_common import wait_key

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_detect")
W, H = 1280, 720
TOP = 112                      # HUD 高度
PW, PH = 640, 480              # 单幅视图尺寸（正好 2x 放大 640x480 深度图）
HIST_Y = 694                   # 直方图柱基线（柱顶 624 > 视图底 592，不重叠）
HIST_H = 66
HBINS = (0, 300, 600, 1000, 1500, 2000, 3000, 4000, 8000, 65536)
VIEWS = ("mask", "stability", "raw")


def color_by_level(pct):
    """有效率分级配色：绿 / 黄 / 红"""
    if pct >= 60:
        return (120, 230, 120), "OK"
    if pct >= 30:
        return (90, 210, 240), "LOW"
    return (90, 110, 245), "BAD"


def depth_view(depth, valid):
    """深度伪彩（0.3~4m），无效像素画成纯黑"""
    g = np.clip((depth.astype(np.float32) - 300.0) / (4000.0 - 300.0), 0, 1)
    g8 = (g * 255).astype(np.uint8)
    img = cv2.applyColorMap(g8, cv2.COLORMAP_TURBO)
    img[~valid] = (0, 0, 0)
    return img


def mask_view(valid, rows=6, cols=8):
    """有效掩码：绿=有深度、暗红=空洞，并叠 8x6 网格与每格有效率"""
    h, w = valid.shape
    vis = np.zeros((h, w, 3), np.uint8)
    vis[valid] = (80, 220, 80)
    vis[~valid] = (45, 35, 85)
    rh, rw = h // rows, w // cols
    for r in range(rows):
        for c in range(cols):
            blk = valid[r * rh:(r + 1) * rh, c * rw:(c + 1) * rw]
            pct = blk.mean() * 100.0
            col, _lv = color_by_level(pct)
            txt = f"{pct:.0f}"
            cv2.putText(vis, txt, (c * rw + rw // 2 - 12, r * rh + rh // 2 + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, (c * rw + rw // 2 - 12, r * rh + rh // 2 + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    for r in range(1, rows):
        cv2.line(vis, (0, r * rh), (w, r * rh), (70, 70, 70), 1)
    for c in range(1, cols):
        cv2.line(vis, (c * rw, 0), (c * rw, h), (70, 70, 70), 1)
    return vis


def stability_view(acc):
    """多帧累积：亮=一直有深度，暗=一直空，灰=时有时无"""
    a8 = np.clip(acc * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(a8, cv2.COLORMAP_VIRIDIS)


def draw_hist(canvas, ratio):
    """底部深度直方图（竖柱）"""
    n = len(ratio)
    x0, bw = 16, (W - 32) // n
    top = HIST_Y - HIST_H
    peak = max(ratio.max(), 1e-6)
    for i in range(n):
        r = ratio[i]
        bx = x0 + i * bw
        bh = int(HIST_H * (r / peak))
        col = (110, 210, 130) if HBINS[i] >= 600 else (110, 130, 225)
        cv2.rectangle(canvas, (bx + 4, HIST_Y - bh), (bx + bw - 10, HIST_Y - 1), col, -1)
        pct = f"{r * 100:.0f}%"
        cv2.putText(canvas, pct, (bx + 4, HIST_Y - bh - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        lo, hi = HBINS[i], HBINS[i + 1]
        lab = (f"{lo / 1000:.1f}-{hi / 1000:.1f}" if lo >= 1000
               else f"{lo / 1000:.1f}-{min(hi, 9999) / 1000:.1f}")
        if hi >= 65536:
            lab = ">8"
        cv2.putText(canvas, lab, (bx + 4, HIST_Y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1, cv2.LINE_AA)


def draw_hud(canvas, res, fps, view, npts_txt=""):
    col, lv = color_by_level(res["valid"])
    cv2.putText(canvas, "Depth Monitor  |  raw depth stream (no calibration involved)",
                (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (215, 215, 215), 1, cv2.LINE_AA)
    big = f"valid {res['valid']:.1f}%"
    cv2.putText(canvas, big, (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(canvas, big, (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 1.15, col, 2, cv2.LINE_AA)
    cv2.putText(canvas, lv, (250, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

    l2 = (f"fps {fps:4.1f}   z=0 {res['zero']:4.1f}%   <0.3m {res['tiny']:4.1f}%   "
          f">8m {res['huge']:4.1f}%")
    if "med" in res:
        l2 += f"   median {res['med'] / 1000:.2f}m   p5 {res['p5'] / 1000:.2f}m   p95 {res['p95'] / 1000:.2f}m"
    cv2.putText(canvas, l2, (16, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (185, 185, 185), 1, cv2.LINE_AA)
    tag = f"right: {view}"
    cv2.putText(canvas, tag, (W - 16 - 11 * len(tag), 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 200, 255), 1, cv2.LINE_AA)
    hint = "m=view  f=mirror  r=reset  s=save  q/close=quit"
    cv2.putText(canvas, hint, (W - 16 - 11 * len(hint), 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="实时深度监视 / 空洞诊断")
    ap.add_argument("--mirror", action="store_true", help="镜像深度图（与相机同向）")
    ap.add_argument("--shot", default="", help="出一张图后退出")
    ap.add_argument("--sec", type=float, default=0.0, help="跑 N 秒自动退出并打印平均帧率")
    args = ap.parse_args()

    ds = cp.DepthSource()
    ds.start()
    state = dict(mirror=args.mirror, view=0, acc=None, fps=0.0)
    win = "DepthView  [m=view  f=mirror  r=reset  s=save  q=quit]"
    if not args.shot:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    t_prev, last_ts = time.time(), None
    t0, n_new = t_prev, 0
    try:
        while True:
            if args.sec and time.time() - t0 > args.sec:
                dur = time.time() - t0
                print(f"{n_new} 新帧 / {dur:.1f}s = {n_new / max(dur, 1e-6):.1f} fps "
                      f"(不含重复帧)")
                break
            it = ds.latest()
            if it is None:
                if cv2.waitKey(10) == 27:
                    break
                continue
            depth, ts = it[0], it[1] if len(it) > 1 else None
            if ts is not None and ts == last_ts:
                time.sleep(0.004)                 # 还是旧帧：不重复统计、不虚增 fps
                continue
            last_ts = ts
            n_new += 1
            if state["mirror"]:
                depth = cv2.flip(depth, 1)
            valid, res = analyze(depth)

            # 稳定度累积（指数滑动平均，约 30 帧的"记忆"）
            if state["acc"] is None or state["acc"].shape != valid.shape:
                state["acc"] = valid.astype(np.float32)
            else:
                state["acc"] += (valid.astype(np.float32) - state["acc"]) * 0.06

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                state["fps"] = 0.9 * state["fps"] + 0.1 / dt if state["fps"] else 1.0 / dt

            view = VIEWS[state["view"]]
            canvas = np.zeros((H, W, 3), np.uint8)
            canvas[TOP:TOP + PH, 0:PW] = depth_view(depth, valid)
            if view == "mask":
                canvas[TOP:TOP + PH, PW:W] = mask_view(valid)
            elif view == "stability":
                canvas[TOP:TOP + PH, PW:W] = stability_view(state["acc"])
            else:
                canvas[TOP:TOP + PH, PW:W] = cv2.cvtColor(
                    (valid * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            draw_hud(canvas, res, state["fps"], view)
            draw_hist(canvas, np.histogram(depth, bins=HBINS)[0].astype(np.float64)
                      / max(depth.size, 1))
            cv2.line(canvas, (PW, TOP), (PW, TOP + PH), (60, 60, 60), 1)

            if args.shot:
                cp.imwrite_u(args.shot, canvas)
                print(f"有效 {res['valid']:.1f}%  z=0 {res['zero']:.1f}%  "
                      f"<0.3m {res['tiny']:.1f}%  >8m {res['huge']:.1f}%")
                if "med" in res:
                    print(f"中位 {res['med']:.0f}mm  p5 {res['p5']:.0f}mm  p95 {res['p95']:.0f}mm")
                for line in grid_table(valid):
                    print("   " + line)
                print("存", args.shot)
                break

            cv2.imshow(win, canvas)
            k = wait_key(win, 1)      # 点窗口 X 也能退出
            if k in (27, ord("q")):
                break
            if k == ord("m"):
                state["view"] = (state["view"] + 1) % len(VIEWS)
            elif k == ord("f"):
                state["mirror"] = not state["mirror"]
                state["acc"] = None
            elif k == ord("r"):
                state["acc"] = None
            elif k == ord("s"):
                os.makedirs(OUT, exist_ok=True)
                fn = os.path.join(OUT, time.strftime("depthview_%H%M%S.png"))
                cp.imwrite_u(fn, canvas)
                print("saved", fn)
    finally:
        cv2.destroyAllWindows()
        ds.close()


if __name__ == "__main__":
    main()
