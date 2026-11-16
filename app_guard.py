"""app_guard.py —— 第四个应用：近处障碍 / 安全距离提示

干什么：把视场按左右切成三列，每列实时给出"最近的有效深度"，再按阈值分级
（危险 / 警告 / 安全），危险时全屏闪红边 + 大字提示。相当于一个只看得见
前方 3 个扇区的电子围栏。

为什么按列而不是按整幅取最小值：整幅取 min 只会告诉你"有东西很近"，但不知道
在哪一侧，操作上没法用。分三列后能直接说"右侧 0.42m"，人/车该往哪让一目了然。

为什么用 5% 分位而不是 min：深度图边缘常有个别飞点（遮挡边界、多路径），
直接取 min 会被这些噪点骗到，5% 分位对少量离群点免疫。

场景：取货口净空判断、AGV/叉车前方避障示意、机械臂工作区入侵提示。

用法：
    python app_guard.py                  # 警告 0.8m / 危险 0.5m
    python app_guard.py --warn 1.2 --danger 0.7
    python app_guard.py --cols 5         # 切成 5 个扇区
    python app_guard.py --shot out.png   # 出一张图后退出

操作：
    s = 存图    m = 彩色/深度伪彩    [ / ] = 警告阈值 ±0.1m    q / ESC = 退出
"""
import argparse
import os
import time

import cv2
import numpy as np

import capture_pair as cp
from ui_common import wait_key
from app_align import adapt_cal, latest_calib, load_calib
from app_pointcloud import CloudBuilder

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "captures")

# 分级配色（BGR）
C_DANGER = (40, 40, 235)
C_WARN = (30, 165, 255)
C_SAFE = (90, 210, 120)
C_NONE = (150, 150, 150)


def zone_near(pts, uc, vc, cols, cw, q=5.0):
    """把视场按 x 切成 cols 列，返回每列最近距离(米) + 该列点数；无点返回 nan"""
    out_d = np.full(cols, np.nan)
    out_n = np.zeros(cols, np.int32)
    if len(pts) == 0:
        return out_d, out_n
    z = pts[:, 2]
    edges = np.linspace(0, cw, cols + 1)
    for i in range(cols):
        m = (uc >= edges[i]) & (uc < edges[i + 1])
        n = int(m.sum())
        out_n[i] = n
        if n >= 30:                       # 点太少的分位不可信
            out_d[i] = float(np.percentile(z[m], q))
        elif n > 0:
            out_d[i] = float(z[m].min())
    return out_d, out_n


def level_of(d, warn, danger):
    if not np.isfinite(d):
        return 2, C_NONE               # 2 = 无数据
    if d < danger:
        return 0, C_DANGER
    if d < warn:
        return 1, C_WARN
    return 3, C_SAFE


def draw(canvas, state, warn, danger, cols, cw, ch):
    """危险区着色 + 三区距离条 + 告警横幅"""
    pts, uc, vc, zones, counts = (state["pts"], state["uc"], state["vc"],
                                 state["zones"], state["counts"])
    mir = state["mirror"]
    # 镜像只翻画面内容：点的显示列要跟着反，否则染色会左右错位
    uc_disp = (cw - 1 - uc) if mir else uc
    # 1) 危险/警告区的点直接染成对应颜色（比画热图直观，也不会糊住细节）
    if len(pts):
        z = pts[:, 2]
        ui = uc_disp.astype(np.int32)
        vi = vc.astype(np.int32)
        dm = z < danger
        wm = (z >= danger) & (z < warn)
        canvas[vi[dm], ui[dm]] = C_DANGER
        canvas[vi[wm], ui[wm]] = C_WARN

    # 分区编号跟着镜像走：画面镜像后 Z1 显示的仍是最左边那一区
    if mir:
        zones = zones[::-1]

    # 2) 三区距离条（画在顶部，避开右上角的镜像按钮）
    bar_y, bar_h = 48, 26
    for i in range(cols):
        x0 = int(cw * i / cols)
        x1 = int(cw * (i + 1) / cols)
        lv, col = level_of(zones[i], warn, danger)
        cv2.rectangle(canvas, (x0 + 4, bar_y), (x1 - 4, bar_y + bar_h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (x0 + 4, bar_y), (x1 - 4, bar_y + bar_h), col, 2)
        if np.isfinite(zones[i]):
            txt = f"Z{i + 1} {zones[i] * 100:5.1f}cm"
        else:
            txt = f"Z{i + 1}  ---  "
        cv2.putText(canvas, txt, (x0 + 12, bar_y + 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, txt, (x0 + 12, bar_y + 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col if lv != 2 else (200, 200, 200), 2, cv2.LINE_AA)

    # 3) 最危险的那一区：画圈 + 大字，并在画面上闪边框
    if len(zones):
        idx = int(np.nanargmin(zones)) if np.isfinite(zones).any() else -1
        if idx >= 0 and zones[idx] < warn:
            lv, col = level_of(zones[idx], warn, danger)
            label = "DANGER" if lv == 0 else "WARN"
            big = f"{label}  Z{idx + 1}  {zones[idx] * 100:.0f} cm"
            cv2.putText(canvas, big, (24, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.putText(canvas, big, (24, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, col, 3, cv2.LINE_AA)
            if lv == 0 and int(time.time() * 4) % 2 == 0:   # 4Hz 闪红边
                cv2.rectangle(canvas, (0, 0), (cw - 1, ch - 1), C_DANGER, 10)
    draw_button(canvas, cw, mir)


def btn_rect(cw):
    """镜像按钮的矩形（右上角），鼠标点击命中判定也用它"""
    return (cw - 152, 8, cw - 8, 42)


def draw_button(canvas, cw, on):
    x0, y0, x1, y1 = btn_rect(cw)
    col = (70, 170, 70) if on else (95, 95, 95)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), col, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (235, 235, 235), 1)
    # 图标：左右两个三角对着一根竖线，表示"镜像"
    xc, yc, r = x0 + 20, (y0 + y1) // 2, 8
    cv2.line(canvas, (xc, yc - r), (xc, yc + r), (245, 245, 245), 1, cv2.LINE_AA)
    cv2.fillPoly(canvas, [np.array([[xc - r, yc], [xc - 3, yc - r], [xc - 3, yc + r]])],
                 (245, 245, 245))
    cv2.fillPoly(canvas, [np.array([[xc + r, yc], [xc + 3, yc - r], [xc + 3, yc + r]])],
                 (245, 245, 245))
    txt = "MIRROR ON" if on else "MIRROR OFF"
    cv2.putText(canvas, txt, (x0 + 34, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(canvas, txt, (x0 + 34, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (245, 245, 245), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="近处障碍 / 安全距离提示")
    ap.add_argument("--calib", default="")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--backend", default="ffmpeg", choices=["ffmpeg", "dshow"])
    ap.add_argument("--device", default=cp.FFMPEG_DEVICE)
    ap.add_argument("--warn", type=float, default=0.80, help="警告阈值(米)")
    ap.add_argument("--danger", type=float, default=0.50, help="危险阈值(米)")
    ap.add_argument("--cols", type=int, default=3, help="分区数")
    ap.add_argument("--shot", default="", help="出一张图后退出")
    ap.add_argument("--mirror", action="store_true", help="启动即镜像画面")
    args = ap.parse_args()

    cal = load_calib(args.calib or latest_calib())
    depth = cp.DepthSource()
    depth.start()
    if args.backend == "ffmpeg":
        color = cp.ColorSourceFFmpeg(args.width, args.height, 30, device=args.device)
    else:
        idx = cp.find_orbbec_camera(verbose=False)
        color = cp.ColorSource(idx, args.width, args.height, 30)
    color.start()

    cal, dsz, csz = adapt_cal(cal, depth.mode, color.mode)
    builder = CloudBuilder(cal, dsz, csz)
    cw, ch = csz[1], csz[0]

    state = dict(pts=None, uc=None, vc=None, zones=None, counts=None,
                 show_depth=False, mirror=args.mirror)
    warn, danger = args.warn, args.danger
    win = "Guard  [s=save  m=mode  f=mirror  [/]=warn+-0.1  q=quit]"

    def on_mouse(ev, x, y, flags, _p):
        if ev == cv2.EVENT_LBUTTONDOWN:
            x0, y0, x1, y1 = btn_rect(cw)
            if x0 <= x <= x1 and y0 <= y <= y1:
                state["mirror"] = not state["mirror"]

    if not args.shot:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, on_mouse)
        print(f"安全距离：警告 {warn}m / 危险 {danger}m，{args.cols} 区。"
              f"点右上角 MIRROR 按钮（或按 f）镜像画面，q 退出")

    t_prev, fps, n = time.time(), 0.0, 0
    try:
        while True:
            dz = depth.latest()
            cz = color.latest()
            if dz is None or cz is None:
                time.sleep(0.01)
                continue
            img = cp.ColorSourceFFmpeg.decode(cz[0]) if getattr(color, "is_jpeg", False) else cz[0]
            pts, rgb, uc, vc, ratio = builder.build_full(cv2.flip(dz[0], 1), img)
            zones, counts = zone_near(pts, uc, vc, args.cols, cw)
            state.update(pts=pts, uc=uc, vc=vc, zones=zones, counts=counts)

            canvas = cv2.flip(img, 1) if state["mirror"] else img.copy()
            if state["show_depth"]:
                g = np.clip((pts[:, 2] - 0.3) / 5.7, 0, 1)
                m = np.zeros((ch, cw), np.uint8)
                uc_disp = (cw - 1 - uc) if state["mirror"] else uc
                m[vc.astype(int), uc_disp.astype(int)] = (g * 255).astype(np.uint8)
                canvas = cv2.addWeighted(canvas, 0.65,
                                         cv2.applyColorMap(m, cv2.COLORMAP_TURBO), 0.35, 0)

            draw(canvas, state, warn, danger, args.cols, cw, ch)

            n += 1
            if n % 5 == 0:
                now = time.time()
                fps = 0.8 * fps + 0.2 * (5.0 / max(now - t_prev, 1e-6))
                t_prev = now
            head = (f"{fps:4.1f} fps | {len(pts) // 1000}k pts | warn {warn:.1f} "
                    f"danger {danger:.1f} | valid {ratio * 100:4.1f}%")
            cv2.putText(canvas, head, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, head, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (210, 240, 210), 1, cv2.LINE_AA)

            if args.shot:
                cp.imwrite_u(args.shot, canvas)
                zt = "  ".join(f"Z{i + 1}={zones[i] * 100:.1f}cm" if np.isfinite(zones[i])
                               else f"Z{i + 1}=---" for i in range(args.cols))
                print(f"{zt} | 点数 {len(pts)} | 存 {args.shot}")
                break

            cv2.imshow(win, canvas)
            k = wait_key(win, 1)      # 点窗口 X 也能退出
            if k in (27, ord("q")):
                break
            if k == ord("s"):
                fn = os.path.join(OUT_DIR, time.strftime("guard_%H%M%S.png"))
                cp.imwrite_u(fn, canvas)
                print("saved captures/" + os.path.basename(fn))
            elif k == ord("m"):
                state["show_depth"] = not state["show_depth"]
            elif k == ord("f"):
                state["mirror"] = not state["mirror"]
            elif k == ord("["):
                warn = max(0.4, warn - 0.1)
                print(f"警告阈值 {warn:.1f}m")
            elif k == ord("]"):
                warn = min(3.0, warn + 0.1)
                print(f"警告阈值 {warn:.1f}m")
    finally:
        if not args.shot:
            cv2.destroyAllWindows()
        color.close()
        depth.close()


if __name__ == "__main__":
    main()
