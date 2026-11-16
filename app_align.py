"""app_align.py —— 第一个应用：深度-彩色实时对齐 + 鼠标测距

为什么先做这个：它是所有后续应用（点云、尺寸测量、抓取）的地基 ——
深度和彩色必须先能落在同一套像素坐标里，后面才谈得上"在彩色图上量距离"。
所需的一切今天已经就绪：双流取流（capture_pair）+ 立体标定（calib/*/stereo_calib.json）。

做法（标准流程，逐像素前向投影）：
    1. 深度像素 (u,v,z) 反投影成深度相机坐标系下的三维点（用深度内参 K_d/D_d）
    2. 用外参 R,T 换到彩色相机坐标系
    3. 用彩色内参 K_c/D_c 投影到彩色图上的像素位置
    4. 按 z 做"近者优先"的泼溅（splat），得到彩色分辨率下的深度图 Zmap
    5. 空洞用形态学膨胀补，然后彩色图上叠一层深度伪彩

两个必须记住的坑（都已在标定阶段踩过）：
  * Astra 的 IR/深度图是**水平镜像**的。标定时 K_d 是在翻转后的图上估的，
    所以运行时也必须先 cv2.flip(depth,1)，否则对齐会整体左右错位。
  * 深度量程是 0.6~8m（规格书），0.6m 以内读数不可信 —— 近处会成片空洞。

用法：
    python app_align.py                      # 自动找最新一份标定
    python app_align.py --calib calib/20261116_154823/stereo_calib.json
    python app_align.py --backend dshow      # 彩色改用 OpenCV 取流（默认 ffmpeg）

操作：
    鼠标移动 = 实时测距   左键 = 锁定/解锁   m = 切换显示模式
    s = 存图   q / ESC = 退出
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

import capture_pair as cp
from ui_common import wait_key

HERE = os.path.dirname(os.path.abspath(__file__))

# 深度伪彩的映射区间（米）。规格书量程 0.6-8m，常用的桌面/人手场景落在
# 0.6~2.5m，这个区间能把近处层次拉开；可通过 --near/--far 改。
NEAR_M, FAR_M = 0.30, 2.50


# --------------------------------------------------------------- 标定加载 ---
def latest_calib():
    """挑 calib/ 下最新的一份 stereo_calib.json（按修改时间）"""
    cand = glob.glob(os.path.join(HERE, "calib", "*", "stereo_calib.json"))
    if not cand:
        raise SystemExit("calib/ 下没有 stereo_calib.json，先做一次引导标定")
    return max(cand, key=os.path.getmtime)


def load_calib(path):
    d = json.load(open(path, encoding="utf-8"))
    # 统一 float32：逐像素运算要跑 30fps，float64 会把内参/外参的乘法整体升精度，
    # 实测这一项就吃掉一半时间（307k 点 × 十几次数组运算）。
    out = dict(
        K_d=np.array(d["depth"]["K"], np.float32),
        D_d=np.array(d["depth"]["D"], np.float32).ravel(),
        K_c=np.array(d["color"]["K"], np.float32),
        D_c=np.array(d["color"]["D"], np.float32).ravel(),
        # 标定里 T 的单位是毫米，这里统一换成米，和深度值同量纲
        R=np.array(d["R_depth_to_color"], np.float32),
        T=(np.array(d["T_depth_to_color"], np.float32).ravel() / 1000.0),
        poses=d.get("poses"), rms=d.get("stereo_rms_px"),
        board=(d.get("board") or {}).get("kind"),
        size_d=tuple(d["depth"].get("size") or ()),
        size_c=tuple(d["color"].get("size") or ()),
    )
    return out


def fit_K(K, cal_size, real_size):
    """标定时的分辨率与实际取流不一致时，按比例缩放内参。

    正常情况下两边都是 640x480 / 1280x720，这个函数是道保险：
    比如以后改成 640x480 彩色取流，直接用 1280 的内参会把点全投到画面外。

    ★ 参数必须是 (宽, 高)。曾经有调用方按 (高, 宽) 传进来，sw 算成 0.56，
      彩色内参 fx 被压到一半，投影只覆盖画面左侧 2/3 —— 表现为"画面右侧
      永远测不到深度"。所以这里做一次宽高比反号检测，传反了自动纠正。
    """
    if not cal_size or not real_size:
        return K
    if len(cal_size) < 2 or len(real_size) < 2:
        return K
    # 宽高顺序反了：一边 w>h 另一边 w<h，几乎不可能是真的配置变化
    if (real_size[0] > real_size[1]) != (cal_size[0] > cal_size[1]):
        print(f"  ! fit_K: 分辨率顺序疑似传反 {real_size}（标定 {cal_size}），"
              f"按 (宽,高) 纠正")
        real_size = (real_size[1], real_size[0])
    sw = real_size[0] / float(cal_size[0])
    sh = real_size[1] / float(cal_size[1])
    if abs(sw - 1) < 1e-6 and abs(sh - 1) < 1e-6:
        return K
    K2 = K.copy()
    K2[0, 0] *= sw
    K2[1, 1] *= sh
    K2[0, 2] *= sw
    K2[1, 2] *= sh
    return K2


def adapt_cal(cal, depth_mode, color_mode):
    """把内参适配到实际取流分辨率。

    返回 (cal, dsz, csz)，其中 dsz/csz 是 **(高, 宽)**，给 Aligner / CloudBuilder 用。
    fit_K 需要的是 (宽, 高)，两者的转换只在这一处做 —— 之前三个应用各自
    传 (高, 宽) 进去，fx 被算成 0.56 倍，投影只覆盖画面左侧 2/3，
    表现为"画面右边永远测不到深度"。
    """
    cal["K_d"] = fit_K(cal["K_d"], cal["size_d"], (depth_mode[0], depth_mode[1]))
    cal["K_c"] = fit_K(cal["K_c"], cal["size_c"], (color_mode[0], color_mode[1]))
    return cal, (depth_mode[1], depth_mode[0]), (color_mode[1], color_mode[0])


class Aligner:
    """把深度图重投影到彩色相机视角。

    输出刻意做成**彩色图的一半分辨率**：640x480 的深度点泼溅到 1280x720 上，
    每个点只覆盖 ~3 个彩色像素，六成以上的格子是空的，补洞要么慢要么糊；
    泼到 640x360 上则几乎铺满（覆盖率 >100%），再线性放大回 1280x720 显示，
    既没有空洞也不糊。测量读数不依赖这张图 —— 它直接用全分辨率的投影结果查。
    """

    def __init__(self, cal, depth_size, color_size):
        self.cal = cal
        self.dh, self.dw = depth_size
        self.ch, self.cw = color_size
        self.hh, self.hw = self.ch // 2, self.cw // 2
        v, u = np.mgrid[0:self.dh, 0:self.dw].astype(np.float32)
        self.u, self.v = u, v
        self._flat = np.zeros(self.hh * self.hw, np.float32)

    def warp(self, depth_mm):
        """深度(毫米, 已翻转) -> (半分辨率深度图, 投影结果)。

        返回 (zmap, proj)，proj = (uc, vc, zc, mask)：彩色全分辨率像素坐标与
        相机系深度（米），供光标采样用 —— 读数因此是全分辨率精度。
        """
        cal = self.cal
        z = depth_mm.astype(np.float32)
        z *= 0.001                                       # 米
        ok = z > 0

        # 1) 反投影到深度相机坐标系（注意：用翻转后的图，K_d 才成立）
        Kd = cal["K_d"]
        fx, fy, cx, cy = Kd[0, 0], Kd[1, 1], Kd[0, 2], Kd[1, 2]
        x = (self.u - cx) * z
        x /= fx
        y = (self.v - cy) * z
        y /= fy

        # 2) 深度系 -> 彩色系：P_c = R·P_d + T
        R, T = cal["R"], cal["T"]
        xc = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z + T[0]
        yc = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z + T[1]
        zc = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z + T[2]
        ok &= zc > 0.05                                  # 只保留相机前方的点

        # 3) 归一化 + 畸变 + 投影
        xn = xc / zc
        yn = yc / zc
        k1, k2, p1, p2, k3 = cal["D_c"][:5]
        r2 = xn * xn
        r2 += yn * yn
        # rad = 1 + k1·r² + k2·r⁴ + k3·r⁶，用 Horner 少开几份临时数组
        rad = r2 * k3
        rad += k2
        rad *= r2
        rad += k1
        rad *= r2
        rad += 1.0
        t1 = 2.0 * p1 * xn * yn
        xd = xn * rad + t1 + p2 * (r2 + 2.0 * xn * xn)
        yd = yn * rad + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        Kc = cal["K_c"]
        uc = xd * Kc[0, 0]
        uc += Kc[0, 2]
        vc = yd * Kc[1, 1]
        vc += Kc[1, 2]

        # 4) 半分辨率泼溅，近者优先。
        #    不做 argsort：307k 点泼到 230k 个格子上必然大量碰撞，但碰撞的
        #    几乎都是相邻深度像素（z 只差几毫米），按栅格顺序直接写只在物体
        #    轮廓上留 1px 的错层，肉眼不可见；省下的 5ms 是 30fps 的关键。
        #    测量读数不走这张图（见 sample，用全分辨率投影结果），精度不受影响。
        ui = np.rint(uc * 0.5)
        vi = np.rint(vc * 0.5)
        m = ok & (ui >= 0) & (ui < self.hw) & (vi >= 0) & (vi < self.hh)
        self._flat.fill(0.0)
        idx = vi[m].astype(np.int32)
        idx *= self.hw
        idx += ui[m].astype(np.int32)
        self._flat[idx] = zc[m]
        zmap = self._flat.reshape(self.hh, self.hw)

        # 5) 补洞：半分辨率下残留很少，两轮 3x3 膨胀即可（>99% 铺满）
        ker = np.ones((3, 3), np.uint8)
        for _ in range(2):
            hole = zmap == 0
            if not hole.any():
                break
            zmap[hole] = cv2.dilate(zmap, ker)[hole]
        return zmap, (uc, vc, zc, m)

    @staticmethod
    def sample(proj, x, y, radius=5.0):
        """光标 (x,y) 附近的有效深度中位数（米），没有则 0。

        直接在全分辨率投影结果上查，不受显示降采样的影响。
        """
        uc, vc, zc, m = proj
        sel = m & (np.abs(uc - x) <= radius) & (np.abs(vc - y) <= radius)
        if not sel.any():
            return 0.0
        return float(np.median(zc[sel]))


def colorize(zmap, near=NEAR_M, far=FAR_M):
    """深度 -> 伪彩（固定量程，避免自动归一化导致的"颜色忽深忽浅"）"""
    g = np.clip((zmap - near) / max(far - near, 1e-6), 0, 1)
    g8 = (g * 255).astype(np.uint8)
    g8[zmap <= 0] = 0
    return cv2.applyColorMap(g8, cv2.COLORMAP_TURBO)


def main():
    ap = argparse.ArgumentParser(description="深度-彩色对齐 + 测距（第一个应用）")
    ap.add_argument("--calib", default="", help="标定 JSON，默认取 calib/ 下最新一份")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--backend", default="ffmpeg", choices=["ffmpeg", "dshow"])
    ap.add_argument("--device", default=cp.FFMPEG_DEVICE, help="ffmpeg 后端的 DirectShow 设备名")
    ap.add_argument("--near", type=float, default=NEAR_M, help="伪彩近端(米)")
    ap.add_argument("--far", type=float, default=FAR_M, help="伪彩远端(米)")
    ap.add_argument("--alpha", type=float, default=0.45, help="深度叠加不透明度")
    args = ap.parse_args()

    cpath = args.calib or latest_calib()
    cal = load_calib(cpath)
    print(f"标定: {os.path.relpath(cpath, HERE)}  "
          f"{cal['poses']} 组姿态 / RMS {cal['rms']:.3f}px / 板型 {cal['board']}")

    depth = cp.DepthSource()
    depth.start()
    if args.backend == "ffmpeg":
        color = cp.ColorSourceFFmpeg(args.width, args.height, 30, device=args.device)
    else:
        idx = cp.find_orbbec_camera(verbose=False)
        color = cp.ColorSource(idx, args.width, args.height, 30)
    color.start()
    print(f"深度 {depth.mode[0]}x{depth.mode[1]}   彩色 {color.mode[0]}x{color.mode[1]}")
    # 取流分辨率与标定分辨率不一致时按比例换算，避免点全投到画面外
    cal, dsz, csz = adapt_cal(cal, depth.mode, color.mode)
    if cal["size_d"] and tuple(cal["size_d"]) != (depth.mode[0], depth.mode[1]):
        print(f"  ! 深度分辨率与标定不同（标定 {cal['size_d']}），已缩放内参")
    if cal["size_c"] and tuple(cal["size_c"]) != (color.mode[0], color.mode[1]):
        print(f"  ! 彩色分辨率与标定不同（标定 {cal['size_c']}），已缩放内参")

    align = Aligner(cal, dsz, csz)

    state = dict(mx=-1, my=-1, lock=False, mode=0, z=0.0)
    MODES = ("叠加 overlay", "并排 side-by-side", "仅深度 depth")

    def on_mouse(ev, x, y, flags, _p):
        state["mx"], state["my"] = x, y
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["lock"] = not state["lock"]

    win = "Depth-Color Align  [mouse=measure  m=mode  s=save  q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    t_prev, fps = time.time(), 0.0
    n = 0
    try:
        while True:
            dz = depth.latest()
            cz = color.latest()
            if dz is None or cz is None:
                time.sleep(0.01)
                continue
            depth_mm = dz[0]
            # ffmpeg 后端给的是 MJPG 原始字节，dshow 后端给的是已解码的 BGR
            img = cp.ColorSourceFFmpeg.decode(cz[0]) if getattr(color, "is_jpeg", False) else cz[0]

            # ★ 关键：Astra 的深度/IR 是水平镜像的，K_d 是在翻转后的图上标的
            depth_mm = cv2.flip(depth_mm, 1)
            zmap, proj = align.warp(depth_mm)

            # 鼠标位置的深度：直接用全分辨率投影结果查 5px 邻域的中位数
            state["z"] = align.sample(proj, state["mx"], state["my"]) \
                if 0 <= state["mx"] < align.cw and 0 <= state["my"] < align.ch else 0.0

            # 统计：本帧有效深度占比，能立刻看出"量程/遮挡"是否合适
            cov = float(proj[3].mean()) * 100.0
            # 半分辨率 -> 彩色全分辨率（线性插值：不糊也不留空洞）
            zfull = cv2.resize(zmap, (align.cw, align.ch), interpolation=cv2.INTER_LINEAR)

            if state["mode"] == 2:
                canvas = colorize(zfull, args.near, args.far)
            else:
                dc = colorize(zfull, args.near, args.far)
                canvas = cv2.addWeighted(img, 1 - args.alpha, dc, args.alpha, 0)
                if state["mode"] == 1:
                    small = cv2.resize(dc, (img.shape[1] // 2, img.shape[0] // 2),
                                       interpolation=cv2.INTER_AREA)
                    canvas = np.hstack([img, small])

            # 十字准星 + 读数
            mx, my = state["mx"], state["my"]
            if 0 <= mx < align.cw and 0 <= my < align.ch:
                col = (60, 220, 255) if state["lock"] else (255, 255, 255)
                cv2.drawMarker(canvas, (mx, my), col, cv2.MARKER_CROSS, 26, 1)
                z = state["z"]
                txt = f"Z {z * 1000:6.1f} mm" if z > 0 else "Z   ---  (no depth)"
                cv2.putText(canvas, txt, (mx + 14, max(24, my - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(canvas, txt, (mx + 14, max(24, my - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)

            n += 1
            if n % 5 == 0:
                now = time.time()
                fps = 0.8 * fps + 0.2 * (5.0 / max(now - t_prev, 1e-6))
                t_prev = now
            head = f"{fps:4.1f} fps | depth {cov:5.1f}% | {MODES[state['mode']]}"
            cv2.putText(canvas, head, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, head, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 240, 200), 1, cv2.LINE_AA)
            cv2.imshow(win, canvas)

            k = wait_key(win, 1)      # 点窗口 X 也能退出
            if k in (27, ord("q")):
                break
            if k == ord("m"):
                state["mode"] = (state["mode"] + 1) % 3
            if k == ord("s"):
                fn = time.strftime("align_%H%M%S.png")
                cp.imwrite_u(os.path.join(HERE, "captures", fn), canvas)
                print("saved captures/" + fn)
    finally:
        cv2.destroyAllWindows()
        color.close()
        depth.close()


if __name__ == "__main__":
    main()
