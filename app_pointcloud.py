"""app_pointcloud.py —— 第二个应用：实时彩色 3D 点云

深度反投影 + 外参变换 + 彩色采样，得到彩色相机坐标系下的 (X,Y,Z,RGB)，
然后自己做一个轻量点云渲染器（numpy 全向量化 + 近者优先写像素），
不依赖 open3d / matplotlib —— 项目只保留 numpy+cv2+pillow 的依赖面。

为什么它是所有应用的地基：
  * 点云 = 三维真值。测距只是一个数的验证，点云能一眼看出整幅视场里
    标定是否一致（平面应该是平的、直线应该是直的、角落不该翘起来）。
  * 有了带 RGB 的 PLY，后续尺寸测量、抓取、建图都能直接复用。

坐标约定（彩色相机坐标系，单位米）：
    X 向右   Y 向下   Z 向前（光轴）—— 与 OpenCV 一致，右手系。

渲染原理（无 GPU、无第三方库）：
    鼠标拖动 = 绕 Y/X 旋转   滚轮 = 缩放   双击 = 复位
    旋转后透视投影到屏幕 → 按 Z 从远到近排序 → 直接写像素（近的覆盖远的）
    150k 点一次排序+写像素约 12ms，30fps 够用。

用法：
    python app_pointcloud.py                     # 实时
    python app_pointcloud.py --shot out.png      # 抓一帧渲染成图片后退出
    python app_pointcloud.py --accum 5           # 累积 5 帧（去噪、补空洞）

操作：
    鼠标拖动 = 旋转   滚轮 = 缩放   r/双击 = 复位   1/2/3 = 正视/俯视/侧视
    c = 彩色/伪彩切换   g = 坐标轴   s = 存 PLY   q / ESC = 退出
"""
import argparse
import os
import time

import cv2
import numpy as np

import capture_pair as cp
from ui_common import wait_key
from app_align import adapt_cal, latest_calib, load_calib

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "captures")

# 只保留这个区间内的点（米）。近于 0.3m 的读数不可信（规格书 0.6m 起，
# 留点余量），远于 6m 的结构光已经很稀疏，画上去只是噪声。
Z_LO, Z_HI = 0.30, 6.00


# ------------------------------------------------------------- 点云构建 ---
def project_to_image(pts, K, D, w, h):
    """彩色系 3D 点 → 彩色图像素坐标（含径向+切向畸变）。

    测量功能要靠它把拟合出的矩形角点画回彩色图；点云构建也用它取颜色。
    """
    X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
    zn = np.maximum(Z, 1e-6)
    xn, yn = X / zn, Y / zn
    k1, k2, p1, p2, k3 = D[:5]
    r2 = xn * xn + yn * yn
    rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    u = xd * K[0, 0] + K[0, 2]
    v = yd * K[1, 1] + K[1, 2]
    ok = (Z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, ok


class CloudBuilder:
    """一帧深度 + 一帧彩色 → 彩色相机坐标系下的点云"""

    def __init__(self, cal, depth_size, color_size):
        self.cal = cal
        dh, dw = depth_size
        self.dh, self.dw = dh, dw
        self.ch, self.cw = color_size
        v, u = np.mgrid[0:dh, 0:dw].astype(np.float32)
        self.u, self.v = u, v
        self.Kd = cal["K_d"]
        self.Kc = cal["K_c"]
        self.R, self.T = cal["R"], cal["T"]
        self.Dc = cal["D_c"]

    def build_full(self, depth_mm, img):
        """返回 (pts, rgb, uc, vc, ratio)：彩色系点云 + 每个点的彩色图像素坐标"""
        z = depth_mm.astype(np.float32) * 0.001
        ok = z > 0

        # 1) 反投影到深度相机系（K_d 是在翻转后的图上标的）
        Kd = self.Kd
        x = (self.u - Kd[0, 2]) * z
        x /= Kd[0, 0]
        y = (self.v - Kd[1, 2]) * z
        y /= Kd[1, 1]

        # 2) 深度系 → 彩色系
        R, T = self.R, self.T
        X = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z + T[0]
        Y = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z + T[1]
        Z = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z + T[2]
        ok &= (Z > Z_LO) & (Z < Z_HI)

        # 3) 投回彩色图：既用来取色，也给测量做框选索引
        uc, vc, kok = project_to_image(np.stack([X, Y, Z], -1), self.Kc, self.Dc,
                                       self.cw, self.ch)
        ok &= kok

        pts = np.stack([X[ok], Y[ok], Z[ok]], axis=1)
        ui = uc[ok].astype(np.int32)
        vi = vc[ok].astype(np.int32)
        rgb = img[vi, ui][:, ::-1]            # BGR -> RGB
        return (np.ascontiguousarray(pts), np.ascontiguousarray(rgb),
                np.ascontiguousarray(uc[ok]), np.ascontiguousarray(vc[ok]),
                float(ok.mean()))

    def build(self, depth_mm, img):
        """返回 (pts Nx3 float32 米, rgb Nx3 uint8, 有效点比例)"""
        pts, rgb, _, _, ratio = self.build_full(depth_mm, img)
        return pts, rgb, ratio


# --------------------------------------------------------------- 渲染器 ---
class CloudView:
    """纯 numpy 点云渲染：透视投影 + 近者优先写像素"""

    def __init__(self, w=1000, h=750, fov_deg=50.0):
        self.w, self.h = w, h
        self.f = 0.5 * h / np.tan(np.radians(fov_deg) * 0.5)
        self.yaw, self.pitch = 0.0, 0.0
        self.zoom = 1.0
        self.bg = (24, 24, 26)

    def reset(self):
        self.yaw = self.pitch = 0.0
        self.zoom = 1.0

    def project(self, pts):
        """旋转 + 透视 → (u, v, depth, keep_mask)"""
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp_, sp = np.cos(self.pitch), np.sin(self.pitch)
        # 先绕 Y（yaw），再绕 X（pitch）
        x1 = cy * x + sy * z
        z1 = -sy * x + cy * z
        y1 = cp_ * y - sp * z1
        z2 = sp * y + cp_ * z1
        keep = z2 > 0.05
        f = self.f * self.zoom
        zz = np.maximum(z2, 1e-6)
        u = f * x1 / zz + self.w * 0.5
        v = f * y1 / zz + self.h * 0.5
        keep &= (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        return u, v, z2, keep

    def render(self, pts, rgb, depth_mode=False, axes=True, size=1):
        img = np.empty((self.h, self.w, 3), np.uint8)
        img[:] = self.bg
        if pts is None or len(pts) == 0:
            return img
        u, v, z2, keep = self.project(pts)
        if not keep.any():
            return img
        uu, vv, zz = u[keep], v[keep], z2[keep]
        col = rgb[keep]
        if depth_mode:
            g = np.clip((zz - Z_LO) / (Z_HI - Z_LO), 0, 1)
            col = cv2.applyColorMap((g * 255).astype(np.uint8).reshape(-1, 1),
                                    cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1]
        # 远 → 近 排序，后写的（近的）覆盖先写的
        order = np.argsort(-zz)
        ui = uu[order].astype(np.int32)
        vi = vv[order].astype(np.int32)
        flat = vi * self.w + ui
        flat_img = img.reshape(-1, 3)
        flat_img[flat] = col[order]
        if size >= 2:
            # 邻域补点：右/下各补一格，视觉上把稀疏点点成小方块
            for dx, dy in ((1, 0), (0, 1), (1, 1)):
                ok2 = (ui + dx < self.w) & (vi + dy < self.h)
                flat_img[(vi[ok2] + dy) * self.w + ui[ok2] + dx] = col[order][ok2]
        if axes:
            self._draw_axes(img)
        return img

    def _draw_axes(self, img, L=0.35):
        """原点处的 XYZ 轴：X 红 / Y 绿 / Z 蓝（相机系方向）"""
        o = np.array([0.8, 0.55, 1.6], np.float32)   # 放在视野左下前方，不挡主体
        segs = [((1, 60, 60), o + [L, 0, 0]), ((60, 200, 60), o + [0, L, 0]),
                ((80, 120, 255), o + [0, 0, L])]
        for col, p in segs:
            pair = np.stack([o, p])
            u, v, _, k = self.project(pair)
            if k.all():
                cv2.line(img, (int(u[0]), int(v[0])), (int(u[1]), int(v[1])), col, 2, cv2.LINE_AA)


# ------------------------------------------------------------ PLY 导出 ---
def save_ply(path, pts, rgb):
    """binary_little_endian PLY，带顶点色（MeshLab / CloudCompare 直接打开）"""
    n = len(pts)
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(rec.tobytes())
    return path


# ------------------------------------------------------------------ 主流程 ---
def main():
    ap = argparse.ArgumentParser(description="实时彩色 3D 点云")
    ap.add_argument("--calib", default="")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--backend", default="ffmpeg", choices=["ffmpeg", "dshow"])
    ap.add_argument("--device", default=cp.FFMPEG_DEVICE)
    ap.add_argument("--accum", type=int, default=1, help="累积帧数（>1 去噪补洞，会稍卡）")
    ap.add_argument("--shot", default="", help="渲染一帧存到这个路径后退出")
    ap.add_argument("--depth-color", action="store_true", help="按深度伪彩显示")
    ap.add_argument("--size", type=int, default=1, choices=[1, 2], help="点大小")
    args = ap.parse_args()

    cpath = args.calib or latest_calib()
    cal = load_calib(cpath)

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
    if not args.shot:
        print(f"标定 {os.path.relpath(cpath, HERE)} | 深度 {dsz[1]}x{dsz[0]} | 彩色 {csz[1]}x{csz[0]}")

    view = CloudView()
    state = dict(show_depth=args.depth_color, axes=True, drag=False,
                 lx=0, ly=0, pts=None, rgb=None, ratio=0.0)
    win = "PointCloud  [drag=rotate  wheel=zoom  r=reset  c=color  s=save  q=quit]"

    def on_mouse(ev, x, y, flags, _p):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = True
            state["lx"], state["ly"] = x, y
        elif ev == cv2.EVENT_LBUTTONUP:
            state["drag"] = False
        elif ev == cv2.EVENT_LBUTTONDBLCLK:
            view.reset()
        elif ev == cv2.EVENT_MOUSEMOVE and state["drag"]:
            dx, dy = x - state["lx"], y - state["ly"]
            state["lx"], state["ly"] = x, y
            view.yaw += dx * 0.008
            view.pitch = float(np.clip(view.pitch + dy * 0.008, -1.5, 1.5))
        elif ev == cv2.EVENT_MOUSEWHEEL:
            step = 1.1 if flags > 0 else 1 / 1.1
            view.zoom = float(np.clip(view.zoom * step, 0.15, 8.0))

    if not args.shot:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, on_mouse)

    # 累积缓冲（--accum>1 时把多帧点拼在一起）
    acc_pts, acc_rgb = [], []
    t_prev, fps, n, frames = time.time(), 0.0, 0, 0

    def compose(dz, img):
        """取一帧 → 更新点云（含累积）"""
        d = cv2.flip(dz, 1)                     # Astra 深度/IR 水平镜像
        pts, rgb, ratio = builder.build(d, img)
        nonlocal acc_pts, acc_rgb
        if args.accum > 1:
            acc_pts.append(pts)
            acc_rgb.append(rgb)
            if len(acc_pts) > args.accum:
                acc_pts.pop(0)
                acc_rgb.pop(0)
            pts = np.concatenate(acc_pts)
            rgb = np.concatenate(acc_rgb)
        return pts, rgb, ratio

    try:
        while True:
            dz = depth.latest()
            cz = color.latest()
            if dz is None or cz is None:
                time.sleep(0.01)
                continue
            img = cp.ColorSourceFFmpeg.decode(cz[0]) if getattr(color, "is_jpeg", False) else cz[0]
            pts, rgb, ratio = compose(dz[0], img)
            state["pts"], state["rgb"] = pts, rgb

            canvas = view.render(pts, rgb, depth_mode=state["show_depth"],
                                 axes=state["axes"], size=args.size)

            if args.shot:
                cp.imwrite_u(args.shot, canvas)
                print(f"点云 {len(pts)} 点 | 有效 {ratio * 100:.1f}% | 存 {args.shot}")
                break

            frames += 1
            if frames % 5 == 0:
                now = time.time()
                fps = 0.8 * fps + 0.2 * (5.0 / max(now - t_prev, 1e-6))
                t_prev = now
            head = (f"{fps:4.1f} fps | {len(pts) // 1000}k pts | valid {ratio * 100:4.1f}%"
                    f" | yaw {np.degrees(view.yaw):5.0f} pitch {np.degrees(view.pitch):4.0f}"
                    f" zoom {view.zoom:.2f}")
            cv2.putText(canvas, head, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, head, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (210, 240, 210), 1, cv2.LINE_AA)
            cv2.imshow(win, canvas)

            k = wait_key(win, 1)      # 点窗口 X 也能退出
            if k in (27, ord("q")):
                break
            if k == ord("r"):
                view.reset()
            elif k == ord("c"):
                state["show_depth"] = not state["show_depth"]
            elif k == ord("g"):
                state["axes"] = not state["axes"]
            elif k == ord("1"):
                view.reset()
            elif k == ord("2"):
                view.yaw, view.pitch, view.zoom = 0.0, -1.35, 1.0     # 俯视
            elif k == ord("3"):
                view.yaw, view.pitch, view.zoom = 1.35, 0.0, 1.0      # 侧视
            elif k == ord("s") and state["pts"] is not None:
                fn = os.path.join(OUT_DIR, time.strftime("cloud_%H%M%S.ply"))
                save_ply(fn, state["pts"], state["rgb"])
                print(f"saved captures/{os.path.basename(fn)} ({len(state['pts'])} pts)")
    finally:
        if not args.shot:
            cv2.destroyAllWindows()
        color.close()
        depth.close()


if __name__ == "__main__":
    main()
