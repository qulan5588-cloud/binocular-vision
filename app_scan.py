"""app_scan.py —— 室内扫描建模：端着相机走一圈，拼出彩色点云 + 俯视平面图

流程（KinectFusion 精简版，纯 numpy，不依赖 scipy/open3d）：

    深度帧 → 顶点图 + 法线图（3 级金字塔）
           → 帧间"投影法 ICP"（point-to-plane，coarse→fine，匀速预测给初值）
           → 累乘位姿 G（相机系 → 世界系，世界系 = 第一帧相机系）
           → 体素去重融合成全局彩色点云
           → 俯视投影成平面图（5cm 格，按高度着色）+ 轨迹

为什么用投影法 ICP 而不是 KD-tree：深度图天生结构化，对应点直接按像素查，
不用建近邻索引；6 万点 3 层 10 次迭代约 25ms，够实时。

坐标约定：相机系 x 右 / y 下 / z 前（OpenCV）。世界系 y 向下为"低"，
平面图取 (x, z) 平面、高度用 y 着色。

已知局限（别当成 SLAM）：
  * 没有 IMU —— 平面图默认你端着相机基本水平走，机身倾斜会有剪切；
  * 没有回环检测 —— 绕回起点会有累积漂移，走远了建议分段扫描再拼；
  * 结构光在强反光/吸光面丢点，那部分在点云里就是洞（物理限制）。

用法：
    python app_scan.py                  # 交互扫描，边扫边看平面图
    python app_scan.py --voxel 0.03     # 体素更粗（点更少、更快）
    python app_scan.py --sec 20 --out captures/scan_demo   # 定时自测并保存

按键：空格 暂停/继续，s 保存（ply+平面图+轨迹），r 清空重扫，
      v 开关 3D 预览，g 开关米格，[ ] 平面图高度切片上下移，
      - / = 调切片带宽，q / 窗口 X 退出。
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

import capture_pair as cp
from app_align import Aligner, adapt_cal, latest_calib, load_calib
from app_pointcloud import CloudView, save_ply
from ui_common import wait_key

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(HERE, "captures")

MIN_D, MAX_D = 0.45, 6.00      # 有效深度区间（米）
VOXEL_DEF = 0.02               # 融合体素边长（米）
CELL = 0.05                    # 平面图格子边长（米）
DMAX = (0.20, 0.10, 0.06)      # 各层 ICP 残差门限（米）
MIN_INLIER = 300               # 最细层最少内点数，低于判跟丢
BG = (20, 21, 25)
SNAP_EVERY = 5                 # 每 N 帧刷新一次全局点云快照（其余帧用缓存）
_LUT = cv2.applyColorMap(np.linspace(0, 255, 256).astype(np.uint8).reshape(-1, 1),
                        cv2.COLORMAP_TURBO).reshape(-1, 3)


# ============================================================ 几何基础 ===
def skew(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]], np.float64)


def se3_exp(xi):
    """6 维 [旋转, 平移] → 4x4，标准 SE(3) 指数映射（转角大了也准）"""
    w = np.asarray(xi[:3], np.float64)
    v = np.asarray(xi[3:], np.float64)
    th = float(np.linalg.norm(w))
    if th < 1e-8:
        R = np.eye(3) + skew(w)
        t = v
    else:
        K = skew(w / th)
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
        V = np.eye(3) + ((1 - np.cos(th)) / th) * K + ((th - np.sin(th)) / th) * (K @ K)
        t = V @ v
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rot_angle(R):
    c = (np.trace(R[:3, :3]) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def vertex_map(z, K):
    """深度图（米，0=无效）→ 顶点图 (h,w,3) + 有效掩码"""
    h, w = z.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.arange(w, dtype=np.float32)[None, :]
    v = np.arange(h, dtype=np.float32)[:, None]
    ok = z > 0.10
    zz = np.where(ok, z, 0.0).astype(np.float32)
    X = (u - cx) / fx * zz
    Y = (v - cy) / fy * zz
    return np.stack([X, Y, zz], -1), ok


def normal_map(V, ok, dmax=0.06):
    """顶点图 → 法线图（中心差分 + 深度跳变处屏蔽 + 统一朝向相机）"""
    N = np.zeros_like(V)
    dx = np.zeros_like(V)
    dy = np.zeros_like(V)
    dx[:, 1:-1] = V[:, 2:] - V[:, :-2]
    dy[1:-1, :] = V[2:, :] - V[:-2, :]
    z = V[..., 2]
    okx = ok.copy()
    okx[:, 1:-1] &= np.abs(z[:, 2:] - z[:, :-2]) < dmax
    okx[:, 0] = False
    okx[:, -1] = False
    oky = ok.copy()
    oky[1:-1, :] &= np.abs(z[2:, :] - z[:-2, :]) < dmax
    oky[0, :] = False
    oky[-1, :] = False
    good = okx & oky
    n = np.cross(dx, dy)
    ln = np.linalg.norm(n, axis=2)
    good &= ln > 1e-9
    n[good] /= ln[good][:, None]
    flip = good & ((n * V).sum(-1) > 0)      # 背对相机的翻过来
    n[flip] *= -1.0
    N[good] = n[good]
    return N, good


class Frame:
    """一帧深度 → 3 级金字塔（顶点/法线/掩码/该层内参）

    金字塔直接建在**抽稀后的深度**上（默认 stride=2 → 320x240 起），不是先算
    全分辨率再降采样：640x480 的顶点+法线要 73ms，而精度上并没有用 ——
    融合本来也只取 1/2 分辨率的点，ICP 最细层只用 2.4 万点。
    这样每帧 ~25ms，实时才够。
    """

    def __init__(self, z, K, stride=2, nlevels=3):
        self.levels = []
        for l in range(nlevels):
            s = stride * 2 ** l
            zz = np.ascontiguousarray(z[::s, ::s])
            Kl = np.array(K, np.float32)
            Kl[0, 0] /= s
            Kl[1, 1] /= s
            Kl[0, 2] = (K[0, 2] - (s - 1) * 0.5) / s
            Kl[1, 2] = (K[1, 2] - (s - 1) * 0.5) / s
            V, okv = vertex_map(zz, Kl)
            N, okn = normal_map(V, okv)
            self.levels.append(dict(V=V, N=N, okv=okv, okn=okn, K=Kl))


# ================================================================ ICP ===
def icp(prev, cur, T0, iters=(10, 6, 4)):
    """当前帧 → 上一帧 的相对位姿 T（4x4）。

    残差 r = n·(T·p_cur − q_prev)；未知增量作用在"上一帧坐标系"，雅可比行
    为 [p × n, n]；每层若干次高斯牛顿，带 LM 阻尼防奇异。
    """
    T = np.array(T0, np.float64)
    st = dict(inliers=0, rms=0.0, ok=False)
    for l in (2, 1, 0):
        P = prev.levels[l]
        C = cur.levels[l]
        Kl = P["K"]
        fx, fy, cx, cy = Kl[0, 0], Kl[1, 1], Kl[0, 2], Kl[1, 2]
        hh, ww = P["okv"].shape
        Vpf = P["V"].reshape(-1, 3)
        Npf = P["N"].reshape(-1, 3)
        oknf = P["okn"].ravel()
        src = C["V"][C["okv"]]
        if len(src) < 100:
            continue
        if l == 0 and len(src) > 24000:          # 最细层抽稀，省一半时间
            src = src[::len(src) // 24000 + 1]
        for _ in range(iters[2 - l]):
            R, t = T[:3, :3], T[:3, 3]
            p = src @ R.T + t
            z = p[:, 2]
            valid = z > 0.05
            iz = 1.0 / np.maximum(z, 1e-6)
            ui = np.rint(p[:, 0] * iz * fx + cx).astype(np.int32)
            vi = np.rint(p[:, 1] * iz * fy + cy).astype(np.int32)
            valid &= (ui >= 0) & (ui < ww) & (vi >= 0) & (vi < hh)
            if valid.sum() < 60:
                break
            ui = np.clip(ui, 0, ww - 1)
            vi = np.clip(vi, 0, hh - 1)
            idx = vi * ww + ui
            sel = np.flatnonzero(valid)
            sel = sel[oknf[idx[sel]]]            # 目标点必须有法线
            if len(sel) < 60:
                break
            q = Vpf[idx[sel]]
            n = Npf[idx[sel]]
            pp = p[sel]
            r = ((pp - q) * n).sum(1)
            keep = np.abs(r) < DMAX[2 - l]
            if keep.sum() < 50:
                break
            pp, n, r = pp[keep], n[keep], r[keep]
            A = np.concatenate([np.cross(pp, n), n], axis=1)
            ATA = A.T @ A
            ATb = -A.T @ r
            # 退化方向阻尼：正对白墙/大片地板时，横向平移在几何上没约束，
            # 硬解会"顺着墙滑走"（实测侧滑 37cm）。这里对 ATA 做特征分解，
            # 把远小于主方向的小特征值抬上去 —— 等价于在这些方向上相信
            # 匀速预测（增量趋 0），而不是让噪声决定往哪滑。
            w, Vv = np.linalg.eigh(ATA)
            wmax = float(w.max()) if len(w) else 0.0
            if wmax <= 0:
                break
            floor_w = 3e-3 * wmax
            w2 = np.where(w < floor_w, floor_w, w)
            xi = Vv @ ((Vv.T @ ATb) / (w2 + 1e-9 * wmax))
            T = se3_exp(xi) @ T
            if l == 0:
                st["inliers"] = int(len(r))
                st["rms"] = float(np.sqrt((r * r).mean()))
    st["ok"] = st["inliers"] >= MIN_INLIER
    return T, st


# ================================================================ 地图 ===
def voxel_dedupe(pts, col, v):
    """按体素去重（同格只留第一个点）—— 别让重复观测把点云堆到几百万"""
    if len(pts) == 0:
        return pts, col
    key = np.round(pts / v).astype(np.int32)
    kv = np.ascontiguousarray(key).view([('a', '<i4'), ('b', '<i4'),
                                        ('c', '<i4')]).ravel()
    _, idx = np.unique(kv, return_index=True)
    idx.sort()
    return pts[idx], col[idx]


class Map:
    def __init__(self, voxel=VOXEL_DEF, compact_at=900000):
        self.voxel = voxel
        self.compact_at = compact_at
        self.pts = np.zeros((0, 3), np.float32)
        self.col = np.zeros((0, 3), np.uint8)
        self.buf_p, self.buf_c = [], []
        self.traj = []
        self.fused = 0
        self._view = None

    def reset(self):
        keep = (self.voxel, self.compact_at)
        self.__init__(*keep)

    def add(self, pts, col, center):
        if len(pts):
            self.buf_p.append(pts)
            self.buf_c.append(col)
        self.traj.append(np.asarray(center, np.float32))
        self.fused += 1
        self._view = None
        if sum(len(p) for p in self.buf_p) > self.compact_at:
            self.compact()

    def compact(self):
        if not self.buf_p:
            return
        P = np.concatenate([self.pts] + self.buf_p)
        C = np.concatenate([self.col] + self.buf_c)
        self.pts, self.col = voxel_dedupe(P, C, self.voxel)
        self.buf_p, self.buf_c = [], []
        self._view = None

    def view(self):
        """给显示用的合并视图：不做去重（去重要 80ms+），格子着色本身就会合并"""
        if self._view is None:
            if self.buf_p:
                self._view = (np.concatenate([self.pts] + self.buf_p),
                              np.concatenate([self.col] + self.buf_c))
            else:
                self._view = (self.pts, self.col)
        return self._view

    def cloud(self):
        """保存用的去重视图"""
        P, C = self.view()
        return voxel_dedupe(P, C, self.voxel)


# ============================================================== 渲染 ===
def floor_basis(P, cell=0.25, tol=0.06):
    """从全局点云估地板平面 → (中心c, 轴e1, 轴e2, 法线n, d0)，失败返回 None。

    长扫描的姿态漂移会把世界系带歪，固定"水平切片"就切不到地板了 ——
    所以平面图要投影到**真实地板平面**上。做法：(x,z) 25cm 网格每格取
    最低点（y 最大），这批点大部分贴着地板；再对它做鲁棒 PCA 平面拟合
    （3 轮剔除离群，容差 6cm，被家具/噪声带歪的点会被丢掉）。
    n 保证朝"下"（y 分量为正）；e1 尽量贴着世界 x，e2 朝世界 +z 方向，
    这样平面图方向和旧版（右=x、下=前）一致。
    """
    if len(P) < 6000:
        return None
    key = np.ascontiguousarray(np.round(P[:, [0, 2]] / cell)).view(
        [("a", "<f4"), ("b", "<f4")]).ravel()
    uk, inv = np.unique(key, return_inverse=True)
    idx = np.lexsort((P[:, 1], inv))          # 组内按 y 升序
    last = np.searchsorted(inv[idx], np.arange(len(uk)), side="right") - 1
    cnts = np.bincount(inv, minlength=len(uk))
    keep_c = cnts >= 8
    if keep_c.sum() < 60:                     # 阈值按点云规模自适应，别写死 200
        return None
    pts = P[idx[last[keep_c]]]                # 每格最低点（剔除稀疏的边缘碎格：
                                              # 那种格子里常常只有天花板/墙点）
    # 播种：真实扫描的 y 被漂移抹开，纯随机 3 点 RANSAC 命中率低（实测
    # 155 格只剩 24% 内点）。先用"每格最低点 y 的众数"圈出最厚的一层
    # 当地板候选（地板是覆盖面积最大的最低面），再在这个邻域里随机采样
    # 做 RANSAC，最后 PCA 精修。
    lo_h, hi_h = float(pts[:, 1].min()), float(pts[:, 1].max())
    hist, edges = np.histogram(pts[:, 1], bins=np.arange(lo_h, hi_h + 0.06, 0.05))
    mb = int(np.argmax(hist))
    ymode = float(0.5 * (edges[mb] + edges[mb + 1]))
    seed = pts[np.abs(pts[:, 1] - ymode) < 0.15]
    if len(seed) < 30:
        seed = pts
    rng = np.random.default_rng(7)
    best = None
    for _ in range(600):
        s = seed[rng.choice(len(seed), 3, replace=False)]
        nv = np.cross(s[1] - s[0], s[2] - s[0])
        nn = float(np.linalg.norm(nv))
        if nn < 1e-9:
            continue
        nv /= nn
        if abs(nv[1]) < 0.5:                  # 地板法线不会歪离竖直 45° 以上
            continue
        d = pts @ nv - s[0] @ nv
        inl = int((np.abs(d) < tol).sum())
        if best is None or inl > best[0]:
            best = (inl, nv, float(s[0] @ nv))
    if best is None or best[0] < max(25, int(0.15 * len(pts))):
        return None
    nv, d0s = best[1], best[2]
    if nv[1] < 0:
        nv, d0s = -nv, -d0s
    pts = pts[np.abs(pts @ nv - d0s) < tol]   # 只留 RANSAC 内点再精修
    for _ in range(4):
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        n = vt[2]
        d = (pts - c) @ n
        keep = np.abs(d) < tol
        if keep.sum() < max(25, len(pts) // 3):
            return None
        if keep.all():
            break
        pts = pts[keep]
    if n[1] < 0:
        n = -n
    d0 = float(c @ n)
    e1 = np.array([1.0, 0, 0]) - n * n[0]
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([0.0, 0, 1]) - n * n[2]
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 = e2 / np.linalg.norm(e2)
    if e2[2] < 0:
        e2 = -e2
    return c, e1, e2, n, d0


def make_snapshot(m, snap, floor=None):
    P, _C = m.view()
    if len(P) == 0:
        snap.update(P=P, box=(0.0, 1.0, 0.0, 1.0), ylo=0.0, yhi=1.0)
        return snap
    S = P[::max(1, len(P) // 120000)]        # 分位数只看抽样，省时间
    snap["P"] = P
    if floor is not None:                    # 视野自适应改在地板平面坐标系里算
        c, e1, e2, n, d0 = floor
        us, vs = S @ e1, S @ e2
        hs = S @ n - d0
        snap["hlo"], snap["hhi"] = (float(np.percentile(hs, 2)),
                                    float(np.percentile(hs, 98)))
    else:
        us, vs = S[:, 0], S[:, 2]
    snap["box"] = (float(np.percentile(us, 1)), float(np.percentile(us, 99)),
                   float(np.percentile(vs, 1)), float(np.percentile(vs, 99)))
    ylo, yhi = np.percentile(S[:, 1], [2, 98])
    if yhi - ylo < 0.4:
        yhi = ylo + 0.4
    snap["ylo"], snap["yhi"] = float(ylo), float(yhi)
    return snap


def render_plan(img, snap, traj, G, show_grid=True, band=None, floor=None):
    """俯视平面图：5cm 格按高度着色 + 轨迹 + 相机视锥 + 1m 比例尺

    band=(中心, 半带宽) 时只渲染该高度带内的点 —— 不切片的话地板和
    天花板叠在同一格取均值，整张图就是彩虹糊团（实测教训）。
    floor 给定时整个图投影在**地板平面**上：水平坐标 = (P·e1, P·e2)，
    "高度" = P·n − d0（地板上方高度），切片也沿地板法线切 —— 长扫描
    姿态漂移把世界系带歪后，只有这样才能切出真正的地板。
    """
    h, w = img.shape[:2]
    img[:] = BG
    P = snap["P"]
    if floor is not None:
        _c, e1, e2, n, d0 = floor
        px, py, ph = P @ e1, P @ e2, P @ n - d0
        if band is not None and len(P):
            sel = np.abs(ph - band[0]) <= band[1]
            px, py, ph = px[sel], py[sel], ph[sel]
        t = np.asarray(traj, np.float32).reshape(-1, 3)
        tpx, tpy = t @ e1, t @ e2
        gp = G[:3, 3]
        gx, gy = float(gp @ e1), float(gp @ e2)
        f3 = G[:3, 2]
        fx, fz = float(f3 @ e1), float(f3 @ e2)      # 光轴在平面上的投影
    else:
        px, py, ph = P[:, 0], P[:, 2], P[:, 1]
        if band is not None and len(P):
            sel = np.abs(ph - band[0]) <= band[1]
            px, py, ph = px[sel], py[sel], ph[sel]
        t = np.asarray(traj, np.float32).reshape(-1, 3)
        tpx, tpy = t[:, 0], t[:, 2]
        gx, gy = float(G[0, 3]), float(G[2, 3])
        fx, fz = float(G[0, 2]), float(G[2, 2])
    if len(P) == 0 and len(traj) == 0:
        cv2.putText(img, "开始扫描：端着相机慢慢走一圈，边扫边看", (30, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (130, 136, 148), 1, cv2.LINE_AA)
        cv2.putText(img, "每帧位移别超过 10cm；转身慢一点，别让画面全是白墙",
                    (30, h // 2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 106, 118), 1,
                    cv2.LINE_AA)
        return img

    xlo, xhi, zlo, zhi = snap["box"]
    pad = 0.8
    dxm = max(xhi - xlo, 1.2) + 2 * pad
    dzm = max(zhi - zlo, 1.2) + 2 * pad
    ppm_t = float(np.clip(min((w - 40) / dxm, (h - 40) / dzm), 8.0, 150.0))
    if snap.get("ppm") is None:
        snap["ppm"] = ppm_t
        snap["cx"] = (xlo + xhi) * 0.5
        snap["cz"] = (zlo + zhi) * 0.5
    else:                                    # 平滑一下，别一冒点就跳视野
        snap["ppm"] = 0.9 * snap["ppm"] + 0.1 * ppm_t
        snap["cx"] = 0.9 * snap["cx"] + 0.1 * (xlo + xhi) * 0.5
        snap["cz"] = 0.9 * snap["cz"] + 0.1 * (zlo + zhi) * 0.5
    ppm, cxw, czw = snap["ppm"], snap["cx"], snap["cz"]

    def w2s(x, z):
        return int(w * 0.5 + (x - cxw) * ppm), int(h * 0.5 + (z - czw) * ppm)

    if show_grid:
        step = 1.0 if ppm < 55 else 0.5
        x0 = np.ceil((cxw - w * 0.5 / ppm) / step) * step
        gx = x0
        while gx < cxw + w * 0.5 / ppm:
            sx, _ = w2s(gx, czw)
            if 0 <= sx < w:
                cv2.line(img, (sx, 0), (sx, h), (34, 36, 42), 1)
                if abs(gx - round(gx)) < 1e-6:
                    cv2.putText(img, f"{gx:+.0f}m", (sx + 3, h - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (72, 76, 86), 1)
            gx += step
        z0 = np.ceil((czw - h * 0.5 / ppm) / step) * step
        gz = z0
        while gz < czw + h * 0.5 / ppm:
            _, sy = w2s(cxw, gz)
            if 0 <= sy < h:
                cv2.line(img, (0, sy), (w, sy), (34, 36, 42), 1)
                if abs(gz - round(gz)) < 1e-6:
                    cv2.putText(img, f"{gz:+.0f}m", (4, sy - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (72, 76, 86), 1)
            gz += step

    # --- 点 → 5cm 格子的平均高度。直接画散点会被地面糊死，取均值才有"立面"
    if len(px):
        sx = (px - cxw) * ppm + w * 0.5
        sy = (py - czw) * ppm + h * 0.5
        cpx = max(1, int(round(CELL * ppm)))
        ncx, ncy = int(w // cpx) + 2, int(h // cpx) + 2
        ix = np.floor(sx / cpx).astype(np.int64)
        iy = np.floor(sy / cpx).astype(np.int64)
        ins = (ix >= 0) & (ix < ncx) & (iy >= 0) & (iy < ncy)
        if ins.any():
            k = iy[ins] * ncx + ix[ins]
            cnt = np.bincount(k, minlength=ncx * ncy)
            acc = np.bincount(k, weights=ph[ins], minlength=ncx * ncy)
            kk = np.flatnonzero(cnt > 0)
            ym = acc[kk] / cnt[kk]
            if band is not None:                 # 切片模式：颜色只在该高度带内展开
                g = np.clip((ym - (band[0] - band[1])) / (2 * band[1]), 0, 1)
            else:                                # 全高度视图：按快照里的高度分位
                hlo = snap.get("hlo", snap["ylo"])
                hhi = snap.get("hhi", snap["yhi"])
                g = np.clip((ym - hlo) / max(1e-6, hhi - hlo), 0, 1)
            small = np.zeros((ncy, ncx, 3), np.uint8)
            small[kk // ncx, kk % ncx] = _LUT[(g * 255).astype(np.int32)]
            big = cv2.resize(small, (ncx * cpx, ncy * cpx),
                             interpolation=cv2.INTER_NEAREST)
            mm = cv2.resize(small.any(2).astype(np.uint8), (ncx * cpx, ncy * cpx),
                            interpolation=cv2.INTER_NEAREST)
            hh, ww = min(h, big.shape[0]), min(w, big.shape[1])
            sel = mm[:hh, :ww] > 0
            view = img[:hh, :ww]
            view[sel] = big[:hh, :ww][sel]

    if len(tpx) > 1:
        step = max(1, len(tpx) // 500)
        pts2 = np.stack([w2s(u, v) for u, v in zip(tpx[::step], tpy[::step])], 0)
        cv2.polylines(img, [pts2.astype(np.int32)], False, (60, 200, 255), 2,
                      cv2.LINE_AA)
    if len(tpx):
        sx, sy = w2s(tpx[-1], tpy[-1])
        f = np.array([fx, fz])                    # 光轴在平面上的投影
        nf = float(np.linalg.norm(f))
        if nf > 1e-6:
            f /= nf
            rgt = np.array([-f[1], f[0]])
            c0 = np.array([sx, sy])
            tri = np.array([c0 + f * 28, c0 + rgt * 14 - f * 7,
                            c0 - rgt * 14 - f * 7], np.int32)
            cv2.fillConvexPoly(img, tri, (90, 235, 150))
        cv2.circle(img, (sx, sy), 4, (255, 255, 255), -1, cv2.LINE_AA)

    bar = max(10, int(round(1.0 * ppm)))
    bx, by = 24, h - 34
    cv2.line(img, (bx, by), (bx + bar, by), (200, 205, 215), 2)
    for ex in (bx, bx + bar):
        cv2.line(img, (ex, by - 5), (ex, by + 5), (200, 205, 215), 2)
    cv2.putText(img, "1 m", (bx + bar + 8, by + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 205, 215), 1, cv2.LINE_AA)
    cv2.putText(img, "TOP VIEW   cell 5cm", (24, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 156, 168), 1, cv2.LINE_AA)
    if band is not None:
        col = (90, 235, 150) if len(px) else (80, 90, 255)
        base = "floor" if floor is not None else "y"
        cv2.putText(img, f"slice {base} {band[0]:+.2f} +/-{band[1]:.2f} m   "
                         "[ ] move  - = width", (24, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    return img


def draw_stats(img, lines, x0, y0, dy=22):
    for i, (txt, col) in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(img, txt, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(img, txt, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1,
                    cv2.LINE_AA)


# ============================================================ 保存 ===
def save_all(m: Map, out_base, meta, band=None, floor=None):
    os.makedirs(out_base, exist_ok=True)
    P, C = m.cloud()
    if len(P):
        save_ply(os.path.join(out_base, "cloud.ply"), P, C)
    snap = make_snapshot(m, dict(ppm=None), floor)
    plan = np.zeros((1080, 1080, 3), np.uint8)
    render_plan(plan, snap, m.traj, np.eye(4), True, band=band, floor=floor)
    cp.imwrite_u(os.path.join(out_base, "plan.png"), plan)
    if band is not None:                       # 同时存一张全高度的，方便对照
        plan2 = np.zeros((1080, 1080, 3), np.uint8)
        render_plan(plan2, snap, m.traj, np.eye(4), True, floor=floor)
        cp.imwrite_u(os.path.join(out_base, "plan_full.png"), plan2)
    if len(m.traj):
        with open(os.path.join(out_base, "traj.csv"), "w", encoding="utf-8") as f:
            f.write("x,y,z\n")
            for p in m.traj:
                f.write(f"{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}\n")
    path = 0.0
    if len(m.traj) > 1:
        path = float(np.linalg.norm(np.diff(np.array(m.traj, np.float32), axis=0),
                                    axis=1).sum())
    meta = dict(meta)
    meta.update(points=int(len(P)), keyframes=int(len(m.traj)), path_m=path)
    with open(os.path.join(out_base, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return out_base


# ============================================================== 主流程 ===
def main():
    ap = argparse.ArgumentParser(description="室内扫描建模：点云拼接 + 平面图")
    ap.add_argument("--voxel", type=float, default=VOXEL_DEF)
    ap.add_argument("--out", default="")
    ap.add_argument("--sec", type=float, default=0.0, help="跑多少秒后自动保存退出")
    ap.add_argument("--stride", type=int, default=2, help="融合时的深度抽稀步长")
    ap.add_argument("--view3d", action="store_true", help="启动即开 3D 预览")
    ap.add_argument("--no-window", action="store_true", help="不开窗口（管道自测用）")
    ap.add_argument("--no-autosave", action="store_true",
                    help="退出时不自动保存（默认退出就存，省得忘了按 s）")
    args = ap.parse_args()

    depth = cp.DepthSource()
    depth.start()
    color = cp.ColorSourceFFmpeg(1280, 720, 30, device=cp.FFMPEG_DEVICE)
    color.start()
    cal = load_calib(latest_calib())
    cal, dsz, csz = adapt_cal(cal, depth.mode, color.mode)
    align = Aligner(cal, dsz, csz)
    Kd = np.array(cal["K_d"], np.float32)
    print(f"深度 {dsz[1]}x{dsz[0]}  彩色 {csz[1]}x{csz[0]}  "
          f"fx={Kd[0, 0]:.1f} cx={Kd[0, 2]:.1f}", flush=True)

    m = Map(args.voxel)
    G = np.eye(4)
    prev = None
    T_pred = np.eye(4)
    G_fused = np.eye(4)
    lost = 0
    paused = False
    show3d = args.view3d
    show_grid = True
    curve = CloudView(320, 180, fov_deg=55)
    curve.bg = BG
    preview = None
    frame_i = 0
    prev_ts = None
    fps = 0.0
    t_prev = time.time()
    t0 = t_prev
    stats = dict(inliers=0, rms=0.0, ok=False)
    snap = dict(P=np.zeros((0, 3), np.float32), box=(0, 1, 0, 1),
                ylo=0.0, yhi=1.0, ppm=None, cx=0.0, cz=0.0)
    band = [0.10, 0.20]     # [切片中心, 半带宽]：地板坐标系里的"地板上方高度"
    fl = None               # 地板平面 (c,e1,e2,n,d0)；点攒够后周期性重估
    npts = 0
    out_path = args.out or os.path.join(CAP, time.strftime("scan_%Y%m%d_%H%M%S"))

    win = "Indoor Scan  [space=pause  s=save  r=reset  v=3d  q=quit]"
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    print("开始扫描：端着相机慢慢走一圈（每帧位移 <10cm），边扫边看平面图。", flush=True)

    def grab_color(cz):
        if cz is None:
            return None
        return (cp.ColorSourceFFmpeg.decode(cz[0])
                if getattr(color, "is_jpeg", False) else cz[0])

    try:
        while True:
            dz = depth.latest()
            cz = color.latest()
            if dz is None:
                time.sleep(0.004)
                continue
            ts = dz[1] if len(dz) > 1 else None
            if ts is not None and ts == prev_ts:
                time.sleep(0.003)
                continue
            prev_ts = ts
            frame_i += 1
            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            # 深度是硬件水平镜像的：先翻正。K_d 是在翻正后的图上标的，
            # 不翻的话配准出来的 x 和彩色也对不上
            d = cv2.flip(dz[0], 1)
            fr = Frame(d.astype(np.float32) * 0.001, Kd, args.stride)

            # ------------------ 配准 ------------------
            if prev is not None and not paused:
                T_rel, stats = icp(prev, fr, T_pred)
                dt_t = float(np.linalg.norm(T_rel[:3, 3]))
                dt_r = rot_angle(T_rel)
                if stats["ok"] and dt_t < 0.45 and dt_r < 25.0:
                    G = G @ T_rel
                    T_pred = T_rel
                    lost = 0
                else:
                    lost += 1
                    T_pred = np.eye(4)      # 丢掉速度模型，下帧从"没动"重猜
                    stats["ok"] = False
            prev = fr

            # ------------------ 融合 ------------------
            moved = (float(np.linalg.norm(G[:3, 3] - G_fused[:3, 3])) > 0.012
                     or rot_angle(G[:3, :3].T @ G_fused[:3, :3]) > 0.6)
            if not paused and lost == 0 and moved:
                V = fr.levels[0]["V"].reshape(-1, 3)      # 已是抽稀后的网格
                keep = (V[:, 2] > MIN_D) & (V[:, 2] < MAX_D)
                rgb = np.zeros((len(V), 3), np.uint8)
                proj = align.warp(d)[1] if keep.any() else None
                if proj is not None:
                    uc, vc, _zc, msk = proj
                    keep &= msk[::args.stride, ::args.stride].ravel()
                    img = grab_color(cz) if keep.any() else None
                    if img is not None and keep.any():
                        ui = np.clip(uc[::args.stride, ::args.stride].ravel()[keep].astype(np.int32),
                                     0, img.shape[1] - 1)
                        vi = np.clip(vc[::args.stride, ::args.stride].ravel()[keep].astype(np.int32),
                                     0, img.shape[0] - 1)
                        rgb[keep] = img[vi, ui][:, ::-1]
                if keep.any():
                    Pw = V[keep] @ G[:3, :3].T + G[:3, 3]
                    m.add(Pw.astype(np.float32), rgb[keep], G[:3, 3])
                    G_fused = G.copy()

            if frame_i % SNAP_EVERY == 1:
                make_snapshot(m, snap, fl)
                npts = len(snap["P"])
                # 地板平面重估：长扫描的姿态漂移会把世界系带歪，固定水平
                # 切片切不到地板（实机 36 分钟扫描实测）—— 周期性从点云
                # 重新拟合地板平面，平面图和切片都贴着真实地板走
                if frame_i % 60 == 1 and npts > 15000:
                    nf = floor_basis(snap["P"][::max(1, npts // 60000)])
                    if nf is not None:
                        if fl is None:
                            print(f"地板平面已锁定（法线 {nf[3].round(2)}），"
                                  "平面图投影到地板上、切片沿地板法线", flush=True)
                        fl = nf

            # ------------------ 画面 ------------------
            canvas = np.zeros((720, 1280, 3), np.uint8)
            render_plan(canvas[:, :956], snap, m.traj, G, show_grid,
                        band=tuple(band) if fl is not None else None, floor=fl)

            live = grab_color(cz)
            live = (cv2.resize(live, (320, 180), interpolation=cv2.INTER_AREA)
                    if live is not None else np.full((180, 320, 3), 24, np.uint8))
            canvas[0:180, 956:1276] = live
            cv2.rectangle(canvas, (956, 0), (1275, 179), (60, 64, 72), 1)
            if lost:
                cv2.putText(canvas, f"LOST {lost}f", (964, 172),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 90, 255), 2, cv2.LINE_AA)

            if show3d:
                if frame_i % 3 == 1 and npts:
                    P3, C3 = m.view()
                    if len(P3) > 240000:
                        st3 = len(P3) // 240000 + 1
                        P3, C3 = P3[::st3], C3[::st3]
                    curve.yaw = -float(np.arctan2(G[2, 0], G[0, 0]))
                    curve.pitch = 0.72
                    preview = curve.render(P3, C3, depth_mode=False, axes=False)
                if preview is not None:
                    canvas[188:368, 956:1276] = preview
                    cv2.rectangle(canvas, (956, 188), (1275, 367), (60, 64, 72), 1)

            col_ok = (90, 235, 150) if (lost == 0 and stats["ok"]) else (80, 90, 255)
            y0 = 400 if show3d else 214
            lines = [
                (f"status  {'PAUSED' if paused else ('LOST' if lost else 'TRACKING')}", col_ok),
                (f"frames  {frame_i}   fused {m.fused}", (215, 218, 226)),
                (f"points  {npts / 1000:.0f}k   voxel {args.voxel * 100:.0f}cm",
                 (215, 218, 226)),
                (f"inliers {stats['inliers']}  rms {stats['rms'] * 1000:.1f}mm",
                 (200, 204, 212)),
                (f"pose    x{G[0, 3]:+.2f} y{G[1, 3]:+.2f} z{G[2, 3]:+.2f}", (200, 204, 212)),
                (f"fps     {fps:4.1f}", (160, 166, 178)),
            ]
            draw_stats(canvas, lines, 966, y0)
            for i, s in enumerate(["space pause", "s save", "r reset", "v 3d",
                                   "g grid", "q quit", "[ ] slice", "- = width"]):
                cv2.putText(canvas, s, (966 + (i % 2) * 130, 700 - (i // 2) * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 146, 158), 1,
                            cv2.LINE_AA)

            if not args.no_window:
                cv2.imshow(win, canvas)
                k = wait_key(win, 1)
            else:
                k = -1
            if k in (27, ord("q")):
                break
            if k == ord(" "):
                paused = not paused
            elif k == ord("g"):
                show_grid = not show_grid
            elif k == ord("v"):
                show3d = not show3d
                preview = None
            elif k == ord("r"):
                m.reset()
                G = np.eye(4)
                prev = None
                T_pred = np.eye(4)
                G_fused = np.eye(4)
                lost = 0
                snap["ppm"] = None
                fl = None                 # 重扫后地板平面要重新拟合
                print("地图已清空", flush=True)
            elif k == ord("["):
                band[0] -= 0.10
            elif k == ord("]"):
                band[0] += 0.10
            elif k == ord("-"):
                band[1] = max(0.05, band[1] - 0.05)
            elif k == ord("="):
                band[1] = min(1.20, band[1] + 0.05)
            elif k == ord("s"):
                save_all(m, out_path, dict(frames=frame_i, voxel=args.voxel,
                                           fx=float(Kd[0, 0]), cx=float(Kd[0, 2])),
                         tuple(band) if fl is not None else None, fl)
                print("saved", out_path, flush=True)
            if args.sec and time.time() - t0 > args.sec:
                save_all(m, out_path, dict(frames=frame_i, voxel=args.voxel,
                                           fx=float(Kd[0, 0]), cx=float(Kd[0, 2])),
                         tuple(band) if fl is not None else None, fl)
                print(f"{args.sec:.0f}s 到点，已保存 {out_path}  点数 {npts}", flush=True)
                break
    finally:
        cv2.destroyAllWindows()
        # 退出就存：扫完直接关窗口是最自然的动作，别让整趟数据烂在内存里。
        # 按过 s 再退出会覆盖一次（内容更全），不影响。
        if not args.no_autosave and m.fused:
            try:
                save_all(m, out_path, dict(frames=frame_i, voxel=args.voxel,
                                           fx=float(Kd[0, 0]), cx=float(Kd[0, 2]),
                                           auto=True),
                         tuple(band) if fl is not None else None, fl)
                print(f"退出自动保存 {out_path}  （{m.fused} 帧融合）", flush=True)
            except Exception as e:                            # noqa: BLE001
                print(f"自动保存失败：{e}（数据丢失，下次请按 s）", flush=True)
        color.close()
        depth.close()


if __name__ == "__main__":
    main()
