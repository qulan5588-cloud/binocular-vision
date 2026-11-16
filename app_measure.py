"""app_measure.py —— 第三个应用：物体平面尺寸测量

干什么：在画面上拖一个框，程序把框内的深度点拟合成一个平面，再在平面内
求最小外接矩形，直接给出 **长 × 宽（毫米）**、平面到相机的距离、倾角。

为什么不是"点两个点量距离"：两点量的是"你点的这两个像素之间"，手一抖误差
就进结果，而且只能量直线。平面法更适合真实场景 —— 贴平面物体（纸张、板材、
箱面、零件底面）的尺寸和摆放角度一次全出来，平面拟合后面做抓取定位也用得上。

算法三步：
  1. RANSAC 拟合平面：3 点定平面 → 数内点 → 取最优 → SVD 精化法向。
     阈值默认 3mm（≈1m 处深度噪声 1σ）：太小把平面切碎，太大把背景吃进来。
     为保实时，RANSAC 只在 ≤4000 点的子集上跑，内点判定和精化用全量点。
  2. 平面内建正交基 (e1,e2)，把内点投成 2D。
  3. 2D 凸包 + 最小外接旋转矩形（cv2.minAreaRect）→ 宽高即尺寸。

精度预期（规格书 1m 处 3mm）：1m 距离量 A4 纸，长边误差应在 1% 内。
量具对答案：拿已知尺寸的纸/板比一比。

用法：
    python app_measure.py
    python app_measure.py --thresh 0.004     # 远处/噪声大时放宽到 4mm
    python app_measure.py --rect 420,180,860,560 --shot out.png   # 无人值守出图

操作：
    左键拖框 = 测量（松开即算）   u = 按当前帧重算   r = 清除
    m = 彩色/深度叠加切换   s = 存图   p = 存 PLY（平面内的点）   q = 退出
"""
import argparse
import os
import time

import cv2
import numpy as np

import capture_pair as cp
from app_align import fit_K, latest_calib, load_calib
from app_pointcloud import CloudBuilder, project_to_image, save_ply

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "captures")


# --------------------------------------------------------------- 平面拟合 ---
def fit_plane(pts, thresh=0.003, iters=300, seed=0, max_sample=4000):
    """RANSAC 拟平面 → 全量内点 + SVD 精化。

    返回 (法向 n, 平面中心 p0, 内点布尔掩码, 内点 RMS 米)；点太少返回 None。
    """
    n_pts = len(pts)
    if n_pts < 50:
        return None
    rng = np.random.default_rng(seed)
    # 子集采样：框内可能几万点，全量子集跑 300 次代价太高
    sub = pts if n_pts <= max_sample else pts[rng.choice(n_pts, max_sample, replace=False)]
    m = len(sub)
    triples = rng.integers(0, m, (iters, 3))
    best_cnt, best = 0, None
    for a, b, c in triples:
        p0, p1, p2 = sub[a], sub[b], sub[c]
        nv = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nv)
        if ln < 1e-9:
            continue
        nv = nv / ln
        cnt = int((np.abs((sub - p0) @ nv) < thresh).sum())
        if cnt > best_cnt:
            best_cnt, best = cnt, (nv, p0)
    if best is None:
        return None

    # 全量判定内点，再用内点做最小二乘（SVD 最小奇异向量 = 法向）
    nv, p0 = best
    inl = np.abs((pts - p0) @ nv) < thresh
    if inl.sum() < 50:
        return None
    P = pts[inl]
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    nv = vt[2] / np.linalg.norm(vt[2])
    if nv[2] > 0:                        # 统一让法向朝相机（Z 分量为负）
        nv = -nv
    resid = (P - c) @ nv
    return nv, c, inl, float(np.sqrt((resid ** 2).mean()))


def plane_dims(P, nv, p0, q=0.01):
    """平面内点 → 尺寸/角点/倾角。

    ★ 为什么不用 cv2.minAreaRect：它求的是包住**所有点**的矩形，深度噪声会把
      极值点推到真实边界之外（实测 3mm 噪声就让 A4 偏大 11mm）。改成"主轴对齐后
      取 1%/99% 分位"：均匀分布下分位点几乎不受噪声影响，再按 0.98 反推真实跨度，
      误差降到 0.1mm 量级（见 _detect/measure_selftest.py）。
    """
    ax = np.array([1.0, 0, 0])
    if abs(nv @ ax) > 0.9:
        ax = np.array([0, 1.0, 0])
    e1 = np.cross(nv, ax)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nv, e1)
    rel = P - p0
    ab = np.stack([rel @ e1, rel @ e2], axis=1)

    # 主轴（PCA）：矩形点云的协方差主轴就是边的方向，比 minAreaRect 稳
    c2 = ab.mean(axis=0)
    cov = np.cov((ab - c2).T)
    _, evecs = np.linalg.eigh(cov)
    main = evecs[:, int(np.argmax(np.linalg.eigvalsh(cov)))]
    th = float(np.arctan2(main[1], main[0]))
    ct, st = np.cos(th), np.sin(th)
    rot = (ab - c2) @ np.array([[ct, -st], [st, ct]])      # 转到轴对齐

    lo = np.percentile(rot, q * 100.0, axis=0)
    hi = np.percentile(rot, (1.0 - q) * 100.0, axis=0)
    size = (hi - lo) / (1.0 - 2.0 * q)                     # 反推真实跨度
    mid = (lo + hi) * 0.5
    lo2, hi2 = mid - size * 0.5, mid + size * 0.5          # 与实际尺寸一致的角点
    corners_rot = np.array([[lo2[0], lo2[1]], [hi2[0], lo2[1]],
                            [hi2[0], hi2[1]], [lo2[0], hi2[1]]])
    back = np.array([[ct, st], [-st, ct]])
    corners_ab = corners_rot @ back.T + c2
    corners = p0 + corners_ab[:, 0:1] * e1 + corners_ab[:, 1:2] * e2
    center = p0 + c2[0] * e1 + c2[1] * e2
    long_m, short_m = float(max(size)), float(min(size))
    return dict(long_mm=long_m * 1000.0, short_mm=short_m * 1000.0,
                corners3d=corners, center3d=center,
                tilt_deg=float(np.degrees(np.arccos(min(1.0, abs(nv[2]))))))


def measure_region(pts, uc, vc, box, thresh=0.003):
    """框选 → 平面 → 尺寸。box=(x0,y0,x1,y1) 为彩色图像素坐标"""
    x0, y0, x1, y1 = box
    sel = (uc >= x0) & (uc <= x1) & (vc >= y0) & (vc <= y1)
    n_sel = int(sel.sum())
    if n_sel < 200:
        return None
    P = pts[sel]
    fp = fit_plane(P, thresh=thresh)
    if fp is None:
        return None
    nv, c, inl, rms = fp
    Q = P[inl]
    d = plane_dims(Q, nv, c)
    d.update(n_pts=n_sel, n_inl=int(inl.sum()),
             inl_ratio=float(inl.sum() / n_sel), rms_mm=rms * 1000.0,
             dist_mm=float(np.median(Q[:, 2]) * 1000.0),
             inl_mask=np.flatnonzero(sel)[inl],      # 相对 pts 的索引，用于画点/存盘
             box=box)
    return d


# ---------------------------------------------------------------- 叠加绘制 ---
def draw(canvas, state, cal, pts, uc, vc):
    b = state["box"]
    if b:
        cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), (90, 200, 255), 1, cv2.LINE_AA)
    r = state["res"]
    if not r:
        return
    K, D = cal["K_c"], cal["D_c"]
    # 平面内点：绿点，一眼看出程序认为哪些点属于这个平面
    idx = r["inl_mask"]
    canvas[vc[idx].astype(int), uc[idx].astype(int)] = (80, 230, 80)
    # 外接矩形：3D 角点投回图像
    u, v, ok = project_to_image(r["corners3d"], K, D, canvas.shape[1], canvas.shape[0])
    if ok.all():
        for i in range(4):
            p1 = (int(u[i]), int(v[i]))
            p2 = (int(u[(i + 1) % 4]), int(v[(i + 1) % 4]))
            cv2.line(canvas, p1, p2, (0, 235, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, p1, 3, (0, 120, 255), -1, cv2.LINE_AA)
    # 尺寸大字
    cu, cvv, _ = project_to_image(r["center3d"].reshape(1, 3), K, D,
                                  canvas.shape[1], canvas.shape[0])
    tx = int(np.clip(cu[0] - 120, 8, canvas.shape[1] - 360))
    ty = int(np.clip(cvv[0], 60, canvas.shape[0] - 40))
    big = f"{r['long_mm']:.1f} x {r['short_mm']:.1f} mm"
    cv2.putText(canvas, big, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(canvas, big, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 235, 255), 2, cv2.LINE_AA)
    detail = (f"dist {r['dist_mm']:.0f}mm  tilt {r['tilt_deg']:.1f}deg  "
              f"inlier {r['inl_ratio'] * 100:.0f}%  rms {r['rms_mm']:.2f}mm  n={r['n_inl']}")
    cv2.putText(canvas, detail, (tx, ty + 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, detail, (tx, ty + 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ------------------------------------------------------------------ 主流程 ---
def main():
    ap = argparse.ArgumentParser(description="平面尺寸测量")
    ap.add_argument("--calib", default="")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--backend", default="ffmpeg", choices=["ffmpeg", "dshow"])
    ap.add_argument("--device", default=cp.FFMPEG_DEVICE)
    ap.add_argument("--thresh", type=float, default=0.003, help="平面内点阈值(米)")
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--depth", action="store_true", help="叠加深度伪彩（诊断空洞用）")
    ap.add_argument("--update-every", type=int, default=10, help="每隔多少帧重算一次")
    ap.add_argument("--shot", default="", help="存一张结果图后退出")
    ap.add_argument("--rect", default="", help="预设框 x0,y0,x1,y1（配 --shot 无人值守）")
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

    dsz = (depth.mode[1], depth.mode[0])
    csz = (color.mode[1], color.mode[0])
    cal["K_d"] = fit_K(cal["K_d"], cal["size_d"], dsz)
    cal["K_c"] = fit_K(cal["K_c"], cal["size_c"], csz)
    builder = CloudBuilder(cal, dsz, csz)

    state = dict(drag=False, sx=0, sy=0, cx=0, cy=0, box=None, res=None,
                 show_depth=args.depth, refresh=True)
    if args.rect:
        state["box"] = tuple(int(v) for v in args.rect.split(","))

    win = "Measure  [drag=box  u=refresh  r=clear  m=overlay  s=save  p=ply  q=quit]"

    def on_mouse(ev, x, y, flags, _p):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state.update(drag=True, sx=x, sy=y, cx=x, cy=y, res=None)
        elif ev == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state["cx"], state["cy"] = x, y
        elif ev == cv2.EVENT_LBUTTONUP and state["drag"]:
            state.update(drag=False, cx=x, cy=y, refresh=True)
            b = (min(state["sx"], x), min(state["sy"], y),
                 max(state["sx"], x), max(state["sy"], y))
            state["box"] = b if (b[2] - b[0] > 8 and b[3] - b[1] > 8) else None

    if not args.shot:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, on_mouse)
        print(f"标定 {os.path.relpath(cpath, HERE)} | 拖框测量   u 重算   r 清除   q 退出")

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

            # 框选后：立刻算一次，之后每 update_every 帧跟随刷新（物体动了结果跟着动）
            if state["box"] and (state["refresh"] or n % args.update_every == 0):
                state["res"] = measure_region(pts, uc, vc, state["box"], args.thresh)
                state["refresh"] = False

            canvas = img.copy()
            if state["show_depth"]:
                m = np.zeros(csz, np.uint8)      # csz=(h,w)
                g = np.clip((pts[:, 2] - 0.3) / 5.7, 0, 1)
                m[vc.astype(int), uc.astype(int)] = (g * 255).astype(np.uint8)
                canvas = cv2.addWeighted(canvas, 1 - args.alpha,
                                         cv2.applyColorMap(m, cv2.COLORMAP_TURBO),
                                         args.alpha, 0)

            draw(canvas, state, cal, pts, uc, vc)

            n += 1
            if n % 5 == 0:
                now = time.time()
                fps = 0.8 * fps + 0.2 * (5.0 / max(now - t_prev, 1e-6))
                t_prev = now
            head = f"{fps:4.1f} fps | pts {len(pts) // 1000}k | valid {ratio * 100:4.1f}%"
            cv2.putText(canvas, head, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, head, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (210, 240, 210), 1, cv2.LINE_AA)

            if args.shot:
                cp.imwrite_u(args.shot, canvas)
                r = state["res"]
                if r:
                    print(f"{r['long_mm']:.1f} x {r['short_mm']:.1f} mm | 距 {r['dist_mm']:.0f} mm"
                          f" | 倾角 {r['tilt_deg']:.1f}° | 内点 {r['inl_ratio'] * 100:.0f}%"
                          f" | RMS {r['rms_mm']:.2f} mm")
                else:
                    print("!! 框内没测出平面（点太少或深度大片空洞）")
                print("saved " + args.shot)
                break

            cv2.imshow(win, canvas)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if k == ord("r"):
                state.update(box=None, res=None)
            elif k == ord("u"):
                state["refresh"] = True
            elif k == ord("m"):
                state["show_depth"] = not state["show_depth"]
            elif k == ord("s"):
                fn = os.path.join(OUT_DIR, time.strftime("measure_%H%M%S.png"))
                cp.imwrite_u(fn, canvas)
                print("saved captures/" + os.path.basename(fn))
            elif k == ord("p") and state["res"]:
                r = state["res"]
                fn = os.path.join(OUT_DIR, time.strftime("measure_%H%M%S.ply"))
                save_ply(fn, pts[r["inl_mask"]], rgb[r["inl_mask"]])
                print(f"saved captures/{os.path.basename(fn)} ({r['n_inl']} pts)")
    finally:
        if not args.shot:
            cv2.destroyAllWindows()
        color.close()
        depth.close()


if __name__ == "__main__":
    main()
