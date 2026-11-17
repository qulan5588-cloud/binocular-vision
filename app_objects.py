"""app_objects.py —— 第五个应用：物体分割 / 计数 / 粗尺寸

干什么：把视场里的物体一块块分出来，给每块标上序号、距离、宽×高、体积估算，
并统计一共几个。这是"货位有没有货、有几个"的最小可用版本。

为什么不用 3D 点云欧式聚类：
    3D 聚类要么引 open3d（重依赖），要么自己写体素+并查集（慢、参数难调）。
    而深度图天生是 2D 网格，**相邻像素深度差超过阈值就是物体边界**——
    一次 connectedComponents 就把物体分开了，几毫秒完成，比 3D 聚类快两个数量级。
    代价是"贴在一起且深度连续"的两个物体算一个，实际场景够用。

为什么先剔平面：
    桌面、地面、背后墙面都是巨大的连通块。不剔掉的话：① 它们占满统计列表；
    ② 放在桌面上的两个东西会被桌面连成一片。所以先 RANSAC 找最大平面挖掉，
    剩下的才是"摆在平面上的东西"。（--keep-plane 可关闭）

为什么不取 min/max 求尺寸：
    和 app_measure 同一个坑——极值把深度噪声当成物体边缘。用 1%/99% 分位反推
    跨度，3mm 噪声下误差能压到 1mm 级。

场景：货位/托盘占用检测、来料计数、桌面零件清点、包装箱到位判断。

用法：
    python app_objects.py                     # 默认：剔平面 + 计数
    python app_objects.py --keep-plane        # 不剔平面（整片场景分块）
    python app_objects.py --jump 60           # 跳变阈值放宽（斜面/弱纹理场景）
    python app_objects.py --min-area 800      # 只关心更大的物体
    python app_objects.py --zmax 1.5          # 只看 1.5m 内（操作区）
    python app_objects.py --shot out.png      # 出一张图后退出

操作：
    鼠标点右上角 MIRROR = 镜像画面（文字不镜像）
    y = 语义分割(YOLO) / 几何分割 切换 —— YOLO 能认出类别（bottle/suitcase/
        person/...），贴着的两个东西也能分开；几何模式靠深度跳变，不认识
        物体但对任意形状都工作
    p = 平面剔除开关    d = 深度伪彩叠加    s = 存图    q / ESC = 退出

YOLO 模式说明：模型 models/yolo11n_seg.onnx（COCO 80 类）在彩色图上出实例
掩码，掩码内的深度点（经标定投影）用 0.5%/99.5% 分位量长宽高 —— 识别归
RGB、量尺寸归深度，互不掺假。DirectML 推理约 8ms，4060 上整链 15ms。
"""
import argparse
import os
import time

import cv2
import numpy as np

import capture_pair as cp
from ui_common import wait_key
from app_align import adapt_cal, latest_calib, load_calib
from app_guard import btn_rect, draw_button
from app_pointcloud import Z_LO, Z_HI, project_to_image

try:
    from yolo_seg import YoloSeg, draw_dets, MODEL_PATH as YOLO_PATH
except Exception:                                  # onnxruntime 装不上也不碍事
    YoloSeg = None
    YOLO_PATH = ""

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "captures")

PALETTE = [(80, 200, 255), (120, 255, 160), (255, 160, 120), (200, 120, 255),
           (255, 220, 90), (110, 220, 255), (180, 255, 120), (255, 120, 200)]
C_PLANE = (115, 115, 115)          # 被剔掉的平面：灰点


# ------------------------------------------------------------- 平面剔除 ---
def ransac_plane(P, thresh=0.012, iters=300, subset=4000, seed=7):
    """最大平面 RANSAC。P: Nx3(m)。返回 (法向 nv, 平面上一点 p0, 内点掩码)"""
    n = len(P)
    if n < 100:
        return None
    rng = np.random.default_rng(seed)
    S = P if n <= subset else P[rng.choice(n, subset, replace=False)]
    best_cnt, best = -1, None
    for _ in range(iters):
        i, j, k = rng.choice(len(S), 3, replace=False)
        nv = np.cross(S[j] - S[i], S[k] - S[i])
        ln = np.linalg.norm(nv)
        if ln < 1e-9:
            continue
        nv /= ln
        cnt = int((np.abs((S - S[i]) @ nv) < thresh).sum())
        if cnt > best_cnt:
            best_cnt, best = cnt, (nv, S[i])
    if best is None:
        return None
    nv, p0 = best
    # 用内点 SVD 精化：三个随机点定的平面法向噪声很大，精化后内点才可信
    inl = np.abs((P - p0) @ nv) < thresh
    if inl.sum() >= 100:
        Q = P[inl]
        c = Q.mean(axis=0)
        nv = np.linalg.svd(Q - c, full_matrices=False)[2][2]
        p0 = c
        inl = np.abs((P - p0) @ nv) < thresh
    return nv, p0, inl


# --------------------------------------------------------------- 分割 ---
def segment(ok, z, jump_m, grow=1):
    """深度不连续分割。ok=允许分割的像素掩码，z=深度(m)。

    做法：把"两侧深度差 > jump"的像素连成边界带并挖掉，剩下的做 8 邻域连通域。
    边界带两侧都标 + 膨胀 1px，确保两个物体之间留出真正的空隙（不然 8 邻域会
    从缝隙里绕过去连上）。
    """
    zf = np.where(ok, z, 0.0)
    edge = np.zeros_like(ok)
    d = np.abs(np.diff(zf, axis=1))
    hit = (d > jump_m) & ok[:, :-1] & ok[:, 1:]
    edge[:, :-1] |= hit
    edge[:, 1:] |= hit
    d = np.abs(np.diff(zf, axis=0))
    hit = (d > jump_m) & ok[:-1, :] & ok[1:, :]
    edge[:-1, :] |= hit
    edge[1:, :] |= hit
    if grow:
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8),
                          iterations=grow) > 0
    m = (ok & ~edge).astype(np.uint8)
    _num, labels = cv2.connectedComponents(m, connectivity=8)
    labels[~ok] = 0
    return labels


# ------------------------------------------------------------ 每块统计 ---
def blob_stats(P, labels, lab_id, bbox, uc, vc, uid):
    x0, y0, w, h = bbox
    m = labels[y0:y0 + h, x0:x0 + w] == lab_id
    n = int(m.sum())
    if n < 30:
        return None
    subP = P[y0:y0 + h, x0:x0 + w][m]
    # 0.5%/99.5% 分位：砍掉噪声极值，再按 0.99 反推真实跨度。
    # 分位不能砍太狠——掠射方向的可见面在图像上只有几十列像素，砍 1% 就等于
    # 砍掉一整列（十几毫米），所以这里用 0.5% 而不是 app_measure 的 1%。
    lo, hi = np.percentile(subP, 0.5, axis=0), np.percentile(subP, 99.5, axis=0)
    span = (hi - lo) / 0.99
    cen = 0.5 * (lo + hi)
    uu = uc[y0:y0 + h, x0:x0 + w][m]
    vv = vc[y0:y0 + h, x0:x0 + w][m]
    order = np.argsort(subP[:, 2])          # 用最近 30% 的点当"该块的距离"
    near = subP[order[:max(1, int(0.3 * n))], 2]
    return dict(id=uid, lab=lab_id, bbox=(x0, y0, w, h), n=n,
                z=float(np.median(near)),
                w=float(span[0]), h=float(span[1]), d=float(span[2]),
                cx=float(cen[0]), cy=float(cen[1]), cz=float(cen[2]),
                u=float(np.median(uu)), v=float(np.median(vv)))


def remove_planes(P, ok, thresh=0.012, max_planes=3, min_frac=0.15):
    """迭代剔除**背景大平面**（墙 / 桌面 / 地面）。

    ① 只剔一个平面不够：墙剔掉后桌面还在，而桌子和摆在上面的物体常常在侧面
       平滑相接（深度连续），一旦桌面留下，物体就会被桌面连成一大块。
    ② 但也不能见平面就剔——箱子的正面本身就是平的！所以判据用**占整幅的比例**：
       只有大过 min_frac 的平面才算"背景"。箱子正面顶多占几个百分点，剔不到。
    """
    plane = np.zeros_like(ok)
    rest = ok.copy()
    n0 = float(ok.sum())
    for _ in range(max_planes):
        if rest.sum() < 800:
            break
        rp = ransac_plane(P[rest], thresh=thresh)
        if rp is None:
            break
        _nv, _p0, inl = rp
        if inl.sum() < min_frac * n0:      # 太小 → 是物体表面，不是背景
            break
        m = np.zeros(ok.size, bool)
        m[np.flatnonzero(rest.ravel())[inl]] = True
        m = m.reshape(ok.shape)
        plane |= m
        rest &= ~m
    return plane


def analyze(depth_mm, cal, dsz, csz, args, plane_on=True):
    dh, dw = dsz
    ch, cw = csz
    z = depth_mm.astype(np.float32) * 0.001
    v, u = np.mgrid[0:dh, 0:dw].astype(np.float32)

    Kd, Kc, R, T = cal["K_d"], cal["K_c"], cal["R"], cal["T"]
    x = (u - Kd[0, 2]) * z / Kd[0, 0]
    y = (v - Kd[1, 2]) * z / Kd[1, 1]
    X = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z + T[0]
    Y = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z + T[1]
    Z = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z + T[2]

    ok = (z > 0) & (Z > max(Z_LO, args.zmin)) & (Z < min(Z_HI, args.zmax))
    P = np.stack([X, Y, Z], axis=-1)
    uc, vc, kok = project_to_image(P, Kc, cal["D_c"], cw, ch)
    ok &= kok

    plane = np.zeros_like(ok)
    if plane_on and ok.sum() > 500:
        plane = remove_planes(P, ok, thresh=args.plane_thresh,
                              max_planes=args.max_planes)

    labels = segment(ok & ~plane, z, args.jump * 0.001)

    objs = []
    if labels.any():
        n_lab, _, stats, _ = cv2.connectedComponentsWithStats(
            (labels > 0).astype(np.uint8), connectivity=8)
        max_area = args.max_area * dh * dw
        cand = [(i, int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, n_lab)
                if args.min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]
        cand.sort(key=lambda t: -t[1])
        for i, _a in cand[:args.top]:
            bb = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                  int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
            s = blob_stats(P, labels, i, bb, uc, vc, len(objs) + 1)
            if s:
                objs.append(s)
    return dict(objs=objs, labels=labels, P=P, ok=ok, plane=plane,
                uc=uc, vc=vc, ratio=float(ok.mean()))


# ---------------------------------------------------- YOLO 模式的深度统计 ---
def yolo_stats(det, P, ok, plane, uc, vc, uid, band=0.10):
    """把一个 YOLO 实例映射到深度点云上量尺寸。

    掩码在彩色图坐标系；深度点经标定投影落在哪个彩色像素，就查那个像素
    的掩码 —— 得到 (深度网格形状) 的布尔图，再取其中的点云统计。
    排除已剔的背景平面，挡住掩码边缘"透过物体看到的后墙"。

    再做一层**深度主簇**过滤：RGB 掩码边缘的遮挡泄漏会让物体身后的背景
    深度点混进掩码区（实机首测：椅子量出 4.2m 进深，其实是整间屋）。
    按 z 做 10cm 直方图，取最密峰所在的连续带（计数 > 峰值 20% 才续上），
    只对带内点量尺寸 —— 物体自身是实心连续的，背景是零散泄漏。
    """
    ch, cw = uc.shape
    # 无效投影：NaN/inf（深度 0）以及除以近零深度产生的 1e32 级巨值 ——
    # 后者是"有限但溢出 int32"的 cast，必须先钳到画布范围
    uc = np.clip(np.where(np.isfinite(uc), uc, 0.0), 0.0, cw - 1)
    vc = np.clip(np.where(np.isfinite(vc), vc, 0.0), 0.0, ch - 1)
    vc_i = vc.astype(np.int32)
    uc_i = uc.astype(np.int32)
    m = det["mask"][vc_i, uc_i] & ok & ~plane
    n = int(m.sum())
    if n < 30:
        return None
    subP = P[m]
    z = subP[:, 2]

    # 深度主簇
    zlo = float(np.percentile(z, 1)) - band
    hist, edges = np.histogram(z, bins=np.arange(zlo, zlo + np.ptp(z) + 2 * band, band))
    if hist.max() < 15:
        return None
    peak = int(hist.argmax())
    lo_i = hi_i = peak
    while lo_i > 0 and hist[lo_i - 1] > 0.2 * hist[peak]:
        lo_i -= 1
    while hi_i < len(hist) - 1 and hist[hi_i + 1] > 0.2 * hist[peak]:
        hi_i += 1
    sel = (z >= edges[lo_i]) & (z < edges[hi_i + 1])
    subP = subP[sel]
    if len(subP) < 60:
        return None

    lo, hi = np.percentile(subP, 0.5, axis=0), np.percentile(subP, 99.5, axis=0)
    span = (hi - lo) / 0.99
    cen = 0.5 * (lo + hi)
    order = np.argsort(subP[:, 2])
    near = subP[order[:max(1, int(0.3 * len(subP)))], 2]
    x0, y0, w, h = det["box"]
    return dict(id=uid, n=len(subP), z=float(np.median(near)),
                w=float(span[0]), h=float(span[1]), d=float(span[2]),
                cx=float(cen[0]), cy=float(cen[1]), cz=float(cen[2]),
                u=float(x0 + w / 2), v=float(y0 + h / 2),
                name=det["name"], conf=det["conf"], mask=det["mask"])


# --------------------------------------------------------------- 绘制 ---
BOX_EDGES = ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
             (0, 4), (1, 5), (2, 6), (3, 7))


def draw(canvas, res, args, plane_on, csz, cal):
    ch, cw = csz
    mir = args.mirror
    flip_u = (lambda a: (cw - 1 - a)) if mir else (lambda a: a)

    if res.get("yolo") is not None:
        # YOLO 模式：掩码叠加 + 类别框 + 3D 包围盒
        dets, stats = res["yolo_dets"], res["yolo"]
        vis = []
        for d in dets:
            dd = dict(d)
            if mir:                       # 画面翻了，掩码和框跟着翻
                dd["mask"] = d["mask"][:, ::-1]
                x0, y0, w, h = d["box"]
                dd["box"] = (cw - 1 - (x0 + w - 1), y0, w, h)
            vis.append(dd)
        names = {i: s["name"] for i, s in enumerate(stats)}
        draw_dets(canvas, vis, names)
        for s in stats:
            col = (90, 235, 150)
            corners = np.array([[s["cx"] + sx * s["w"] / 2,
                                 s["cy"] + sy * s["h"] / 2,
                                 s["cz"] + sz * s["d"] / 2]
                                for sx in (-1, 1) for sy in (-1, 1)
                                for sz in (-1, 1)], np.float32)
            cu, cv2_, cok = project_to_image(corners, cal["K_c"], cal["D_c"],
                                             cw, ch)
            if cok.all():
                p2 = np.stack([flip_u(cu), cv2_], axis=1).astype(np.int32)
                for a, b in BOX_EDGES:
                    cv2.line(canvas, tuple(p2[a]), tuple(p2[b]), col, 1,
                             cv2.LINE_AA)
        n = len(stats)
        head = f"{n} object{'s' if n != 1 else ''}  (yolo seg)"
    else:
        if plane_on and res["plane"].any():
            pm = res["plane"]
            canvas[res["vc"][pm].astype(int),
                   flip_u(res["uc"][pm]).astype(int)] = C_PLANE

        for o in res["objs"]:
            col = PALETTE[(o["id"] - 1) % len(PALETTE)]
            x0, y0, w, h = o["bbox"]
            sub = res["labels"][y0:y0 + h, x0:x0 + w] == o["lab"]
            uu = res["uc"][y0:y0 + h, x0:x0 + w][sub]
            vv = res["vc"][y0:y0 + h, x0:x0 + w][sub]
            canvas[vv.astype(int), flip_u(uu).astype(int)] = col

            # 3D 包围盒 8 角投回彩色图连线
            corners = np.array([[o["cx"] + sx * o["w"] / 2,
                                 o["cy"] + sy * o["h"] / 2,
                                 o["cz"] + sz * o["d"] / 2]
                                for sx in (-1, 1) for sy in (-1, 1)
                                for sz in (-1, 1)], np.float32)
            cu, cv2_, cok = project_to_image(corners, cal["K_c"], cal["D_c"],
                                             cw, ch)
            if cok.all():
                p2 = np.stack([flip_u(cu), cv2_], axis=1).astype(np.int32)
                for a, b in BOX_EDGES:
                    cv2.line(canvas, tuple(p2[a]), tuple(p2[b]), col, 1,
                             cv2.LINE_AA)

            txt = (f"#{o['id']} {o['z'] * 10:.0f}dm "
                   f"{o['w'] * 100:.0f}x{o['h'] * 100:.0f}cm")
            tx, ty = int(flip_u(o["u"])), int(o["v"])
            cv2.putText(canvas, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        col, 1, cv2.LINE_AA)

        n = len(res["objs"])
        head = f"{n} object{'s' if n != 1 else ''}"
    cv2.putText(canvas, head, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(canvas, head, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 235, 140) if n else (175, 175, 175), 2, cv2.LINE_AA)
    draw_button(canvas, cw, mir)


def main():
    ap = argparse.ArgumentParser(description="物体分割 / 计数 / 粗尺寸")
    ap.add_argument("--calib", default="")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--backend", default="ffmpeg", choices=["ffmpeg", "dshow"])
    ap.add_argument("--device", default=cp.FFMPEG_DEVICE)
    ap.add_argument("--jump", type=float, default=40.0, help="深度跳变阈值(mm)")
    ap.add_argument("--min-area", type=int, default=400, help="最小块面积(深度图像素)")
    ap.add_argument("--max-area", type=float, default=0.35, help="最大块面积(占画面比例)")
    ap.add_argument("--plane-thresh", type=float, default=0.012, help="平面内点阈值(m)")
    ap.add_argument("--max-planes", type=int, default=3, help="最多剔除几个大平面")
    ap.add_argument("--zmin", type=float, default=0.40, help="只关心此距离之外(m)")
    ap.add_argument("--zmax", type=float, default=6.00, help="只关心此距离之内(m)")
    ap.add_argument("--top", type=int, default=12, help="最多标注几块")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO 置信度阈值")
    ap.add_argument("--no-yolo", action="store_true", help="禁用语义分割（只用几何分割）")
    ap.add_argument("--keep-plane", action="store_true", help="不剔平面")
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
    ch, cw = csz
    win = "Objects  [y=yolo/geom  p=plane  d=depth  f=mirror  s=save  q=quit]"
    plane_on = not args.keep_plane
    show_depth = False

    yolo = None
    yolo_mode = False
    if YoloSeg is not None and not args.no_yolo and os.path.exists(YOLO_PATH):
        yolo = YoloSeg()
        yolo_mode = True
        print(f"YOLO 语义分割就绪（{yolo.provider}，COCO 80 类），"
              f"y 键可切回几何分割")
    elif not args.no_yolo:
        print("提示：未找到 models/yolo11n_seg.onnx 或 onnxruntime，"
              "只有几何分割模式（pip 装 onnxruntime-directml 可启用 YOLO）")

    def on_mouse(ev, x, y, _f, _p):
        if ev == cv2.EVENT_LBUTTONDOWN:
            x0, y0, x1, y1 = btn_rect(cw)
            if x0 <= x <= x1 and y0 <= y <= y1:
                args.mirror = not args.mirror

    if not args.shot:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, on_mouse)
        print(f"物体分割：{'剔平面' if plane_on else '不剔平面'}，跳变 {args.jump:.0f}mm，"
              f"最小块 {args.min_area}px。点右上角 MIRROR 镜像（或按 f），q 退出")

    t_prev, fps, n = time.time(), 0.0, 0
    try:
        while True:
            dz = depth.latest()
            cz = color.latest()
            if dz is None or cz is None:
                time.sleep(0.01)
                continue
            img = cp.ColorSourceFFmpeg.decode(cz[0]) if getattr(color, "is_jpeg", False) else cz[0]
            res = analyze(cv2.flip(dz[0], 1), cal, dsz, csz, args, plane_on)

            if yolo_mode and yolo is not None:
                dets = yolo.detect(img, conf=args.conf, max_det=args.top)
                stats = []
                for d in dets:
                    s = yolo_stats(d, res["P"], res["ok"], res["plane"],
                                   res["uc"], res["vc"], len(stats) + 1)
                    if s:
                        stats.append(s)
                res["yolo"] = stats
                res["yolo_dets"] = dets[:len(stats)]
            else:
                res["yolo"] = None

            canvas = cv2.flip(img, 1) if args.mirror else img.copy()
            if show_depth:
                g = np.clip((res["P"][..., 2] - 0.3) / 5.7, 0, 1)
                m = np.zeros((ch, cw), np.uint8)
                sel = res["ok"]
                ucd = (cw - 1 - res["uc"]) if args.mirror else res["uc"]
                m[res["vc"][sel].astype(int), ucd[sel].astype(int)] = \
                    (g[sel] * 255).astype(np.uint8)
                canvas = cv2.addWeighted(canvas, 0.65,
                                         cv2.applyColorMap(m, cv2.COLORMAP_TURBO), 0.35, 0)

            draw(canvas, res, args, plane_on, csz, cal)

            n += 1
            if n % 5 == 0:
                now = time.time()
                fps = 0.8 * fps + 0.2 * (5.0 / max(now - t_prev, 1e-6))
                t_prev = now
            head = (f"{fps:4.1f} fps | {'yolo ' + yolo.provider.split('Execution')[0] if (yolo_mode and yolo) else 'geom'} | "
                    f"jump {args.jump:.0f}mm | min {args.min_area}px | "
                    f"{'plane-removed' if plane_on else 'raw'} | valid {res['ratio'] * 100:.0f}%")
            cv2.putText(canvas, head, (12, ch - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, head, (12, ch - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (210, 240, 210), 1, cv2.LINE_AA)

            if args.shot:
                cp.imwrite_u(args.shot, canvas)
                for s in res.get("yolo") or res["objs"]:
                    if "name" in s:
                        print(f"  {s['name']:<12s} conf={s['conf']:.2f}  "
                              f"z={s['z']:.3f}m  "
                              f"{s['w'] * 1000:.0f}x{s['h'] * 1000:.0f}x"
                              f"{s['d'] * 1000:.0f}mm  "
                              f"vol≈{s['w'] * s['h'] * s['d'] * 1000:.1f}L")
                    else:
                        print(f"  #{s['id']}  {s['n']:6d}pt  z={s['z']:.3f}m  "
                              f"{s['w'] * 1000:.0f}x{s['h'] * 1000:.0f}x"
                              f"{s['d'] * 1000:.0f}mm  "
                              f"vol≈{s['w'] * s['h'] * s['d'] * 1000:.1f}L")
                mode = "YOLO 语义分割" if res.get("yolo") is not None else "几何分割"
                print(f"共 {len(res.get('yolo') or res['objs'])} 块（{mode}）| "
                      f"存 {args.shot}")
                break

            cv2.imshow(win, canvas)
            k = wait_key(win, 1)      # 点窗口 X 也能退出
            if k in (27, ord("q")):
                break
            if k == ord("s"):
                fn = os.path.join(OUT_DIR, time.strftime("objects_%H%M%S.png"))
                cp.imwrite_u(fn, canvas)
                print("saved captures/" + os.path.basename(fn))
            elif k == ord("y"):
                if yolo is not None:
                    yolo_mode = not yolo_mode
                    print("分割模式：", "YOLO 语义分割" if yolo_mode else "几何（深度跳变）")
                else:
                    print("YOLO 不可用（缺模型或 onnxruntime）")
            elif k == ord("p"):
                plane_on = not plane_on
                print("平面剔除", "开" if plane_on else "关")
            elif k == ord("d"):
                show_depth = not show_depth
            elif k == ord("f"):
                args.mirror = not args.mirror
    finally:
        if not args.shot:
            cv2.destroyAllWindows()
        color.close()
        depth.close()


if __name__ == "__main__":
    main()
