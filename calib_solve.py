# -*- coding: utf-8 -*-
"""
深度+彩色 联合标定 —— 求解
------------------------------------------------------------------
输入: calib_capture.py 产出的 <dir>/pairs.csv + 成对 PNG
流程:
  1) 彩色图检测棋盘格角点 -> cv2.calibrateCamera -> 彩色内参 K_c, D_c
  2) IR 图检测角点(共镜头=深度相机内参)     -> cv2.calibrateCamera -> K_d, D_d
  3) cv2.stereoCalibrate(CALIB_FIX_INTRINSIC) -> 外参 R, T (深度相机 -> 彩色相机)
  4) 验证: 把棋盘格 3D 点经 [R|T] 投影回彩色图叠加保存
输出:
  <dir>/stereo_calib.json   全部标定参数 + 误差（后续脚本直接读取）
  <dir>/verify_*.png        投影验证叠加图

用法:
    python calib_solve.py --dir calib_raw
    python calib_solve.py --selftest          # 合成数据自检（不需要相机）
"""
import argparse, json, os, sys, time

import cv2
import numpy as np

from calib_capture import (BOARD, BOARDS, board_label, ir_to_8bit,
                          detect_corners, imwrite_u, set_board)


# --------------------------------------------------------------- 基础 ---
def board_points():
    """标定点的物理坐标（mm），Z=0 平面。

    顺序必须与 calib_capture.detect_corners() 的返回顺序严格一致。
    """
    cols, rows = BOARD["cols"], BOARD["rows"]
    sq = BOARD["square_mm"]
    pts = np.zeros((cols * rows, 3), np.float32)
    idx = 0

    if BOARD["kind"] == "circles":
        # 非对称圆点阵：findCirclesGrid 沿 (±0.5, 0.5) 格基对角遍历，
        # 第 k 个点不是"逐行扫描"，而是按列分组：k = c*cols + t。
        # 这个对应关系是对着 OpenCV 实测出来的（合成真实点阵，20 点全部
        # 一一命中、残差 0.00px）；把板子转 180° 后顺序不变，
        # 所以圆点板不存在棋盘的 180° 简并。
        #   c : 第几列（共 rows 列），x 递增
        #   t : 该列内的第几个点（共 cols 个），y 按半间距交错
        half = sq / 2.0
        for c in range(rows):
            for t in range(cols):
                pts[idx] = [c * half, (2 * t + (c % 2)) * half, 0]
                idx += 1
        return pts

    for r in range(rows):
        for c in range(cols):
            pts[idx] = [(c + 0) * sq, (-r) * sq, 0]  # 行方向取负, 保证角点顺序与检测结果一致
            idx += 1
    return pts


OBJP = board_points()


def refresh_objp():
    """按当前 BOARD 重新烘焙物面坐标。换标定板后必须调用，
    否则会出现"检测按新板、几何按旧板"的错位。"""
    global OBJP
    OBJP = board_points()
    return OBJP

# 外参合理性闸门：Astra Pro Plus 的 IR(深度) 与彩色模组刚性近平行装配，
# 正确解旋转角只有零点几度、基线约 25~40mm。超过即说明解出了镜像/漂移解。
ANG_MAX = 15.0        # deg
ANG_T_MIN, ANG_T_MAX = 10.0, 150.0   # mm


def calibrate_one(objps, imgps, size, fix_k3=True):
    flags = 0
    if fix_k3:
        flags |= cv2.CALIB_FIX_K3
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objps, imgps, size, None, None, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    return rms, K, D, rvecs, tvecs


def stereo_solve(objps, p_ir, p_c, K_d, D_d, K_c, D_c, size_c):
    """外参求解。
    默认 CALIB_USE_INTRINSIC_GUESS：在内参初值基础上联合优化内外参。
    合成数据实验证明其外参精度远优于固定内参（平移误差 ~0.5mm vs ~10mm）。

    返回 (rms, R, T, K_d, D_d, K_c, D_c)。注意 stereoCalibrate 会把内参
    一起精化，外参与精化后的内参是一体的 —— 验证投影必须用返回值里
    精化后的 K/D，否则会出现"RMS 很低但验证图乱掉"的假象。
    """
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
    # OpenCV 5 的 stereoCalibrate 声明为 InputOutputArray：会把传入的
    # cameraMatrix/distCoeffs numpy 数组就地改写，返回值也可能与传入共享
    # 内存。2026-11-16 踩过：多假设循环里第二个假设拿到被污染的初值，
    # JSON 里写出的内参也变成污染后的值（RMS/T 反而正常，极具迷惑性）。
    # 出入口全部深拷贝，彻底隔离。
    out = cv2.stereoCalibrate(
        objps, p_ir, p_c, K_d.copy(), D_d.copy(), K_c.copy(), D_c.copy(), size_c,
        flags=flags, criteria=crit)
    rms = out[0]
    K_d2 = np.array(out[1], copy=True)
    D_d2 = np.array(out[2], copy=True)
    K_c2 = np.array(out[3], copy=True)
    D_c2 = np.array(out[4], copy=True)
    R = np.array(out[5], copy=True)
    T = np.array(out[6], copy=True)
    return rms, R, T, K_d2, D_d2, K_c2, D_c2


# --------------------------------------------------------------- 求解 ---
def load_pairs(dirname):
    import csv
    pairs = []
    with open(os.path.join(dirname, "pairs.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c, i = row["color_png"], row["ir_png"]
            pc, pi = os.path.join(dirname, c), os.path.join(dirname, i)
            if os.path.exists(pc) and os.path.exists(pi):
                pairs.append((pc, pi))
    return pairs


def detect_all(pairs):
    ok_c, ok_i, used = [], [], []
    for k, (pc, pi) in enumerate(pairs):
        cimg = cv2.imdecode(np.fromfile(pc, dtype=np.uint8), cv2.IMREAD_COLOR)
        ir16 = cv2.imdecode(np.fromfile(pi, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if cimg is None or ir16 is None:
            print(f"    [skip] {os.path.basename(pc)} 读取失败"); continue
        # Astra 系列的 IR（与深度共镜头）图像是水平镜像的：实测同一块板，
        # 彩色板心 u 与 IR 板心 u 满足 u_color + u_ir ≈ 1。镜像无法用旋转
        # 表达，不翻回来立体标定必炸（任何点序假设都收敛到 38px 的废解）。
        # 翻转后整条链路（检测/内参/外参/验证图）都按标准针孔朝向走。
        # 注意：运行时用 K_d 投影深度图前，深度图必须做同样的水平翻转。
        ir16 = cv2.flip(ir16, 1)
        cok, cpts = detect_corners(cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY))
        iok, ipts = detect_corners(ir_to_8bit(ir16))
        if cok and iok:
            ok_c.append(cpts.reshape(-1, 1, 2).astype(np.float32))
            ok_i.append(ipts.reshape(-1, 1, 2).astype(np.float32))
            used.append((pc, cimg, ir16))
            print(f"    [{k+1:02d}] OK   {os.path.basename(pc)}")
        else:
            print(f"    [{k+1:02d}] skip {'彩色' if not cok else ''}"
                  f"{'IR' if not iok else ''} 未检出 -> {os.path.basename(pc)}")
    return ok_c, ok_i, used


def run_solve(pairs, out_dir, n_verify=5):
    print(f"  检测角点 ({len(pairs)} 组) ...")
    p_c, p_d, used = detect_all(pairs)
    n = len(p_c)
    if n < 10:
        raise SystemExit(f"有效姿态仅 {n} 组(<10)，请回 calib_capture.py 补采")
    print(f"  有效姿态: {n} 组")

    size_c = (used[0][1].shape[1], used[0][1].shape[0])
    size_d = (used[0][2].shape[1], used[0][2].shape[0])
    objps = [OBJP.reshape(-1, 1, 3).astype(np.float32)] * n

    print("  [1/2] 彩色内参标定 ...")
    rms_c, K_c, D_c, rv_c, tv_c = calibrate_one(objps, p_c, size_c)
    print(f"        RMS = {rms_c:.3f} px   fx,fy = {K_c[0,0]:.1f},{K_c[1,1]:.1f}")

    print("  [2/2] 深度相机(IR 共镜头)内参标定 ...")
    rms_d, K_d, D_d, rv_d, tv_d = calibrate_one(objps, p_d, size_d)
    print(f"        RMS = {rms_d:.3f} px   fx,fy = {K_d[0,0]:.1f},{K_d[1,1]:.1f}")

    # ---- 外参：两种角点顺序假设都解，选物理合理的那个 ----
    # 棋盘格：10x7 格在"旋转180°+黑白反转"下与自身等价，OpenCV 在彩色流和
    #   IR 流里挑的起点角点可能相反 —— flip 用整段反序。
    # 圆点板：CGB-020 的错排点阵左右镜像对称（镜像后 (col+row)%2 奇偶不变），
    #   同一块板存在两个几何等价的解释，两路各自收敛到哪一个纯属检测器内部
    #   实现。2026-11-16 实测（15 组真机数据，彩色 0 号点在左上、IR 0 号点在
    #   右下）：IR 镜像翻转后的检出序对应"镜像解释"，flip 必须用镜像置换 ——
    #   用棋盘那套反序会得到 RMS 15px/|T|3.6m 的废解；镜像置换得到
    #   RMS 0.204px / 旋转 0.42° / |T| 12.5mm，是唯一物理合理的组合。
    # 两个模组物理上近平行装配，正确解的旋转角应接近 0°——以此裁决。
    if BOARD["kind"] == "circles":
        o = OBJP[:, :2]
        mirrored = np.column_stack([o[:, 0].max() - o[:, 0], o[:, 1]])
        flip_perm = np.array([int(np.argmin(np.linalg.norm(o - m, axis=1)))
                              for m in mirrored])
        flips = [("原序   ", None), ("IR 镜像序", flip_perm)]
    else:
        flips = [("原序   ", None), ("IR 反序  ", "rev")]
    print(f"  [3/3] 外参 (深度 -> 彩色)，{len(flips)} 种角点顺序假设 ...")
    cands = []
    for tag, perm in flips:
        if perm is None:
            q = p_d
        elif isinstance(perm, str):
            q = [p[::-1].copy() for p in p_d]
        else:
            q = [p[perm].copy() for p in p_d]
        try:
            rms_s, R, T, K_d2, D_d2, K_c2, D_c2 = stereo_solve(
                objps, q, p_c, K_d, D_d, K_c, D_c, size_c)
        except cv2.error as e:
            print(f"        {tag} 解算失败: {e}")
            continue
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        tn = float(np.linalg.norm(T))
        ok_phys = ang <= ANG_MAX and ANG_T_MIN <= tn <= ANG_T_MAX
        print(f"        {tag}: 立体RMS={rms_s:.3f} px"
              f"  旋转={ang:.2f} deg  |T|={tn:.1f} mm"
              + ("" if ok_phys else "   <- 物理不合理,弃用"))
        cands.append(dict(flip=tag, rms=rms_s, R=R, T=T, ang=ang, tn=tn,
                          K_d=K_d2, D_d=D_d2, K_c=K_c2, D_c=D_c2))

    good = [c for c in cands if c["ang"] <= ANG_MAX and ANG_T_MIN <= c["tn"] <= ANG_T_MAX]
    if not good:
        det = ";  ".join(f"旋转{c['ang']:.1f}°/|T|{c['tn']:.0f}mm" for c in cands) or "两假设均解算失败"
        raise RuntimeError(
            f"外参解算结果物理不合理（{det}）。"
            f"正确解的旋转角应 < {ANG_MAX:.0f}°、|T| 在 {ANG_T_MIN:.0f}~{ANG_T_MAX:.0f} mm。"
            "最常见原因是姿态多样性不足：板基本没动、或角点全挤在画面中间时内参本身解不准，"
            "外参必然漂移。请用引导标定重新采集（默认的「中间横带」方案即可，"
            "别选「只摆中间」），不要重复使用本组数据。")
    c = min(good, key=lambda x: x["rms"])
    if c["flip"] != "原序   ":
        print(f"        -> 采用 {c['flip'].strip()} 的解（OpenCV 两流起点约定不同，属正常现象）")
    rms_s, R, T = c["rms"], c["R"], c["T"]
    K_d, D_d, K_c, D_c = c["K_d"], c["D_d"], c["K_c"], c["D_c"]
    ang = c["ang"]
    print(f"        立体重投影 RMS = {rms_s:.3f} px")
    print(f"        平移 T = {T.ravel()} mm   旋转角 = {ang:.2f} deg")

    # ---- 验证叠加图：棋盘 3D 点(板坐标) -> 深度相机 -> 彩色相机 投影 ----
    # 注意每帧深度位姿必须在"精化后的 K_d/D_d"下重求（solvePnP）：
    # calibrate_one 的 rv_d/tv_d 配的是精化前内参，与 stereo 精化的 R/T
    # 不配套 —— 混用会出现 RMS 0.2px 但叠加图整体偏十几像素的假象。
    n_verify = min(n_verify, n)
    for k in range(n_verify):
        pc, cimg, _ = used[k]
        ok, rv_k, tv_k = cv2.solvePnP(OBJP.reshape(-1, 1, 3).astype(np.float32),
                                      p_d[k], K_d, D_d)
        rvec = cv2.Rodrigues(R @ rv_k)[0]
        tvec = R @ tv_k + T.reshape(3, 1)
        proj, _ = cv2.projectPoints(OBJP, rvec, tvec, K_c, D_c)
        vis = cimg.copy()
        for p in proj.reshape(-1, 2).astype(int):
            cv2.circle(vis, tuple(p), 4, (0, 0, 255), -1)
        imwrite_u(os.path.join(out_dir, f"verify_{os.path.basename(pc)}"), vis)

    result = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        board=dict(kind=BOARD["kind"], cols=BOARD["cols"], rows=BOARD["rows"],
                   square_mm=BOARD["square_mm"]),
        color=dict(size=list(size_c), K=K_c.tolist(), D=D_c.ravel().tolist(),
                   rms_px=rms_c, note="K/D 为立体联合精化后的值"),
        depth=dict(size=list(size_d), K=K_d.tolist(), D=D_d.ravel().tolist(),
                   rms_px=rms_d,
                   note="K/D 为立体联合精化后的值；坐标基准是【水平翻转后】的"
                        "深度/IR 图像（Astra IR 传感器成镜像）。运行时用 K_d "
                        "投影深度前必须先 cv2.flip(depth, 1)"),
        R_depth_to_color=R.tolist(),
        T_depth_to_color=T.ravel().tolist(),
        T_units="mm", T_direction="depth camera -> color camera",
        stereo_rms_px=rms_s, poses=n,
    )
    jpath = os.path.join(out_dir, "stereo_calib.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {jpath}")
    return result


# --------------------------------------------------------------- 自检 ---
def selftest(n_poses=30, noise_px=0.3, seed=7):
    """合成数据自检：生成带噪声的虚拟姿态，验证整条标定管线精度。
    姿态多样性按实测建议（大角度倾斜、更近距离、更大横向偏移）设置。
    """
    rng = np.random.default_rng(seed)
    size_c, size_d = (1280, 720), (640, 480)
    K_c = np.array([[985.0, 0, 640], [0, 985.0, 360], [0, 0, 1]])
    D_c = np.array([[-0.31, 0.14, 0.0, 0.0]])
    K_d = np.array([[470.0, 0, 320], [0, 470.0, 240], [0, 0, 1]])
    D_d = np.array([[-0.48, 0.26, 0.0, 0.0]])
    # 真值外参：深度 -> 彩色（mm）
    ax = rng.normal(0, 0.03, 3); ax /= np.linalg.norm(ax)
    R_gt = cv2.Rodrigues(ax * rng.uniform(0.05, 0.12))[0]
    T_gt = np.array([[25.0], [-8.0], [12.0]])

    objp = OBJP.reshape(-1, 1, 3).astype(np.float32)
    p_c, p_d, objps = [], [], []
    for _ in range(n_poses):
        for _try in range(50):
            rvec = rng.normal(0, 0.30, 3)
            t = np.array([rng.uniform(-210, 210), rng.uniform(-130, 130),
                          rng.uniform(550, 1150)])
            X_c, _ = cv2.projectPoints(objp, rvec, t.reshape(3, 1), K_c, D_c)
            X_d = (R_gt.T @ (t.reshape(3, 1) - T_gt))
            r_d = cv2.Rodrigues(R_gt.T @ cv2.Rodrigues(rvec)[0])[0]
            Y_d, _ = cv2.projectPoints(objp, r_d, X_d, K_d, D_d)
            if (X_c.reshape(-1, 2).min(0) > 10).all() and \
               (X_c.reshape(-1, 2) < [size_c[0]-10, size_c[1]-10]).all() and \
               (Y_d.reshape(-1, 2).min(0) > 5).all() and \
               (Y_d.reshape(-1, 2) < [size_d[0]-5, size_d[1]-5]).all():
                p_c.append((X_c + rng.normal(0, noise_px, X_c.shape)).astype(np.float32))
                p_d.append((Y_d + rng.normal(0, noise_px, Y_d.shape)).astype(np.float32))
                objps.append(objp)
                break
        else:
            raise SystemExit("自检无法生成姿态")

    print(f"[自检] 合成 {n_poses} 组姿态  噪声 σ={noise_px}px")
    rms_c, Kc, Dc, rv_c, tv_c = calibrate_one(objps, p_c, size_c)
    rms_d, Kd, Dd, rv_d, tv_d = calibrate_one(objps, p_d, size_d)
    rms_s, R, T, _, _, _, _ = stereo_solve(objps, p_d, p_c, Kd, Dd, Kc, Dc, size_c)
    ang = np.degrees(np.arccos(np.clip((np.trace(R @ R_gt.T) - 1) / 2, -1, 1)))
    terr = np.linalg.norm(T.ravel() - T_gt.ravel())

    print(f"  彩色内参 RMS={rms_c:.3f}px  fx误差={Kc[0,0]-985:+.2f}  "
          f"cx误差={Kc[0,2]-640:+.2f}")
    print(f"  深度内参 RMS={rms_d:.3f}px  fx误差={Kd[0,0]-470:+.2f}  "
          f"cx误差={Kd[0,2]-320:+.2f}")
    print(f"  外参: 旋转误差={ang:.4f} deg   平移误差={terr:.3f} mm")
    # 10 种子稳定性实测: 旋转 max 0.36deg, 平移 max 1.5mm -> 阈值放宽 2 倍
    ok = rms_c < 0.6 and rms_d < 0.6 and ang < 0.7 and terr < 3.0
    print(f"  自检{'通过 ✅' if ok else '未达标 ❌'}")
    return ok


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="深度+彩色联合标定求解")
    ap.add_argument("--dir", default=os.path.join(here, "calib_raw"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--board", default="chess", choices=list(BOARDS),
                    help="标定板类型，必须与采集时一致：chess / circles")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if args.board != BOARD["kind"]:
        set_board(args.board)
        refresh_objp()
    print(f"  标定板: {board_label()}")

    pairs = load_pairs(args.dir)
    if not pairs:
        raise SystemExit(f"{args.dir} 里没有成对数据，先运行 calib_capture.py")
    run_solve(pairs, args.dir)


if __name__ == "__main__":
    main()
