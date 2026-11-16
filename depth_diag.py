"""depth_diag.py —— 诊断"某些区域没有深度"的成因

现象：app_align 的 HUD 显示 depth 只有 3.4%，画面上大片区域是空洞。
空洞只有两种来源，必须先分开：
  A) 原始深度图本身就空  → 传感器问题（环境光/反射率/距离/USB 带宽/双流争抢）
  B) 原始有，投影后丢了    → 标定或映射问题
本脚本只回答 A：直接读 OpenNI2 的深度流（不经过任何标定、投影、变换），
统计每一帧的有效率、空洞的空间分布、深度值直方图。

三种跑法：
    python depth_diag.py                 # 单流（只开深度）→ 基线
    python depth_diag.py --dual          # 双流（深度+彩色 1280x720）→ 看带宽争抢
    python depth_diag.py --dual --sec 10 # 指定双流持续秒数

输出：
    * 每帧 valid% 的时间序列（看是否稳定、是否抖动式掉零）
    * 8x6 网格的空洞分布 ASCII 表（看空洞是"成片按区域"还是"全图稀碎"）
    * _detect/depth_diag_*.png：左=原始深度伪彩，右=有效掩码
    * 深度值直方图（看是否有偏置/截断）
"""
import argparse
import os
import time

import cv2
import numpy as np

import capture_pair as cp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_detect")

# 规格书的可靠量程（毫米）。注意深度流给的就是毫米。
Z_MIN, Z_MAX = 300, 8000


def analyze(d):
    """单帧统计：有效掩码、有效率、深度分布"""
    z = d.astype(np.float32)
    valid = (z >= Z_MIN) & (z <= Z_MAX)
    res = dict(valid=float(valid.mean() * 100.0),
               zero=float((d == 0).mean() * 100.0),
               tiny=float(((d > 0) & (d < Z_MIN)).mean() * 100.0),
               huge=float((d > Z_MAX).mean() * 100.0))
    if valid.any():
        v = z[valid]
        res.update(med=float(np.median(v)), p5=float(np.percentile(v, 5)),
                   p95=float(np.percentile(v, 95)))
    return valid, res


def grid_table(valid, rows=6, cols=8):
    """把有效掩码按 rows x cols 平均，输出 ASCII 表（行=画面从上到下）"""
    h, w = valid.shape
    rh, rw = h // rows, w // cols
    lines = []
    for r in range(rows):
        cells = []
        for c in range(cols):
            blk = valid[r * rh:(r + 1) * rh, c * rw:(c + 1) * rw]
            cells.append(f"{blk.mean() * 100:4.0f}")
        lines.append(" ".join(cells))
    return lines


def hist_text(d, bins=(0, 300, 600, 1000, 1500, 2000, 3000, 4000, 8000, 65536)):
    h = np.histogram(d, bins=bins)[0].astype(np.float64)
    h /= max(h.sum(), 1)
    out = []
    for i in range(len(bins) - 1):
        out.append(f"    {bins[i]:>5}-{bins[i + 1]:<6} {h[i] * 100:5.1f}%")
    return out


def save_vis(tag, depth, valid):
    """左：深度伪彩（0.3~4m）；右：有效掩码（白=有效）"""
    z = depth.astype(np.float32)
    g = np.clip((z - 300) / (4000 - 300), 0, 1)
    g8 = (g * 255).astype(np.uint8)
    g8[~valid] = 0
    left = cv2.applyColorMap(g8, cv2.COLORMAP_TURBO)
    right = (valid.astype(np.uint8) * 255)
    right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    both = np.hstack([left, right])
    fn = os.path.join(OUT, f"depth_diag_{tag}.png")
    os.makedirs(OUT, exist_ok=True)
    cp.imwrite_u(fn, both)
    return fn


def run_single(n_frames):
    print("== A) 纯深度流（只开 depth，排除带宽争抢）==")
    ds = cp.DepthSource()
    ds.start()
    print(f"   mode = {ds.mode}")
    t0 = time.time()
    frames, valid_s, stats = [], [], []
    while len(frames) < n_frames and time.time() - t0 < 20:
        it = ds.latest()
        if it is None:
            time.sleep(0.01)
            continue
        d = it[0]
        v, s = analyze(d)
        frames.append(d)
        valid_s.append(s["valid"])
        stats.append(s)
        time.sleep(0.03)
    ds.close()
    dur = time.time() - t0
    print(f"   取到 {len(frames)} 帧 / {dur:.1f}s = {len(frames) / dur:.1f} fps")
    report(frames, valid_s, stats, "single")
    return frames, valid_s


def run_dual(sec):
    print("== B) 双流（depth + color 1280x720，模拟 app_align 的真实负载）==")
    ds = cp.DepthSource()
    ds.start()
    col = cp.ColorSourceFFmpeg(1280, 720, 30, device=cp.FFMPEG_DEVICE)
    col.start()
    print(f"   depth mode = {ds.mode}   color mode = {col.mode}")
    t0 = time.time()
    frames, valid_s, stats = [], [], []
    while time.time() - t0 < sec:
        it = ds.latest()
        if it is not None:
            d = it[0]
            v, s = analyze(d)
            frames.append(d)
            valid_s.append(s["valid"])
            stats.append(s)
        time.sleep(0.03)
    dur = time.time() - t0
    print(f"   取到 {len(frames)} 帧 / {dur:.1f}s = {len(frames) / dur:.1f} fps")
    col.close()
    ds.close()
    report(frames, valid_s, stats, "dual")
    return frames, valid_s


def report(frames, valid_s, stats, tag):
    if not frames:
        print("   !! 一帧都没取到")
        return
    vs = np.array(valid_s)
    print(f"   valid%  中位 {np.median(vs):5.1f}  最小 {vs.min():5.1f}  最大 {vs.max():5.1f}"
          f"  标准差 {vs.std():5.1f}")
    print("   逐帧:")
    print("     " + " ".join(f"{x:4.0f}" for x in vs))
    last = stats[-1]
    print(f"   末帧构成: z=0 {last['zero']:5.1f}% | 0<z<0.3m {last['tiny']:5.1f}%"
          f" | 有效 {last['valid']:5.1f}% | >8m {last['huge']:5.1f}%")
    if "med" in last:
        print(f"   深度范围: p5 {last['p5']:.0f}mm  中位 {last['med']:.0f}mm  p95 {last['p95']:.0f}mm")
    print("   深度直方图:")
    for line in hist_text(frames[-1]):
        print(line)
    valid, _ = analyze(frames[-1])
    print("   空洞空间分布（8 列 x 6 行，每格=该块 valid%）:")
    for line in grid_table(valid):
        print("     " + line)
    # 多帧叠加：常空的地方最黑，闪空的地方灰
    acc = np.zeros(frames[0].shape, np.float32)
    for d in frames[-min(len(frames), 20):]:
        v, _ = analyze(d)
        acc += v
    acc /= min(len(frames), 20)
    fn = save_vis(tag, frames[-1], valid)
    acc8 = (acc * 255).astype(np.uint8)
    cp.imwrite_u(os.path.join(OUT, f"depth_diag_{tag}_stability.png"),
                 cv2.applyColorMap(acc8, cv2.COLORMAP_VIRIDIS))
    print(f"   图: {os.path.relpath(fn, HERE)}  +  depth_diag_{tag}_stability.png"
          "（稳定度：亮=一直有 depth，暗=一直空，灰=时有时无）")


def main():
    ap = argparse.ArgumentParser(description="深度空洞诊断")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--dual", action="store_true")
    ap.add_argument("--sec", type=float, default=6.0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.dual:
        run_dual(args.sec)
    else:
        run_single(args.frames)


if __name__ == "__main__":
    main()
