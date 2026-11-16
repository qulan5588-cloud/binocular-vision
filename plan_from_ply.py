"""plan_from_ply.py —— 用当前的地板平面算法重渲已保存的扫描

扫描数据（cloud.ply + traj.csv）存下来以后，渲染逻辑还在迭代 —— 这个
工具不碰相机，直接读 PLY 重新拟合地板、重新出平面图，用于：
  * 救旧扫描（算法升级后不必重扫）；
  * 想换个高度切片 / 换个层看家具，不用再端相机走一圈。

用法：
    python plan_from_ply.py captures/scan_20261116_214401
    python plan_from_ply.py captures/scan_xxx --band 0.8 --width 0.3
    python plan_from_ply.py captures/scan_xxx --size 1600
"""
import argparse
import csv
import os

import cv2
import numpy as np

import app_scan as S


def load_ply(path):
    """读 binary_little_endian PLY（本项目的 save_ply 格式：xyz + uchar rgb）"""
    with open(path, "rb") as fh:
        head = b""
        while b"end_header\n" not in head:
            head += fh.read(1)
        n = 0
        for line in head.decode("ascii").splitlines():
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
        rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                 ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        buf = fh.read(rec.itemsize * n)
    rec = np.frombuffer(buf, dtype=rec.dtype)
    pts = np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float32)
    col = np.stack([rec["red"], rec["green"], rec["blue"]], 1).copy()
    return pts, col


def main():
    ap = argparse.ArgumentParser(description="重渲已保存扫描的平面图")
    ap.add_argument("dir", help="扫描目录（含 cloud.ply / traj.csv）")
    ap.add_argument("--band", type=float, default=0.10, help="切片中心（地板上方 m）")
    ap.add_argument("--width", type=float, default=0.20, help="切片半带宽（m）")
    ap.add_argument("--size", type=int, default=1080, help="输出图边长")
    ap.add_argument("--suffix", default="_refit", help="输出文件名后缀")
    args = ap.parse_args()

    pts, _col = load_ply(os.path.join(args.dir, "cloud.ply"))
    print(f"点云 {len(pts) / 1000:.0f}k", flush=True)
    fl = S.floor_basis(pts)
    if fl is None:
        print("地板拟合失败：点太少或全是竖直面")
        return
    _c, _e1, _e2, n, _d0 = fl
    print(f"地板法线 {n.round(3)}（竖直度 {np.degrees(np.arccos(min(1, abs(n[1])))):.1f}° 偏离竖直）",
          flush=True)

    traj = []
    tp = os.path.join(args.dir, "traj.csv")
    if os.path.isfile(tp):
        with open(tp, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        traj = [np.array([float(r["x"]), float(r["y"]), float(r["z"])], np.float32)
                for r in rows]

    m = S.Map()
    m.pts = pts
    m.col = _col
    snap = dict(ppm=None)
    S.make_snapshot(m, snap, fl)

    for tag, band in (("", (args.band, args.width)), ("_full", None)):
        img = np.zeros((args.size, args.size, 3), np.uint8)
        S.render_plan(img, snap, traj, np.eye(4), True, band=band, floor=fl)
        out = os.path.join(args.dir, f"plan{args.suffix}{tag}.png")
        cv2.imwrite(out, img)
        print("saved", out, flush=True)


if __name__ == "__main__":
    main()
