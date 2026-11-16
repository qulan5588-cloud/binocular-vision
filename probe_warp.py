"""probe_warp.py —— 逐级检查 Aligner.warp 的丢点环节

背景：原始深度有效率 ~49%，但 app_align HUD 显示 cov 只有 3.4%。
本脚本取一帧真实深度，跑 warp 的每一步，打印每步剩多少点，
并输出投影坐标的分布范围，定位丢失发生在哪一级。
"""
import os
import sys

import cv2
import numpy as np

import capture_pair as cp
from app_align import Aligner, latest_calib, load_calib

HERE = os.path.dirname(os.path.abspath(__file__))

cal = load_calib(latest_calib())
print("calib K_d=\n", cal["K_d"], "\nT=", cal["T"])

ds = cp.DepthSource()
ds.start()
frame = None
for _ in range(120):
    it = ds.latest()
    if it is not None:
        frame = it[0]
    # 不 sleep，尽快拿满缓冲
ds.close()
if frame is None:
    sys.exit("没取到深度帧")

d = cv2.flip(frame, 1)          # 与 app_align 相同：先翻转
z = d.astype(np.float32) * 0.001
ok0 = z > 0
print(f"\n[0] flip 后有效深度        {ok0.mean()*100:6.1f}%")

al = Aligner(cal, (d.shape[0], d.shape[1]), (720, 1280))
u, v = al.u, al.v
Kd = cal["K_d"]
x = (u - Kd[0, 2]) * z / Kd[0, 0]
y = (v - Kd[1, 2]) * z / Kd[1, 1]

R, T = cal["R"], cal["T"]
xc = R[0, 0]*x + R[0, 1]*y + R[0, 2]*z + T[0]
yc = R[1, 0]*x + R[1, 1]*y + R[1, 2]*z + T[1]
zc = R[2, 0]*x + R[2, 1]*y + R[2, 2]*z + T[2]
ok1 = ok0 & (zc > 0.05)
print(f"[1] zc>0.05 后             {ok1.sum()/307200*100:6.1f}%")

xn, yn = xc / np.maximum(zc, 1e-6), yc / np.maximum(zc, 1e-6)
k1, k2, p1, p2, k3 = cal["D_c"][:5]
r2 = xn*xn + yn*yn
rad = 1 + k1*r2 + k2*r2**2 + k3*r2**3
xd = xn*rad + 2*p1*xn*yn + p2*(r2 + 2*xn*xn)
yd = yn*rad + p1*(r2 + 2*yn*yn) + 2*p2*xn*yn
Kc = cal["K_c"]
uc = xd*Kc[0, 0] + Kc[0, 2]
vc = yd*Kc[1, 1] + Kc[1, 2]

m_in = ok1 & (uc >= 0) & (uc < 1280) & (vc >= 0) & (vc < 720)
print(f"[2] 投影落入 1280x720 内    {m_in.sum()/307200*100:6.1f}%")
for name, arr, msk in (("uc", uc, ok1), ("vc", vc, ok1)):
    a = arr[msk]
    print(f"    {name}: p1={np.percentile(a,1):8.1f}  中位={np.median(a):8.1f}"
          f"  p99={np.percentile(a,99):8.1f}")

# 最终跑一遍真正的 warp，验证一致
zmap, proj = al.warp(d)
print(f"[3] Aligner.warp 的 cov     {proj[3].mean()*100:6.1f}%")
