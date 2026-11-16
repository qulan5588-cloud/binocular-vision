"""对比不同 OpenCV 读取方式的实际吞吐，找出彩色流的瓶颈。"""
import time
import cv2
import numpy as np

W, H, FPS = 1280, 720, 30


def bench(name, cap, decode):
    for _ in range(10):
        cap.read()
    t0 = time.time()
    n = 0
    while time.time() - t0 < 5.0:
        ok, f = cap.read()
        if ok:
            n += 1
    dt = time.time() - t0
    rate = n / dt
    extra = ""
    if n and decode:
        extra = f"  shape={f.shape}"
    elif n and not decode:
        extra = f"  raw_bytes={f.size}"
    print(f"  {name:<42} {rate:5.1f} fps{extra}")
    cap.release()
    return rate


def open_cap(backend, w=W, h=H, fourcc="MJPG", bufsize=None, convert_rgb=True):
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if bufsize is not None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, bufsize)
    if not convert_rgb:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    return cap


print("=" * 62)
print(f" OpenCV 读取吞吐对比  {W}x{H} MJPG   (目标 {FPS} fps)")
print("=" * 62)

cap = open_cap(cv2.CAP_DSHOW, bufsize=1)
if cap:
    bench("DSHOW  BUFFERSIZE=1  (当前配置)", cap, True)

cap = open_cap(cv2.CAP_DSHOW)
if cap:
    bench("DSHOW  默认 BUFFERSIZE", cap, True)

cap = open_cap(cv2.CAP_DSHOW, convert_rgb=False)
if cap:
    r = bench("DSHOW  CONVERT_RGB=0 (不解码)", cap, False)
    print(f"       -> MJPG 原始流 {r:.1f} fps，约 {r * 8 / 1024:.1f} MB/s")

cap = open_cap(cv2.CAP_MSMF)
if cap:
    bench("MSMF   MJPG", cap, True)
else:
    print("  MSMF   不可用")

cap = open_cap(cv2.CAP_DSHOW, fourcc="YUY2")
if cap:
    bench("DSHOW  YUY2 (无压缩)", cap, True)

print("-" * 62)
