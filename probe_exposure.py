# -*- coding: utf-8 -*-
"""验证：暗光下 UVC 自动曝光降帧；手动锁短曝光能否恢复 30fps（v2）"""
import time
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)   # 直读原始 MJPG，不解码
time.sleep(1.5)


def rate(secs, label):
    for _ in range(15):
        cap.read()
    t0, n = time.time(), 0
    while time.time() - t0 < secs:
        ok, _f = cap.read()
        if ok:
            n += 1
    print(f"  {label:40s} {n/secs:5.1f} fps")
    return n / secs


print("自动曝光（当前状态）:")
rate(6, "auto")

print("\n尝试锁手动曝光:")
for exp in (-5, -6, -7):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    time.sleep(1.5)
    r = rate(5, f"exp={exp}  (auto={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):.2f}, "
                f"exp={cap.get(cv2.CAP_PROP_EXPOSURE):.2f})")
    if r > 26:
        print(f"  -> exp={exp} 帧率恢复")
        break

cap.release()
