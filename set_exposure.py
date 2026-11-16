# -*- coding: utf-8 -*-
"""用 OpenCV 锁定手动曝光并释放相机，让外部工具能独立测帧率"""
import sys
import time
import cv2

exp = float(sys.argv[1]) if len(sys.argv) > 1 else -5.0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FPS, 30)
time.sleep(1.0)

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 手动
cap.set(cv2.CAP_PROP_EXPOSURE, exp)
time.sleep(1.0)
print(f"已设置手动曝光 exp={exp}  "
      f"(读回 auto={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):.2f}, "
      f"exp={cap.get(cv2.CAP_PROP_EXPOSURE):.2f})")

# 保持 3 秒让相机应用设置，然后不 release 复位（避免 AE 恢复）
time.sleep(3.0)
cap.release()
print("相机已释放，立即用 ffmpeg 数帧率验证")
