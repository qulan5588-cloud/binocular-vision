"""yolo_seg.py —— YOLO11-seg ONNX 推理包装（DirectML / CPU 通吃）

为什么走 ONNX + onnxruntime-directml 而不是 torch：
    Windows 上 onnxruntime-gpu 要自己装 CUDA + cuDNN（几个 GB、版本对齐地狱），
    DirectML 直接吃任意 DX12 GPU（4060 / 核显都行），pip 一个包就完事。
    torch 只在导出 ONNX 时用一次（CPU 就够），运行时不进 venv。

模型：models/yolo11n_seg.onnx（COCO 80 类，imgsz=640，约 6.4MB）
    导出：from ultralytics import YOLO; YOLO("yolo11n-seg.pt").export(
              format="onnx", imgsz=640, opset=17, dynamic=False)

输出格式（ultralytics 约定）：
    output0: (1, 4+80+32, 8400)  前面 4 = xywh(letterbox 系)、80 = 各类分数、
                                 32 = mask 系数；8400 = 各尺度 anchor 总数
    output1: (1, 32, 160, 160)   原型掩码（proto）

用法：
    det = YoloSeg()                     # 默认 models/yolo11n_seg.onnx
    dets = det.detect(bgr, conf=0.35)   # -> [{cls,name,conf,box,mask}, ...]
    mask 是与输入同尺寸的 bool 图（float 版为 dets[i]["maskf"]）
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "yolo11n_seg.onnx")

# COCO 80 类（YOLO 训练序）
COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]

IMG_SIZE = 640


class YoloSeg:
    def __init__(self, path=MODEL_PATH, providers=None):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        import onnxruntime as ort
        if providers is None:
            avail = set(ort.get_available_providers())
            want = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider")
                    if p in avail]
            providers = (want + ["CPUExecutionProvider"]) if want \
                else ["CPUExecutionProvider"]
        self.ort = ort.InferenceSession(path, providers=providers)
        self.provider = self.ort.get_providers()[0]
        self.inp = self.ort.get_inputs()[0]
        self.is_bgr_out = False

    # ---------------------------------------------------------- 预处理 ---
    def _letterbox(self, bgr):
        h, w = bgr.shape[:2]
        s = min(IMG_SIZE / w, IMG_SIZE / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dx, dy = (IMG_SIZE - nw) // 2, (IMG_SIZE - nh) // 2
        canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 114, np.uint8)
        canvas[dy:dy + nh, dx:dx + nw] = small
        # 记录映射参数：原图坐标 = (letterbox 坐标 - pad) / s
        return canvas, (s, dx, dy, w, h)

    # ---------------------------------------------------------- 后处理 ---
    def detect(self, bgr, conf=0.35, iou=0.55, max_det=20, mask_thresh=0.5):
        lb, (s, dx, dy, w, h) = self._letterbox(bgr)
        x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]                      # 1x3x640x640

        out0, proto = self.ort.run(None, {self.inp.name: x})
        out0 = out0[0].T                                    # (8400, 116)
        nb, ncls, ncoef = 4, len(COCO), out0.shape[1] - 4 - len(COCO)
        boxes_lb = out0[:, :nb]
        scores = out0[:, nb:nb + ncls]
        coeffs = out0[:, nb + ncls:]                        # (8400, 32)

        cls_id = scores.argmax(axis=1)
        cls_conf = scores.max(axis=1)
        keep = cls_conf >= conf
        if not keep.any():
            return []
        boxes_lb, cls_conf, cls_id, coeffs = (boxes_lb[keep], cls_conf[keep],
                                              cls_id[keep], coeffs[keep])
        idx = cv2.dnn.NMSBoxes(boxes_lb.tolist(), cls_conf.tolist(), conf, iou)
        idx = np.array(idx).reshape(-1)
        boxes_lb, cls_conf, cls_id, coeffs = (boxes_lb[idx], cls_conf[idx],
                                              cls_id[idx], coeffs[idx])
        boxes_lb = boxes_lb[:max_det]
        cls_conf, cls_id, coeffs = cls_conf[:max_det], cls_id[:max_det], coeffs[:max_det]

        # 掩码：系数点乘 proto -> sigmoid -> 阈值化（proto 在 letterbox 系 160x160）
        p = proto[0].reshape(ncoef, -1)                     # 32 x 25600
        mf = (coeffs @ p).reshape(-1, 160, 160)
        mf = 1.0 / (1.0 + np.exp(-mf))
        mres = cv2.resize(mf.transpose(1, 2, 0), (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_LINEAR)
        if mres.ndim == 2:                      # 单检出时 resize 会压掉通道维
            mres = mres[:, :, None]
        mf = mres.transpose(2, 0, 1)

        dets = []
        for i in range(len(boxes_lb)):
            cx, cy, bw, bh = boxes_lb[i]
            # letterbox -> 原图
            x0 = (cx - bw / 2 - dx) / s
            y0 = (cy - bh / 2 - dy) / s
            x1 = (cx + bw / 2 - dx) / s
            y1 = (cy + bh / 2 - dy) / s
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(w - 1, int(x1)), min(h - 1, int(y1))
            # 掩码裁回原图有效区
            m = np.zeros((h, w), bool)
            sx0, sy0 = max(0, dx), max(0, dy)
            m_part = mf[i][sy0:sy0 + h, sx0:sx0 + w] > mask_thresh
            m_part = m_part[:h - sy0 if sy0 + h > h else h, :w - sx0 if sx0 + w > w else w]
            m[sy0:, sx0:][:m_part.shape[0], :m_part.shape[1]] = m_part
            dets.append(dict(cls=int(cls_id[i]), name=COCO[int(cls_id[i])],
                             conf=float(cls_conf[i]),
                             box=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                             mask=m, maskf=mf[i]))
        dets.sort(key=lambda d: -d["conf"])
        return dets


def draw_dets(canvas, dets, depth_stats=None):
    """把检测结果画到彩色画布上。depth_stats: {i: str} 额外文字（如尺寸）。"""
    pal = [(80, 200, 255), (120, 255, 160), (255, 160, 120), (200, 120, 255),
           (255, 220, 90), (110, 220, 255), (180, 255, 120), (255, 120, 200)]
    for i, d in enumerate(dets):
        col = pal[i % len(pal)]
        x0, y0, w, h = d["box"]
        overlay = canvas[y0:y0 + h, x0:x0 + w]
        mo = d["mask"][y0:y0 + h, x0:x0 + w]
        overlay[mo] = (overlay[mo] * 0.45 + np.array(col) * 0.55).astype(np.uint8)
        cv2.rectangle(canvas, (x0, y0), (x0 + w - 1, y0 + h - 1), col, 1, cv2.LINE_AA)
        txt = f"{d['name']} {d['conf']:.2f}"
        extra = (depth_stats or {}).get(i)
        if extra:
            txt += "  " + extra
        ty = y0 - 8 if y0 >= 22 else y0 + 18
        cv2.putText(canvas, txt, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, txt, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    col, 1, cv2.LINE_AA)
    return canvas
