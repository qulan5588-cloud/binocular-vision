"""app_gesture.py —— 手势交互：食指当光标、捏合当点击

MediaPipe HandLandmarker（`models/hand_landmarker.task`，7.8MB，CPU 单帧
10-20ms）给出 21 个手部关键点，在此之上做三件事：

  1. **手势识别**：按"指尖到手腕的距离 vs 该指 PIP 到手腕的距离"判每根手指
     是否伸直（比看角度稳，手掌朝向变了也不失效），组合出
     握拳 / 张开掌 / 指向 / V / 拇指 / 捏合。
  2. **虚拟光标**：食指尖做指针（EMA 平滑），食指捏拇指 = 点击（带
     迟滞，避免抖动连点）。
  3. **可点界面**：四个按钮 —— 镜像 / 深度叠加 / 截图 / 重置 —— 放在
     画面**右侧的独立面板**里，不压画面。光标悬停 0.9 秒自动触发
     （dwell click），也可以用鼠标直接点按钮。

镜像与坐标：相机画面水平镜像后，手部骨架、光标、深度叠加都必须在同一
套"画布坐标"里，否则镜像时骨架子会跑到反方向（手在右、骨架在左）。
统一由 `View` 做 相机像素 -> 画布 的映射，需要深度时再 `x_back` 反算
回相机坐标（深度投影 proj 永远是相机坐标系）。

按钮在画面外，手指够不到怎么办？画面最右侧 16%（EDGE_BAND）是"推出去"
的过渡带：手指越过这条缝，光标就从画面边缘滑到面板上，纵向选按钮。

为什么要这个：以后在仓储场景"用手指着货位比划一下"比掏鼠标快，
这套光标 + 按钮的骨架可以直接复用到货位选择上。

用法：
    python app_gesture.py               # 实时手势交互
    python app_gesture.py --mirror      # 画面镜像（与人对面操作时更自然）
    python app_gesture.py --image a.png # 单张图跑识别（不占相机/不下载模型以外的资源）
    python app_gesture.py --shot out.png

按键：m 镜像，f 深度叠加，s 截图，r 重置，q / 窗口 X 退出。
"""
import argparse
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision

import capture_pair as cp
from app_align import Aligner, adapt_cal, latest_calib, load_calib
from ui_common import wait_key

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_detect")
MODEL = os.path.join(HERE, "models", "hand_landmarker.task")

# MediaPipe 手部 21 点的骨架连线（1.x 不再提供 solutions.HAND_CONNECTIONS）
HAND_CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
             (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
             (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

TIP, PIP = (4, 8, 12, 16, 20), (3, 6, 10, 14, 18)
DWELL_SEC = 0.9           # 悬停多久自动触发
CLICK_COOLDOWN = 1.0      # 触发后的冷却，防连点
PINCH_ON = 0.30           # 捏合判定：指尖距 < 0.30 * 掌心长度
PINCH_OFF = 0.45          # 松开判定（迟滞），避免边界抖动反复点击

PANEL_W = 292             # 画面右侧的按钮面板宽度（在画面之外，不遮挡）
BTN_W, BTN_H, BTN_GAP = 248, 96, 16
BTN_PAD_X = (PANEL_W - BTN_W) // 2
BTN_Y0 = 58

EDGE_BAND = 0.16          # 画面最右侧这一段：手指伸进去，光标滑出画面到面板


# --------------------------------------------------------------- 手势 ---
def finger_states(lm, w):
    """返回 5 根手指是否伸直 [拇指, 食指, 中指, 无名指, 小指]。

    判据：指尖到手腕的距离 > 该指 PIP 关节到手腕的距离 × 1.08。
    比"看弯曲角度"稳：手掌朝上/朝下、侧对相机都不受影响。
    拇指单独处理 —— 它的 PIP 离手腕本来就近，直接比会一直判成弯的，
    改用"指尖到食指根的距离 vs 拇指 IP 到食指根的距离"。
    """
    wrist = np.array([lm[0].x * w, lm[0].y * w])
    out = []
    for tip, pip in zip(TIP, PIP):
        t = np.array([lm[tip].x * w, lm[tip].y * w])
        p = np.array([lm[pip].x * w, lm[pip].y * w])
        out.append(float(np.linalg.norm(t - wrist)) > float(np.linalg.norm(p - wrist)) * 1.08)
    idx_mcp = np.array([lm[5].x * w, lm[5].y * w])
    th_tip = np.array([lm[4].x * w, lm[4].y * w])
    th_ip = np.array([lm[3].x * w, lm[3].y * w])
    out[0] = float(np.linalg.norm(th_tip - idx_mcp)) > float(np.linalg.norm(th_ip - idx_mcp)) * 1.06
    return out


def palm_len(lm, w):
    """掌心尺度：手腕到中指根，用来把像素距离归一化（不同距离的手可比）"""
    a = np.array([lm[0].x * w, lm[0].y * w])
    b = np.array([lm[9].x * w, lm[9].y * w])
    return max(float(np.linalg.norm(a - b)), 1.0)


def pinch_distance(lm, w):
    """拇指尖到食指尖的距离 / 掌心长度。捏合时接近 0，张开约 1.2~1.8"""
    a = np.array([lm[4].x * w, lm[4].y * w])
    b = np.array([lm[8].x * w, lm[8].y * w])
    return float(np.linalg.norm(a - b)) / palm_len(lm, w)


def gesture_name(lm, w, pinching):
    st = finger_states(lm, w)
    n = sum(st)
    if pinching:
        return "PINCH", st
    if n == 0:
        return "FIST", st
    if n == 5:
        return "OPEN", st
    if st[1] and not any(st[2:]):
        return "POINT", st
    if st[1] and st[2] and not any(st[3:]):
        return "VICTORY", st
    if st[0] and not any(st[1:]):
        return "THUMB", st
    return f"{n} FINGER" + ("S" if n > 1 else ""), st


# --------------------------------------------------------------- 绘制 ---
class View:
    """相机像素坐标 <-> 画布坐标 的映射（镜像时 x 反转）。

    MediaPipe 始终吃**原始相机帧**（画面中途翻转会让 VIDEO 模式的跟踪器
    跟丢/跳一下），所以关键点坐标是相机坐标；显示镜像只体现在这里。
    以前骨架直接用相机坐标画到镜像画布上，结果"手在右、骨架在左"。
    """

    def __init__(self, w, h, mirror):
        self.w, self.h, self.m = int(w), int(h), bool(mirror)

    def pt(self, p):
        """landmark（归一化）-> 画布整数坐标，夹在画面内，别画到面板上"""
        x = p.x * self.w
        if self.m:
            x = self.w - 1 - x
        return (int(min(max(x, 0), self.w - 1)),
                int(min(max(p.y * self.h, 0), self.h - 1)))

    def x_back(self, cx):
        """画布 x -> 相机 x（深度投影 proj 是相机坐标系，取深度必须反算）"""
        return int(self.w - 1 - cx) if self.m else int(cx)

    def in_panel(self, cx):
        return cx >= self.w


# --------------------------------------------------------------- 绘制 ---
def draw_hand(canvas, lm, view, col):
    pts = [view.pt(p) for p in lm]
    for a, b in HAND_CONN:
        cv2.line(canvas, pts[a], pts[b], (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(canvas, pts[a], pts[b], col, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = 6 if i in TIP else 3
        cv2.circle(canvas, p, r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p, r, col, 2, cv2.LINE_AA)


def button_rect(i, vw):
    """按钮矩形。x 从画面右边界 vw 起算 —— 按钮全在画面外的面板里"""
    x0 = vw + BTN_PAD_X
    y0 = BTN_Y0 + i * (BTN_H + BTN_GAP)
    return x0, y0, x0 + BTN_W, y0 + BTN_H


def cursor_target(view, lm):
    """食指尖 -> 画布坐标。指尖压到画面最右侧 EDGE_BAND 带里，光标滑出画面进面板。

    按钮搬到画面外后指尖永远够不到，所以留这条"推出去"的过渡带：
    位置连续（缝上等于画面右边界），只是越过后横向比例变快，纵向不变，
    于是"手往右一推 + 上下移"就能选到四个按钮。
    """
    tx, ty = view.pt(lm[8])
    seam = int(view.w * (1.0 - EDGE_BAND))
    if tx >= seam:
        t = (tx - seam) / max(1.0, float(view.w - 1 - seam))
        tx = view.w + int(min(t, 1.0) * (PANEL_W - 74))
    return tx, ty


def panel_status(state):
    return ["ON" if state["mirror"] else "OFF",
            "ON" if state["show_depth"] else "OFF", "", ""]


def draw_panel(canvas, vw, labels, hover, flash, hits, state):
    hh, ww = canvas.shape[:2]
    cv2.rectangle(canvas, (vw, 0), (ww - 1, hh - 1), (22, 23, 28), -1)
    cv2.line(canvas, (vw, 0), (vw, hh - 1), (54, 58, 66), 2)
    cv2.putText(canvas, "GESTURE", (vw + BTN_PAD_X, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (110, 116, 130), 1, cv2.LINE_AA)
    st = panel_status(state)
    for i, (key, label) in enumerate(labels):
        x0, y0, x1, y1 = button_rect(i, vw)
        on = (i == hover)
        col = (90, 235, 150) if on else (80, 84, 94)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (30, 32, 38), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), col, 3 if on else 2)
        if on:
            # 悬停进度条：填满即 dwell 触发
            p = min(1.0, flash[i] / DWELL_SEC)
            cv2.rectangle(canvas, (x0, y1 - 6), (x0 + int((x1 - x0) * p), y1),
                          (90, 235, 150), -1)
        cv2.putText(canvas, label, (x0 + 16, y0 + 44), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, (x0 + 16, y0 + 44), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68, col if on else (210, 214, 222), 2, cv2.LINE_AA)
        sub = f"hotkey {key}   x{hits[i]}"
        if st[i]:
            sub += f"   {st[i]}"
        cv2.putText(canvas, sub, (x0 + 16, y0 + 74), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (130, 136, 148), 1, cv2.LINE_AA)
    # 面板底部操作提示（原来压在画面右下角）
    hint = ["point = move", "pinch / hover = click", "mouse click ok",
            "m f s r   q = quit"]
    for i, s in enumerate(hint):
        cv2.putText(canvas, s, (vw + BTN_PAD_X, hh - 96 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 156, 168), 1, cv2.LINE_AA)


def draw_cursor(canvas, x, y, pinching, prog):
    r = 26 if pinching else 20
    col = (80, 220, 255) if pinching else (240, 240, 240)
    cv2.circle(canvas, (x, y), r, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), r, col, 2, cv2.LINE_AA)
    if prog > 0:
        cv2.ellipse(canvas, (x, y), (r + 6, r + 6), -90, 0, int(360 * min(prog, 1.0)),
                    (90, 235, 150), 3, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), 3, col, -1, cv2.LINE_AA)


def draw_hud(canvas, gname, nfingers, dist, fps, nhands, msg):
    txt = f"{gname}"
    cv2.putText(canvas, txt, (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (0, 0, 0), 7, cv2.LINE_AA)
    cv2.putText(canvas, txt, (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (90, 235, 150), 3, cv2.LINE_AA)
    l2 = f"fingers {nfingers}   " + (f"{dist:.2f}m   " if dist else "--   ") + \
         f"hands {nhands}   fps {fps:4.1f}"
    cv2.putText(canvas, l2, (26, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, l2, (26, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (215, 215, 215), 1, cv2.LINE_AA)
    if msg:
        cv2.putText(canvas, msg, (26, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, msg, (26, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (90, 235, 150), 2, cv2.LINE_AA)


# --------------------------------------------------------------- 主流程 ---
def main():
    ap = argparse.ArgumentParser(description="手势交互：食指光标 + 捏合点击")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--shot", default="")
    ap.add_argument("--image", default="")
    ap.add_argument("--conf", type=float, default=0.5, help="手部检测置信度")
    args = ap.parse_args()

    if not os.path.exists(MODEL):
        raise SystemExit(f"缺模型 {MODEL}\n先下载 hand_landmarker.task 放到 models/")
    base = mp_py.BaseOptions(model_asset_path=MODEL)

    if args.image:                                    # 单图模式：只验证识别
        det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base, running_mode=vision.RunningMode.IMAGE,
            num_hands=2, min_hand_detection_confidence=args.conf))
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"读不到 {args.image}")
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = time.time()
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        dt = (time.time() - t) * 1000
        canvas = img.copy()
        nhands = len(res.hand_landmarks or [])
        gname, nf = "NONE", 0
        vimg = View(w, h, False)
        for i, lm in enumerate(res.hand_landmarks or []):
            pd = pinch_distance(lm, w)
            g, st = gesture_name(lm, w, pd < PINCH_ON)
            draw_hand(canvas, lm, vimg, (90, 235, 150) if i == 0 else (255, 190, 110))
            if i == 0:
                gname, nf = g, sum(st)
            print(f"  手{i + 1}: {g}  手指={sum(st)}  捏合距离={pd:.2f}")
        draw_hud(canvas, gname, nf, 0.0, 0.0, nhands, f"{dt:.0f} ms  {nhands} hand(s)")
        out = args.shot or os.path.join(OUT, "gesture_image.png")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        cp.imwrite_u(out, canvas)
        print(f"{args.image}: {nhands} 只手 -> {out}")
        return

    det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=base, running_mode=vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=args.conf))

    depth = cp.DepthSource()
    depth.start()
    color = cp.ColorSourceFFmpeg(1280, 720, 30, device=cp.FFMPEG_DEVICE)
    color.start()
    cal = load_calib(latest_calib())
    cal, _dsz, csz = adapt_cal(cal, depth.mode, color.mode)
    align = Aligner(cal, _dsz, csz)

    labels = [("m", "MIRROR"), ("f", "DEPTH"), ("s", "SNAPSHOT"), ("r", "RESET")]
    state = dict(mirror=args.mirror, show_depth=False, msg="",
                 hover=-1, hover_t=0.0, hold=[0.0] * 4, hits=[0] * 4,
                 cooldown=0.0, cx=0, cy=0, show_depth_map=None, proj=None,
                 pinch_prev=False, fps=0.0, n_calls=0)
    win = "Gesture  [point=move  pinch=click  q=quit]"
    clicks = []
    if not args.shot:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

        def on_mouse(event, x, y, flags, param):
            # 回调里只记账，绝不做重活（Popen/IO 会卡住消息泵，窗口就"点不动"）
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks.append((x, y))

        cv2.setMouseCallback(win, on_mouse)
    print("手势交互已启动：食指尖悬停按钮 0.9 秒或捏合即触发；"
          "手指推到画面最右侧会滑出画面进入按钮面板。q 退出", flush=True)

    def trigger(i, now):
        key = labels[i][0]
        state["hits"][i] += 1
        state["cooldown"] = now + CLICK_COOLDOWN
        state["hover_t"] = 0.0
        if key == "m":
            state["mirror"] = not state["mirror"]
            # 镜像翻转后画布坐标整体反了，光标锚点必须重seed，否则会从旧位置飘过去
            state["cx"] = state["cy"] = 0
            state["msg"] = "MIRROR " + ("ON" if state["mirror"] else "OFF")
        elif key == "f":
            state["show_depth"] = not state["show_depth"]
            state["msg"] = "DEPTH " + ("ON" if state["show_depth"] else "OFF")
        elif key == "s":
            os.makedirs(OUT, exist_ok=True)
            fn = os.path.join(OUT, time.strftime("gesture_%H%M%S.png"))
            state["pending_save"] = fn
            state["msg"] = "SAVED " + os.path.basename(fn)
        elif key == "r":
            state["hits"] = [0] * 4
            state["msg"] = "RESET"

    t_prev = time.time()
    last_ts = None
    frame_i = 0
    try:
        while True:
            cz = color.latest()
            if cz is None:
                time.sleep(0.005)
                continue
            cts = cz[1] if len(cz) > 1 else None
            if cts is not None and cts == last_ts:
                time.sleep(0.004)
                continue
            last_ts = cts
            img = cp.ColorSourceFFmpeg.decode(cz[0]) if getattr(color, "is_jpeg", False) else cz[0]
            h, w = img.shape[:2]
            frame_i += 1

            # 1) 手部关键点（VIDEO 模式：内部带跟踪，比逐帧 IMAGE 稳且快）
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ts_ms = int(time.time() * 1000)
            if ts_ms <= state["n_calls"]:
                ts_ms = state["n_calls"] + 1
            state["n_calls"] = ts_ms
            res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                data=rgb), ts_ms)
            hands = res.hand_landmarks or []

            # 2) 深度：手掌中心距离（隔帧刷新，warp 22ms）
            dz = depth.latest()
            if dz is not None and frame_i % 2 == 1:
                _zmap, state["proj"] = align.warp(cv2.flip(dz[0], 1))

            # 画布 = 画面 + 右侧面板；镜像/非镜像只在这里体现一次
            view = View(w, h, state["mirror"])
            vw = w
            canvas = np.zeros((h, vw + PANEL_W, 3), np.uint8)
            canvas[:, :vw] = cv2.flip(img, 1) if state["mirror"] else img
            if state["show_depth"] and state["proj"] is not None:
                uc, vc, zc, m = state["proj"]
                dm = np.zeros((h, w), np.uint8)
                g = np.clip((zc[m] - 0.3) / 3.7, 0, 1)
                dm[vc[m].astype(int), uc[m].astype(int)] = (g * 255).astype(np.uint8)
                if state["mirror"]:
                    dm = cv2.flip(dm, 1)
                canvas[:, :vw] = cv2.addWeighted(
                    canvas[:, :vw], 0.62,
                    cv2.applyColorMap(dm, cv2.COLORMAP_TURBO), 0.38, 0)

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                state["fps"] = 0.9 * state["fps"] + 0.1 / dt if state["fps"] else 1.0 / dt

            # 3) 每只手：手势 + 光标
            gname, nf, dist = "NONE", 0, 0.0
            target = None
            pinching = False
            for hi, lm in enumerate(hands):
                pd = pinch_distance(lm, w)
                on = pd < (PINCH_OFF if state["pinch_prev"] else PINCH_ON)
                if hi == 0:
                    pinching = on
                    gname, st = gesture_name(lm, w, on)
                    nf = sum(st)
                # 骨架走 view，镜像时才跟手在一边（以前画在相机坐标上，镜像即错位）
                draw_hand(canvas, lm, view,
                          (90, 235, 150) if hi == 0 else (255, 190, 110))

            if hands:
                lm = hands[0]
                tx, ty = cursor_target(view, lm)
                # EMA 平滑：关键点本身有 2~3px 抖动，直接当地标会抖得点不中
                if state["cx"] == 0 and state["cy"] == 0:
                    state["cx"], state["cy"] = tx, ty
                else:
                    state["cx"] = int(0.55 * state["cx"] + 0.45 * tx)
                    state["cy"] = int(0.55 * state["cy"] + 0.45 * ty)
                # 手掌中心距离（lm[9] 中指根）：proj 是相机坐标系，先反算回相机 x
                if state["proj"] is not None:
                    hx, hy = view.pt(lm[9])
                    dist = align.sample(state["proj"], view.x_back(hx), hy)
                for i in range(len(labels)):
                    x0, y0, x1, y1 = button_rect(i, vw)
                    if x0 <= state["cx"] <= x1 and y0 <= state["cy"] <= y1:
                        target = i
                        break

            # 鼠标点面板按钮（手举累了可以直接点）
            while clicks:
                mx, my = clicks.pop(0)
                for i in range(len(labels)):
                    x0, y0, x1, y1 = button_rect(i, vw)
                    if x0 <= mx <= x1 and y0 <= my <= y1:
                        if now > state["cooldown"]:
                            trigger(i, now)
                        break

            # 4) 悬停累积 / 捏合点击
            if target != state["hover"]:
                state["hover"] = target
                state["hover_t"] = 0.0
            elif target is not None:
                state["hover_t"] += dt
            for i in range(len(labels)):
                state["hold"][i] = state["hover_t"] if i == target else 0.0

            if now > state["cooldown"]:
                if pinching and not state["pinch_prev"] and target is not None:
                    trigger(target, now)
                elif state["hover_t"] > DWELL_SEC and target is not None:
                    trigger(target, now)
            state["pinch_prev"] = pinching

            if state.get("pending_save"):
                cp.imwrite_u(state["pending_save"], canvas)
                print("saved", state["pending_save"], flush=True)
                state["pending_save"] = ""

            # 5) 绘制
            draw_panel(canvas, vw, labels, state["hover"], state["hold"],
                       state["hits"], state)
            if hands:
                fx, fy = view.pt(hands[0][8])
                if view.in_panel(state["cx"]):
                    # 光标滑出画面时拉一条细线回指尖，免得看着像"光标丢了"
                    cv2.line(canvas, (fx, fy), (state["cx"], state["cy"]),
                             (90, 235, 150), 1, cv2.LINE_AA)
                draw_cursor(canvas, state["cx"], state["cy"], pinching,
                            state["hover_t"] / DWELL_SEC if target is not None else 0.0)
            draw_hud(canvas, gname, nf, dist, state["fps"], len(hands), state["msg"])

            if args.shot:
                cp.imwrite_u(args.shot, canvas)
                print(f"手势 {gname} 手指数 {nf} 距离 {dist:.2f}m 手数 {len(hands)}")
                break

            cv2.imshow(win, canvas)
            k = wait_key(win, 1)      # 点窗口 X 也能退出
            if k in (27, ord("q")):
                break
            if k == ord("m"):
                trigger(0, now)
            elif k == ord("f"):
                trigger(1, now)
            elif k == ord("s"):
                trigger(2, now)
            elif k == ord("r"):
                trigger(3, now)
    finally:
        cv2.destroyAllWindows()
        color.close()
        depth.close()


if __name__ == "__main__":
    main()
