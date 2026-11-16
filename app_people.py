"""app_people.py —— 人数统计 + 人脸追踪

一件事分两层做：
  1) **检测**：YuNet（OpenCV 自带的 cv2.FaceDetectorYN，227KB，1280x720 单帧 21ms）。
     OpenCV 5 的精简包里已经不附带人脸 Haar xml 了，而且 Haar 对侧脸/小脸误检多。
  2) **追踪**：自写轻量多目标跟踪 —— 按"中心距 / 人脸尺度"贪心配对上一帧轨迹，
     连续命中 3 帧才确认、丢失 12 帧注销。这样 ID 稳定，不会每帧新编号。
     （不用 cv2 的 KCF/CSRT：那几个是为单目标设计的，重建成本高，也没法处理进出。）

在此基础上做人数统计：
  * 当前人数 = 已确认轨迹数
  * 累计进出 = 人脸中心跨越画面中的计数线（带 ±30px 滞后带，防在线上抖动重复计数）
  * 每人一个距离读数（深度图对齐后取人脸框中心区域的中位深度）

用法：
    python app_people.py                    # 实时人数统计 + 追踪
    python app_people.py --fast             # 960x540 检测（约 12ms，省一半 CPU）
    python app_people.py --line 0.5         # 计数线位置（画面宽度比例，默认 0.5）
    python app_people.py --thresh 0.7       # 检测置信度阈值（默认 0.6）
    python app_people.py --mirror           # 画面镜像显示（计数方向不变）
    python app_people.py --shot out.png     # 抓一帧出图后退出
    python app_people.py --image a.png      # 单张图片检测出图（不占相机）

按键：m = 镜像，l = 显隐计数线，[ / ] = 检测阈值 ±0.05，s = 存图，
      窗口 X 或 q = 退出。
"""
import argparse
import math
import os
import time

import cv2
import numpy as np

import capture_pair as cp
from app_align import Aligner, adapt_cal, latest_calib, load_calib
from ui_common import wait_key

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_detect")
MODEL = os.path.join(HERE, "models", "face_detection_yunet_2023mar.onnx")

CONFIRM_HITS = 3          # 连续命中多少帧才认作"在场的人"
MAX_MISS = 12             # 连续丢失多少帧后注销轨迹
MIN_BOX = 26              # 人脸框最小边长(px)：更小的当噪声，远处小人脸不要
LINE_DEAD = 30            # 计数线两侧的滞后带(px)
MATCH_COST = 1.2          # 匹配代价上限（中心距 / 人脸尺度）

PALETTE = [(90, 200, 255), (120, 235, 150), (255, 190, 110),
           (215, 150, 255), (255, 150, 190), (150, 220, 255),
           (190, 255, 130), (130, 180, 255)]


# ----------------------------------------------------------------- 追踪 ---
class Track:
    """一条人脸轨迹。未确认（tentative）不计数、画虚线框。"""

    __slots__ = ("tid", "box", "hits", "miss", "side", "dist", "score", "vx", "vy")

    def __init__(self, tid, box, score=0.0):
        self.tid = tid
        self.box = np.asarray(box, np.float32)     # x, y, w, h
        self.hits = 1
        self.miss = 0
        self.side = 0                              # 相对计数线的历史位置：-1 左 / +1 右
        self.dist = 0.0                            # 人脸距离(米)，0 = 无深度
        self.score = score
        self.vx = 0.0                              # 中心速度(px/帧)，用于预测
        self.vy = 0.0

    def predict(self):
        """下一帧中心位置的预测（匀速模型）。

        交叉时非它不可：两个人迎面走过，在画面上重合的那一两帧里，只按
        "谁离得近"配对必然配反，两条轨迹就互相拖拽、卡在计数线两侧来回
        跳，永远不算穿越（自测 case 4 就是这个现象）。
        """
        return self.cx + self.vx, self.cy + self.vy

    @property
    def confirmed(self):
        return self.hits >= CONFIRM_HITS

    @property
    def cx(self):
        return float(self.box[0] + self.box[2] * 0.5)

    @property
    def cy(self):
        return float(self.box[1] + self.box[3] * 0.5)


def match_tracks(tracks, dets):
    """按「中心距 / 人脸尺度」贪心配对，返回 ([(轨迹下标, 检测下标)], 未匹配检测下标)"""
    cand = []
    for di, d in enumerate(dets):
        dcx, dcy = d[0] + d[2] * 0.5, d[1] + d[3] * 0.5
        for ti, t in enumerate(tracks):
            pcx, pcy = t.predict()          # 跟"预测位置"比，而不是当前位置
            dist = math.hypot(dcx - pcx, dcy - pcy)
            scale = max(40.0, 0.5 * (t.box[2] + d[2]))
            cand.append((dist / scale, ti, di))
    cand.sort(key=lambda c: c[0])
    used_t, used_d, pairs = set(), set(), []
    for cost, ti, di in cand:
        if cost > MATCH_COST:
            break                       # 已按代价升序，后面的只会更差
        if ti in used_t or di in used_d:
            continue
        pairs.append((ti, di))
        used_t.add(ti)
        used_d.add(di)
    return pairs, [i for i in range(len(dets)) if i not in used_d]


def update_side(t, line_x):
    """越线判定：返回 +1 从左到右、-1 从右到左、0 没越线。

    滞后带的意义：人站在线上时框中心会在 line_x 上下抖，没有滞后带就会
    反复"进进出出"，计数直接翻倍。
    """
    if t.cx < line_x - LINE_DEAD:
        s = -1
    elif t.cx > line_x + LINE_DEAD:
        s = 1
    else:
        return 0                        # 落在滞后带里，归属不变
    if t.side == 0:
        t.side = s                      # 首次确定归属，不算越线
        return 0
    if s == t.side:
        return 0
    t.side = s
    return s


def step_tracks(tracks, dets, next_id):
    """一帧的轨迹更新。返回 (tracks, next_id, 新增轨迹数)"""
    pairs, unmatched = match_tracks(tracks, dets)
    for ti, di in pairs:
        t = tracks[ti]
        old_cx, old_cy = t.cx, t.cy
        t.box = dets[di]
        # 速度做指数平滑：单帧跳变不至于把预测带飞
        t.vx = 0.6 * t.vx + 0.4 * (t.cx - old_cx)
        t.vy = 0.6 * t.vy + 0.4 * (t.cy - old_cy)
        t.hits += 1
        t.miss = 0
    for ti, t in enumerate(tracks):
        if all(ti != p[0] for p in pairs):
            t.miss += 1
    new = 0
    for di in unmatched:
        tracks.append(Track(next_id, dets[di]))
        next_id += 1
        new += 1
    return [t for t in tracks if t.miss <= MAX_MISS], next_id, new


# ----------------------------------------------------------------- 绘制 ---
def tid_color(tid):
    return PALETTE[tid % len(PALETTE)]


def draw_track(canvas, t, mirror, w_img):
    col = tid_color(t.tid)
    x, y, bw, bh = t.box
    if mirror:
        x = w_img - x - bw
    p1 = (int(x), int(y))
    p2 = (int(x + bw), int(y + bh))
    if t.confirmed:
        cv2.rectangle(canvas, p1, p2, col, 2)
        cv2.circle(canvas, (int(x + bw / 2), int(y + bh / 2)), 3, col, -1)
    else:
        # 未确认：细框，避免误检闪一下就当成"人"
        cv2.rectangle(canvas, p1, p2, (120, 120, 120), 1)

    dist = f"{t.dist:.2f}m" if t.dist else "--"
    label = f"#{t.tid} {dist}"
    ly = max(int(y) - 8, 16)
    cv2.putText(canvas, label, (int(x), ly), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, label, (int(x), ly), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, col, 2, cv2.LINE_AA)


def draw_line(canvas, line_x):
    h = canvas.shape[0]
    # 黑描边 + 橙线：白墙/白纸背景上才看得清
    cv2.line(canvas, (line_x, 0), (line_x, h), (0, 0, 0), 5, cv2.LINE_AA)
    cv2.line(canvas, (line_x, 0), (line_x, h), (60, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "IN >", (line_x + 10, h - 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, "IN >", (line_x + 10, h - 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (60, 200, 255), 2, cv2.LINE_AA)


def draw_hud(canvas, count, n_in, n_out, peak, fps, thresh, n_hidden):
    big = f"PEOPLE {count}"
    cv2.putText(canvas, big, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (0, 0, 0), 7, cv2.LINE_AA)
    cv2.putText(canvas, big, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (120, 235, 150), 3, cv2.LINE_AA)

    l2 = f"IN {n_in}   OUT {n_out}   PEAK {peak}   fps {fps:4.1f}   thr {thresh:.2f}"
    if n_hidden:
        l2 += f"   (-{n_hidden} 未确认)"
    cv2.putText(canvas, l2, (22, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, l2, (22, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (215, 215, 215), 1, cv2.LINE_AA)

    hint = "m=mirror  l=line  [/]=thresh  s=save  q/close=quit"
    h, w = canvas.shape[:2]
    cv2.putText(canvas, hint, (w - 16 - 9 * len(hint), 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (170, 170, 170), 1, cv2.LINE_AA)


def face_depth(zmap, box, scale):
    """人脸框中心 40% 区域的中位深度(米)。zmap 是彩色图 scale 分之一分辨率"""
    x, y, bw, bh = box
    x0 = int((x + bw * 0.3) / scale)
    x1 = int((x + bw * 0.7) / scale)
    y0 = int((y + bh * 0.3) / scale)
    y1 = int((y + bh * 0.7) / scale)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = zmap[max(y0, 0):y1, max(x0, 0):x1]
    v = roi[roi > 0]
    return float(np.median(v)) if v.size else 0.0


def detect_faces(det, img, scale=1.0):
    """返回 Nx4 人脸框（原图坐标）。scale<1 时把图缩小检测再放大坐标"""
    if scale != 1.0:
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = img
    h, w = small.shape[:2]
    det.setInputSize((w, h))
    _n, faces = det.detect(small)
    out = []
    if faces is None:
        return out
    inv = 1.0 / scale
    for f in faces:
        bw, bh = f[2] * inv, f[3] * inv
        if bw < MIN_BOX or bh < MIN_BOX:
            continue
        out.append(np.array([f[0] * inv, f[1] * inv, bw, bh], np.float32))
    return out


# ------------------------------------------------------------------ 主流程 ---
def run_image(path, args):
    """单张图片：检测 + 画框，不碰相机（方便验证算法与界面）"""
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"读不到图片：{path}")
    det = cv2.FaceDetectorYN.create(MODEL, "", (320, 320), args.thresh, 0.3, 5000)
    dets = detect_faces(det, img)
    canvas = img.copy()
    for i, d in enumerate(dets):
        t = Track(i + 1, d)
        t.hits = CONFIRM_HITS
        draw_track(canvas, t, args.mirror, img.shape[1])
    draw_hud(canvas, len(dets), 0, 0, len(dets), 0.0, args.thresh, 0)
    out = args.shot or os.path.join(OUT, "people_image.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cp.imwrite_u(out, canvas)
    print(f"{path}: 检出 {len(dets)} 张脸 -> {out}")
    for i, d in enumerate(dets):
        print(f"  #{i + 1} box={[int(v) for v in d]}")


def main():
    ap = argparse.ArgumentParser(description="人数统计 + 人脸追踪")
    ap.add_argument("--fast", action="store_true", help="960x540 检测，省一半 CPU")
    ap.add_argument("--line", type=float, default=0.5, help="计数线位置（画面宽度比例）")
    ap.add_argument("--thresh", type=float, default=0.6, help="检测置信度阈值")
    ap.add_argument("--mirror", action="store_true", help="画面镜像显示")
    ap.add_argument("--shot", default="", help="抓一帧出图后退出")
    ap.add_argument("--image", default="", help="单张图片检测（不占相机）")
    args = ap.parse_args()

    if args.image:
        run_image(args.image, args)
        return

    det = cv2.FaceDetectorYN.create(MODEL, "", (320, 320), args.thresh, 0.3, 5000)
    dscale = 0.75 if args.fast else 1.0

    depth = cp.DepthSource()
    depth.start()
    color = cp.ColorSourceFFmpeg(1280, 720, 30, device=cp.FFMPEG_DEVICE)
    color.start()
    cal = load_calib(latest_calib())
    cal, dsz, csz = adapt_cal(cal, depth.mode, color.mode)
    align = Aligner(cal, dsz, csz)

    state = dict(tracks=[], next_id=1, n_in=0, n_out=0, peak=0,
                 mirror=args.mirror, show_line=True, thresh=args.thresh,
                 fps=0.0)
    win = "People  [m=mirror  l=line  []=thresh  s=save  q=quit]"
    if not args.shot:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    print("人数统计已启动：q 退出。计数线可 --line 0.5 调整", flush=True)

    zmap = None
    t_prev = time.time()
    last_color_ts = None
    frame_i = 0
    try:
        while True:
            cz = color.latest()
            if cz is None:
                time.sleep(0.005)
                continue
            cts = cz[1] if len(cz) > 1 else None
            if cts is not None and cts == last_color_ts:
                time.sleep(0.004)          # 同一帧不重复检测（会虚增 fps）
                continue
            last_color_ts = cts
            img = cp.ColorSourceFFmpeg.decode(cz[0]) if getattr(color, "is_jpeg", False) else cz[0]
            h, w = img.shape[:2]
            line_x = int(w * args.line)

            # 1) 检测 + 追踪
            dets = detect_faces(det, img, dscale)
            tracks, next_id, _new = step_tracks(state["tracks"], dets,
                                                state["next_id"])
            state["tracks"], state["next_id"] = tracks, next_id

            # 2) 深度（隔帧刷新：warp 22ms，摊薄到 11ms/帧）
            frame_i += 1
            dz = depth.latest()
            if dz is not None and frame_i % 2 == 1:
                zmap, _proj = align.warp(cv2.flip(dz[0], 1))
            if zmap is not None:
                s = align.cw / float(zmap.shape[1])
                for t in tracks:
                    t.dist = face_depth(zmap, t.box, s)

            # 3) 越线计数（只算已确认的人）
            for t in tracks:
                if not t.confirmed:
                    continue
                crossed = update_side(t, line_x)
                if crossed > 0:
                    state["n_in"] += 1
                elif crossed < 0:
                    state["n_out"] += 1
            count = sum(1 for t in tracks if t.confirmed)
            state["peak"] = max(state["peak"], count)

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                state["fps"] = 0.9 * state["fps"] + 0.1 / dt if state["fps"] else 1.0 / dt

            # 4) 画
            canvas = cv2.flip(img, 1) if state["mirror"] else img.copy()
            if state["show_line"]:
                draw_line(canvas, w - line_x if state["mirror"] else line_x)
            for t in tracks:
                draw_track(canvas, t, state["mirror"], w)
            n_hidden = sum(1 for t in tracks if not t.confirmed)
            draw_hud(canvas, count, state["n_in"], state["n_out"], state["peak"],
                     state["fps"], state["thresh"], n_hidden)

            if args.shot:
                cp.imwrite_u(args.shot, canvas)
                print(f"人数 {count}  IN {state['n_in']}  OUT {state['n_out']}  "
                      f"未确认 {n_hidden}  轨迹 {len(tracks)}")
                for t in tracks:
                    print(f"  #{t.tid} box={[int(v) for v in t.box]} "
                          f"hits={t.hits} dist={t.dist:.2f}m")
                print("存", args.shot)
                break

            cv2.imshow(win, canvas)
            k = wait_key(win, 1)      # 点窗口 X 也能退出
            if k in (27, ord("q")):
                break
            if k == ord("m"):
                state["mirror"] = not state["mirror"]
            elif k == ord("l"):
                state["show_line"] = not state["show_line"]
            elif k == ord("["):
                state["thresh"] = max(0.3, state["thresh"] - 0.05)
                det = cv2.FaceDetectorYN.create(MODEL, "", (320, 320),
                                                state["thresh"], 0.3, 5000)
                print(f"检测阈值 {state['thresh']:.2f}")
            elif k == ord("]"):
                state["thresh"] = min(0.95, state["thresh"] + 0.05)
                det = cv2.FaceDetectorYN.create(MODEL, "", (320, 320),
                                                state["thresh"], 0.3, 5000)
                print(f"检测阈值 {state['thresh']:.2f}")
            elif k == ord("s"):
                os.makedirs(OUT, exist_ok=True)
                fn = os.path.join(OUT, time.strftime("people_%H%M%S.png"))
                cp.imwrite_u(fn, canvas)
                print("saved", fn)
    finally:
        cv2.destroyAllWindows()
        color.close()
        depth.close()


if __name__ == "__main__":
    main()
