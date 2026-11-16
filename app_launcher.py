"""app_launcher.py —— 工具箱入口：一个窗口选应用，顺带管住"谁在用相机"

为什么要有它：应用攒到 5 个之后，日常用起来有两个麻烦——
  ① 记不住哪个脚本配哪些参数；
  ② 关窗口没关干净时，残留进程占着相机，新开的应用**不会报错**，只会跑出
     一堆假数据（2026-11-16 那次"深度空洞"就是这么来的，排查了半天）。

设计要点（都是踩过的坑）：
  * 鼠标回调里**只记一个待办索引**，真正的启动放到主循环里做。回调里直接
    Popen 会阻塞 highgui 的消息泵，窗口立刻变成"点 X 关不掉、按键没反应"。
  * 子进程的 stdout/stderr 重定向到 `_detect/launch_<脚本>.log`，不让它继承
    本进程的管道——继承既会污染终端，也可能把启动卡在管道缓冲上。
  * 子进程退出码非 0 时，把日志最后一行显示到状态栏，而不是让窗口一闪而过。
  * 坐标按窗口实际显示尺寸换算：Windows 高 DPI（125%/150%）下鼠标坐标与
    图像像素不是 1:1，不换算就会"点了没反应"。
  * 窗口点 X 要能退出：cv2 的 X 不会结束 waitKey 循环，得查 WND_PROP_VISIBLE。

界面用 Pillow 渲染（cv2 的 putText 不支持中文），再转 BGR 给 cv2 显示。
只在状态变化时重绘，一次约 15ms。

用法：python app_launcher.py
操作：点卡片 或 数字键 1-6 启动；r 刷新；q / ESC / 点 X 退出
"""
import os
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import camera_lock

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "_detect")
W, H = 1120, 1000         # 9 张卡片 2 列 5 行

BG = (20, 22, 26)
CARD = (32, 35, 42)
CARD_HOVER = (46, 51, 62)
BORDER = (52, 57, 68)
TEXT = (238, 240, 244)
DIM = (150, 156, 168)
ACCENT_OK = (110, 200, 130)
ACCENT_BAD = (90, 110, 235)

APPS = [
    dict(key="1", name="对齐测距", script="app_align.py", color=(255, 168, 90),
         desc="把深度图贴到彩色视角，鼠标指哪儿量哪儿，约 26fps",
         cmd="python app_align.py"),
    dict(key="2", name="3D 点云", script="app_pointcloud.py", color=(140, 220, 255),
         desc="带 RGB 的点云，拖动旋转，可导出 PLY 给 MeshLab",
         cmd="python app_pointcloud.py"),
    dict(key="3", name="尺寸测量", script="app_measure.py", color=(150, 240, 160),
         desc="拖框框住平面，自动拟合并量长宽，1m 处误差约 1mm",
         cmd="python app_measure.py"),
    dict(key="4", name="安全距离", script="app_guard.py", color=(110, 170, 255),
         desc="视场分三区报最近距离，近了标红闪边，AGV 前方净空",
         cmd="python app_guard.py"),
    dict(key="5", name="物体计数", script="app_objects.py", color=(230, 160, 255),
         desc="按深度跳变分割物体，剔掉桌面墙面，数个数并粗估尺寸",
         cmd="python app_objects.py"),
    dict(key="6", name="深度监视", script="app_depthview.py", color=(255, 210, 110),
         desc="实时看深度：有效率、8x6 网格缺口、直方图、稳定度累积",
         cmd="python app_depthview.py"),
    dict(key="7", name="人数统计", script="app_people.py", color=(120, 235, 150),
         desc="人脸追踪带编号和距离，跨线计数进出人数",
         cmd="python app_people.py"),
    dict(key="8", name="手势交互", script="app_gesture.py", color=(90, 220, 255),
         desc="食指当光标、捏合当点击，悬停自动触发按钮",
         cmd="python app_gesture.py"),
    dict(key="9", name="室内扫描", script="app_scan.py", color=(255, 150, 130),
         desc="端着相机慢慢走一圈，拼出彩色点云 + 俯视平面图，可导出 PLY",
         cmd="python app_scan.py"),
]


# --------------------------------------------------------------- 字体 ---
def _font(size, bold=False):
    name = "msyhbd.ttc" if bold else "msyh.ttc"
    cands = [os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", name),
             os.path.expanduser("~/.fonts/" + name),
             "/usr/share/fonts/truetype/" + name,
             os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", "simhei.ttf")]
    for p in cands:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


F_TITLE = _font(30, True)
F_SUB = _font(14)
F_NAME = _font(21, True)
F_DESC = _font(13)
F_CMD = _font(12)
F_KEY = _font(17, True)
F_STATUS = _font(13)


# --------------------------------------------------------------- 布局 ---
PAD = 34
GAP = 18
TOP = 132
CARD_W = (W - PAD * 2 - GAP) // 2
CARD_H = 152


def card_rect(i):
    r, c = divmod(i, 2)
    x = PAD + c * (CARD_W + GAP)
    y = TOP + r * (CARD_H + GAP)
    return x, y, x + CARD_W, y + CARD_H


# --------------------------------------------------------------- 绘制 ---
def render(hover=-1, focus=-1, msg="", msg_ok=True, running=""):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((PAD, 34), "双目相机工具箱", font=F_TITLE, fill=TEXT)
    d.text((PAD + 2, 74), "Orbbec Astra Pro Plus · 深度 640×480 + 彩色 1280×720",
           font=F_SUB, fill=DIM)

    busy = camera_lock.peek()
    ok = busy is None
    col = ACCENT_OK if ok else ACCENT_BAD
    label = "相机空闲" if ok else "相机被占用"
    tw = d.textlength(label, font=F_STATUS)
    bx, by = W - PAD - 12 - tw, 46
    d.ellipse([bx - 18, by + 3, bx - 8, by + 13], fill=col)
    d.text((bx, by), label, font=F_STATUS, fill=col)
    if not ok:
        who = f"{busy['name']} (PID {busy['pid']})"
        d.text((W - PAD - d.textlength(who, font=F_CMD), by + 22), who,
               font=F_CMD, fill=DIM)

    for i, a in enumerate(APPS):
        x0, y0, x1, y1 = card_rect(i)
        on = (i == hover) or (i == focus)
        d.rounded_rectangle([x0, y0, x1, y1], radius=10,
                            fill=CARD_HOVER if on else CARD,
                            outline=a["color"] if on else BORDER, width=2)
        d.rounded_rectangle([x0, y0, x0 + 5, y1], radius=3, fill=a["color"])

        bxx, byy = x0 + 22, y0 + 22
        d.rounded_rectangle([bxx, byy, bxx + 30, byy + 30], radius=8,
                            fill=tuple(min(255, c + 40) for c in a["color"]))
        kw = d.textlength(a["key"], font=F_KEY)
        d.text((bxx + 15 - kw / 2, byy + 4), a["key"], font=F_KEY, fill=(24, 24, 28))

        d.text((x0 + 64, y0 + 24), a["name"], font=F_NAME, fill=TEXT)
        d.text((x0 + 64, y0 + 56), a["desc"], font=F_DESC, fill=DIM)
        d.text((x0 + 22, y1 - 30), "$ " + a["cmd"], font=F_CMD,
               fill=(120, 128, 142))

    d.line([(PAD, H - 46), (W - PAD, H - 46)], fill=BORDER, width=1)
    tip = "点卡片或按数字键启动 · r 刷新 · q 退出"
    d.text((PAD, H - 34), tip, font=F_STATUS, fill=DIM)
    if msg:
        c = ACCENT_OK if msg_ok else ACCENT_BAD
        d.text((PAD + d.textlength(tip, font=F_STATUS) + 22, H - 34), msg,
               font=F_STATUS, fill=c)

    cal = os.path.join(HERE, "calib")
    info = "无标定"
    if os.path.isdir(cal):
        subs = sorted([s for s in os.listdir(cal)
                       if os.path.exists(os.path.join(cal, s, "stereo_calib.json"))])
        if subs:
            info = "标定 " + subs[-1]
    if running:
        info = f"● {running} 运行中"
        d.text((W - PAD - d.textlength(info, font=F_CMD), H - 33), info,
               font=F_CMD, fill=ACCENT_OK)
    else:
        d.text((W - PAD - d.textlength(info, font=F_CMD), H - 33), info,
               font=F_CMD, fill=DIM)

    return np.array(img)[:, :, ::-1].copy()


# --------------------------------------------------------------- 交互 ---
def hit(x, y):
    for i in range(len(APPS)):
        x0, y0, x1, y1 = card_rect(i)
        if x0 <= x <= x1 and y0 <= y <= y1:
            return i
    return -1


def tail_line(path, n=1, maxlen=90):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return ""
        return lines[-n][:maxlen]
    except OSError:
        return ""


def launch(i, state):
    """在主循环里调用（不在鼠标回调里）。返回 (消息, 是否成功)"""
    a = APPS[i]
    busy = camera_lock.peek()
    if busy:
        return (f"相机被 {busy['name']}（PID {busy['pid']}）占用，先关掉它", False)
    path = os.path.join(HERE, a["script"])
    if not os.path.exists(path):
        return (f"找不到 {a['script']}", False)

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "launch_" + a["script"].replace(".py", "") + ".log")
    try:
        log = open(log_path, "w", encoding="utf-8", errors="replace")
        t0 = time.time()
        p = subprocess.Popen([sys.executable, a["script"]], cwd=HERE,
                             stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT)
    except Exception as e:                                   # noqa: BLE001
        return (f"启动失败：{e}", False)
    state["proc"] = dict(p=p, log=log, path=log_path, name=a["name"], t0=t0)
    return (f"正在启动 {a['name']}（{time.time() - t0:.1f}s）", True)


def stop_child(state, wait=1.5):
    """退出前结束还活着的子应用。

    不这么做的话，关掉工具箱后子应用会变成孤儿继续跑：窗口留在屏幕上，
    相机锁也一直不放，下次启动就报"相机被占用"——看起来就是"关不掉"。
    """
    pr = state.get("proc")
    if not pr:
        return ""
    p = pr["p"]
    if p.poll() is not None:
        return ""
    try:
        p.terminate()
    except Exception:                                        # noqa: BLE001
        pass
    t0 = time.time()
    while time.time() - t0 < wait and p.poll() is None:
        time.sleep(0.05)
    if p.poll() is None:
        # venv 的 python.exe 是个 redirector，只杀父进程会留下孙子继续占相机
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
    return pr["name"]


def poll_proc(state):
    """检查子进程状态，返回 (消息, 是否ok, 运行中名字)。消息为空表示不更新"""
    pr = state.get("proc")
    if not pr:
        return "", True, ""
    rc = pr["p"].poll()
    if rc is None:
        name = pr["name"]
        if time.time() - pr["t0"] < 3.0:
            return f"正在启动 {name}…", True, name
        return f"{name} 运行中", True, name
    try:
        pr["log"].close()
    except OSError:
        pass
    state["proc"] = None
    if rc == 0:
        return f"{pr['name']} 已结束", True, ""
    tail = tail_line(pr["path"])
    return (f"{pr['name']} 启动失败（退出码 {rc}）：{tail}", False, "")


def main():
    win = "Toolbox"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    state = dict(hover=-1, focus=-1, msg="", ok=True, pending=-1, proc=None)
    frame = render()
    last_msg, last_h, last_f = state["msg"], -1, -1
    dirty = True

    def to_img(x, y):
        """窗口显示坐标 → 图像像素（高 DPI / 窗口缩放时按比例换算）"""
        try:
            _x, _y, dw, dh = cv2.getWindowImageRect(win)
        except Exception:                                    # noqa: BLE001
            return x, y
        if dw > 0 and dh > 0 and (dw != W or dh != H):
            return int(x * W / dw), int(y * H / dh)
        return x, y

    def on_mouse(ev, x, y, _f, _p):
        # 回调里只记录意图，绝不在这里启动进程（会阻塞消息泵，窗口随即卡死）
        if ev == cv2.EVENT_MOUSEMOVE:
            h = hit(*to_img(x, y))
            if h != state["hover"]:
                state["hover"] = h
        elif ev == cv2.EVENT_LBUTTONDOWN:
            h = hit(*to_img(x, y))
            print(f"[click] ({x},{y}) -> 卡片 {h}", flush=True)
            if h >= 0:
                state["pending"] = h

    cv2.setMouseCallback(win, on_mouse)
    print("工具箱已启动：点卡片或按 1-6 启动应用，q 退出", flush=True)

    while True:
        # 1) 执行待启动请求（一帧只做一次，避免连点开出一串实例）
        if state["pending"] >= 0:
            i, state["pending"] = state["pending"], -1
            if state.get("proc"):
                state["msg"], state["ok"] = "上一个应用还在运行，先关掉它的窗口", False
            else:
                state["focus"] = i
                state["msg"], state["ok"] = launch(i, state)
                print("[launch]", state["msg"], flush=True)

        # 2) 跟踪子进程 + 重绘（消息或悬停状态变化时）
        msg, ok, running = poll_proc(state)
        if msg:
            state["msg"], state["ok"] = msg, ok
        if (state["msg"] != last_msg or state["hover"] != last_h
                or state["focus"] != last_f or dirty):
            last_msg, last_h, last_f = state["msg"], state["hover"], state["focus"]
            dirty = False
            frame = render(state["hover"], state["focus"], state["msg"],
                           state["ok"], running)
        cv2.imshow(win, frame)

        k = cv2.waitKey(30) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("r"):
            state["msg"], state["ok"] = "状态已刷新", True
            state["focus"] = -1
        if ord("1") <= k <= ord("9"):
            state["pending"] = k - ord("1")

        # 4) 点窗口 X 也要能退出：cv2 的 X 不会结束 waitKey 循环
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    stopped = stop_child(state)
    if stopped:
        print(f"已一并结束 {stopped}", flush=True)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
