# -*- coding: utf-8 -*-
"""
Orbbec Astra Pro Plus 深度 + 彩色双流同步采集
------------------------------------------------------------------
深度：OpenNI2（ctypes 直接调用，免 GUI / 免编译）
彩色：ffmpeg 管道直取 MJPG 裸流（默认，可跑满 30fps）
      OpenCV + DirectShow（备用，注意 OpenCV 5.0 的 MJPG 解码很慢）

用法:
    python capture_pair.py                              # 预览窗口；空格/回车抓帧，a 自动，q 退出
    python capture_pair.py --auto 2 --frames 20          # 每 2 秒抓一组，共 20 组
    python capture_pair.py --no-window --auto 0.5 --frames 10
    python capture_pair.py --width 1920 --height 1080    # 彩色用全分辨率
    python capture_pair.py --out D:\\data\\calib_01
    python capture_pair.py --color-backend dshow         # 回退到 OpenCV 取流

输出（<out> 目录）:
    color_000001.jpg        彩色原图（ffmpeg 后端，MJPG 原始流直存，零损失）
    color_000001.png        彩色原图（dshow 后端）
    depth_000001.png        深度原图（16bit，单位 mm，0=无效）
    depth_000001_vis.png    深度伪彩（仅供肉眼检查）
    captures.csv            时间戳 / 同步误差 / 深度统计
    device.txt              设备与流参数
"""
import argparse, collections, csv, ctypes, glob, os, subprocess, sys, threading, time
from ctypes import (c_void_p, c_int, c_char_p, c_uint16, c_uint64,
                    POINTER, byref, Structure)

import numpy as np
import cv2

# --------------------------------------------------------------- OpenNI2 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_oni_dir():
    """定位 OpenNI2 运行时目录（含 OpenNI2.dll）。

    换机器调试时常常没有 _driver 那一坨几百 MB 的原始 SDK 包，所以这里
    按优先级依次找，找到第一个含 OpenNI2.dll 的目录就用它：
      1) 环境变量 OPENNI2_LIB / OPENNI2_REDIST（官方安装包会设置后者）
      2) 本项目 _driver/openni2/*/Win64-Release/sdk/libs（随仓库带的精简运行时）
      3) 项目内 third_party/openni2
      4) 系统安装目录 Program Files[ (x86)]/OpenNI2/Redist
    都找不到就返回 None，由 _load_oni() 抛出带操作指引的错误，
    不会在 import 阶段就崩掉整个程序。
    """
    cands = []
    for env in ("OPENNI2_LIB", "OPENNI2_REDIST"):
        v = os.environ.get(env)
        if v:
            cands.append(v)
    cands += sorted(glob.glob(os.path.join(
        BASE_DIR, "_driver", "openni2", "*", "Win64-Release", "sdk", "libs")))
    cands.append(os.path.join(BASE_DIR, "third_party", "openni2"))
    for pf in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
        if pf:
            cands.append(os.path.join(pf, "OpenNI2", "Redist"))
            cands.append(os.path.join(pf, "OpenNI2"))
    for d in cands:
        if d and os.path.isfile(os.path.join(d, "OpenNI2.dll")):
            return d
    return None


ONI_LIB_DIR = _find_oni_dir()
if ONI_LIB_DIR:
    try:
        os.add_dll_directory(ONI_LIB_DIR)
    except (AttributeError, OSError):
        pass          # 非 Windows 或目录异常时忽略，真正加载时会报明确错误

ONI_API_VERSION = 2003
ONI_SENSOR_DEPTH = 3
ONI_SENSOR_IR = 1
ONI_STREAM_PROPERTY_VIDEO_MODE = 3

PIXFMT = {100: "DEPTH_1_MM", 101: "DEPTH_100_UM", 102: "DEPTH_1_UM",
          202: "GRAY8", 203: "GRAY16", 204: "RGB888"}


class OniVideoMode(Structure):
    _fields_ = [("pixelFormat", c_int), ("resolutionX", c_int),
                ("resolutionY", c_int), ("fps", c_int)]


class OniFrame(Structure):
    _fields_ = [("dataSize", c_int), ("data", c_void_p), ("sensorType", c_int),
                ("timestamp", c_uint64), ("frameIndex", c_int), ("width", c_int),
                ("height", c_int), ("videoMode", OniVideoMode),
                ("croppingEnabled", c_int), ("cropOriginX", c_int),
                ("cropOriginY", c_int), ("stride", c_int)]


class OniDeviceInfo(Structure):
    _fields_ = [("uri", ctypes.c_char * 256), ("vendor", ctypes.c_char * 256),
                ("name", ctypes.c_char * 256), ("serialNumber", ctypes.c_char * 256),
                ("usbVendorId", c_uint16), ("usbProductId", c_uint16)]


def _load_oni():
    if not ONI_LIB_DIR:
        raise RuntimeError(
            "找不到 OpenNI2 运行时（OpenNI2.dll）。请任选一种方式：\n"
            "  1) 把 OpenNI2 的 Redist 目录解压到 本项目/third_party/openni2/；\n"
            "  2) 安装官方 OpenNI2 / Orbbec 驱动（会设置 OPENNI2_REDIST 环境变量）；\n"
            "  3) 设置环境变量 OPENNI2_LIB 指向含 OpenNI2.dll 的目录。\n"
            "  下载：https://www.orbbec.com/developers/openni-sdk/")
    lib = ctypes.CDLL(os.path.join(ONI_LIB_DIR, "OpenNI2.dll"))

    def s(n, r, a):
        f = getattr(lib, n)
        f.restype, f.argtypes = r, a
        return f

    def s_opt(n, r, a):
        """可选符号：SDK 里没有该函数时返回 None，不影响其余功能"""
        try:
            return s(n, r, a)
        except AttributeError:
            return None

    return dict(
        init=s("oniInitialize", c_int, [c_int]),
        list=s("oniGetDeviceList", c_int, [POINTER(POINTER(OniDeviceInfo)), POINTER(c_int)]),
        open=s("oniDeviceOpen", c_int, [c_char_p, POINTER(c_void_p)]),
        create=s("oniDeviceCreateStream", c_int, [c_void_p, c_int, POINTER(c_void_p)]),
        start=s("oniStreamStart", c_int, [c_void_p]),
        stop=s("oniStreamStop", c_int, [c_void_p]),
        read=s("oniStreamReadFrame", c_int, [c_void_p, POINTER(POINTER(OniFrame))]),
        getprop=s("oniStreamGetProperty", c_int, [c_void_p, c_int, c_void_p, POINTER(c_int)]),
        release=s("oniFrameRelease", None, [POINTER(OniFrame)]),
        devclose=s_opt("oniDeviceClose", c_int, [c_void_p]),
        shutdown=s_opt("oniShutdown", None, []),
    )


class DepthSource(threading.Thread):
    """后台线程持续取帧，保留最近 30 帧供配对；带卡死看门狗"""

    def __init__(self, use_ir=False, buffer=30):
        super().__init__(daemon=True)
        self.api = _load_oni()
        if self.api["init"](ONI_API_VERSION) != 0:
            raise RuntimeError("oniInitialize 失败")
        plist = POINTER(OniDeviceInfo)()
        n = c_int(0)
        self.api["list"](byref(plist), byref(n))
        if n.value == 0:
            raise RuntimeError("OpenNI2 未枚举到设备（若刚异常退出，请拔插一次相机）")
        info = plist[0]
        self.uri = info.uri.decode(errors="ignore")
        self.name = info.name.decode(errors="ignore")
        self.vid, self.pid = info.usbVendorId, info.usbProductId

        self.dev = c_void_p()
        if self.api["open"](info.uri, byref(self.dev)) != 0:
            raise RuntimeError("打开设备失败")

        self.stream = c_void_p()
        stype = ONI_SENSOR_IR if use_ir else ONI_SENSOR_DEPTH
        if self.api["create"](self.dev, stype, byref(self.stream)) != 0:
            raise RuntimeError(f"创建{'IR' if use_ir else '深度'}流失败")

        vm = OniVideoMode()
        sz = c_int(ctypes.sizeof(OniVideoMode))
        self.api["getprop"](self.stream, ONI_STREAM_PROPERTY_VIDEO_MODE, byref(vm), byref(sz))
        self.mode = (vm.resolutionX, vm.resolutionY, vm.fps, vm.pixelFormat)

        self._buf = collections.deque(maxlen=buffer)
        self._lock = threading.Lock()
        self._running = False
        self.count = 0
        self.error = None
        self.last_ts = 0.0
        self.stalled = False

    def run(self):
        pf = POINTER(OniFrame)()
        while self._running:
            try:
                rc = self.api["read"](self.stream, byref(pf))
            except OSError:
                # oniStreamStop 期间 read 可能落在已释放的对象上（访问冲突）。
                # 这是停流的正常副作用，线程安静退出即可。
                break
            if rc != 0 or not pf:
                self.error = f"读帧失败 rc={rc}"
                break
            f = pf.contents
            arr = np.frombuffer(ctypes.string_at(f.data, f.dataSize),
                                dtype=np.uint16).reshape(f.height, f.width).copy()
            ts = time.time()
            with self._lock:
                self._buf.append((arr, ts, int(f.frameIndex), int(f.timestamp)))
                self.last_ts = ts
            self.count += 1
            self.api["release"](pf)

    def start(self):
        if self.api["start"](self.stream) != 0:
            raise RuntimeError("深度流启动失败")
        self._running = True
        super().start()
        t0 = time.time()
        while self.latest() is None and time.time() - t0 < 6:
            time.sleep(0.02)
        if self.latest() is None:
            # 必须显式 stop，否则 OpenNI 的读帧线程会留在内核态，后续调用永久死锁
            self._running = False
            try:
                self.api["stop"](self.stream)
            except Exception:
                pass
            self._join_reader()
            raise RuntimeError("6 秒内没有收到深度帧 —— 相机可能死锁，请拔插一次 USB")

    def latest(self):
        with self._lock:
            return self._buf[-1] if self._buf else None

    def snapshot(self):
        with self._lock:
            return list(self._buf)

    def healthy(self, timeout=4.0):
        return self.last_ts > 0 and (time.time() - self.last_ts) < timeout

    def _join_reader(self, timeout=2.0):
        """等读帧线程真正退出。返回 True 表示已退出。

        oniStreamStop 只是唤醒阻塞中的 read，线程可能还没跑完这一轮。
        若此时就 oniDeviceClose / oniShutdown，读线程会踩已释放的对象，
        在内核里抛 access violation（表现为线程里 OSError 刷屏，
        严重时导致下次 oniInitialize 失败）。
        """
        if not self.is_alive() or threading.current_thread() is self:
            return True
        self.join(timeout)
        return not self.is_alive()

    def close(self):
        """停流。顺序很关键。

        先让读帧线程自然退出（它读到下一帧、发现 _running 为假就跳出循环），
        再调 oniStreamStop。反过来先 stop 的话，阻塞中的 oniStreamReadFrame
        永远等不到数据，线程会永久卡在驱动层（实测 join 2 秒也不返回，
        之后连 oniDeviceClose 都不敢做，只能泄漏）。
        """
        self._running = False
        self._join_reader(timeout=1.5)
        try:
            self.api["stop"](self.stream)
        except Exception:
            pass

    def release(self):
        """彻底释放：停流 -> 关设备 -> 卸载 OpenNI。
        只给需要反复开关相机的场景（GUI、标定切换 IR/深度）用；
        命令行脚本 close() 就够了。
        不调用的话第二次 oniDeviceOpen 会因设备仍被占用而失败。"""
        if getattr(self, "_released", False):
            return
        self._released = True
        self.close()
        # 线程若仍卡在 read 里，宁可不关设备（会泄漏），也不要在它脚下拆对象
        if not self._join_reader(timeout=0.1):
            self.error = self.error or "读帧线程未能退出，已跳过设备关闭"
            return
        try:
            f = self.api.get("devclose")
            if f:
                f(self.dev)
        except Exception:
            pass
        try:
            f = self.api.get("shutdown")
            if f:
                f()
        except Exception:
            pass


def _fourcc_str(v):
    """CAP_PROP_FOURCC 返回的整数转成 'MJPG' / 'YUY2' 这样的可读串"""
    v = int(v)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


class ColorSource(threading.Thread):
    def __init__(self, index, width, height, fps):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开彩色相机 index={index}")
        # 注意：必须先设分辨率再设 FOURCC，之后还要复验。
        # 若 FOURCC 没生效会退化成 YUY2（该相机 1080p YUY2 仅 3fps），
        # 表现为"能出图但帧率极低/严重卡顿"。
        target = cv2.VideoWriter_fourcc(*"MJPG")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FOURCC, target)
        for _ in range(4):
            if _fourcc_str(self.cap.get(cv2.CAP_PROP_FOURCC)) == "MJPG":
                break
            self.cap.set(cv2.CAP_PROP_FOURCC, target)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FOURCC, target)
        self.fourcc = _fourcc_str(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.mode = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     int(self.cap.get(cv2.CAP_PROP_FPS)))
        self._latest = None
        self._lock = threading.Lock()
        self._running = False
        self.count = 0
        self.error = None
        self.last_ts = 0.0

    def run(self):
        fails = 0
        while self._running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                fails += 1
                self.error = "彩色读帧失败"
                if fails > 200:
                    break
                time.sleep(0.01)
                continue
            fails = 0
            ts = time.time()
            with self._lock:
                self._latest = (frame, ts)
                self.last_ts = ts
            self.count += 1

    def start(self):
        self._running = True
        super().start()
        t0 = time.time()
        while self.latest() is None and time.time() - t0 < 6:
            time.sleep(0.02)
        if self.latest() is None:
            raise RuntimeError("6 秒内没有收到彩色帧")

    def latest(self):
        with self._lock:
            return self._latest

    def healthy(self, timeout=2.0):
        return self.last_ts > 0 and (time.time() - self.last_ts) < timeout

    def close(self):
        self._running = False
        try:
            self.cap.release()
        except Exception:
            pass


# ------------------------------------------------- 彩色：ffmpeg 管道取流 ---
FFMPEG_EXE = None
FFMPEG_DEVICE = "USB Camera"


def find_ffmpeg():
    """定位 ffmpeg.exe：先查 PATH，再查常见安装位置"""
    global FFMPEG_EXE
    if FFMPEG_EXE is not None:
        return FFMPEG_EXE
    import shutil as _sh
    cand = _sh.which("ffmpeg")
    if not cand:
        for p in (r"C:\Users\Lenovo\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe",):
            if os.path.exists(p):
                cand = p
                break
    FFMPEG_EXE = cand
    return cand


class ColorSourceFFmpeg(threading.Thread):
    """用 ffmpeg `-c:v copy` 直取 UVC 的 MJPG 裸流，Python 侧完全不解码。

    背景：OpenCV 5.0 的 MJPG 解码吞吐只有约 6.6 MP/s
    （1080p 3.2fps / 720p 7.1fps），而相机本身能稳定输出 30fps。
    本类把解码彻底去掉，抓到 JPEG 原始字节直接入队，
    落盘就是 .jpg，省掉"解码 + 重新编码"两次开销。
    实测 720p 双流可稳定 30fps，同步误差中位 6ms。
    """
    is_jpeg = True

    def __init__(self, width, height, fps, device=None, buffer=45):
        super().__init__(daemon=True)
        exe = find_ffmpeg()
        if not exe:
            raise RuntimeError("找不到 ffmpeg.exe，请安装或用 --color-backend dshow")
        self.device = device or FFMPEG_DEVICE
        cmd = [exe, "-hide_banner", "-loglevel", "error",
               "-f", "dshow", "-rtbufsize", "256M",
               "-video_size", f"{width}x{height}", "-framerate", str(fps),
               "-i", f"video={self.device}",
               "-c:v", "copy", "-f", "mjpeg", "pipe:1"]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL,
                                  stdin=subprocess.DEVNULL, bufsize=0)
        self._buf = collections.deque(maxlen=buffer)
        self._lock = threading.Lock()
        self._running = True
        self.count = 0
        self.error = None
        self.last_ts = 0.0
        self.fourcc = "MJPG"
        self.mode = (width, height, fps)

    def run(self):
        acc = bytearray()
        fd = self.p.stdout.fileno()
        while self._running:
            try:
                chunk = os.read(fd, 1 << 18)
            except OSError as e:
                if self._running:
                    self.error = f"读管道失败: {e}"
                break
            if not chunk:
                if self._running:
                    self.error = "ffmpeg 管道关闭"
                break
            acc += chunk
            start = 0
            while True:
                s = acc.find(b"\xff\xd8", start)
                if s < 0:
                    start = max(0, len(acc) - 1)
                    break
                e = acc.find(b"\xff\xd9", s + 2)
                if e < 0:
                    start = s
                    break
                jpg = bytes(acc[s:e + 2])
                start = e + 2
                ts = time.time()
                with self._lock:
                    self._buf.append((jpg, ts))
                    self.last_ts = ts
                self.count += 1
            if start:
                del acc[:start]

    def start(self):
        super().start()
        t0 = time.time()
        while self.latest() is None and time.time() - t0 < 8:
            time.sleep(0.03)
        if self.latest() is None:
            self.close()
            raise RuntimeError("8 秒内没有收到彩色帧 —— ffmpeg 管道没起来")

    def latest(self):
        with self._lock:
            return self._buf[-1] if self._buf else None

    def snapshot(self):
        with self._lock:
            return list(self._buf)

    def healthy(self, timeout=2.0):
        return self.last_ts > 0 and (time.time() - self.last_ts) < timeout

    @staticmethod
    def decode(jpg_buf):
        return cv2.imdecode(np.frombuffer(jpg_buf, dtype=np.uint8), cv2.IMREAD_COLOR)

    def close(self):
        self._running = False
        try:
            self.p.kill()
        except Exception:
            pass


def find_orbbec_camera(max_index=8, verbose=True):
    """优先返回能跑 >=1920 宽的那个 UVC 相机"""
    fallback = None
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok = cap.grab()
        cap.release()
        if verbose:
            print(f"    [扫描] index={i}  {w}x{h}  grab={ok}")
        if ok and w >= 1920:
            return i
        if ok and fallback is None:
            fallback = i
    return fallback


def set_camera_exposure(exp=-6.0, device_index=0, auto=False):
    """设置彩色相机（UVC）的曝光模式。必须在启动 ffmpeg/DShow 彩色流之前
    调用（此时相机未被占用）。设置在相机上保持，释放后仍然生效。

    背景：暗光下 UVC 的自动曝光会把曝光时间拉长到超过 1/30s，
    相机随之降到 17~20fps。锁短曝光（如 -6 即 1/64s）可恢复满帧，
    且避免标定时画面亮度漂移。代价：光照不足时画面会偏暗。

    auto=False 时锁定手动曝光 exp（DSHOW 单位 log2 秒）；
    auto=True 时恢复自动曝光。
    返回 (是否成功, 实际生效值)。"""
    last = None
    for idx in (device_index, 0, 1, 2):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        try:
            if auto:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)   # 自动
                time.sleep(1.0)
                return True, cap.get(cv2.CAP_PROP_EXPOSURE)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)       # 手动
            cap.set(cv2.CAP_PROP_EXPOSURE, exp)
            time.sleep(1.0)   # 留时间让相机应用设置
            ok = abs(cap.get(cv2.CAP_PROP_EXPOSURE) - exp) < 2.0
            return ok, cap.get(cv2.CAP_PROP_EXPOSURE)
        finally:
            cap.release()
    return False, last


def imwrite_u(path, img):
    """支持中文路径的图片写入（OpenCV 的 imwrite 在 Windows 上不支持非 ASCII 路径）"""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        print("    !! 编码失败:", path)
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def depth_to_vis(depth):
    d = depth.astype(np.float32)
    v = d[d > 0]
    if v.size == 0:
        return np.zeros((depth.shape[0], depth.shape[1], 3), np.uint8)
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    if hi <= lo:
        hi = lo + 1
    g = (np.clip((d - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    g[depth == 0] = 0
    return cv2.applyColorMap(g, cv2.COLORMAP_JET)


def pick_depth_pair(snapshot, color_ts, window=0.5):
    """在深度缓冲里挑与彩色帧时间戳最接近的一帧"""
    if not snapshot:
        return None
    best = min(snapshot, key=lambda f: abs(f[1] - color_ts))
    if abs(best[1] - color_ts) > window:
        return None
    return best


def main():
    ap = argparse.ArgumentParser(description="Orbbec Astra Pro Plus 深度+彩色同步采集")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "captures"))
    ap.add_argument("--camera", type=int, default=-1, help="彩色相机索引，-1=自动查找")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--auto", type=float, default=0.0, help="自动抓帧间隔(秒)，0=手动")
    ap.add_argument("--frames", type=int, default=0, help="抓满 N 组后退出，0=不限")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--ir", action="store_true", help="采 IR 流代替深度流")
    ap.add_argument("--start-index", type=int, default=1)
    ap.add_argument("--no-scan", action="store_true", help="跳过相机扫描，直接用 index 0")
    ap.add_argument("--color-only", action="store_true", help="只采彩色（深度不可用时自检用）")
    ap.add_argument("--color-backend", default="auto", choices=["auto", "ffmpeg", "dshow"],
                    help="彩色取流方式：ffmpeg 管道(推荐，满30fps) / dshow(OpenCV)")
    ap.add_argument("--device", default=FFMPEG_DEVICE,
                    help=f"ffmpeg 后端使用的 DirectShow 设备名，默认 '{FFMPEG_DEVICE}'")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("=" * 66)
    print(" Orbbec Astra Pro Plus 双流同步采集")
    print("=" * 66)

    depth = None
    if args.color_only:
        print("[1/4] --color-only：跳过深度流")
        w = h = f = fmt = None
    else:
        print("[1/4] 启动深度流 ...")
        depth = DepthSource(use_ir=args.ir)
        w, h, f, fmt = depth.mode
        print(f"    {depth.name}  {depth.vid:04X}:{depth.pid:04X}")
        print(f"    {'IR' if args.ir else '深度'}: {w}x{h} @{f}fps  {PIXFMT.get(fmt, fmt)}")
        depth.start()

    print("[2/4] 定位彩色相机 ...")
    backend = args.color_backend
    if backend == "auto":
        backend = "ffmpeg" if find_ffmpeg() else "dshow"
    try:
        if backend == "ffmpeg":
            print(f"    后端 ffmpeg 管道  设备 '{args.device}'")
            color = ColorSourceFFmpeg(args.width, args.height, args.fps, args.device)
        else:
            idx = args.camera
            if idx < 0:
                idx = 0 if args.no_scan else find_orbbec_camera()
                if idx is None:
                    print("    !! 未找到可用彩色相机"); sys.exit(1)
                print(f"    后端 OpenCV DSHOW  选定 index={idx}")
            color = ColorSource(idx, args.width, args.height, args.fps)
        print("[3/4] 启动彩色流 ...")
        cw, ch, cf = color.mode
        print(f"    彩色: {cw}x{ch} @{cf}fps MJPG")
        color.start()
    except BaseException:
        # 彩色一路失败时必须先关深度流，否则 OpenNI 句柄泄漏 -> 下次启动死锁，
        # 只能物理拔插 USB 才能恢复。
        if depth is not None:
            try:
                depth.close()
                time.sleep(0.25)
            except Exception:
                pass
        raise

    print("[4/4] 预热 1.5 秒（等自动曝光稳定）...")
    time.sleep(1.5)

    with open(os.path.join(args.out, "device.txt"), "w", encoding="utf-8") as fp:
        if depth:
            fp.write(f"device_name={depth.name}\n")
            fp.write(f"device_uri={depth.uri}\n")
            fp.write(f"vid_pid={depth.vid:04X}:{depth.pid:04X}\n")
            fp.write(f"stream={'IR' if args.ir else 'DEPTH'} {w}x{h}@{f} {PIXFMT.get(fmt, fmt)}\n")
        else:
            fp.write("stream=COLOR ONLY\n")
        fp.write(f"color={cw}x{ch}@{cf} MJPG backend={backend}\n")
        fp.write("depth_unit=mm (uint16 little endian, 0 = invalid)\n")

    csv_path = os.path.join(args.out, "captures.csv")
    new_csv = not os.path.exists(csv_path)
    cfp = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(cfp)
    if new_csv:
        writer.writerow(["index", "color_png", "depth_png", "color_ts", "depth_ts",
                         "sync_diff_ms", "depth_valid_pct", "depth_median_mm", "depth_frame_idx"])

    win = "Orbbec Depth + Color   [SPACE]=capture  [A]=auto  [Q]=quit"
    show = not args.no_window
    if show:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 1280, 480)
        except Exception as e:
            print("    预览窗口不可用，转无界面:", e)
            show = False

    state = {"idx": args.start_index, "saved": 0}

    def write_color(idx_name, payload):
        """ffmpeg 后端存原始 .jpg 字节；dshow 后端走 imwrite"""
        if getattr(color, "is_jpeg", False):
            with open(os.path.join(args.out, idx_name), "wb") as fp:
                fp.write(payload)
        else:
            imwrite_u(os.path.join(args.out, idx_name), payload)

    def save_pair():
        c = color.latest()
        if c is None:
            print("    !! 彩色数据未就绪"); return False
        cimg, cts = c
        i = state["idx"]
        is_jpeg = getattr(color, "is_jpeg", False)
        cname = f"color_{i:06d}." + ("jpg" if is_jpeg else "png")
        if depth is None:
            write_color(cname, cimg)
            writer.writerow([i, cname, "", f"{cts:.4f}", "", "", "", "", ""])
            cfp.flush()
            if is_jpeg:
                print(f"    [{i:06d}] 彩色已保存 {len(cimg)/1024:.0f} KB (MJPG 原始流)")
            else:
                print(f"    [{i:06d}] 彩色已保存 {cimg.shape[1]}x{cimg.shape[0]}")
            state["idx"] += 1
            state["saved"] += 1
            return True
        best = None
        deadline = time.time() + 0.4
        while time.time() < deadline:
            best = pick_depth_pair(depth.snapshot(), cts)
            if best and abs(best[1] - cts) < 0.02:
                break
            time.sleep(0.008)
        if best is None:
            print("    !! 找不到时间相近的深度帧（深度流可能已停）"); return False
        darr, dts, didx, _ddev = best
        diff = abs(cts - dts) * 1000.0

        f2, f3 = f"depth_{i:06d}.png", f"depth_{i:06d}_vis.png"
        write_color(cname, cimg)
        imwrite_u(os.path.join(args.out, f2), darr)
        imwrite_u(os.path.join(args.out, f3), depth_to_vis(darr))
        v = darr[darr > 0]
        pct = v.size / darr.size * 100
        med = float(np.median(v)) if v.size else 0.0
        writer.writerow([i, cname, f2, f"{cts:.4f}", f"{dts:.4f}",
                         f"{diff:.1f}", f"{pct:.1f}", f"{med:.0f}", didx])
        cfp.flush()
        print(f"    [{i:06d}] 保存完成  同步误差={diff:5.1f}ms  有效={pct:4.1f}%  中位={med:6.0f}mm")
        state["idx"] += 1
        state["saved"] += 1
        return True

    auto_on = args.auto > 0
    auto_next = time.time() + args.auto if auto_on else None
    last_show = 0.0
    stall_warned = False

    print("-" * 66)
    print(" 运行中 …  " + (f"自动模式，每 {args.auto}s 一组" if auto_on else "手动模式（空格抓帧）"))
    print("-" * 66)

    try:
        while True:
            now = time.time()

            if depth and not depth.healthy() and not stall_warned:
                print("    ⚠ 深度流已 4 秒没有新帧，可能卡死。请拔插相机后重新运行。")
                stall_warned = True
            if depth and depth.error:
                print("    深度错误:", depth.error); break

            if auto_on and auto_next and now >= auto_next:
                save_pair()
                auto_next = now + args.auto
                if args.frames and state["saved"] >= args.frames:
                    break

            if args.frames and not auto_on and state["saved"] >= args.frames:
                break

            if show and now - last_show > 0.25:
                # JPEG 后端解码较慢，预览降到 ~4fps，不影响采集线程
                last_show = now
                c = color.latest(); d = depth.latest() if depth else None
                if c:
                    cimg, cts = c
                    if getattr(color, "is_jpeg", False):
                        cimg = color.decode(cimg)
                    if cimg is not None and d:
                        darr, dts = d[0], d[1]
                        vis = cv2.resize(depth_to_vis(darr), (cimg.shape[1], cimg.shape[0]))
                        disp = np.hstack([cimg, vis])
                        diff = abs(cts - dts) * 1000
                        txt = (f"frame {d[2]}  sync {diff:.0f}ms  saved {state['saved']}"
                               + ("  [AUTO]" if auto_on else ""))
                        cv2.putText(disp, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        cv2.imshow(win, disp)
                    elif cimg is not None:
                        cv2.imshow(win, cimg)

            if show:
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), 27):
                    break
                if k in (ord(' '), 13, ord('s')):
                    save_pair()
                    if args.frames and state["saved"] >= args.frames:
                        break
                if k == ord('a'):
                    auto_on = not auto_on
                    auto_next = time.time() + args.auto if auto_on else None
                    print(f"    自动模式 -> {'开' if auto_on else '关'}")
            elif not auto_on:
                # 无窗口且未开自动：每秒抓一组
                if auto_next is None:
                    auto_next = now + 1.0
                if now >= auto_next:
                    save_pair()
                    auto_next = now + 1.0
            else:
                time.sleep(0.005)
    except KeyboardInterrupt:
        print("\n    用户中断")
    finally:
        cfp.close()
        if depth:
            depth.close()
        color.close()
        if show:
            cv2.destroyAllWindows()

    print("-" * 66)
    print(f" 结束：本次保存 {state['saved']} 组 -> {args.out}")
    if color.error:
        print(" 彩色错误:", color.error)


if __name__ == "__main__":
    main()
