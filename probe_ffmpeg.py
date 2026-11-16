"""验证 ffmpeg 管道直取 MJPG 裸流的可达帧率，并与深度流做同步统计。

思路：OpenCV 5.0 的 MJPG 解码吞吐只有 ~6.6 MP/s（1080p 仅 3.2fps），
而 ffmpeg `-c:v copy` 不解码，能拿到相机原始 30fps 数据。
采集中直接以 .jpg 落盘（相机本来就输出 MJPG，省掉一次解码+编码）。
"""
import argparse
import collections
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capture_pair as cp
import cv2

FFMPEG = r"C:\Users\Lenovo\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
DEVICE = "USB Camera"


class FfColor(threading.Thread):
    def __init__(self, w, h, fps, buf=40):
        super().__init__(daemon=True)
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-f", "dshow", "-rtbufsize", "256M",
            "-video_size", f"{w}x{h}", "-framerate", str(fps),
            "-i", f"video={DEVICE}",
            "-c:v", "copy", "-f", "mjpeg", "pipe:1",
        ]
        self.logpath = os.path.join(__import__("tempfile").gettempdir(), "ffcolor.log")
        self.logf = open(self.logpath, "w", encoding="utf-8", errors="ignore")
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                  stderr=self.logf, bufsize=1 << 20)
        self.buf = collections.deque(maxlen=buf)
        self.lock = threading.Lock()
        self.count = 0
        self._running = True
        self.error = None

    def run(self):
        acc = bytearray()
        fd = self.p.stdout.fileno()
        while self._running:
            # 直接 os.read 绕过 BufferedReader，减少一次拷贝和缓冲层
            try:
                chunk = os.read(fd, 1 << 18)
            except OSError as e:
                self.error = f"读管道失败: {e}"
                break
            if not chunk:
                self.error = "ffmpeg 管道关闭"
                break
            acc += chunk
            start = 0
            while True:
                s = acc.find(b"\xff\xd8", start)
                if s < 0:
                    start = len(acc) - 1 if len(acc) else 0
                    if start < 0:
                        start = 0
                    break
                e = acc.find(b"\xff\xd9", s + 2)
                if e < 0:
                    start = s
                    break
                jpg = bytes(acc[s:e + 2])
                start = e + 2
                with self.lock:
                    self.buf.append((jpg, time.time()))
                    self.count += 1
            if start:
                del acc[:start]

    def latest(self):
        with self.lock:
            return self.buf[-1] if self.buf else None

    def close(self):
        self._running = False
        try:
            self.p.kill()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--color-only", action="store_true")
    args = ap.parse_args()

    print("=" * 62)
    print(f" ffmpeg 管道采集  {args.width}x{args.height}@{args.fps}  {args.secs:.0f}s")
    print("=" * 62)

    depth = None
    color = None
    try:
        if not args.color_only:
            depth = cp.DepthSource(use_ir=False)
            w, h, f, fmt = depth.mode
            print(f"  深度  {depth.name}  {w}x{h}@{f}")
            depth.start()

        color = FfColor(args.width, args.height, args.fps)
        color.start()
        t0 = time.time()
        while color.latest() is None and time.time() - t0 < 8:
            time.sleep(0.05)
        if color.latest() is None:
            print("  !! 8 秒没有彩色帧，管线可能没起来")
            return 1
        first = color.latest()[0]
        probe = cv2.imdecode(__import__("numpy").frombuffer(first, "uint8"), cv2.IMREAD_COLOR)
        if probe is None:
            print("  !! 收到数据但不是有效 JPEG")
            return 1
        print(f"  彩色  首帧 {probe.shape[1]}x{probe.shape[0]}  {len(first)/1024:.0f} KB/帧")

        time.sleep(1.0)
        c0 = color.count
        d0 = depth.count if depth else 0
        sync = []
        sizes = []
        t0 = time.time()
        while time.time() - t0 < args.secs:
            c = color.latest()
            d = depth.latest() if depth else None
            if c and d:
                sync.append(abs(c[1] - d[1]) * 1000.0)
            time.sleep(0.02)
        dt = time.time() - t0

        cfps = (color.count - c0) / dt
        dfps = (depth.count - d0) / dt if depth else 0.0

        print("-" * 62)
        if depth:
            print(f"  深度  {depth.count - d0:5d} 帧  =  {dfps:5.1f} fps")
        print(f"  彩色  {color.count - c0:5d} 帧  =  {cfps:5.1f} fps")
        if sync:
            s = sorted(sync)
            print(f"  同步差 均值 {sum(sync)/len(sync):.1f} ms  中位 {s[len(s)//2]:.1f} ms"
                  f"  P95 {s[int(len(s)*0.95)-1]:.1f} ms  最大 {max(s):.1f} ms")
        print("-" * 62)
        ok = cfps >= args.fps * 0.9 and (not depth or dfps >= 29)
        print("  判定：" + ("通过" if ok else "不达标"))
        return 0 if ok else 2
    finally:
        if color:
            color.close()
        if depth:
            try:
                depth.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
