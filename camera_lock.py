"""camera_lock.py —— 相机互斥锁：同一台相机同一时刻只允许一个进程占用

为什么需要它：Astra 只有一台。两个进程同时打开 OpenNI2 时，**不会报错**，
而是深度流大面积失效——2026-11-16 就踩过一次：界面上 HUD 显示 depth 只有
3.4%、画面大片黑洞，排查了半天以为是传感器坏了/标定错了，最后发现是
.venv 版和系统 Python 版两个 app_align 同时在跑。

所以把这件事变成硬约束：进程打开深度设备前先抢锁，抢不到就明确报错退出，
而不是悄悄跑出一堆假数据。

实现要点：
  * 锁文件 `.camera.lock`（JSON：pid / name / 启动时间），进程退出自动清理
    （atexit），被 kill -9 也会残留——所以判活看 PID 而不是看文件在不在。
  * Windows 上不能用 os.kill(pid, 0) 判活：它会真的去终止进程。改用
    OpenProcess + GetExitCodeProcess。
  * 拿不到锁时抛 CameraBusy，消息里带占用者名字和 PID，直接能定位。
"""
import atexit
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK_PATH = os.path.join(HERE, ".camera.lock")
_STILL_ACTIVE = 259


class CameraBusy(RuntimeError):
    """相机已被别的进程占用"""


def pid_alive(pid):
    """进程是否还活着（跨平台，Windows 走 Win32 API）"""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, int(pid))     # QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == _STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def peek():
    """读锁文件 → dict(pid, name, ts) 或 None。持有者已死也算 None"""
    try:
        with open(LOCK_PATH, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or not pid_alive(int(d.get("pid", 0))):
        return None
    return d


class CameraLock:
    """进程内相机锁。用 with 语句或 acquire/release 手动配对"""

    def __init__(self, name=None):
        self.name = name or os.path.basename(sys.argv[0] or "python")
        self.pid = os.getpid()
        self.held = False
        self._cleanup_registered = False

    def acquire(self, wait=0.0, poll=0.15):
        """抢锁。wait>0 时最多等这么久（秒），否则立刻抛 CameraBusy"""
        t_end = time.time() + max(0.0, wait)
        while True:
            cur = peek()
            if cur is None:
                self._write()
                return self
            if int(cur.get("pid", -1)) == self.pid:
                self.held = True                    # 同一个进程重复 acquire
                return self
            if time.time() >= t_end:
                raise CameraBusy(
                    f"相机正被 {cur.get('name', '?')}（PID {cur.get('pid')}）占用。\n"
                    f"  先关掉那个窗口，或执行：Stop-Process -Id {cur.get('pid')} -Force")
            time.sleep(poll)

    def release(self):
        if not self.held:
            return
        self.held = False
        try:
            cur = peek()
            if cur and int(cur.get("pid", -1)) == self.pid:
                os.remove(LOCK_PATH)
        except OSError:
            pass

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_exc):
        self.release()

    # ------------------------------------------------------------ 内部 ---
    def _write(self):
        try:
            with open(LOCK_PATH, "w", encoding="utf-8") as f:
                json.dump({"pid": self.pid, "name": self.name,
                           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
        except OSError:
            # 磁盘只读之类的极端情况：不阻塞主流程，退化成"无锁"
            self.held = True
            return
        self.held = True
        if not self._cleanup_registered:
            atexit.register(self.release)   # 进程正常退出/异常退出都会清
            self._cleanup_registered = True


def describe():
    """给人看的一句话状态，launcher 用它显示"""
    cur = peek()
    if cur is None:
        return "相机空闲"
    return f"相机被 {cur['name']} 占用（PID {cur['pid']}，{cur.get('ts', '')}）"


if __name__ == "__main__":
    print(describe())
