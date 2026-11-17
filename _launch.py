"""_launch.py —— 统一启动入口（双击 start.bat，或 python _launch.py [命令]）

默认行为：**直接进主界面**（`app_launcher.py`，十张功能卡片），跟双击一个
exe 的体验一致——多数时候人就是想看看有哪些功能，不该先答一道菜单题。

要精确启动某个东西时给参数（也可以给 start.bat 传，方便做桌面快捷方式）：
    python _launch.py             # 主界面（app_launcher.py）
    python _launch.py menu        # 旧的交互菜单（扫描/语音/自检…）
    python _launch.py scan        # 室内扫描
    python _launch.py tsdf        # 扫描 + TSDF 三维重建
    python _launch.py voice       # 语音控制
    python _launch.py both        # 扫描 + 语音（语音另开窗口）
    python _launch.py check       # 麦克风自检
    python _launch.py list        # 输入设备列表

为什么还要包这一层：直接跑脚本得先记着"用 .venv 里的 python"、"语音要单独
开窗口"、"相机麦要手动 --device 2"，换机器/换人就踩坑；这里统一收口，并且
**启动前看一眼内存**——16GB 机器上 VS Code 全开能吃掉 4GB，扫描会丢帧。
"""
import ctypes
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x10)

# 低于这个可用物理内存就别硬上了：扫描要 0.4GB、TSDF 0.3GB、语音 small 复核
# 提交峰值能到 3.8GB，余量太小会丢帧（双目同步差被打散）甚至分配失败。
MEM_WARN_GB = 3.0


class _MEM(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def mem_gb():
    """(总内存, 可用内存) 单位 GB"""
    m = _MEM()
    m.dwLength = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return 0.0, 0.0
    return m.ullTotalPhys / 2 ** 30, m.ullAvailPhys / 2 ** 30


def check_env(interactive=True):
    """启动前体检：内存余量 + 是不是在 .venv 里跑

    interactive=False 时不等人回车——进主界面用不着拦一道，提醒一句就够；
    真正要开扫描/TSDF 时才值得让人确认一下。
    """
    total, avail = mem_gb()
    venv_py = os.path.join(HERE, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_py) and os.path.normcase(PY) != os.path.normcase(venv_py):
        print(f"⚠ 当前解释器不是 .venv 里的（{PY}）")
        print(f"  建议用 {venv_py}，否则可能缺 opencv/onnxruntime 等依赖")
    if total and avail < MEM_WARN_GB:
        print(f"⚠ 可用内存只剩 {avail:.1f} GB / 共 {total:.1f} GB")
        print("  扫描时容易丢帧、双目同步变差；建议先关掉 VS Code/浏览器再扫")
        if interactive:
            print("  （继续也没关系，按回车开始）")
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass


def run(script, args=(), new_console=False):
    """跑一个脚本；new_console=True 时另开一个控制台窗口（语音要独立窗口看日志）"""
    cmd = [PY, os.path.join(HERE, script), *args]
    try:
        if new_console:
            subprocess.Popen(cmd, cwd=HERE, creationflags=NEW_CONSOLE)
            print(f"  已另开窗口：{script} {' '.join(args)}")
            return 0
        return subprocess.call(cmd, cwd=HERE)
    except FileNotFoundError:
        print(f"✗ 找不到 {script}")
        return 2
    except KeyboardInterrupt:
        return 130


# (键, 说明, 执行函数)
ACTIONS = {
    "1": ("室内扫描（实时平面图 + 切片）", lambda: run("app_scan.py")),
    "2": ("室内扫描 + TSDF 三维重建（导出 mesh.ply）",
          lambda: run("app_scan.py", ["--tsdf"])),
    "3": ("语音控制（自动挑麦，控制 Indoor Scan 窗口）", lambda: run("voice_cmd.py")),
    "4": ("扫描 + 语音一起开（语音另开窗口）",
          lambda: (run("voice_cmd.py", new_console=True), run("app_scan.py"))[1]),
    "5": ("功能启动台（全部卡片）", lambda: run("app_launcher.py")),
    "6": ("麦克风自检（录 4 秒看电平 + 试识别）",
          lambda: run("voice_cmd.py", ["--check"])),
    "7": ("列出输入设备 + 现场电平", lambda: run("voice_cmd.py", ["--list"])),
}

ALIAS = {"scan": "1", "tsdf": "2", "voice": "3", "both": "4", "toolbox": "5",
         "launcher": "5", "check": "6", "list": "7"}


def menu():
    total, avail = mem_gb()
    print("=" * 60)
    print(f"  双目视觉 · 启动台      内存 {avail:.1f}/{total:.1f} GB 可用")
    print("=" * 60)
    for k, (desc, _fn) in ACTIONS.items():
        print(f"  {k}. {desc}")
    print("  0. 退出")
    print("-" * 60)
    try:
        c = input("选择：").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return 0
    if c in ("0", "q", ""):
        return 0
    c = ALIAS.get(c, c)
    if c not in ACTIONS:
        print(f"没有 {c!r} 这个选项")
        return 2
    check_env()
    print(f"→ {ACTIONS[c][0]}\n")
    return ACTIONS[c][1]()


def main():
    arg = sys.argv[1].strip().lower() if len(sys.argv) > 1 else ""
    # 无参数 = 直接进主界面（十张卡片），不再让人先答一道菜单题
    if arg in ("", "main", "home", "ui", "gui", "toolbox", "launcher"):
        check_env(interactive=False)
        return run("app_launcher.py")
    if arg == "menu":
        return menu()
    key = ALIAS.get(arg, arg)
    if key not in ACTIONS:
        print(__doc__)
        return 2
    # 开相机/跑 GPU 的那几项才值得拦一下确认内存
    check_env(interactive=key in ("1", "2", "4"))
    return ACTIONS[key][1]()


if __name__ == "__main__":
    sys.exit(main() or 0)
