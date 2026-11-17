"""voice_cmd.py —— 本地语音控制（P3）：说话 → faster-whisper 转写 → 关键词 → 给应用窗口发键

为什么独立进程 + PostMessage，而不是把语音塞进各 app 主循环：
  * 各 app 已经有键盘接口（s 保存 / space 暂停 / …），语音只是"替你按键"，
    独立进程发 WM_CHAR 就行——这条远程按键链路昨天已实测（窗口不用聚焦）；
  * whisper 转写一次 300~600ms，塞进主循环会把 19fps 的扫描卡成幻灯片。

模型：faster-whisper，**两级**——
  * 一级 base int8（145MB，约 1s/句）：正常音量下够准，延迟低；
  * 二级 small int8（461MB，约 3s/句）：一级没匹配上关键词时复核一遍。
    实测同一个很轻的输入（峰值 0.03），base 听成「暴存」、small 能听对
    「保存」——相机麦克风/远场说话就靠这一层兜住。small 只在需要时才加载。
两级都靠关键词**精确匹配**，匹配不上就不发键（宁可不执行，也不乱按键）。
实测：干净 TTS 6 句全对、-20dB 弱信号仍能发键、-30dB 由 small 兜住。

用法：
    python voice_cmd.py                     # 默认控制 Indoor Scan（自动挑麦克风）
    python voice_cmd.py --check             # 麦克风自检（录 4 秒看电平 + 识别）
    python voice_cmd.py --list              # 列出输入设备 + 现场电平（挑设备用）
    python voice_cmd.py --device 2          # 指定麦克风（--list 里的编号）
    python voice_cmd.py --target People     # 控制指定窗口（标题子串）
    python voice_cmd.py --wav a.wav         # 用录音文件跑完整链路（不占麦克风）
    python voice_cmd.py --text "帮我保存"    # 离线测关键词映射（不开麦）
    python voice_cmd.py --model small       # 一级就用 small（更准，慢 3 倍）
    python voice_cmd.py --no-fallback       # 关掉 small 复核（只要 base）

窗口内按键：t 切换目标窗口，q 退出。

指令表（中文，说一句发一键）：
    开始 / 继续 / 暂停 / 停一下 → 空格（暂停开关）
    保存 → s            重扫 / 清空 → r
    三维 / 预览 → v      网格 → g          全高度 → x
    上一层 → ]           下一层 → [        加宽 → =    变窄 → -
    退出 / 关闭 → q

踩过的坑（2026-11-17）：
  * **默认输入设备可能是相机自带的麦克风**。本机默认是 Astra 的 mic，44.1kHz
    采出来 RMS≈1e-5（等于死信号），而笔记本阵列麦克风正常。所以启动时先探
    电平，死设备自动跳到下一个能用并打印出来——不然表现就是"说了没反应"。
  * **必须按设备原生采样率开流**。用 16kHz 直接开某些 MME 设备不报错，但
    read() 回来全是 0（不是异常，是静音），比报错更难查。所以按设备默认
    采样率采，攒够整句再一次性重采样到 16k 给 whisper。
"""
import argparse
import ctypes
import os
import queue
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_detect"))

WM_CHAR = 0x0102
SR = 16000                      # whisper 要求 16kHz
BLOCK_MS = 30                   # 音频块大小
GATE_MIN = 0.0025               # 触发门限下限（RMS）——相机麦克风电平很低，别定高
GATE_RATIO = 3.0                # 门限 = max(GATE_MIN, 比率×噪声底)
GATE_MAX = 0.050                # 门限上限：环境很吵时别把自己顶死
AGC_TARGET = 0.20               # 送 whisper 前把峰值抬到这个量级
AGC_MAX = 8.0                   # 最大增益 18dB，再高就是纯噪声了
SILENCE_MS = 900                # 静音多久算一句话说完
MIN_MS = 400                    # 短于这个时长的不要（关门声、咳嗽）
MAX_MS = 7000                   # 单句上限
COOLDOWN_MS = 1500              # 发键后的冷却，防一句"保存保存"发两次
DEAD_PEAK = 1e-4                # 峰值低于这个「且」几乎全是零 → 死麦
DEAD_ZEROS = 0.97               # 全零样本占比超过这个才算"设备没出数据"
ALIVE_PEAK = 3e-3               # 峰值够高，直接判定活麦（不用再看零值占比）
# 深度相机自带的远场麦克风：能开流、也能采到一丁点抖动，但人声进去只有
# 1e-5~1e-3 量级，等于听不见。它常常正好是 Windows 默认输入（插上相机就被
# 设成默认），所以不能"默认设备能用就用它"——名字像相机麦的，一律让给笔记本
# 阵列麦。实测本机：#1 Astra 麦 peak=3e-5 zeros=0.77，#2 阵列麦 peak=1e-3 zeros=0.22。
CAMERA_MIC_HINTS = ("orbbec", "astra", "深度相机", "depth camera")
# "声音映射器"只是把系统默认设备再暴露一遍，它指向的就是上面那台相机麦，
# 选它等于没换设备，所以排到最后。
MAPPER_HINTS = ("声音映射器", "sound mapper")

# 顺序就是优先级：先长后短、先具体后笼统（"全高度"必须先于"高度"类词，
# "退出"必须后于"保存"——一句里出现两个词时按这里的前后取）
KEYWORDS = [
    (["全高度", "全部显示", "看全部"], "x", "全高度视图"),
    (["上一层", "切片上移", "往上切", "上移一层"], "]", "切片上移"),
    (["下一层", "切片下移", "往下切", "下移一层"], "[", "切片下移"),
    (["加宽", "带宽一点", "带宽一些"], "=", "切片加宽"),
    (["变窄", "带宽窄一点", "窄一点"], "-", "切片变窄"),
    (["暂停", "停一下", "停一下扫描", "停止"], " ", "暂停/继续"),
    (["继续", "开始扫描", "开始"], " ", "暂停/继续"),
    (["保存", "存一下", "存盘"], "s", "保存"),
    (["重扫", "重新扫", "重来", "清空"], "r", "清空重扫"),
    (["三维", "3D", "立体", "预览"], "v", "3D 预览"),
    (["网格"], "g", "网格开关"),
    (["退出", "关闭", "关掉"], "q", "退出"),
]

TARGETS = ["Indoor Scan", "Objects", "People", "Gesture", "Depth", "Toolbox"]


def map_text(text):
    """转写文本 → (按键, 说明)。匹配不上返回 ("", "")——宁可不执行。

    一句话里命中多个词时**取最先出现的那个**，而不是按词表顺序。实测反例：
    弱信号下 base 输出「保存 第1层 下一层 下一层…」，按词表顺序会命中"下一层"
    发出 `[`（用户明明说的是"保存"），按位置取就能发对 `s`。
    """
    t = (text or "").replace(" ", "").replace("，", "").replace("。", "")
    best = ("", "")
    pos_best = len(t) + 1
    for kws, key, desc in KEYWORDS:
        for k in kws:
            i = t.find(k)
            if 0 <= i < pos_best:
                pos_best = i
                best = (key, desc)
    return best


def match_keys(text):
    """文本里命中的全部指令 → [(位置, 按键, 说明)]，按位置排序、按按键去重

    为什么要"全部"：一句话里出现两个不同指令时（常见于 whisper 把提示词
    幻觉成词表），发哪个都是猜。上层据此判"歧义 → 让 small 复核 → 仍歧义
    就不发键"，比盲目取第一个安全。
    """
    t = (text or "").replace(" ", "").replace("，", "").replace("。", "")
    hits = []
    for kws, key, desc in KEYWORDS:
        pos = min([i for i in (t.find(k) for k in kws) if i >= 0], default=-1)
        if pos < 0:
            continue
        if any(h[1] == key for h in hits):        # 同一个键算一条（"三维预览"→v）
            continue
        hits.append((pos, key, desc))
    return sorted(hits)


def find_target(sub):
    """按标题子串找可见窗口，返回 (hwnd, 标题) 或 None（grab_window 的 find）"""
    from grab_window import find
    hits = find(sub)
    return hits[0] if hits else None


def send_key(hwnd, ch):
    """远程发键：PostMessage WM_CHAR，目标窗口不需要聚焦（已实测）"""
    u32 = ctypes.windll.user32
    u32.PostMessageW(hwnd, WM_CHAR, ord(ch), 0)


# ------------------------------------------------------------ 麦克风 ---
def list_inputs():
    """所有输入设备 [(编号, 名字, 默认采样率)]"""
    import sounddevice as sd
    return [(i, d["name"], int(d["default_samplerate"]))
            for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0]


def is_dead(info):
    """设备是不是"根本没出数据"

    只靠峰值判不出来：安静房间里好麦克风峰值也只有 3e-5，和死设备一样。
    真正的区别是**样本是否精确为 0**——死设备/静音端点整段 0.0，活麦克风
    永远有芯片底噪抖动（零值占比 0.1~0.5）。所以两个条件同时成立才算死。
    """
    if not info:
        return True
    return info["peak"] < DEAD_PEAK and info["zeros"] > DEAD_ZEROS


def is_camera_mic(name):
    """名字像深度相机自带的远场麦（声音小到听不见，但开流不报错）"""
    n = (name or "").lower()
    return any(h in n for h in CAMERA_MIC_HINTS)


def is_mapper(name):
    """"声音映射器"：只是系统默认设备的别名，选它等于没换设备"""
    n = (name or "").lower()
    return any(h in n for h in MAPPER_HINTS)


def _grab(dev, rate, seconds):
    """按指定采样率录一小段，返回 float32 单声道；开不了流抛异常"""
    import sounddevice as sd
    with sd.InputStream(device=dev, samplerate=rate, channels=1,
                        dtype="float32") as s:
        t0 = time.time()
        buf = []
        while time.time() - t0 < seconds:
            d, _ = s.read(max(1, int(rate * 0.05)))
            buf.append(d[:, 0])
        return np.concatenate(buf)


def probe_device(dev, seconds=0.45, rates=None):
    """录一小段判断麦克风是否真的在工作 → {rate, rms, peak, zeros} / None

    **必须多采样率都试**：本机实测同一台笔记本阵列麦克风，44100Hz 采出来
    峰值 1e-4（等于死），48000Hz 却有 0.08 —— Windows 共享模式的重采样会
    把某些组合搞成全静音，而且不报错。哪个速率峰值高就用哪个。

    判"死麦"也不能用 rms：安静房间里好麦克风 rms 也只有 1e-5 量级，和真死
    的设备分不开。peak 才分得开（正常麦克风永远有底噪抖动）。
    """
    import sounddevice as sd
    try:
        info = sd.query_devices(dev)
    except Exception:                                    # noqa: BLE001
        return None
    if rates is None:
        rates = [int(info["default_samplerate"]), 48000, 44100]
    best = None
    seen = set()
    for rate in rates:
        if rate <= 0 or rate in seen:
            continue
        seen.add(rate)
        try:
            x = _grab(dev, rate, seconds)
        except Exception:                                # noqa: BLE001
            continue
        cur = dict(rate=rate,
                   rms=float(np.sqrt(np.mean(x * x))),
                   peak=float(np.abs(x).max()),
                   zeros=float((x == 0).mean()))
        if best is None or cur["peak"] > best["peak"]:
            best = cur
        if not is_dead(best):                            # 已经确定是活的，不用再试
            break
    return best


def probe_best(dev, tries=2, seconds=0.6):
    """探几次，取峰值最高的那一份

    单次探测抖得厉害：同一台笔记本阵列麦，这一秒 peak=1e-3、下一秒 6e-5。
    采样率那一轮已经按"探到活麦就停"，这里再按轮次取最大，避免一次碰巧静音
    就把好麦判成死麦。
    """
    best = None
    for _ in range(max(1, tries)):
        info = probe_device(dev, seconds)
        if info and (best is None or info["peak"] > best["peak"]):
            best = info
        if best and best["peak"] >= ALIVE_PEAK:
            break
    return best


def choose_device(prefer=None, verbose=True, max_probe=3):
    """挑麦克风 → (dev, rate, name, info)

    规则按优先级：
      1. `--device N` 给了就用它（用户说了算，打不开才报错）；
      2. 默认设备**能用、且不是相机远场麦/映射器**→ 用它（尊重系统里的选择）；
      3. 否则在剩下的真麦克风里挑**峰值最高**的那台（最多探 3 台，每台探 2 次）；
      4. 一台真麦都挑不出来 → 退回默认设备并明确告警（总比没有强）。
    """
    import sounddevice as sd
    rows = list_inputs()
    if prefer is not None:
        dev = int(prefer)
        info = probe_device(dev, 0.8)
        name = sd.query_devices(dev)["name"]
        if info is None:
            raise SystemExit(f"#{dev} 打不开（--list 看看编号）")
        return dev, info["rate"], name, info

    d0 = sd.default.device[0]
    d0 = int(d0) if d0 is not None and int(d0) >= 0 else None
    d0_name = sd.query_devices(d0)["name"] if d0 is not None else ""
    d0_info = None
    if d0 is not None:
        d0_info = probe_best(d0, 2, 0.8)
        if d0_info and verbose:
            print(f"  默认 #{d0} {d0_name[:40]}: peak={d0_info['peak']:.5f} "
                  f"zeros={d0_info['zeros']:.2f} @{d0_info['rate']}Hz", flush=True)
        if d0_info and not is_dead(d0_info) and not is_camera_mic(d0_name) \
                and not is_mapper(d0_name):
            return d0, d0_info["rate"], d0_name, d0_info

    best = None
    tried = 0
    for dev, name, _rate in rows:
        if dev == d0 or is_camera_mic(name) or is_mapper(name):
            continue
        if tried >= max_probe:
            break
        tried += 1
        info = probe_best(dev, 2, 0.6)
        if not info:
            continue
        if verbose:
            print(f"  #{dev:<3} {name[:40]:<42} peak={info['peak']:.5f} "
                  f"zeros={info['zeros']:.2f} @{info['rate']}Hz", flush=True)
        if is_dead(info):
            continue
        if best is None or info["peak"] > best[1]["peak"]:
            best = (dev, info, name)

    if best is not None:
        dev, info, name = best
        if verbose:
            why = "是相机远场麦（声音太小）" if is_camera_mic(d0_name) else "没有数据（peak≈0）"
            print(f"  → 默认设备{why}，改用 #{dev} {name[:30]}", flush=True)
        return dev, info["rate"], name, info

    if d0 is not None and d0_info is not None:
        print(f"⚠ 只找到相机/映射器麦克风（#{d0} {d0_name[:24]}），它声音很小、"
              f"可能听不见——在 Windows「声音设置」里把它禁用，或用 --device N 指定",
              flush=True)
        return d0, d0_info["rate"], d0_name, d0_info
    return None


def resample(x, sr_in, sr_out=SR):
    """线性插值重采样（语音够用，省一个 scipy 依赖）"""
    if sr_in == sr_out or len(x) == 0:
        return np.asarray(x, np.float32)
    n = max(1, int(round(len(x) * sr_out / float(sr_in))))
    xi = np.linspace(0.0, len(x) - 1.0, n)
    return np.interp(xi, np.arange(len(x)), x).astype(np.float32)


# --------------------------------------------------------------- 状态窗 ---
def _font(size, bold=False):
    from PIL import ImageFont
    name = "msyhbd.ttc" if bold else "msyh.ttc"
    p = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", name)
    return ImageFont.truetype(p, size) if os.path.exists(p) \
        else ImageFont.load_default()


class Status:
    """小状态窗（Pillow 渲染中文）：目标窗口 / 状态 / 听到的指令"""

    W, H = 400, 168
    BG = (22, 24, 28)
    TEXT = (235, 238, 243)
    DIM = (140, 146, 158)
    OK = (110, 200, 130)
    REC = (110, 170, 255)
    BAD = (90, 110, 235)

    def __init__(self, win):
        import cv2
        from PIL import Image, ImageDraw
        self.cv2 = cv2
        self.Image = Image
        self.ImageDraw = ImageDraw
        self.win = win
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        self.f_t = _font(17, True)
        self.f_s = _font(14)
        self.f_l = _font(16, True)
        self._img = None

    def show(self, target, state, heard, desc, matched, level=0.0, gate=0.0,
             dead=False, mic=""):
        """state: listen / rec / busy；heard 有变化或状态变化才重绘"""
        img = self.Image.new("RGB", (self.W, self.H), self.BG)
        d = self.ImageDraw.Draw(img)
        d.text((18, 12), "语音控制", font=self.f_t, fill=self.TEXT)
        d.text((18, 42), f"目标窗口：{target or '（没找到）'}", font=self.f_s,
               fill=self.OK if target else self.BAD)
        st_txt = {"listen": "● 监听中，说指令…", "rec": "● 录音中…",
                  "busy": "● 识别中…"}.get(state, "")
        st_col = {"listen": self.DIM, "rec": self.REC, "busy": self.OK}.get(
            state, self.DIM)
        d.text((18, 66), st_txt, font=self.f_s, fill=st_col)
        if heard:
            col = self.OK if matched else self.DIM
            d.text((18, 92), f"「{heard}」", font=self.f_l, fill=col)
            d.text((18, 116), f"→ {desc}" if matched else "（没有匹配的指令）",
                   font=self.f_s, fill=col)
        else:
            d.text((18, 92), "说「保存」「暂停」「全高度」…", font=self.f_s,
                   fill=self.DIM)
            d.text((18, 116), "t 切换目标 · q 退出", font=self.f_s, fill=self.DIM)

        # 电平条：说话时能看见条形跳，用户才有"麦克风在工作"的确定感
        x0, x1, y0, y1 = 18, self.W - 18, 142, 152
        d.rectangle([x0, y0, x1, y1], outline=(58, 62, 70))
        frac = min(1.0, level / 0.15) if level else 0.0
        if frac > 0:
            d.rectangle([x0, y0, x0 + int((x1 - x0) * frac), y1],
                        fill=self.REC if state == "rec" else (70, 110, 160))
        if gate > 0:
            gx = x0 + int((x1 - x0) * min(1.0, gate / 0.15))
            d.line([gx, y0 - 2, gx, y1 + 2], fill=(230, 170, 90), width=1)
        tip = "MIC 无信号！--list 挑设备" if dead else f"level {level:.3f}"
        d.text((18, y0 + 12), tip, font=self.f_s,
               fill=self.BAD if dead else self.DIM)
        if mic:
            d.text((150, y0 + 12), mic[:30], font=self.f_s, fill=self.DIM)

        arr = np.array(img)[:, :, ::-1].copy()
        self.cv2.imshow(self.win, arr)


# --------------------------------------------------------------- 模型 ---
def load_model(name):
    """faster-whisper int8（纯 CPU）。本地有就用本地，没有走 hf-mirror 下"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from faster_whisper import WhisperModel
    local = os.path.join(HERE, "models", "whisper", name)
    kw = dict(device="cpu", compute_type="int8", cpu_threads=8)
    if os.path.isdir(local) and os.path.exists(os.path.join(local, "model.bin")):
        return WhisperModel(local, **kw)
    print(f"本地没有 {name}，开始下载（hf-mirror，约几百 MB）…", flush=True)
    return WhisperModel(name, download_root=os.path.join(HERE, "models", "whisper"),
                        **kw)


# 提示词只放「词表」，不放句子。放整句（"以下是室内扫描软件的简短中文指令："）
# 时，输入一旦偏弱 whisper 会把提示句本身当内容吐出来——实测就是这样听到了
# "这次的室内扫描软件 继续 保存…"，然后一个关键词都匹配不上。
PROMPT = ("暂停 开始 保存 重扫 三维 网格 全高度 上一层 下一层 "
          "加宽 变窄 退出 清空")
NO_SPEECH_MAX = 0.55          # VAD 概率超过这个当"没有人说话"（幻觉的主要来源）


_MODELS = {}


def get_model(name):
    """按名字取模型（缓存，small 只在第一次复核时才真正加载）"""
    if name not in _MODELS:
        _MODELS[name] = load_model(name)
    return _MODELS[name]


def transcribe(model, audio, want_conf=False):
    """float32 16k → 文本（可选返回置信度）。短指令贪心解码就够，别上 beam"""
    segments, _info = model.transcribe(
        audio, language="zh", beam_size=1, vad_filter=False,
        condition_on_previous_text=False, initial_prompt=PROMPT)
    segs = list(segments)
    text = "".join(s.text for s in segs).strip()
    conf = dict(no_speech=max([s.no_speech_prob for s in segs], default=0.0),
                logprob=min([s.avg_logprob for s in segs], default=0.0))
    # 幻觉的典型形态是"同一段词反复念"（下一层 下一层 下一层…）：这类输出
    # 既可能是空音频，也可能是把噪声硬解成词，一律不认，交给 small 复核
    toks = [w for w in text.replace("，", " ").replace("。", " ").split() if w]
    rep = max([toks.count(w) for w in set(toks)], default=0) if toks else 0
    conf["repeat"] = rep
    conf["suspect"] = rep >= 3 or len(toks) > 8
    conf["speech"] = conf["no_speech"] <= NO_SPEECH_MAX and not conf["suspect"]
    return (text, conf) if want_conf else text


# --------------------------------------------------------------- 自检 ---
def check_mic(args, seconds=4.0):
    """麦克风自检：录几秒 → 打印实时电平 → 识别 → 不发键

    "语音控制没反应"九成是麦克风这一层：系统默认输入可能是相机自带的远场
    麦克风，或者输入音量被调到很低，表现都是"说了没反应"。这条命令把电平
    和识别结果都摊开给用户看，不用猜。
    """
    import sounddevice as sd
    picked = choose_device(args.device)
    if not picked:
        print("没有能用的输入设备（--list 看看）")
        return
    dev, rate, name, info = picked
    print(f"设备 #{dev} {name} @{rate}Hz")
    print(f"请对着麦克风说一句「保存」（录 {seconds:.0f} 秒）…", flush=True)
    buf = []
    with sd.InputStream(device=dev, samplerate=rate, channels=1,
                        dtype="float32") as st:
        t0 = time.time()
        while time.time() - t0 < seconds:
            d, _ = st.read(int(rate * 0.08))
            x = d[:, 0]
            buf.append(x.copy())
            lev = float(np.abs(x).max())
            bar = "#" * min(40, int(lev * 200))
            print(f"\r  {lev:.4f} |{bar:<40}|", end="", flush=True)
    print()
    a = np.concatenate(buf)
    peak, rms = float(np.abs(a).max()), float(np.sqrt(np.mean(a ** 2)))
    print(f"峰值 {peak:.4f}   有效值 {rms:.5f}")
    if peak < 0.003:
        print("✗ 电平太低：Windows 设置 → 系统 → 声音 → 输入，把输入音量调大，")
        print("  或换一台麦克风（--list / --device N）。这样说话 whisper 只会瞎猜。")
        return
    model = get_model(args.model)
    gain = min(AGC_MAX, AGC_TARGET / max(peak, 1e-6)) if peak < AGC_TARGET else 1.0
    heard, conf = transcribe(model, np.clip(resample(a, rate) * gain, -1, 1),
                             want_conf=True)
    key, desc = map_text(heard) if conf["speech"] else ("", "")
    print(f"识别：{heard!r}  （增益 {20 * np.log10(gain):.0f}dB, "
          f"no_speech={conf['no_speech']:.2f}）")
    if key:
        print(f"✓ 会对目标窗口发出 {key!r}（{desc}）")
    elif heard:
        print("△ 听到了但没匹配上指令词（换句更短的，或者说「保存」「暂停」）")
    else:
        print("✗ 没识别出内容：声音太轻或被判为无语音，检查输入音量/距离")


# --------------------------------------------------------------- 主循环 ---
def run(args):
    import cv2
    import sounddevice as sd
    from ui_common import wait_key

    if args.text:
        key, desc = map_text(args.text)
        print(f"「{args.text}」 -> {'按 ' + repr(key) + '（' + desc + '）' if key else '无匹配'}")
        return

    if args.check:
        check_mic(args)
        return

    if args.list:
        print("输入设备（判死看两点：peak<1e-4 且 整段几乎全零；静音时活麦 peak 也只有 1e-5）")
        for dev, name, rate in list_inputs():
            info = probe_device(dev, 0.5)
            tag = "默认" if dev == sd.default.device[0] else "    "
            if not info:
                print(f"  {tag} #{dev:<3} {name[:44]:<46} 打不开")
                continue
            mark = "  ← 死麦" if is_dead(info) else "  ← 可用"
            print(f"  {tag} #{dev:<3} {name[:44]:<46} {info['rate']}Hz  "
                  f"peak={info['peak']:.5f} zeros={info['zeros']:.2f}{mark}")
        return

    win = "VoiceCmd  [t=target  q=quit]"
    st = Status(win)

    target_hwnd, target_title = (None, "")
    target_sub = args.target
    cur = find_target(target_sub)
    if cur:
        target_hwnd, target_title = cur

    frames_q = queue.Queue()
    stop_evt = threading.Event()
    stream = None

    if args.wav:
        # 录音文件按真实时间节奏喂进主循环 —— 走的是和麦克风完全相同的
        # 门限/切句/转写/发键链路，只是音源换成文件（不占麦克风，能回归测试）
        audio_in, sr_in = _read_wav(args.wav)
        audio_in = resample(audio_in, sr_in)
        rate, dev, dev_name = SR, None, args.wav
        level0 = float(np.sqrt(np.mean(audio_in ** 2)))
        dead = False
        print(f"注入 {args.wav}：{len(audio_in) / SR:.1f}s  rms={level0:.4f}",
              flush=True)

        def feed():
            step = SR * BLOCK_MS // 1000
            t0 = time.time()
            i = 0
            while i < len(audio_in) and not stop_evt.is_set():
                frames_q.put(audio_in[i:i + step])
                i += step
                time.sleep(max(0.0, t0 + i / SR - time.time()))
            tail = np.zeros(int(SR * 1.4), np.float32)   # 尾部静音，好让句子收尾
            for j in range(0, len(tail), step):
                if stop_evt.is_set():
                    break
                frames_q.put(tail[j:j + step])
                time.sleep(BLOCK_MS / 1000.0)

        threading.Thread(target=feed, daemon=True).start()
    else:
        print("挑麦克风（每个设备试 0.6s，看有没有真的数据）…", flush=True)
        picked = choose_device(args.device)
        if not picked:
            raise SystemExit("没有能用的麦克风输入设备（--list 看看）")
        dev, rate, dev_name, info0 = picked
        level0 = float(info0.get("rms", 0.0))
        dead = is_dead(info0)
        print(f"麦克风 #{dev} {dev_name} @{rate}Hz  peak={info0.get('peak', 0):.5f}"
              f" rms={level0:.5f}"
              + ("  ⚠ 没采到数据（整段全零）：换一台试试 --device N"
                 if dead else ""), flush=True)
    if target_title:
        print(f"目标窗口：{target_title}", flush=True)
    else:
        print(f"目标窗口：没找到（标题含 {target_sub!r}），先开扫描窗口，或按 t 切换",
              flush=True)

    model = load_model(args.model)
    mic_label = f"#{dev} {str(dev_name)[:24]}" if dev is not None else "（注入文件）"
    print(f"模型就绪（{args.model} int8 CPU）。开始监听。", flush=True)

    def cb(indata, _f, _t, _s):
        frames_q.put(indata[:, 0].copy())

    if not args.wav:
        blocksize = max(64, int(rate * BLOCK_MS / 1000))
        stream = sd.InputStream(device=dev, samplerate=rate, blocksize=blocksize,
                                channels=1, dtype="float32", callback=cb)
        stream.start()

    noise = max(0.002, level0)
    rec, rec_ms, sil_ms = [], 0.0, 0.0
    state = "listen"
    cool_until = 0.0
    last_shown = ("", "", "", "")
    last_refind = 0.0
    level = level0
    gate = GATE_MIN
    t_last_draw = 0.0
    zero_since = time.time() if dead else None

    def pump_frames():
        """取走队列里全部块，返回 [(音频块, rms)]（顺带更新噪声底）"""
        out = []
        nonlocal noise
        while True:
            try:
                f = frames_q.get_nowait()
            except queue.Empty:
                return out
            rms = float(np.sqrt(np.mean(f * f)) + 1e-12)
            out.append((f, rms))
            if state == "listen":
                noise = 0.98 * noise + 0.02 * rms

    try:
        while True:
            now = time.time()
            blocks = pump_frames()
            for f, rms in blocks:
                level = rms if state != "listen" else 0.7 * level + 0.3 * rms
                gate = min(GATE_MAX, max(GATE_MIN, GATE_RATIO * noise))
                if state == "listen" and rms > gate and now > cool_until:
                    state, rec, rec_ms, sil_ms = "rec", [f], BLOCK_MS, 0.0
                elif state == "rec":
                    rec.append(f)
                    rec_ms += BLOCK_MS
                    sil_ms = 0.0 if rms > gate * 0.5 else sil_ms + BLOCK_MS
                    if (sil_ms >= SILENCE_MS and rec_ms >= MIN_MS) or rec_ms >= MAX_MS:
                        state = "busy"
                        break

            if state == "busy":
                st.show(target_title, "busy", "", "", False, level, gate,
                        mic=mic_label)
                audio = np.concatenate(rec)
                peak = float(np.abs(audio).max())
                heard, conf = "", dict(no_speech=0.0, speech=True)
                gain = 1.0                                     # 复核时也要用
                loud = peak >= max(0.0015, 1.8 * noise)
                if not loud:
                    # 太轻了，别送给 whisper —— 它会拿提示词凑一句出来
                    print(f"[{time.strftime('%H:%M:%S')}] 声音太小"
                          f"（peak={peak:.4f}）跳过", flush=True)
                else:
                    # 自动增益：相机麦克风/远场说话峰值只有 0.01 量级，whisper
                    # 对这么轻的输入几乎必然胡编。抬到 0.2 量级再识别。
                    gain = 1.0
                    if peak < AGC_TARGET:
                        gain = min(AGC_MAX, AGC_TARGET / max(peak, 1e-6))
                    try:
                        heard, conf = transcribe(
                            model, np.clip(resample(audio, rate) * gain, -1, 1),
                            want_conf=True)
                        if gain > 1.5:
                            print(f"（已增益 {20 * np.log10(gain):.0f}dB）",
                                  end="", flush=True)
                    except Exception as e:                       # noqa: BLE001
                        print(f"转写失败：{e}", flush=True)
                if heard and not conf["speech"]:
                    why = ("重复词×%d" % conf["repeat"]) if conf.get("suspect") \
                        else ("no_speech=%.2f" % conf["no_speech"])
                    print(f"[{time.strftime('%H:%M:%S')}] 可能是幻觉（{why}）："
                          f"{heard!r}", flush=True)
                    heard = ""
                hits = match_keys(heard)
                key, desc = (hits[0][1], hits[0][2]) if len(hits) == 1 else ("", "")
                if len(hits) > 1:
                    print(f"   命中多个指令 {[h[1] for h in hits]}，先不乱发", flush=True)
                # 没匹配上 / 有歧义 → 用 small 复核（弱麦克风时 base 常听错）
                if not key and loud and not args.no_fallback and \
                        args.model != "small":
                    try:
                        model2 = get_model("small")
                        heard2, conf2 = transcribe(
                            model2, np.clip(resample(audio, rate) * gain, -1, 1),
                            want_conf=True)
                        hits2 = match_keys(heard2) if conf2["speech"] else []
                        print(f"   复核（small）：{heard2!r} -> "
                              f"{[h[1] for h in hits2] or '无匹配'}", flush=True)
                        if len(hits2) == 1:
                            heard, key, desc = heard2, hits2[0][1], hits2[0][2]
                    except Exception as e:                       # noqa: BLE001
                        print(f"复核失败：{e}", flush=True)
                print(f"[{time.strftime('%H:%M:%S')}] heard={heard!r} "
                      f"-> {key or '无匹配'}", flush=True)
                if key and target_hwnd:
                    send_key(target_hwnd, key)
                    cool_until = time.time() + COOLDOWN_MS / 1000
                    print(f"   已发送 {key!r} → {target_title}", flush=True)
                elif key:
                    print("   没有目标窗口，按键没发出去（按 t 切换）", flush=True)
                st.show(target_title, "listen", heard, desc or heard, bool(key),
                        level, gate, mic=mic_label)
                last_shown = (target_title, heard, desc, key)
                state = "listen"
                rec, rec_ms, sil_ms = [], 0.0, 0.0
            else:
                # 电平条每 100ms 刷一次；文字部分变了也刷
                if now - t_last_draw > 0.1:
                    t_last_draw = now
                    st.show(target_title, state, "", "", False, level, gate,
                            dead, mic=mic_label)
                    last_shown = (target_title, "", "", "")

            # 死信号告警：连续 3 秒 rms 还是 0，状态窗标红（注入模式不报）
            if level <= DEAD_PEAK / 10 and not args.wav:
                zero_since = zero_since or time.time()
                if time.time() - zero_since > 3.0 and not dead:
                    dead = True
                    print("⚠ 麦克风连续 3 秒没数据（rms≈0）：换设备 --list / --device N",
                          flush=True)
            else:
                zero_since = None
                dead = False

            # 目标窗口可能被关掉重开：每 2 秒查一次可见性，没了就重找
            if now - last_refind > 2.0:
                last_refind = now
                u32 = ctypes.windll.user32
                if not target_hwnd or not u32.IsWindowVisible(target_hwnd):
                    c = find_target(target_sub)
                    if c:
                        target_hwnd, target_title = c
                        print(f"目标窗口：{target_title}", flush=True)
                        last_shown = ("", "", "", "")

            k = wait_key(win, 30)
            if k in (27, ord("q")):
                break
            if k == ord("t"):
                vis = [find_target(s) for s in TARGETS]
                vis = [(h, t) for h, t in vis if h]
                if vis:
                    names = [t for _, t in vis]
                    i = (names.index(target_title) + 1) % len(names) \
                        if target_title in names else 0
                    target_hwnd, target_title = vis[i][0], vis[i][1]
                    print(f"目标切换：{target_title}", flush=True)
                    last_shown = ("", "", "", "")
            # 窗口 X 退出：cv2 的 X 不会结束 waitKey 循环，得查可见性
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        stop_evt.set()
        if stream is not None:
            stream.stop()
            stream.close()
        cv2.destroyAllWindows()


def _read_wav(path):
    """读 wav → (float32 单声道, 采样率)"""
    import wave
    with wave.open(path) as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
        sw = w.getsampwidth()
    if sw == 2:
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
    else:
        x = np.frombuffer(raw, np.uint8).astype(np.float32) / 128.0 - 1.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x.astype(np.float32), sr


def main():
    ap = argparse.ArgumentParser(description="本地语音控制（whisper 关键词 → 应用窗口按键）")
    ap.add_argument("--target", default="Indoor Scan", help="目标窗口标题子串")
    ap.add_argument("--model", default="base", help="一级 whisper 模型（base/small）")
    ap.add_argument("--no-fallback", action="store_true",
                    help="一级没匹配上时不再用 small 复核")
    ap.add_argument("--device", type=int, default=None,
                    help="麦克风设备编号（--list 查看；默认自动挑能出数据的）")
    ap.add_argument("--list", action="store_true", help="列出输入设备 + 现场电平")
    ap.add_argument("--check", action="store_true",
                    help="麦克风自检：录 4 秒、看电平、试识别（不发键）")
    ap.add_argument("--wav", default="", help="用录音文件跑完整链路（不占麦克风）")
    ap.add_argument("--text", default="", help="离线测关键词映射，不开麦")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
