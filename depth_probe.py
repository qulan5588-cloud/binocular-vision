# -*- coding: utf-8 -*-
"""
Orbbec Astra Pro Plus 深度流验证脚本（纯 ctypes，无需 GUI / 编译）
用法: python depth_probe.py [输出目录]
"""
import ctypes, os, sys, time, zlib, struct
from ctypes import c_void_p, c_uint32, c_int, c_float, c_char_p, POINTER, byref
import numpy as np

VIEWER_DIR = r"D:\双目视觉\_driver\viewer"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else r"D:\双目视觉\_detect"
os.makedirs(OUT_DIR, exist_ok=True)
os.add_dll_directory(VIEWER_DIR)

lib = ctypes.CDLL(os.path.join(VIEWER_DIR, "OrbbecSDK.dll"))
PP = POINTER(c_void_p)

def sig(name, restype, argtypes):
    f = getattr(lib, name)
    f.restype = restype
    f.argtypes = argtypes
    return f

ob_create_context            = sig("ob_create_context", c_void_p, [PP])
ob_create_context_with_config= sig("ob_create_context_with_config", c_void_p, [c_char_p, PP])
ob_query_device_list         = sig("ob_query_device_list", c_void_p, [c_void_p, PP])
ob_device_list_device_count  = sig("ob_device_list_device_count", c_uint32, [c_void_p, PP])
ob_device_list_get_serial    = sig("ob_device_list_get_device_serial_number", c_char_p, [c_void_p, c_uint32, PP])
ob_create_pipeline           = sig("ob_create_pipeline", c_void_p, [PP])
ob_pipeline_start            = sig("ob_pipeline_start", None, [c_void_p, PP])
ob_pipeline_wait_for_frameset= sig("ob_pipeline_wait_for_frameset", c_void_p, [c_void_p, c_uint32, PP])
ob_frameset_depth_frame      = sig("ob_frameset_depth_frame", c_void_p, [c_void_p, PP])
ob_frameset_color_frame      = sig("ob_frameset_color_frame", c_void_p, [c_void_p, PP])
ob_video_frame_width         = sig("ob_video_frame_width", c_uint32, [c_void_p, PP])
ob_video_frame_height        = sig("ob_video_frame_height", c_uint32, [c_void_p, PP])
ob_frame_data                = sig("ob_frame_data", c_void_p, [c_void_p, PP])
ob_frame_data_size           = sig("ob_frame_data_size", c_uint32, [c_void_p, PP])
ob_frame_format              = sig("ob_frame_format", c_int, [c_void_p, PP])
ob_depth_frame_value_scale   = sig("ob_depth_frame_get_value_scale", c_float, [c_void_p, PP])
ob_frame_index               = sig("ob_frame_index", ctypes.c_uint64, [c_void_p, PP])
ob_delete_frame              = sig("ob_delete_frame", None, [c_void_p, PP])
ob_pipeline_stop             = sig("ob_pipeline_stop", None, [c_void_p, PP])
ob_error_message             = sig("ob_error_message", c_char_p, [c_void_p])

def err_text(e):
    if not e:
        return ""
    e = e.contents.value if hasattr(e, "contents") else e
    return ob_error_message(e) or ""

def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    open(path, "wb").write(png)

print("=" * 60)
print("Orbbec Astra Pro Plus 深度流验证")
print("=" * 60)

e = c_void_p()
CFG = r"D://双目视觉//_driver//viewer//OrbbecSDKConfig_v1.0.xml"
ctx = ob_create_context_with_config(CFG.encode(), byref(e))
if not ctx:
    print("创建 context 失败:", err_text(e)); sys.exit(1)

print("[1] 枚举设备 ...")
devs = ob_query_device_list(ctx, byref(e))
if not devs:
    print("查询设备列表失败:", err_text(e)); sys.exit(1)
n = ob_device_list_device_count(devs, byref(e))
print("    发现设备数量:", n)
if n == 0:
    print("    !! 没有任何 Orbbec 设备被枚举到"); sys.exit(1)
for i in range(n):
    sn = ob_device_list_get_serial(devs, i, byref(e))
    print(f"    设备[{i}] 序列号: {sn.decode() if sn else '(空)'}")

print("[2] 创建 pipeline 并启动默认流 ...")
pipe = ob_create_pipeline(byref(e))
if not pipe:
    print("创建 pipeline 失败:", err_text(e)); sys.exit(1)
t0 = time.time()
ob_pipeline_start(pipe, byref(e))
if e:
    print("    启动失败:", err_text(e)); sys.exit(1)
print(f"    启动成功，耗时 {time.time()-t0:.2f}s")

print("[3] 接收帧 ...")
got = 0
depth_arr = None
info = {}
fps_t0 = None
NF = 30
for k in range(NF):
    fs = ob_pipeline_wait_for_frameset(pipe, 2000, byref(e))
    if not fs:
        print("    第", k, "次等待帧超时:", err_text(e))
        continue
    if fps_t0 is None:
        fps_t0 = time.time()
    df = ob_frameset_depth_frame(fs, byref(e))
    if not df:
        print("    该帧集里没有深度帧:", err_text(e))
        continue
    if k == 0:
        w = ob_video_frame_width(df, byref(e))
        h = ob_video_frame_height(df, byref(e))
        fmt = ob_frame_format(df, byref(e))
        scale = ob_depth_frame_value_scale(df, byref(e))
        sz = ob_frame_data_size(df, byref(e))
        ptr = ob_frame_data(df, byref(e))
        info = dict(w=w, h=h, fmt=fmt, scale=scale, size=sz)
        buf = ctypes.string_at(ptr, sz)
        arr = np.frombuffer(buf, dtype=np.uint16).reshape(h, w).copy()
        depth_arr = arr
        print(f"    深度帧: {w}x{h}  format={fmt}(8=Y16)  size={sz}B  value_scale={scale}")
    if k == NF - 1:
        cf = ob_frameset_color_frame(fs, byref(e))
        if cf:
            cw = ob_video_frame_width(cf, byref(e))
            ch = ob_video_frame_height(cf, byref(e))
            csz = ob_frame_data_size(cf, byref(e))
            print(f"    彩色帧: {cw}x{ch}  size={csz}B  format={ob_frame_format(cf, byref(e))}")
    ob_delete_frame(fs, byref(e))
    got += 1

dt = time.time() - fps_t0 if fps_t0 else 0
print(f"    收到 {got}/{NF} 个帧集，用时 {dt:.2f}s -> {got/dt if dt else 0:.1f} fps")

if depth_arr is None:
    print("[!] 未取到深度帧，退出"); sys.exit(2)

print("[4] 深度数据分析 ...")
valid = depth_arr[depth_arr > 0]
scale = info["scale"] or 1.0
print(f"    分辨率      : {info['w']}x{info['h']}")
print(f"    有效像素    : {valid.size} / {depth_arr.size}  ({valid.size/depth_arr.size*100:.1f}%)")
if valid.size:
    print(f"    原始值范围  : {depth_arr.min()} ~ {depth_arr.max()}")
    mm = depth_arr.astype(np.float64) * scale
    vv = valid.astype(np.float64) * scale
    print(f"    物理距离    : {vv.min():.0f} mm ~ {vv.max():.0f} mm  (中位 {np.median(vv):.0f} mm)")
    print(f"    中心 20x20  : {np.median(mm[info['h']//2-10:info['h']//2+10, info['w']//2-10:info['w']//2+10]):.0f} mm")

# 灰度映射保存（0=黑）
norm = np.zeros_like(depth_arr, dtype=np.float64)
if valid.size:
    mm_all = depth_arr.astype(np.float64) * scale
    lo, hi = np.percentile(vv, 2), np.percentile(vv, 98)
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((mm_all - lo) / (hi - lo), 0, 1)
    norm[depth_arr == 0] = 0
gray = (norm * 255).astype(np.uint8)
rgb = np.stack([gray, gray, gray], axis=-1)
p1 = os.path.join(OUT_DIR, "depth_raw.png")
write_png(p1, rgb)
print(f"    深度图已保存: {p1}")

np.save(os.path.join(OUT_DIR, "depth_raw.npy"), depth_arr)
print(f"    原始数据已保存: {os.path.join(OUT_DIR, 'depth_raw.npy')}")

ob_pipeline_stop(pipe, byref(e))
print("[5] 完成，深度流验证通过 ✅")
