# -*- coding: utf-8 -*-
"""
Orbbec Astra Pro Plus 深度流验证（OpenNI2 + ctypes，无需 GUI / 编译）
"""
import ctypes, os, sys, time, zlib, struct
from ctypes import c_void_p, c_int, c_char_p, c_uint16, c_uint64, POINTER, byref, Structure
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "_detect")
os.makedirs(OUT_DIR, exist_ok=True)
# 复用 capture_pair.py 的动态查找逻辑定位 OpenNI2.dll
sys.path.insert(0, BASE)
import capture_pair as _cp  # noqa: E402
LIB_DIR = _cp.ONI_LIB_DIR
os.add_dll_directory(LIB_DIR)

ONI_API_VERSION = 2003
ONI_SENSOR_DEPTH = 3
ONI_SENSOR_COLOR = 2
ONI_STREAM_PROPERTY_VIDEO_MODE = 3
ONI_MAX_STR = 256


class OniVideoMode(Structure):
    _fields_ = [("pixelFormat", c_int),
                ("resolutionX", c_int),
                ("resolutionY", c_int),
                ("fps", c_int)]


class OniFrame(Structure):
    _fields_ = [("dataSize", c_int),
                ("data", c_void_p),
                ("sensorType", c_int),
                ("timestamp", c_uint64),
                ("frameIndex", c_int),
                ("width", c_int),
                ("height", c_int),
                ("videoMode", OniVideoMode),
                ("croppingEnabled", c_int),
                ("cropOriginX", c_int),
                ("cropOriginY", c_int),
                ("stride", c_int)]


class OniDeviceInfo(Structure):
    _fields_ = [("uri", ctypes.c_char * ONI_MAX_STR),
                ("vendor", ctypes.c_char * ONI_MAX_STR),
                ("name", ctypes.c_char * ONI_MAX_STR),
                ("serialNumber", ctypes.c_char * ONI_MAX_STR),
                ("usbVendorId", c_uint16),
                ("usbProductId", c_uint16)]


lib = ctypes.CDLL(os.path.join(LIB_DIR, "OpenNI2.dll"))

def sig(name, restype, argtypes):
    f = getattr(lib, name); f.restype = restype; f.argtypes = argtypes; return f

oniInitialize       = sig("oniInitialize", c_int, [c_int])
oniGetVersion       = sig("oniGetVersion", c_char_p, [])
oniGetDeviceList    = sig("oniGetDeviceList", c_int, [POINTER(POINTER(OniDeviceInfo)), POINTER(c_int)])
oniDeviceOpen       = sig("oniDeviceOpen", c_int, [c_char_p, POINTER(c_void_p)])
oniDeviceCreateStream = sig("oniDeviceCreateStream", c_int, [c_void_p, c_int, POINTER(c_void_p)])
oniStreamStart      = sig("oniStreamStart", c_int, [c_void_p])
oniStreamStop       = sig("oniStreamStop", c_int, [c_void_p])
oniStreamReadFrame  = sig("oniStreamReadFrame", c_int, [c_void_p, POINTER(POINTER(OniFrame))])
oniStreamGetProperty = sig("oniStreamGetProperty", c_int, [c_void_p, c_int, c_void_p, POINTER(c_int)])
oniFrameRelease     = sig("oniFrameRelease", None, [POINTER(OniFrame)])


def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) \
          + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    open(path, "wb").write(png)


print("=" * 62)
print(" Orbbec Astra Pro Plus 深度流验证 (OpenNI2)")
print("=" * 62)

st = oniInitialize(ONI_API_VERSION)
print("[0] oniInitialize ->", st, "(0=成功)  OpenNI2 版本:", oniGetVersion().decode())
if st != 0:
    sys.exit(1)

print("[1] 枚举设备 ...")
plist = POINTER(OniDeviceInfo)()
cnt = c_int(0)
st = oniGetDeviceList(byref(plist), byref(cnt))
print("    oniGetDeviceList ->", st, " 设备数:", cnt.value)
if cnt.value == 0:
    print("    !! 没有枚举到任何 OpenNI2 设备")
    sys.exit(1)

target = None
for i in range(cnt.value):
    d = plist[i]
    uri = d.uri.decode(errors="ignore")
    name = d.name.decode(errors="ignore")
    vend = d.vendor.decode(errors="ignore")
    sn = d.serialNumber.decode(errors="ignore")
    print(f"    设备[{i}] name='{name}' vendor='{vend}' SN='{sn}' VID:PID={d.usbVendorId:04X}:{d.usbProductId:04X}")
    print(f"           uri={uri}")
    if d.usbProductId in (0x060F, 0x060f) or "060F" in uri.upper():
        target = d
if target is None:
    target = plist[0]
    print("    (未匹配 060F，使用第一个设备)")

print("[2] 打开设备 ...")
dev = c_void_p()
st = oniDeviceOpen(target.uri, byref(dev))
print("    oniDeviceOpen ->", st, " handle=", dev)
if st != 0 or not dev:
    sys.exit(1)

print("[3] 创建深度流 ...")
dep = c_void_p()
st = oniDeviceCreateStream(dev, ONI_SENSOR_DEPTH, byref(dep))
print("    oniDeviceCreateStream(DEPTH) ->", st)
if st != 0 or not dep:
    sys.exit(1)

vm = OniVideoMode()
sz = c_int(ctypes.sizeof(OniVideoMode))
st = oniStreamGetProperty(dep, ONI_STREAM_PROPERTY_VIDEO_MODE, byref(vm), byref(sz))
print(f"    当前深度模式: {vm.resolutionX}x{vm.resolutionY} @{vm.fps}fps  pixelFormat={vm.pixelFormat} (100=mm, 101=0.1mm)")

st = oniStreamStart(dep)
print("    oniStreamStart ->", st)
if st != 0:
    sys.exit(1)

print("[4] 读取深度帧 ...")
frames = []
t0 = time.time()
first = None
for k in range(30):
    pf = POINTER(OniFrame)()
    st = oniStreamReadFrame(dep, byref(pf))
    if st != 0 or not pf:
        print("    读帧失败:", st); break
    f = pf.contents
    if k == 0:
        w, h = f.width, f.height
        first = (w, h, f.videoMode.pixelFormat, f.dataSize, f.stride)
        buf = ctypes.string_at(f.data, f.dataSize)
        arr = np.frombuffer(buf, dtype=np.uint16).reshape(h, w).copy()
        frames.append(arr)
        print(f"    首帧: {w}x{h} dataSize={f.dataSize} stride={f.stride} pixelFormat={f.videoMode.pixelFormat}")
    if k == 29:
        print(f"    末帧 index={f.frameIndex} timestamp={f.timestamp}")
    oniFrameRelease(pf)
dt = time.time() - t0
print(f"    读帧结束: 共 {k+1} 帧, 用时 {dt:.2f}s -> {(k+1)/dt:.1f} fps")

oniStreamStop(dep)

if not frames:
    print("[!] 未取到深度帧"); sys.exit(2)

depth = frames[0]
pfmt = first[2]
scale = 0.1 if pfmt == 101 else 1.0   # mm
print("[5] 深度数据分析 ...")
valid = depth[depth > 0]
print(f"    分辨率     : {depth.shape[1]}x{depth.shape[0]}")
print(f"    有效像素   : {valid.size}/{depth.size} ({valid.size/depth.size*100:.1f}%)")
if valid.size:
    mm = valid.astype(np.float64) * scale
    print(f"    原始值范围 : {depth.min()} ~ {depth.max()}")
    print(f"    物理距离   : {mm.min():.0f} mm ~ {mm.max():.0f} mm  中位 {np.median(mm):.0f} mm")
    h, w = depth.shape
    c = depth[h//2-20:h//2+20, w//2-20:w//2+20].astype(np.float64) * scale
    cv = c[c > 0]
    if cv.size:
        print(f"    画面中心   : {np.median(cv):.0f} mm")

mm_all = depth.astype(np.float64) * scale
norm = np.zeros_like(mm_all)
if valid.size:
    lo, hi = np.percentile(valid.astype(np.float64)*scale, 2), np.percentile(valid.astype(np.float64)*scale, 98)
    if hi <= lo: hi = lo + 1
    norm = np.clip((mm_all - lo) / (hi - lo), 0, 1)
    norm[depth == 0] = 0
gray = (norm * 255).astype(np.uint8)
p1 = os.path.join(OUT_DIR, "depth_openni.png")
write_png(p1, np.stack([gray, gray, gray], -1))
np.save(os.path.join(OUT_DIR, "depth_openni.npy"), depth)
print(f"    深度图   : {p1}")
print(f"    原始数据 : {os.path.join(OUT_DIR, 'depth_openni.npy')}")
print("[6] 完成 —— 深度流可用 ✅")
