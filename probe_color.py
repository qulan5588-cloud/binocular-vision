# -*- coding: utf-8 -*-
"""探测 OpenNI2 是否支持 COLOR 流，以及可用分辨率/格式"""
import ctypes, os, sys
from ctypes import c_void_p, c_int, c_char_p, c_uint16, c_uint64, POINTER, byref, Structure

LIB_DIR = r"D:\双目视觉\_driver\openni2\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs"
os.add_dll_directory(LIB_DIR)

class OniVideoMode(Structure):
    _fields_ = [("pixelFormat", c_int), ("resolutionX", c_int),
                ("resolutionY", c_int), ("fps", c_int)]

class OniFrame(Structure):
    _fields_ = [("dataSize", c_int), ("data", c_void_p), ("sensorType", c_int),
                ("timestamp", c_uint64), ("frameIndex", c_int), ("width", c_int),
                ("height", c_int), ("videoMode", OniVideoMode), ("croppingEnabled", c_int),
                ("cropOriginX", c_int), ("cropOriginY", c_int), ("stride", c_int)]

class OniDeviceInfo(Structure):
    _fields_ = [("uri", ctypes.c_char * 256), ("vendor", ctypes.c_char * 256),
                ("name", ctypes.c_char * 256), ("serialNumber", ctypes.c_char * 256),
                ("usbVendorId", c_uint16), ("usbProductId", c_uint16)]

lib = ctypes.CDLL(os.path.join(LIB_DIR, "OpenNI2.dll"))
def sig(n, r, a):
    f = getattr(lib, n); f.restype = r; f.argtypes = a; return f

oniInitialize = sig("oniInitialize", c_int, [c_int])
oniGetDeviceList = sig("oniGetDeviceList", c_int, [POINTER(POINTER(OniDeviceInfo)), POINTER(c_int)])
oniDeviceOpen = sig("oniDeviceOpen", c_int, [c_char_p, POINTER(c_void_p)])
oniDeviceCreateStream = sig("oniDeviceCreateStream", c_int, [c_void_p, c_int, POINTER(c_void_p)])
oniStreamStart = sig("oniStreamStart", c_int, [c_void_p])
oniStreamStop = sig("oniStreamStop", c_int, [c_void_p])
oniStreamReadFrame = sig("oniStreamReadFrame", c_int, [c_void_p, POINTER(POINTER(OniFrame))])
oniStreamGetProperty = sig("oniStreamGetProperty", c_int, [c_void_p, c_int, c_void_p, POINTER(c_int)])
oniFrameRelease = sig("oniFrameRelease", None, [POINTER(OniFrame)])

print("oniInitialize:", oniInitialize(2003))
plist = POINTER(OniDeviceInfo)(); cnt = c_int(0)
oniGetDeviceList(byref(plist), byref(cnt))
print("设备数:", cnt.value)
if cnt.value == 0:
    sys.exit(1)
uri = plist[0].uri
dev = c_void_p()
print("open:", oniDeviceOpen(uri, byref(dev)))

for st_name, st_val in [("DEPTH", 3), ("COLOR", 2), ("IR", 1)]:
    s = c_void_p()
    rc = oniDeviceCreateStream(dev, st_val, byref(s))
    print(f"\n--- {st_name} (type={st_val}) createStream rc={rc} handle={s.value}")
    if rc != 0 or not s.value:
        continue
    vm = OniVideoMode(); sz = c_int(ctypes.sizeof(OniVideoMode))
    rc2 = oniStreamGetProperty(s, 3, byref(vm), byref(sz))
    print(f"    默认模式 rc={rc2}: {vm.resolutionX}x{vm.resolutionY} @{vm.fps} fmt={vm.pixelFormat}")
    rc3 = oniStreamStart(s)
    print(f"    start rc={rc3}")
    if rc3 == 0:
        pf = POINTER(OniFrame)()
        rc4 = oniStreamReadFrame(s, byref(pf))
        if rc4 == 0 and pf:
            f = pf.contents
            print(f"    首帧: {f.width}x{f.height} size={f.dataSize} stride={f.stride} fmt={f.videoMode.pixelFormat}")
            oniFrameRelease(pf)
        else:
            print(f"    读帧失败 rc={rc4}")
        oniStreamStop(s)
