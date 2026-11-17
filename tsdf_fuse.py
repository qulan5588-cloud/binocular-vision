"""tsdf_fuse.py —— 可选的 Open3D TSDF 融合后端（app_scan 默认不启用）

app_scan 默认的「2cm 体素去重」是零依赖的：点云直接累积，随时可回溯、可离线
重渲。它的短板是只有表面点、没有法线/网格，而且噪声不做平均——薄墙（门板、
隔断）扫两遍会糊成两层。

这里给一个可选后端：同样的深度帧 + 位姿喂给 Open3D 的 VoxelBlockGrid（TSDF），
在 GPU 上做截断符号距离的加权融合，能导出：

    * 点云（带颜色，取等值面 tsdf=0，比原始点更薄更准）
    * 三角网格 mesh.ply（可量面积/体积，可用 MeshLab 看）
    * 法线

代价是依赖重：open3d 0.20 的 Windows CUDA wheel（open3d_cuda，127MB）+ pip 的
nvidia-* 运行库（cublas/cusolver/cusparse/npp，合计约 1.1GB）。所以：

    * 默认关，`python app_scan.py --tsdf` 才用
    * 没装 open3d / 没有 N 卡 → 自动退化到 CPU 版 VoxelBlockGrid（慢，但能用）
    * 完全没装 open3d → 抛 TsdfUnavailable，主程序打印一句就去用默认后端

用法：
    from tsdf_fuse import TsdfFuse, available, TsdfUnavailable
    ts = TsdfFuse(voxel=0.02)
    ts.integrate(depth_m, K, G_cam2world, color_bgr)   # 每帧一次
    P, C = ts.cloud()
    ts.save_mesh("out/mesh.ply")
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


class TsdfUnavailable(RuntimeError):
    """没有可用的 open3d（没装 / DLL 缺）"""


_O3D = None
_DEVICE = None


def load(verbose=False):
    """导入 open3d 并把 pip 装的 CUDA 运行库目录挂到 DLL 搜索路径上

    open3d_cuda 的 Open3D.dll 静态依赖 cublas64_12 / cusolver64_11 /
    npp*64_12，这些 DLL 在 site-packages/nvidia/*/bin 里，pip 不会自动加到
    PATH，所以必须自己 add_dll_directory，否则 import 直接 DLL load failed。
    """
    global _O3D, _DEVICE
    if _O3D is not None:
        return _O3D
    bins = glob.glob(os.path.join(sys.prefix, "Lib", "site-packages",
                                  "nvidia", "*", "bin"))
    for d in bins:
        try:
            os.add_dll_directory(d)
        except OSError:
            pass
    try:
        import open3d as o3d
    except Exception as e:                        # noqa: BLE001
        raise TsdfUnavailable(f"open3d 不可用：{type(e).__name__}: {e}") from e
    _O3D = o3d
    if verbose:
        print(f"open3d {o3d.__version__}  CUDA={o3d.core.cuda.is_available()}")
    return o3d


def pick_device():
    """优先 CUDA:0；没有 CUDA 版就用 CPU:0"""
    o3d = load()
    return "CUDA:0" if o3d.core.cuda.is_available() else "CPU:0"


class TsdfFuse:
    """一帧一帧地 integrate；随时可导出点云/网格"""

    def __init__(self, voxel=0.02, device=None, block_count=60000,
                 depth_max=6.0, trunc=8.0, weight=1.0):
        o3d = load()
        o3c = o3d.core
        from open3d.t.geometry import VoxelBlockGrid

        self.o3d = o3d
        self.o3c = o3c
        self.voxel = float(voxel)
        self.depth_max = float(depth_max)
        self.weight = float(weight)    # 提取阈值：默认 3.0 会把只融合过 1-2 帧的
                                       # 体素全滤掉（扫描前期基本没点），降到 1.0
        self.trunc = float(trunc)
        self.device = o3c.Device(device or pick_device())
        self._vbg = VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1,), (1,), (3,)),
            voxel_size=self.voxel, block_resolution=8,
            block_count=int(block_count), device=self.device)
        self.frames = 0
        self._t = 0.0

    # ---------------------------------------------------------- 融合 --
    def integrate(self, depth_m, K, G, color_bgr=None):
        """depth_m: (H,W) 米；K: 3x3（对应 depth_m 分辨率）；G: 相机→世界 4x4

        color_bgr 可选，(H,W,3) uint8，必须已经对齐到深度图（app_scan 里用
        Aligner 生成的 proj 重采样好再传进来）。
        """
        o3d, o3c = self.o3d, self.o3c
        dev = self.device
        if color_bgr is None:
            # 纯深度融合：float32 + depth_scale=1.0
            z = np.ascontiguousarray(depth_m, np.float32)
            scale = 1.0
        else:
            # 彩色融合：Open3D 只接受 (uint16 mm, uint8) 这一对组合，
            # (Float32, UInt8) 会直接抛 Unsupported input data type combination
            z = np.clip(np.asarray(depth_m, np.float32) * 1000.0, 0,
                        65535).astype(np.uint16)
            scale = 1000.0
        # 0.20 的 API 收 t.geometry.Image，不是裸 Tensor
        dt = o3d.t.geometry.Image(o3c.Tensor(z, device=dev))
        intr = o3c.Tensor(np.asarray(K, np.float64), device=dev)
        ext = o3c.Tensor(np.linalg.inv(np.asarray(G, np.float64)).astype(np.float64),
                         device=dev)
        blocks = self._vbg.compute_unique_block_coordinates(
            dt, intr, ext, scale, self.depth_max, self.trunc)
        if color_bgr is None:
            self._vbg.integrate(blocks, dt, intr, ext, scale, self.depth_max,
                                self.trunc)
        else:
            rgb = np.ascontiguousarray(color_bgr[:, :, ::-1], np.uint8)
            ct = o3d.t.geometry.Image(o3c.Tensor(rgb, device=dev))
            self._vbg.integrate(blocks, dt, ct, intr, ext, scale,
                                self.depth_max, self.trunc)
        if "CUDA" in str(dev):
            o3c.cuda.synchronize()
        self.frames += 1

    # ---------------------------------------------------------- 导出 --
    def cloud(self):
        """等值面点云：返回 (points[N,3] float32 世界系, colors[N,3] uint8 RGB)"""
        o3c = self.o3c
        pcd = self._vbg.extract_point_cloud(
            weight_threshold=self.weight).to(o3c.Device("CPU"))
        P = pcd.point.positions.numpy().astype(np.float32)
        if "colors" in pcd.point:
            C = np.clip(pcd.point.colors.numpy() * 255.0, 0, 255).astype(np.uint8)
        else:
            C = np.full((len(P), 3), 200, np.uint8)
        return P, C

    def mesh(self):
        """三角网格（顶点带颜色），返回 o3d.t.geometry.TriangleMesh（CPU）"""
        return self._vbg.extract_triangle_mesh(
            weight_threshold=self.weight).to(self.o3c.Device("CPU"))

    def save_mesh(self, path):
        m = self.mesh()
        ok = self.o3d.t.io.write_triangle_mesh(path, m, write_vertex_colors=True)
        if not ok:
            raise RuntimeError(f"写网格失败：{path}")
        return path

    def stats(self):
        return dict(backend=f"tsdf {self.device}",
                    voxel=self.voxel, frames=self.frames)


def available():
    """能不能用（不抛异常），给 app_scan 判断要不要提示"""
    try:
        load()
        return True, pick_device()
    except Exception as e:                        # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    ok, info = available()
    print(("TSDF 可用，设备 " + info) if ok else ("TSDF 不可用：" + info))
