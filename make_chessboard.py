# -*- coding: utf-8 -*-
"""
生成棋盘格标定板 PDF（精确物理尺寸，可直接打印）
------------------------------------------------------------------
用法:
    python make_chessboard.py                          # 默认 9x6 内角点, 25mm 方格
    python make_chessboard.py --cols 9 --rows 6 --square 25 --paper A4L
    python make_chessboard.py --measured 95            # 打印机缩到95%，自动补偿放大
    python make_chessboard.py --paper A3L --square 40

输出:
    chessboard_<cols>x<rows>_<square>mm.pdf            打印时务必选「实际大小/100%缩放」
    chessboard_<cols>x<rows>_<square>mm_cal95.pdf      补偿版（--measured 时生成）
说明:
    "9x6 内角点" = 横向 10 个方格 x 纵向 7 个方格
    PDF 内部以毫米精确绘制（1pt = 25.4/72 mm），并附带 100mm 校验尺

--measured 用法:
    打印一版后用尺子量底部校验尺。若量得 95mm 而不是 100mm，说明打印机
    把整页缩到了 95%（常见原因：页面高 210mm 而可打印高只有约 200mm，
    驱动自动缩小适配）。传 --measured 95 后本脚本把内容整体放大 100/95，
    并在校验尺标注放大后的长度，打印出来量到的正好是 100mm。
"""
import argparse
import os

MM2PT = 72.0 / 25.4

PAPER = {
    "A4L": (297.0, 210.0),
    "A4P": (210.0, 297.0),
    "A3L": (420.0, 297.0),
    "A3P": (297.0, 420.0),
}

# 版面参数（单位 mm）。底部留给校验尺与说明文字，顶部少量余量。
MARGIN_BOTTOM = 18.0
MARGIN_TOP = 6.0
RULER_Y = 8.5          # 校验尺基线到页底的距离
NOTE_Y = 2.2           # 说明文字基线到页底的距离


def layout(cols, rows, square_mm, paper, factor=1.0):
    """算出版面几何。坐标统一用 mm，输出时再换算成 pt。

    factor: 补偿系数。打印机会缩小多少，就预先放大多少。
    返回 (pw, ph, ox, oy, bw, bh, sq, ruler_len, ruler_y)
    """
    pw, ph = PAPER[paper]
    sq = square_mm * factor
    bw, bh = (cols + 1) * sq, (rows + 1) * sq       # 棋盘外框
    avail_h = ph - MARGIN_BOTTOM - MARGIN_TOP
    if bw + 12 > pw or bh > avail_h:
        raise SystemExit(
            f"棋盘 {bw:.1f} x {bh:.1f} mm 放不进 {paper} 的 "
            f"{pw:.0f} x {ph:.0f} mm（可用 {pw - 12:.0f} x {avail_h:.0f} mm）。\n"
            f"请减小 --square，或改用 --paper A3L / A3P。")
    ox = (pw - bw) / 2.0                          # 水平居中
    oy = MARGIN_BOTTOM + (avail_h - bh) / 2.0     # 垂直：在可用区内居中
    ruler_len = 100.0 * factor                    # 打印被缩小后正好 100mm
    return pw, ph, ox, oy, bw, bh, sq, ruler_len, RULER_Y


def build_pdf(path, cols, rows, square_mm, paper, factor=1.0, measured=None):
    pw, ph, ox, oy, bw, bh, sq, rl, ruler_y = layout(
        cols, rows, square_mm, paper, factor)

    P = lambda v: v * MM2PT  # mm -> pt

    ops = [f"1 1 1 rg 0 0 {P(pw):.2f} {P(ph):.2f} re f"]
    # 黑色方格：(col+row) 为偶数的格子涂黑（左下角为原点）
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (c + r) % 2 == 0:
                x, y, s = P(ox + c * sq), P(oy + r * sq), P(sq)
                ops.append(f"0 0 0 rg {x:.2f} {y:.2f} {s:.2f} {s:.2f} re f")
    # 外框细线（便于裁剪对齐）
    ops.append(f"0.3 0.3 0.3 RG 0.5 w {P(ox - 3):.2f} {P(oy - 3):.2f} "
               f"{P(bw + 6):.2f} {P(bh + 6):.2f} re S")

    # 校验尺（水平，位于棋盘下方）。画的是放大后的长度，打印缩回后正好 100mm。
    rx = (pw - rl) / 2.0
    ops.append(f"0 0 0 RG 1 w {P(rx):.2f} {P(ruler_y):.2f} m "
               f"{P(rx + rl):.2f} {P(ruler_y):.2f} l S")
    for i in range(11):  # 每 10mm 一个刻度
        t = rl / 10 * i
        h = 6.0 if i % 5 == 0 else 3.5
        ops.append(f"{P(rx + t):.2f} {P(ruler_y):.2f} m "
                   f"{P(rx + t):.2f} {P(ruler_y + h):.2f} l S")
    # 0 / 100mm 数字标注（右端文字右对齐，避免出界）
    ops.append(f"BT /F1 7 Tf 1 0 0 1 {P(rx) - 2:.2f} {P(ruler_y + 7):.2f} Tm (0) Tj ET")
    ops.append(f"BT /F1 7 Tf 1 0 0 1 {P(rx + rl) - 21:.2f} {P(ruler_y + 7):.2f} "
               f"Tm (100mm) Tj ET")

    # 说明文字（必须纯 ASCII：PDF 里用的是 Type1 Helvetica）
    if measured and abs(factor - 1.0) > 1e-6:
        note = (f"COMPENSATED for {measured:g}% print scale "
                f"(content scaled {factor:.4f}x). After printing, the ruler "
                f"below must measure exactly 100 mm. "
                f"{cols}x{rows} inner corners, square = {square_mm:g} mm.")
    else:
        note = (f"{cols}x{rows} inner corners, square = {square_mm:g} mm. "
                f"Print at 100% scale (Actual size), "
                f"then verify the ruler below measures exactly 100 mm.")
    ops.append(f"BT /F1 7 Tf 0.15 0.15 0.15 rg 1 0 0 1 {P(18):.2f} {P(NOTE_Y):.2f} "
               f"Tm ({note}) Tj ET")

    content = "\n".join(ops).encode("latin-1", "replace")

    # ---- 组装 PDF 文件结构 ----
    objs = []
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")                             # 1
    objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")                     # 2
    objs.append(                                                                  # 3
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {P(pw):.2f} {P(ph):.2f}] "
        f"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode())
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")        # 4
    objs.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")  # 5

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()

    with open(path, "wb") as f:
        f.write(bytes(out))
    return bw, bh, sq


def make_preview(path, cols, rows, square_mm, paper, factor=1.0, measured=None, dpi=60):
    """生成整页预览 PNG（按 PDF 同样的版式渲染，仅供肉眼确认）"""
    import cv2
    import numpy as np
    pw, ph, ox, oy, bw, bh, sq, rl, ruler_y = layout(
        cols, rows, square_mm, paper, factor)
    n = dpi / 25.4  # 每毫米像素数
    W, H = int(pw * n), int(ph * n)
    img = np.full((H, W), 255, np.uint8)

    px = lambda v: int(round(v * n))          # mm -> 像素
    # 黑色方格（PDF 的 y 向上，预览 y 向下，需翻转）
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (c + r) % 2 == 0:
                x0 = px(ox + c * sq)
                y0 = px(ph - (oy + (r + 1) * sq))
                s = max(1, px(sq))
                img[y0:y0 + s, x0:x0 + s] = 0
    cv2.rectangle(img, (px(ox - 3), px(ph - (oy + bh + 3))),
                  (px(ox + bw + 3), px(ph - (oy - 3))), 90, 1)
    # 校验尺
    ry = px(ph - ruler_y)
    rx, rl_px = px((pw - rl) / 2.0), px(rl)
    cv2.line(img, (rx, ry), (rx + rl_px, ry), 0, 1)
    for i in range(11):
        t = rl_px // 10 * i
        h = px(6.0) if i % 5 == 0 else px(3.5)
        cv2.line(img, (rx + t, ry), (rx + t, ry - h), 0, 1)
    cv2.putText(img, "0", (rx - 4, ry - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 0, 1)
    cv2.putText(img, "100mm", (rx + rl_px - 30, ry - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, 0, 1)
    tag = f" [compensated x{factor:.3f} for {measured:g}% scale]" if measured else ""
    cv2.putText(img, f"{cols}x{rows} corners, square={square_mm:g}mm{tag}",
                (px(14), px(ph - NOTE_Y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 40, 1)

    ok, buf = cv2.imencode(".png", img)
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="生成棋盘格标定板 PDF")
    ap.add_argument("--cols", type=int, default=9, help="横向内角点数")
    ap.add_argument("--rows", type=int, default=6, help="纵向内角点数")
    ap.add_argument("--square", type=float, default=25.0, help="方格边长 mm")
    ap.add_argument("--paper", default="A4L", choices=list(PAPER))
    ap.add_argument("--measured", type=float, default=None,
                    help="打印后校验尺实测长度 mm（如 95）。给了就按 100/实测 补偿放大")
    args = ap.parse_args()

    factor = 1.0
    if args.measured is not None:
        if not (50.0 <= args.measured <= 130.0):
            raise SystemExit(f"--measured {args.measured} 不合理，应在 50~130 之间")
        factor = 100.0 / args.measured

    suffix = f"_cal{args.measured:g}" if args.measured else ""
    pdf = os.path.join(here, f"chessboard_{args.cols}x{args.rows}_{args.square:g}mm{suffix}.pdf")
    png = os.path.join(here, f"chessboard_{args.cols}x{args.rows}_{args.square:g}mm{suffix}_preview.png")
    bw, bh, sq = build_pdf(pdf, args.cols, args.rows, args.square, args.paper,
                           factor, args.measured)
    make_preview(png, args.cols, args.rows, args.square, args.paper,
                 factor, args.measured)

    print(f"棋盘外框: {bw:.1f} x {bh:.1f} mm   纸张: {args.paper}")
    if args.measured:
        print(f"补偿: 实测 {args.measured:g}mm -> 放大 {factor:.4f} 倍")
        print(f"      文件上方格 = {sq:.3f} mm（打印后应回到 {args.square:g} mm）")
        print(f"      文件上校验尺 = {100 * factor:.2f} mm（打印后应量得 100 mm）")
    print(f"PDF  : {pdf}")
    print(f"预览 : {png}")
    print("打印时选「实际大小 / 100% 缩放」，打印后用尺子核对底部 100mm 校验尺。")


if __name__ == "__main__":
    main()
