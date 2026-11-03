"""符号检测数据集：**把谱表带切成固定尺度的输入 + 生成中心热力图目标**。

设计要点（每一条都是踩过坑之后的取舍）
--------------------------------------
1. **输入用"去线之前"的二值图**，不用 `remove_staff_lines` 的输出。
   理由：修复 12/19 与 §10.8 三次都出在"去线"这一步上
   （线粗估计错 → 符头被抹掉；符杠被当符头）。让 CNN 直接看**原始墨迹**，
   它自己学会忽略谱线 —— 等于把"去线"这个最脆弱的环节从检测链路里拿掉。
   ⚠️ 代价：模型必须真的学到"谱线不算字"，这要在训练集里出现过各种线粗/模糊。

2. **按 space 归一化尺度**（`SPACE_PX` 像素 = 1 space）。
   真实照片的 space 是 19.5~24.9px、合成页可以 16~32px ——
   不归一化的话，模型要同时学"字形"和"尺度"两件事，数据量根本不够。

3. **目标是每类一张中心热力图**（输出 stride 2），不是框回归。
   中心点对"符头/休止符/临时记号/附点"这类**紧凑字形**是够的，
   解码直接复用 `detect._nms_peaks`（与模板法同一条后处理路径，便于对照）。
   框大小第一版用**各类字形的物理尺寸常数**补齐（见 `CLASS_BOX_SPACE`）。

4. **标签只取 `GlyphBox`**（字形级真值），不是 `Event.bbox`
   （那是"符头+符干+临时记号"的并集，会把符干训成符头）。

用法（在 omr/ 下）：
    python -m tools.probe_cnn_data --n 4 --out out/cnn/data_demo
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 预测的类别（**索引 0 固定是背景**，与热力图通道一一对应）
CLASSES: tuple[str, ...] = (
    "notehead_filled",
    "notehead_hollow",
    "rest_whole",
    "rest_half",
    "rest_quarter",
    "accidental_sharp",
    "accidental_flat",
    "accidental_natural",
    "augmentation_dot",
)
N_CLASS = len(CLASSES)
CLS_TO_IDX = {c: i + 1 for i, c in enumerate(CLASSES)}

#: 各类字形的**物理尺寸**（单位 space，宽 × 高）—— 从 `render.py` 的绘制代码反推。
#: 第一版不做框回归，解码时用它把中心点补成框。
CLASS_BOX_SPACE: dict[str, tuple[float, float]] = {
    "notehead_filled": (1.18, 0.95),
    "notehead_hollow": (1.18, 0.95),
    "rest_whole": (1.15, 0.48),
    "rest_half": (1.15, 0.48),
    "rest_quarter": (0.68, 2.40),
    "accidental_sharp": (1.01, 1.90),
    "accidental_flat": (1.01, 1.90),
    "accidental_natural": (1.01, 1.90),
    "augmentation_dot": (0.30, 0.30),
}

#: 归一化尺度：1 space 映射成多少像素（训练与推理**必须一致**）
SPACE_PX = 16.0
#: 谱表上下各留多少 space（要装得下加线、符干、符杠、连音线）
PAD_SPACE = 2.6
#: 输出相对输入的步长
STRIDE = 2
#: 热力图高斯半径（输出像素）
SIGMA_PX = 2.0


@dataclass
class BandMeta:
    """把"带内坐标"映回"原图坐标"所需的一切。"""

    x0: float
    y0: float
    scale: float          # 带内像素 / 原图像素
    space: float          # 原图上的谱间距（像素）
    w_in: int
    h_in: int

    def back_x(self, x_out: float) -> float:
        return self.x0 + x_out / self.scale

    def back_y(self, y_out: float) -> float:
        return self.y0 + y_out / self.scale


def band_geometry(geom: dict, img_shape: tuple[int, int], space_px: float = SPACE_PX,
                  pad_space: float = PAD_SPACE
                  ) -> tuple[int, int, int, int, float]:
    """算出该谱表要裁的矩形与缩放系数。

    返回 `(x0, y0, x1, y1, scale)`（原图坐标，右/下开区间）。
    """
    h, w = img_shape[:2]
    space = float(geom["space"])
    lines = [float(y) for y in geom["line_ys"]]
    y0 = int(np.floor(min(lines) - pad_space * space))
    y1 = int(np.ceil(max(lines) + pad_space * space))
    x0 = int(np.floor(float(geom["x0"]) - 0.6 * space))
    x1 = int(np.ceil(float(geom["x1"]) + 0.6 * space))
    # 裁剪带要**整段按同一尺度**缩放，所以先夹到图像内，再算 scale
    y0c, y1c = max(0, y0), min(h, y1)
    x0c, x1c = max(0, x0), min(w, x1)
    if y1c <= y0c or x1c <= x0c:
        return 0, 0, 1, 1, 1.0
    # ⚠️ scale 必须按**未裁剪的带高**算：否则裁到页边的谱表会被放大成另一个尺度，
    #    训练/推理就会看到两种尺度（这是最容易悄悄发生的一类不一致）。
    scale = space_px / max(1e-6, space)
    return x0, y0, x1, y1, scale


def extract_band(binary: np.ndarray, geom: dict, space_px: float = SPACE_PX,
                 pad_space: float = PAD_SPACE,
                 x_window: tuple[int, int] | None = None
                 ) -> tuple[np.ndarray, BandMeta]:
    """裁出谱表带并缩放到固定尺度。

    Parameters
    ----------
    binary
        去线**之前**的二值图（bool）。
    x_window
        可选的**列窗口**（原图坐标），用来随机取一段（训练时当增广用）。
        若给出，则只保留与该窗口相交的部分，并在输出里平移。
    """
    h, w = binary.shape[:2]
    x0, y0, x1, y1, scale = band_geometry(geom, binary.shape, space_px, pad_space)
    if x_window is not None:
        x0, x1 = max(x0, int(x_window[0])), min(x1, int(x_window[1]))
    # 先把"带"补成完整矩形（越界的部分留白），保证 scale 与几何一致
    bx0, by0 = max(0, x0), max(0, y0)
    bx1, by1 = min(w, x1), min(h, y1)
    sub = np.zeros((max(1, y1 - y0), max(1, x1 - x0)), dtype=np.float32)
    if bx1 > bx0 and by1 > by0:
        sub[(by0 - y0):(by1 - y0), (bx0 - x0):(bx1 - x0)] = \
            binary[by0:by1, bx0:bx1].astype(np.float32)

    out_h = max(1, int(round(sub.shape[0] * scale)))
    out_w = max(1, int(round(sub.shape[1] * scale)))
    band = _resize_bool(sub, out_h, out_w)
    meta = BandMeta(x0=float(x0), y0=float(y0), scale=float(scale),
                    space=float(geom["space"]), w_in=out_w, h_in=out_h)
    return band, meta


def _resize_bool(a: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """最近邻缩放（**不用插值**：二值图插值会造出灰度台阶，等于凭空加噪声）。"""
    if a.shape[0] == out_h and a.shape[1] == out_w:
        return a.astype(np.float32, copy=False)
    yi = (np.arange(out_h) * (a.shape[0] / out_h)).astype(np.int64)
    xi = (np.arange(out_w) * (a.shape[1] / out_w)).astype(np.int64)
    yi = np.clip(yi, 0, a.shape[0] - 1)
    xi = np.clip(xi, 0, a.shape[1] - 1)
    return a[yi][:, xi].astype(np.float32, copy=False)


def _gauss(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    yy = np.arange(h, dtype=np.float32)[:, None] - cy
    xx = np.arange(w, dtype=np.float32)[None, :] - cx
    d2 = yy * yy + xx * xx
    return np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)


def rasterize_points(points, meta: BandMeta, out_h: int, out_w: int,
                     stride: int = STRIDE, sigma: float = SIGMA_PX,
                     classes: tuple[str, ...] = CLASSES) -> np.ndarray:
    """把一串 **(类别名, 原图 x, 原图 y)** 栅格化成热力图目标。

    ★ 单独抽出来的原因：**半监督/自训练**要用"别的检测器给出的框"当目标
    （见 `tools/selftrain_cnn.py`），那些框不是 `render.GlyphBox`，
    但它们必须和渲染真值走**完全同一套栅格化**，否则"合成训出来的"和
    "真实微调用的"就是两种目标，模型会被两套口径来回拉扯。
    """
    oh = max(1, out_h // stride)
    ow = max(1, out_w // stride)
    tgt = np.zeros((len(classes), oh, ow), dtype=np.float32)
    for cls, px, py in points:
        ci = CLS_TO_IDX.get(cls)
        if ci is None:
            continue
        bx = (px - meta.x0) * meta.scale
        by = (py - meta.y0) * meta.scale
        if not (-4 <= bx <= out_w + 4 and -4 <= by <= out_h + 4):
            continue
        ox = (bx + 0.5) / stride - 0.5
        oy = (by + 0.5) / stride - 0.5
        y0 = max(0, int(oy - 4 * sigma))
        y1 = min(oh, int(oy + 4 * sigma) + 1)
        x0 = max(0, int(ox - 4 * sigma))
        x1 = min(ow, int(ox + 4 * sigma) + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        blob = _gauss(y1 - y0, x1 - x0, ox - x0, oy - y0, sigma)
        ch = tgt[ci - 1, y0:y1, x0:x1]
        np.maximum(ch, blob, out=ch)
    return tgt


def rasterize(glyphs: list, meta: BandMeta, out_h: int, out_w: int,
              stride: int = STRIDE, sigma: float = SIGMA_PX,
              classes: tuple[str, ...] = CLASSES) -> np.ndarray:
    """把该谱表的字形框转成 `(N_CLASS, out_h/stride, out_w/stride)` 的热力图目标。

    ★ **不设"背景"通道**：每个类各一张中心热力图，"这里没有东西"由
    "所有通道都低"来表达。多一个"背景"通道会让它在损失里占掉一大块
    （它几乎处处为正），反而把真正要学的类压下去。
    """
    return rasterize_points([(g.cls, g.cx, g.cy) for g in glyphs],
                            meta, out_h, out_w, stride=stride, sigma=sigma,
                            classes=classes)


def glyphs_of_staff(glyphs: list, si: int) -> list:
    return [g for g in glyphs if g.staff == si]


def jitter_stroke(band: np.ndarray, rng, p: float = 0.5) -> np.ndarray:
    """随机把笔画**加粗/变细 1 像素**（增广）。

    ★ 为什么必须做：真实手机翻拍上，谱线粗细约 **0.17 space**（4px @ space 23），
    而 `render._lw` 画出来的是 **0.11 space**（2~3px）。
    也就是说**合成页的笔画天生比真实照片细 1.5 倍** ——
    模型在细笔画上学到的"墨迹形状"，遇到粗笔画会偏移，
    而且这类差异在合成留出集上**永远测不出来**（留出集也一样细）。

    直接在**归一化之后**的带上做形态学膨胀/腐蚀（1px ≈ 0.06 space），
    正好补上这一档差距，而且**不影响标签坐标**（字形的中心不动）。
    """
    import numpy as np
    if rng.random() >= p:
        return band
    b = band > 0.5
    grow = bool(rng.random() < 0.5)
    nh = np.zeros_like(b)
    nv = np.zeros_like(b)
    # 4 邻域膨胀 / 腐蚀（用移位实现，不依赖 cv2）
    for sh, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        if axis == 0:
            n = np.roll(b, sh, axis=0)
            n[0 if sh > 0 else -1, :] = b[0 if sh > 0 else -1, :]
        else:
            n = np.roll(b, sh, axis=1)
            n[:, 0 if sh > 0 else -1] = b[:, 0 if sh > 0 else -1]
        if grow:
            nh |= n
            nv |= n
        else:
            nh &= n
            nv &= n
    out = (nh | nv) if grow else (nh & nv)
    return out.astype(np.float32)


def make_sample(res, si: int, *, space_px: float = SPACE_PX,
                x_window: tuple[int, int] | None = None,
                stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray, BandMeta]:
    """从一页里取出第 `si` 条谱表的 (输入, 目标, 元信息)。

    ⚠️ 训练与推理**必须都调这一个函数**：输入怎么裁、怎么缩放、目标怎么栅格化，
    只有一处实现 —— 否则训出来的模型和线上看到的输入不是一回事。
    """
    from . import preprocess as pp
    geom = res.staff_geoms[si]
    # 与流水线一致地做二值化（合成灰度图 → 二值）
    binary = pp.preprocess(res.image, pp.PreprocessConfig()).binary
    band, meta = extract_band(binary, geom, space_px=space_px,
                              x_window=x_window)
    tgt = rasterize(glyphs_of_staff(res.glyphs, si), meta, meta.h_in, meta.w_in,
                    stride=stride)
    return band, tgt, meta
