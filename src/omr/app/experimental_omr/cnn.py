"""CNN 符号检测器：**谱表带 → 每类中心热力图 → 符号检出**。

它替换的是检测链路里的**符号级**那一半（符头 / 休止符 / 临时记号 / 附点），
而**谱号 / 拍号 / 小节线仍然走原来的几何方法**（那几项本来就工作得不错，
见 `detect.find_clef` / `find_time_signature` / `find_barlines`）。
这样 A/B 对照就非常干净：**下游 `reconstruct` 一字不改，只换检测器**。

为什么值得换
------------
模板法在这三类上反复失败，而且每修一次都是在给判据打补丁：
  * 符杠白缝被当成空心符头（修复 12）
  * 符杠被当成休止符（修复 13）
  * 符杠被当成实心符头（修复 19，占剩余多出量的 ~20%）
  * 去线阈值跨过符头尺寸 → 符头被整片抹掉（§10.8）
CNN 的两个结构性优势：
  1. **输入是去线之前的原始墨迹**（`dataset` 的设计），不依赖 `remove_staff_lines`；
  2. 判据是**学出来的**，不是"一个阈值对应一种字形"，不会再出现
     "修好 A 就把 B 放进来"的跷跷板。

⚠️ 诚实的前提：这条路要成立，训练集必须覆盖真实照片的分布。
合成页只能覆盖"几何 + 退化"，覆盖不了真正的刻谱字体与透视
（见 `synthdata` 的说明与 §7.1 的合成集盲区）。

用法
----
    from omr.cnn import CnnDetector
    det = CnnDetector("out/cnn/symbolnet.pt")
    page = recognize(img, PipelineConfig(detector="cnn", cnn_weights=...))
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import dataset as ds
from . import detect as dt
from .detect import Detection, GlyphSet
from .staff import StaffGeometry

#: 解码默认参数（在验证集上定，见 `tools/train_symbol_cnn.py --sweep`）
DEFAULT_MIN_SCORE = 0.35
DEFAULT_NMS_RADIUS = 3


def geom_of_staff(staff: StaffGeometry) -> dict:
    """把 `StaffGeometry` 转成 `dataset` 需要的那几个字段。

    ★ 训练用的是 `render.staff_geoms` 的字典，推理用的是这里的 `StaffGeometry` ——
    **两者必须是同一组字段**，否则训练/推理的输入不是一回事。
    （本项目的老毛病：两条路径不同构。所以这里刻意只做一次转换。）
    """
    return {"space": float(staff.space),
            "line_ys": [float(y) for y in staff.line_ys],
            "x0": float(staff.x0),
            "x1": float(staff.x1),
            "clef": str(getattr(staff, "clef", "treble"))}


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------


def build_net(n_out: int = ds.N_CLASS, ch: int = 24):
    """小型 U-Net 风格全卷积网络：输入 1×H×W，输出 `n_out×H/2×W/2`。

    为什么用 FCN 而不是"先切候选再分类"：
      * 候选生成这一步（连通域 / 模板峰）在真实照片上本身就是坏的
        （谱线残渣把整条谱表连成一片，见 §7.1 #11），
        在坏候选上做分类只能补救一部分；
      * FCN 直接在**整条谱表带**上给逐像素判断，不需要候选；
      * 输出 stride 2 已经足够定位（1 space = 16px 输入 → 8px 输出），
        再细只会让显存和推理时间变差。
    """
    import torch
    import torch.nn as nn

    def blk(cin: int, cout: int) -> "nn.Sequential":
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    class SymbolNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # 第一层就 stride=2：输出分辨率是输入的 1/2，
            # 所以最细的那级 skip 也在 1/2 上（不然拼接的尺寸对不上）。
            self.stem = nn.Sequential(
                nn.Conv2d(1, ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            )
            self.down1 = nn.Sequential(nn.MaxPool2d(2), blk(ch, ch * 2))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), blk(ch * 2, ch * 4))
            self.up1 = nn.Sequential(nn.Conv2d(ch * 4 + ch * 2, ch * 2, 3,
                                               padding=1, bias=False),
                                     nn.BatchNorm2d(ch * 2), nn.ReLU(inplace=True))
            self.up0 = nn.Sequential(nn.Conv2d(ch * 2 + ch, ch, 3, padding=1,
                                               bias=False),
                                     nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
            self.head = nn.Conv2d(ch, n_out, 1)

        def forward(self, x):                            # (B,1,H,W)
            e0 = self.stem(x)                            # H/2  ← 输出分辨率
            e1 = self.down1(e0)                          # H/4
            e2 = self.down2(e1)                          # H/8
            u1 = torch.nn.functional.interpolate(
                e2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            u1 = self.up1(torch.cat([u1, e1], dim=1))
            u0 = torch.nn.functional.interpolate(
                u1, size=e0.shape[-2:], mode="bilinear", align_corners=False)
            u0 = self.up0(torch.cat([u0, e0], dim=1))
            return self.head(u0)                         # (B,n_out,H/2,W/2)

    return SymbolNet()


# ---------------------------------------------------------------------------
# 解码：热力图 -> Detection
# ---------------------------------------------------------------------------


def _grid_xy(i: int, stride: int) -> float:
    """输出格点下标 -> 带内像素坐标（取像素中心，避免整体半像素偏移）。

    ⚠️ 这个映射必须在**栅格化目标**和**解码**里是同一个式子，
    而且在 `dataset.rasterize` 里用的是它的逆。半像素偏移看起来无所谓，
    但它会系统地影响 y —— 而 y 决定音高（音高间距只有 0.5 space）。
    """
    return (i + 0.5) * stride - 0.5


def decode_heatmaps(prob: np.ndarray, meta: "ds.BandMeta", *,
                    classes: tuple[str, ...] = ds.CLASSES,
                    min_score: float = DEFAULT_MIN_SCORE,
                    nms_radius: int = DEFAULT_NMS_RADIUS,
                    stride: int = ds.STRIDE) -> list[Detection]:
    """`(N_CLASS, h, w)` 概率图 -> `Detection` 列表（原图坐标）。"""
    out: list[Detection] = []
    space = meta.space
    for ci, cname in enumerate(classes):
        m = prob[ci]
        if m.size == 0:
            continue
        for xo, yo, v in dt._nms_peaks(m, min_score, nms_radius):
            bx = _grid_xy(xo, stride)
            by = _grid_xy(yo, stride)
            # 带内 -> 原图（含半像素修正，见 `BandMeta.back_x` 的说明）
            px = meta.x0 + (bx + 0.5) / meta.scale - 0.5
            py = meta.y0 + (by + 0.5) / meta.scale - 0.5
            bw_s, bh_s = ds.CLASS_BOX_SPACE.get(cname, (1.0, 1.0))
            bw, bh = bw_s * space, bh_s * space
            out.append(Detection(cname, float(px), float(py), float(v),
                                 (float(px - bw / 2), float(py - bh / 2),
                                  float(bw), float(bh)),
                                 extra={"measure": "cnn_heatmap",
                                        "value": float(v)}))
    # 同一位置只留得分最高的那个（与模板法的 `_dedupe_cross_class` 同一意图）
    return dt._dedupe_cross_class(out, space, rx=0.5, ry=0.5)


# ---------------------------------------------------------------------------
# 检测器（实现 detect.Detector 协议）
# ---------------------------------------------------------------------------


class CnnDetector:
    """`Detector` 协议实现：CNN 出符号，几何方法出谱号/拍号/小节线。"""

    name = "cnn"

    def __init__(self, weights: str | Path,
                 device: str = "cpu",
                 min_score: float = DEFAULT_MIN_SCORE,
                 nms_radius: int = DEFAULT_NMS_RADIUS) -> None:
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.min_score = float(min_score)
        self.nms_radius = int(nms_radius)
        ck = torch.load(str(weights), map_location=self.device)
        n_out = int(ck.get("n_out", ds.N_CLASS))
        ch = int(ck.get("ch", 24))
        self.net = build_net(n_out=n_out, ch=ch).to(self.device)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.classes = tuple(ck.get("classes", ds.CLASSES))
        self.space_px = float(ck.get("space_px", ds.SPACE_PX))
        self.stride = int(ck.get("stride", ds.STRIDE))
        self.weights_path = str(weights)

    # -- 推理 ---------------------------------------------------------------
    def _forward_band(self, band: np.ndarray) -> np.ndarray:
        """(H,W) float {0,1} -> (N_CLASS,h,w) 概率。"""
        t = self.torch.from_numpy(band[None, None].astype(np.float32)).to(self.device)
        with self.torch.no_grad():
            logit = self.net(t)
            prob = self.torch.sigmoid(logit)[0].cpu().numpy()
        return prob

    def detect(self, ink_without_staff: np.ndarray, staff: StaffGeometry,
               binary: np.ndarray | None = None) -> GlyphSet:
        gs = GlyphSet(staff=staff, ink=ink_without_staff)
        space = staff.space

        # --- 几何那一半（与模板法**共用同一批函数**）------------------------
        clef_name, clef_score, clef_bbox = dt.find_clef(ink_without_staff, staff)
        gs.clef, gs.clef_score = clef_name, clef_score
        ts_x = (clef_bbox[0] + clef_bbox[2] + 0.35 * space) if clef_bbox \
            else (staff.x0 + 3.0 * space)
        ts, ts_score, ts_bbox = dt.find_time_signature(ink_without_staff, staff, ts_x)
        gs.time_signature, gs.ts_score = ts, ts_score
        music_x0 = (ts_bbox[0] + ts_bbox[2] + 0.5 * space) if ts_bbox \
            else (ts_x + 2.6 * space)
        off_staff: list[float] = []
        gs.barlines = dt.find_barlines(ink_without_staff, staff, binary=binary,
                                       rejected=off_staff)
        gs.barlines_off_staff = off_staff

        # --- 符号那一半（CNN）----------------------------------------------
        # ★ 输入要**去线之前**的原始墨迹：CNN 的任务之一就是自己忽略谱线。
        #   拿不到 binary 时退回 `ink_without_staff` —— 但那条路会让
        #   "训练时看到的"和"推理时看到的"不一致，所以只当兜底并在结果里标注。
        src = binary if binary is not None else ink_without_staff
        band, meta = ds.extract_band(src, geom_of_staff(staff),
                                     space_px=self.space_px)
        prob = self._forward_band(band)
        dets = decode_heatmaps(prob, meta, classes=self.classes,
                               min_score=self.min_score,
                               nms_radius=self.nms_radius, stride=self.stride)
        gs.detections = [d for d in dets if d.x >= music_x0]
        return gs
