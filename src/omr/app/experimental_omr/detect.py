"""
符号检测 —— 从"去掉谱线的二值图"里找出各种音乐符号的位置。

⚠️ 关于检测器的定位（务必先读）
------------------------------
这里实现的是 **Baseline A：几何/模板法检测器**。

  优点：零训练成本、纯确定性、参数可解释、能立刻在合成图上跑通并建立精度基线。
  缺点：只对干净、标准排版的图像有效。**真实拍摄场景必然需要 CNN。**

所以这个模块的真正价值不是"能用"，而是：
  1. 把 `Detector` 接口固定下来，让 YOLO 能直接替换进来（见 `OnnxDetector`）；
  2. 提供一条**无需标注数据就能量化全链路**的回归基线；
  3. 把"检测"和"语义重建"的边界划清楚 —— 前者可下沉 DPU，后者只能留 PS。

三个关键设计决策（都是踩过坑之后定下来的）
------------------------------------------
1. **实心符头用"圆盘内墨迹占比"，空心符头用"环占比 − λ×孔内占比"。**
   一开始我用"圆盘占比 ≥ 0.78 判实心、否则判空心"，结果符尾(flag)的
   圆盘占比 ≈ 0.40、空心符头 ≈ 0.54，两者太近，符尾被大量误判成空心符头。
   "孔内墨迹占比"才是真正的判别量：空心符头的孔是空的，符尾没有孔。
2. **小节线判据必须用"精确谱表高度覆盖率 ≥ 0.92"。**
   符干也有 3.4/4.0 = 85% 的覆盖率，用 0.80 的阈值会把每根符干都当成小节线
   —— 实测直接导致小节数从 4 变成 14，后面全错。
3. **拍号搜索窗口要按谱号字形的实际右边界定位，不能用模板中心。**
   模板中心 ≠ 字形中心（字形在模板里是偏心的），差出 1.5 个 space 就足以
   让窗口切掉半个数字，进而匹配出垃圾结果。

对应架构：本模块整体属于可下沉到 PL/DPU 的部分；
         而 reconstruct.py 属于必须留在 PS 的部分。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .render import (ACC_W, NOTEHEAD_H, NOTEHEAD_W, MUSIC_CP, _draw_accidental,
                     _draw_rest, _draw_treble_clef, _font_size_for_height,
                     _glyph_mask, music_font_path)
from .staff import StaffGeometry, horizontal_run_length

# ---------------------------------------------------------------------------
# 符号类别
# ---------------------------------------------------------------------------

SYM_NOTEHEAD_FILLED = "notehead_filled"
SYM_NOTEHEAD_HOLLOW = "notehead_hollow"
SYM_REST_WHOLE = "rest_whole"
SYM_REST_HALF = "rest_half"
SYM_REST_QUARTER = "rest_quarter"
SYM_ACC_SHARP = "accidental_sharp"
SYM_ACC_FLAT = "accidental_flat"
SYM_ACC_NATURAL = "accidental_natural"
SYM_DOT = "augmentation_dot"
SYM_CLEF_TREBLE = "clef_treble"

ACCIDENTAL_TO_SYM = {
    "sharp": SYM_ACC_SHARP,
    "flat": SYM_ACC_FLAT,
    "natural": SYM_ACC_NATURAL,
}
SYM_TO_ACCIDENTAL = {v: k for k, v in ACCIDENTAL_TO_SYM.items()}


@dataclass
class Detection:
    """一个被检出的符号。坐标是原图像素（页面坐标系）。"""

    symbol: str
    x: float
    y: float
    score: float                       # 0..1，检测置信度
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.symbol}@({self.x:.0f},{self.y:.0f}) s={self.score:.2f}"


@dataclass
class GlyphSet:
    """一页的原始符号检测结果（尚未赋予乐理含义）。"""

    detections: list[Detection] = field(default_factory=list)
    barlines: list[float] = field(default_factory=list)
    clef: str = "treble"
    clef_score: float = 0.0
    time_signature: tuple[int, int] | None = None
    ts_score: float = 0.0
    clef_bbox: tuple[float, float, float, float] | None = None
    raw_accidentals: list[Detection] = field(default_factory=list)
    key_fifths: int | None = None
    key_score: float = 0.0
    staff: StaffGeometry | None = None
    #: 去掉谱线后的墨迹图 —— 语义重建阶段做符头/符干/符尾关联时要用
    ink: np.ndarray | None = None
    #: ★ 被判为"符杠白缝"而**丢弃**的空心符头候选（详见 `_is_beam_gap`）。
    #:   必须留下来：万一判据误杀，这是唯一能看见"丢了什么"的地方 ——
    #:   本项目多次吃亏于"静默丢弃"（工具/实现不一致那一类坑）。
    beam_gaps: list[Detection] = field(default_factory=list)
    #: ★ 被判为"不坐落在谱表上"而**丢弃**的小节线候选（详见 `_staff_side_hits`）。
    barlines_off_staff: list[float] = field(default_factory=list)
    #: ★ 被判为"尺寸不像附点"而**丢弃**的附点候选（详见 `_is_dot_sized`）。
    dots_big_blob: list[Detection] = field(default_factory=list)
    #: ★ 被判为"跨谱表重复检出"而**丢弃**的幽灵（详见 `drop_cross_staff_ghosts`）。
    ghosts: list[Detection] = field(default_factory=list)

    def by_symbol(self, sym: str) -> list[Detection]:
        return [d for d in self.detections if d.symbol == sym]

    def noteheads(self) -> list[Detection]:
        return [d for d in self.detections
                if d.symbol in (SYM_NOTEHEAD_FILLED, SYM_NOTEHEAD_HOLLOW)]


# ---------------------------------------------------------------------------
# 模板相关性
# ---------------------------------------------------------------------------


def _correlate(ink: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    """模板内的墨迹占比图（**整图**版本，只在小图上用）。

    用 filter2D 做相关（OpenCV 的 filter2D 实际是相关而非卷积，
    正好是我们想要的方向，模板不会被翻转）。

    ⚠️ 大图上不要用这个，用 `_correlate_sub` + `_crop_for`。
    实测在一张 1780x1506 的多谱表页上，全图相关让 `find_time_signature`
    一个函数就吃掉 612ms（10 个数字模板 × 全图 filter2D），
    而它真正要看的只是谱号右侧一小块。整条链路因此慢 14 倍。
    """
    if cv2 is None:
        raise RuntimeError("模板相关性需要 OpenCV")
    n = int(tmpl.sum())
    if n <= 0:
        return np.zeros_like(ink, dtype=np.float32)
    k = (tmpl.astype(np.float32) / n)
    return cv2.filter2D(ink.astype(np.float32), -1, k,
                        borderType=cv2.BORDER_CONSTANT)


def _staff_roi(staff: StaffGeometry, shape: tuple[int, int],
               pad_space: float = 8.0) -> tuple[int, int, int, int]:
    """一个谱表的搜索区域：整宽 + 谱表上下各留若干 space。

    留 margin 是因为加线、符干、连线都会伸到谱表外。
    """
    h, w = shape
    pad = pad_space * staff.space
    return (0, int(max(0, staff.y_top_line - pad)),
            w, int(min(h, staff.y_bottom_line + pad)))


def _crop_for(ink: np.ndarray, roi: tuple[int, int, int, int],
              tmpl: np.ndarray) -> tuple[np.ndarray, int, int]:
    """按 roi（再外扩半个模板尺寸）裁剪，返回 (子图, x 偏移, y 偏移)。

    ROI 之外的结果我们不关心，所以不计算 —— 这是纯优化，不改变任何结果。
    """
    th, tw = tmpl.shape
    x0, y0, x1, y1 = roi
    x0 = int(max(0, x0 - tw // 2))
    y0 = int(max(0, y0 - th // 2))
    x1 = int(min(ink.shape[1], x1 + tw // 2 + 1))
    y1 = int(min(ink.shape[0], y1 + th // 2 + 1))
    if x1 <= x0 or y1 <= y0:
        return ink[0:0, 0:0], 0, 0
    return ink[y0:y1, x0:x1], x0, y0


def _correlate_sub(sub: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    """在子图上做模板相关。语义与 `_correlate` 完全一致。"""
    if cv2 is None:
        raise RuntimeError("模板相关性需要 OpenCV")
    n = int(tmpl.sum())
    if n <= 0 or sub.size == 0:
        return np.zeros(sub.shape, dtype=np.float32)
    k = (tmpl.astype(np.float32) / n)
    return cv2.filter2D(sub.astype(np.float32), -1, k,
                        borderType=cv2.BORDER_CONSTANT)


def _nms_peaks(score: np.ndarray, min_score: float, radius: int,
               max_n: int = 4096) -> list[tuple[int, int, float]]:
    """在得分图上做非极大值抑制，返回 [(x, y, score), ...]，按 score 降序。"""
    s = score.copy()
    out: list[tuple[int, int, float]] = []
    r = max(1, int(radius))
    for _ in range(max_n):
        idx = int(np.argmax(s))
        y, x = divmod(idx, s.shape[1])
        v = float(s[y, x])
        if v < min_score:
            break
        out.append((x, y, v))
        y0, y1 = max(0, y - r), min(s.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(s.shape[1], x + r + 1)
        s[y0:y1, x0:x1] = -1e9
    return out


def _tmpl_from_draw(w: int, h: int, fn) -> np.ndarray:
    """把 render.py 里的字形绘制函数渲染成 bool 模板（True = 墨迹）。"""
    from PIL import Image, ImageDraw
    img = Image.new("L", (max(3, w), max(3, h)), 255)
    d = ImageDraw.Draw(img)
    fn(d)
    return np.asarray(img) < 128


def _glyph_bbox_in_image(peak_x: int, peak_y: int, tmpl: np.ndarray
                         ) -> tuple[float, float, float, float]:
    """由相关峰位置反推字形在原图中的包围盒。

    filter2D 把模板锚在中心，所以模板左上角 = (peak_x - tw/2, peak_y - th/2)；
    再叠加字形在模板内部的重心/包围盒偏移，就得到字形在原图中的位置。
    """
    th, tw = tmpl.shape
    ys, xs = np.nonzero(tmpl)
    if xs.size == 0:
        return (float(peak_x), float(peak_y), 0.0, 0.0)
    x0 = peak_x - tw / 2.0 + float(xs.min())
    y0 = peak_y - th / 2.0 + float(ys.min())
    return (x0, y0, float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))


def _dedupe_cross_class(dets: list[Detection], space: float,
                        rx: float = 0.7, ry: float = 0.7) -> list[Detection]:
    """跨类别去重：同一个小区域里只保留得分最高的那个。

    必需的一步。因为不同类别的模板会互相"蹭"到对方的墨迹
    （典型例子：四分休止符的折线会同时蹭到全休止符模板，如果只对
    每个类别各自做 NMS，就会在同一个位置输出两个休止符）。
    """
    keep: list[Detection] = []
    for d in sorted(dets, key=lambda x: -x.score):
        if any(abs(d.x - k.x) <= rx * space and abs(d.y - k.y) <= ry * space
               for k in keep):
            continue
        keep.append(d)
    keep.sort(key=lambda x: x.x)
    return keep


def _dedupe_accidentals(dets: list[Detection], space: float) -> list[Detection]:
    """Retain a clear sharp candidate when flat/natural templates overlap it.

    On printed sharps the simplified flat template can score slightly higher
    than the sharp template at the same physical symbol.  The margin is small,
    so only apply this tie break to a strong sharp response in one cluster.
    """
    for sharp in dets:
        if sharp.symbol != SYM_ACC_SHARP or sharp.score < 0.68:
            continue
        rivals = [other for other in dets if other.symbol != SYM_ACC_SHARP and
                  abs(other.x - sharp.x) <= 0.6 * space and
                  abs(other.y - sharp.y) <= 0.6 * space]
        if rivals and max(r.score for r in rivals) - sharp.score <= 0.10:
            sharp.score = min(1.0, sharp.score + 0.08)
    kept = _dedupe_cross_class(dets, space, rx=0.6, ry=0.6)
    # Preserve close alternatives for semantic reading.  Deleting the runner
    # up made a softly photographed sharp indistinguishable from a natural,
    # even when both templates independently responded to the same glyph.
    for item in kept:
        if item.symbol != SYM_ACC_NATURAL:
            continue
        rivals = [other for other in dets if other.symbol == SYM_ACC_SHARP
                  and abs(other.x - item.x) <= 0.30 * space
                  and abs(other.y - item.y) <= 0.40 * space]
        if rivals:
            runner = max(rivals, key=lambda other: other.score)
            if runner.score >= 0.65 and item.score - runner.score <= 0.06:
                item.extra["ambiguous_sharp_score"] = runner.score
    return kept


# ---------------------------------------------------------------------------
# 符头
# ---------------------------------------------------------------------------


def _limit_to_roi(score: np.ndarray, ox: int, oy: int,
                  roi: tuple[int, int, int, int]) -> np.ndarray:
    """把子图得分限制回 roi 内（ROI 外置为极小值）。

    `_crop_for` 会把裁剪范围外扩半个模板尺寸，好让贴着 ROI 边缘的模板
    也能看到完整上下文。但**峰值只能出现在 ROI 以内**，
    否则等于悄悄把搜索范围放大了 —— 语义就变了。
    """
    x0, y0, x1, y1 = roi
    out = np.full(score.shape, -1e9, dtype=score.dtype)
    ax0 = max(0, x0 - ox)
    ay0 = max(0, y0 - oy)
    ax1 = min(score.shape[1], x1 - ox)
    ay1 = min(score.shape[0], y1 - oy)
    if ax1 > ax0 and ay1 > ay0:
        out[ay0:ay1, ax0:ax1] = score[ay0:ay1, ax0:ax1]
    return out


def _bounding_ink(ink: np.ndarray, gx: int, gy: int, dir_y: int,
                  limit: float) -> tuple[float, float]:
    """从 (gx, gy) 沿 ±y 找**第一块墨**，返回 (横向游程, 沿射线厚度)。

    游程是"垂直于射线"方向的连续墨长，厚度是"沿射线"方向的连续墨长。
    两者都以像素返回，由调用方按 space 归一化。

    ★ 这个函数的物理含义：**空心符头的孔是被"自己的环"包住的**，
    所以从孔心往外遇到的第一块墨必然是环的弧 —— 又短又薄。
    若遇到的是又长又厚的横条，那这个"孔"就不是符头的孔。
    """
    h, w = ink.shape
    y = gy + dir_y
    steps = 0
    max_steps = int(limit)
    while 0 <= y < h and steps < max_steps and not ink[y, gx]:
        y += dir_y
        steps += 1
    if not (0 <= y < h) or steps >= max_steps or not ink[y, gx]:
        return 0.0, 0.0            # 该方向在 limit 内没有墨：不构成"被包住"
    x0 = gx
    while x0 - 1 >= 0 and ink[y, x0 - 1]:
        x0 -= 1
    x1 = gx
    while x1 + 1 < w and ink[y, x1 + 1]:
        x1 += 1
    run = float(x1 - x0 + 1)
    t = 0
    yy = y
    while 0 <= yy < h and ink[yy, gx]:
        t += 1
        yy += dir_y
    return run, float(t)


def _is_beam_gap(ink: np.ndarray, gx: int, gy: int, space: float,
                 run_beam: float, thk_beam: float, thk_hard: float,
                 run_hard: float, reach: float) -> bool:
    """判断一个"空心符头"候选其实是**符杠之间的白缝**。

    ★ 这是本项目最后一个、也是最隐蔽的一类伪检测。
    在密集十六分音符的符杠音群里，相邻符杠之间有一条白缝，
    左右被**符干**封住 —— 于是它局部看起来就是一个"环内空白"的图形，
    `ring − λ·hole` 分数高达 0.6，稳稳通过 `hollow_thr`。
    实测 101910 的一个谱表里因此凭空冒出 13 个"二分音符"，
    再叠加附点就成了刷屏的 `1440` tick。

    为什么不能靠调 `hollow_thr` 解决
    --------------------------------
    在合成图（21 space）上量过：**真空心符头的分数只有 0.69**，
    而符杠白缝有 0.55~0.69 —— 两者**直接重叠**。
    原因有二：(a) 环模板与实际字形有半像素级几何差；
    (b) 坐在谱线旁的符头，其上弧会被去线工序啃掉。
    所以"提高阈值"必然两边都伤，必须换一个**结构判据**。

    判据（全部按 space 归一化，可跨分辨率迁移）
    ------------------------------------------
    真空心符头（22 个合成标定样本 + 2 个真实照片实测，全部人眼确认）：
        run ≤ 1.00 space，thk ≤ 0.25 space
    真符杠白缝 / 谱号环（真实照片实测，人眼逐个确认）：
        run 最大 20.4 space，thk 最大 2.0 space —— 两者至少一项远超上界

    于是四条并列（任一成立即判为符杠白缝）：
      1. 上下**两侧**都是长横条      -> 白缝被两根符杠夹住
      2. 一侧"又长又厚"              -> 复用修复 4 的符杠判据（长 且 厚）
      3. 一侧厚度 ≥ thk_hard (0.5)   -> 真符头的上下弧实测 ≤0.25，2 倍余量
      4. 一侧长度 > run_hard (3.0)   -> 真符头的上下弧实测 ≤1.00，3 倍余量

    ⚠️ 为什么不能用"单侧长度 > 1.2 space"这种更紧的判据
    ----------------------------------------------------
    实测有一个真符头的弧长正好是 **1.00 space**，而某个假候选是 **1.09 space**
    —— 直接重叠。所以紧阈值必然两边都伤。**必须靠"厚度"和"超长"这两个
    与真符头差 2~3 倍的量来分开，而把它们做成 OR 的并列条件。**

    ⚠️ 已知残余风险（合成集**测不出来**，必须记住）
    -------------------------------------------------
    `render.py` **不画加线**（`find_ledger_lines` 是死代码），所以合成集里
    根本没有"加线 / 连音线 / 重音记号出现在符头上下"的场景。
    若某个真符头的环弧被去线工序啃掉，而上下恰好有长横条
    （加线、连音线、谱线残渣），就会被误杀成符杠白缝。
    真实照片上暂无这样的样本，**上板前应补一份带加线的合成集**。

    ⚠️ 本函数**不负责**"谱号环"这一类误检
    ------------------------------------
    真实照片上另有 3 个假空心落在**高音谱号的圈里**（谱号本身就是个空心环）。
    它们的局部几何与真符头**完全无法区分**（run 1.09 / thk 0.25，全在真符头范围内）。
    根因是别处：`find_clef` 被**书页边缘的暗带**骗走，返回了错包围盒，
    于是 `music_x0` 太小、谱号没被排除 —— 见 `find_clef` 的说明。
    """
    if space <= 0:
        return False
    r_u, t_u = _bounding_ink(ink, gx, gy, -1, reach * space)
    r_d, t_d = _bounding_ink(ink, gx, gy, +1, reach * space)
    ru, rd, tu, td = r_u / space, r_d / space, t_u / space, t_d / space
    rmax, tmax = max(ru, rd), max(tu, td)
    if ru > run_beam and rd > run_beam:
        return True                      # 1. 两侧都被长横条封住
    if rmax > run_beam and tmax >= thk_beam:
        return True                      # 2. "又长又厚" = 符杠
    if tmax >= thk_hard:
        return True                      # 3. 厚到不可能是环弧
    if rmax > run_hard:
        return True                      # 4. 长到不可能是环弧
    return False


def _is_beam_bar(ink: np.ndarray, gx: int, gy: int, space: float,
                 bar_run: float = 2.5, bar_thk: float = 0.80) -> bool:
    """候选处的墨迹是否"**又长又薄**"—— 也就是符杠 / 谱线残渣，而不是符头。

    ★★ 这是"实心符头"分支缺的那一条尺寸判据（**PC 侧独有**，
    PL 侧不做符号检测，所以不涉及跨三层）。

    机理：`find_noteheads` 的实心判据只有"**圆盘模板内墨迹占比 ≥ 0.80**"。
    一条符杠是**实心长条**，一个 `1.18×0.95 space` 的圆盘可以**整个落在墨里**
    → 得分 1.00，与真符头**完全一样**；再叠上 NMS（半径 0.55 space），
    一条 4 space 长的符杠会给出 **3~4 个**"实心符头"。
    这与修复 12（符杠白缝）、修复 13（符杠被当休止符）是**同一根因的第三个出口**。

    实测（`tools/probe_noteheads.py`，**有真值的密集符杠对照页**）：

        真符头 56 个 / 检出 104 个 → **假阳性 48 个**，全部是符杠
        假阳性：run = **3.20 space**（= 符杠长度）、厚 = **0.53~0.57 space**（= 梁厚）
        真符头：厚 = 0.96~1.10 space（符头高 0.95）

    真实照片（865 个检出）的 run×厚 交叉表里，`run ≥ 2.5 且 厚 < 0.80` 一格有
    **70 个**（8%）—— 这些就是符杠/谱线残渣。

    ⚠️ **不能只用 run 判**：密集段落里真符头会互相接触、还连着符杠，
    run 可以到 8~17 space（实测真实照片上真符头的 run 最大 21.8）。
    真正的分界量是**厚度**：实心符头 ≈ 0.95 space 高，符杠只有 0.3~0.6 space 厚。
    ⚠️ 阈值也不能压到 1.5 space：真实谱面上的符杠只有 **2.10~2.33 space** 长
    （修复 13 实测），压太低会连带把"符头 + 邻接符干"一起误杀。
    """
    a, b = _h_run_span(ink, gy, gx)
    if a < 0:
        return False
    if (b - a + 1) / space < bar_run:
        return False
    return _v_extent(ink, gy, gx, int(round(space))) / space <= bar_thk


def _hollow_font_evidence(binary: np.ndarray, cx: float, cy: float,
                          font_mask: np.ndarray | None) -> float:
    """Compare the original ink around a head with an engraved hollow rim."""
    if font_mask is None:
        return 0.0
    mh, mw = font_mask.shape
    best = 0.0
    for dx in (-4, -2, 0, 2, 4):
        for dy in (-4, -2, 0, 2, 4):
            xx, yy = int(round(cx + dx - mw / 2)), int(round(cy + dy - mh / 2))
            if xx < 0 or yy < 0 or xx + mw > binary.shape[1] or yy + mh > binary.shape[0]:
                continue
            patch = binary[yy:yy + mh, xx:xx + mw]
            on = float(patch[font_mask].mean())
            off = float(patch[~font_mask].mean())
            best = max(best, on - 0.5 * off)
    return best


def _hollow_stem_evidence(ink: np.ndarray | None, cx: float, cy: float,
                          space: float) -> bool:
    """Check for a long stroke adjoining either side of a hollow head."""
    if ink is None:
        return False
    h, w = ink.shape
    for side in (-1, 1):
        for offset in np.arange(0.40, 0.81, 0.08):
            x = int(round(cx + side * offset * space))
            if not 0 <= x < w:
                continue
            for direction in (-1, 1):
                ya = int(round(cy + direction * 0.5 * space))
                yb = int(round(cy + direction * 4.0 * space))
                lo, hi = max(0, min(ya, yb)), min(h, max(ya, yb))
                if hi > lo and np.count_nonzero(ink[lo:hi, x]) >= 2.5 * space:
                    return True
    return False


def find_hollow_notehead_holes(binary: np.ndarray, staff: StaffGeometry,
                               ink_without_staff: np.ndarray | None = None) -> list[Detection]:
    """Find hollow heads from enclosed paper regions in the original binary image.

    The ring template misses tilted engraving fonts and holes crossed by staff
    lines.  A closed paper component is much less sensitive to the outer font
    shape.  The size gate keeps clef loops, accidentals, and barline gaps out.
    Two small diagonal holes may be joined when a ledger line splits a head.
    """
    if cv2 is None:
        return []
    space = staff.space
    x0, y0, x1, y1 = _staff_roi(staff, binary.shape, pad_space=6.0)
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(binary.shape[1], int(x1)), min(binary.shape[0], int(y1))
    if x1 <= x0 or y1 <= y0:
        return []
    paper = (~binary[y0:y1, x0:x1]).astype(np.uint8)
    _, _, stats, centers = cv2.connectedComponentsWithStats(paper, 8)
    small: list[tuple[float, float, int, int, int]] = []
    out: list[Detection] = []
    tw = NOTEHEAD_W * space
    th = NOTEHEAD_H * space
    font_mask = None
    if music_font_path():
        cp = MUSIC_CP["notehead_hollow"]
        font_mask, _, _ = _glyph_mask(cp, _font_size_for_height(cp, th))

    def font_evidence(cx: float, cy: float) -> float:
        """Outer head-shape check; enclosed holes also occur inside sharps/flags."""
        return _hollow_font_evidence(binary, cx, cy, font_mask)

    def make_head(cx: float, cy: float, confidence: float, kind: str) -> Detection:
        return Detection(SYM_NOTEHEAD_HOLLOW, cx, cy, confidence,
                         (cx - tw / 2, cy - th / 2, tw, th),
                         extra={"measure": kind, "value": confidence})

    def stem_evidence(cx: float, cy: float) -> bool:
        return _hollow_stem_evidence(ink_without_staff, cx, cy, space)

    for i in range(1, len(stats)):
        bx, by, width, height, area = (int(v) for v in stats[i])
        if bx == 0 or by == 0 or bx + width >= x1 - x0 or by + height >= y1 - y0:
            continue
        cx, cy = float(centers[i, 0] + x0), float(centers[i, 1] + y0)
        if (0.52 <= width / space <= 1.05 and
                0.53 <= height / space <= 0.85 and
                0.18 <= area / (space * space) <= 0.44):
            # A single enclosed white patch is also common in accidentals,
            # flags, and slurs.  Require the *outside* to match a hollow head.
            shape = font_evidence(cx, cy)
            if shape >= 0.74 or (shape >= 0.68 and stem_evidence(cx, cy)):
                out.append(make_head(cx, cy, 0.90, "enclosed_hole"))
        elif (0.30 <= width / space <= 0.75 and
              0.25 <= height / space <= 0.55 and
              0.06 <= area / (space * space) <= 0.25):
            small.append((cx, cy, width, height, area))

    for i, (ax, ay, aw, ah, aa) in enumerate(small):
        for bx, by, bw, bh, ba in small[i + 1:]:
            dx, dy = abs(ax - bx) / space, abs(ay - by) / space
            if not (0.35 <= dx <= 0.80 and 0.20 <= dy <= 0.60):
                continue
            total_w = abs(ax - bx) + (aw + bw) / 2
            total_h = abs(ay - by) + (ah + bh) / 2
            if not (0.85 <= total_w / space <= 1.50 and
                    0.55 <= total_h / space <= 1.25):
                continue
            weight = aa + ba
            out.append(make_head((ax * aa + bx * ba) / weight,
                                 (ay * aa + by * ba) / weight,
                                 0.88, "split_enclosed_hole"))

    # Some photographed whole notes have a pale upper/lower rim.  Thresholding
    # leaves two compact black lobes instead of an enclosed white component.
    # Require symmetric, isolated lobes with the full width of a whole head;
    # stems and adjacent filled heads are taller or have different areas.
    if ink_without_staff is not None:
        _, _, black_stats, black_centers = cv2.connectedComponentsWithStats(
            ink_without_staff[y0:y1, x0:x1].astype(np.uint8), 8)
        lobes: list[tuple[float, float, int, int, int]] = []
        for i in range(1, len(black_stats)):
            bx, by, width, height, area = (int(v) for v in black_stats[i])
            if bx == 0 or by == 0 or bx + width >= x1 - x0 or by + height >= y1 - y0:
                continue
            if (0.75 <= width / space <= 1.25 and
                    0.80 <= height / space <= 1.50 and
                    0.35 <= area / (space * space) <= 0.85):
                lobes.append((float(black_centers[i, 0] + x0),
                              float(black_centers[i, 1] + y0), width, height, area))
        for i, (ax, ay, aw, ah, aa) in enumerate(lobes):
            for bx, by, bw, bh, ba in lobes[i + 1:]:
                if not (0.95 <= abs(bx - ax) / space <= 1.60 and
                        abs(by - ay) / space <= 0.25 and
                        max(aw, bw) / min(aw, bw) <= 1.25 and
                        max(ah, bh) / min(ah, bh) <= 1.25 and
                        max(aa, ba) / min(aa, ba) <= 1.35):
                    continue
                cx, cy = (ax + bx) / 2, (ay + by) / 2
                if not any(abs(cx - h.x) <= space and abs(cy - h.y) <= space
                           for h in out):
                    out.append(make_head(cx, cy, 0.86, "split_whole_lobes"))
    return _dedupe_cross_class(out, space, rx=0.6, ry=0.6)


def _recover_stacked_hollow_heads(
        accepted: list[Detection], beam_gap_rejected: list[Detection],
        binary: np.ndarray, ink: np.ndarray, space: float,
        font_mask: np.ndarray) -> list[Detection]:
    """Recover a rim rejected because the other hollow chord tone looks thick."""
    recovered: list[Detection] = []
    for det in beam_gap_rejected:
        if det.score < 0.40:
            continue
        if not any(
                head.symbol == SYM_NOTEHEAD_HOLLOW
                and head.score >= 0.40
                and abs(head.x - det.x) <= 0.25 * space
                and 0.85 * space <= abs(head.y - det.y) <= 1.15 * space
                for head in accepted):
            continue
        if (_hollow_font_evidence(binary, det.x, det.y, font_mask) < 0.68
                or not _hollow_stem_evidence(ink, det.x, det.y, space)):
            continue
        det.extra["recovered_stacked_hollow"] = True
        recovered.append(det)
    return recovered


def _probe_stacked_hollow_heads(
        accepted: list[Detection], ring_scores: np.ndarray,
        ox: int, oy: int, binary: np.ndarray, ink: np.ndarray,
        space: float, font_mask: np.ndarray,
        seeds: list[Detection] | None = None) -> list[Detection]:
    """Probe one staff space from a strong hollow head for a missed chord tone."""
    recovered: list[Detection] = []
    h, w = ring_scores.shape
    radius = max(1, int(round(.15 * space)))
    for head in list(accepted if seeds is None else seeds):
        if head.symbol != SYM_NOTEHEAD_HOLLOW or head.score < .45:
            continue
        for direction in (-1, 1):
            expected_y = head.y + direction * space
            if any(other.symbol == SYM_NOTEHEAD_HOLLOW
                   and abs(other.x - head.x) <= .35 * space
                   and abs(other.y - expected_y) <= .35 * space
                   for other in accepted + recovered):
                continue
            cx, cy = int(round(head.x)) - ox, int(round(expected_y)) - oy
            x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
            y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            patch = ring_scores[y0:y1, x0:x1]
            py, px = np.unravel_index(int(np.argmax(patch)), patch.shape)
            score = float(patch[py, px])
            if score < .34:
                continue
            x, y = float(px + x0 + ox), float(py + y0 + oy)
            font_evidence = _hollow_font_evidence(binary, x, y, font_mask)
            if (not ((head.score >= .48 and score >= .34 and font_evidence >= .65)
                     or (score >= .40 and font_evidence >= .63))
                    or not _hollow_stem_evidence(ink, x, y, space)):
                continue
            tw = NOTEHEAD_W * space
            th = NOTEHEAD_H * space
            recovered.append(Detection(
                SYM_NOTEHEAD_HOLLOW, x, y, score,
                (x - tw / 2, y - th / 2, tw, th),
                extra={"measure": "stacked_hollow_probe", "value": score}))
    return recovered


def find_noteheads(ink: np.ndarray, staff: StaffGeometry,
                   filled_thr: float = 0.80,
                   hollow_thr: float = 0.55,
                   lambda_hole: float = 1.0,
                   reject_beam_gap: bool = True,
                   beam_run: float = 1.5,
                   beam_thk: float = 0.35,
                   beam_thk_hard: float = 0.5,
                   beam_run_hard: float = 3.0,
                   beam_reach: float = 2.0,
                   reject_beam_bar: bool = True,
                   bar_run: float = 2.5,
                   bar_thk: float = 0.80,
                   binary: np.ndarray | None = None,
                   rejected: list[Detection] | None = None) -> list[Detection]:
    """找符头（实心 / 空心分别用不同的判别量）。

    实心符头：圆盘内墨迹占比 ≥ filled_thr。
        实心符头 ≈ 1.0；符尾 ≈ 0.40；符干 ≈ 0.10；休止符 ≈ 0.25。
    空心符头：环内占比 − λ × 孔内占比 ≥ hollow_thr。
        空心符头 ≈ 1.0 − 1.0×0.0 = 1.0；
        实心符头 ≈ 1.0 − 1.0×1.0 = 0（正确排除）；
        符尾 ≈ 0.5 − 1.0×0.8 = −0.3（正确排除）。

    ⚠️ 上述理想值只在**完美字形**上成立。实测合成图上真空心符头只有 0.69，
    与"符杠白缝"的 0.55~0.69 重叠，所以必须再叠一层结构判据
    `_is_beam_gap`（见其 docstring）。

    ⚠️ 环厚必须和 render.py 用**同一个取整方式**（`int()` 而不是 `round()`）。
    两者差 1 个像素就会让环模板与字形不重合，空心符头得分掉到 0.79 附近，
    再叠加谱线去除留下的缺口就掉到阈值以下 —— 实测直接导致二分音符全部漏检。
    """
    space = staff.space
    tw = int(round(NOTEHEAD_W * space))
    th = int(round(NOTEHEAD_H * space))
    w = max(2, int(space * 0.17))              # ★ 与 render._draw_notehead 保持一致
    radius = max(2, int(round(space * 0.55)))

    disc = _tmpl_from_draw(tw, th, lambda d: d.ellipse([0, 0, tw - 1, th - 1], fill=0))
    ring = _tmpl_from_draw(tw, th,
                           lambda d: d.ellipse([0, 0, tw - 1, th - 1],
                                               outline=0, width=w))
    inner = _tmpl_from_draw(tw, th,
                            lambda d: d.ellipse([w, w, tw - 1 - w, th - 1 - w], fill=0))
    hole = inner & ~ring

    roi = _staff_roi(staff, ink.shape, pad_space=6.0)
    sub, ox, oy = _crop_for(ink, roi, disc)
    disc_frac = _limit_to_roi(_correlate_sub(sub, disc), ox, oy, roi)
    hollow_score = _limit_to_roi(
        _correlate_sub(sub, ring) - lambda_hole * _correlate_sub(sub, hole),
        ox, oy, roi)

    fallback_mask = None
    if binary is not None and music_font_path():
        cp = MUSIC_CP["notehead_hollow"]
        fallback_mask, _, _ = _glyph_mask(cp, _font_size_for_height(cp, NOTEHEAD_H * space))
    search_hollow_thr = min(hollow_thr, 0.35) if fallback_mask is not None else hollow_thr

    out: list[Detection] = []
    beam_gap_rejected: list[Detection] = []
    for x, y, v in _nms_peaks(disc_frac, filled_thr, radius):
        gx, gy = x + ox, y + oy
        det = Detection(SYM_NOTEHEAD_FILLED, float(gx), float(gy), float(v),
                        _glyph_bbox_in_image(gx, gy, disc),
                        extra={"measure": "disc_fill", "value": float(v)})
        # ★ "又长又薄" ⇒ 符杠/谱线残渣，不是符头（见 `_is_beam_bar`）。
        #   圆盘占比判据对符杠**恒为满分**，所以必须叠这一条尺寸判据。
        if reject_beam_bar and _is_beam_bar(ink, gx, gy, space, bar_run, bar_thk):
            if rejected is not None:
                rejected.append(det)
            continue
        out.append(det)
    for x, y, v in _nms_peaks(hollow_score, search_hollow_thr, radius):
        gx, gy = x + ox, y + oy
        det = Detection(SYM_NOTEHEAD_HOLLOW, float(gx), float(gy), float(v),
                        _glyph_bbox_in_image(gx, gy, ring),
                        extra={"measure": "ring_minus_hole", "value": float(v)})
        if reject_beam_gap and _is_beam_gap(ink, gx, gy, space,
                                            beam_run, beam_thk, beam_thk_hard,
                                            beam_run_hard, beam_reach):
            beam_gap_rejected.append(det)
            if rejected is not None:
                rejected.append(det)
            continue
        if v < hollow_thr:
            font_evidence = _hollow_font_evidence(binary, gx, gy, fallback_mask)
            if font_evidence < (0.65 if v < 0.45 else 0.60):
                continue
            if v < 0.45 and not _hollow_stem_evidence(ink, gx, gy, space):
                continue
        out.append(det)

    if binary is not None:
        out.extend(find_hollow_notehead_holes(binary, staff, ink))

    # In a close half-note chord, the lower head's rim can make one side of
    # the upper ring look too thick for the beam-gap guard.  Recover only a
    # rejected ring with its own font outline and stem beside an independently
    # accepted hollow head exactly one staff space away at the same x.
    if binary is not None and fallback_mask is not None:
        recovered = _recover_stacked_hollow_heads(
            out, beam_gap_rejected, binary, ink, space, fallback_mask)
        out.extend(recovered)
        if rejected is not None:
            for det in recovered:
                rejected.remove(det)
        probed = _probe_stacked_hollow_heads(
            out, hollow_score, ox, oy, binary, ink, space, fallback_mask)
        out.extend(probed)
        if probed:
            # A three-tone whole-note chord can have two consecutive heads
            # hidden by overlapping rims.  Limit propagation to one extra
            # staff step and require the same independent shape evidence.
            chained = _probe_stacked_hollow_heads(
                out, hollow_score, ox, oy, binary, ink, space, fallback_mask,
                seeds=probed)
            out.extend(chained)
            probed.extend(chained)
        if rejected is not None and probed:
            rejected[:] = [det for det in rejected
                           if not any(det.symbol == SYM_NOTEHEAD_HOLLOW
                                      and abs(det.x - head.x) <= .20 * space
                                      and abs(det.y - head.y) <= .20 * space
                                      for head in probed)]

    # 实心与空心互为排斥：同一位置只留得分高的那个
    return _dedupe_cross_class(out, space, rx=0.6, ry=0.6)


def _dot_size_limit_space(space: float) -> float:
    """附点墨迹允许的最大尺寸（单位 space）。

    ★ 从 `render._draw_dot` 的**物理尺寸**导出，不写死常数：
      附点是个半径 `max(1.5, 0.15·space)` 的实心圆 → 直径 ≈ `0.30·space`。
      实测（合成页，光栅化后）：

          真附点 `max(run, ext)` = **0.33~0.38 space**（≈1.1~1.3 倍直径）
          假附点（落在隔壁符头上）= **0.76~1.15 space**

      取 **2.5 倍直径**作上界 → 0.75·space，两侧各留约 1.25 倍余量。
    """
    diam = 2.0 * max(1.5, 0.15 * space)          # 与 render._draw_dot 同一个式子
    return 2.5 * diam / max(1e-9, space)


def _is_dot_sized(ink: np.ndarray, x: int, y: int, space: float,
                  limit_space: float | None = None) -> bool:
    """候选处的墨迹是否**只是一个附点那么大**。

    ★★ 附点判据的核心，也是"视觉相似 ≠ 可判别"的又一个实例：

    `find_dots` 的做法是"在每个符头右方 `[+0.95, +2.1]·space` 的窗口里，
    取 7×7 实心圆模板的最大响应"。但那个窗口里**经常有整个隔壁符头**：
    密集十六分音符的中心间距只有约 1.3·space，落在窗口内 ——
    而实心符头对实心圆模板的响应是 **1.00（满分）**，
    与真附点**完全无法用分数区分**（实测两边都是 1.00）。

    实测证据（`tools/probe_dots.py`，**一个附点都不放**的对照页）：
        25 个检出**全部为假**，score 全是 1.00，
        `max(run, ext)` 是 **0.76~1.15**，而候选到右侧符头的距离只有 **0.10~0.33 space**。
    真实照片上 581 个候选的 `max(run, ext)` 中位数是 **1.33**、最大 21.9
    （很多候选落在符杠 / 谱线残渣上）。

    所以真正的判别量是**墨迹尺寸**：附点只有 0.30·space 见方，
    而符头是 1.18 × 0.95、符杠可以横贯 20+ space。

    ⚠️ **不能用"候选离右侧符头很近"来判**：密集段落里**真附点**与它右边那个
    符头在 x 上只差 `1.3 - 1.21 = 0.09 space` —— 那个判据会把密集段落里的
    真附点全部误杀（提示：标定页故意做得稀疏，所以标定页上它"看起来成立"）。
    """
    if limit_space is None:
        limit_space = _dot_size_limit_space(space)
    lim = limit_space * space
    a, b = _h_run_span(ink, y, x)
    if a >= 0 and (b - a + 1) > lim:
        return False
    return _v_extent(ink, y, x, int(round(space))) <= lim


def _find_compact_dot(ink: np.ndarray, head: Detection,
                      space: float) -> Detection | None:
    """Recover a detached dot when the fixed template misses a small photo blob.

    Restrict the component centre to the note's immediate right and require
    both dimensions, area, and fill to look like a compact spot.  A neighboring
    notehead, staff remnant, or accidental is much larger or elongated.
    """
    if cv2 is None:
        return None
    h, w = ink.shape
    x0 = max(0, int(round(head.x + 0.90 * space)))
    x1 = min(w, int(round(head.x + 1.60 * space)) + 1)
    y0 = max(0, int(round(head.y - 0.75 * space)))
    y1 = min(h, int(round(head.y + 0.75 * space)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    n, _labels, stats, centers = cv2.connectedComponentsWithStats(
        ink[y0:y1, x0:x1].astype(np.uint8), 8)
    candidates: list[tuple[float, Detection]] = []
    for i in range(1, n):
        left, top, width, height, area = [int(v) for v in stats[i]]
        # A beam/staff remnant clipped by the ROI can masquerade as a small
        # connected component.  A real isolated dot is wholly inside it.
        if (left <= 0 or top <= 0 or left + width >= x1 - x0
                or top + height >= y1 - y0):
            continue
        cx, cy = float(centers[i, 0] + x0), float(centers[i, 1] + y0)
        dx = cx - head.x
        if not (0.95 * space <= dx <= 1.40 * space
                and abs(cy - head.y) <= 0.68 * space):
            continue
        if not (0.16 * space <= width <= 0.55 * space
                and 0.16 * space <= height <= 0.55 * space
                and 0.025 * space * space <= area <= 0.25 * space * space):
            continue
        if not (0.60 <= width / height <= 1.65 and area / (width * height) >= 0.42):
            continue
        detection = Detection(SYM_DOT, cx, cy, 0.65,
                              (float(left + x0), float(top + y0),
                               float(width), float(height)),
                              {"component_fallback": True})
        candidates.append((abs(dx / space - 1.2) + 0.25 * abs(cy - head.y) / space,
                           detection))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def find_dots(ink: np.ndarray, staff: StaffGeometry, heads: list[Detection],
              min_frac: float = 0.62,
              reject_big_blob: bool = True,
              size_ratio: float = 2.5,
              rejected: list[Detection] | None = None) -> list[Detection]:
    """找附点（延长记号的小圆点）。

    ★ 附点只在"某个符头的右方"才有意义，所以**不做全局搜索**，
    而是对每个符头开一个受限窗口去搜。

    为什么必须这样：附点模板只是一个 7×7 的小实心圆，任何一处较厚的
    墨迹（符头边缘、符干根部、符尾、谱号笔画）都能让它得高分。
    全局搜索实测在每个符头上都找出 2~3 个假附点，共 34 个。
    把它绑在符头上之后，一个符头最多贡献一个附点。

    ★★ 但"绑在符头上"**并不够**：窗口里常常有整个隔壁符头，
    而实心符头让模板拿满分。必须再叠一条**尺寸**判据
    （`_is_dot_sized`）—— 见它的 docstring。
    """
    space = staff.space
    r = max(2, int(round(space * 0.15)))
    d2 = 2 * r + 1
    dot = _tmpl_from_draw(d2, d2, lambda d: d.ellipse([0, 0, d2 - 1, d2 - 1], fill=0))

    out: list[Detection] = []
    for hd in heads:
        # ★ 窗口左边界必须让开符头本体（见上方说明），只在该窗口内做相关。
        # 空心符头的中心来自白孔，实际附点可能离孔中心仅 0.7 space。
        # 大墨块仍由下方的尺寸判据挡住。
        dot_start = 0.55 if hd.symbol == SYM_NOTEHEAD_HOLLOW else 0.95
        roi = (int(round(hd.x + space * dot_start)), int(round(hd.y - space * 0.7)),
               int(round(hd.x + space * 2.1)), int(round(hd.y + space * 0.7)))
        sub, ox, oy = _crop_for(ink, roi, dot)
        if sub.size == 0:
            continue
        sc = _limit_to_roi(_correlate_sub(sub, dot), ox, oy, roi)
        i = int(np.argmax(sc))
        yy, xx = divmod(i, sc.shape[1])
        v = float(sc[yy, xx])
        det = None
        gx, gy = xx + ox, yy + oy
        if v >= min_frac:
            det = Detection(SYM_DOT, float(gx), float(gy), v,
                            _glyph_bbox_in_image(gx, gy, dot))
        # ★ 尺寸判据：窗口里很可能是**隔壁那个整个符头**（实心符头让模板拿满分），
        #   或者符杠 / 谱线残渣。附点只有 0.30·space 见方，见 `_is_dot_sized`。
        # 实拍补光页的附点会轻微扩散，空心符头的真附点实测竖向 17 px
        # （space 21.76，原界限 16.32 px）。只给空心符头留这段余量。
        allowed_ratio = max(size_ratio, 2.85) if hd.symbol == SYM_NOTEHEAD_HOLLOW else size_ratio
        if det is not None and reject_big_blob and not _is_dot_sized(
                ink, gx, gy, space, allowed_ratio / 2.5 * _dot_size_limit_space(space)):
            if rejected is not None:
                rejected.append(det)
            det = None
        if det is None:
            det = _find_compact_dot(ink, hd, space)
        if det is not None:
            out.append(det)
    return out


# ---------------------------------------------------------------------------
# 休止符 / 临时记号
# ---------------------------------------------------------------------------


def _h_run_span(ink: np.ndarray, y: int, x: int) -> tuple[int, int]:
    """(y, x) 所在水平连续墨迹段的 (起点, 终点)。不是墨迹则返回 (-1, -1)。"""
    h, w = ink.shape
    if not (0 <= y < h) or not (0 <= x < w) or not ink[y, x]:
        return -1, -1
    a = x
    while a > 0 and ink[y, a - 1]:
        a -= 1
    b = x
    while b + 1 < w and ink[y, b + 1]:
        b += 1
    return a, b


def _v_extent(ink: np.ndarray, y: int, x: int, half: int) -> int:
    """(y, x) 处竖直方向的连续墨迹长度（最多上下各 half 像素）。"""
    h = ink.shape[0]
    if not (0 <= y < h) or not (0 <= x < ink.shape[1]) or not ink[y, x]:
        return 0
    a = y
    while a - 1 >= 0 and a - 1 > y - half and ink[a - 1, x]:
        a -= 1
    b = y
    while b + 1 < h and b + 1 < y + half and ink[b + 1, x]:
        b += 1
    return b - a + 1


#: 全 / 二分休止符字形的**物理尺寸**（单位：space），
#: ★ 必须与 `render._draw_rest` 里的 `rw` / `rh` 保持一致 —— 那是唯一真源。
#: 判据从它导出（`_exceeds_glyph_box`），而不是写死像素常数。
_REST_GLYPH_W = 1.15
_REST_GLYPH_H = 0.48


def _exceeds_glyph_box(ink: np.ndarray, y: int, x: int, space: float,                       rw_max: float, rh_max: float) -> bool:
    """判断 (y, x) 处的墨迹是否**超出了休止符字形应有的方框**。

    ★ 判据直接从 `render._draw_rest` 的**物理尺寸**导出，不写死常数：

        全休止符 = 二分休止符 = 实心矩形 `rw = 1.15·space` 宽 × `rh = 0.48·space` 高
        四分休止符 = Z 形折线，`0.64·space` 宽 × `2.4·space` 高（又高又瘦）

    所以"紧凑"这条只对**全 / 二分**成立：它俩是个小矩形，
    任何方向上的墨迹都不该超出这个方框太多。实测（真实照片，见
    `tools/probe_rests.py`）：

        | 对象 | 字形实测 | 是否合法 |
        |---|---|---|
        | 全休止符（合成） | 1.19 × 0.52 space | ✅ |
        | 全休止符（真实印刷） | 1.33~1.46 × 0.50~0.62 space | ✅ |
        | **符杠**（两根横梁，`_looks_like_bar` 差 0.09 space 没拦住） | **2.10~2.33** × 0.44~0.56 | ❌ 横向超框 |
        | **符干 / 谱号笔画** | 0.83~0.90 × **0.93~3.01** | ❌ 纵向超框 |
        | **实心符头**（`find_noteheads` 漏检时） | 0.87 × **0.92** | ❌ 纵向超框 |

    两个上界各留 ~1.2 倍余量（真样本最坏 1.46 / 0.62）；关键在于
    **符杠只在横向超、符干/符头只在纵向超**，所以两条必须并列（OR），
    单看任何一条都会漏掉另一半。

    为什么不对四分休止符用这一条：它本身就是 **2.4 space 高**的折线，
    纵向上与符干同量级，用了必然误杀真四分休止符。
    """
    a, b = _h_run_span(ink, y, x)
    if a >= 0 and (b - a + 1) > rw_max:
        return True
    ext = _v_extent(ink, y, x, int(round(space)))
    return ext > rh_max


def _looks_like_bar(ink: np.ndarray, y: int, x: int,
                    space: float, max_run_space: float,
                    min_thick_space: float = 0.28,
                    reach: int = 4) -> bool:
    """判断 (y, x) 处是不是一条**又长又厚**的横条（符杠），而不是紧凑字形。

    ★ 判据 = 横向连续游程过长 **且** 该横条有足够厚度，两条缺一不可。

    为什么要"且厚度"（这一步是被误杀风险逼出来的）
    ----------------------------------------------
    只按"横向过长"判断会误杀**真休止符**：二分休止符恰好坐在谱表中间线上，
    一旦去谱线在该处留下残渣，休止符所在行的游程被撑长，真的也会被毙掉。
    而"残渣"和"符杠"在几何上有确定的区别：
        符杠    厚约 0.4 space（粗细与符干同量级）
        线残渣  厚约 0.17 space（就是 4 px 的谱线）
        休止符  高约 0.5 space、宽仅 1.6 space（不长）
    所以"长 + 厚"才等于符杠；"长 + 薄"只是线残渣，不构成否决理由。

    只采样游程中段的几列量厚度，而不是逐列量 —— 候选数量很少（每谱表几个），
    而这个量只用于否决，采 5 列足够稳。
    """
    h, w = ink.shape
    if not (0 <= y < h) or not (0 <= x < w):
        return False
    for yy in (y - reach, y, y + reach):
        a, b = _h_run_span(ink, yy, x)
        if a < 0:
            continue
        if (b - a + 1) <= max_run_space * space:
            continue
        # 长条：再验厚度，排除"去谱线残渣"
        span = b - a + 1
        cols = [a + int(round(span * f)) for f in (0.2, 0.35, 0.5, 0.65, 0.8)]
        thick = [_v_extent(ink, yy, min(w - 1, max(0, c)), int(round(space)))
                 for c in cols]
        thick = [t for t in thick if t > 0]
        if thick and float(np.median(thick)) >= min_thick_space * space:
            return True
    return False


def find_rests(ink: np.ndarray, staff: StaffGeometry,
               heads: list[Detection] | None = None,
               min_score: float = 0.55,
               lambda_off: float = 0.5,
               max_run_space: float = 2.4,
               min_thick_space: float = 0.28,
               box_w_ratio: float = 1.5,
               box_h_ratio: float = 1.7) -> list[Detection]:
    """找休止符（全/二分/四分）。

    模板用 render.py 里**同一个**绘制函数渲染，且把模板的 y=0 当作谱表顶线，
    这样休止符在谱表中的相对位置天然就被模板编码进去了。

    ★ 必须做四件事，否则会大面积误检（实测每个符头内部都会匹配出一个全休止符）：
      1. **限制相关峰的纵向范围**：同一套坐标约定下，所有休止符模板的相关峰
         都落在谱表顶线下方 1.85 个 space 处。
      2. **排除已判定为符头的位置**：全休止符字形就是一个实心矩形，
         放到实心符头内部，ON 占比能到 0.95 —— 形状本身完全不具区分度。
      3. **加"框外应为空白"的惩罚**：真正休止符的字形周围是空的，
         而符头附近有符干、符尾，框外墨迹多。
      4. **排除"横条"**（见 `_looks_like_bar`）。

    ★ 为什么还要第 4 条（真实照片上逼出来的）
    ------------------------------------------
    二分/全休止符的模板都是**实心矩形**，而**符杠也是横条**。
    模板嵌在一条比它长的符杠里照样能拿高分，因为第 3 条的"框外惩罚"
    只作用在**模板自己的包围盒之内**，管不到向外延伸出去的那一截。
    合成图上符杠不会压在谱表中间线上，所以这个坑一直没暴露；
    真实钢琴谱上符杠压在中间线很常见 —— 实测一张真实照片里一个小节
    凭空多出两个 `rest(960)`（共 1920 tick，等于整小节的期望值），
    tick 误差 +2820 里绝大部分来自这两项。实测加上第 4 条后，
    该谱表的 tick 误差从 3780 降到 2100。

    ★ 试过但**失败**的做法（留作教训）
    ----------------------------------
    第一版用"候选所在**连通域**的包围盒不得超过 3 个 space"。
    真实照片上确实修好了，但**合成退化集的模糊场景从 100% 掉到 73.3%**
    —— 模糊会让真休止符与相邻符干/符头**连成一体**，包围盒被撑大，
    于是真休止符被误杀。绝对尺寸类判据在模糊面前不成立；
    改成"形状特征"类判据（水平过长 / 有斜率）才既有效又不误杀。
    """
    space = staff.space
    h, w = ink.shape
    out: list[Detection] = []
    tw = int(round(1.6 * space))
    th = int(round(3.7 * space))
    y_peak = staff.y_top_line + th / 2.0
    half_win = int(round(space * 0.5))
    ya = max(0, int(round(y_peak - half_win)))
    yb = min(h, int(round(y_peak + half_win)) + 1)
    if yb - ya < 3:
        return []

    roi = (0, ya, int(ink.shape[1]), yb)

    for name, sym in (("whole", SYM_REST_WHOLE),
                      ("half", SYM_REST_HALF),
                      ("quarter", SYM_REST_QUARTER)):
        tmpl = _tmpl_from_draw(tw, th,
                               lambda d, name=name, tw=tw, space=space:
                               _draw_rest(d, tw / 2.0, 0.0, space, name))
        if tmpl.sum() < 8:
            continue
        ys, xs = np.nonzero(tmpl)
        gx0, gy0 = float(xs.mean()), float(ys.mean())
        sub, ox, oy = _crop_for(ink, roi, tmpl)
        if sub.size == 0:
            continue
        on = _correlate_sub(sub, tmpl)
        off_box = ~tmpl
        off = _correlate_sub(sub, off_box) if off_box.sum() else np.zeros_like(on)
        score = _limit_to_roi(on - lambda_off * off, ox, oy, roi)
        for x, y, v in _nms_peaks(score, min_score,
                                  radius=max(2, int(round(space * 0.6)))):
            # ★ 必须加回裁剪偏移 (ox, oy)。`_nms_peaks` 给的是**子图**坐标，
            # 忘了加偏移的后果很隐蔽：单行页面的谱表带 ROI 恰好从 (0,0) 开始，
            # 偏移为 0 时一切正常；一旦页面上有多行谱表，下面的谱表就整体错位。
            # 实测在带退化的第 2 页上偏移 81px，导致"排除符头位置"的判据失效，
            # 每个符头旁边都多检出一个二分休止符。
            cx = x + ox + gx0 - tw / 2.0
            cy = y + oy + gy0 - th / 2.0
            if heads and any(abs(cx - hd.x) <= space * 1.1
                             and abs(cy - hd.y) <= space * 1.0 for hd in heads):
                continue
            # 第 4 条：排除"又长又厚"的横条（符杠）
            if max_run_space > 0 and _looks_like_bar(
                    ink, int(round(cy)), int(round(cx)), space,
                    max_run_space, min_thick_space):
                continue
            # 第 5 条：全 / 二分休止符必须**紧凑**（见 `_exceeds_glyph_box`）。
            # 第 4 条只处理"又长又厚"，实测有符杠横向游程 **2.31 space**、
            # 而阈值是 2.4 —— 差 0.09 space（1~2 个像素）就漏过去了，
            # 与交接文档"修复 2"是同一类"阈值只差 1 个像素"的坑。
            # 另有符干 / 漏检符头在**纵向**超出（0.92~3.01 space），
            # 而这两个方向的失效互相独立，所以必须分别卡。
            if name in ("whole", "half") and _exceeds_glyph_box(
                    ink, int(round(cy)), int(round(cx)), space,
                    _REST_GLYPH_W * box_w_ratio * space,
                    _REST_GLYPH_H * box_h_ratio * space):
                continue
            out.append(Detection(sym, float(cx), float(cy), float(v),
                                 _glyph_bbox_in_image(x + ox, y + oy, tmpl)))
    return out


def find_accidentals(ink: np.ndarray, staff: StaffGeometry,
                     min_score: float = 0.55, lambda_off: float = 0.5) -> list[Detection]:
    """找临时记号（升/降/还原）。

    ★ 必须加"字形框之外应为空白"的惩罚。
    还原号（♪ 形的两条竖线 + 两条横线）墨迹很稀疏，ON 像素只占模板包围盒的
    不到 20%。单纯用 ON 占比的话，任何一个符头压在模板下都能拿到 0.63~0.88
    的高分 —— 实测每个符头旁边都会多出一个假还原号（共 17 个）。
    加上框外惩罚后，真记号（周围是空白）得分基本不变，压在符头上的假记号
    会掉到阈值以下。
    """
    space = staff.space
    out: list[Detection] = []
    tw = int(round(ACC_W * space * 1.6))
    th = int(round(space * 2.3))
    roi = _staff_roi(staff, ink.shape, pad_space=3.0)
    for kind, sym in ACCIDENTAL_TO_SYM.items():
        def fn(d, kind=kind, tw=tw, th=th, space=space):
            _draw_accidental(d, 0.0, th / 2.0, space, kind)
        templates = [("geometry", _tmpl_from_draw(tw, th, fn), min_score)]
        # The regression pages use a music font, while the older detector only
        # searched for simplified hand-drawn accidentals.  Keep both families:
        # a real score can have either style, and the stronger local response
        # wins during cross-class deduplication.
        if music_font_path():
            px = _font_size_for_height(MUSIC_CP[sym], 1.9 * space)
            mask, _xoff, _yoff = _glyph_mask(MUSIC_CP[sym], px)
            pad = max(2, int(round(0.22 * space)))
            templates.append(("music_font", np.pad(mask, pad),
                              max(min_score, 0.67)))
        for family, tmpl, threshold in templates:
            ys, xs = np.nonzero(tmpl)
            if xs.size == 0:
                continue
            gh, gw = tmpl.shape
            gx0, gy0 = float(xs.mean()), float(ys.mean())
            sub, ox, oy = _crop_for(ink, roi, tmpl)
            if sub.size == 0:
                continue
            on = _correlate_sub(sub, tmpl)
            off_box = ~tmpl
            off = _correlate_sub(sub, off_box) if off_box.sum() else np.zeros_like(on)
            score = _limit_to_roi(on - lambda_off * off, ox, oy, roi)
            for x, y, v in _nms_peaks(score, threshold,
                                      radius=max(2, int(round(space * 0.5)))):
                cx = x + ox + gx0 - gw / 2.0
                cy = y + oy + gy0 - gh / 2.0
                out.append(Detection(sym, float(cx), float(cy), float(v),
                                     _glyph_bbox_in_image(x + ox, y + oy, tmpl),
                                     {"template_family": family}))
    return out


# ---------------------------------------------------------------------------
# 小节线
# ---------------------------------------------------------------------------


def _staff_side_hits(binary: np.ndarray, staff: StaffGeometry, x: int,
                     reach_space: float = 1.5, rowpad: int = 3) -> tuple[int, int]:
    """在该列的左右两侧、沿 5 条谱线行找墨迹，返回 (左侧命中行数, 右侧命中行数)。

    ★ 物理含义：**小节线必须坐落在谱表上**。
    真节线是**穿过谱线**的一根竖线，所以它旁边（同一行上）一定有谱线；
    而书页边缘 / 装订阴影那根竖线左右两侧**根本没有谱线**（谱线还没开始）。

    这与 `find_barlines` 原有的两个判据（覆盖率 ≥0.90、水平游程 ≤ 2.5·线宽）
    是**正交**的：那两个只描述"这根竖线自己长什么样"，
    而书页边缘那根线**细、竖直、且贯穿整个谱表高度**，与真节线**同形** ——
    实测 101910 的假节线就来自 x≈110（书页边缘），
    它切出来的小节一个音符都没有，成为"不足方向"的主要来源。

    ⚠️ 必须在**二值图**（去线之前）上测：谱线在 `ink`（去线后）里已经被抹掉了，
    拿它来测会连真节线也一起否掉（实测真节线也报 0/5）。

    ⚠️ 判定要用 `max(左,右)` 而不是 `min`：
    **谱表起始处的开谱节线左侧本来就没有谱线**，用 `min` 会把它们全部误杀，
    进而把小节一路扩到页面左边、连谱号一起吞进来。
    """
    h, w = binary.shape
    reach = max(3, int(round(reach_space * staff.space)))
    left = right = 0
    for y in staff.line_ys:
        yi = int(round(y))
        if not (0 <= yi < h):
            continue
        ya, yb = max(0, yi - rowpad), min(h, yi + rowpad + 1)
        if binary[ya:yb, max(0, x - reach):max(0, x - 3)].any():
            left += 1
        if binary[ya:yb, min(w, x + 4):min(w, x + reach + 1)].any():
            right += 1
    return left, right


def find_barlines(ink: np.ndarray, staff: StaffGeometry,
                  cover_frac: float = 0.90,
                  max_h_run: float | None = None,
                  merge_gap_frac: float = 0.8,
                  binary: np.ndarray | None = None,
                  on_staff_need: int = 2,
                  rejected: list[float] | None = None) -> list[float]:
    """找小节线。

    ★ 需要**两个**判据同时满足，缺一不可：

    1. 在精确谱表高度范围内，列墨迹覆盖率 ≥ cover_frac（默认 0.90）。
       单用这个不够 —— "符干 + 符头"在同一列上也能凑到 95%：
       一个 F4（step+1）带朝上符干，符干从 y=163 到 91.6，
       符头又从 153 到 173，并集 81 px / 85 px = 95%。

    2. 该列至少 75% 的有墨迹行的水平游程 ≤ max_h_run。
       小节线偶尔与连音线交叉，不能再用整列最大值一票否决；
       符干经过符头时，有较多行的水平游程明显变宽。

    ★ `max_h_run` 默认**按谱线粗细推出**，而不是写死的常数。
    踩过的坑：真实照片上这条写死成 5，结果**一条小节线都找不到**。
    实测某张真实钢琴谱，真小节线列的最大水平游程是 **6**（真谱线粗细 4 px，
    小节线用同一支笔，横向就是 4~6 px），而符干/符头那类假候选是
    14 / 26 / 29 / 30 / 31 / 32 / 33 / 64 —— 分辨力绰绰有余，
    只是阈值卡在 5 上，把真的全毙了。合成图上谱线只有 2 px，
    游程 2~3，写死 5 恰好能过，所以这个坑在合成集上一直没暴露。
    按 `2.5 * thickness` 取值：粗细 2 -> 5（与原来的 5 完全一致，
    合成黄金基准不受影响），粗细 4 -> 10，粗细 5 -> 12.5（仍远小于 14）。

    另外：终止小节线是双竖线，会被识别成两条很近的线，
    必须合并（否则会多出一个空小节，导致小节编号全部错位）。
    """
    h, w = ink.shape
    y0 = max(0, int(round(staff.y_top_line)))
    y1 = min(h, int(round(staff.y_bottom_line)) + 1)
    if y1 - y0 < 4:
        return []
    band = ink[y0:y1, :]
    cover = band.sum(axis=0)
    need = cover_frac * (y1 - y0)
    if max_h_run is None:
        max_h_run = max(5.0, 2.5 * staff.thickness)
    # ★ 水平游程只在谱表带内计算即可 —— 水平游程本来就不会跨行，
    # 所以这是**精确等价**的，不是近似。整图算的话，一张 10 行谱表的
    # A4 页每行都要过一遍 8.7M 像素，10 遍下来光这一项就是秒级开销。
    hrl = horizontal_run_length(band)
    # 真节线可能与连音线/符杠交叉，个别行的水平游程会很长；
    # 用单列最大值会把整根节线误杀。要求至少 75% 的有墨迹行仍像细竖线，
    # 同时挡住符干与符头构成的假节线（符头占据较多宽游程行）。
    thin_rows = (band & (hrl <= max_h_run)).sum(axis=0)
    thin_fraction = thin_rows / np.maximum(cover, 1)
    cols = np.where((cover >= need) & (thin_fraction >= 0.75))[0]
    if cols.size == 0:
        return []

    groups: list[list[int]] = [[int(cols[0])]]
    for c in cols[1:]:
        if c - groups[-1][-1] <= 2:
            groups[-1].append(int(c))
        else:
            groups.append([int(c)])
    centers = [float(np.mean(g)) for g in groups if (g[-1] - g[0] + 1) <= 0.9 * staff.space]

    # 合并双竖线（终止小节线）
    merged: list[float] = []
    for c in centers:
        if merged and c - merged[-1] < merge_gap_frac * staff.space:
            merged[-1] = (merged[-1] + c) / 2.0
        else:
            merged.append(c)

    # ★ 第 3 条判据：**必须坐落在谱表上**（见 `_staff_side_hits`）。
    #   前两条判据（覆盖率、水平游程）描述的是"这根竖线自己长什么样"，
    #   而书页边缘 / 装订阴影那根竖线**细、竖直、贯穿全谱表高度**，
    #   与真节线**同形** —— 实测它造出了 101910 每个系统开头的 2~3 个空小节，
    #   也就是"不足方向"（105,750 tick，69 个小节）的主体。
    #   ⚠️ 需要**二值图**；`binary=None` 时跳过这一条（保持旧行为，
    #   这样直接调用 `detect_symbols` 的诊断工具不会与流水线走出两条路 ——
    #   流水线 `pipeline.recognize` 一定会把 binary 传下来）。
    if binary is not None and on_staff_need > 0:
        kept: list[float] = []
        for c in merged:
            lf, rt = _staff_side_hits(binary, staff, int(round(c)))
            if max(lf, rt) >= on_staff_need:
                kept.append(c)
            elif rejected is not None:
                rejected.append(c)
        merged = kept
    return merged


# ---------------------------------------------------------------------------
# 系统级小节线统一
# ---------------------------------------------------------------------------


def find_system_barlines(ink: np.ndarray, y_top: float, y_bottom: float,
                         space: float, thickness: float,
                         cover_frac: float = 0.90,
                         merge_gap_frac: float = 0.8) -> list[float]:
    """在**整条系统带**（高音顶线 → 低音底线）上找小节线。

    用一个只有两条"谱线"的代理几何调用 `find_barlines` ——
    它只用到 y_top_line / y_bottom_line / space / thickness / merge_gap_frac，
    所以代理足够了，不必改动已验证的 `find_barlines`。
    """
    proxy = StaffGeometry(index=-1,
                          line_ys=[float(y_top), float(y_bottom)],
                          space=float(space), x0=0.0, x1=float(ink.shape[1]),
                          thickness=float(thickness))
    return find_barlines(ink, proxy, cover_frac=cover_frac,
                         merge_gap_frac=merge_gap_frac)


def unify_system_barlines(sys_ink: np.ndarray, glyph_sets: list,
                          grp: list[int], y_top: float, y_bottom: float,
                          space: float, cover_min: float = 0.70,
                          merge_frac: float = 0.9,
                          left_edge: float | None = None,
                          right_edge: float | None = None) -> int:
    """把同一系统内各谱表的小节线**并成一套共享边界**，就地写回 `glyph_sets`。

    为什么必须统一（真实照片上量出来的）
    ------------------------------------
    钢琴谱的小节线是贯穿两条谱表的**同一条线**，所以同一系统内两条谱表的
    小节数必然相等。实测却发现大量不等的情况：
        101910 系统配对的小节数是 (2,4) (3,4) (5,3) (6,5) —— 5 对里只 1 对相等
        102013 是 (5,5) (4,4) (6,6) (7,7) (6,4) (4,4) —— 6 对里 5 对相等
    而小节边界错一个小节，该小节内的音符数就错一份，tick 直接成倍膨胀。
    实测精度与这个一致性**强相关**：102013 一致性最高、时值膨胀最小
    (1.0~3.3×)，其中一致性最好的 #7 是 **7/7 小节全对、比例 1.00**；
    101910 一致性最差、膨胀最大 (2.1~7.0×)。

    做法（三选一里最稳的那个）
    --------------------------
    * 不用"直接在系统带上检测"：有些钢琴谱两条谱表的小节线是**分开画**的
      （中间留空隙），整带检测会因覆盖率不足把真小节线全丢掉。
    * 不用"只取某一条谱表的结果"：那条谱表可能恰好漏检。
    * **取各谱表结果的并集，再用系统带的列覆盖率互相验证**：
      真小节线在整条系统带上都有墨迹，而某条谱表自己误检出来的假线过不了这一关。
      系统内部要求较高的全带覆盖率；靠近谱线两端的小节线允许较低覆盖率。
      远在本页谱线左端以左的书页边缘候选直接丢弃。

    返回保留的小节线数量。
    """
    cands: list[float] = []
    for i in grp:
        cands.extend(glyph_sets[i].barlines)
    if not cands:
        return 0
    cands.sort()

    merged: list[float] = []
    for c in cands:
        if merged and c - merged[-1] < merge_frac * space:
            merged[-1] = (merged[-1] + c) / 2.0
        else:
            merged.append(c)

    y0 = max(0, int(round(y_top)))
    y1 = min(sys_ink.shape[0], int(round(y_bottom)) + 1)
    span = max(1, y1 - y0)
    width = sys_ink.shape[1]
    keep: list[tuple[float, float]] = []
    for c in merged:
        # 同一页各系统的谱线左端接近；远在其左侧的书页暗边即使
        # 贯穿整条系统，也不是小节线。
        if left_edge is not None and c < left_edge - 2.0 * space:
            continue
        xc = int(round(c))
        # 两条谱表的原始候选可能相差数像素。直接取平均会把竖线
        # 坐标放在白缝里；以合并坐标为中心重新找覆盖峰，宽度按谱间距定。
        radius = max(2, int(round(0.6 * space)))
        candidates = range(max(0, xc - radius), min(width, xc + radius + 1))
        best_x, best = max(
            ((x, float(sys_ink[y0:y1, x].sum()) / span) for x in candidates),
            key=lambda item: (item[1], -abs(item[0] - c)),
        )
        # 系统内部必须有大部分高度的连贯竖线。合成拍照回归中真节线
        # 经过去线后有约 0.73 的列覆盖率；0.80 会把 20 小节削成 10 小节。
        # 0.70 仍高于同页符干假候选约 0.28 的覆盖率。
        # 开谱节线可能被
        # 谱号/花括号遮住；末尾的终止线可能只画在两条谱表上，
        # 不贯穿中间的间隙。低覆盖率仅在已定位谱表端点附近放行。
        near_left = left_edge is not None and c <= left_edge + 7.0 * space
        near_right = right_edge is not None and c >= right_edge - 4.0 * space
        if best >= cover_min or (near_left and best >= 0.5) or (near_right and best >= 0.55):
            # A stem immediately after a barline can line up on both piano
            # staves and cover 70–80% of the full system.  If it is within
            # 1.4 spaces of a near-complete vertical barline, that stronger
            # line is the shared measure boundary.  Partial real barlines
            # remain eligible when there is no such full-height neighbor.
            if keep and best_x - keep[-1][0] <= 1.4 * space:
                previous_x, previous_cover = keep[-1]
                if max(best, previous_cover) >= 0.92 and min(best, previous_cover) <= 0.83:
                    if best > previous_cover:
                        keep[-1] = (float(best_x), best)
                    continue
            keep.append((float(best_x), best))

    # In a system whose adjacent measure lines both cross nearly the whole
    # grand staff, a much weaker line between them is usually an aligned
    # pair of stems.  The raw 0.70 threshold is needed when genuine barlines
    # are partial, so only apply this relative check with two strong anchors.
    keep = [entry for index, entry in enumerate(keep)
            if not (0 < index < len(keep) - 1
                    and entry[1] < 0.82
                    and keep[index - 1][1] >= 0.94
                    and keep[index + 1][1] >= 0.94)]

    for i in grp:
        glyph_sets[i].barlines = [x for x, _ in keep]
    return len(keep)


# ---------------------------------------------------------------------------
# 谱号 / 拍号
# ---------------------------------------------------------------------------


def find_clef(ink: np.ndarray, staff: StaffGeometry,
              x_limit_space: float = 8.0,
              x_start: float | None = None
              ) -> tuple[str, float, tuple[float, float, float, float] | None]:
    """在谱表最左侧找谱号。

    Returns
    -------
    (谱号名, 得分, 字形包围盒)
        包围盒是**字形**在原图中的位置，不是模板位置 —— 拍号的搜索窗口
        要靠它来定位（见 find_time_signature 的注释）。

    搜索窗口由 `staff.x0` 决定。旧的水平范围检测会把书页暗边当成
    谱线左端，导致窗口完全避开谱号；现在 `staff_h_extent` 要求
    五线的长距离共同支撑，再以该位置搜索。当前模板仅可靠返回
    高音谱号候选；钢琴大谱表的低音谱号由版式规则覆盖。
    """
    space = staff.space
    tw = int(round(3.0 * space))
    th = int(round(4.4 * space))
    tmpl = _tmpl_from_draw(tw, th,
                           lambda d: _draw_treble_clef(d, 0.0, 0.0, 4.0 * space, space))
    left = staff.x0 if x_start is None else x_start
    roi = (int(left), int(staff.y_top_line - space),
           int(round(left + x_limit_space * space)),
           int(staff.y_bottom_line + space))
    sub, ox, oy = _crop_for(ink, roi, tmpl)
    if sub.size == 0:
        return "treble", 0.0, None
    sc = _limit_to_roi(_correlate_sub(sub, tmpl), ox, oy, roi)
    idx = int(np.argmax(sc))
    y, x = divmod(idx, sc.shape[1])
    v = float(sc[y, x])
    if v <= 0:
        return "treble", 0.0, None
    return "treble", v, _glyph_bbox_in_image(x + ox, y + oy, tmpl)


#: 合法拍号分母 —— 只能是 2 的幂。见 find_time_signature 里的说明。
_VALID_TS_DENOMINATORS: tuple[int, ...] = (1, 2, 4, 8, 16)


def find_time_signature(ink: np.ndarray, staff: StaffGeometry,
                        x_start: float, x_limit_space: float = 5.0,
                        min_score: float = 0.65, max_gap_space: float = 0.9
                        ) -> tuple[tuple[int, int] | None, float,
                                   tuple[float, float, float, float] | None]:
    """在谱号右侧找拍号：分子在上两间、分母在下两间。

    ★ 搜索窗口必须由**谱号字形的右边界**推出，不能用谱号模板的中心。
    字形在模板里是偏心的（模板中心 ≠ 字形重心），差出 1.5 个 space
    就足以切掉半个数字 —— 实测这会让 4/4 被识别成 1/1。

    同时要求分子分母横向对齐（|Δx| ≤ max_gap_space），
    否则"1"这种细笔画数字会到处乱匹配。
    """
    from PIL import Image, ImageDraw, ImageFont
    space = staff.space
    h, w = ink.shape
    x0 = int(round(x_start))
    x1 = min(w, int(round(x_start + x_limit_space * space)))
    if x1 - x0 < 4:
        return None, 0.0, None

    tw, th = int(space * 2.0), int(space * 2.2)
    # ★ 只在谱号右侧这一小块里做相关。整图相关会让这个函数吃掉 600ms 以上
    # （10 个数字模板 × 全图 filter2D），而它真正要看的只有几十个像素见方。
    roi = (max(0, int(round(x_start - space * 0.6))),
           max(0, int(round(staff.y_top_line - space * 0.6))),
           x1,
           min(h, int(round(staff.y_bottom_line + space * 1.4))))
    probe = np.zeros((th, tw), dtype=bool)
    probe[0, 0] = True
    sub, ox, oy = _crop_for(ink, roi, probe)
    if sub.size == 0:
        return None, 0.0, None

    try:
        font = ImageFont.load_default(size=int(space * 2.55))
    except TypeError:
        font = ImageFont.load_default()

    digits: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n in range(10):
        img = Image.new("L", (tw, th), 255)
        d = ImageDraw.Draw(img)
        bb = d.textbbox((0, 0), str(n), font=font)
        d.text((tw / 2 - (bb[2] - bb[0]) / 2 - bb[0],
                th / 2 - (bb[3] - bb[1]) / 2 - bb[1]), str(n), font=font, fill=0)
        t = np.asarray(img) < 128
        if t.sum() < 8:
            continue
        # ★ 必须加"数字框之外应为空白"的惩罚。
        # 光看 ON 占比时，细笔画数字（尤其 "1"）是灾难：把 "1" 的模板扣在
        # "4" 上面，那条竖笔画能对上 "4" 的竖笔，ON 占比就有 0.9+，
        # 于是 4/4 被识别成 1/4（实测在带轻微退化的图上必现）。
        # 加上框外惩罚后，"1" 扣在 "4" 上会因为框内多出一大堆墨迹而被压下去。
        on = _correlate_sub(sub, t)
        off_box = ~t
        off = _correlate_sub(sub, off_box) if off_box.sum() else np.zeros_like(on)
        digits[n] = (t, _limit_to_roi(on - 0.5 * off, ox, oy, roi))

    if not digits:
        return None, 0.0, None

    y_num = staff.y_top_line + space * 1.0
    y_den = staff.y_top_line + space * 3.0
    half = int(round(space * 1.2))
    a_x = max(0, x0 - ox)
    b_x = min(sub.shape[1], x1 - ox)

    # 分子分母必须**联合搜索**并满足"横向对齐"。按行独立取最优会漏掉这个约束，
    # 导致细笔画数字（如 1）到处乱匹配。
    cands_num: list[tuple[int, float, int, int]] = []
    cands_den: list[tuple[int, float, int, int]] = []
    for n, (_t, sc) in digits.items():
        for target_y, bucket in ((y_num - oy, cands_num), (y_den - oy, cands_den)):
            if b_x - a_x < 3:
                continue
            lo = max(0, int(round(target_y - half)))
            hi = min(sc.shape[0], int(round(target_y + half)) + 1)
            if hi <= lo:
                continue
            # 用副本做抑制。原先直接改视图会连带污染另一个数字的得分图。
            work = sc[lo:hi, a_x:b_x].copy()
            for _ in range(5):
                i = int(np.argmax(work))
                y, x = divmod(i, work.shape[1])
                v = float(work[y, x])
                if v < min_score * 0.6:
                    break
                bucket.append((n, v, x + a_x + ox, lo + y + oy))
                yy0 = max(0, y - 3); yy1 = min(work.shape[0], y + 4)
                xs0 = max(0, x - 3); xs1 = min(work.shape[1], x + 4)
                work[yy0:yy1, xs0:xs1] = -1e9

    cands_num.sort(key=lambda t: -t[1])
    cands_den.sort(key=lambda t: -t[1])

    # ★ 乐理硬约束：**分母必须是 2 的幂**，分子是 1..32 的正整数。
    #   踩过的坑：原来分子分母都允许 0~9 任意组合，于是模板匹配能输出
    #   "1/7" 这种根本不存在的拍号。而它一旦被当成确定拍号返回，
    #   reconstruct 就会拿 1920/7 = 274 tick 当"期望小节长度"，
    #   整页每个小节的节拍校验、纠错、置信度全部跟着错（实测一张真实
    #   钢琴谱上出现 12 处 "期望 274 tick"）。合成退化集上永远不会暴露
    #   这个问题 —— 合成页面的拍号本来就是合法的，模板匹配没有机会
    #   输出非法值，所以这条约束必须显式写出来。
    cands_num = [c for c in cands_num if 1 <= c[0] <= 32]
    cands_den = [c for c in cands_den if c[0] in _VALID_TS_DENOMINATORS]

    best: tuple[float, int, int, int, int, np.ndarray] | None = None
    for n_num, v_num, xn, yn in cands_num[:6]:
        for n_den, v_den, xd, yd in cands_den[:6]:
            if abs(xn - xd) > max_gap_space * space:
                continue
            s = min(v_num, v_den)
            if best is None or s > best[0]:
                t = digits[n_num][0]
                best = (s, n_num, n_den, xn, yn, t)
    if best is None:
        return None, 0.0, None
    s, n_num, n_den, xn, yn, t = best
    bbox = (xn - t.shape[1] / 2.0, y_num - space * 1.1,
            float(t.shape[1]), float(space * 3.2))
    # ★ 低于 min_score 时**只返回包围盒、不返回拍号**。
    #   踩过的坑：原来这里从不检查 min_score（该参数只在候选收集时用作
    #   `min_score * 0.6` 的截断），于是真实照片上模板相关性只有 0.4~0.6
    #   的胡猜也会被当成确定的拍号返回。实测把 4/4 认成 (1,7)，而
    #   reconstruct 会拿 1920/7 = 274 tick 当"期望小节长度"，纠错阶段
    #   于是把整页的音符时长全改错，单小节 tick 误差冲到 +47000。
    #   返回 None 会走 reconstruct 里 `time_signature or default_time_signature`
    #   的兜底（默认 4/4），那是本来就设计好的路径；包围盒仍然有用 ——
    #   调用方拿它定"乐音区起点"，跟拍号对不对无关。
    if s < min_score:
        return None, float(s), bbox
    return (n_num, n_den), float(s), bbox


# ---------------------------------------------------------------------------
# 总入口
# ---------------------------------------------------------------------------


def music_start_x(staff: StaffGeometry,
                  clef_bbox: tuple[float, float, float, float] | None,
                  time_signature: tuple[int, int] | None,
                  time_bbox: tuple[float, float, float, float] | None) -> float:
    """只在确实识别出拍号时才用拍号框排除引子区。"""
    if clef_bbox is not None:
        after_clef = clef_bbox[0] + clef_bbox[2] + 0.35 * staff.space
    else:
        after_clef = staff.x0 + 3.0 * staff.space
    if time_signature is not None and time_bbox is not None:
        return time_bbox[0] + time_bbox[2] + 0.5 * staff.space
    return after_clef + 0.5 * staff.space


def detect_symbols(ink_without_staff: np.ndarray, staff: StaffGeometry,
                   clef: str = "treble",
                   binary: np.ndarray | None = None,
                   expected_meter: tuple[int, int] | None = None) -> GlyphSet:
    """在单个谱表上做全套符号检测。

    Parameters
    ----------
    ink_without_staff
        已去除谱线的 bool 墨迹图（True = 墨迹）。
    staff
        该谱表的几何信息（staff.py 的输出）。
    binary
        **去线之前**的二值图。给得出就用来做"节线必须坐落在谱表上"的判据
        （谱线在 `ink_without_staff` 里已被抹掉，只有它能提供这个证据）。
        传 `None` 则跳过该判据 —— 但**流水线与诊断工具必须走同一条路**，
        所以 `pipeline.recognize` 一定传它，诊断工具也应当传。
    """
    space = staff.space
    gs = GlyphSet(staff=staff, ink=ink_without_staff)

    # Detect barlines first: a weak left staff line can put x0 beyond the
    # actual opening barline.  That barline is independent evidence for where
    # the clef search must begin.  Otherwise the clef template can lock onto
    # a time-signature digit and hide the real key signature before it.
    off_staff: list[float] = []
    gs.barlines = find_barlines(ink_without_staff, staff,
                                binary=binary, on_staff_need=1,
                                rejected=off_staff)
    gs.barlines_off_staff = off_staff
    clef_left = staff.x0
    clef_limit = 8.0
    if gs.barlines:
        opening = min(gs.barlines)
        if staff.x0 - 5.0 * space <= opening <= staff.x0 + 1.0 * space:
            clef_left = min(clef_left, opening)
            if opening < staff.x0 - 0.5 * space:
                clef_limit = 4.8  # stop before the nearby time-signature digits

    # 1) 谱号（限定在谱表最左侧）
    clef_name, clef_score, clef_bbox = find_clef(ink_without_staff, staff,
                                                x_start=clef_left,
                                                x_limit_space=clef_limit)
    gs.clef, gs.clef_score = clef_name, clef_score
    gs.clef_bbox = clef_bbox

    # 2) 拍号：窗口由谱号字形的右边界推出
    if clef_bbox is not None:
        ts_x = clef_bbox[0] + clef_bbox[2] + 0.35 * space
    else:
        ts_x = staff.x0 + 3.0 * space
    ts, ts_score, ts_bbox = find_time_signature(ink_without_staff, staff, ts_x)
    # A manually confirmed meter is stronger evidence than a template that
    # reads noteheads as digits.  A conflicting bbox must not hide actual
    # notes to its left; retain the score for diagnostics.
    meter_conflict = expected_meter is not None and ts is not None and ts != expected_meter
    if expected_meter is not None and ts != expected_meter:
        ts = None
    gs.time_signature, gs.ts_score = ts, ts_score

    # 3) 拍号右侧才是真正的乐音区域；谱号和拍号都要排除掉，
    #    否则它们的笔画会被当成符头
    music_x0 = music_start_x(staff, clef_bbox, ts, ts_bbox)

    # 5) 符头
    beam_gaps: list[Detection] = []
    heads = [h for h in find_noteheads(ink_without_staff, staff, binary=binary,
                                       rejected=beam_gaps)
             if h.x >= music_x0]
    gs.beam_gaps = [b for b in beam_gaps if b.x >= music_x0]

    # 6) 附点：只在各符头右侧的受限窗口里搜（全局搜会到处误检）
    dot_rejects: list[Detection] = []
    dots = find_dots(ink_without_staff, staff, heads, rejected=dot_rejects)
    gs.dots_big_blob = dot_rejects

    # 7) 休止符：三种休止符模板会互相蹭墨迹 -> 跨类别去重
    rests = _dedupe_cross_class(
        [r for r in find_rests(ink_without_staff, staff, heads=heads)
         if r.x >= music_x0],
        space, rx=0.8, ry=0.8)
    # 安全网：符头与休止符之间再去一次重。全休止符字形就是个实心矩形，
    # 放到实心符头内部 ON 占比能到 0.95，形状上根本无法区分。
    # 这里**永远优先保留符头**（符头有"圆盘/环+孔"两个独立判据，更可靠）。
    rests = [r for r in rests
             if not any(abs(r.x - h.x) <= space * 0.8 and abs(r.y - h.y) <= space * 0.8
                        for h in heads)]
    # Whole/half-rest rectangles can also fire on a beam at the *same onset*
    # as a notehead several staff steps below.  This pipeline reconstructs a
    # single voice per staff, so a rest cannot occupy that note's x position.
    rests = [r for r in rests
             if r.symbol not in (SYM_REST_WHOLE, SYM_REST_HALF) or
             not any(abs(r.x - h.x) <= 0.55 * space for h in heads)]
    # When a conflicting time template is located wholly before the first
    # note, a rest response inside that same box is the time glyph seen twice.
    # If a note lies in the box instead, it may be a false time template on
    # real music; keep the note and any genuine surrounding rest candidates.
    if meter_conflict and ts_bbox is not None and heads:
        bx, by, bw, bh = ts_bbox
        if bx + bw < min(head.x for head in heads):
            rests = [r for r in rests
                     if not (bx <= r.x <= bx + bw and
                             by <= r.y <= by + bh)]

    # 8) 临时记号：升/降/还原号模板形状相近，同样跨类别去重
    gs.raw_accidentals = find_accidentals(ink_without_staff, staff)
    accs = _dedupe_accidentals(
        [a for a in gs.raw_accidentals if a.x >= music_x0], space)

    gs.detections = heads + dots + rests + accs
    return gs


def drop_cross_staff_ghosts(glyph_sets: list, staves: list,
                            tol_space: float = 0.2) -> int:
    """同一个字形被相邻两条谱表各检一次时，**只保留更近的那条谱表**里的那一份。

    ★ 这是"幽灵音符"——一类**任何单符号判据都抓不到**的误检。

    机理：`_staff_roi` 给每个谱表的搜索区上下各留 **6·space** 的余量
    （理由是对的：加线、符干、连音线都会伸出谱表）。但真实钢琴谱
    **系统内间距只有 6.2~7.4 space**，于是相邻两条谱表的 ROI **必然重叠**，
    夹在中间的那些字形会被**各检一次**。实测三张照片共 **159 个**这样的重复检出，
    而且**每一个都落在自己谱表 5 条线之外**。

    危害：多出来的那一份会落进**另一条谱表**的小节里，让那条谱表的 tick
    凭空变大（而且用的是**错误的谱号**去算音高，所以音高也是错的）。
    字形本身完全合法（符头形状、得分 1.00），所以"距离/尺寸/形状"类判据
    一条都拦不住 —— **只能用"同一个物理位置被检了两遍"来发现**。

    归属判据：取两条谱表之间的**中点**。中点以上归上面那条、以下归下面那条
    （就是按谱表做 Voronoi 划分）。这与刻谱惯例一致：音符属于它所在的那条谱表。

    保守之处：**只在两条谱表确实都检出了同一个字形时才丢**
    （丢掉远的那一份）。若近的那条谱表没检出它，就原样保留 ——
    这样"跨谱表去重"**不会让任何音符凭空消失**，只消除重复。
    """
    n = len(staves)
    if n < 2:
        return 0
    dropped = 0
    for i in range(n - 1):
        j = i + 1
        si, sj = staves[i], staves[j]
        tol = max(2.0, tol_space * 0.5 * (si.space + sj.space))
        mid = (si.y_bottom_line + sj.y_top_line) / 2.0
        used: set[int] = set()
        kill_i: list[Detection] = []
        kill_j: list[Detection] = []
        # ⚠️ 必须遍历**副本**：一边 `for d in ...detections` 一边 `remove`，
        #    删掉一个元素会让迭代器跳过下一个 —— 实测因此少丢了 47/159 个幽灵。
        for d in list(glyph_sets[i].detections):
            for k, e in enumerate(glyph_sets[j].detections):
                if k in used:
                    continue
                if abs(d.x - e.x) <= tol and abs(d.y - e.y) <= tol:
                    used.add(k)
                    # 同一个字形：谁在下半区就归下面那条谱表
                    if d.y >= mid:
                        kill_i.append(d)
                    else:
                        kill_j.append(e)
                    dropped += 1
                    break
        for d in kill_i:
            glyph_sets[i].detections.remove(d)
            glyph_sets[i].ghosts.append(d)
        for e in kill_j:
            glyph_sets[j].detections.remove(e)
            glyph_sets[j].ghosts.append(e)
    return dropped


# ---------------------------------------------------------------------------
# 可插拔接口 —— 真实场景要换成这个
# ---------------------------------------------------------------------------


class Detector(Protocol):
    """检测器接口。实现它就能无缝替换 Baseline A。"""

    name: str

    def detect(self, ink_without_staff: np.ndarray, staff: StaffGeometry,
               binary: np.ndarray | None = None) -> GlyphSet:
        ...


class TemplateDetector:
    """Baseline A：几何/模板法。只适合干净图像。"""

    name = "template-geometric"

    def __init__(self, expected_meter: tuple[int, int] | None = None):
        self.expected_meter = expected_meter

    def detect(self, ink_without_staff: np.ndarray, staff: StaffGeometry,
               binary: np.ndarray | None = None) -> GlyphSet:
        return detect_symbols(ink_without_staff, staff, binary=binary,
                              expected_meter=self.expected_meter)


class OnnxDetector:
    """Baseline B（待接入）：ONNX 目标检测模型 —— 真实拍摄场景的必经之路。

    接入步骤（P2 阶段）：
      1. 用 DeepScores / Camera-PrIMuS 训练 YOLOv8n（INT8 量化）。
      2. 整页按 tile（640×640，50% 重叠）推理，NMS 合并。
      3. 把检测框按中心点归入对应谱表（用 staff.py 的几何）。
      4. 用 DPU（Vitis AI）或 CPU（onnxruntime / NEON）跑推理。

    与 TemplateDetector 的接口完全一致，所以 reconstruct.py 不需要任何改动。
    """

    name = "onnx-yolo"

    def __init__(self, model_path: str, conf: float = 0.35,
                 classes: dict[int, str] | None = None, tile: int = 640,
                 overlap: float = 0.5, providers: list[str] | None = None):
        self.model_path = model_path
        self.conf = conf
        self.tile = tile
        self.overlap = overlap
        self.classes = classes or {}
        self.providers = providers
        self._session = None

    def _load(self):
        if self._session is None:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                self.model_path, providers=self.providers
                or ["CPUExecutionProvider"])
        return self._session

    def detect(self, ink_without_staff: np.ndarray, staff: StaffGeometry,
               binary: np.ndarray | None = None) -> GlyphSet:
        raise NotImplementedError(
            "P2 阶段接入：需要先训练好检测模型。\n"
            "在此之前请使用 TemplateDetector —— 它的输出接口与这里完全一致，\n"
            "所以 reconstruct.py / export.py / evaluate.py 都不需要改动。"
        )
