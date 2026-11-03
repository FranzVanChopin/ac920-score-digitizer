"""
合成乐谱渲染器 —— P1 的"免费 ground truth"。

为什么需要它
------------
真实乐谱的标注极其昂贵。但如果**我们自己把乐谱渲染出来**，就同时得到了
图像和 100% 精确的标准答案，可以自动算准确率、做回归测试。

⚠️ 边界要说清楚
   这个渲染器画的是**理想化/几何化的乐谱**（干净背景、标准字形、无畸变），
   它的用途是 **pipeline 回归 + 精度度量**，不是模拟真实拍摄。
   真实场景的鲁棒性必须靠真实照片验证 —— 用手机拍几张真乐谱就是最好的测试集。

   所以：`render.py` 保证"算法对不对"，真实照片保证"能不能用"。

提供的退化模拟
--------------
`degrade()` 可以叠加光照梯度、高斯噪声、模糊、旋转、JPEG 压缩，
用来专门测试 preprocess.py 的各个模块 —— 因为退化是已知的，
效果好不好可以直接量化。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import theory
from .model import PPQ, BBox, Event, Measure, Part, Score, Staff, Voice

# ---------------------------------------------------------------------------
# 排版常量（比例参照 SMuFL / Gould《Behind Bars》的常规值，以 staff space 为单位）
# ---------------------------------------------------------------------------

NOTEHEAD_W = 1.18   # 符头宽 / space
NOTEHEAD_H = 0.95   # 符头高 / space
STEM_LEN = 3.4      # 符干长 / space
LINE_W = 0.11       # 线宽 / space
CLEF_W = 2.6        # 谱号占宽 / space
TS_W = 1.9          # 拍号占宽 / space
ACC_W = 0.85        # 临时记号占宽 / space
MEASURE_PAD = 1.2   # 小节线两侧留白 / space
NOTE_MIN_GAP = 1.6  # 相邻音符最小水平间距 / space


#: 当前笔画粗细倍率（由 `RenderOptions.stroke_scale` 在 `render_score` 开头设置）。
#: ⚠️ 用模块级变量而不是给每个 `_draw_*` 加参数，是因为 `_lw()` 被十几个绘制函数调用，
#: 逐个改签名会牵动整套渲染代码；而它是**渲染期的全局设置**，语义上确实只有一份。
#: 代价：非线程安全（本项目渲染是单线程的）。
_STROKE_SCALE: float = 1.0


def _lw(space: float) -> int:
    return max(2, int(round(space * LINE_W * _STROKE_SCALE)))


# ---------------------------------------------------------------------------
# 字形绘制
# ---------------------------------------------------------------------------


#: 简化 G 谱号用的常量保持不变；下面这一组是**真字体模式**要用的码位
#: （Unicode Musical Symbols 区块，Windows 的 `seguisym.ttf`/Segoe UI Symbol 全覆盖）
MUSIC_CP = {
    "notehead_filled": 0x1D158,
    # U+1D15E 是带符干的“二分音符”；把整枚字缩成符头高度会让
    # 可见符头只有约 7 px 宽，产生不真实的漏检。U+1D157 才是空心符头。
    "notehead_hollow": 0x1D157,
    "rest_whole": 0x1D13B,
    "rest_half": 0x1D13C,
    "rest_quarter": 0x1D13D,
    "accidental_sharp": 0x1D130,
    "accidental_flat": 0x1D12C,
    "accidental_natural": 0x1D12E,
    "augmentation_dot": 0x1D16D,
    "clef_treble": 0x1D11E,
    "clef_bass": 0x1D122,
    "barline": 0x1D100,
}
#: 真字体候选路径（按顺序找第一个存在的）
MUSIC_FONT_CANDIDATES = (
    "C:/Windows/Fonts/seguisym.ttf",
    "C:/Windows/Fonts/SansSerifCollection.ttf",
    "C:/Windows/Fonts/l_10646.ttf",
    "omr/assets/Bravura.otf",
)
_MUSIC_FONT_PATH: str | None = None
_GLYPH_CACHE: dict[tuple[int, int], tuple[np.ndarray, int, int]] = {}


def music_font_path() -> str | None:
    """找一份能画音乐符号的字体；找不到返回 None（自动退回几何画法）。"""
    global _MUSIC_FONT_PATH
    if _MUSIC_FONT_PATH is not None:
        return _MUSIC_FONT_PATH or None
    import os
    for p in MUSIC_FONT_CANDIDATES:
        if os.path.exists(p):
            _MUSIC_FONT_PATH = p
            return p
    _MUSIC_FONT_PATH = ""
    return None


def _glyph_mask(cp: int, px: int) -> tuple[np.ndarray, int, int]:
    """把码位渲染成二值掩码（True = 墨），并返回墨迹左上角相对字形的偏移。"""
    key = (cp, px)
    hit = _GLYPH_CACHE.get(key)
    if hit is not None:
        return hit
    from PIL import Image as _I
    from PIL import ImageDraw as _D
    try:
        f = ImageFont.truetype(music_font_path() or "", px)
    except Exception:                                        # noqa: BLE001
        f = ImageFont.load_default(size=px)
    pad = px
    im = _I.new("L", (px * 3 + pad, px * 3 + pad), 255)
    _D.Draw(im).text((pad, pad), chr(cp), font=f, fill=0)
    a = np.asarray(im) < 128
    ys, xs = np.nonzero(a)
    if xs.size == 0:
        m = np.zeros((1, 1), dtype=bool)
        _GLYPH_CACHE[key] = (m, 0, 0)
        return _GLYPH_CACHE[key]
    sub = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    off_x = int(xs.min()) - pad
    off_y = int(ys.min()) - pad
    _GLYPH_CACHE[key] = (sub, off_x, off_y)
    return _GLYPH_CACHE[key]


def _font_size_for_height(cp: int, target_h: float, probe: int = 96) -> int:
    """求出"让该字形的**墨迹高度** ≈ target_h 像素"所需的字号。"""
    m, _ox, _oy = _glyph_mask(cp, probe)
    ih = max(1, m.shape[0])
    return max(4, int(round(probe * target_h / ih)))


def _draw_music_glyph(img: "Image.Image", cp: int, cx: float, cy: float,
                      space: float, h_space: float
                      ) -> tuple[float, float, float, float]:
    """用**真音乐字体**画一个字形，按"墨迹高 = h_space·space"定字号，
    并把墨迹中心对齐到 (cx, cy)。返回墨迹在原图上的包围盒。

    ★ 为什么要用真字体：`render.py` 原来是用椭圆/矩形/折线**近似**画符头与休止符，
    而真实照片上印的是真正的刻谱字形。CNN 只在近似字形上训练，
    遇到真字形会有**域间隙**，而且这个间隙在合成留出集上永远测不出来。
    本机 `C:/Windows/Fonts/seguisym.ttf` 覆盖了全部需要的音乐码位（实测见
    `tools/probe_fonts.py`）。
    """
    px = _font_size_for_height(cp, max(3.0, h_space * space))
    m, off_x, off_y = _glyph_mask(cp, px)
    mh, mw = m.shape
    # ★★ 记框必须记**粘贴的位置**，不能记"文字原点推回去的位置"。
    #   踩过的坑：返回 `(cx-mw/2-off_x, cy-mh/2-off_y)`，
    #   于是**画对了、框错了**（真值框整体偏了 off_x/off_y 像素），
    #   表现是"框内墨占比 0.1、图看着又没问题" —— 正是最难查的那一类。
    #   这里让"粘贴坐标"和"返回的框"用**同一个变量**，从结构上不可能再错开。
    paste_x = int(round(cx - mw / 2.0))
    paste_y = int(round(cy - mh / 2.0))
    sub = Image.fromarray(np.where(m, 0, 255).astype(np.uint8))
    img.paste(sub, (paste_x, paste_y))
    return (float(paste_x), float(paste_y), float(mw), float(mh))


def _draw_treble_clef(d: ImageDraw.ImageDraw, x: float, top: float, bottom: float, space: float) -> None:
    """简化的 G 谱号：一个螺旋 + 一条下行钩。

    不是精确的 Bravura 字形，但几何特征（跨整个谱表高度 + 位于谱表左侧 +
    中部有一个卷曲）足以让"找谱号"这一步有真实感。
    """
    w = _lw(space)
    h = bottom - top
    cx = x + CLEF_W * space * 0.45
    # 主螺旋（谱表中央）
    mid = top + h * 0.52
    r = space * 0.95
    d.arc([cx - r, mid - r, cx + r, mid + r], start=200, end=160, fill=0, width=w)
    # 上方弧线
    d.arc([cx - r * 0.75, top + h * 0.06, cx + r * 0.75, top + h * 0.46],
          start=180, end=20, fill=0, width=w)
    # 竖直主干
    d.line([(cx, top + h * 0.14), (cx, bottom - h * 0.12)], fill=0, width=w)
    # 下方钩
    rr = space * 0.6
    d.arc([cx - rr * 1.6, bottom - h * 0.24, cx + rr * 0.4, bottom + rr * 0.6],
          start=280, end=120, fill=0, width=w)


def _draw_bass_clef(d: ImageDraw.ImageDraw, x: float, top: float, bottom: float,
                    space: float) -> None:
    """简化的 F 谱号（低音谱号）：一个卷曲 + 两个点。

    ★ 为什么必须有它：真实测试集是**钢琴大谱表**，下面那条谱表全是低音谱号；
    而 `render.py` 原来只有高音谱号。一个只有高音谱号的合成集，
    训不出能用在真实照片上的模型 —— 这是"合成集盲区"（§7.1）的又一处。
    几何要点：卷曲的**大圆在第四线附近**（F 线），两个点夹在第三、四线之间的两侧。
    """
    w = _lw(space)
    h = bottom - top
    cx = x + CLEF_W * space * 0.42
    f_line = top + space * 3.0                    # 第四线（自上而下第 4 条）
    # 主体：从上方起笔、向右下再折向左下的粗弧
    r = space * 0.95
    d.arc([cx - r, f_line - r * 1.25, cx + r, f_line + r * 0.55],
          start=300, end=170, fill=0, width=max(2, int(space * 0.16)))
    d.line([(cx + r * 0.62, f_line - r * 1.5), (cx + r * 0.62, f_line - r * 0.1)],
           fill=0, width=max(2, int(space * 0.16)))
    # 两个点（在 F 线右侧上下各一个）
    dr = max(1.5, space * 0.13)
    for dy in (-space * 0.5, space * 0.5):
        cy = f_line + dy
        d.ellipse([cx + r * 1.15 - dr, cy - dr, cx + r * 1.15 + dr, cy + dr], fill=0)


def _draw_ledger_lines(d: ImageDraw.ImageDraw, cx: float, step: int, space: float,
                       y_bottom_line: float) -> list[tuple[float, float, float, float]]:
    """加线（ledger lines）：符头超出谱表时必须画的短横线。

    ★ 为什么现在才加：`detect.find_ledger_lines` 一直是**死代码**，
    `render.py` 也不画 —— 于是合成集里从来没有"符头上下出现长横条"这个场景，
    而修复 12（空心判据）与修复 19（"又长又薄就不是符头"判据）的**残余风险
    全都落在这里**：加线正是"长而薄的横条，且穿过符头"。
    真实钢琴谱里加线非常常见，**不画就等于在骗自己的门槛**。

    返回画过的每条线的 (x0, y, x1, 线宽)，供真值日志使用。
    """
    out: list[tuple[float, float, float, float]] = []
    lw = _lw(space)
    half = space * 0.72                     # 加线比符头略宽
    # `step` 的刻度：0 = 最下面那条谱线，8 = 最上面那条线，每 +1 是半个 space。
    # 所以加线只出现在**偶数** step 上：外面第一条在 10（或 -2），再往外 12（-4）……
    if 0 <= step <= 8:
        return out
    ss = [s for s in range(10, int(step) + 1, 2)] if step > 8 \
        else [s for s in range(-2, int(step) - 1, -2)]
    for s in ss:
        y = y_bottom_line - s * space / 2.0
        d.line([(cx - half, y), (cx + half, y)], fill=0, width=lw)
        out.append((cx - half, y, cx + half, float(lw)))
    return out


def _draw_time_signature(d: ImageDraw.ImageDraw, x: float, top: float, space: float,
                         ts: tuple[int, int]) -> None:
    """拍号：分子在上两间，分母在下两间。用位图字体放大绘制。"""
    num, den = ts
    try:
        font = ImageFont.load_default(size=int(space * 2.55))
    except TypeError:  # 老版本 Pillow 不支持 size 参数
        font = ImageFont.load_default()
    y_num = top + space * 1.0
    y_den = top + space * 3.0
    for text, cy in ((str(num), y_num), (str(den), y_den)):
        bb = d.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        d.text((x + TS_W * space / 2 - tw / 2 - bb[0], cy - th / 2 - bb[1]),
               text, font=font, fill=0)


def _draw_notehead(d: ImageDraw.ImageDraw, cx: float, cy: float, space: float,
                   hollow: bool) -> None:
    w = NOTEHEAD_W * space
    h = NOTEHEAD_H * space
    box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    if hollow:
        d.ellipse(box, outline=0, width=max(2, int(space * 0.17)))
    else:
        d.ellipse(box, fill=0)


def _draw_dot(d: ImageDraw.ImageDraw, cx: float, cy: float, space: float) -> None:
    """附点（延长记号）。半径约 0.15 * space。"""
    r = max(1.5, space * 0.15)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)


def _draw_slur(d: ImageDraw.ImageDraw, x0: float, y0: float,
               x1: float, y1: float, space: float) -> tuple[float, float, float, float]:
    """Draw a curved phrase mark and return its bounding box."""
    if x1 <= x0:
        return x0, y0, 0.0, 0.0
    start = np.asarray((x0, y0 + 0.72 * space), dtype=np.float64)
    end = np.asarray((x1, y1 + 0.72 * space), dtype=np.float64)
    control = np.asarray(((x0 + x1) / 2, max(y0, y1) + 1.55 * space),
                         dtype=np.float64)
    points = []
    for t in np.linspace(0.0, 1.0, 33):
        point = (1 - t) ** 2 * start + 2 * t * (1 - t) * control + t ** 2 * end
        points.append((float(point[0]), float(point[1])))
    d.line(points, fill=0, width=max(2, _lw(space) // 2))
    ys = [point[1] for point in points]
    return x0, min(ys), x1 - x0, max(ys) - min(ys)


def _stem_x(head_cx: float, space: float, up: bool) -> float:
    """符干贴的那一列 x（符头左/右边缘向内半个线宽）。

    ★ 单独抽出来是为了让"画符干的"和"画符杠/符尾的"**用同一个 x** ——
    两条路径各算一次，迟早会算出两个不同的值（本项目踩过这类坑）。
    """
    w = _lw(space)
    return (head_cx + (NOTEHEAD_W * space / 2 - w * 0.5) if up
            else head_cx - (NOTEHEAD_W * space / 2 - w * 0.5))


def _draw_stem(d: ImageDraw.ImageDraw, head_cx: float, cy: float, space: float,
               up: bool, hollow: bool) -> float:
    """画符干，返回符干末端的 y。"""
    w = _lw(space)
    # 符干贴在符头的一侧
    x = _stem_x(head_cx, space, up)
    y_end = cy - STEM_LEN * space if up else cy + STEM_LEN * space
    d.line([(x, cy), (x, y_end)], fill=0, width=w)
    return y_end


def _draw_flags(d: ImageDraw.ImageDraw, stem_x: float, y_end: float, space: float,
                up: bool, count: int) -> None:
    """符尾（八分音符 1 条，十六分 2 条）。"""
    sgn = 1.0 if up else -1.0
    for i in range(count):
        y0 = y_end + sgn * i * space * 0.62
        w = space * 0.75
        h = space * 0.95
        if up:
            pts = [(stem_x, y0), (stem_x + w, y0 + h * 0.45), (stem_x + w * 0.86, y0 + h * 0.95),
                   (stem_x, y0 + h * 0.55)]
        else:
            pts = [(stem_x, y0), (stem_x + w, y0 - h * 0.45), (stem_x + w * 0.86, y0 - h * 0.95),
                   (stem_x, y0 - h * 0.55)]
        d.polygon(pts, fill=0)


#: 能被符杠连起来的时值（八分及更短）
BEAMABLE_FLAGS = {"eighth": 1, "triplet_eighth": 1,
                  "dotted_eighth": 1, "sixteenth": 2}
#: 一组符杠最多连几个音符（真实版面上常见 2~4 个）
BEAM_GROUP_MAX = 4


def _plan_beams(meas: list, weights: list[float], unit: float, cx0: float,
                y_bottom_line: float, space: float, clef: str,
                mid_y: float, idx0: int = 0) -> dict[int, dict]:
    """预先规划小节内的**符杠分组**，并给出每条音符的符干末端 y。

    ★ 为什么必须"先规划再画"（而不是边画边接）：符杠是一条**从组首符干末端
    连到组尾符干末端**的斜梁，**组内每一根符干都要正好停在梁上**。
    若按每个音符各自的固定符干长度先画、再补一根梁，中间那些符干就会
    **戳出梁外**（长短不一），与真实版面不符 —— 而"符干末端有没有横条"
    正是时值判据（`reconstruct.count_beams`）要量的东西，
    几何不对就会把探针也一起带偏。

    返回 `{音符序号: {"group": [...], "beam_y": y, "is_last": bool, ...}}`。
    """
    # 1. 先按与主循环**完全一致**的规则算位置（同一套 weights / unit）
    infos: list[dict] = []
    cx = cx0
    for i, spec in enumerate(meas):
        # ⚠️ `weights` 是**整首曲子的全局权重表**（间距按全局分配），
        #    所以这里要加上本小节在全局里的起始下标 —— 用错下标会让
        #    "探针算出来的位置"和"主循环画出来的位置"不一致，
        #    于是梁和符干对不上（本项目最容易犯的那类错：两条路径不同构）。
        w_here = unit * weights[idx0 + i]
        hx = cx + w_here / 2
        cx += w_here
        if spec.is_rest:
            infos.append({"rest": True, "i": i})
            continue
        letter, octave, _off = theory.parse_pitch_name(spec.pitch)
        step = theory.pitch_to_step(letter, octave, clef)
        cy = theory.step_to_y(step, y_bottom_line, space)
        infos.append({
            "rest": False, "i": i, "hx": hx, "cy": cy,
            "up": cy > mid_y,
            "flags": BEAMABLE_FLAGS.get(spec.duration, 0),
            "hollow": spec.ticks >= 2 * PPQ,
        })

    # 2. 分组：相邻的"可连梁"音符连成一组；休止符、长时值、或超过上限就断开
    plan: dict[int, dict] = {}
    gid = 0
    run: list[dict] = []

    def flush(run: list[dict]) -> None:
        nonlocal gid
        if len(run) < 2:
            return                      # 单个音符仍然画符尾，不连梁
        gid += 1
        # 组内统一方向：以**多数**为准（真实版面也是这样，取组的重心）
        ups = sum(1 for n in run if n["up"])
        up = ups * 2 >= len(run)
        tips = []
        for n in run:
            sign = -1.0 if up else 1.0
            tips.append(n["cy"] + sign * STEM_LEN * space)
        # 梁的上下沿：从组首末端连一条线，再按最大梁数给厚度
        y_first, y_last = tips[0], tips[-1]
        n_beams = max(n["flags"] for n in run)
        for k, n in enumerate(run):
            t = 0.0 if len(run) == 1 else k / (len(run) - 1)
            beam_y = y_first + (y_last - y_first) * t
            plan[n["i"]] = {
                "group": [m["i"] for m in run], "k": k, "n": len(run),
                "up": up, "beam_y": beam_y, "beams": n_beams,
                "flags": n["flags"], "hx": n["hx"], "cy": n["cy"],
            }
        # 组首/组尾的梁线端点（供画斜梁用）
        for k, n in enumerate(run):
            plan[n["i"]]["y_first"] = y_first
            plan[n["i"]]["y_last"] = y_last

    for n in infos:
        if n.get("rest") or n["flags"] == 0 or n["hollow"]:
            if run:
                flush(run)
                run = []
            continue
        # 方向变了也断开（真实版面一个梁组方向一致）
        if run and n["up"] != run[0]["up"]:
            flush(run)
            run = []
        run.append(n)
        if len(run) >= BEAM_GROUP_MAX:
            flush(run)
            run = []
    if run:
        flush(run)
    return plan


def _draw_beam(d: ImageDraw.ImageDraw, group: list[dict], space: float,
               up: bool, beams: int) -> None:
    """画一组符杠：主梁 + （十六分时的）次梁。

    ★ 梁厚取 `0.35 * space`，梁间**中心距**取 `0.62 * space` —— 这两个数是
    **从真实照片反推的**，不是随手定的：

    * 真实照片上检出的符杠，`_v_extent` 量出来的厚度是 **0.53~0.57 space**
      （含二值化的膨胀）；画厚了会让**两根梁之间的白缝消失**；
    * 白缝一旦消失，十六分音符的双梁就变成"**一根 1.1 space 厚的实心横条**" ——
      而实心符头恰好是 0.95 space 高，于是渲染出来的**合成页会比真实版面更容易
      骗过判据**（实测：`beams16` 对照页上多检出 60 个"音符"，
      其中一部分正是"双梁糊成一块"造成的）。
    * 真实版面的双梁在手机翻拍分辨率下**是分得开的**（裁图可见两条梁之间有条白缝），
      所以合成页也必须画得开，否则**合成集与真实场景就不是同一个问题了**。
    """
    if len(group) < 2:
        return
    thick = max(2.0, space * 0.35)
    sign = -1.0 if up else 1.0
    y_first, y_last = group[0]["beam_y"], group[-1]["beam_y"]
    x_first, x_last = group[0]["stem_x"], group[-1]["stem_x"]
    if x_last < x_first:
        x_first, x_last = x_last, x_first
        y_first, y_last = y_last, y_first
    def segment(x0: float, y0: float, x1: float, y1: float,
                level: int) -> None:
        # Secondary beams lie towards the heads, inside the primary beam.
        off = -sign * level * space * 0.62
        d.polygon([(x0, y0 + off - thick / 2),
                   (x1, y1 + off - thick / 2),
                   (x1, y1 + off + thick / 2),
                   (x0, y0 + off + thick / 2)], fill=0)

    segment(x_first, y_first, x_last, y_last, 0)
    # A second beam belongs only to notes requiring a second flag.  In a
    # mixed eighth/sixteenth group, extending it over all stems would print
    # the eighths as sixteenths while keeping eighths in the ground truth.
    for level in range(1, beams):
        i = 0
        while i < len(group):
            if group[i]["flags"] <= level:
                i += 1
                continue
            start = i
            while i + 1 < len(group) and group[i + 1]["flags"] > level:
                i += 1
            end = i
            if end > start:
                segment(group[start]["stem_x"], group[start]["beam_y"],
                        group[end]["stem_x"], group[end]["beam_y"], level)
            else:
                # A lone sixteenth uses a short hook.  It reaches less than
                # halfway to its neighbor, so that neighbor remains an eighth.
                neighbor = start + 1 if start + 1 < len(group) else start - 1
                a = group[start]
                b = group[neighbor]
                segment(a["stem_x"], a["beam_y"],
                        a["stem_x"] + 0.42 * (b["stem_x"] - a["stem_x"]),
                        a["beam_y"] + 0.42 * (b["beam_y"] - a["beam_y"]),
                        level)
            i += 1


def _draw_accidental(d: ImageDraw.ImageDraw, x: float, cy: float, space: float,
                     kind: str) -> None:
    """升号 / 降号 / 还原号。"""
    w = _lw(space)
    h = space * 1.9
    cw = ACC_W * space
    if kind == "sharp":
        d.line([(x + cw * 0.30, cy - h / 2), (x + cw * 0.30, cy + h / 2)], fill=0, width=w)
        d.line([(x + cw * 0.72, cy - h / 2), (x + cw * 0.72, cy + h / 2)], fill=0, width=w)
        d.line([(x, cy - space * 0.28), (x + cw, cy - space * 0.50)], fill=0, width=w)
        d.line([(x, cy + space * 0.50), (x + cw, cy + space * 0.28)], fill=0, width=w)
    elif kind == "flat":
        d.line([(x + cw * 0.22, cy - h / 2), (x + cw * 0.22, cy + h * 0.30)], fill=0, width=w)
        d.arc([x + cw * 0.10, cy - space * 0.10, x + cw * 1.05, cy + space * 1.15],
              start=270, end=90, fill=0, width=w)
    elif kind == "natural":
        d.line([(x + cw * 0.25, cy - h / 2), (x + cw * 0.25, cy + h * 0.35)], fill=0, width=w)
        d.line([(x + cw * 0.75, cy - h * 0.35), (x + cw * 0.75, cy + h / 2)], fill=0, width=w)
        d.line([(x + cw * 0.25, cy - space * 0.22), (x + cw * 0.75, cy - space * 0.42)], fill=0, width=w)
        d.line([(x + cw * 0.25, cy + space * 0.42), (x + cw * 0.75, cy + space * 0.22)], fill=0, width=w)
    else:
        raise ValueError(f"未知临时记号: {kind}")


def _draw_rest(d: ImageDraw.ImageDraw, cx: float, top: float, space: float, kind: str) -> None:
    """休止符。draw 的 y 以谱表 top 为基准。"""
    w = _lw(space)
    line_y = lambda i: top + (4 - i) * space  # i=0 是最下面那条线
    rw = space * 1.15
    rh = space * 0.48
    if kind == "whole":
        y = line_y(3)          # 全休止符挂在第 4 线下方
        d.rectangle([cx - rw / 2, y, cx + rw / 2, y + rh], fill=0)
    elif kind == "half":
        y = line_y(2)          # 二分休止符坐在第 3 线上
        d.rectangle([cx - rw / 2, y - rh, cx + rw / 2, y], fill=0)
    elif kind == "quarter":
        # 简化四分休止符：一个 Z 形折线
        y0, y1 = line_y(3) - space * 0.2, line_y(1) + space * 0.2
        pts = [(cx - space * 0.28, y0), (cx + space * 0.28, y0 + space * 0.5),
               (cx - space * 0.30, y0 + space * 1.1), (cx + space * 0.34, y1),
               (cx - space * 0.05, y1 - space * 0.55)]
        d.line(pts, fill=0, width=w, joint="curve")
    else:
        raise ValueError(f"未知休止符: {kind}")


def _draw_barline(d: ImageDraw.ImageDraw, x: float, top: float, space: float, final: bool) -> None:
    w = _lw(space)
    bottom = top + 4 * space
    d.line([(x, top), (x, bottom)], fill=0, width=w)
    if final:
        d.line([(x + space * 0.45, top), (x + space * 0.45, bottom)], fill=0, width=w * 2)


# ---------------------------------------------------------------------------
# 乐谱定义
# ---------------------------------------------------------------------------


@dataclass
class NoteSpec:
    """一个待渲染的事件。`pitch` 为音名字符串（如 "F#4"），休止符用 pitch="" 。"""

    pitch: str
    duration: str          # 时值名，支持 "dotted_quarter"
    accidental: str | None = None   # 仅当需要显式写出临时记号时
    is_rest: bool = False
    chord_pitches: tuple[str, ...] = ()  # 与主音同一 onset 的附加音高
    slur_start: bool = False
    slur_end: bool = False

    @property
    def ticks(self) -> int:
        return theory.ticks(self.duration)


#: 默认测试旋律：贝多芬《欢乐颂》开头，C 大调，4/4，4 小节单声部。
#:
#: 特意覆盖了最容易出错的几类情况：
#:   * 第 2 小节一个显式升号   -> 测临时记号的渲染/检测/应用
#:   * 第 3 小节一个四分休止符 -> 测休止符通路
#:   * 第 4 小节附点四分 + 八分 + 二分 -> 测附点、符尾、空心符头
DEFAULT_MELODY: list[list[NoteSpec]] = [
    [NoteSpec("E4", "quarter"), NoteSpec("E4", "quarter"),
     NoteSpec("F4", "quarter"), NoteSpec("G4", "quarter")],
    [NoteSpec("G4", "quarter"), NoteSpec("F4", "quarter", accidental="sharp"),
     NoteSpec("E4", "quarter"), NoteSpec("D4", "quarter")],
    [NoteSpec("", "quarter", is_rest=True), NoteSpec("C4", "quarter"),
     NoteSpec("D4", "quarter"), NoteSpec("E4", "quarter")],
    [NoteSpec("E4", "dotted_quarter"), NoteSpec("D4", "eighth"),
     NoteSpec("D4", "half")],
]


@dataclass
class GlyphBox:
    """**字形级**真值框（不是事件级）。

    为什么需要它：`Event.bbox` 是"符头 + 符干 + 临时记号"的并集
    （为了方便 UI 高亮），**不能当符号检测的真值** ——
    拿它训检测器，等于把符干、临时记号也算进符头里。
    所以渲染时顺手把每一个**画出来的字形**的框记下来。

    `cls` 用与 `detect.py` 的 `SYM_*` **同一套名字**（符头/休止符/临时记号/附点/谱号），
    另外补上检测链路里需要但不能算"符号"的几何：`stem` / `beam` / `barline` /
    `ledger` / `staff_line`。**同名同义**是刻意的：训练标签和推理输出对不上名，
    后面每一处转换都是出错的机会。
    """

    cls: str
    x: float
    y: float
    w: float
    h: float
    #: 该字形属于第几条谱表（多谱表页面上用来分组/归属）
    staff: int = -1

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def as_list(self) -> list[float]:
        return [float(self.x), float(self.y), float(self.w), float(self.h)]


@dataclass
class RenderOptions:
    space: float = 21.0          # 谱线间距（像素）。真实 A4 拍摄下约 21px
    margin_x: float = 60.0
    margin_y: float = 90.0
    page_w: int = 1700
    page_h: int = 420
    clef: str = "treble"
    key_fifths: int = 0          # 默认 C 大调（无升降号）
    time_signature: tuple[int, int] = (4, 4)
    title: str = "Ode to Joy (synthetic)"
    #: ★ 用**真音乐字体**（Segoe UI Symbol 的 Unicode 音乐符号）画符头/休止符/
    #:   临时记号/附点，而不是几何近似。默认关：打开会改变合成页的字形，
    #:   而 `detect.py` 里的模板是按"几何近似字形"做的 ——
    #:   **模板法的黄金基准不能被改**。CNN 训练集用它（见 `synthdata`）。
    use_music_font: bool = False
    #: 笔画粗细倍率。★ 为什么要这个开关：**真实手机翻拍的笔画比合成页粗**。
    #: 实测真实照片的谱线粗细是 **4px @ space 19.5~24.9 ⇒ 0.16~0.20 space**，
    #: 而 `LINE_W=0.11` 画出来只有 0.11 space（约 2~3px）。
    #: 差 1.5 倍不是小数：二值化后的墨迹分布、去线判据、模板响应都会跟着变。
    #: 合成测试集要能代表真实照片，就必须能画出"粗笔画"这一档。
    stroke_scale: float = 1.0
    #: Optional shared horizontal widths for paired staves.  Each measure's
    #: notes are spaced inside this width so the barlines align vertically.
    measure_content_widths: list[float] | None = None


@dataclass
class RenderResult:
    image: np.ndarray            # uint8 灰度图，0=黑 255=白
    truth: Score                 # 精确 ground truth
    options: RenderOptions
    #: 谱表几何，供调试/评测用
    staff_top: float = 0.0
    y_bottom_line: float = 0.0
    #: ★ **字形级真值框**（训练检测器/评测检测器用），见 `GlyphBox`
    glyphs: list[GlyphBox] = field(default_factory=list)
    #: ★ **每条谱表的几何**（space / 5 条线的 y / 左右范围 / 谱号）。
    #:   `truth.parts[0].staves` 在多系统页面上是**合并成一个**的
    #:   （见 `stack_systems`），拿不到"第 k 行谱表在哪" ——
    #:   而裁"谱表带"训练 CNN 恰恰需要它。所以单独记一份。
    staff_geoms: list[dict] = field(default_factory=list)


def render_score(
    melody: list[list[NoteSpec]] | None = None,
    options: RenderOptions | None = None,
    seed: int = 0,
) -> RenderResult:
    """把旋律渲染成灰度图，并返回精确的 ground truth。

    Returns
    -------
    RenderResult
        `image` 是 uint8 灰度图（0=黑，255=白），`truth` 是与图像逐像素对齐的标准答案。
    """
    melody = melody or DEFAULT_MELODY
    o = options or RenderOptions()

    img = Image.new("L", (o.page_w, o.page_h), 255)
    d = ImageDraw.Draw(img)

    global _STROKE_SCALE
    _STROKE_SCALE = float(o.stroke_scale)

    space = o.space
    top = o.margin_y
    bottom = top + 4 * space
    y_bottom_line = bottom
    lw = _lw(space)

    # --- 谱表 -------------------------------------------------------------
    #: ★ 字形级真值框的收集器（见 `GlyphBox`）。渲染与"记真值"在**同一处**做，
    #:   这样真值框不可能和画出来的东西对不上 —— 反过来（先画完再另写一套
    #:   几何推导来算框）就是本项目反复踩的"两条路径不同构"。
    gbox: list[GlyphBox] = []
    for i in range(5):
        y = top + i * space
        d.line([(o.margin_x, y), (o.page_w - o.margin_x, y)], fill=0, width=lw)
        gbox.append(GlyphBox("staff_line", o.margin_x, y - lw / 2,
                             o.page_w - 2 * o.margin_x, float(lw)))

    # --- 谱号 + 拍号 -------------------------------------------------------
    x = o.margin_x + space * 0.5
    if o.clef == "bass":
        _draw_bass_clef(d, x, top, bottom, space)
        gbox.append(GlyphBox("clef_bass", x - space * 0.2, top - space * 0.3,
                             CLEF_W * space, 4 * space + space * 0.6))
    else:
        _draw_treble_clef(d, x, top, bottom, space)
        gbox.append(GlyphBox("clef_treble", x - space * 0.2, top - space * 0.1,
                             CLEF_W * space, 4 * space + space * 0.6))
    x += CLEF_W * space
    _draw_time_signature(d, x, top, space, o.time_signature)
    gbox.append(GlyphBox("ts_num", x, top + space * 0.2, TS_W * space, space * 1.6))
    gbox.append(GlyphBox("ts_den", x, top + space * 2.2, TS_W * space, space * 1.6))
    x += TS_W * space + MEASURE_PAD * space

    # --- 预先算好每个音符的位置，保证间距均匀 -------------------------------
    all_notes = [n for meas in melody for n in meas]
    # 按"占位宽度"分配水平空间：时值越长占位越宽，但至少 NOTE_MIN_GAP
    span = o.page_w - o.margin_x - x
    weights = [max(NOTE_MIN_GAP, 1.0 + 0.9 * math.log2(n.ticks / PPQ + 1.6)) for n in all_notes]
    total_w = sum(weights)
    # 每个小节末尾留出小节线空间
    n_measures = len(melody)
    bar_space = MEASURE_PAD * space * 2 * n_measures
    unit = (span - bar_space) / total_w
    if unit <= 0:
        raise ValueError("页面太窄，放不下内容；增大 page_w 或减小 space")
    if o.measure_content_widths is not None:
        if (len(o.measure_content_widths) != n_measures or
                any(width <= 0 for width in o.measure_content_widths) or
                abs(sum(o.measure_content_widths) - (span - bar_space)) > 1.0):
            raise ValueError("measure_content_widths must fit the available measure span")

    # --- 逐小节渲染 --------------------------------------------------------
    truth_staff = Staff(
        number=1,
        line_ys=[top + i * space for i in range(5)],
        space=space,
        clef=o.clef,
        key_fifths=o.key_fifths,
        x0=o.margin_x,
        x1=o.page_w - o.margin_x,
    )
    truth = Score(
        parts=[Part(name="Melody", staves=[truth_staff])],
        title=o.title,
        confidence=1.0,
    )

    key_map = theory.key_signature_map(o.key_fifths)
    idx = 0
    cx = x
    for mi, meas in enumerate(melody):
        m_num = mi + 1
        m_events: list[Event] = []
        onset = 0
        triplet_index = 0
        measure_acc: dict[tuple[str, int], int] = {}
        open_slur: tuple[float, float] | None = None
        # ★ 先规划本小节的**符杠分组**（含每根符干应该停在哪条梁上）——
        #   必须"先规划再画"：梁是一条从组首符干末端连到组尾符干末端的斜线，
        #   组内每根符干都要正好停在梁上，否则中间那些符干会戳出梁外。
        measure_unit = (o.measure_content_widths[mi] /
                        sum(weights[idx:idx + len(meas)])) if o.measure_content_widths else unit
        beam_plan = _plan_beams(meas, weights, measure_unit, cx, y_bottom_line, space,
                                o.clef, top + 2 * space, idx0=idx)
        mi_idx = 0
        for spec in meas:
            w_here = measure_unit * weights[idx]
            idx += 1
            bplan = beam_plan.get(mi_idx)
            mi_idx += 1

            if spec.duration == "triplet_eighth":
                tuplet = ("start", "continue", "stop")[triplet_index]
                triplet_index = (triplet_index + 1) % 3
            else:
                if triplet_index:
                    raise ValueError("Triplet eighths must occur in complete groups of three")
                tuplet = None

            if spec.is_rest:
                if spec.chord_pitches:
                    raise ValueError("A rest cannot contain chord pitches")
                rx = cx + w_here / 2
                _rk = spec.duration if spec.duration in ("whole", "half", "quarter") else "quarter"
                if o.use_music_font and music_font_path():
                    # 真字体：字号按"全/二分休止符乐谱惯例高 0.48 space" 定，
                    # 但**框以实际墨迹为准**（返回的 bbox 直接当真值）
                    _h = 0.48 if _rk in ("whole", "half") else 2.4
                    _bb = _draw_music_glyph(img, MUSIC_CP[f"rest_{_rk}"], rx,
                                            top + space * (1.0 if _rk == "whole"
                                                           else 2.0), space, _h)
                    gbox.append(GlyphBox(f"rest_{_rk}", *_bb))
                else:
                    _draw_rest(d, rx, top, space, _rk)
                    if _rk in ("whole", "half"):
                        _ly = top + space if _rk == "whole" else top + 2 * space
                        _ry = _ly if _rk == "whole" else _ly - space * 0.48
                        gbox.append(GlyphBox(f"rest_{_rk}", rx - space * 0.575, _ry,
                                             space * 1.15, space * 0.48))
                    else:
                        gbox.append(GlyphBox("rest_quarter", rx - space * 0.34,
                                             top + space * 1.0, space * 0.68,
                                             space * 2.4))
                bbox = BBox(rx - space * 0.6, top + space, space * 1.2, space * 3)
                m_events.append(Event(
                    type="rest", onset_ticks=onset, duration_ticks=spec.ticks,
                    confidence=1.0, bbox=bbox,
                ))
                onset += spec.ticks
                cx += w_here
                continue

            letter, octave, off_in_name = theory.parse_pitch_name(spec.pitch)
            step = theory.pitch_to_step(letter, octave, o.clef)
            cy = theory.step_to_y(step, y_bottom_line, space)
            hx = cx + w_here / 2
            if spec.slur_start:
                open_slur = (hx, cy)

            # 应用调号 / 临时记号，得到实际发声音高
            explicit = spec.accidental
            offset = theory.resolve_accidental(letter, explicit, o.key_fifths,
                                               measure_acc, octave)
            if explicit:
                measure_acc[(letter, octave)] = theory.ACCIDENTAL_OFFSET[explicit]

            # 临时记号字形
            acc_x = None
            if explicit:
                # Leave a real gap between the accidental and notehead.
                # The old 1.15-space placement made a sharp's right stroke
                # overlap the head, erasing much of its identifying shape.
                acc_x = hx - space * 1.65
                if o.use_music_font and music_font_path():
                    _bb = _draw_music_glyph(img, MUSIC_CP[f"accidental_{explicit}"],
                                            acc_x + ACC_W * space / 2, cy, space, 1.9)
                    gbox.append(GlyphBox(f"accidental_{explicit}", *_bb))
                else:
                    _draw_accidental(d, acc_x, cy, space, explicit)
                    gbox.append(GlyphBox(f"accidental_{explicit}", acc_x - space * 0.08,
                                         cy - space * 0.95, ACC_W * space + space * 0.16,
                                         space * 1.9))

            # 加线（符头超出谱表时）
            for _lx0, _ly, _lx1, _llw in _draw_ledger_lines(d, hx, step, space,
                                                            y_bottom_line):
                gbox.append(GlyphBox("ledger", _lx0, _ly - _llw / 2,
                                     _lx1 - _lx0, _llw))

            # 符头
            hollow = spec.ticks >= 2 * PPQ
            if o.use_music_font and music_font_path():
                _bb = _draw_music_glyph(
                    img, MUSIC_CP["notehead_hollow" if hollow else "notehead_filled"],
                    hx, cy, space, NOTEHEAD_H)
                gbox.append(GlyphBox("notehead_hollow" if hollow
                                     else "notehead_filled", *_bb))
            else:
                _draw_notehead(d, hx, cy, space, hollow)
                gbox.append(GlyphBox("notehead_hollow" if hollow else "notehead_filled",
                                     hx - NOTEHEAD_W * space / 2,
                                     cy - NOTEHEAD_H * space / 2,
                                     NOTEHEAD_W * space, NOTEHEAD_H * space))

            chord_members: list[tuple[str, int, int, float]] = []
            for chord_pitch in spec.chord_pitches:
                chord_letter, chord_octave, chord_written_offset = theory.parse_pitch_name(chord_pitch)
                chord_step = theory.pitch_to_step(chord_letter, chord_octave, o.clef)
                chord_y = theory.step_to_y(chord_step, y_bottom_line, space)
                for _lx0, _ly, _lx1, _llw in _draw_ledger_lines(
                        d, hx, chord_step, space, y_bottom_line):
                    gbox.append(GlyphBox("ledger", _lx0, _ly - _llw / 2,
                                         _lx1 - _lx0, _llw))
                if o.use_music_font and music_font_path():
                    _bb = _draw_music_glyph(
                        img, MUSIC_CP["notehead_hollow" if hollow else "notehead_filled"],
                        hx, chord_y, space, NOTEHEAD_H)
                    gbox.append(GlyphBox("notehead_hollow" if hollow
                                         else "notehead_filled", *_bb))
                else:
                    _draw_notehead(d, hx, chord_y, space, hollow)
                    gbox.append(GlyphBox("notehead_hollow" if hollow else "notehead_filled",
                                         hx - NOTEHEAD_W * space / 2,
                                         chord_y - NOTEHEAD_H * space / 2,
                                         NOTEHEAD_W * space, NOTEHEAD_H * space))
                chord_members.append((chord_letter, chord_octave,
                                      chord_written_offset, chord_y))

            # 附点（时值后缀 dotted_）
            dot_x = None
            if spec.duration.startswith("dotted_"):
                dot_x = hx + NOTEHEAD_W * space / 2 + space * 0.62
                # An augmentation dot next to a note on a staff/ledger line
                # belongs in the space above it.  Putting it on the line
                # merges it with staff ink and creates invalid printed truth.
                dot_y = cy - space / 2 if step % 2 == 0 else cy
                if o.use_music_font and music_font_path():
                    _bb = _draw_music_glyph(img, MUSIC_CP["augmentation_dot"],
                                            dot_x, dot_y, space, 0.30)
                    gbox.append(GlyphBox("augmentation_dot", *_bb))
                else:
                    _draw_dot(d, dot_x, dot_y, space)
                    _dr = max(1.5, space * 0.15)
                    gbox.append(GlyphBox("augmentation_dot", dot_x - _dr, dot_y - _dr,
                                         2 * _dr, 2 * _dr))

            # 符干与符尾 / **符杠**
            up = cy > (top + 2 * space)   # 中线以下符干朝上
            y_end = None
            if spec.duration != "whole":
                if bplan is not None:
                    # ★ 连梁：符干画到**梁线**上（组内统一方向、统一梁线），
                    #   组尾那一个负责把整条梁画出来。
                    up = bool(bplan["up"])
                    stem_x = _stem_x(hx, space, up)
                    y_end = float(bplan["beam_y"])
                    d.line([(stem_x, cy), (stem_x, y_end)], fill=0, width=lw)
                    bplan["stem_x"] = stem_x
                    gbox.append(GlyphBox("stem", stem_x - lw / 2, min(cy, y_end),
                                         float(lw), abs(y_end - cy) + lw))
                    if bplan["k"] == bplan["n"] - 1:
                        _draw_beam(d, [beam_plan[i] for i in bplan["group"]],
                                   space, up, int(bplan["beams"]))
                        if (len(bplan["group"]) == 3 and
                                all(meas[i].duration == "triplet_eighth"
                                    for i in bplan["group"])):
                            bx = sum(beam_plan[i]["hx"] for i in bplan["group"]) / 3
                            by3 = (min(beam_plan[i]["beam_y"] for i in bplan["group"])
                                   - space * 1.05 if up else
                                   max(beam_plan[i]["beam_y"] for i in bplan["group"])
                                   + space * 0.8)
                            try:
                                font = ImageFont.truetype("arial.ttf", max(12, int(space * 0.8)))
                            except OSError:
                                font = ImageFont.load_default()
                            d.text((bx, by3), "3", fill=0, font=font, anchor="mm")
                            gbox.append(GlyphBox("tuplet_number", bx - space * 0.3,
                                                 by3 - space * 0.45,
                                                 space * 0.6, space * 0.9))
                        grp = [beam_plan[i] for i in bplan["group"]]
                        sx = [g["stem_x"] for g in grp]
                        by = [g["beam_y"] for g in grp]
                        _bt = max(2.0, space * 0.35)
                        _nb = int(bplan["beams"])
                        _top = min(by[0], by[-1]) - (_nb - 1) * space * 0.62 - _bt / 2
                        _bot = max(by[0], by[-1]) + _bt / 2
                        gbox.append(GlyphBox("beam", min(sx), _top,
                                             max(sx) - min(sx) or 1.0, _bot - _top))
                else:
                    y_end = _draw_stem(d, hx, cy, space, up, hollow)
                    stem_x = _stem_x(hx, space, up)
                    gbox.append(GlyphBox("stem", stem_x - lw / 2, min(cy, y_end),
                                         float(lw), abs(y_end - cy) + lw))
                    flags = BEAMABLE_FLAGS.get(spec.duration, 0)
                    if flags:
                        _draw_flags(d, stem_x, y_end, space, up, flags)
                        _fs = 1.0 if up else -1.0
                        gbox.append(GlyphBox("flag", stem_x, min(y_end, y_end + _fs * (flags - 1) * space * 0.62) - (space * 0.5 if up else 0.0),
                                             space * 0.75, space * 0.95 + (flags - 1) * space * 0.62))

            # bbox：把符头 + 符干 + 临时记号都包进去，方便 UI 高亮
            x0 = hx - NOTEHEAD_W * space / 2
            x1 = hx + NOTEHEAD_W * space / 2
            if acc_x is not None:
                x0 = min(x0, acc_x)
            y0 = min(cy - NOTEHEAD_H * space / 2, y_end if y_end is not None else cy)
            y1 = max(cy + NOTEHEAD_H * space / 2, y_end if y_end is not None else cy)

            m_events.append(Event(
                type="note",
                onset_ticks=onset,
                duration_ticks=spec.ticks,
                pitch=theory.midi_pitch(letter, octave, offset),
                written_name=theory.pitch_name(letter, octave, off_in_name),
                sounding_name=theory.pitch_name(letter, octave, offset),
                accidental=explicit,
                tuplet=tuplet,
                slur="start" if spec.slur_start else "stop" if spec.slur_end else None,
                confidence=1.0,
                bbox=BBox(x0, y0, x1 - x0, y1 - y0),
                stem="up" if up else "down",
                chord_id=(m_num * 100 + mi_idx if chord_members else None),
            ))
            for chord_letter, chord_octave, chord_written_offset, chord_y in chord_members:
                chord_offset = theory.resolve_accidental(
                    chord_letter, None, o.key_fifths, measure_acc, chord_octave)
                m_events.append(Event(
                    type="note", onset_ticks=onset, duration_ticks=spec.ticks,
                    pitch=theory.midi_pitch(chord_letter, chord_octave, chord_offset),
                    written_name=theory.pitch_name(chord_letter, chord_octave,
                                                   chord_written_offset),
                    sounding_name=theory.pitch_name(chord_letter, chord_octave,
                                                    chord_offset),
                    confidence=1.0,
                    bbox=BBox(hx - NOTEHEAD_W * space / 2,
                              chord_y - NOTEHEAD_H * space / 2,
                              NOTEHEAD_W * space, NOTEHEAD_H * space),
                    chord_id=m_num * 100 + mi_idx,
                ))
            if spec.slur_end and open_slur is not None:
                slur_box = _draw_slur(d, open_slur[0], open_slur[1],
                                      hx, cy, space)
                gbox.append(GlyphBox("slur", *slur_box))
                open_slur = None
            onset += spec.ticks
            cx += w_here

        if triplet_index:
            raise ValueError("Triplet eighths must occur in complete groups of three")

        # 小节线
        bar_x = cx + MEASURE_PAD * space
        is_final = mi == len(melody) - 1
        _draw_barline(d, bar_x, top, space, is_final)
        _bw = space * 0.9 if is_final else lw * 1.5
        gbox.append(GlyphBox("barline", bar_x - _bw / 2, top - space * 0.1,
                             _bw, 4 * space + space * 0.2))
        cx = bar_x + MEASURE_PAD * space

        truth_staff.measures.append(Measure(
            number=m_num,
            time_signature=o.time_signature,
            voices=[Voice(id=1, events=m_events)],
            bbox=BBox(o.margin_x, top, bar_x - o.margin_x, 4 * space),
        ))

    arr = np.asarray(img, dtype=np.uint8)
    for g in gbox:                       # 单页就是第 0 条谱表
        g.staff = 0
    return RenderResult(image=arr, truth=truth, options=o,
                        staff_top=top, y_bottom_line=y_bottom_line,
                        glyphs=gbox,
                        staff_geoms=[{"space": space,
                                      "line_ys": [top + i * space for i in range(5)],
                                      "x0": float(o.margin_x),
                                      "x1": float(o.page_w - o.margin_x),
                                      "clef": o.clef}])


def stack_systems(results: list["RenderResult"], space: float,
                  gap_space: float = 3.0, margin_x: float = 40.0,
                  margin_y: float = 60.0) -> "RenderResult":
    """把若干个单谱表页纵向叠成一张多谱表页（模拟真实 A4 乐谱）。

    为什么需要它：单行谱表的页面太"矮"，整页缩放到 1080p 后谱线间距依然够读，
    所以永远走不到 `scroll` 取景分支。**真实的 A4 乐谱一页有 8~12 行谱表**，
    整页塞进 1080p 会让谱线只剩 6px —— 这正是必须做滚动跟随的原因。
    没有这种测试页，滚动路径就只在单元测试里被验证过，没在整页上跑过。
    """
    from PIL import Image as _Image

    if not results:
        raise ValueError("至少需要一个 RenderResult")

    gap = int(round(gap_space * space))
    widths = [r.image.shape[1] for r in results]
    page_w = max(widths) + int(round(2 * margin_x))
    heights = [r.image.shape[0] for r in results]
    page_h = int(round(2 * margin_y)) + sum(heights) + gap * (len(results) - 1)

    canvas = _Image.new("L", (page_w, page_h), 255)
    truth = Score(title=results[0].truth.title, confidence=1.0)
    part = Part(name="Melody")
    lines: list[float] = []
    #: ★ 每张子图的字形真值框也要跟着平移 —— 否则多谱表页的 `GlyphBox`
    #:   会全部落在"原图坐标系"里，训检测器时标签和图像错位（而且是**整体错位**，
    #:   看起来"标签都在图外"，很容易被误判成"渲染没记框"）。
    glyphs: list[GlyphBox] = []
    geoms: list[dict] = []
    y_cursor = int(round(margin_y))
    # ★ 小节号必须**连续重编**。每个 render_score 出来的谱表都是从 1 开始编号的，
    # 直接把 10 个谱表叠起来会得到 1,2,1,2,... 的小节号 —— 真值本身自相矛盾，
    # 基于 (小节号, onset) 对齐的评测于是完全失效（实测准确率掉到 3.8%）。
    measure_no = 0

    for ri, r in enumerate(results):
        src = _Image.fromarray(r.image)
        # 原图的谱表左边缘在 r.options.margin_x，这里重新贴到 margin_x
        dx = int(round(margin_x)) - int(round(r.options.margin_x))
        canvas.paste(src, (dx, y_cursor))
        dy = float(y_cursor)

        st_src = r.truth.parts[0].staves[0]
        lines.extend([y + dy for y in st_src.line_ys])
        for g in r.glyphs:
            glyphs.append(GlyphBox(g.cls, g.x + dx, g.y + dy, g.w, g.h, staff=ri))
        for gm in r.staff_geoms:
            geoms.append({**gm,
                          "line_ys": [y + dy for y in gm["line_ys"]],
                          "x0": float(gm["x0"]) + dx,
                          "x1": float(gm["x1"]) + dx})

        new_measures: list[Measure] = []
        for m in st_src.measures:
            evs: list[Event] = []
            for v in m.voices:
                for e in v.events:
                    ev = Event(**{**e.__dict__})
                    if e.bbox is not None:
                        ev.bbox = BBox(e.bbox.x + dx, e.bbox.y + dy, e.bbox.w, e.bbox.h)
                    evs.append(ev)
            mb = None
            if m.bbox is not None:
                mb = BBox(m.bbox.x + dx, m.bbox.y + dy, m.bbox.w, m.bbox.h)
            measure_no += 1
            new_measures.append(Measure(number=measure_no, page=1,
                                        time_signature=m.time_signature,
                                        voices=[Voice(id=1, events=evs)], bbox=mb))
        part.staves.append(Staff(number=ri + 1, measures=new_measures,
                                 space=space, clef=st_src.clef,
                                 key_fifths=st_src.key_fifths,
                                 x0=st_src.x0 + dx, x1=st_src.x1 + dx))
        y_cursor += heights[ri] + gap

    # 合并所有谱表到同一个 Staff（本项目首版单声部，谱表几何按第一行记，
    # 供取景决策用 space 即可）
    merged = Staff(number=1, measures=[m for s in part.staves for m in s.measures],
                   line_ys=lines, space=space, clef=part.staves[0].clef,
                   key_fifths=part.staves[0].key_fifths,
                   x0=margin_x, x1=page_w - margin_x)
    truth.parts = [Part(name="Melody", staves=[merged])]

    arr = np.asarray(canvas, dtype=np.uint8)
    opts = RenderOptions(space=space, page_w=page_w, page_h=page_h)
    return RenderResult(image=arr, truth=truth, options=opts,
                        staff_top=margin_y, y_bottom_line=margin_y + 4 * space,
                        glyphs=glyphs, staff_geoms=geoms)


def make_a4_like_page(systems: int = 10, space: float = 21.0,
                      seed: int = 0) -> "RenderResult":
    """造一张接近真实 A4 的多谱表乐谱页（默认 10 行谱表）。

    尺寸约 1700 x 1500+，等比放大到 A4/300DPI 就是 2480x3508 —— 也就是
    真实拍摄会得到的图像。用它来验证滚动取景。

    ★★ 踩过的坑：这个函数的 `seed` 以前**根本没被用上** ——
    `rng = np.random.default_rng(seed)` 建完就在函数末尾 `del rng` 了，
    旋律完全由固定的 `scales` 表决定。后果不是"图不对"，
    而是**统计上彻底虚假**：所有 seed 出的是同一张图，
    所以"多种子平均"其实是同一张图跑 N 遍，`min == max == mean`，
    看起来"方差为 0、结果非常稳定"，实际上一个样本都没增加。

    这个坑很隐蔽：脚本跑得通、数字很整齐、报告很好看，
    只有去核对"不同 seed 的图像是否真的不同"才会发现。
    现在 seed 真的参与造旋律（每个谱表从音池里随机取 8 个音）。
    音池固定在同一批音上，所以**不会**跑出谱表范围影响基准准确率。
    """
    rng = np.random.default_rng(seed)
    scales = [
        ["E4", "F4", "G4", "A4", "G4", "F4", "E4", "D4"],
        ["C4", "E4", "G4", "C5", "B4", "G4", "E4", "C4"],
        ["G4", "A4", "B4", "C5", "D5", "C5", "B4", "A4"],
        ["F4", "A4", "C5", "A4", "F4", "D4", "F4", "A4"],
    ]
    pool = sorted({p for sc in scales for p in sc})
    results = []
    for i in range(systems):
        # seed=0 时保持原样（老用例、老基线全部不变，避免把历史结论作废）
        if seed == 0:
            pat = scales[i % len(scales)]
        else:
            idx = rng.integers(0, len(pool), size=8)
            pat = [pool[int(k)] for k in idx]
        # 全用四分音符：每小节 4 拍，拍号校验和 infer_time_signature 都不会被扰动
        meas = [[NoteSpec(p, "quarter") for p in pat[:4]],
                [NoteSpec(p, "quarter") for p in pat[4:]]]
        r = render_score(meas, RenderOptions(space=space,
                                             title=f"A4-like page ({systems} systems)"))
        results.append(r)
    return stack_systems(results, space=space)


# ---------------------------------------------------------------------------
# 退化模拟 —— 用来专测 preprocess.py
# ---------------------------------------------------------------------------


@dataclass
class DegradeOptions:
    illumination: float = 0.55     # 光照梯度强度 0..1（模拟侧光/整体明暗不均）
    noise_sigma: float = 6.0       # 高斯噪声标准差
    blur_sigma: float = 0.8        # 高斯模糊
    rotation_deg: float = 0.0      # 旋转（模拟倾斜拍摄）
    jpeg_quality: int = 0          # >0 时做一次 JPEG 往返，模拟压缩
    speckle: float = 0.0           # 椒盐噪声比例 0..1
    shadow: float = 0.0            # 局部阴影强度 0..1（★★ 见下）


def _apply_shadow(x: np.ndarray, strength: float, softness: float = 0.06) -> np.ndarray:
    """叠加一片局部阴影（偏心的椭圆暗区，边界柔和）。

    ★★ 这一项比 `illumination` 重要得多，是区分"局部自适应二值化够不够用"的
    关键测试。原因：

    全局线性梯度是**平滑**的 —— 一个 31×31 的局部窗口里，梯度几乎不变，
    所以 Sauvola 之类的局部方法天然就能跟上，根本不需要先做光照归一化。
    实测：光照 0.8 的线性梯度下，"不做任何归一化"和"完整链"都是 100%。

    真实拍摄难的是**阴影边界**（手指、书脊、灯光遮挡）：亮度在几十个像素内
    发生阶跃。这时局部窗口跨越边界，均值被严重拉偏，边界两侧会一边糊成黑块、
    一边把纸纹判成墨迹 —— 这才是必须做光照归一化的真正理由。
    """
    h, w = x.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.72, h * 0.30
    rx, ry = w * 0.35, h * 0.55
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    s = 1.0 / (1.0 + np.exp((d - 1.0) / softness))
    return x * (1.0 - strength * s)


def degrade(img: np.ndarray, opt: DegradeOptions | None = None, seed: int = 0) -> np.ndarray:
    """对干净渲染图叠加各种退化。

    `illumination`（平滑梯度）与 `shadow`（局部阴影边界）是两项不同的考验：
    前者几乎任何局部自适应方法都能扛，**后者才是真正的难点**。
    用两组对比可以直接量化"光照归一化到底救回了多少"。
    """
    o = opt or DegradeOptions()
    rng = np.random.default_rng(seed)
    x = img.astype(np.float32)

    # 光照梯度：沿 x 和 y 各一个线性梯度 + 一个低频起伏
    if o.illumination > 0:
        h, w = x.shape
        gx = np.linspace(-1.0, 1.0, w)[None, :]
        gy = np.linspace(-1.0, 1.0, h)[:, None]
        grad = 1.0 - o.illumination * (0.45 * gx + 0.30 * gy + 0.25 * gx * gy)
        lowfreq = 1.0 + 0.08 * np.sin(np.linspace(0, 5 * math.pi, w))[None, :]
        x = x * grad * lowfreq

    # 局部阴影（阶跃型亮度不连续）
    if o.shadow > 0:
        x = _apply_shadow(x, o.shadow)

    if o.blur_sigma > 0:
        x = np.asarray(Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(o.blur_sigma)), dtype=np.float32)

    if o.noise_sigma > 0:
        x = x + rng.normal(0.0, o.noise_sigma, x.shape)

    if o.rotation_deg:
        im = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
        im = im.rotate(o.rotation_deg, resample=Image.BICUBIC, fillcolor=255)
        x = np.asarray(im, dtype=np.float32)

    if o.speckle > 0:
        mask = rng.random(x.shape) < o.speckle
        x[mask] = rng.choice([0.0, 255.0], size=int(mask.sum()))

    if o.jpeg_quality > 0:
        import io
        im = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=o.jpeg_quality)
        buf.seek(0)
        x = np.asarray(Image.open(buf).convert("L"), dtype=np.float32)

    return np.clip(x, 0, 255).astype(np.uint8)
