"""
五线谱检测与五线谱去除。

这两步是整个 OMR 里**确定性最强、最适合放 PL** 的部分：
  * 谱线检测 = 水平投影 + 分组，纯整数运算
  * 谱线去除 = 垂直游程长度过滤，逐像素可流水

而它们也恰好是最关键的一步：谱表几何（staff space、底线 y）是后续
**所有音高判定的基准**。谱表定错了，后面全错。

谱线去除的难点
--------------
直接把谱线所在的整行抹掉是不行的 —— 符头、符干、加线都跨在谱线上，
全抹会把它们切断。正确做法是**只抹"垂直游程长度很短"的谱线像素**：
  * 纯谱线像素：垂直游程 ≈ 线宽（2~3 px）
  * 符头/符干跨线的像素：垂直游程几十个像素
所以用一个游程长度阈值就能干净地分开两者。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import theory


# ---------------------------------------------------------------------------
# 游程长度（向量化实现）
# ---------------------------------------------------------------------------


def _run_down(b: np.ndarray) -> np.ndarray:
    """每个墨迹像素往上数的连续墨迹长度（含自身）。"""
    h = b.shape[0]
    idx = np.arange(h, dtype=np.int32)[:, None]
    last_nonink = np.where(~b, idx, -1)
    last_nonink = np.maximum.accumulate(last_nonink, axis=0)
    return np.where(b, idx - last_nonink, 0).astype(np.int32)


def vertical_run_length(b: np.ndarray) -> np.ndarray:
    """每个墨迹像素所属的**垂直连续游程总长度**。

    这是谱线去除的核心量。全向量化，无 Python 循环 —— 这个形式
    直接对应 HLS 里的两遍扫描（正扫 + 反扫）。
    """
    return (_run_down(b) + _run_down(b[::-1, :])[::-1] - 1).astype(np.int32)


def horizontal_run_length(b: np.ndarray) -> np.ndarray:
    """每个墨迹像素所属的**水平连续游程总长度**。

    小节线检测的关键判据。单靠"竖向覆盖率"区分不开小节线和符干：
    一根符干加一个符头，在符干那一列上竖向覆盖率也能到 95%。
    但小节线在**每一行**上的水平游程都只有线宽（2~3 px），
    而符干那一列经过符头时水平游程有 20+ px。用这个量一刀切开。
    """
    return vertical_run_length(np.ascontiguousarray(b.T)).T


# ---------------------------------------------------------------------------
# 谱线检测
# ---------------------------------------------------------------------------


@dataclass
class StaffGeometry:
    """一个谱表的几何信息。后续音高判定全部依赖它。"""

    index: int                      # 自上而下第几个谱表（0 起）
    line_ys: list[float]            # 五条谱线的 y（自上而下，y 递增）
    space: float                    # 谱线间距（staff space），像素
    x0: float                       # 谱表水平范围
    x1: float
    thickness: float = 0.0          # 估计的谱线粗细
    local_offsets: list[tuple[float, float]] = field(default_factory=list)
    centre_anchor_safe: bool = False

    @property
    def y_top_line(self) -> float:
        return self.line_ys[0]

    @property
    def y_bottom_line(self) -> float:
        """最下面那条谱线的 y。theory.y_to_step 的基准。"""
        return self.line_ys[-1]

    @property
    def y_middle_line(self) -> float:
        return self.line_ys[2]

    def y_to_step(self, y: float, x: float | None = None) -> float:
        """y 坐标 -> 谱位；有 x 时修正照片中倾斜/弯曲的局部谱线。"""
        offset = 0.0
        if x is not None and self.local_offsets:
            offset = float(np.interp(x, [point[0] for point in self.local_offsets],
                                     [point[1] for point in self.local_offsets]))
        return theory.y_to_step(y, self.y_bottom_line + offset, self.space)

    def step_to_y(self, step: float) -> float:
        return theory.step_to_y(step, self.y_bottom_line, self.space)

    def step_to_pitch(self, step: int, clef: str = "treble") -> tuple[str, int]:
        return theory.step_to_pitch(step, clef)

    def __str__(self) -> str:
        return (f"Staff#{self.index} space={self.space:.2f} "
                f"y=[{self.y_top_line:.1f}..{self.y_bottom_line:.1f}] "
                f"x=[{self.x0:.0f}..{self.x1:.0f}]")


def detect_staff_line_ys(binary: np.ndarray, min_coverage: float = 0.5,
                         max_thickness: int = 16) -> list[float]:
    """找出所有谱线的 y 坐标（像素，浮点）。

    做法：谱线的特征是其所在行的墨迹覆盖率远高于普通文字行
    （谱线几乎横跨整个谱表宽度，而文字行只有零散笔画）。
    所以对行投影取**相对**阈值即可分离 —— 这一点很重要：
    即使因为轻微残余倾斜让谱线在竖直方向散开、绝对覆盖率掉到 35%，
    只要它仍然远高于文字行，相对阈值就依然有效。

    Parameters
    ----------
    min_coverage
        相对于"覆盖率最高的那一行"的比例阈值。
    max_thickness
        墨迹带的行数上限，超过的当成文字块丢弃。
        注意不能设太小：倾斜残差会让谱线散成 5~10 行。
    """
    if binary.ndim != 2:
        raise ValueError("binary 必须是二维数组")
    row_ink = binary.sum(axis=1).astype(np.float64)
    peak = row_ink.max()
    if peak <= 0:
        return []
    thr = min_coverage * peak
    rows = np.where(row_ink >= thr)[0]
    if rows.size == 0:
        return []

    # 把连续的行聚成一个"线带"，用墨迹量做加权质心
    groups: list[list[int]] = [[int(rows[0])]]
    for r in rows[1:]:
        if r - groups[-1][-1] <= 2:      # 允许空隙，抗噪
            groups[-1].append(int(r))
        else:
            groups.append([int(r)])

    ys: list[float] = []
    for g in groups:
        if len(g) > max_thickness:
            continue                     # 太厚，是文字块而不是谱线
        w = row_ink[g]
        if w.sum() <= 0:
            continue
        ys.append(float((np.asarray(g) * w).sum() / w.sum()))
    return ys


def group_staves(
    line_ys: list[float],
    min_space: float = 4.0,
    spacing_tol: float = 0.30,
) -> list[StaffGeometry]:
    """把零散的谱线 y 聚成"每 5 条一个谱表"。

    判据：连续 5 条线的 4 个间隔必须彼此接近（相对偏差 < spacing_tol），
    且间隔不小于 min_space。不满足就往下错一位再试 —— 这样能容忍
    混进来的杂线（加线、小节线的残渣等）。
    """
    ys = sorted(line_ys)
    staves: list[StaffGeometry] = []
    i = 0
    while i <= len(ys) - 5:
        cand = ys[i:i + 5]
        gaps = np.diff(cand)
        mean_gap = float(gaps.mean())
        if mean_gap < min_space:
            i += 1
            continue
        if (gaps.min() >= mean_gap * (1 - spacing_tol)
                and gaps.max() <= mean_gap * (1 + spacing_tol)):
            staves.append(StaffGeometry(
                index=len(staves),
                line_ys=[float(v) for v in cand],
                space=mean_gap,
                x0=0.0, x1=0.0,
            ))
            i += 5
        else:
            i += 1
    return staves


def staff_h_extent(binary: np.ndarray, staff: StaffGeometry,
                   pad: int = 2) -> tuple[float, float]:
    """求谱表水平范围，避免把书页边缘的竖向暗带当成谱表左端。"""
    h, w = binary.shape
    y0 = max(0, int(round(staff.y_top_line)) - pad)
    y1 = min(h, int(round(staff.y_bottom_line)) + pad + 1)
    band = binary[y0:y1, :]
    cols = np.where(band.any(axis=0))[0]
    if cols.size == 0:
        return 0.0, float(w)
    left = float(cols[0])
    right = float(cols[-1])
    if len(staff.line_ys) != 5:
        return left, right

    # 真谱线在多个 y 高度上横贯页面；边缘暗带虽然也穿过 5 条线，
    # 却只有几十像素宽。沿 x 平滑后，只保留足够长的共同墨迹区间。
    line_pad = max(pad, int(round(0.35 * staff.space)))
    support = np.zeros(w, dtype=np.uint8)
    for y in staff.line_ys:
        center = int(round(y))
        lo, hi = max(0, center - line_pad), min(h, center + line_pad + 1)
        support += binary[lo:hi, :].any(axis=0).astype(np.uint8)
    window = max(31, int(round(3.5 * staff.space)))
    smooth = np.convolve((support >= 4).astype(np.float32),
                         np.ones(window, dtype=np.float32) / window, mode="same")
    mask = smooth >= 0.55
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    min_length = max(120, int(round(8 * staff.space)))
    long_runs = [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_length]
    if long_runs:
        left = float(max(0, min(a for a, _ in long_runs) - window // 2))
    return left, right


def unwrap_staff_line_offsets(samples: list[tuple[float, float]],
                              space: float) -> list[tuple[float, float]]:
    """Resolve one-line aliases using continuity and the page-wide anchor.

    The first sample can land on an adjacent staff line.  In that case a
    left-to-right unwrap follows the wrong branch across the entire system.
    The median raw offset supplies an independent page-wide anchor while the
    unwrapped differences retain real slope and curvature.
    """
    if not samples or space <= 0:
        return samples
    result = [samples[0]]
    for x, value in samples[1:]:
        previous = result[-1][1]
        choices = (value - space, value, value + space)
        result.append((x, min(choices, key=lambda item: abs(item - previous))))
    raw_median = float(np.median([value for _, value in samples]))
    unwrapped_median = float(np.median([value for _, value in result]))
    shift = round((raw_median - unwrapped_median) / space) * space
    return [(x, value + shift) for x, value in result]


def estimate_local_line_offsets(binary: np.ndarray, staff: StaffGeometry,
                                sample_space: float = 4.0,
                                window_space: float = 4.0) -> list[tuple[float, float]]:
    """Measure the common vertical drift of five staff lines across the page.

    The global line positions are averages over the page. In a photographed
    system they can be more than half a staff step away from the true line at
    either end, which turns a correctly found notehead into a wrong pitch.
    A horizontal window gives the staff line much more support than a notehead.
    Each sample needs at least three agreeing lines; a short median filter
    suppresses local interference from beams and text.
    """
    h, w = binary.shape
    if len(staff.line_ys) != 5 or staff.space <= 0:
        return []
    half_width = max(12, int(round(window_space * staff.space / 2)))
    stride = max(12, int(round(sample_space * staff.space)))
    # A photographed page can drift by nearly one whole staff space from the
    # page-wide average.  At 0.75 space the true line falls outside the search
    # range and the previous line (one space above) can win instead.
    search = max(3, int(round(1.20 * staff.space)))
    start = max(half_width, int(round(staff.x0)))
    stop = min(w - half_width, int(round(staff.x1)))
    if stop <= start:
        return []
    samples: list[tuple[float, float]] = []
    for x in range(start, stop + 1, stride):
        left, right = x - half_width, x + half_width + 1
        low = max(0, int(round(staff.y_top_line)) - search - 2)
        high = min(h, int(round(staff.y_bottom_line)) + search + 3)
        profile = binary[low:high, left:right].sum(axis=1)
        scores = []
        for offset in range(-search, search + 1):
            total = 0.0
            for nominal_y in staff.line_ys:
                center = int(round(nominal_y)) + offset - low
                if 0 <= center < len(profile):
                    total += float(profile[max(0, center - 1):center + 2].max(initial=0))
            scores.append(total)
        peak = max(scores)
        if peak >= 3.0 * 0.55 * (right - left):
            best = int(np.argmax(scores)) - search
            plateau = [offset - search for offset, score in enumerate(scores)
                       if abs(offset - search - best) <= 2 and score >= 0.96 * peak]
            samples.append((float(x), float(np.mean(plateau))))
    if len(samples) < 2:
        return []
    values = [point[1] for point in samples]
    smoothed = []
    for index, (x, value) in enumerate(samples):
        # Preserve the ends: a photograph may slope by a full staff step
        # across the page, so averaging neighbouring windows biases the ends.
        nearby = values[index - 1:index + 2] if 0 < index < len(values) - 1 else [value]
        smoothed.append((x, float(np.median(nearby))))
    # A long beam can impersonate one of the five lines for several adjacent
    # samples. Reject a short offset plateau that jumps by almost a full staff
    # space while the surrounding lines remain continuous.
    short_values = [value for _, value in smoothed]
    cleaned = []
    for index, (x, value) in enumerate(smoothed):
        broad = short_values[max(0, index - 4):min(len(smoothed), index + 5)]
        reference = float(np.median(broad))
        cleaned.append((x, reference if abs(value - reference) > 0.45 * staff.space else value))
    unwrapped = unwrap_staff_line_offsets(cleaned, staff.space)
    # The five detected line positions are measured near the centre of the
    # staff. A periodic comb can nevertheless follow the adjacent line for
    # most of the local samples. If the measured centre is almost exactly one
    # whole line away, anchor that branch back to the detected geometry.
    # Leave half-line discrepancies alone: they may reflect a real bad staff
    # candidate and cannot be resolved from this evidence alone.
    middle_x = (staff.x0 + staff.x1) / 2.0
    middle_offset = float(np.interp(middle_x,
                                    [x for x, _ in unwrapped],
                                    [offset for _, offset in unwrapped]))
    line_shift = round(middle_offset / staff.space)
    if (staff.centre_anchor_safe and line_shift
            and abs(middle_offset) >= .65 * staff.space
            and abs(middle_offset - line_shift * staff.space) <= .30 * staff.space):
        unwrapped = [(x, offset - line_shift * staff.space)
                     for x, offset in unwrapped]
    return unwrapped


def estimate_line_thickness(binary: np.ndarray,
                            staff: StaffGeometry,
                            run_lengths: np.ndarray | None = None) -> float:
    """估计谱线粗细：取谱线中心行垂直游程的低四分位数。

    `run_lengths` 可以传入预先算好的垂直游程图。**多谱表页面上必须传** ——
    这个量只跟二值图有关、与谱表无关，但整图算一次就是 O(像素数)；
    一张 10 行谱表的 A4 页 8.7M 像素，重复算 20 多次直接是秒级开销。
    """
    rl = run_lengths if run_lengths is not None else vertical_run_length(binary)
    h, _ = binary.shape
    vals: list[np.ndarray] = []
    for y in staff.line_ys:
        yi = int(round(y))
        if 0 <= yi < h:
            band = rl[yi, :]
            m = band > 0
            if m.any():
                vals.append(band[m])
    if not vals:
        return 2.0
    # 实拍谱中符干、字形和连音线会在谱线中心行形成很长的竖向游程；
    # 首系统可占过半墨迹，使中位数从真实线宽 2~3 px 涨到 14~18 px。
    # 低四分位数仍由横向谱线主导，可防止去线步骤把符头一并抹掉。
    return max(2.0, float(np.percentile(np.concatenate(vals), 25)))


def detect_staves(binary: np.ndarray, min_coverage: float = 0.5,
                  spacing_tol: float = 0.30,
                  run_lengths: np.ndarray | None = None) -> list[StaffGeometry]:
    """完整的谱表检测：找谱线 -> 分组成谱表 -> 补全几何信息。"""
    ys = detect_staff_line_ys(binary, min_coverage=min_coverage)
    staves = group_staves(ys, spacing_tol=spacing_tol)
    for s in staves:
        s.x0, s.x1 = staff_h_extent(binary, s)
        s.thickness = estimate_line_thickness(binary, s, run_lengths=run_lengths)
    return staves


def group_systems(staves: list[StaffGeometry],
                  glyph_sets: list | None = None,
                  binary: np.ndarray | None = None,
                  gap_max_space: float = 9.0,
                  span_min_frac: float = 0.5) -> list[list[int]]:
    """把谱表按**系统**（system）分组。返回每组在 `staves` 里的下标列表。

    这一步是多声部的前置：同一系统内的谱表是**同时**发声的（钢琴的大谱表），
    不同系统之间才是**先后**关系。搞错了会得到"左右手一前一后弹"的输出。

    ★ 主判据：**小节线是否延伸进两条谱表之间的间隙**
    ----------------------------------------------
    刻谱惯例：同一个系统内的谱表，小节线是**贯穿整个系统**的同一条竖线，
    所以它必然出现在两条谱表之间的**间隙**里；而不同系统之间是空白带的
    分界，上一条谱表的小节线**不会**伸进那条空白带。
    于是"取上一条谱表的小节线 x，看间隙那些列上是否有墨迹贯穿"就是
    直接证据，而且它**不依赖任何阈值标定**。

    ★ 为什么不用"左边缘墨迹"（第一版的做法，实测失败）
    ------------------------------------------------
    第一版看间隙左边缘在 x ∈ [x0, x0+k·space] 内的墨迹行占比，想靠**花括号**判断。
    101852 上它完美分离（相连对 0.92~1.00，不相连对 0.00），
    但 101910 上**每一对都只有 0.00~0.09** —— 原因有二：
      1. `staff.x0` 不可靠：101910 的 x0 是 3/14/16/102/109/117 这种乱值，
         因为**页面边缘的背景墨迹**被判成了墨迹，`staff_h_extent` 于是指到了图边，
         以它为基准的窗口根本没框住花括号。
      2. 窗口一旦放宽到 8~12 个 space 就会碰到**小节线**，
         而小节线是贯穿间隙的 -> `band.any(axis=1)` 整列变真 -> 占比恒为 1.000，
         判据彻底失去分辨力（实测宽窗口下 101910 全对都是 1.000）。
    换成"小节线是否伸进间隙"之后，这两个问题都不存在：小节线位置由
    `find_barlines` 的覆盖率+游程双重判据给出，比 `x0` 可靠得多。

    ★ 几何上界只作安全网
    --------------------
    实测系统内间隙 6.4~7.3 space、系统间 ≥ 9.7 space，所以 `gap_max_space=9.0`
    落在两者中间。它只用来避免把相隔很远的谱表拿去问小节线（那种情况必然为假，
    白花时间）。**决定权在小节线证据**，不在这个数。
    没有 `glyph_sets` 时退化为纯几何判断（只按 `gap_max_space`），
    这样只关心"行间距大不大"的调用方仍可用。
    """
    if not staves:
        return []
    if len(staves) < 2:
        return [[0]]

    groups: list[list[int]] = [[0]]
    for i in range(len(staves) - 1):
        upper, lower = staves[i], staves[i + 1]
        gap = lower.y_top_line - upper.y_bottom_line
        same = gap < gap_max_space * lower.space
        if same and glyph_sets is not None and binary is not None:
            bars = list(getattr(glyph_sets[i], "barlines", []) or [])
            same = _barlines_span_gap(binary, upper, lower, bars, span_min_frac)
        if same:
            groups[-1].append(i + 1)
        else:
            groups.append([i + 1])
    return groups


def _barlines_span_gap(binary: np.ndarray, upper: StaffGeometry,
                       lower: StaffGeometry, bars: list[float],
                       min_frac: float) -> bool:
    """上一条谱表的小节线里，有多大比例真的延伸进了两条谱表之间的间隙。"""
    h, w = binary.shape
    y0 = int(round(upper.y_bottom_line)) + 1
    y1 = int(round(lower.y_top_line))
    if y1 - y0 < 2:
        return True                    # 间隙太小，视为相连
    if not bars:
        return False
    span = y1 - y0
    good = 0
    for c in bars:
        xc = int(round(c))
        best = 0.0
        for x in range(max(0, xc - 2), min(w, xc + 3)):
            best = max(best, float(binary[y0:y1, x].sum()) / span)
            if best >= 0.6:
                break
        if best >= 0.6:
            good += 1
    return good / len(bars) >= min_frac


# ---------------------------------------------------------------------------
# 谱线去除
# ---------------------------------------------------------------------------


def remove_staff_lines(binary: np.ndarray, staff: StaffGeometry,
                       thickness_factor: float = 1.8,
                       band_factor: float = 1.6,
                       run_lengths: np.ndarray | None = None) -> np.ndarray:
    """只抹掉"细"的谱线像素，保留跨线的符头/符干。

    Parameters
    ----------
    thickness_factor
        游程长度阈值 = 谱线粗细 * 该系数。大于它的一律保留。
        太小 -> 抹不干净，残留断线会污染后续检测；
        太大 -> 把符干切断。
    band_factor
        只在谱线上下各 band_factor*thickness 的范围内动手，避免误伤别处。
    run_lengths
        可传入预先算好的垂直游程图（多谱表时**务必传**，见 estimate_line_thickness）。
    """
    if not staff.line_ys:
        return binary
    h, _ = binary.shape
    rl = run_lengths if run_lengths is not None else vertical_run_length(binary)
    thr = max(2.0, staff.thickness * thickness_factor)
    band_h = max(1.0, staff.thickness * band_factor)

    kill = np.zeros_like(binary)
    for y in staff.line_ys:
        y0 = max(0, int(round(y - band_h)))
        y1 = min(h, int(round(y + band_h)) + 1)
        if y1 <= y0:
            continue
        sub = np.zeros_like(binary)
        sub[y0:y1, :] = True
        kill |= sub & (rl <= thr)
    return binary & ~kill


def remove_all_staff_lines(binary: np.ndarray,
                           staves: list[StaffGeometry],
                           run_lengths: np.ndarray | None = None,
                           **kw) -> np.ndarray:
    """对所有谱表逐一去线。"""
    rl = run_lengths if run_lengths is not None else vertical_run_length(binary)
    out = binary
    for s in staves:
        out = remove_staff_lines(out, s, run_lengths=rl, **kw)
    return out


# ---------------------------------------------------------------------------
# 供调试/可视化
# ---------------------------------------------------------------------------


def find_ledger_lines(binary_without_staff: np.ndarray,
                      staff: StaffGeometry,
                      min_width: float = 0.6) -> list[tuple[float, int]]:
    """找加线（ledger line）。返回 [(y, step), ...]。

    加线是短横线，位于谱表上方或下方，是音高判定的重要线索
    （它决定了超出谱表范围的音高往哪个方向数）。
    """
    h, _ = binary_without_staff.shape
    results: list[tuple[float, int]] = []
    probe = binary_without_staff.copy()
    for direction, step0 in ((+1, 9), (-1, -1)):
        step = step0
        while True:
            y = staff.step_to_y(step)
            yi = int(round(y))
            if yi < 0 or yi >= h:
                break
            y0, y1 = max(0, yi - 2), min(h, yi + 3)
            band = probe[y0:y1, :]
            if not band.any():
                break
            row = band.any(axis=0)
            # 找最长的连续水平段
            best = cur = 0
            for v in row:
                cur = cur + 1 if v else 0
                best = max(best, cur)
            if best >= min_width * staff.space * 3:
                results.append((float(y), step))
            step += 2 * direction
            if abs(step) > 24:
                break
    return results


# ---------------------------------------------------------------------------
# 梳状（matched filter）谱表检测
# ---------------------------------------------------------------------------
#
# 为什么需要第二套检测器
# ----------------------
# `detect_staff_line_ys` + `group_staves` 的判据是"某一行墨迹 >= 0.5 * 全图峰值"。
# 合成图上整页亮度一致，这个判据很干脆；**真实照片上光照沿页面衰减，
# 同一页里谱线的行墨迹可以差 3 倍**（实测一张真实钢琴谱：中部谱线单行
# 2400~2500，页面上下两端的只有 800~1450）。于是固定相对阈值必然顾此失彼：
#   阈值高 -> 页面两端的谱表整条落选
#   阈值低 -> 文字行混进候选，把后面的贪心 5 线分组带偏
# 实测那张 5 组大谱表（= 10 条谱表）的真实照片，这套流水线只给出 5 条。
#
# 梳状滤波换掉整个思路：不再逐行判断"这行够不够黑"，而是拿
# **"5 条等距线"这个几何约束**去整体匹配：
#
#     score(y, s) = sum_{i=0..4} row_ink(round(y + i*s))
#                   - sum_{i=0..3} row_ink(round(y + i*s + s/2))
#
# 减掉"线间"那一项是为了压制"把瘦梳子塞进一块厚墨迹"的退化解：
# 真谱表线黑、线间相对空白，score 高；厚墨块里线上线下一样黑，score 趋 0。
# （注意密集钢琴谱的线间有符头/符干/符杠，本来就不空，所以这一项只是
#  辅助，真正决定性的是下面那条硬约束。）
#
# 硬约束：**5 条线里最弱的一条不得低于最强的一条的 `min_line_rel` 倍**。
# 这一条专治错位解 —— 半个梳子搭在谱表上、另外半个搭在空白处时，
# 总分可能只比正确解低 30%，但最弱线只有最强的 10%，一票否决。
# 实测一张 10 条谱表的真实照片里，前 16 个候选中错位解全被这条剔除。
#
# 为什么它对 PL 友好
# ------------------
#   * 只需要**行投影**（长度 H 的一维数组）：流式扫一遍二值图即可累加，
#     一块 BRAM 就装得下
#   * s 的上下限与量化步长都是编译期常数，定点实现不含除法
#   * 两条约束都是整数比较；搜索量 = H * S * 5，与图像宽度无关
#
# 代价：单个谱表的 s 不随 x 变化（不支持页内弯曲）。真实照片上是靠
# s 的搜索范围覆盖页面整体的透视缩放，这一点实测足够（同页 s 从 22.5 变到 24.0）。


@dataclass
class CombConfig:
    """梳状谱表检测参数。"""

    n_lines: int = 5                # 一个谱表几条线
    s_min: float = 15.0             # 谱间距搜索下限（px）
    s_max: float = 40.0             # 上限
    s_step: float = 0.25            # 量化步长
    min_line_rel: float = 0.55      # 最弱线 / 最强线 的下限
    mid_rel: float = 0.70           # 线间墨迹 / 线上墨迹 的上限
    nms_factor: float = 4.0         # 非极大抑制半径 = nms_factor * s
    min_peak_frac: float = 0.15     # 最强线墨迹 >= 全图行墨迹峰值 * 该比例
    refine: int = 3                 # 每条线用 ±refine 行做质心精修
    space_tol: float = 0.08         # 同页谱间距一致性：|s - 中位数| / 中位数 的上限
    thickness_max: float = 0.0      # 谱线粗细上限，0 = 不启用


def row_ink_profile(binary: np.ndarray) -> np.ndarray:
    """行墨迹投影。梳状检测唯一需要的中间量。"""
    return binary.sum(axis=1).astype(np.int64)


def prune_overlapping_staves(staves: list[StaffGeometry],
                             row: np.ndarray) -> list[StaffGeometry]:
    """Keep the better-supported five-line group when two candidates share lines.

    A false comb solution can start on the bottom line of a real staff and use
    note beams as its other four "lines".  The two five-line ranges then touch
    or overlap, which distinct piano staves cannot do.  Compare the weakest
    refined line's full-page ink support, rather than only the raw comb score.
    """
    kept: list[StaffGeometry] = []

    def weakest_line(staff: StaffGeometry) -> int:
        return min(int(row[min(max(0, round(y)), len(row) - 1)])
                   for y in staff.line_ys)

    for staff in sorted(staves, key=lambda item: item.y_top_line):
        while kept and staff.y_top_line <= kept[-1].y_bottom_line + .2 * min(
                staff.space, kept[-1].space):
            if weakest_line(staff) <= weakest_line(kept[-1]):
                break
            kept.pop()
        else:
            kept.append(staff)
    for index, staff in enumerate(kept):
        staff.index = index
    return kept


def prune_weak_staff_candidates(staves: list[StaffGeometry],
                                row: np.ndarray) -> list[StaffGeometry]:
    """Discard five-line patterns whose weakest line has little page coverage.

    Tempo text or note beams can make five roughly equally spaced row peaks.
    Compare each candidate's weakest line with its own horizontal span, then
    use the page median as a relative threshold. This accommodates uneven
    illumination across photographed pages.
    """
    if len(staves) < 3:
        return staves

    def coverage(staff: StaffGeometry) -> float:
        weakest = min(int(row[min(max(0, round(y)), len(row) - 1)])
                      for y in staff.line_ys)
        return weakest / max(1.0, staff.x1 - staff.x0)

    supports = [coverage(staff) for staff in staves]
    floor = 0.25 * float(np.median(supports))
    kept = [staff for staff, support in zip(staves, supports)
            if support >= floor]
    for index, staff in enumerate(kept):
        staff.index = index
    return kept


def _comb_arrays(row: np.ndarray, spaces: np.ndarray, n_lines: int):
    """返回 (line_vals, mid_sum)，line_vals 形状 (H, S, n_lines)。"""
    h = row.size
    ys = np.arange(h, dtype=np.float64)[:, None]
    vals = np.empty((h, spaces.size, n_lines), dtype=np.int64)
    for i in range(n_lines):
        idx = np.clip(np.round(ys + i * spaces[None, :]).astype(np.int64), 0, h - 1)
        vals[:, :, i] = row[idx]
    mid = np.zeros((h, spaces.size), dtype=np.int64)
    for i in range(n_lines - 1):
        idx = np.clip(np.round(ys + (i + 0.5) * spaces[None, :]).astype(np.int64), 0, h - 1)
        mid += row[idx]
    return vals, mid


def detect_staves_comb(binary: np.ndarray,
                       cfg: CombConfig | None = None,
                       run_lengths: np.ndarray | None = None) -> list[StaffGeometry]:
    """用"5 条等距线"的几何约束检测谱表（对光照衰减免疫）。

    与 `detect_staves` 的关系：保留后者不动（合成图上是黄金基准，
    且已被三层验证链覆盖），本函数是真实照片上更稳的第二条路径。
    """
    cfg = cfg or CombConfig()
    if binary.ndim != 2:
        raise ValueError("binary 必须是二维数组")
    h = binary.shape[0]
    row = row_ink_profile(binary)
    peak = int(row.max())
    if peak <= 0:
        return []

    spaces = np.arange(cfg.s_min, cfg.s_max + 1e-9, cfg.s_step)
    vals, mid = _comb_arrays(row, spaces, cfg.n_lines)
    line_sum = vals.sum(axis=2)
    vmin = vals.min(axis=2)
    vmax = vals.max(axis=2)

    # 四条约束：
    #   1) 线间要有对比（真谱表线黑、线间空白；塞进厚墨块的假解两边一样黑）
    #   2) 最弱线不能太弱（专治错位解：半个梳子在谱表上、半个在空白里）
    #   3) 最强线要有足够墨迹（滤掉纯噪声区域）
    #   4) 线间墨迹不能超过线上的 `mid_rel` 倍
    ok = (line_sum - mid > 0)
    ok &= (vmin * 100 >= int(cfg.min_line_rel * 100) * vmax)
    ok &= (vmax >= cfg.min_peak_frac * peak)
    ok &= (mid * 100 <= int(cfg.mid_rel * 100) * line_sum)
    if not ok.any():
        return []

    score = np.where(ok, line_sum - mid, 0)
    flat = score.ravel()
    order = np.argsort(-flat)
    ncols = spaces.size

    picked: list[tuple[float, float]] = []      # (y, s)
    for idx in order:
        val = int(flat[idx])
        if val <= 0:
            break
        y = float(idx // ncols)
        s = float(spaces[int(idx % ncols)])
        radius = cfg.nms_factor * s
        if any(abs(y - py) < radius for py, _ in picked):
            continue
        picked.append((y, s))

    # 精修：把每条线吸附到邻域内墨迹的质心，几何精度直接决定音高判定
    staves: list[StaffGeometry] = []
    for y, s in picked:
        line_ys: list[float] = []
        for i in range(cfg.n_lines):
            yc = y + i * s
            lo = max(0, int(round(yc)) - cfg.refine)
            hi = min(h, int(round(yc)) + cfg.refine + 1)
            w = row[lo:hi]
            if w.sum() > 0:
                line_ys.append(float((np.arange(lo, hi) * w).sum() / w.sum()))
            else:
                line_ys.append(yc)
        gaps = np.diff(line_ys)
        staves.append(StaffGeometry(
            index=0,
            line_ys=line_ys,
            space=float(gaps.mean()),
            x0=0.0, x1=0.0,
        ))

    staves.sort(key=lambda st: st.y_top_line)
    for i, st in enumerate(staves):
        st.index = i
        st.x0, st.x1 = staff_h_extent(binary, st)
        st.thickness = estimate_line_thickness(binary, st, run_lengths=run_lengths)

    # 同页一致性：**一页纸上所有谱表的谱间距必然相同**（差异只来自透视，
    # 实测一页里 22.5 ~ 24.0，约 ±3%）。用中位数当基准，把明显不合群的
    # 候选剔掉 —— 实测这一条就能清掉"塞进厚墨块"的假解（它们借着
    # 墨块里的任意等距结构凑分数，s 会明显偏离）。
    if cfg.space_tol > 0 and len(staves) >= 3:
        s_med = float(np.median([st.space for st in staves]))
        staves = [st for st in staves
                  if abs(st.space - s_med) <= cfg.space_tol * s_med]
    if cfg.thickness_max > 0:
        staves = [st for st in staves if st.thickness <= cfg.thickness_max]

    return prune_weak_staff_candidates(prune_overlapping_staves(staves, row), row)


def detect_staves_multistrip(binary: np.ndarray,
                             run_lengths: np.ndarray | None = None
                             ) -> list[StaffGeometry]:
    """Find slanted staves by voting across three narrower horizontal strips.

    A whole-page row projection can lose a line when perspective spreads its
    ink over many rows. Each strip has less vertical drift. A candidate must
    appear in at least two strips, which rejects local text and beam patterns.
    This is a recovery path for photographed piano pages, not a replacement
    for the cheaper whole-page detector when it already found every staff.
    """
    if binary.ndim != 2:
        raise ValueError("binary 必须是二维数组")
    width = binary.shape[1]
    spans = ((.12, .42), (.35, .65), (.58, .88))
    clusters: list[list[tuple[int, float, StaffGeometry]]] = []
    for strip_index, (left_frac, right_frac) in enumerate(spans):
        left, right = round(width * left_frac), round(width * right_frac)
        if right - left < 120:
            return []
        strip_runs = (run_lengths[:, left:right]
                      if run_lengths is not None else None)
        for staff in detect_staves_comb(binary[:, left:right],
                                        run_lengths=strip_runs):
            x_center = (left + right) / 2.0
            best = None
            best_distance = float("inf")
            for cluster in clusters:
                if any(index == strip_index for index, _, _ in cluster):
                    continue
                centre_y = float(np.median([s.y_top_line for _, _, s in cluster]))
                centre_space = float(np.median([s.space for _, _, s in cluster]))
                distance = abs(staff.y_top_line - centre_y)
                if (distance <= 1.6 * centre_space and
                        abs(staff.space - centre_space) <= .2 * centre_space and
                        distance < best_distance):
                    best, best_distance = cluster, distance
            if best is None:
                best = []
                clusters.append(best)
            best.append((strip_index, x_center, staff))

    recovered: list[StaffGeometry] = []
    runs = run_lengths if run_lengths is not None else vertical_run_length(binary)
    middle_left, middle_right = round(width * spans[1][0]), round(width * spans[1][1])
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        centre = next((staff for index, _, staff in cluster if index == 1), None)
        if centre is not None:
            line_ys = list(centre.line_ys)
        else:
            # The outer strips have equal displacement from the page centre.
            line_ys = [float(np.mean([staff.line_ys[line] for _, _, staff in cluster]))
                       for line in range(5)]
        space = float(np.diff(line_ys).mean())
        # A continuous double beam can impersonate the top or bottom staff
        # line. Thin horizontal ink distinguishes the actual line (2–3 px)
        # from a 10 px beam; require a strong sixth thin line outside the
        # detected comb before moving it by one space.
        half_band = max(2, round(.15 * space))
        thin_support = []
        for line in range(-1, 6):
            y = round(line_ys[0] + line * space)
            lo, hi = max(0, y - half_band), min(binary.shape[0], y + half_band + 1)
            band = binary[lo:hi, middle_left:middle_right]
            run_band = runs[lo:hi, middle_left:middle_right]
            thin_support.append(int((band & (run_band <= 4)).sum()))
        prev_line, *five_lines, next_line = thin_support
        down_reference = float(np.median(five_lines[1:]))
        up_reference = float(np.median(five_lines[:-1]))
        shift_down = (down_reference > 0 and five_lines[0] < .4 * down_reference
                      and next_line > .6 * down_reference
                      and next_line > .45 * (middle_right - middle_left))
        shift_up = (up_reference > 0 and five_lines[-1] < .4 * up_reference
                    and prev_line > .6 * up_reference
                    and prev_line > .45 * (middle_right - middle_left))
        shifted_by_thin_line = shift_down != shift_up
        if shift_down and not shift_up:
            line_ys = [y + space for y in line_ys]
        elif shift_up and not shift_down:
            line_ys = [y - space for y in line_ys]
        staff = StaffGeometry(index=len(recovered), line_ys=line_ys,
                              space=space, x0=0.0, x1=0.0)
        # Three independent strips must agree on the centre line before the
        # offset tracker may use it to resolve a one-line alias. A displaced
        # strip candidate can mean that the central comb itself skipped a
        # line, so anchoring to it would change true pitches by a whole line.
        staff.centre_anchor_safe = (shifted_by_thin_line or
                                    (len(cluster) == 3 and
                                     all(abs(candidate.y_top_line - line_ys[0])
                                         <= .85 * space
                                         for _, _, candidate in cluster)))
        staff.x0, staff.x1 = staff_h_extent(binary, staff)
        left_edge = round(width * spans[0][0])
        left_sample = next((candidate for index, _, candidate in cluster
                            if index == 0), None)
        if left_sample is not None and left_sample.x0 > 2:
            staff.x0 = min(staff.x0, left_edge + left_sample.x0)
        # A row measured over the entire slanted page can cross symbols
        # instead of a line, overstating the thickness of a true staff.
        staff.thickness = float(np.median(
            [candidate.thickness for _, _, candidate in cluster]))
        if staff.thickness <= .30 * staff.space:
            recovered.append(staff)
    recovered.sort(key=lambda staff: staff.y_top_line)
    for index, staff in enumerate(recovered):
        staff.index = index
    return recovered
