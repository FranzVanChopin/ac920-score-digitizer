"""
语义重建 — 技一堆符叀变成"一段音乀　
这是整个项目**工作量最大、也最不可能放刀FPGA 一*的部分　原因：它是图/序列推理，分支多、规则杂、形状不规则，天生不适合定长流水线　对应架构：本模块永远留在 PS（C++/Python）　
单声部的重建顺序
----------------
  1. 符头 + 符干 + 符尾  ->  时倀  2. 符头 y 位置 + 谱号 + 调号 + 临时记号  ->  音高
  3. 挀x 排序 + 小节线切刀 ->  小节归属
  4. 小节内顺序累劀 ->  onset
  5. 节拍约束校验  ->  发现/纠正时值错诀
其中笀5 步是最便宜、最有效的自动纠错手段：只要某小节总时值对不上拍号＀就一定有音符时值判错了。真实系统里这一步能救回相当一部分识别错误　"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import theory
from .detect import (SYM_ACC_FLAT, SYM_ACC_NATURAL, SYM_ACC_SHARP, SYM_DOT,
                     SYM_NOTEHEAD_FILLED, SYM_NOTEHEAD_HOLLOW,
                     SYM_REST_HALF, SYM_REST_QUARTER, SYM_REST_WHOLE,
                     SYM_TO_ACCIDENTAL, Detection, GlyphSet)
from .model import (PPQ, BBox, Event, Measure, MeasureIssue, Part, Score, Staff,
                    Voice, validate_measures)
from .render import NOTEHEAD_W, STEM_LEN
from .staff import StaffGeometry, vertical_run_length


# ---------------------------------------------------------------------------
# 符干与符尀# ---------------------------------------------------------------------------


@dataclass
class StemInfo:
    present: bool = False
    direction: str = "up"          # "up" | "down"
    flags: int = 0
    #: ★ 符干末端的**横梁数**（符杠）。真实密集段落里音符靠符杠连成组，
    #:   而 `flags` 只数得到**符尾**，数不到横梁 → 时值被放大 4~8 倍。
    #:   见 `count_beams`。
    beams: int = 0
    length: float = 0.0
    tip_y: float = 0.0


def count_beams(ink: np.ndarray, hx: float, hy: float, space: float,
                stem: "StemInfo", min_run_space: float = 1.8,
                tip_span_space: float = 2.4,
                min_thick_space: float = 0.0) -> int:
    """数符干末端有几根**横梁**（符杠）　"
    ☀为什么必须单独数梁，而不能只数符尀    ------------------------------------
    `analyze_stem` 数的昀*符尾**：在符干末端右侧开窗口、用**竖直跨度**换算条数
    ＀ �?�?0.95·space）。但真实乐谱里密集音符用的是**符杠** — 一杀*横梁**＀    它在该窗口里的竖直跨度只有梁厚（0.1~0.4 space），换算出来 `flags = 0`
    ↀ时值取 `PPQ = 480`（四分音符）＀*时值被放大 4~8 倀*�?
    实测（`tools/probe_stems.py`，三张真实照片）＀50 个有符干的音符里
    **120 个（18%）是"现判据给 flags=0、但符干末端确有横条"**　    接上本判据之后（与符尾取 max）：**多出釀120,750 ↀ97,530（−19%＀*�?    **未归囀57,120 ↀ42,720**、tick 正确 **62 ↀ69**　
    ⚠️ **合成集测不出杀*：`render.py` **根本不画符杠**（只有符尾）＀    所以默认合成页上本判据恒为 0、与旧判据完全一致（实测巀**0 tick**）　    要让它被门槛真正覆盖，得先让 `render.py` 能画符杠　
    判据（复用本项目已验证过皀**符杠 = 又长又厚的横杀*"）：
    在符干末端朝向符头那一侧的窗口里，逐行所穿过符干附近　    且长庀≀`min_run_space` 的横杀，把**相邻行合幀*成一根梁，数出梁数　    只认横条不认竖笔画是刻意的：符干只有 2~4 px 宽，不会达标　    """
    h, w = ink.shape
    head_w = NOTEHEAD_W * space
    sx = hx + head_w / 2 if stem.direction == "up" else hx - head_w / 2
    tip = int(round(stem.tip_y))
    span = int(round(tip_span_space * space))
    if stem.direction == "up":
        y0, y1 = tip, tip + span + 1
    else:
        y0, y1 = tip - span, tip + 1
    y0, y1 = max(0, y0), min(h, y1)
    xc = int(round(sx))
    lo = max(0, xc - int(round(0.5 * space)))
    hi = min(w, xc + int(round(0.5 * space)) + 1)
    if y1 <= y0 or hi <= lo:
        return 0
    min_run = max(3.0, min_run_space * space)
    hit = np.zeros(y1 - y0, dtype=bool)
    for i, y in enumerate(range(y0, y1)):
        row = ink[y]
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        st = np.flatnonzero(d == 1)
        en = np.flatnonzero(d == -1)
        for a, b in zip(st.tolist(), en.tolist()):
            if a <= hi and b >= lo and (b - a) >= min_run:
                hit[i] = True
                break
    if not hit.any():
        return 0
    d2 = np.diff(np.concatenate(([0], hit.view(np.int8), [0])))
    starts = np.flatnonzero(d2 == 1)
    ends = np.flatnonzero(d2 == -1)
    if min_thick_space > 0:
        # 可选：要求每根"�?自身有足够厚度（把谱线残渀0.17·space 之类挡在外面）
        need = min_thick_space * space
        starts = [s for s, e in zip(starts.tolist(), ends.tolist()) if (e - s) >= need]
    return len(starts)


def short_tail_rows(ink: np.ndarray, y0: int, y1: int,
                    x0: int, x1: int, space: float) -> np.ndarray:
    """Rows with a short curved flag stroke, excluding long lines and beams."""
    h, w = ink.shape
    if y1 <= y0 or x1 <= x0:
        return np.empty(0, dtype=np.int32)
    margin = int(round(2.0 * space))
    lo, hi = max(0, x0 - margin), min(w, x1 + margin)
    limit = 1.8 * space
    accepted = []
    for y in range(max(0, y0), min(h, y1)):
        row = ink[y, lo:hi].astype(np.int8)
        edges = np.diff(np.concatenate(([0], row, [0])))
        for start, end in zip(np.flatnonzero(edges == 1),
                              np.flatnonzero(edges == -1)):
            if (start + lo < x1 and end + lo > x0
                    and end - start <= limit):
                accepted.append(y - y0)
                break
    return np.asarray(accepted, dtype=np.int32)


def longest_tail_run(rows: np.ndarray, space: float) -> tuple[float, int]:
    """Longest dense flag stroke, excluding distant noteheads and staff noise.

    Chords can place another head within the tail search window.  Measuring
    from the first accepted row to the last then converts their separation
    into a fictitious second flag.  True double flags overlap vertically in
    the supported engraving styles, while unrelated blobs have a clear gap.
    """
    if rows.size == 0:
        return 0.0, 0
    gap = max(2, int(round(0.18 * space)))
    cuts = np.flatnonzero(np.diff(rows) > gap) + 1
    groups = np.split(rows, cuts)
    best = max(groups, key=lambda group: (group[-1] - group[0] + 1, group.size))
    return float(best[-1] - best[0] + 1), int(best.size)


def analyze_stem(ink: np.ndarray, hx: float, hy: float, space: float,
                 run_lengths: np.ndarray | None = None) -> StemInfo:
    """在符头两侧找符干，并由符干末端判断符尾数量　"
    符干一定贴在符头的左边缘或右边缘上，所以只在两个边缘附近几列里找　    符尾是从符干末端向外伸出的笔画，它的**竖直延伸长度**决定符尾条数＀        1 杀≀0.95 * space
        2 杀≀1.57 * space

    `run_lengths`
        可传入预先算好的垂直游程图　*一张页面上有几十个符头＀        每个都重算一遍整图游程是纯粹的浪贀* — 多谱表页面上这一顀        实测占了总耗时的一大半。调用方（reconstruct）每个谱表只算一次　    """
    h, w = ink.shape
    head_w = NOTEHEAD_W * space
    lw = max(2, int(round(space * 0.11)))
    rl = run_lengths if run_lengths is not None else vertical_run_length(ink)

    best: StemInfo | None = None
    for xc in (hx + head_w / 2 - lw * 0.5, hx - head_w / 2 + lw * 0.5):
        for dx in range(-lw, lw + 1):
            x = int(round(xc + dx))
            if x < 0 or x >= w:
                continue
            yi = int(round(hy))
            if yi < 0 or yi >= h or not ink[yi, x]:
                continue
            run = int(rl[yi, x])
            if run < 2.2 * space:
                continue
            # 找该列上包含 yi 的连续段上下边界
            a = yi
            while a - 1 >= 0 and ink[a - 1, x]:
                a -= 1
            b = yi
            while b + 1 < h and ink[b + 1, x]:
                b += 1
            above = hy - a
            below = b - hy
            direction = "up" if above > below else "down"
            tip = float(a) if direction == "up" else float(b)
            if best is None or run > best.length:
                # ⚠️ 这里**必须用关键字参数**。
                # 踩过的坑：原来是位置参数 `StemInfo(True, direction, 0, float(run), tip)`。
                # 后来往 `StemInfo` 中间插了一一`beams` 字段（为了记符杠梁数），
                # 位置参数就整体错位了 — `length` 拿到亀tip　*`tip_y` 变成亀0.0**）
                # 于是符尾搜索窗口全算错：合成门槛 100.0%/98.7% 掉到 97.3%/96.0%。
                # 真实照片多出釀120,750 涨到 223,080。
                # 而且**回退亀使用 beams"的代码也修不奀*，因为字段本身还在，
                # 错位就一直存圀— 这类"静默错位"极难从症状反推。
                # **凡是绀dataclass 加字段，都要先把所有构造点改成关键字参数　*
                best = StemInfo(present=True, direction=direction, flags=0,
                                length=float(run), tip_y=tip)
    if best is None:
        return StemInfo(present=False)

    # --- 数符尀---
    sx = hx + head_w / 2 if best.direction == "up" else hx - head_w / 2
    y_lo = int(best.tip_y) - 1
    y_hi = int(best.tip_y + (space * 2.1 if best.direction == "up"
                             else -space * 2.1))
    y0, y1 = (max(0, y_lo), min(h, y_hi + 1)) if best.direction == "up" \
        else (max(0, y_hi), min(h, y_lo + 1))
    # ★ 搜索区的左边界必须把**符干本身**让开。
    # 符尾是从符干末端向右侧伸出的笔画，符干只有 lw 宽，所以从 sx+2*lw 起搜
    # 就能完全避开符干。用 sx+lw 是不够的：实测符干右边缘正好压在第 1 列上，
    # 而搜索区高 2.1*space，于是"没符尾的四分音符"量到的竖直跨度 = 整个搜索区
    # 高度 -> 算出 flags=3 -> 时值变成三十二分音符。
    # 这个 bug 只在符干方向为 down 时暴露（up 方向符干在左、符尾也在右，
    # 搜索区天然不含符干），所以特别隐蔽。
    x0 = int(max(0, np.ceil(sx + max(2 * lw, 0.32 * space))))
    x1 = int(min(w, np.ceil(sx + space * 0.85)))
    if y1 > y0 and x1 > x0:
        rows = short_tail_rows(ink, y0, y1, x0, x1, space)
        if rows.size:
            extent, dense_rows = longest_tail_run(rows, space)
            # ★ 符尾条数的换算要扣掉"第一条符尾自身的高度"。
            # 渲染几何：第 i 条符尾起于 tip + i*0.62*space，自身高 0.95*space。
            # 所以 1 条的竖直跨度 ≈ 0.95*space，2 条 ≈ 1.57*space。
            # 一开始直接用 extent/(0.62*space) 四舍五入，1 条符尾算出 1.54 -> 2 条，
            # 八分音符被当成十六分音符（实测时值直接减半）。
            n_est = (extent / space - 0.95) / 0.62 + 1.0
            # Current renderer/validated photo set reaches sixteenths.  The
            # broad tail window often touches a neighboring head and reports
            # three flags; do not invent a 32nd without three separate tails.
            # Staff-line remnants and beam fragments may occur at two distant
            # rows in this window.  Their span resembles a flag, but a real
            # curved flag occupies a substantial number of distinct rows.
            if extent >= 0.55 * space and dense_rows >= 0.40 * space:
                best.flags = int(np.clip(round(n_est), 1, 2))
                best.length = max(best.length, extent)
    # ★ 再数一次**横梁**（符杠），与符尾取**较大者**。
    #   取 max 而不是替换，是刻意的保守选择：符尾与符杠都是"从符干末端伸出的尾"，
    #   真实乐谱里每个音符只会有其中一种，所以取大的那个就是"数到的尾数"；
    #   而 max 保证**时值只可能变短、不可能变长** ——
    #   于是这次改动在方向上只会减少"多出量"，不会引入新的膨胀。
    best.beams = count_beams(ink, hx, hy, space, best)
    # A dotted eighth followed by a sixteenth often uses one long beam and
    # one short hook.  The strict beam counter sees only the long beam, while
    # the flag rows have two separated strokes.  Retain that second stroke
    # as visual evidence for the later measure-level duration repair.
    if best.beams == 1 and y1 > y0 and x1 > x0:
        rows = short_tail_rows(ink, y0, y1, x0, x1, space)
        if rows.size:
            longest, _ = longest_tail_run(rows, space)
            if (rows[-1] - rows[0] + 1 >= longest + 0.4 * space
                    and count_beams(ink, hx, hy, space, best,
                                    min_run_space=0.9) >= 2):
                best.flags = max(best.flags, 2)
    return best


def bridge_beam_count(ink: np.ndarray, left: Detection, right: Detection,
                      left_stem: StemInfo, right_stem: StemInfo,
                      space: float, tip_reach_space: float = 0.65,
                      require_pair_spacing: bool = False,
                      far_reach_space: float = 1.25) -> int:
    """Count beams halfway between two neighboring stems.

    At a stem tip, a staff remnant or notehead can look like a second beam.
    The midpoint between stems has only the continuous beam, so it supplies
    independent evidence and can recover a stem lost during line removal.
    """
    # Dense sixteenths in engraved piano scores can be only 1.15 spaces
    # apart. The old 1.55-space minimum discarded their visible beams.
    if not (1.10 * space <= right.x - left.x <= 7.0 * space):
        return 0
    if not (left_stem.present or right_stem.present):
        return 0
    if left_stem.present and right_stem.present and left_stem.direction != right_stem.direction:
        return 0
    direction = left_stem.direction if left_stem.present else right_stem.direction
    offset = NOTEHEAD_W * space / 2 * (1 if direction == "up" else -1)
    tips = [s.tip_y for s in (left_stem, right_stem) if s.present]
    ty = float(np.mean(tips))
    # A wider reach is available to the caller only when a neighboring bridge
    # already proves that this is a two-beam group.  Unconditional widening
    # mistakes slurs and other nearby marks for second beams.
    y0 = max(0, int(round(ty - (tip_reach_space if direction == "up" else far_reach_space) * space)))
    y1 = min(ink.shape[0], int(round(ty + (far_reach_space if direction == "up" else tip_reach_space) * space)))
    if y1 <= y0:
        return 0
    counts = []
    for fraction in (0.15, 0.35, 0.50, 0.65, 0.85):
        sx = int(round(left.x + fraction * (right.x - left.x) + offset))
        if not (0 <= sx < ink.shape[1]):
            return 0
        col = ink[y0:y1, sx].astype(np.int8)
        edges = np.diff(np.concatenate(([0], col, [0])))
        runs = [(y0 + a, y0 + b) for a, b in zip(np.flatnonzero(edges == 1),
                                                np.flatnonzero(edges == -1))]
        good = [(a, b) for a, b in runs
                if 0.15 * space <= b - a <= 0.90 * space and
                abs((a + b) / 2 - ty) <= (1.75 if tip_reach_space > 0.65 else
                                            max(1.15, far_reach_space)) * space]
        if require_pair_spacing and len(good) >= 2:
            centres = [(a + b) / 2 for a, b in good]
            paired = any(0.45 * space <= v - u <= 0.90 * space
                         for u, v in zip(centres, centres[1:]))
            counts.append(2 if paired else 1)
        else:
            counts.append(min(3, len(good)))
    return min(counts)


def recover_missing_beamed_stems(ink: np.ndarray,
                                 head_stems: list[tuple[Detection, StemInfo]],
                                 space: float) -> None:
    """Bridge one or two absent stems inside a continuous double-beam group.

    The two surviving endpoint stems fix direction and approximate beam height.
    Every adjacent gap must independently contain two parallel beams, which
    keeps isolated slurs and unstemmed half notes from becoming sixteenths.
    """
    for i, (left, left_stem) in enumerate(head_stems):
        if not left_stem.present:
            continue
        for gap_count in (1, 2):
            j = i + gap_count + 1
            if j >= len(head_stems):
                continue
            right, right_stem = head_stems[j]
            middle = head_stems[i + 1:j]
            if (not right_stem.present
                    or right_stem.direction != left_stem.direction
                    or any(stem.present or head.symbol != SYM_NOTEHEAD_FILLED
                           for head, stem in middle)):
                continue
            run = head_stems[i:j + 1]
            if any(not 1.55 * space <= b.x - a.x <= 2.50 * space
                   for (a, _), (b, _) in zip(run, run[1:])):
                continue
            tips = []
            for head, stem in run:
                if stem.present:
                    tips.append(stem)
                else:
                    fraction = (head.x - left.x) / (right.x - left.x)
                    tip = left_stem.tip_y + fraction * (right_stem.tip_y - left_stem.tip_y)
                    tips.append(StemInfo(present=True, direction=left_stem.direction,
                                         tip_y=tip))
            counts = [bridge_beam_count(ink, a, b, sa, sb, space)
                      for (a, _), (b, _), sa, sb in
                      zip(run, run[1:], tips, tips[1:])]
            if counts and min(counts) >= 2:
                for (_, stem), trial in zip(middle, tips[1:-1]):
                    stem.present = True
                    stem.direction = trial.direction
                    stem.tip_y = trial.tip_y
                break


def regularize_beamed_stems(ink: np.ndarray,
                           head_stems: list[tuple[Detection, StemInfo]],
                           space: float) -> None:
    """Use a measured bridge to correct local beam overcounts or missing stems."""
    recover_missing_beamed_stems(ink, head_stems, space)
    edges = [bridge_beam_count(ink, left, right, ls, rs, space)
             for (left, ls), (right, rs) in zip(head_stems, head_stems[1:])]

    def has_neighboring_bridge(counts: list[int], i: int, count: int) -> bool:
        # Consecutive heads of one chord have essentially the same x.  Their
        # intervening edge has count zero, but it must not split a continuous
        # beam group into unrelated two-note fragments.
        for direction in (-1, 1):
            j = i + direction
            while (0 <= j < len(counts) and counts[j] == 0
                   and abs(head_stems[j + 1][0].x - head_stems[j][0].x)
                   <= 0.55 * space):
                j += direction
            if 0 <= j < len(counts) and counts[j] == count:
                return True
        return False

    baseline_edges = edges.copy()
    for i, count in enumerate(baseline_edges):
        if count != 1 or not has_neighboring_bridge(baseline_edges, i, 2):
            continue
        (left, ls), (right, rs) = head_stems[i:i + 2]
        if bridge_beam_count(ink, left, right, ls, rs, space,
                             tip_reach_space=1.6,
                             require_pair_spacing=True) >= 2:
            edges[i] = 2
    supported: set[int] = set()
    for i, count in enumerate(edges):
        if count <= 0:
            continue
        left, ls = head_stems[i]
        right, rs = head_stems[i + 1]
        linked = has_neighboring_bridge(edges, i, count)
        if not linked and right.x - left.x < 1.90 * space:
            continue
        supported.update((i, i + 1))
        beam_direction = ls.direction if ls.present else rs.direction
        for head, stem in ((left, ls), (right, rs)):
            if not stem.present:
                stem.present = True
                stem.direction = beam_direction
                stem.tip_y = ls.tip_y if ls.tip_y else rs.tip_y
            # A sixteenth mixed with eighths can have a short secondary hook
            # that never reaches the bridge midpoint.  Count short, thick
            # local strokes only after a continuous primary bridge is proven.
            local = count_beams(ink, head.x, head.y, space, stem,
                                min_run_space=0.75, tip_span_space=1.05,
                                min_thick_space=0.15)
            stem.beams = max(count, min(2, local))
            # A third beam sits about 1.3 spaces below the stem tip in this
            # engraving.  The short local window above sees only two beams.
            # Require both a connected double-beam bridge and three long,
            # thick horizontal strokes at the stem before assigning a 32nd.
            if (count >= 2 and
                    count_beams(ink, head.x, head.y, space, stem,
                                min_run_space=1.8, tip_span_space=2.4,
                                min_thick_space=0.15) >= 3 and
                    bridge_beam_count(ink, left, right, ls, rs, space,
                                      far_reach_space=1.8) >= 3):
                stem.beams = max(stem.beams, 3)
    # A beam must connect two stems.  A long horizontal stroke at one stem tip
    # alone can be a staff remnant or slur, so it is insufficient evidence.
    for i, (_, stem) in enumerate(head_stems):
        if i not in supported:
            stem.beams = 0


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class ReconstructConfig:
    """重建参数。"""

    clef: str = "treble"
    key_fifths: int = 0                     # 调号（0 = C 大调/无升降号）
    key_changes: tuple[tuple[float, int], ...] = ()  # (小节左节线 x, 新调号)
    clef_changes: tuple[tuple[float, str], ...] = ()  # (图像 x, 新谱号)
    default_time_signature: tuple[int, int] = (4, 4)
    #: ★ 是否保留**没有内容的空小节**。
    #:   默认 False（丢弃）—— 单谱表场景下这是对的，理由见函数体内那段长注释：
    #:   多出来的一条假节线会凭空造出一个空小节，把后面所有音符的 key 整体平移。
    #:   但**多谱表系统内必须为 True**：同一系统内各谱表的小节边界是同一条节线，
    #:   小节数必然相等。若某小节只在高音谱表有内容、低音谱表为空，
    #:   逐谱表独立丢弃就会让两条谱表从此**整体错位**（左右手对不上）。
    #:   实测真实钢琴谱上系统内小节数经常不等，直接原因就是这里逐谱表独立决策。
    keep_empty: bool = False
    #: 成对谱表共享起止小节边界；None 保留单谱表自动判断。
    include_leading_measure: bool | None = None
    include_trailing_measure: bool | None = None
    #: ★ 和弦分组：x 相差不超过 `chord_dx` 个 space 的符头视为**同一 onset**，
    #:   合成一个和弦（共享 onset/duration，带同一个 chord_id）。
    #:   取 0.55 是很保守的：同符干的符头和弦 x 完全相同，含二度和弦会错开约
    #:   一个符头宽（≈1 space），而密集十六分音符之间的 x 间距通常 ≥1.3 space，
    #:   所以这个阈值只合并**确定同 onset** 的符头，不会把顺序音符并成一串。
    chord_dx: float = 0.55
    #: 空心符头且无符干 -> 全音符；空心 + 符干 -> 二分音符
    hollow_no_stem_is_whole: bool = True
    #: 附点搜索窗口（相对 space）
    # The validated font and photo use centres roughly 1.2–1.35 spaces apart.
    # A wider window attached a following natural sign to the preceding note.
    dot_dx: float = 1.4
    dot_dy: float = 0.68
    #: 临时记号搜索窗口（相寀space），只看符头左侧
    acc_dx: float = 1.75
    acc_dy: float = 1.0
    min_confidence: float = 0.0
    #: ★ 是否用拍号约束自动纠正时值误判。强烈建议开启（见 repair_durations）。
    repair: bool = True
    #: 纠错时允许的最大时值改动倍率（3.0 = 允许 ×3 或 ÷3）
    repair_max_ratio: float = 3.0
    #: ★ 一个小时内最多允许改动几个音符。太多就说明不是"几处误判"，
    #: 而是整体识别失败（比如漏检了音符），此时纠错只会扭曲本来正确的音符。
    repair_max_changes: int = 2
    #: ★ 纠错方案的总代价上限。代价定义见 `_change_cost`。
    #: 0.8 = 一处符尾修正；1.0 = 一处附点修正；1.6 = 两处符尾修正。
    repair_max_cost: float = 1.6


# ---------------------------------------------------------------------------
# 时值判定
# ---------------------------------------------------------------------------

#: 符干符尾组合 -> 基础时值（tick）
_STEM_DURATION = {
    (True, 0): PPQ,               # 四分
    (True, 1): PPQ // 2,          # 八分
    (True, 2): PPQ // 4,          # 十六分
    (True, 3): PPQ // 8,          # 三十二分
}


def note_duration_ticks(hollow: bool, stem: StemInfo, dotted: bool,
                        cfg: ReconstructConfig) -> int:
    """由字形特征判定时值。"""
    if hollow and not stem.present:
        base = 4 * PPQ                      # 全音符
    elif hollow and stem.present:
        base = 2 * PPQ                      # 二分音符
    elif not hollow and stem.present:
        # 真正检测到横梁时，以横梁数为准。符尾窗口也会扫到邻音、
        # 横梁和残余谱线；清晰合成页上大量单横梁八分音符被它误报为
        # 3 条符尾，原来的 max(flags, beams) 因而缩成三十二分音符。
        tail_count = stem.beams if stem.beams > 0 else stem.flags
        base = _STEM_DURATION.get(
            (True, min(tail_count, 3)), PPQ)
    else:
        # 实心且无符干：不合法的字形，按最短的合理值处理并降置信度
        base = PPQ
    return int(base * 1.5) if dotted else int(base)


# ---------------------------------------------------------------------------
# 主流稀# ---------------------------------------------------------------------------


def _associate_accidental(head: Detection, accs: list[Detection],
                          space: float, cfg: ReconstructConfig,
                          head_stems: list[tuple[Detection, StemInfo]] | None = None,
                          ink: np.ndarray | None = None) -> Detection | None:
    """给符头找它左边紧邻的临时记号　"""
    best, best_d = None, 1e9
    for a in accs:
        dx = head.x - a.x
        # A glyph centre inside the notehead/stem envelope cannot be a
        # preceding accidental.  This rejects false flat templates formed by
        # the note's own downward stem without excluding normally spaced glyphs.
        if dx < 0.65 * space or dx > cfg.acc_dx * space:
            continue
        if abs(a.y - head.y) > cfg.acc_dy * space:
            continue
        # The two crossbars of a sharp straddle its target staff step.  A
        # response one step above/below commonly belongs to another head in
        # the same chord; assigning it twice corrupts the rest of the measure.
        if a.symbol == SYM_ACC_SHARP and abs(a.y - head.y) > 0.22 * space:
            continue
        # A strong sharp beside the upper head of a chord can also match the
        # natural template about one staff step away.  Both responses have
        # the same x because they come from one printed glyph; accepting the
        # lower response changes every later note on that step in the bar.
        if (a.symbol == SYM_ACC_NATURAL
                and any(
                    sharp.symbol == SYM_ACC_SHARP
                    and abs(sharp.x - a.x) <= 0.15 * space
                    and 0.55 * space <= abs(a.y - sharp.y) <= 1.10 * space
                    and sharp.score >= a.score + 0.10
                    and any(
                        other is not head
                        and abs(other.x - head.x) <= 0.30 * space
                        and abs(other.y - sharp.y) <= 0.22 * space
                        for other, _ in head_stems or [])
                    for sharp in accs)):
            continue
        # A flat's bowl aligns with the target but the template centre sits
        # above it.  Resolve ambiguous geometry first: a natural response can
        # actually be a flat, and must not claim the higher chord tone.  Do
        # this before selecting the nearest candidate so the true flat remains
        # available as an alternative.
        if (ink is not None
                and resolve_accidental_symbol(ink, a, space) == SYM_ACC_FLAT
                and a.y - head.y > -0.15 * space):
            continue
        # Dense chords can make a real notehead and its stem look like a sharp
        # (two dark lobes with a vertical stroke).  The glyph centre then sits
        # on a different, already detected stemmed notehead.  An accidental
        # occupies the free space before its target, not another notehead.
        on_other_head = any(
            other is not head and stem.present
            and abs(a.x - other.x) <= 0.45 * space
            and abs(a.y - other.y) <= 1.2 * space
            for other, stem in head_stems or []
        )
        if on_other_head:
            continue
        # A neighboring note's stem/beam can match the sharp/natural template.
        # It is already explained by that note and cannot also be an accidental.
        on_other_stem = False
        for other, stem in head_stems or []:
            if other is head or not stem.present:
                continue
            stem_x = other.x + NOTEHEAD_W * space / 2 if stem.direction == "up" \
                else other.x - NOTEHEAD_W * space / 2
            lo, hi = sorted((other.y, stem.tip_y))
            if (abs(a.x - stem_x) <= 0.45 * space and
                    lo - 0.25 * space <= a.y <= hi + 0.25 * space):
                on_other_stem = True
                break
        if on_other_stem:
            continue
        # The accidental should be centered on the note's staff step.  Several
        # template responses can share the same x while landing on adjacent
        # steps; x distance alone selects the wrong glyph in that case.
        d = abs(dx - space * 0.85) + 0.5 * abs(a.y - head.y)
        if d < best_d:
            best, best_d = a, d
    return best


def refine_flat_natural(ink: np.ndarray, accidental: Detection,
                        space: float) -> str:
    """Distinguish a flat's one long stem from a natural's two stems.

    Staff-line removal can erase the thin crossbars of a natural and leave a
    convincing flat template response.  The two long vertical strokes remain
    separated by about half a staff space in both symbols and photographs.
    """
    if accidental.symbol not in (SYM_ACC_FLAT, SYM_ACC_NATURAL):
        return accidental.symbol
    xc = float(accidental.x)
    yc = int(round(accidental.y))
    x0 = max(0, int(round(xc - 0.55 * space)))
    x1 = min(ink.shape[1], int(round(xc + 0.55 * space)) + 1)
    y0 = max(0, int(round(yc - space)))
    y1 = min(ink.shape[0], int(round(yc + space)) + 1)
    if x1 <= x0 or y1 <= y0:
        return accidental.symbol
    columns = np.count_nonzero(ink[y0:y1, x0:x1], axis=0)
    left = columns[np.arange(x0, x1) < xc - 0.08 * space]
    right = columns[np.arange(x0, x1) > xc + 0.08 * space]
    if left.size == 0 or right.size == 0:
        return accidental.symbol
    left_peak, right_peak = int(left.max()), int(right.max())
    if (left_peak >= 0.9 * space and right_peak >= 0.9 * space
            and right_peak >= 0.62 * left_peak):
        return SYM_ACC_NATURAL
    if left_peak >= 0.9 * space and right_peak < 0.62 * left_peak:
        return SYM_ACC_FLAT
    return accidental.symbol


def resolve_accidental_symbol(ink: np.ndarray, accidental: Detection,
                              space: float) -> str:
    refined = refine_flat_natural(ink, accidental, space)
    if (refined == SYM_ACC_NATURAL
            and accidental.extra.get("ambiguous_sharp_score", 0.0) >= 0.65):
        return SYM_ACC_SHARP
    return refined


def drop_stem_ghosts(head_stems: list[tuple[Detection, StemInfo]],
                     space: float) -> list[tuple[Detection, StemInfo]]:
    """Remove filled-head responses centred on a different note's stem.

    A true chord head sits beside its shared stem, while a false template
    response on a long stem is centred directly on it.  Require a stemless
    response well away from the real head so short adjacent chord steps stay.
    """
    kept: list[tuple[Detection, StemInfo]] = []
    for head, stem in head_stems:
        if head.symbol != SYM_NOTEHEAD_FILLED or stem.present:
            kept.append((head, stem))
            continue
        ghost = False
        for other, other_stem in head_stems:
            if other is head or not other_stem.present:
                continue
            if abs(head.y - other.y) < 1.5 * space:
                continue
            stem_x = other.x + (NOTEHEAD_W * space / 2 if other_stem.direction == "up"
                                else -NOTEHEAD_W * space / 2)
            lo, hi = sorted((other.y, other_stem.tip_y))
            if (abs(head.x - stem_x) <= 0.38 * space and
                    lo + 0.3 * space <= head.y <= hi - 0.3 * space):
                ghost = True
                break
        if not ghost:
            kept.append((head, stem))
    return kept


def recover_natural_from_hollow_response(
        ink: np.ndarray,
        head_stems: list[tuple[Detection, StemInfo]],
        space: float,
        accidentals: list[Detection] = ()) -> tuple[list[tuple[Detection, StemInfo]], dict[int, float]]:
    """Treat a split-hole response immediately before a head as a natural.

    A photographed natural has two long uprights and can enclose a small white
    patch.  The hollow-head fallback then reports it as a whole note.  Require
    both uprights and a filled head immediately to its right before making
    this reinterpretation.  A stemmed false response or a small vertical
    offset also needs an independently detected natural sign at that position.
    """
    kept: list[tuple[Detection, StemInfo]] = []
    recovered: dict[int, float] = {}
    for head, stem in head_stems:
        if (head.symbol != SYM_NOTEHEAD_HOLLOW
                or head.extra.get("measure") != "split_enclosed_hole"):
            kept.append((head, stem))
            continue
        matching = []
        for other, other_stem in head_stems:
            if (other.symbol != SYM_NOTEHEAD_FILLED or not other_stem.present
                    or not 0.70 * space <= other.x - head.x <= 1.15 * space
                    or abs(other.y - head.y) > 0.45 * space):
                continue
            glyph_support = any(
                abs(acc.x - head.x) <= 0.50 * space
                and abs(acc.y - other.y) <= 0.35 * space
                and 0.65 * space <= other.x - acc.x <= 1.75 * space
                and resolve_accidental_symbol(ink, acc, space) == SYM_ACC_NATURAL
                for acc in accidentals)
            if abs(other.y - head.y) > (0.45 if glyph_support else 0.25) * space:
                continue
            if stem.present and not glyph_support:
                continue
            matching.append((other, other_stem))
        if not matching or refine_flat_natural(
                ink, Detection(SYM_ACC_FLAT, head.x, head.y, head.score),
                space) != SYM_ACC_NATURAL:
            kept.append((head, stem))
            continue
        target = min(matching, key=lambda item: abs(item[0].x - head.x))[0]
        recovered[id(target)] = max(recovered.get(id(target), 0.0), head.score)
    return kept, recovered


def drop_beam_ghosts(ink: np.ndarray,
                     head_stems: list[tuple[Detection, StemInfo]],
                     space: float) -> list[tuple[Detection, StemInfo]]:
    """Discard apparent heads lying on a measured beam between real stems.

    A sloped beam can be too short in any one image row for the detector's
    horizontal-run filter.  Several high-scoring false heads then appear
    along it.  Require two separate noteheads, aligned stem tips and a bridge
    measured at five intermediate columns before removing a response.
    """
    bridges: list[tuple[int, int, float, float]] = []
    for i, (left, ls) in enumerate(head_stems):
        if not ls.present:
            continue
        for j in range(i + 1, len(head_stems)):
            right, rs = head_stems[j]
            if right.x - left.x > 7.0 * space:
                break
            if not rs.present or ls.direction != rs.direction:
                continue
            if bridge_beam_count(ink, left, right, ls, rs, space) > 0:
                bridges.append((i, j, ls.tip_y, rs.tip_y))
    kept: list[tuple[Detection, StemInfo]] = []
    for k, item in enumerate(head_stems):
        head, stem = item
        ghost = False
        for i, j, y0, y1 in bridges:
            if k in (i, j):
                continue
            left, right = head_stems[i][0], head_stems[j][0]
            margin = 0.35 if stem.present else 0.2
            if not (left.x - margin * space <= head.x <= right.x + margin * space):
                continue
            if min(abs(head.y - left.y), abs(head.y - right.y)) < 1.5 * space:
                continue
            t = (head.x - left.x) / (right.x - left.x)
            beam_y = y0 + t * (y1 - y0)
            if abs(head.y - beam_y) <= 0.85 * space:
                ghost = True
                break
            # A lower beam can itself pass the filled-head template.  Its
            # response may appear to have a stem: it is actually the same
            # vertical stroke as the nearby real note, read in reverse.
            # Require the independently measured bridge and both endpoints
            # of that reversed stroke before suppressing a possible chord.
            if stem.present:
                for endpoint_index in (i, j):
                    real, real_stem = head_stems[endpoint_index]
                    if (stem.direction != real_stem.direction
                            and abs(head.x - real.x) <= 0.35 * space
                            and abs(head.y - real_stem.tip_y) <= 1.10 * space
                            and abs(stem.tip_y - real.y) <= 0.40 * space
                            and abs(head.y - real.y) >= 1.5 * space):
                        ghost = True
                        break
                if ghost:
                    break
        if not ghost:
            kept.append(item)
    return kept


def rest_inside_continuous_beam(rest: Detection,
                                head_stems: list[tuple[Detection, StemInfo]],
                                ink: np.ndarray, space: float) -> bool:
    """Reject a small rest response on the bridge between two beamed notes.

    The printed triplet numeral can match the half-rest template.  Its x and
    y then lie inside the continuous beam while the actual noteheads and
    their two stems remain independently visible.
    """
    if rest.symbol not in (SYM_REST_HALF, SYM_REST_WHOLE):
        return False
    for (left, ls), (right, rs) in zip(head_stems, head_stems[1:]):
        if not (ls.present and rs.present and ls.direction == rs.direction):
            continue
        if not (left.x + 0.25 * space < rest.x < right.x - 0.25 * space):
            continue
        if abs(rest.y - (ls.tip_y + rs.tip_y) / 2) > 1.2 * space:
            continue
        if bridge_beam_count(ink, left, right, ls, rs, space) > 0:
            return True
    return False


def extreme_head_has_ledger_support(head: Detection, staff: StaffGeometry,
                                    ink: np.ndarray) -> bool:
    """Require ledger lines for a head over four spaces beyond the staff.

    Such a pitch needs several visible ledger lines in conventional notation.
    A staff crossing or a beam in the grand-staff gap can otherwise produce a
    convincing isolated notehead far outside the intended staff.
    """
    space = staff.space
    top, bottom = staff.y_top_line, staff.y_bottom_line
    edge = top if head.y < top else bottom
    sign = -1 if head.y < top else 1
    outside = abs(head.y - edge) / space
    if outside <= 4.0:
        return True
    count = 0
    x0 = max(0, int(round(head.x - 1.2 * space)))
    x1 = min(ink.shape[1], int(round(head.x + 1.2 * space)) + 1)
    centre = head.x - x0
    for step in range(1, int(outside) + 1):
        ledger_y = int(round(edge + sign * step * space))
        for y in range(max(0, ledger_y - 2), min(ink.shape[0], ledger_y + 3)):
            row = ink[y, x0:x1].astype(np.int8)
            edges = np.diff(np.concatenate(([0], row, [0])))
            if any(a <= centre + 0.4 * space and b >= centre - 0.4 * space and
                   b - a >= 1.3 * space
                   for a, b in zip(np.flatnonzero(edges == 1),
                                   np.flatnonzero(edges == -1))):
                count += 1
                break
        if count >= 2:
            return True
    return False


def _find_dot(head: Detection, dots: list[Detection],
              space: float, cfg: ReconstructConfig) -> Detection | None:
    allowed_dy = max(cfg.dot_dy, 0.7) if head.symbol == SYM_NOTEHEAD_HOLLOW else cfg.dot_dy
    for d in dots:
        dx = d.x - head.x
        # The right rim of a hollow head can look like an isolated dot after
        # line removal.  A genuine augmentation dot lies beyond that rim.
        min_dx = 0.65 if head.symbol == SYM_NOTEHEAD_HOLLOW else 0.0
        if (0 < dx and min_dx * space <= dx <= cfg.dot_dx * space
                and abs(d.y - head.y) <= allowed_dy * space):
            return d
    return None


def _dot_is_flat_lower_tip(dot: Detection, accs: list[Detection],
                           space: float) -> bool:
    """Reject a dot response immediately below a strong flat's bounding box."""
    for acc in accs:
        if acc.symbol != SYM_ACC_FLAT or acc.score < 0.72:
            continue
        x, y, w, h = acc.bbox
        bottom = y + h
        if (x <= dot.x <= x + w and
                bottom - 0.10 * space <= dot.y <= bottom + 0.25 * space):
            return True
    return False


def reconstruct(glyphs: GlyphSet, cfg: ReconstructConfig | None = None,
                title: str = "") -> tuple[Score, list]:
    """把符号检测结果重建为乐谱　"
    Returns
    -------
    (Score, issues)
        `issues` 是节拍校验发现的问题列表（model.validate_measures 的输出）　    """
    cfg = cfg or ReconstructConfig()
    staff_geo = glyphs.staff
    if staff_geo is None:
        raise ValueError("GlyphSet 缺少 staff 几何信息")
    ink = glyphs.ink
    if ink is None:
        raise ValueError("GlyphSet 缺少去线后的墨迹图（语义重建做符幀符尾关联时需要）")
    space = staff_geo.space

    clef = glyphs.clef or cfg.clef
    clef_changes = sorted(cfg.clef_changes)

    def clef_at(x: float) -> str:
        current = clef
        for change_x, new_clef in clef_changes:
            if x <= change_x:
                break
            current = new_clef
        return current
    ts = glyphs.time_signature or cfg.default_time_signature

    heads = sorted((hd for hd in glyphs.noteheads()
                    if extreme_head_has_ledger_support(hd, staff_geo, ink)),
                   key=lambda d: d.x)
    rests = sorted(glyphs.by_symbol(SYM_REST_WHOLE) + glyphs.by_symbol(SYM_REST_HALF)
                   + glyphs.by_symbol(SYM_REST_QUARTER), key=lambda d: d.x)
    accs = glyphs.by_symbol(SYM_ACC_SHARP) + glyphs.by_symbol(SYM_ACC_FLAT) \
        + glyphs.by_symbol(SYM_ACC_NATURAL)
    dots = glyphs.by_symbol(SYM_DOT)
    dots = [dot for dot in dots if not _dot_is_flat_lower_tip(dot, accs, space)]

    # --- 1. 把符头变戀待定事件" -----------------------------------------
    # 垂直游程图整页只算这一次，供所有符头的符干/符尾分析复用
    rl = vertical_run_length(ink)

    head_stems = [(hd, analyze_stem(ink, hd.x, hd.y, space,
                                   run_lengths=rl)) for hd in heads]
    head_stems = drop_stem_ghosts(head_stems, space)
    head_stems, recovered_naturals = recover_natural_from_hollow_response(
        ink, head_stems, space, accs)
    head_stems = drop_beam_ghosts(ink, head_stems, space)
    regularize_beamed_stems(ink, head_stems, space)
    rests = [rest for rest in rests
             if not rest_inside_continuous_beam(rest, head_stems, ink, space)]
    raw: list[tuple[float, dict]] = []
    for hd, stem in head_stems:
        hollow = hd.symbol == SYM_NOTEHEAD_HOLLOW
        dot = _find_dot(hd, dots, space, cfg)
        # The compact-component fallback is intentionally weaker evidence
        # than the dot template.  Beam/stem fragments in dense eighth-note
        # groups can create identical tiny components, so require a plain
        # unflagged note for this recovery path.
        if (dot is not None and dot.extra.get("component_fallback")
                and max(stem.flags, stem.beams) > 0):
            dot = None
        dur = note_duration_ticks(hollow, stem, dot is not None, cfg)

        step_f = staff_geo.y_to_step(hd.y, hd.x)
        step = int(round(step_f))
        letter, octave = theory.step_to_pitch(step, clef_at(hd.x))

        acc = (None if id(hd) in recovered_naturals else
               _associate_accidental(hd, accs, space, cfg, head_stems, ink))
        acc_symbol = resolve_accidental_symbol(ink, acc, space) if acc else None
        explicit = ("natural" if id(hd) in recovered_naturals else
                    SYM_TO_ACCIDENTAL.get(acc_symbol) if acc_symbol else None)

        # 置信度：符头匹配分 × 步位取整误差惩罚 × 字形自洽性
        step_err = abs(step_f - step)
        conf = float(hd.score) * max(0.0, 1.0 - step_err / 0.5)
        if not hollow and not stem.present:
            conf *= 0.5                    # 实心无符干是不合法字形
        if acc is not None:
            conf = min(conf, float(acc.score))
        if id(hd) in recovered_naturals:
            conf = min(conf, recovered_naturals[id(hd)])

        raw.append((hd.x, {
            "kind": "note", "letter": letter, "octave": octave,
            "explicit": explicit, "duration": dur, "hollow": hollow,
            "stem": stem, "bbox": hd.bbox, "conf": conf, "step": step,
        }))

    for r in rests:
        kind = {SYM_REST_WHOLE: "whole", SYM_REST_HALF: "half",
                SYM_REST_QUARTER: "quarter"}[r.symbol]
        raw.append((r.x, {
            "kind": "rest", "duration": theory.DURATION_TICKS[kind],
            "bbox": r.bbox, "conf": float(r.score),
        }))

    raw.sort(key=lambda t: t[0])

    # --- 2. 用小节线切分小节 ---------------------------------------------
    # 末尾那条小节线之后通常没有内容，只有在确实还有事件时才追加一一尾小芀。
    # 否则会凭空多出一个空小节，导致小节编号和小节数全部错佀    # （实测：4 小节的乐谱被切成 5 个）。
    barlines = sorted(glyphs.barlines)
    max_event_x = max((x for x, _ in raw), default=-1e18)
    bounds: list[tuple[float, float]] = []
    prev = -1e18
    for b in barlines:
        bounds.append((prev, b))
        prev = b
    if cfg.include_trailing_measure is True or (cfg.include_trailing_measure is None
                                                and max_event_x > prev):
        bounds.append((prev, 1e18))
    if not bounds:
        bounds = [(-1e18, 1e18)]

    # ★★ 空小节一往*不产净*（不只是尾小节）。
    #
    # 踩过的坑（这个坑的代价比看起来大得多）：
    # 上面只挡住了"尾小芀这一种空小节＀*内部的空小节没有任何保护**。
    # 只要节线多检出一条、或者相邻两条节线之间没有内容，
    # 就会多产出一一`events=[]` 的小节。后果不昀多一个小芀这么轀—     # `evaluate.flatten()` 是按 **（小节号, 小节冀tick 偏移＀* 给事件编 key 的，
    # 所什*从多出来的那个空小节起，后面所有音符的 key 全部平移**）
    # 退key 对齐随即全部落空。
    #
    # 实测（probe_perspective.py，透视顀+ 抹纸外背景））
    #     序列编辑距离 = 0        ↀ24 个音符的音高+时倀*全部正确、顺序也寀*
    #     音符级准确率 = 4.2%     ↀ同一份结果！
    #     小节敀真倀识别 = 6/12 ↀ多出杀6 个空小节，且 actual_ticks 全为 0
    # 也就是说"准确率崩亀其实昀*分小节崩亀*�?    # 音符一个都没读错（漏检 12 / 多检 12，正好对称，就是 key 平移的典型特征）。
    #
    # 为什乀丢掉"�?留着"更对：单声部乐谱里一个小节不可能一个事件都没有 —     # 真的是空小节也应该是一个全小节休止符（那是 rest 事件，不是没有事件）。
    # 所什空小芀永远是不妀合并"的假设。
    #
    # ☀丢掉之后怎么还能发现这类问题：看 `EvalReport.n_measures_rec` 与真值小节数
    #   是否相等。上面的实测里它报的昀6/12 — **信号一直都在报告里**）
    #   只是当时没往"分小节错亀这个方向读，光盯着准确率了。
    #   所以这里不再单独留一一丢了几个"的计数器：那个数已经能从
    #   小节数之差和小节冀tick 数看出来，再存一份只是重复。
    measures: list[Measure] = []
    mi = 0
    dropped_empty = 0
    dropped_lead = 0
    active_key = cfg.key_fifths
    for bi, (lo, hi) in enumerate(bounds):
        for change_x, new_key in cfg.key_changes:
            if abs(change_x - lo) <= 0.25 * space:
                active_key = new_key
        # ☀保留 x（元组第一项）：和弦分组要甀x 判断"是否同一 onset"
        items = [(x, d) for x, d in raw if lo < x <= hi]
        # ★★ **引子区不是小芀*（本轮查出，昀不足方向"缺口的主要来源）。
        #
        # 现象：每个系统的**第一个小芀*都是空的，而且成对出现在同一系统皀        # 两条谱表上。三个系绀顀× 若干顀ↀ实测 39 个空小节、缺叀69,990 tick）
        # 占全部不足量＀6,550）的 **80.9%**。
        #
        # 根因：`bounds` 的第一项是 `(-∀ 第一条小节线]`，也就是
        # **谱表左端到第一条小节线之间那一殀*。乐谱的排版约定昀        # "谱号 + 拍号"在第一条小节线**左边**（引子区），音乐从小节线才开姀—         # 所以那一段里根本没有音符、只有谱号笔画。
        #
        # 为什么以前没暴露：`keep_empty=False` 时它本来就会被下面的
        # "空小节不产出"挡掉；但多谱表系统必须用 `keep_empty=True`
        # （见 `ReconstructConfig.keep_empty` 的说明：不然同系统两条谱表会脱节），
        # 于是引子区被当成"空小芀**保了下来**，每个系统凭空多一什1920 tick 皀        # "应有釀　*�?9 引入 `keep_empty` 时没有意识到它会顺带保住引子区　*
        #
        # 判据：引子区釀*必须有音笀*才算一个小节；只有谱号/拍号（或什么都没有）
        # 就不是。若第一条小节线其实是谱表中间那条（开谱节线漏检），
        # 那么该区域里就有真音笀ↀ照旧产出，不影响。
        is_lead = bool(barlines) and bi == 0
        drop_lead = (cfg.include_leading_measure is False
                     or (cfg.include_leading_measure is None
                         and not any(d.get("kind") == "note" for _, d in items)))
        if is_lead and drop_lead:
            dropped_lead += 1
            continue
        if not items and not cfg.keep_empty:
            dropped_empty += 1
            continue
        mi += 1
        events: list[Event] = []
        onset = 0
        measure_acc: dict[tuple[str, int], int] = {}  # 小节内临时记号表，跨小节清空
        # ☀和弦分组＀*同一 onset 上的多个符头必须并成一个和开*，不能当戀        #   一串顺序音符。钢琴谱里同符干多符头极常见；当成顺序音符会让该小节皀        #   tick 直接翻倀— 实测误差归因釀和弦未分绀占多出量皀31.8%
        #   （该数是贪心上界，但这一错误本身是确定存在的）。
        #   判据只用"x 几乎相同"：同符干的符夀x 完全相同，含二度和弦会错开纀        #   一个符头宽（≈1 space），而密集十六分音符之间 x 间距通常 �?.3 space）
        #   所什0.55 space 只会合并**确定吀onset**的符头，不会误并顺序音符。
        #   并组吀onset 只前迀*一欀*（前进整个组的时值），这是关销—         #   原来每个音符各前进一次，正是"tick 翻倀的来源。
        chord_dx = cfg.chord_dx * float(glyphs.staff.space)
        cid = 0
        k = 0
        while k < len(items):
            n = k + 1
            if items[k][1]["kind"] != "rest":
                while (n < len(items) and items[n][1]["kind"] != "rest"
                       and abs(float(items[n][0]) - float(items[k][0]))
                       <= chord_dx):
                    n += 1
            group = items[k:n]
            group_dur = max(int(d["duration"]) for _x, d in group)
            if len(group) > 1:
                # A shared chord stem determines the duration.  Extra heads
                # without an independently detected stem often look like
                # quarters (or wholes for hollow heads); taking the maximum
                # incorrectly lengthens every note in a beamed chord.
                stemmed = [d for _x, d in group
                           if d["kind"] == "note" and d["stem"].present]
                if stemmed:
                    beamed = [d for d in stemmed if d["stem"].beams > 0]
                    tailed = (beamed or [d for d in stemmed
                                         if d["stem"].flags > 0])
                    group_dur = (min(int(d["duration"]) for d in tailed)
                                 if tailed else max(int(d["duration"]) for d in stemmed))
                    # A tall three-note chord may overlap a neighboring rest
                    # or slur.  Its upper head can then acquire a false stem
                    # in the opposite direction with one extra flag.  When
                    # two heads agree on one shared stem and duration, let
                    # that repeated evidence decide the whole chord.
                    if not beamed and len(stemmed) >= 3:
                        for direction in ("up", "down"):
                            same = [d for d in stemmed
                                    if d["stem"].direction == direction]
                            other = [d for d in stemmed
                                     if d["stem"].direction != direction]
                            if (len(same) >= 2 and len(same) > len(other)
                                    and all(d["stem"].flags > 0 for d in same)
                                    and len({int(d["duration"]) for d in same}) == 1
                                    and all(int(d["duration"]) < int(same[0]["duration"])
                                            for d in other)):
                                group_dur = int(same[0]["duration"])
                                break
                cid += 1
                group_chord: int | None = cid
            else:
                group_chord = None
            for _x, d in group:
                if d["kind"] == "rest":
                    events.append(Event(
                        type="rest", onset_ticks=onset,
                        duration_ticks=group_dur,
                        confidence=float(d["conf"]),
                        bbox=BBox.from_list(d["bbox"]),
                    ))
                    continue
                letter, octave = d["letter"], d["octave"]
                off = theory.resolve_accidental(
                    letter, d["explicit"], active_key, measure_acc, octave)
                if d["explicit"]:
                    measure_acc[(letter, octave)] = theory.ACCIDENTAL_OFFSET[d["explicit"]]
                keysig_off = theory.key_signature_map(active_key).get(letter, 0)
                events.append(Event(
                    type="note",
                    onset_ticks=onset,
                    duration_ticks=group_dur,
                    pitch=theory.midi_pitch(letter, octave, off),
                    written_name=theory.pitch_name(letter, octave, 0),
                    sounding_name=theory.pitch_name(letter, octave, off),
                    accidental=d["explicit"],
                    stem=d["stem"].direction if d["stem"].present else None,
                    visual_tail_flags=d["stem"].flags if d["stem"].present else None,
                    visual_tail_beams=d["stem"].beams if d["stem"].present else None,
                    visual_stem_tip_y=d["stem"].tip_y if d["stem"].present else None,
                    chord_id=group_chord,
                    confidence=float(d["conf"]),
                    bbox=BBox.from_list(d["bbox"]),
                    note="" if keysig_off == off or d["explicit"] else "",
                ))
            onset += group_dur
            k = n
        first_note_x = next((x for x, d in items if d["kind"] == "note"), lo)
        measures.append(Measure(number=mi, time_signature=ts,
                                voices=[Voice(id=1, events=events)],
                                key_fifths=active_key, clef=clef_at(first_note_x)))

    # 全空页（一个事件都没有）时仍然要有一个小节，否则下游拿到的谱表是空的。
    # 小节数为 0，导出和显示都会出问题。这里保持和老行为一致。
    if not measures:
        measures = [Measure(number=1, time_signature=ts,
                            voices=[Voice(id=1, events=[])],
                            key_fifths=cfg.key_fifths, clef=clef)]

    staff = Staff(
        number=1,
        measures=measures,
        line_ys=list(staff_geo.line_ys),
        space=space,
        clef=clef,
        key_fifths=cfg.key_fifths,
        x0=staff_geo.x0,
        x1=staff_geo.x1,
    )
    score = Score(parts=[Part(name="Melody", staves=[staff])],
                  title=title, confidence=0.0)

    # --- 3. 节拍约束自动纠错 ----------------------------------------------
    # 顺序很重要：**先纠错，再校验**。这样最终报出来的 issues 是纠错之后
    # 仍然对不上的残留问题，数量少且真的需要人工看。
    issues: list[MeasureIssue] = []
    # ★ 丢掉的空小节**要报出来**，不能悄悄丢。
    #
    # 这是那个 bug 的直接教训：空小节本身是"副产品错误"，但它是个很好的
    # **症状指标** —— 出现了就说明节线多检了（或者这个区域的音符全漏检了）。
    # 静默丢掉会让这个症状消失：以前它会以 `actual_ticks=0` 的形式
    # 冒到 `残留问题` 里，丢掉之后就没人知道了。
    #
    # 约定沿用已有的 `measure=0`（"不属于某个具体小节"的通知，
    # 上面 repair 的提示也是这么标的），pipeline 按这个把它挑出来转发。
    if dropped_empty:
        issues.append(MeasureIssue(
            part="Melody", staff=1, measure=0, expected_ticks=0, actual_ticks=0,
            severity="warning",
            message=(f"丢弃 {dropped_empty} 个没有内容的空小节（节线可能多检，"
                     f"或该区域音符全部漏检）")))
    if dropped_lead:
        issues.append(MeasureIssue(
            part="Melody", staff=1, measure=0, expected_ticks=0, actual_ticks=0,
            severity="warning",
            message=(f"丢弃 {dropped_lead} 个引子区（第一条小节线之前只有谱号/拍号，"
                     f"不是小节）")))
    if cfg.repair:
        for msg in repair_triplets(score, ink, staff_geo):
            issues.append(MeasureIssue(part="Melody", staff=1, measure=0,
                                       expected_ticks=0, actual_ticks=0,
                                       severity="repaired", message=msg))
        for msg in drop_spurious_terminal_rests(score):
            issues.append(MeasureIssue(part="Melody", staff=1, measure=0,
                                       expected_ticks=0, actual_ticks=0,
                                       severity="repaired", message=msg))
        for msg in repair_durations(score, max_ratio=cfg.repair_max_ratio,
                                    max_changes=cfg.repair_max_changes,
                                    max_cost=cfg.repair_max_cost):
            issues.append(MeasureIssue(part="Melody", staff=1, measure=0,
                                       expected_ticks=0, actual_ticks=0,
                                       severity="repaired", message=msg))
    issues.extend(validate_measures(score))

    score.confidence = score.mean_confidence()
    return score, issues


# ---------------------------------------------------------------------------
# 节拍约束自动纠错
# ---------------------------------------------------------------------------
#
# 这是整个后处理里**性价比最髀*的一步　#
# 为什么有效：乐理给了我们一个极强的约束 — 每小节的总时值必须等于拍号　# 识别出的时值一旦不满足，就一定有音符判错了。而且错误类型高度集中在两类：
#   * 符尾数数销     -> 时倀×2 戀÷2（八刀十六分混淆）
#   * 附点漏检/误检   -> 时倀×1.5 戀÷1.5
# 所什在标准时值集合里找一组最小改动、让总时值对上拍叀是一个搜索空闀# 很小、命中率很高的问题　#
# 实测：模板检测器在带轻微退化的图上会把某个四分音符的符尾数戀1
# （时倀480 -> 240），单靠调检测参数很难根治；而节拍约束一步就能改回来　

def _valid_duration_ticks() -> list[int]:
    """所最能写出来皀时值（含附点），单佀tick　"""
    vals: set[int] = set()
    for t in theory.DURATION_TICKS.values():
        vals.add(int(t))
        vals.add(int(round(t * 1.5)))
    return sorted(vals)


VALID_DURATIONS: list[int] = _valid_duration_ticks()


def _triplet_digit_template(space: float) -> np.ndarray:
    """Compact numeral template used only after beam and meter checks."""
    size = max(12, int(space * 0.8))
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    tile = Image.new("L", (size * 3, size * 3), 255)
    ImageDraw.Draw(tile).text((size * 1.5, size * 1.5), "3",
                              fill=0, font=font, anchor="mm")
    ink = np.asarray(tile) < 128
    ys, xs = np.nonzero(ink)
    if not len(xs):
        return np.zeros((1, 1), dtype=bool)
    pad = max(2, int(round(0.25 * space)))
    return np.pad(ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1], pad)


def _triplet_marker_score(ink: np.ndarray, groups: list[list[Event]],
                          space: float, components: tuple[np.ndarray, np.ndarray],
                          template: np.ndarray,
                          allow_connected: bool = False) -> float:
    """Read the small isolated 3 above or below a three-stem beam group."""
    stemmed = [e for group in groups for e in group
               if e.stem in ("up", "down") and e.visual_stem_tip_y is not None]
    if len(stemmed) < 2 or len({e.stem for e in stemmed}) != 1:
        return 0.0
    up = stemmed[0].stem == "up"
    x = sum(group[0].bbox.cx for group in groups if group[0].bbox) / 3
    tips = [float(e.visual_stem_tip_y) for e in stemmed]
    tip = min(tips) if up else max(tips)
    y = tip - 1.05 * space if up else tip + 0.8 * space
    stats, centres = components
    compact = any(
        abs(float(cx) - x) <= 0.8 * space
        and abs(float(cy) - y) <= 1.4 * space
        and 0.30 * space <= int(st[3]) <= 1.5 * space
        and int(st[2]) <= 1.15 * space
        and int(st[4]) >= 0.4 * space
        for st, (cx, cy) in zip(stats, centres)
    )
    if not compact and not allow_connected:
        return 0.0
    h, w = template.shape
    xa = max(0, int(x - 0.65 * space - w / 2))
    xb = min(ink.shape[1], int(x + 0.65 * space + w / 2))
    ya = max(0, int(y - space - h / 2))
    yb = min(ink.shape[0], int(y + space + h / 2))
    region = ink[ya:yb, xa:xb].astype(np.float32)
    if region.shape[0] < h or region.shape[1] < w:
        return 0.0
    on = cv2.matchTemplate(region, template.astype(np.float32),
                           cv2.TM_CCORR) / max(1, int(template.sum()))
    offmask = ~template
    off = cv2.matchTemplate(region, offmask.astype(np.float32),
                            cv2.TM_CCORR) / max(1, int(offmask.sum()))
    score = float(np.max(on - 0.8 * off))
    # A printed numeral may touch its beam after photographic resampling, so
    # connected components no longer expose a compact isolated 3.  Only the
    # caller's meter-and-spacing guarded path may use its raw glyph score.
    return score if compact or score >= 0.58 else 0.0


def _missed_eighth_triplet_beam(ink: np.ndarray,
                               groups: list[list[Event]],
                               space: float) -> bool:
    """Confirm an eighth-note beam when three heads were read as quarters.

    A missing third stem and a chord head at the first onset can interrupt the
    ordinary adjacent-stem regularizer.  Two measured bridges, including one
    spanning all three onsets, are stronger evidence than the local tail count.
    """
    if len(groups) != 3 or not all(group[0].duration_ticks == PPQ for group in groups):
        return False
    first, middle, last = (group[0] for group in groups)
    if any(event.type != "note" or event.bbox is None
           for event in (first, middle, last)):
        return False
    if (first.stem not in ("up", "down") or middle.stem != first.stem
            or first.visual_stem_tip_y is None
            or middle.visual_stem_tip_y is None
            or last.stem not in (None, first.stem)):
        return False
    heads = [Detection(SYM_NOTEHEAD_FILLED, event.bbox.cx, event.bbox.cy, 1.0)
             for event in (first, middle, last)]
    stems = [StemInfo(present=True, direction=event.stem,
                      tip_y=float(event.visual_stem_tip_y))
             for event in (first, middle)]
    last_stem = (StemInfo(present=True, direction=last.stem,
                          tip_y=float(last.visual_stem_tip_y))
                 if last.visual_stem_tip_y is not None and last.stem is not None
                 else StemInfo())
    return (bridge_beam_count(ink, heads[0], heads[1], stems[0], stems[1], space) == 1
            and bridge_beam_count(ink, heads[0], heads[2], stems[0],
                                  last_stem, space) == 1)


def repair_triplets(score: Score, ink: np.ndarray,
                    staff_geo: StaffGeometry) -> list[str]:
    """Convert three single-beamed eighths only with a visible tuplet 3.

    A three-note beam by itself is an ordinary eighth group.  A strong,
    isolated numeral is sufficient; a weaker numeral also needs an overfull
    bar so arbitrary compact marks cannot shorten notes.
    """
    space = float(staff_geo.space)
    y0 = max(0, int(staff_geo.y_top_line - 6 * space))
    y1 = min(ink.shape[0], int(staff_geo.y_bottom_line + 6 * space))
    _n, _labels, local_stats, local_centres = cv2.connectedComponentsWithStats(
        ink[y0:y1].astype(np.uint8), 8)
    stats = local_stats[1:]
    centres = local_centres[1:].copy()
    centres[:, 1] += y0
    template = _triplet_digit_template(space)
    repaired: list[str] = []
    for _part, _staff, measure in score.iter_measures():
        if len(measure.voices) != 1:
            continue
        overflow = measure.actual_ticks() - measure.expected_ticks
        events = measure.voices[0].events
        groups: list[list[Event]] = []
        for event in events:
            if not groups or groups[-1][0].onset_ticks != event.onset_ticks:
                groups.append([event])
            else:
                groups[-1].append(event)
        candidates: list[tuple[float, int, bool]] = []
        for i in range(len(groups) - 2):
            triple = groups[i:i + 3]
            regular_eighths = all(
                group[0].type == "note" and
                group[0].duration_ticks == PPQ // 2 and
                any((e.visual_tail_beams or 0) >= 1 for e in group)
                for group in triple)
            missed_beam = (overflow >= 2 * PPQ and
                           _missed_eighth_triplet_beam(ink, triple, space))
            if not (regular_eighths or missed_beam):
                continue
            if any(group[0].bbox is None for group in triple):
                continue
            xs = [group[0].bbox.cx for group in triple]
            stem_directions = {event.stem for group in triple for event in group
                               if event.stem in ("up", "down")}
            # The slightly wider photographic spacing occurs in an up-stem
            # group with a visible 3.  Widening down-stem groups also matched
            # ordinary beams in two independently inspected real measures.
            spacing_limit = 3.2 if stem_directions == {"up"} else 3.0
            compact_spacing = max(xs[1] - xs[0], xs[2] - xs[1]) <= spacing_limit * space
            confidence = _triplet_marker_score(
                ink, triple, space, (stats, centres), template,
                allow_connected=(not missed_beam and overflow > 0
                                 and compact_spacing))
            # A photographed Verovio numeral typically matches the compact
            # Arial template at 0.62-0.74 after resampling.  Requiring 0.80
            # discards those visible numerals when another error leaves the
            # measure underfull, even though the marker is independently
            # supported by a compact component and a three-stem beam group.
            if confidence >= 0.60 or (not missed_beam and confidence >= 0.35
                                      and overflow > 0):
                candidates.append((confidence, i, missed_beam))
        selected: set[int] = set()
        group_limit = (max(1, int(math.ceil(overflow / (PPQ // 2))))
                       if overflow > 0 else 2)
        for confidence, i, missed_beam in sorted(candidates, reverse=True):
            if len(selected) >= group_limit or any(abs(i - j) < 3 for j in selected):
                continue
            selected.add(i)
            for role, group in zip(("start", "continue", "stop"), groups[i:i + 3]):
                for event in group:
                    event.duration_ticks = PPQ // 3
                group[0].tuplet = role
            repaired.append(f"measure {measure.number}: visible triplet 3, "
                            f"{'missed beam' if missed_beam else 'three eighths'} "
                            f"-> 3x{PPQ // 3} ticks "
                            f"(marker score {confidence:.2f})")
        if selected:
            onset = 0
            for group in groups:
                for event in group:
                    event.onset_ticks = onset
                onset += max(event.duration_ticks for event in group)
    return repaired


def drop_spurious_terminal_rests(score: Score) -> list[str]:
    """Discard a weak trailing rest when preceding notes already fill a bar.

    A rest template often fires on the last note's tail or the end barline.
    The exact meter equality is deliberately strict: it supplies independent
    evidence that this isolated low-confidence rest is impossible here.
    """
    messages: list[str] = []
    for _part, _staff, measure in score.iter_measures():
        if len(measure.voices) != 1:
            continue
        events = measure.voices[0].events
        if (len(events) < 3 or events[-1].type != "rest" or
                events[-1].confidence >= 0.65):
            continue
        preceding = events[:-1]
        if (sum(event.type == "note" for event in preceding) >= 2 and
                sum(event.duration_ticks for event in preceding) == measure.expected_ticks):
            events.pop()
            messages.append(f"小节 {measure.number}: 移除末尾误检休止符（音符时值已填满小节）")
    return messages


def _change_cost(old: int, new: int) -> float:
    """一次时值改动的代价 —— **把识别错误的统计规律编码进去**。"

    ★ 这不是随手选的权重，它决定了纠错到底是在帮忙还是在捣乱。

    识别时值只有两条主要出错路径：
      (1) **符尾数数错**：时值恰好 ×2 / ÷2 / ×4 / ÷4      ← 最常见
      (2) **附点漏检或误检**：时值恰好 ×1.5 / ÷1.5

    如果对所有改动一视同仁地按 |log2(比值)| 计价，会得到反直觉且更差的结果：
    小节 `[480, 240, 480, 480]`（总 1680，按 4/4 应为 1920）里，
      * "把 480 加成附点 720"   比值 1.5，代价 0.58  ← 会被选中
      * "把误判的 240 改回 480" 比值 2.0，代价 1.00
    于是纠错把一个**本来正确**的音符改错了。实测退化集音符级准确率
    从 93.3% 掉到 86.7% —— 纠错反而降低了精度。

    所以符尾类修正必须比附点类修正更便宜。
    """
    if old == new:
        return 0.0
    if old <= 0 or new <= 0:
        return 99.0
    r = new / old
    lg = math.log2(r)
    if abs(lg - round(lg)) < 1e-9:              # ×2 / ÷2 / ×4 / ÷4：符尾数数错
        return 0.8 * abs(lg)
    if abs(r - 1.5) < 1e-9 or abs(r - 2.0 / 3.0) < 1e-9:   # 附点
        return 1.0
    return 1.5 + abs(lg)


def _repair_one_measure(events: list[Event], expected: int,
                        max_ratio: float = 3.0,
                        max_changes: int = 2,
                        max_cost: float = 1.6) -> list[int] | None:
    """在候选时值里找一组"代价最小、总和恰好等于拍号"的改动。"

    用动态规划：`best[s]` = 前 i 个事件凑出总时值 s 的最小代价。
    代价见 `_change_cost`。

    ★ 两个保守性闸门（`max_changes` / `max_cost`）是必需的。
    原因是实测发现：**当整小节识别严重失败时（比如旋转导致音符整个漏检），
    靠改时值去凑拍号只会把本来正确的音符也改错** —— 退化扫描里
    "旋转 1.5°"场景的准确率会因为纠错从 93.3% 掉到 80%。
    一个小节内需要改动的音符数一旦超过 2，通常就不是"几处误判"，
    而是根本性的漏检/多检，这时候正确的做法是**放弃纠错、把问题报出来**。

    ⚠️ 诚实地说：**单靠时值总和存在天然歧义** ——
    `[480,240,480,480]` 补成 4/4，既可以解读为"某个音被漏了附点"，
    也可以解读为"某个音被少数了一条符尾"，两者在时值上都成立。
    这里用错误模型的先验（符尾错 > 附点错）来打破平局，
    并且把**所有被改动的音符降为可疑**，交给人工复核确认。

    Returns
    -------
    新的时值列表；无解或超出保守性闸门时返回 None。
    """
    orig = [e.duration_ticks for e in events]
    if not events or expected <= 0:
        return None
    if sum(orig) == expected:
        return orig

    lo = math.log2(1.0 / max_ratio) - 1e-9
    hi = math.log2(max_ratio) + 1e-9
    cands: list[list[int]] = []
    for d in orig:
        if d <= 0:
            return None
        c = [v for v in VALID_DURATIONS if lo <= math.log2(v / d) <= hi]
        cands.append(sorted(set(c)) or [d])

    # best: 总时倀-> (累计代价, 已选时值元绀
    best: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for i, cs in enumerate(cands):
        nxt: dict[int, tuple[float, tuple[int, ...]]] = {}
        for s, (cost, chosen) in best.items():
            for v in cs:
                ns = s + v
                if ns > expected:
                    continue
                change_cost = _change_cost(orig[i], v)
                # A second local tail on an otherwise single-beamed note is
                # evidence for a sixteenth at the end of a partial beam.
                # Use it only to rank competing meter repairs; the bridge
                # remains the primary duration reading.
                if (orig[i] == 240 and v == 120 and
                        events[i].visual_tail_flags is not None and
                        events[i].visual_tail_flags >= 2 and
                        events[i].visual_tail_beams == 1):
                    change_cost -= 0.25
                nc = cost + change_cost
                if nc > max_cost + 1e-9:
                    continue                      # 闸门 1：累计代价过早剪枝
                cur = nxt.get(ns)
                if cur is None or nc < cur[0]:
                    nxt[ns] = (nc, chosen + (v,))
        best = nxt
        if not best:
            return None
    hit = best.get(expected)
    if hit is None:
        return None
    cost, chosen = hit
    n_changes = sum(1 for o, c in zip(orig, chosen) if o != c)
    if n_changes > max_changes or cost > max_cost + 1e-9:
        return None                               # 闸门 2：改动太多，宁可不改
    return list(chosen)


def repair_durations(score: Score, max_ratio: float = 3.0,
                     max_events: int = 20,
                     max_changes: int = 2,
                     max_cost: float = 1.6) -> list[str]:
    """Repair one locally ambiguous duration when the meter differs by a sixteenth.

    A meter mismatch can also come from a missing or extra symbol.  Forcing
    several high-confidence notes to sum to the expected length was measured
    to lower accuracy on the complex synthetic page, so only a single edit
    backed by low confidence or contradictory tail evidence is applied.
    """
    notes: list[str] = []
    for _p, _s, m in score.iter_measures():
        exp = m.expected_ticks
        if not m.voices:
            continue
        # 只处理恰好一个声部、且确实对不上的小节
        active = [v for v in m.voices if v.events]
        if len(active) != 1:
            continue
        v = active[0]
        before = sum(e.duration_ticks for e in v.events)
        if before == exp:
            continue
        if abs(before - exp) != PPQ // 4:
            continue
        if len(v.events) > max_events:
            continue
        new = _repair_one_measure(v.events, exp, max_ratio=max_ratio,
                                  max_changes=max_changes, max_cost=max_cost)
        if new is None:
            continue
        changed = [(index, event, duration)
                   for index, (event, duration) in enumerate(zip(v.events, new))
                   if event.duration_ticks != duration]
        if len(changed) != 1:
            continue
        index, event, duration = changed[0]
        preceding = v.events[index - 1] if index > 0 else None
        tail_support = (
            before < exp and event.duration_ticks == 120 and duration == 240
            and (event.visual_tail_flags or 0) >= 2
            and (event.visual_tail_beams or 0) == 0
        ) or (
            before > exp and event.duration_ticks == 240 and duration == 120
            and (event.visual_tail_flags or 0) >= 2
            and (event.visual_tail_beams or 0) == 1
        ) or (
            # A dotted eighth followed by a single-beam eighth can hide the
            # short secondary beam hook next to the second stem.  If the
            # measure is over by exactly one sixteenth, the hook is the one
            # plausible local correction even when notehead confidence is high.
            before > exp and event.duration_ticks == 240 and duration == 120
            and index == len(v.events) - 1 and preceding is not None
            and preceding.duration_ticks == 360
            and (preceding.visual_tail_beams or 0) == 1
            and (event.visual_tail_beams or 0) == 1
        )
        if event.confidence > 0.6 and not tail_support:
            continue
        changes: list[str] = []
        onset = 0
        for e, nd in zip(v.events, new):
            if nd != e.duration_ticks:
                label = "rest" if e.type == "rest" else (e.sounding_name or "?")
                changes.append(f"{label} {e.duration_ticks}->{nd}")
                e.note = (e.note + " " if e.note else "") + \
                    f"duration repaired {e.duration_ticks}->{nd}"
                e.duration_ticks = int(nd)
                e.confidence = min(e.confidence, 0.6)   # 纠错过的音符标为可疑
            e.onset_ticks = onset
            onset += e.duration_ticks
        if changes:
            notes.append(f"小节 {m.number}: {before} -> {exp} tick，"
                         f"纠正 {len(changes)} 处（{'; '.join(changes)}）")
    return notes


# ---------------------------------------------------------------------------
# 拍号反推 — 用节拍约束反过来纠正/推断拍号
# ---------------------------------------------------------------------------

COMMON_TIME_SIGNATURES: list[tuple[int, int]] = [
    (4, 4), (3, 4), (2, 4), (2, 2), (6, 8), (3, 8), (12, 8), (9, 8), (5, 4),
]


def infer_time_signature(score: Score,
                         candidates: list[tuple[int, int]] | None = None
                         ) -> tuple[tuple[int, int] | None, list[tuple[int, int]]]:
    """在候选拍号里找"能让所有小节都校验通过"的那个。"

    这是节拍约束的另一种用法：**不依赖拍号字形检测**也能定出拍号。
    返回 (唯一解或 None, 所有通过的候选)。多个候选都通过时返回 None，
    因为此时信息不足（例如 4/4 与 2/2 在纯音符时值上天然无法区分）。
    """
    cands = candidates or COMMON_TIME_SIGNATURES
    ok: list[tuple[int, int]] = []
    for ts in cands:
        good = True
        for _p, _s, m in score.iter_measures():
            if not m.voices:
                continue
            if m.actual_ticks() != theory.measure_ticks(ts):
                good = False
                break
        if good:
            ok.append(ts)
    return (ok[0] if len(ok) == 1 else None), ok
