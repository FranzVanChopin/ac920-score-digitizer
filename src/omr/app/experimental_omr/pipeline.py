"""
端到端编排。

这一层刻意保持"薄"：它只负责把各模块串起来，不做任何算法决策。
好处是 P3 移植到板子时，这个文件的逻辑可以一比一翻译成 C++ 或 Python，
而每个模块也可以单独替换（尤其是 detect.py 的检测器）。

数据流
------
    灰度图
      └─ preprocess()          -> 二值图 (去斜 + 去噪)
           └─ detect_staves()  -> [StaffGeometry]
                └─ 对每个谱表:
                     remove_staff_lines()  -> 去线墨迹
                     detect_symbols()      -> GlyphSet
                     reconstruct()         -> Score(单谱表)
                └─ 合并 -> Score(整页) -> 节拍校验
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from . import detect as detect_mod
from . import preprocess as pp_mod
from . import staff as staff_mod
from .detect import (GlyphSet, SYM_NOTEHEAD_FILLED, SYM_NOTEHEAD_HOLLOW,
                     SYM_REST_HALF, SYM_REST_QUARTER, SYM_REST_WHOLE)
from .model import Part, Score, merge_scores, merge_system_staves, validate_measures
from .preprocess import PreprocessConfig, PreprocessResult
from .key_signature import inline_key_changes, resolve_system_keys
from .inline_clef import find_inline_bass_clefs
from .reconstruct import ReconstructConfig, reconstruct
from .staff import StaffGeometry


@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    reconstruct: ReconstructConfig = field(default_factory=ReconstructConfig)
    #: "template"（Baseline A，几何/模板法）| "onnx"（Baseline B，待接入）
    #: | "cnn"（Baseline C，本项目的谱表带热力图 CNN，需要 `cnn_weights`）
    detector: str = "template"
    onnx_model: str | None = None
    onnx_conf: float = 0.35
    #: CNN 检测器的权重路径（`tools/train_symbol_cnn.py` 的产物）
    cnn_weights: str | None = None
    cnn_device: str = "cpu"
    cnn_min_score: float = 0.35
    cnn_nms_radius: int = 3
    #: 谱表检测时，行墨迹覆盖率的相对阈值（只对 "lines" 检测器有效）
    staff_min_coverage: float = 0.5
    #: 谱表检测器："comb"（默认，5 条等距线整体匹配，抗光照衰减）
    #:             | "lines"（旧的两段式：全局相对阈值 + 贪心 5 线分组）
    #: 真实照片上 "lines" 会把页面两端的谱表整条漏掉（实测 10 条只检出 5 条），
    #: 所以默认走 "comb"；保留 "lines" 是为了合成图上的黄金基准可比。
    staff_detector: str = "comb"
    #: 输入已知是钢琴大谱表时，用系统分组确定上方高音谱号、下方低音谱号。
    piano_grand_staff: bool = False
    #: 从原图人工确认的拍号，可覆盖不可靠的模板数字识别。
    meter_override: tuple[int, int] | None = None
    #: 人工确认的调号；未指定时逐系统从引子区识别。
    key_override: int | None = None
    title: str = ""


@dataclass
class PageResult:
    """一整页的识别结果与全部中间产物（调参/排错都靠它）。"""

    score: Score
    glyphs: list[GlyphSet] = field(default_factory=list)
    #: 逐谱表的识别结果（与 `staves` 一一对应）。`score` 是它们合并后的整页结果，
    #: 但排错时必须能回到"单个谱表"这一层 —— 多声部之前，整页合并是**顺序拼接**，
    #: 只看整页结果无法判断某个谱表自己是好是坏。
    staff_scores: list[Score] = field(default_factory=list)
    staves: list[StaffGeometry] = field(default_factory=list)
    systems: list[list[int]] = field(default_factory=list)
    pre: PreprocessResult | None = None
    issues: list = field(default_factory=list)
    #: 每个谱表去线后的墨迹图，便于可视化
    ink_per_staff: list[np.ndarray] = field(default_factory=list)
    primary_staff_count: int = 0
    supplemental_staff_count: int = 0
    clef_override_count: int = 0
    inline_clef_count: int = 0
    inline_clef_x_by_staff: list[list[float]] = field(default_factory=list)
    suppressed_opening_segments: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues


def make_detector(cfg: PipelineConfig):
    if cfg.detector == "template":
        return detect_mod.TemplateDetector(expected_meter=cfg.meter_override)
    if cfg.detector == "onnx":
        return detect_mod.OnnxDetector(cfg.onnx_model or "", conf=cfg.onnx_conf)
    if cfg.detector == "cnn":
        from .cnn import CnnDetector
        if not cfg.cnn_weights:
            raise ValueError("detector='cnn' 需要 cfg.cnn_weights 指向权重文件"
                             "（先跑 tools/train_symbol_cnn.py）")
        return CnnDetector(cfg.cnn_weights, device=cfg.cnn_device,
                           min_score=cfg.cnn_min_score,
                           nms_radius=cfg.cnn_nms_radius)
    raise ValueError(f"未知检测器: {cfg.detector}")


def has_treble_clef_extents(glyphs: GlyphSet) -> bool:
    """Detect a clear G clef at a system start by ink above and below staff.

    A bass clef sits almost entirely inside its five lines, while a treble
    clef extends beyond both outer lines.  Only the narrow clef column near
    the left edge is measured; requiring both extensions avoids mistaking an
    isolated high note, low chord or page shadow for a clef change.
    """
    staff, ink = glyphs.staff, glyphs.ink
    if staff is None or ink is None:
        return False
    space = staff.space
    x0 = max(0, int(round(staff.x0 + 1.75 * space)))
    x1 = min(ink.shape[1], int(round(staff.x0 + 4.5 * space)))
    top = int(round(staff.y_top_line))
    bottom = int(round(staff.y_bottom_line))
    if x1 <= x0 or top < 0 or bottom >= ink.shape[0]:
        return False
    above = ink[max(0, int(round(top - 1.1 * space))):
                max(0, int(round(top - .15 * space))), x0:x1]
    below = ink[min(ink.shape[0], int(round(bottom + .15 * space))):
                min(ink.shape[0], int(round(bottom + 1.1 * space))), x0:x1]
    far_below = ink[min(ink.shape[0], int(round(bottom + .6 * space))):
                    min(ink.shape[0], int(round(bottom + 1.1 * space))), x0:x1]
    middle = ink[top:bottom + 1, x0:x1]
    if not above.size or not below.size or not middle.size:
        return False
    support = float(middle.sum())
    return (support > 0 and float(above.sum()) >= .08 * support
            and float(below.sum()) >= .11 * support
            and float(far_below.sum()) >= .018 * support)


def apply_piano_grand_staff_clefs(glyph_sets: list[GlyphSet], systems: list[list[int]]) -> int:
    """Set piano clefs by position, allowing a proven treble lower staff."""
    overridden = 0
    for group in systems:
        if len(group) != 2:
            continue
        lower = "treble" if has_treble_clef_extents(glyph_sets[group[1]]) else "bass"
        for index, expected in zip(group, ("treble", lower)):
            if glyph_sets[index].clef != expected:
                glyph_sets[index].clef = expected
                overridden += 1
    return overridden


def apply_meter_override(score: Score, meter: tuple[int, int]) -> None:
    """将人工确认的拍号应用到候选谱，随后再做小节时值检查。"""
    for part in score.parts:
        for staff in part.staves:
            for measure in staff.measures:
                measure.time_signature = meter


def pair_piano_staves(staves: list[StaffGeometry]) -> list[list[int]] | None:
    """仅在每条上下谱表的间隙合理时按钢琴大谱表成对分组。"""
    if not staves or len(staves) % 2:
        return None
    pairs = [[i, i + 1] for i in range(0, len(staves), 2)]
    # Perspective can stretch the first grand-staff gap more than the local
    # line spacing.  A 9-space cap split one photographed pair into two
    # systems, doubling its measure boundaries after independent barline
    # detection.  Keep a conservative upper bound for a piano layout.
    if any(staves[j].y_top_line - staves[i].y_bottom_line > 11.5 * staves[j].space
           for i, j in pairs):
        return None
    return pairs


def align_piano_staff_extents(staves: list[StaffGeometry],
                              pairs: list[list[int]] | None) -> None:
    """Use the common horizontal extent of both staves in a piano system.

    A weak line at one end can move one staff's measured x0 past several real
    notes.  The other staff supplies independent evidence for that edge.
    """
    for pair in pairs or []:
        left = min(staves[i].x0 for i in pair)
        right = max(staves[i].x1 for i in pair)
        for i in pair:
            staves[i].x0 = left
            staves[i].x1 = right


def paired_measure_boundaries(glyph_sets: list[GlyphSet], group: list[int]) -> tuple[bool, bool] | None:
    """成对谱表任一成员有音符/休止符时，两者都保留该首/尾小节。"""
    if len(group) < 2 or not glyph_sets[group[0]].barlines:
        return None
    first = glyph_sets[group[0]].barlines[0]
    last = glyph_sets[group[0]].barlines[-1]
    symbols = {SYM_NOTEHEAD_FILLED, SYM_NOTEHEAD_HOLLOW,
               SYM_REST_WHOLE, SYM_REST_HALF, SYM_REST_QUARTER}
    positions = [d.x for index in group for d in glyph_sets[index].detections
                 if d.symbol in symbols]
    note_positions = [d.x for index in group for d in glyph_sets[index].detections
                      if d.symbol in (SYM_NOTEHEAD_FILLED, SYM_NOTEHEAD_HOLLOW)]
    geoms = [glyph_sets[index].staff for index in group
             if glyph_sets[index].staff is not None]
    opening_bar = bool(geoms) and first <= float(np.median(
        [s.x0 + 7.0 * s.space for s in geoms]))
    # 谱号/拍号偶尔被误报成符头。若第一根线就在谱表左端的引子区，
    # 它是开谱节线；这些误报不能凭空生成一个前置小节。
    end_space = float(np.median([s.space for s in geoms])) if geoms else 0.0
    # 末尾节线旁的笔画容易误报为休止符；仅用明确的符头建立尾小节。
    # 只有休止符的真实尾小节需要更强的休止符/节线联合判据，暂留待复核。
    return (not opening_bar and any(x <= first for x in positions),
            any(x > last + 1.2 * end_space for x in note_positions))


def suppress_short_empty_openings(staff_scores: list[Score],
                                  glyph_sets: list[GlyphSet],
                                  systems: list[list[int]]) -> int:
    """Reject a narrow pre-music segment created by a second opening line.

    Time-signature fragments may look like rests between the opening staff
    line and the actual first measure line.  Require the same narrow segment
    on both piano staves and no reconstructed note on either staff.
    """
    removed = 0
    for group in systems:
        if len(group) != 2:
            continue
        glyphs = glyph_sets[group[0]]
        bars = glyphs.barlines
        staff = glyphs.staff
        if staff is None or len(bars) < 2:
            continue
        space = float(staff.space)
        if (bars[0] > staff.x0 + 3 * space
                or bars[1] - bars[0] >= 10 * space):
            continue
        measures = [staff_scores[i].parts[0].staves[0].measures for i in group]
        if any(len(series) < 2 for series in measures):
            continue
        first_events = [event for series in measures
                        for voice in series[0].voices for event in voice.events]
        if any(event.is_note for event in first_events):
            continue
        for series in measures:
            series.pop(0)
        removed += 1
    return removed


def recognize(gray: np.ndarray, cfg: PipelineConfig | None = None,
              binary_override: np.ndarray | None = None,
              staff_supplement: Callable[[np.ndarray], list[StaffGeometry]] | None = None) -> PageResult:
    """把一页乐谱图像识别成 Score。

    Parameters
    ----------
    gray
        uint8 灰度图、彩色图或二值图都行（内部会统一转灰度）。
    binary_override
        **外部算好的二值图**（bool 或 0/1 uint8）。给了它就不再跑预处理。

        这个接口是为 FPGA 集成准备的：下沉到 PL 之后，预处理由 PL 出结果，
        PS 只负责"拿二值图 → 谱表检测 → 符号检测 → 语义重建"。
        有了它，"PL 输出的二值图"和"Python 算的二值图"可以走**完全相同的下游**，
        于是能直接验证"换成 PL 之后识别精度有没有变化" ——
        这是 FPGA 适配里最要紧的一条验收。
    staff_supplement
        可选的独立谱表定位器，接收已旋正的灰度图。仅补充与原定位
        结果不重合、且间距与本页一致的谱表。
    """
    cfg = cfg or PipelineConfig()

    # --- 1. 预处理 -------------------------------------------------------
    if binary_override is not None:
        b = np.asarray(binary_override)
        binary = (b > 0) if b.dtype != bool else b
        gray_img = pp_mod.ensure_gray(gray)
        # 归一化图没有意义（已经是外部结果），占位成灰度原图
        pre = PreprocessResult(gray=gray_img, normalized=gray_img,
                               binary=binary, skew_angle=0.0,
                               config=cfg.preprocess)
    else:
        pre = pp_mod.preprocess(gray, cfg.preprocess)
        binary = pre.binary

    # --- 2. 谱表检测 -----------------------------------------------------
    # ★ 垂直游程图只跟二值图有关、与谱表无关，所以在整页上**只算一次**，
    # 之后所有谱表复用。一张 10 行谱表的 A4 页有 8.7M 像素，
    # 每行谱表各算一遍的话（检测粗细 + 去线）就是 20 多次全图扫描，秒级开销。
    run_lengths = staff_mod.vertical_run_length(binary)

    if cfg.staff_detector == "comb":
        staves = staff_mod.detect_staves_comb(binary, run_lengths=run_lengths)
    else:
        staves = staff_mod.detect_staves(binary, min_coverage=cfg.staff_min_coverage,
                                         run_lengths=run_lengths)
    if cfg.piano_grand_staff and cfg.staff_detector == "comb":
        local_staves = staff_mod.detect_staves_multistrip(
            binary, run_lengths=run_lengths)
        def spacing_dispersion(items: list[StaffGeometry]) -> float:
            spaces = np.asarray([staff.space for staff in items], dtype=float)
            median = float(np.median(spaces))
            return float(np.sqrt(np.mean(((spaces - median) / median) ** 2)))

        # Both methods may return a complete even count while the global
        # projection has locked one or two staves onto the wrong line comb.
        # Prefer the local geometry only if it substantially reduces the
        # page's otherwise anomalous line-spacing scatter.
        better_local_geometry = False
        if len(local_staves) == len(staves) and len(staves) >= 2:
            global_scatter = spacing_dispersion(staves)
            local_scatter = spacing_dispersion(local_staves)
            better_local_geometry = (global_scatter > .03 and
                                     local_scatter < .025 and
                                     local_scatter < .65 * global_scatter)
        if (len(local_staves) >= 2 and len(local_staves) % 2 == 0 and
                (len(staves) % 2 != 0 or len(local_staves) > len(staves)
                 or better_local_geometry)):
            staves = local_staves
    primary_ids = {id(staff) for staff in staves}
    primary_staff_count = len(staves)
    supplemental_staff_count = 0
    row_profile = staff_mod.row_ink_profile(binary)
    if staff_supplement is not None:
        candidates = staff_supplement(pre.gray)
        reference_space = float(np.median([s.space for s in staves or candidates])) if (staves or candidates) else 0.0
        for candidate in candidates:
            if len(candidate.line_ys) != 5 or candidate.y_top_line < 0 or candidate.y_bottom_line >= binary.shape[0]:
                continue
            if reference_space <= 0 or abs(candidate.space - reference_space) > 0.18 * reference_space:
                continue
            if any(abs(candidate.y_top_line - known.y_top_line) < 2.5 * reference_space for known in staves):
                continue
            candidate.x0, candidate.x1 = staff_mod.staff_h_extent(binary, candidate)
            candidate.thickness = staff_mod.estimate_line_thickness(binary, candidate, run_lengths=run_lengths)
            staves.append(candidate)
        # The independent detector can add a plausible five-line comb made
        # from beams or text.  Such candidates sometimes overlap a real staff
        # by three line spaces, beyond the old top-line-only deduplication.
        # Reject ink bands as thick as a whole staff space, then compare the
        # actual five-line support of all primary and supplemental candidates.
        staves = [staff for staff in staves
                  if staff.thickness <= 0.45 * staff.space]
        staves = staff_mod.prune_overlapping_staves(staves, row_profile)
        primary_staff_count = sum(id(staff) in primary_ids for staff in staves)
        supplemental_staff_count = len(staves) - primary_staff_count
    if not staves:
        return PageResult(
            score=Score(title=cfg.title), pre=pre,
            issues=[], glyphs=[], staves=[],
            primary_staff_count=primary_staff_count,
            supplemental_staff_count=supplemental_staff_count,
        )

    for staff in staves:
        staff.local_offsets = staff_mod.estimate_local_line_offsets(binary, staff)

    piano_pairs = pair_piano_staves(staves) if cfg.piano_grand_staff else None
    align_piano_staff_extents(staves, piano_pairs)

    detector = make_detector(cfg)
    glyph_sets: list[GlyphSet] = []
    staff_scores: list[Score] = []
    ink_list: list[np.ndarray] = []
    notices: list = []

    # --- 3a. 每个谱表：去线 -> 字形检测 --------------------------------
    # ★ 先只做检测、先不重建：小节线要等**系统级统一**之后再定。
    for s in staves:
        ink_full = staff_mod.remove_staff_lines(binary, s, run_lengths=run_lengths)
        # ★ 必须把 `binary`（去线**之前**）一并传下去：
        #   "节线必须坐落在谱表上"这条判据要的谱线证据只存在于 binary 里
        #   （`ink_full` 里谱线已经被抹掉）。见 `detect._staff_side_hits`。
        glyphs = detector.detect(ink_full, s, binary=binary)
        glyph_sets.append(glyphs)
        ink_list.append(ink_full)

    # --- 3b. 跨谱表幽灵去重 ---------------------------------------------
    # ★ 必须放在"逐谱表重建"**之前**：幽灵是落进**别的谱表**里的音符，
    #   一旦重建完再删，小节数/onset 都已经错了。
    #   这里也必须在 `group_systems` 之前 —— 它只动 detections、不动 barlines，
    #   而系统分组依赖 barlines。
    n_ghosts = detect_mod.drop_cross_staff_ghosts(glyph_sets, staves)

    # Supplemental gray-image localization can add a paper edge or tempo
    # symbols with five apparent lines.  Compare their weakest line with
    # primary staves only after cross-staff duplicate heads are removed:
    # the tempo glyph on sonata page 9 initially has 35 head detections but
    # only eight survive duplicate elimination.  A genuine faint staff in
    # the first reference photo still has 19 heads and five barlines.
    primary_support = [min(int(row_profile[round(y)]) for y in staff.line_ys)
                       for staff in staves if id(staff) in primary_ids]
    if primary_support:
        floor = 0.20 * float(np.median(primary_support))

        def near_primary_staff(candidate: StaffGeometry) -> bool:
            return any(
                0 <= known.y_top_line - candidate.y_bottom_line
                <= 4 * candidate.space
                or 0 <= candidate.y_top_line - known.y_bottom_line
                <= 4 * candidate.space
                for known in staves if id(known) in primary_ids)

        kept_indices = [i for i, (staff, glyphs) in enumerate(zip(staves, glyph_sets))
                        if (id(staff) in primary_ids
                            or min(int(row_profile[round(y)]) for y in staff.line_ys)
                            >= floor
                            or len(glyphs.noteheads()) >= 10
                            or (glyphs.barlines and not near_primary_staff(staff)))]
        if len(kept_indices) != len(staves):
            staves = [staves[i] for i in kept_indices]
            glyph_sets = [glyph_sets[i] for i in kept_indices]
            ink_list = [ink_list[i] for i in kept_indices]
            for index, staff in enumerate(staves):
                staff.index = index
            piano_pairs = pair_piano_staves(staves) if cfg.piano_grand_staff else None
            align_piano_staff_extents(staves, piano_pairs)
            primary_staff_count = sum(id(staff) in primary_ids for staff in staves)
            supplemental_staff_count = len(staves) - primary_staff_count
            # A remaining detection can have been marked as a duplicate of
            # a staff just removed.  Restore and re-evaluate the ghosts on
            # the final staff set so genuine notes are not lost with it.
            for glyphs in glyph_sets:
                glyphs.detections.extend(glyphs.ghosts)
                glyphs.ghosts.clear()
            n_ghosts = detect_mod.drop_cross_staff_ghosts(glyph_sets, staves)

    # --- 3c. 系统级小节线统一 -------------------------------------------
    # 同一系统内的谱表（钢琴大谱表的高音+低音）是**同时**发声的，小节线也是
    # 贯穿两条谱表的同一条线，所以它们必须共享同一组小节边界。
    # 实测不统一时两条谱表的小节数经常不等（101910：5 对里只 1 对相等），
    # 从而使小节内音符数成倍错、tick 成倍膨胀。
    # 已知钢琴大谱表优先按谱表几何成对；其他版式仍靠小节线是否延伸进
    # 两条谱表的间隙分组（见 staff.group_systems），故分组前须先检测字形。
    systems = piano_pairs
    if systems is None:
        systems = staff_mod.group_systems(staves, glyph_sets=glyph_sets, binary=binary)
    clef_override_count = apply_piano_grand_staff_clefs(glyph_sets, systems) if cfg.piano_grand_staff else 0
    clef_changes = {
        i: tuple((x, "bass") for x in find_inline_bass_clefs(ink, staves[i]))
        for i, ink in enumerate(ink_list)
    }
    resolve_system_keys(glyph_sets, systems, override=cfg.key_override)
    for grp in systems:
        if len(grp) < 2:
            continue
        grp_staves = [staves[i] for i in grp]
        sys_ink = staff_mod.remove_all_staff_lines(binary, grp_staves,
                                                  run_lengths=run_lengths)
        detect_mod.unify_system_barlines(
            sys_ink, glyph_sets, grp,
            y_top=min(s.y_top_line for s in grp_staves),
            y_bottom=max(s.y_bottom_line for s in grp_staves),
            space=float(np.mean([s.space for s in grp_staves])),
            left_edge=float(np.median([s.x0 for s in staves])),
            right_edge=float(np.median([s.x1 for s in staves])),
        )
    key_changes = ({} if cfg.key_override is not None
                   else inline_key_changes(glyph_sets, systems))

    # --- 3d. 逐谱表重建 ------------------------------------------------
    # ★ 多谱表系统内的成员必须**保留空小节**（`keep_empty`）：
    #   同一系统内各谱表的小节边界是同一条节线，小节数必须相等。
    #   若某小节只在高音谱表有内容，逐谱表独立丢弃就会让两条谱表从此整体错位。
    multi_members = {i for grp in systems if len(grp) >= 2 for i in grp}
    boundary_policies = {
        index: paired_measure_boundaries(glyph_sets, group)
        for group in systems if len(group) >= 2 for index in group
    }
    for i, glyphs in enumerate(glyph_sets):
        rcfg = replace(cfg.reconstruct, key_fifths=glyphs.key_fifths or 0,
                       key_changes=key_changes.get(i, ()),
                       clef_changes=clef_changes[i])
        if i in multi_members and not rcfg.keep_empty:
            rcfg = replace(rcfg, keep_empty=True)
        policy = boundary_policies.get(i)
        if policy is not None:
            rcfg = replace(rcfg, include_leading_measure=policy[0],
                           include_trailing_measure=policy[1])
        sc, sub_issues = reconstruct(glyphs, rcfg, title=cfg.title)
        staff_scores.append(sc)
        # ★ 只转发"不属于某个具体小节"的通知（measure=0）。
        #   reconstruct 里逐小节的报错和下面 validate_measures(page) 是**同一批**，
        #   全转发会重复计数；measure=0 那批是它独有的（比如丢弃空小节的提示），
        #   不转发就会丢 —— 而"丢了什么"正是上一次踩坑的关键信息。
        notices.extend(i2 for i2 in sub_issues if getattr(i2, "measure", 1) == 0)

    suppressed_opening_segments = suppress_short_empty_openings(
        staff_scores, glyph_sets, systems)

    # --- 4. 串接成一首曲子 ------------------------------------------------
    # ★ 一页里的多个谱表是**连续的系统**（真实 A4 乐谱一页 8~12 行），
    # 必须按阅读顺序把小节首尾相接，而不是当成并行声部。
    # 用和"多页合并"完全相同的逻辑 —— 见 model.merge_scores。
    # 系统内**同时**合并、系统间**顺序**拼接（单谱表系统下等价于原来的 merge_scores）
    # 槽位：多成员系统按自上而下顺序；**单成员系统**（某条谱表没检出）按谱号定 ——
    # 低音谱表属于下方槽位。否则会把低音谱表放到上方，等于把左右手搞反。
    slot_of: list[list[int]] = []
    for grp in systems:
        if len(grp) >= 2:
            slot_of.append(list(range(len(grp))))
        else:
            clef = str(getattr(glyph_sets[grp[0]], "clef", "") or "").lower()
            slot_of.append([1 if clef.startswith("bass") else 0])
    page = merge_system_staves(
        [[staff_scores[i] for i in grp] for grp in systems],
        title=cfg.title, slot_of=slot_of)

    if cfg.meter_override is not None:
        apply_meter_override(page, cfg.meter_override)

    # --- 5. 整页节拍校验 --------------------------------------------------
    issues = validate_measures(page)
    issues.extend(notices)

    return PageResult(score=page, glyphs=glyph_sets, staff_scores=staff_scores,
                      staves=staves, systems=systems,
                      pre=pre, issues=issues, ink_per_staff=ink_list,
                      primary_staff_count=primary_staff_count,
                      supplemental_staff_count=supplemental_staff_count,
                      clef_override_count=clef_override_count,
                      inline_clef_count=sum(map(len, clef_changes.values())),
                      suppressed_opening_segments=suppressed_opening_segments,
                      inline_clef_x_by_staff=[
                          [x for x, _clef in clef_changes[i]]
                          for i in range(len(glyph_sets))])
