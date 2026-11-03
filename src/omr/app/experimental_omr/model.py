"""
数据模型 —— 整个项目的地基。

设计要点（见 docs/AC920乐谱识别实施方案.md §6.2）：
  * `voices` 层即使首版只有单声部也必须保留。事后再加要同时改检测后处理、
    符号关联、语义重建、MusicXML 导出、播放调度五个环节。
  * `bbox` + `confidence` 必须存在：前者驱动"点击修正"和"放音高亮跟随"，
    后者是"人工校正"闭环成立的前提。
  * 时间一律用整数 tick，不用分数。绕开连音时值表示的地狱。

坐标约定：全部使用**原图像素坐标系**（矫正后的页面坐标系），
          y 轴向下，原点左上。这样 bbox 可以直接用于 UI 叠加。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 每四分音符的 tick 数。480 是 MIDI 世界的通用值，能被 2/3/4/5/6/8 整除，
#: 足以表示三连音、附点、全音符等。
PPQ = 480

EventType = Literal["note", "rest"]
TieType = Literal["start", "stop", "continue", "none"]

#: 时值（以四分音符为单位）-> 标准名称
DURATION_NAMES = {
    4.0: "whole",
    3.0: "half_dotted",
    2.0: "half",
    1.5: "quarter_dotted",
    1.0: "quarter",
    0.75: "eighth_dotted",
    0.5: "eighth",
    0.25: "sixteenth",
    0.125: "thirty_second",
}

#: 名称 -> 时值（四分音符为单位）
DURATION_VALUES = {v: k for k, v in DURATION_NAMES.items()}


# ---------------------------------------------------------------------------
# 几何
# ---------------------------------------------------------------------------


@dataclass
class BBox:
    """轴对齐包围盒，原图像素坐标。用于 UI 叠加 / 点击修正 / 高亮跟随。"""

    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def to_list(self) -> list[float]:
        return [round(self.x, 2), round(self.y, 2), round(self.w, 2), round(self.h, 2)]

    @classmethod
    def from_list(cls, v: Iterable[float]) -> "BBox":
        x, y, w, h = list(v)[:4]
        return cls(float(x), float(y), float(w), float(h))


# ---------------------------------------------------------------------------
# 音乐事件
# ---------------------------------------------------------------------------


@dataclass
class Event:
    """一个音符或休止符。"""

    type: EventType
    onset_ticks: int
    duration_ticks: int
    #: MIDI 音高 0..127。休止符为 -1。
    pitch: int = -1
    #: 书写音名（未考虑调号的"谱面音名"），如 "C4" / "F#5"。休止符为 ""。
    written_name: str = ""
    #: 调号推导出的音名（实际发声音名，已应用调号但未应用临时记号），如 "F#5"。
    sounding_name: str = ""
    #: 临时记号（仅当该音带显式临时记号时非空）："sharp"|"flat"|"natural"|"double_sharp"|"double_flat"
    accidental: str | None = None
    #: 该 bbox 内是否发生过临时记号（用于人工校正时高亮）
    tie: TieType = "none"
    #: 三连八分音符组中的位置；None 表示普通时值。
    tuplet: Literal["start", "continue", "stop"] | None = None
    #: Phrase slur endpoints for notation export; separate from a sounding tie.
    slur: Literal["start", "stop"] | None = None
    #: 该音符是符干朝上还是朝下（首版单声部不用，为多声部预留）
    stem: str | None = None
    #: OMR 看到的局部符尾/连梁条数，供拍号纠错时打破等代价歧义。
    visual_tail_flags: int | None = None
    visual_tail_beams: int | None = None
    visual_stem_tip_y: float | None = None
    #: 是否属于和弦；和弦内所有音共享 onset 与 duration
    chord_id: int | None = None
    #: 识别置信度 0..1。人工校正 UI 用它高亮可疑项。
    confidence: float = 1.0
    bbox: BBox | None = None
    #: ★ 该事件所在的**页号**（1 起）。多页乐谱合并后，bbox 仍是页内局部坐标，
    #: 必须靠 page 才能定位到具体是哪一页 —— HDMI 放音高亮跟随要用它。
    page: int = 1
    #: 自由备注，便于调试
    note: str = ""

    # -- 便捷属性 ----------------------------------------------------------

    @property
    def is_note(self) -> bool:
        return self.type == "note"

    @property
    def offset_beats(self) -> float:
        """onset，以四分音符为单位。"""
        return self.onset_ticks / PPQ

    @property
    def duration_beats(self) -> float:
        return self.duration_ticks / PPQ

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.bbox is not None:
            d["bbox"] = self.bbox.to_list()
        else:
            d.pop("bbox", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        d = dict(d)
        bbox = d.pop("bbox", None)
        ev = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if bbox is not None:
            ev.bbox = BBox.from_list(bbox)
        return ev


# ---------------------------------------------------------------------------
# 层次结构： Score -> Part -> Staff -> Measure -> Voice -> Event
# ---------------------------------------------------------------------------


@dataclass
class Voice:
    """一个声部。首版每个谱表只放一个 voice，但这一层必须存在。"""

    id: int
    events: list[Event] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "events": [e.to_dict() for e in self.events]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Voice":
        return cls(id=d["id"], events=[Event.from_dict(e) for e in d.get("events", [])])


@dataclass
class Measure:
    number: int
    time_signature: tuple[int, int] = (4, 4)
    voices: list[Voice] = field(default_factory=list)
    #: 当前小节生效的调号；可表示同一页不同系统处的调号变化。
    key_fifths: int | None = None
    #: 谱号可在后续系统或小节变化；None 沿用当前谱表的谱号。
    clef: str | None = None
    #: 小节线的原图 bbox，便于定位
    bbox: BBox | None = None
    #: ★ 该小节来自第几页（1 起）。多页合并后小节号是连续的，
    #: 但页内坐标要靠它才能还原。
    page: int = 1

    @property
    def expected_ticks(self) -> int:
        """按拍号推出的小节总时值（tick）。用于节拍约束校验与自动纠错。"""
        num, den = self.time_signature
        return int(round(num * (4.0 / den) * PPQ))

    def actual_ticks(self) -> int:
        """本小节各声部中最长的时值总量。

        ★ 必须按"**最后一个事件的结束时间**"算，**不能把各事件时值直接相加**。
        踩过的坑：原来写的是 `sum(e.duration_ticks for e in v.events)`，
        而和弦的各音**共享 onset 与 duration**（见 `Event.chord_id`），
        直接相加会把和弦成员重复计数 —— 一个三音和弦被算成 3 倍时值。
        实测：刚加上和弦分组时，小节划分是对的、onset 也只前进一次，
        但总量反而从 352,080 涨到 474,960 tick（2.13 -> 2.04 的实际/期望比
        变成 1.70 -> 2.04），完全是因为这里在重复计数。
        没有和弦时两种算法**完全等价**（顺序事件的时值和 = 末事件的结束时间），
        所以这个改动对单声部用例是逐位不变的。
        """
        best = 0
        for v in self.voices:
            end = max((e.onset_ticks + e.duration_ticks for e in v.events), default=0)
            if end > best:
                best = end
        return best

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "number": self.number,
            "page": self.page,
            "time_signature": list(self.time_signature),
            "voices": [v.to_dict() for v in self.voices],
        }
        if self.bbox is not None:
            d["bbox"] = self.bbox.to_list()
        if self.key_fifths is not None:
            d["key_fifths"] = self.key_fifths
        if self.clef is not None:
            d["clef"] = self.clef
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Measure":
        m = cls(
            number=d["number"],
            time_signature=tuple(d.get("time_signature", (4, 4))),  # type: ignore[arg-type]
            voices=[Voice.from_dict(v) for v in d.get("voices", [])],
            key_fifths=(int(d["key_fifths"]) if d.get("key_fifths") is not None else None),
            clef=d.get("clef"),
            page=int(d.get("page", 1)),
        )
        if d.get("bbox"):
            m.bbox = BBox.from_list(d["bbox"])
        return m


@dataclass
class Staff:
    number: int
    measures: list[Measure] = field(default_factory=list)
    #: 五线谱检测结果：每一条谱线的 y 坐标（原图像素）
    line_ys: list[float] = field(default_factory=list)
    #: 谱线间距（staff space），单位像素。所有几何归一化的基准。
    space: float = 0.0
    clef: str = "treble"
    key_fifths: int = 0
    #: 本谱表在原图中的水平范围
    x0: float = 0.0
    x1: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "clef": self.clef,
            "key_fifths": self.key_fifths,
            "space": round(self.space, 3),
            "line_ys": [round(y, 2) for y in self.line_ys],
            "x0": round(self.x0, 2),
            "x1": round(self.x1, 2),
            "measures": [m.to_dict() for m in self.measures],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Staff":
        return cls(
            number=d["number"],
            measures=[Measure.from_dict(m) for m in d.get("measures", [])],
            line_ys=list(d.get("line_ys", [])),
            space=float(d.get("space", 0.0)),
            clef=d.get("clef", "treble"),
            key_fifths=int(d.get("key_fifths", 0)),
            x0=float(d.get("x0", 0.0)),
            x1=float(d.get("x1", 0.0)),
        )


@dataclass
class Part:
    name: str = "Melody"
    #: 移调半音数（多声部/移调乐器才用得上，先留着）
    transpose: int = 0
    staves: list[Staff] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "transpose": self.transpose,
            "staves": [s.to_dict() for s in self.staves],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Part":
        return cls(
            name=d.get("name", "Melody"),
            transpose=int(d.get("transpose", 0)),
            staves=[Staff.from_dict(s) for s in d.get("staves", [])],
        )


@dataclass
class Score:
    parts: list[Part] = field(default_factory=list)
    title: str = ""
    composer: str = ""
    #: 识别时所用的原图路径，便于回溯
    source_image: str = ""
    #: 整页的识别置信度（各事件置信度的加权平均）
    confidence: float = 1.0
    #: ★ 页表。多页乐谱合并后，事件的 bbox 是**页内局部坐标**，
    #: 要看图/高亮必须先查这张表拿到对应页的位图与尺寸。
    #: 每项形如 {"index": 1, "image": "pages/page_01.png", "w": 1700, "h": 420}
    pages: list[dict[str, Any]] = field(default_factory=list)

    # -- 页表便捷访问 ------------------------------------------------------

    def page_info(self, index: int) -> dict[str, Any] | None:
        for p in self.pages:
            if int(p.get("index", 0)) == index:
                return p
        return None

    def page_size(self, index: int) -> tuple[int, int] | None:
        p = self.page_info(index)
        if p is None:
            return None
        return int(p.get("w", 0)), int(p.get("h", 0))

    # -- 结构便捷访问 ------------------------------------------------------

    def iter_measures(self):
        for p in self.parts:
            for s in p.staves:
                for m in s.measures:
                    yield p, s, m

    def iter_events(self):
        for p in self.parts:
            for s in p.staves:
                for m in s.measures:
                    for v in m.voices:
                        for e in v.events:
                            yield p, s, m, v, e

    def note_count(self) -> int:
        return sum(1 for *_x, e in self.iter_events() if e.is_note)

    def mean_confidence(self) -> float:
        vals = [e.confidence for *_x, e in self.iter_events()]
        return sum(vals) / len(vals) if vals else 0.0

    # -- 序列化 ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": {
                "title": self.title,
                "composer": self.composer,
                "source_image": self.source_image,
                "confidence": round(self.confidence, 4),
                "pages": self.pages,
                "parts": [p.to_dict() for p in self.parts],
            }
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Score":
        d = d.get("score", d)
        return cls(
            parts=[Part.from_dict(p) for p in d.get("parts", [])],
            title=d.get("title", ""),
            composer=d.get("composer", ""),
            source_image=d.get("source_image", ""),
            confidence=float(d.get("confidence", 1.0)),
            pages=list(d.get("pages", [])),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "Score":
        return cls.from_dict(json.loads(s))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "Score":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())


# ---------------------------------------------------------------------------
# 校验：节拍约束 —— 最便宜有效的自动纠错手段
# ---------------------------------------------------------------------------


@dataclass
class MeasureIssue:
    part: str
    staff: int
    measure: int
    expected_ticks: int
    actual_ticks: int
    severity: str  # "error" | "warning" | "repaired"
    #: 自定义说明。非空时优先于自动生成的描述（纠错记录用）。
    message: str = ""

    def __str__(self) -> str:
        if self.message:
            return f"[{self.severity}] {self.message}"
        return (
            f"[{self.severity}] {self.part}/staff{self.staff} 第{self.measure}小节: "
            f"期望 {self.expected_ticks} tick, 实际 {self.actual_ticks} tick "
            f"(差 {self.actual_ticks - self.expected_ticks:+d})"
        )


def validate_measures(score: Score, tolerance_ticks: int = 0) -> list[MeasureIssue]:
    """按拍号检查每小节总时值。

    这是识别后处理里最有价值的一步：只要某一小节的时值总和对不上拍号，
    就说明某个音符的时值判错了（八分/十六分混淆是最常见的）。
    后续可据此做自动纠正，或把该小节标为"可疑"交给用户复核。
    """
    issues: list[MeasureIssue] = []
    for part, staff, m in score.iter_measures():
        exp, act = m.expected_ticks, m.actual_ticks()
        if abs(act - exp) > tolerance_ticks:
            issues.append(
                MeasureIssue(
                    part=part.name,
                    staff=staff.number,
                    measure=m.number,
                    expected_ticks=exp,
                    actual_ticks=act,
                    severity="error",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# 全局时间轴
# ---------------------------------------------------------------------------
#
# 事件的 `onset_ticks` 是**小节内相对值**（真实 OMR 里跨小节的绝对时间要先
# 确定拍号和小节划分才敢算）。导出 MIDI、播放、以及放音高亮跟随都需要
# "全局 tick"，所以这三处**必须共用同一份换算逻辑** —— 否则会出现
# "MIDI 和屏幕上高亮的位置对不上"这种极难排查的问题。
#
# 推进规则：用**实际累加值**（`actual_ticks`），而不是拍号理论值。
# 这样即使某个小节识别错了、时值总和对不上，也不会让后面全部错位。


def measure_spans(score: Score) -> list[tuple[int, int, Measure]]:
    """-> [(起始 tick, 结束 tick, 小节)]，按全局 tick 排序。"""
    spans: list[tuple[int, int, Measure]] = []
    for part in score.parts:
        for st in part.staves:
            base = 0
            for m in st.measures:
                length = m.actual_ticks() if m.actual_ticks() > 0 else m.expected_ticks
                spans.append((base, base + length, m))
                base += length
    spans.sort(key=lambda t: t[0])
    return spans


def iter_absolute_events(score: Score) -> list[tuple[int, Measure, Event]]:
    """-> [(全局起始 tick, 小节, 事件)]，按 tick 排序。

    这是播放、MIDI 导出、放音高亮三者的**唯一权威时间轴**。
    """
    out: list[tuple[int, Measure, Event]] = []
    for part in score.parts:
        for st in part.staves:
            base = 0
            for m in st.measures:
                for v in m.voices:
                    for ev in v.events:
                        out.append((base + ev.onset_ticks, m, ev))
                base += m.actual_ticks() if m.actual_ticks() > 0 else m.expected_ticks
    out.sort(key=lambda t: t[0])
    return out


def total_ticks(score: Score) -> int:
    """整曲总时长（tick）。"""
    spans = measure_spans(score)
    return max((e for _s, e, _m in spans), default=0)


# ---------------------------------------------------------------------------
# 小节串接（多系统 / 多页合并的共用逻辑）
# ---------------------------------------------------------------------------


def merge_system_staves(system_scores: list[list[Score]],
                        title: str = "",
                        slot_of: list[list[int]] | None = None) -> Score:
    """把一页内的各**系统**合并成"系统内**同时**、系统间**顺序**"的结构。

    Parameters
    ----------
    system_scores
        每个系统一个列表，列表内是该系统**自上而下**各谱表的识别结果
        （每个元素是 `reconstruct` 产出的、只含一个谱表的 Score）。
    slot_of
        与 `system_scores` 平行，给出每个成员应落在**哪个槽位**（0 = 最上方）。
        不给时按列表顺序 0,1,2,... 排。
        ★ 需要它的原因：某个系统**少检出一条谱表**时（实测 101852 的 System 1
        只检出了低音谱表），按顺序排会把**低音谱表放到上方槽位**，
        等于把左右手搞反。调用方知道谱号，可以按"低音谱表 = 下方槽位"来指定。

    与 `merge_scores` 的关系（重要）
    -------------------------------
    `merge_scores` 做的是"**首尾相接**"，它明确写了适用范围是**单声部乐谱**：
    钢琴大谱表的高音与低音是**同时**发声的，用首尾相接会得到"左右手一前一后弹"。
    本函数是它的推广：
        * 系统内各谱表 -> 同一个 Part 的不同 **Staff**，小节按**下标对齐**
          （即 `staves[k].measures[j]` 是同一个小节）
        * 系统与系统之间 -> 仍然**顺序**拼接
    **当每个系统都只有一条谱表时，本函数退化成 `merge_scores` 的行为**，
    所以单谱表用例（合成集、单行练习曲）的结果逐位不变。

    ★ 某槽位在该系统没有谱表时怎么办
    --------------------------------
    用**等量空小节占位**。这里的不变量是"跨谱表下标对齐"——
    宁可留下一个空小节（会被节拍校验当场标成 tick 不符，属于**诚实的失败信号**），
    也不能让后续所有小节整体错位（那会变成静默的错误）。
    """
    if not system_scores:
        return Score(title=title)

    if slot_of is None:
        slot_of = [list(range(len(grp))) for grp in system_scores]
    n_slots = max((max(sl) + 1 for sl in slot_of if sl), default=1)

    first_score = next((sc for grp in system_scores for sc in grp), None)
    if first_score is None:
        return Score(title=title)

    base_part = first_score.parts[0] if first_score.parts else Part(name="Piano")
    base_staff = base_part.staves[0] if base_part.staves else Staff(number=1, measures=[])

    def _measures(sc: Score) -> list[Measure]:
        return sc.parts[0].staves[0].measures if sc.parts else []

    merged_staves: list[Staff] = []
    for slot in range(n_slots):
        slot_score = next(
            (sc for grp, slots in zip(system_scores, slot_of)
             for sc, sl in zip(grp, slots) if sl == slot),
            None,
        )
        slot_staff = (slot_score.parts[0].staves[0]
                      if slot_score is not None and slot_score.parts
                      and slot_score.parts[0].staves else base_staff)
        all_measures: list[Measure] = []
        num = 1
        for grp, slots in zip(system_scores, slot_of):
            # 该系统的小节数：以落在本槽位的成员为准；本槽位缺员时用系统内
            # 其余成员的小节数占位（keep_empty 已保证同系统成员小节数相同）
            ref = [_measures(sc) for sc, sl in zip(grp, slots) if sl == slot]
            if ref:
                src = ref[0]
            else:
                other = next((_measures(sc) for sc in grp if _measures(sc)), [])
                ts = other[0].time_signature if other else (4, 4)
                src = [Measure(number=1, time_signature=ts,
                               voices=[Voice(id=1, events=[])]) for _ in other]
            for m in src:
                m.number = num
                num += 1
                all_measures.append(m)
        merged_staves.append(Staff(
            number=slot + 1,
            measures=all_measures,
            line_ys=[],                   # 谱线是页内信息，合并后无意义
            space=slot_staff.space,
            clef=slot_staff.clef,
            key_fifths=slot_staff.key_fifths,
            x0=0.0, x1=0.0,
        ))

    merged = Score(parts=[Part(name=base_part.name, transpose=base_part.transpose,
                               staves=merged_staves)],
                   title=title or first_score.title,
                   composer=first_score.composer)

    seen: set[int] = set()
    for grp in system_scores:
        for sc in grp:
            for p in sc.pages:
                idx = int(p.get("index", 0))
                if idx and idx not in seen:
                    seen.add(idx)
                    merged.pages.append(p)
    merged.pages.sort(key=lambda p: int(p.get("index", 0)))
    merged.confidence = merged.mean_confidence()
    return merged


def merge_scores(scores: list[Score], title: str = "") -> Score:
    """把多段识别结果**首尾相接**成一首曲子。

    两个地方都要用它，而且语义完全一样，所以放在这里：
      * 一页里有多个谱表（真实 A4 乐谱一页 8~12 行谱表）—— 它们是连续的系统
      * 多页乐谱（第 2 页接着第 1 页）

    ★ 关键：**谱表是"接着来"的，不是"同时来"的。**
    踩过的坑：一开始把每个检测到的谱表当成一个独立的 Staff 放进同一个 Part，
    结果成了一首曲子里 10 个并行声部 —— 时间轴上它们全部从 tick 0 开始，
    播放和显示彻底错乱。对于单声部乐谱，正确做法是把小节按
    **从上到下、从左到右的阅读顺序**串起来，小节号连续重编。

    事件的 `page` 与 `bbox` 原样保留（bbox 是页内局部坐标）。
    """
    if not scores:
        return Score(title=title)

    base = scores[0]
    merged_parts: list[Part] = []
    for pi, base_part in enumerate(base.parts):
        merged_staves: list[Staff] = []
        for si, base_staff in enumerate(base_part.staves):
            all_measures: list[Measure] = []
            num = 1
            for sc in scores:
                if pi >= len(sc.parts) or si >= len(sc.parts[pi].staves):
                    continue
                for m in sc.parts[pi].staves[si].measures:
                    m.number = num
                    num += 1
                    all_measures.append(m)
            merged_staves.append(Staff(
                number=base_staff.number,
                measures=all_measures,
                line_ys=[],                   # 谱线是页内信息，合并后无意义
                space=base_staff.space,
                clef=base_staff.clef,
                key_fifths=base_staff.key_fifths,
                x0=0.0, x1=0.0,
            ))
        merged_parts.append(Part(name=base_part.name,
                                 transpose=base_part.transpose,
                                 staves=merged_staves))

    merged = Score(parts=merged_parts, title=title or base.title,
                   composer=base.composer)

    # 页表合并（去重、按页号排序）
    seen: set[int] = set()
    for sc in scores:
        for p in sc.pages:
            idx = int(p.get("index", 0))
            if idx and idx not in seen:
                seen.add(idx)
                merged.pages.append(p)
    merged.pages.sort(key=lambda p: int(p.get("index", 0)))
    merged.confidence = merged.mean_confidence()
    return merged
