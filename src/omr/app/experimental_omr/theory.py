"""
乐理核心：谱面几何 <-> 音高 <-> MIDI 音高。

这是单声部识别里"必须一次做对"的部分（见方案文档 §6.4）。弄错了播放就会错。

核心抽象
--------
用 **diatonic number**（全音阶序号）统一表示"五线谱上的位置"：
    D = octave * 7 + letter_index        letter_index: C=0 D=1 E=2 F=3 G=4 A=5 B=6
它的好处是每在谱上走一格（线->间->线）D 就 +1，与谱号、加线、上下加线完全解耦。

谱面 y 坐标 -> 谱位 step p：
    p = (y_bottom_line - y) / (space / 2)      向上为正
每半个 space 是一个全音阶步。
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import PPQ

# ---------------------------------------------------------------------------
# 音名与半音
# ---------------------------------------------------------------------------

LETTERS = "CDEFGAB"
LETTER_INDEX = {c: i for i, c in enumerate(LETTERS)}
#: 自然音级 -> 相对 C 的半音数
LETTER_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

#: 临时记号 -> 半音偏移
ACCIDENTAL_OFFSET = {
    "double_flat": -2,
    "flat": -1,
    "natural": 0,
    "sharp": 1,
    "double_sharp": 2,
}
#: 半音偏移 -> 标准名（用于反查）
OFFSET_TO_ACCIDENTAL = {v: k for k, v in ACCIDENTAL_OFFSET.items()}

#: 调号 fifths -> 升号出现顺序 / 降号出现顺序
SHARP_ORDER = "FCGDAEB"
FLAT_ORDER = "BEADGCF"


# ---------------------------------------------------------------------------
# 谱号：五线谱最下面那条线（line 0）对应的全音阶序号
# ---------------------------------------------------------------------------

def diatonic_number(letter: str, octave: int) -> int:
    """(音名, 八度) -> 全音阶序号。C4 = 4*7+0 = 28。"""
    return octave * 7 + LETTER_INDEX[letter.upper()]


def number_to_pitch(d: int) -> tuple[str, int]:
    """全音阶序号 -> (音名, 八度)。"""
    return LETTERS[d % 7], d // 7


#: 各谱号下，最下面一条谱线的全音阶序号
CLEF_BOTTOM_LINE: dict[str, int] = {
    # 高音谱号：五条线自下而上 E4 G4 B4 D5 F5
    "treble": diatonic_number("E", 4),   # 30
    # 低音谱号：G2 B2 D3 F3 A3
    "bass": diatonic_number("G", 2),     # 18
    # 中音谱号（C 谱号在第 3 线）：F3 A3 C4 E4 G4
    "alto": diatonic_number("F", 3),     # 24
    # 次中音谱号（C 谱号在第 4 线）：D3 F3 A3 C4 E4
    "tenor": diatonic_number("D", 3),    # 22
}

#: 首选用于显示的谱号名
CLEF_ALIASES = {
    "g": "treble", "gclef": "treble", "violin": "treble",
    "f": "bass", "fclef": "bass",
    "c3": "alto", "c4": "tenor",
}


def step_to_pitch(step: int, clef: str = "treble") -> tuple[str, int]:
    """谱位 step（0 = 最下面那条线，向上为正，可负=下加线） -> (音名, 八度)。"""
    clef = CLEF_ALIASES.get(clef.lower(), clef.lower())
    if clef not in CLEF_BOTTOM_LINE:
        raise ValueError(f"未知谱号: {clef}")
    return number_to_pitch(CLEF_BOTTOM_LINE[clef] + step)


def pitch_to_step(letter: str, octave: int, clef: str = "treble") -> int:
    """(音名, 八度) -> 谱位 step。step_to_pitch 的逆。"""
    clef = CLEF_ALIASES.get(clef.lower(), clef.lower())
    if clef not in CLEF_BOTTOM_LINE:
        raise ValueError(f"未知谱号: {clef}")
    return diatonic_number(letter, octave) - CLEF_BOTTOM_LINE[clef]


def parse_pitch_name(name: str) -> tuple[str, int, int]:
    """解析 "F#4" / "Bb3" / "C4" -> (音名, 八度, 半音偏移)。

    这是本项目里唯一允许出现的音名字符串解析入口，方便以后统一替换。
    """
    s = name.strip()
    if not s:
        raise ValueError("空的音名")
    letter = s[0].upper()
    if letter not in LETTER_INDEX:
        raise ValueError(f"非法音名: {name}")
    i = 1
    suffix = ""
    while i < len(s) and s[i] in "#b♯♭":
        suffix += "#" if s[i] in "#♯" else "b"
        i += 1
    offset_map = {"": 0, "#": 1, "##": 2, "b": -1, "bb": -2}
    if suffix not in offset_map:
        raise ValueError(f"非法临时记号: {suffix!r} in {name!r}")
    octave = int(s[i:])
    return letter, octave, offset_map[suffix]


def y_to_step(y: float, y_bottom_line: float, space: float) -> float:
    """谱面 y 坐标 -> 谱位（浮点，未取整）。向上为正。"""
    if space <= 0:
        raise ValueError("staff space 必须为正")
    return (y_bottom_line - y) / (space / 2.0)


def step_to_y(step: float, y_bottom_line: float, space: float) -> float:
    """谱位 -> 谱面 y 坐标。step_to_y 的逆。"""
    return y_bottom_line - step * (space / 2.0)


# ---------------------------------------------------------------------------
# 调号与临时记号
# ---------------------------------------------------------------------------

def key_signature_map(fifths: int) -> dict[str, int]:
    """调号（fifths, 正=升号数, 负=降号数） -> {音名: 半音偏移}。

    注意：调号作用于**所有八度**上的该音名。
    例如 G 大调 (fifths=1) 下，任何 F 都要升。
    """
    m: dict[str, int] = {}
    if fifths > 0:
        for c in SHARP_ORDER[:fifths]:
            m[c] = 1
    elif fifths < 0:
        for c in FLAT_ORDER[: -fifths]:
            m[c] = -1
    return m


def resolve_accidental(
    letter: str,
    explicit: str | None,
    key_fifths: int = 0,
    measure_accidentals: dict[str | tuple[str, int], int] | None = None,
    octave: int | None = None,
) -> int:
    """决定一个音符最终应用的半音偏移。

    优先级：显式临时记号 > 本小节内已出现的同音名、同八度临时记号
            > 调号 > 0

    Parameters
    ----------
    measure_accidentals
        本小节内已经生效的临时记号表 {(音名, 八度): 偏移}。调用方负责在小节边界清空，
        并在遇到显式临时记号时写入（这就是"跨小节失效"规则的实现位置）。
        未提供 octave 时兼容早期仅按音名建立的调用。
    """
    if explicit:
        return ACCIDENTAL_OFFSET[explicit]
    if measure_accidentals:
        key: str | tuple[str, int] = (letter, octave) if octave is not None else letter
        if key in measure_accidentals:
            return measure_accidentals[key]
    return key_signature_map(key_fifths).get(letter, 0)


def midi_pitch(letter: str, octave: int, offset: int) -> int:
    """(音名, 八度, 半音偏移) -> MIDI 音高。C4 = 60。"""
    return 12 * (octave + 1) + LETTER_SEMITONE[letter.upper()] + offset


def pitch_name(letter: str, octave: int, offset: int) -> str:
    """-> 形如 "F#4" / "Bb3" / "C4" 的字符串。"""
    suffix = ""
    if offset == 1:
        suffix = "#"
    elif offset == -1:
        suffix = "b"
    elif offset == 2:
        suffix = "##"
    elif offset == -2:
        suffix = "bb"
    return f"{letter.upper()}{suffix}{octave}"


def midi_to_pitch_name(midi: int) -> str:
    """调试用：MIDI 音高 -> 通用音名（一律用升号拼写）。"""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}"


# ---------------------------------------------------------------------------
# 时值
# ---------------------------------------------------------------------------

#: 名称 -> tick 数
DURATION_TICKS: dict[str, int] = {
    "whole": 4 * PPQ,
    "half": 2 * PPQ,
    "quarter": 1 * PPQ,
    "eighth": PPQ // 2,
    "triplet_eighth": PPQ // 3,
    "sixteenth": PPQ // 4,
    "thirty_second": PPQ // 8,
}


def ticks(name: str) -> int:
    """时值名称 -> tick。支持附点前缀：'dotted_quarter'。"""
    if name.startswith("dotted_"):
        return int(DURATION_TICKS[name[len("dotted_"):]] * 1.5)
    return DURATION_TICKS[name]


def ticks_to_name(t: int) -> str:
    """tick -> 最接近的时值名称（含附点）。用于导出 MusicXML。"""
    if t == PPQ // 3:
        return "triplet_eighth"
    best, best_err = "quarter", 1 << 30
    for base in DURATION_TICKS:
        if base == "triplet_eighth":
            continue
        for mult, prefix in ((1.0, ""), (1.5, "dotted_")):
            v = DURATION_TICKS[base] * mult
            err = abs(v - t)
            if err < best_err - 1e-9:
                best, best_err = f"{prefix}{base}", err
    return best


def parse_time_signature(s: str | tuple[int, int]) -> tuple[int, int]:
    """'4/4' 或 (4, 4) -> (4, 4)。"""
    if isinstance(s, tuple):
        return s
    num, den = s.split("/")
    return int(num), int(den)


def measure_ticks(time_signature: tuple[int, int]) -> int:
    """按拍号计算一小节应有的总 tick。"""
    num, den = time_signature
    return int(round(num * (4.0 / den) * PPQ))


# ---------------------------------------------------------------------------
# 音高拼写
# ---------------------------------------------------------------------------


@dataclass
class Spelling:
    """一个音符的完整拼写信息。"""

    letter: str
    octave: int
    offset: int  # 半音偏移

    @property
    def midi(self) -> int:
        return midi_pitch(self.letter, self.octave, self.offset)

    @property
    def name(self) -> str:
        return pitch_name(self.letter, self.octave, self.offset)

    @property
    def diatonic(self) -> int:
        return diatonic_number(self.letter, self.octave)


def spell_from_step(step: int, clef: str, offset: int) -> Spelling:
    """谱位 + 谱号 + 半音偏移 -> 完整拼写。"""
    letter, octave = step_to_pitch(step, clef)
    return Spelling(letter, octave, offset)
