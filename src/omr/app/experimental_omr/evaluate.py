"""
精度评测 —— P1 的核心产出之一：**把"准确率"变成一个可以随时跑出来的数字**。

没有这个模块，OMR 项目就会退化成"看起来还行"的主观判断。
有了它，每次改算法/改参数都能立刻知道是变好还是变坏。

指标设计原则
------------
**业务指标是"播出来对不对"，不是 mAP。** 所以指标分三层：
  * 音符级：音高对不对、时值对不对、两者都对（最重要）
  * 结构级：小节数、拍号、时值总和是否合规
  * 序列级：编辑距离（SER）—— 对漏检/多检最敏感

对齐方式
--------
按 **(小节号, 小节内 onset)** 对齐 ground truth 与识别结果。
这比"按出现顺序对齐"更严格也更公平：时间对不上就是错，不给你蒙对的机会。
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

from . import theory
from .model import Event, Score


# ---------------------------------------------------------------------------
# 事件抽取
# ---------------------------------------------------------------------------


def flatten(score: Score) -> dict[tuple[int, int], list[Event]]:
    """-> {(小节号, 小节内onset): [事件...]}。

    同一个 key 下可能有多个事件（和弦），所以值是列表。
    """
    out: dict[tuple[int, int], list[Event]] = {}
    for p in score.parts:
        for s in p.staves:
            for m in s.measures:
                for v in m.voices:
                    for ev in v.events:
                        out.setdefault((m.number, ev.onset_ticks), []).append(ev)
    return out


def sequence_tokens(score: Score) -> list[tuple]:
    """把乐谱压成符号序列，用于算编辑距离。

    token = (音高 或 -1, 时值tick)。顺序 = (小节号, onset)。和弦会被展开成多个 token。
    """
    flat = flatten(score)
    toks: list[tuple] = []
    for key in sorted(flat.keys()):
        evs = sorted(flat[key], key=lambda e: (e.type, e.pitch))
        for ev in evs:
            toks.append((ev.pitch if ev.type == "note" else -1, ev.duration_ticks))
    return toks


def levenshtein(a: list, b: list) -> int:
    """标准编辑距离（插入/删除/替换代价均为 1）。"""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if x == y else 1))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """评测结果。所有比率都是 0..1，越大越好。"""

    n_truth: int = 0
    n_rec: int = 0
    n_matched: int = 0
    n_pitch_ok: int = 0
    n_duration_ok: int = 0
    n_both_ok: int = 0
    n_missed: int = 0
    n_extra: int = 0
    n_rest_truth: int = 0
    n_rest_ok: int = 0

    edit_distance: int = 0
    n_measures_truth: int = 0
    n_measures_rec: int = 0
    ts_truth: tuple[int, int] | None = None
    ts_rec: tuple[int, int] | None = None
    measure_ok: bool = False

    #: 每条差异的说明，便于定位
    diffs: list[str] = dc_field(default_factory=list)

    # -- 派生指标 ---------------------------------------------------------

    @property
    def note_accuracy(self) -> float:
        """音符级准确率：音高 **且** 时值都对，占 ground truth 的比例。

        这是本项目的**头号指标** —— 它直接对应"播出来对不对"。
        """
        return self.n_both_ok / self.n_truth if self.n_truth else 0.0

    @property
    def pitch_accuracy(self) -> float:
        return self.n_pitch_ok / self.n_truth if self.n_truth else 0.0

    @property
    def duration_accuracy(self) -> float:
        return self.n_duration_ok / self.n_truth if self.n_truth else 0.0

    @property
    def recall(self) -> float:
        return self.n_matched / self.n_truth if self.n_truth else 0.0

    @property
    def precision(self) -> float:
        return self.n_matched / self.n_rec if self.n_rec else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def ser(self) -> float:
        """序列编辑距离率（Symbol Error Rate）。对漏检/多检最敏感。"""
        return self.edit_distance / self.n_truth if self.n_truth else 0.0

    # -- ★ 与"分小节"无关的指标 -------------------------------------------
    #
    #  为什么必须单独有这一组（这是踩坑之后加的）：
    #  `flatten()` 是按 **(小节号, 小节内 tick 偏移)** 给事件编 key 的，
    #  所以 `note_accuracy` / `pitch_accuracy` / `ser` 这些都**隐含依赖"
    #  小节切分是对的"**。小节一旦切错（多一个空小节、少一条节线），
    #  从错位处起所有音符的 key 整体平移，这些指标一起崩到接近 0 ——
    #  **而音符可能一个都没读错**。
    #
    #  实测过一次：抹掉纸外背景的一张透视页，
    #      序列编辑距离 = 0        （24 个音符音高+时值全对、顺序也对）
    #      音符级准确率 = 4.2%     （同一份结果）
    #      小节数 6/12             （多出来 6 个空小节）
    #  当时因为只看准确率，把一个"分小节 bug"误读成了"识别鲁棒性差"。
    #
    #  `note_sequence_accuracy` 只看**音符序列本身**（用编辑距离算），
    #  完全不关心它们被分到哪个小节。于是两个指标一起看就能立刻分层：
    #
    #      两个都高            → 真的都对
    #      序列高、准确率低    → **音符没错，是分小节错了**（查节线/小节切分）
    #      两个都低            → 音符本身读错了（查检测/重建）
    #
    #  这是本项目最值得先看的一对数字，`summary()` 里就并排打印。

    @property
    def note_sequence_accuracy(self) -> float:
        """与分小节无关的音符序列准确率 = 1 - SER。

        只比"音符序列"（音高+时值的有序串），不比它们落在哪个小节。
        """
        if not self.n_truth:
            return 0.0
        return max(0.0, 1.0 - self.edit_distance / self.n_truth)

    @property
    def segmentation_ok(self) -> bool:
        """小节数是否与真值一致 —— 分小节对不对的最直接证据。"""
        return (self.n_measures_truth > 0
                and self.n_measures_rec == self.n_measures_truth)

    @property
    def diagnosis(self) -> str:
        """一句话说明"崩的是哪一层"。给报告和日志用。"""
        seq, acc = self.note_sequence_accuracy, self.note_accuracy
        if not self.n_truth:
            return "无 ground truth"
        if seq >= 0.95 and acc >= 0.95:
            return "正常" if self.segmentation_ok else "正常（但小节数不一致）"
        if seq >= 0.95 and acc < 0.95:
            return (f"★ 音符读对了，是**分小节**错了"
                    f"（序列 {seq:.0%} vs 准确率 {acc:.0%}，"
                    f"小节数 {self.n_measures_truth}/{self.n_measures_rec}）")
        if seq < 0.95 <= acc:
            return f"★ 异常：序列 {seq:.0%} 却准确率 {acc:.0%}（不该出现，请查对齐逻辑）"
        return f"音符本身读错（序列 {seq:.0%}）"

    def summary(self) -> str:
        L = []
        a = L.append
        a("=" * 62)
        a("  识别精度报告")
        a("=" * 62)
        a(f"  ground truth 音符数 : {self.n_truth}   (其中休止符 {self.n_rest_truth})")
        a(f"  识别出的事件数      : {self.n_rec}")
        a(f"  对齐上             : {self.n_matched}   漏检 {self.n_missed}   多检 {self.n_extra}")
        a("  " + "-" * 58)
        a(f"  ★ 音符级准确率      : {self.note_accuracy:6.1%}   ({self.n_both_ok}/{self.n_truth})")
        a(f"    音高准确率        : {self.pitch_accuracy:6.1%}   ({self.n_pitch_ok}/{self.n_truth})")
        a(f"    时值准确率        : {self.duration_accuracy:6.1%}   ({self.n_duration_ok}/{self.n_truth})")
        a(f"    Precision/Recall  : {self.precision:5.1%} / {self.recall:5.1%}   F1={self.f1:5.1%}")
        a(f"    序列编辑距离 SER  : {self.ser:6.1%}   (ED={self.edit_distance})")
        a("  " + "-" * 58)
        # ★ 这一对数字要**并排**看：准确率隐含依赖"分小节是对的"，
        #   而序列准确率不依赖。两者差距大就说明崩的是分小节那一层。
        a(f"  ★ 音符序列准确率    : {self.note_sequence_accuracy:6.1%}"
          f"   ← 与分小节**无关**，只比音符序列本身")
        a(f"    分小节          : {'一致' if self.segmentation_ok else '★不一致'}"
          f"   ({self.n_measures_truth}/{self.n_measures_rec} 小节)")
        a(f"    诊断            : {self.diagnosis}")
        a("  " + "-" * 58)
        a(f"  小节数 truth/rec    : {self.n_measures_truth} / {self.n_measures_rec}"
          f"   {'一致' if self.n_measures_truth == self.n_measures_rec else '★不一致'}")
        a(f"  拍号 truth/rec      : {self.ts_truth} / {self.ts_rec}")
        for d in self.diffs[:20]:
            a("    " + d)
        if len(self.diffs) > 20:
            a(f"    ... 另有 {len(self.diffs) - 20} 条差异未显示")
        a("=" * 62)
        return "\n".join(L)

    def to_dict(self) -> dict:
        return {
            "n_truth": self.n_truth,
            "n_rec": self.n_rec,
            "n_matched": self.n_matched,
            "n_missed": self.n_missed,
            "n_extra": self.n_extra,
            "note_accuracy": round(self.note_accuracy, 4),
            "note_sequence_accuracy": round(self.note_sequence_accuracy, 4),
            "segmentation_ok": self.segmentation_ok,
            "diagnosis": self.diagnosis,
            "pitch_accuracy": round(self.pitch_accuracy, 4),
            "duration_accuracy": round(self.duration_accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "ser": round(self.ser, 4),
            "edit_distance": self.edit_distance,
            "n_measures_truth": self.n_measures_truth,
            "n_measures_rec": self.n_measures_rec,
            "measure_ok": self.measure_ok,
        }


# ---------------------------------------------------------------------------
# 评测主函数
# ---------------------------------------------------------------------------


def _desc(ev: Event) -> str:
    if ev.type == "rest":
        return f"rest({ev.duration_ticks})"
    return f"{ev.sounding_name or theory.midi_to_pitch_name(ev.pitch)}({ev.duration_ticks})"


def evaluate(truth: Score, rec: Score, collect_diffs: bool = True) -> EvalReport:
    """把识别结果与 ground truth 比对。"""
    rep = EvalReport()
    ft = flatten(truth)
    fr = flatten(rec)

    rep.n_truth = sum(len(v) for v in ft.values())
    rep.n_rec = sum(len(v) for v in fr.values())
    rep.n_rest_truth = sum(1 for v in ft.values() for e in v if e.type == "rest")

    # --- 逐 key 对齐 -----------------------------------------------------
    for key in sorted(set(ft) | set(fr)):
        t_evs = sorted(ft.get(key, []), key=lambda e: (e.type, e.pitch))
        r_evs = sorted(fr.get(key, []), key=lambda e: (e.type, e.pitch))
        n = min(len(t_evs), len(r_evs))
        for i in range(n):
            t, r = t_evs[i], r_evs[i]
            rep.n_matched += 1
            pitch_ok = (t.pitch == r.pitch) and (t.type == r.type)
            dur_ok = t.duration_ticks == r.duration_ticks
            if pitch_ok:
                rep.n_pitch_ok += 1
            if dur_ok:
                rep.n_duration_ok += 1
            if pitch_ok and dur_ok:
                rep.n_both_ok += 1
                if t.type == "rest":
                    rep.n_rest_ok += 1
            if collect_diffs and not (pitch_ok and dur_ok):
                bad = []
                if not pitch_ok:
                    bad.append("音高")
                if not dur_ok:
                    bad.append("时值")
                rep.diffs.append(
                    f"小节{key[0]} t={key[1]:>4}: {_desc(t):<16} -> {_desc(r):<16}"
                    f"  ✗{'+'.join(bad)}"
                )
        if len(t_evs) > n:
            rep.n_missed += len(t_evs) - n
            if collect_diffs:
                for t in t_evs[n:]:
                    rep.diffs.append(f"小节{key[0]} t={key[1]:>4}: {_desc(t):<16} -> 漏检")
        if len(r_evs) > n:
            rep.n_extra += len(r_evs) - n
            if collect_diffs:
                for r in r_evs[n:]:
                    rep.diffs.append(f"小节{key[0]} t={key[1]:>4}: 多检 {_desc(r)}")

    # --- 序列编辑距离 ----------------------------------------------------
    rep.edit_distance = levenshtein(sequence_tokens(truth), sequence_tokens(rec))

    # --- 结构级 ----------------------------------------------------------
    rep.n_measures_truth = sum(1 for _ in truth.iter_measures())
    rep.n_measures_rec = sum(1 for _ in rec.iter_measures())
    rep.ts_truth = next((m.time_signature for _p, _s, m in truth.iter_measures()), None)
    rep.ts_rec = next((m.time_signature for _p, _s, m in rec.iter_measures()), None)

    # 小节时值是否全部合规（识别结果）
    from .model import validate_measures
    rep.measure_ok = (len(validate_measures(rec)) == 0)
    return rep


# ---------------------------------------------------------------------------
# 门限检查 —— 可以直接拿去做 CI
# ---------------------------------------------------------------------------


GATES = {
    "synthetic": {"note_accuracy": 0.95, "ser": 0.05,
                  "note_sequence_accuracy": 0.95, "segmentation": True},
    "photo": {"note_accuracy": 0.85, "ser": 0.15,
              "note_sequence_accuracy": 0.85, "segmentation": False},
}


def check_gates(rep: EvalReport, profile: str = "synthetic") -> tuple[bool, list[str]]:
    """按门限判断是否达标。返回 (是否通过, 失败原因列表)。

    ★ 除准确率之外还单独门限**序列准确率**和**分小节**：
    准确率低但序列高时，失败原因会直说是"分小节"而不是含糊的"准确率不够"。
    没有这一条，一个分小节 bug 报出来的就是"识别精度不达标"，
    排查方向会直接跑偏（真的跑偏过一次）。
    """
    g = GATES.get(profile, GATES["synthetic"])
    fails: list[str] = []
    if rep.note_accuracy < g["note_accuracy"]:
        fails.append(f"音符级准确率 {rep.note_accuracy:.1%} < 门限 {g['note_accuracy']:.0%}")
    if rep.ser > g["ser"]:
        fails.append(f"SER {rep.ser:.1%} > 门限 {g['ser']:.0%}")
    seq_gate = g.get("note_sequence_accuracy")
    if seq_gate is not None and rep.note_sequence_accuracy < seq_gate:
        fails.append(f"音符序列准确率 {rep.note_sequence_accuracy:.1%} "
                     f"< 门限 {seq_gate:.0%}")
    if g.get("segmentation") and rep.n_truth and not rep.segmentation_ok:
        fails.append(f"分小节不一致：真值 {rep.n_measures_truth} 小节，"
                     f"识别 {rep.n_measures_rec} 小节")
    # ★ 最后补一句"崩在哪一层"，免得又去猜
    if fails and rep.diagnosis not in ("正常",):
        fails.append(f"定位：{rep.diagnosis}")
    return (not fails), fails
