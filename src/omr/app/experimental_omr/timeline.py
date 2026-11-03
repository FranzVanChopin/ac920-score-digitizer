"""
全局时间轴 —— 播放、音频合成、放音高亮跟随**共用的唯一权威**。

为什么必须独立成一个模块
------------------------
"当前演奏到哪了"这个问题有三个消费者：
  1. 音频合成（哪个音符该在什么采样位置发声）
  2. 播放控制（播放/暂停/跳转）
  3. 屏幕高亮跟随（此刻该点亮哪一页的哪些音符）

如果它们各自算一遍 tick->时间的换算，一旦有一处不一致，症状就是
"声音和屏幕上的高亮对不上" —— 这类 bug 极难排查，因为两边单独看都是对的。
所以换算只在一处发生，其余全部从这里取。

两个关键约定
------------
* 事件的 bbox 是**页内像素坐标**，`page` 字段指出在第几页。时间轴负责把
  tick 映射到"第几页"，屏幕那边再负责把页内坐标缩放到屏幕坐标。
* 相邻音的释放音尾会重叠（合成器要加 release），但时间轴上音符的区间是
  **[start, end) 半开区间**，所以"某一刻有哪些音在响"永远只有唯一答案。
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import PPQ, Event, Measure, Score, iter_absolute_events, measure_spans

#: 事件索引的桶大小（tick）。一个四分音符一桶足够 —— 音乐里同一个时刻
#: 在响的音数量很少，桶小一点查询更快。
BUCKET_TICKS = PPQ


@dataclass
class TimedEvent:
    """一个带全局时间的音符事件。"""

    start_tick: int
    end_tick: int
    start_sample: int
    end_sample: int
    event: Event
    measure_number: int
    page: int

    @property
    def duration_ticks(self) -> int:
        return self.end_tick - self.start_tick

    @property
    def pitch(self) -> int:
        return self.event.pitch

    def __str__(self) -> str:
        return (f"{self.event.sounding_name or '?'}({self.pitch}) "
                f"t={self.start_tick}..{self.end_tick} "
                f"smp={self.start_sample}..{self.end_sample} "
                f"小节{self.measure_number} 第{self.page}页")


class Timeline:
    """把乐谱摊平成一条全局时间轴。

    Parameters
    ----------
    tempo_bpm
        四分音符的速度。OMR 识别不出速度（乐谱上通常只有一个意大利语术语），
        所以这里用固定值，后续可以做成可调（甚至从术语猜）。
    """

    def __init__(self, score: Score, tempo_bpm: float = 100.0,
                 sample_rate: int = 22050):
        self.score = score
        self.tempo_bpm = float(tempo_bpm)
        self.sample_rate = int(sample_rate)

        # 小节区间（全局 tick）
        self.spans: list[tuple[int, int, Measure]] = measure_spans(score)
        self.total_ticks = max((e for _s, e, _m in self.spans), default=0)
        self.total_seconds = self.tick_to_seconds(self.total_ticks)
        self.total_samples = self.tick_to_sample(self.total_ticks)

        # 音符事件（休止符不进时间轴 —— 它们不发声，也没有可高亮的符头）
        self.events: list[TimedEvent] = []
        for tick, m, ev in iter_absolute_events(score):
            if ev.type != "note" or ev.pitch < 0:
                continue
            dur = max(1, ev.duration_ticks)
            self.events.append(TimedEvent(
                start_tick=tick,
                end_tick=tick + dur,
                start_sample=self.tick_to_sample(tick),
                end_sample=self.tick_to_sample(tick + dur),
                event=ev,
                measure_number=m.number,
                page=int(m.page),
            ))

        # 桶索引：按起始 tick 分桶，并把跨越多个桶的长音放进每个覆盖到的桶，
        # 这样 events_at() 是 O(桶内事件数) 而不是 O(全曲事件数) ——
        # 逐帧查询（60fps）时这个差别是决定性的。
        self._buckets: dict[int, list[TimedEvent]] = {}
        for te in self.events:
            b0 = te.start_tick // BUCKET_TICKS
            b1 = (te.end_tick - 1) // BUCKET_TICKS
            for b in range(b0, b1 + 1):
                self._buckets.setdefault(b, []).append(te)

        # 页序（供应屏幕滚动使用）
        self.pages: list[int] = sorted({int(m.page) for _s, _e, m in self.spans}) \
            or ([1] if self.spans else [])

    # -- 换算 --------------------------------------------------------------

    def tick_to_seconds(self, tick: float) -> float:
        return tick / PPQ * 60.0 / self.tempo_bpm

    def tick_to_sample(self, tick: float) -> int:
        return int(round(self.tick_to_seconds(tick) * self.sample_rate))

    def sample_to_tick(self, sample: float) -> int:
        return int(round(sample / self.sample_rate * self.tempo_bpm / 60.0 * PPQ))

    def seconds_to_tick(self, sec: float) -> int:
        return int(round(sec * self.tempo_bpm / 60.0 * PPQ))

    # -- 查询 --------------------------------------------------------------

    def events_at(self, tick: int) -> list[TimedEvent]:
        """此刻正在响的音符。区间是半开的 [start, end)，所以答案唯一。"""
        cand = self._buckets.get(int(tick) // BUCKET_TICKS, ())
        return [te for te in cand if te.start_tick <= tick < te.end_tick]

    def measure_at(self, tick: int) -> Measure | None:
        for s, e, m in self.spans:
            if s <= tick < e:
                return m
        return self.spans[-1][2] if self.spans else None

    def page_at(self, tick: int) -> int:
        """此刻该显示第几页。越界时钳到首/尾页。"""
        if not self.spans:
            return 1
        for s, e, m in self.spans:
            if s <= tick < e:
                return int(m.page)
        return int(self.spans[-1][2].page) if tick > 0 else int(self.spans[0][2].page)

    def page_span_ticks(self, page: int) -> tuple[int, int] | None:
        """某一页覆盖的 tick 区间 —— 屏幕上"翻页"要用。"""
        sel = [(s, e) for s, e, m in self.spans if int(m.page) == page]
        if not sel:
            return None
        return min(s for s, _ in sel), max(e for _, e in sel)

    def progress(self, tick: int) -> float:
        """0..1 的播放进度。"""
        return min(1.0, max(0.0, tick / self.total_ticks)) if self.total_ticks else 0.0

    def tick_at_page_start(self, page: int) -> int:
        span = self.page_span_ticks(page)
        return span[0] if span else 0

    # -- 诊断 --------------------------------------------------------------

    def summary(self) -> str:
        return (f"时间轴: {len(self.events)} 个音符, {len(self.spans)} 个小节, "
                f"{len(self.pages)} 页, {self.total_ticks} tick = "
                f"{self.total_seconds:.2f}s @ {self.tempo_bpm:g}bpm "
                f"({self.sample_rate}Hz, {self.total_samples} 采样)")
