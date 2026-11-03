"""
播放控制与音频后端。

架构要点：**时钟与后端解耦**。
播放位置由 `time.monotonic()` 驱动，而不是由音频设备上报 —— 因为
"把渲染好的 WAV 交给系统播放器"这条路上，播放器不回传位置。
这带来一个好处：同一套位置逻辑既能带声音跑，也能在**没有音频设备的
环境里静音跑**（CI、板子调试、还没接 DAC 的阶段），
所以高亮跟随的代码可以在有声音之前就全部开发和验证完。

后端选择
--------
* Windows  -> `winsound`（只吃 WAV，正好我们渲染的就是 WAV）
* Linux    -> `aplay` / `paplay` / `ffplay` 依次尝试（板子上是 ALSA，aplay 够用）
* 都没有    -> `NullBackend`：静音但时钟照走，功能与高亮验证不受影响
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .model import Score
from .synth import (AudioCheck, SynthConfig, pcm_stats, render_pcm, verify_audio,
                    write_wav)
from .timeline import TimedEvent, Timeline

#: 默认把试听 WAV 写在工程目录下（而不是系统临时目录）——
#: 沙箱/无权限环境下更可靠，用户也能直接找到文件。
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "out"


# ---------------------------------------------------------------------------
# 音频后端
# ---------------------------------------------------------------------------


class AudioBackend(Protocol):
    name: str

    def available(self) -> bool: ...
    def start(self, wav_path: Path) -> None: ...
    def stop(self) -> None: ...


class NullBackend:
    """静音后端：不出声，但**时钟照走**。

    存在的意义：让播放控制与高亮跟随的逻辑可以在没有音频设备时
    完整开发和验证 —— 包括在板子上还没接 DAC 的阶段。
    """

    name = "null"

    def available(self) -> bool:
        return True

    def start(self, wav_path: Path) -> None:
        return None

    def stop(self) -> None:
        return None


class SystemBackend:
    """调用系统播放器播 WAV。"""

    name = "system"

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._player: str | None = None
        self._winsound = None
        if sys.platform.startswith("win"):
            try:
                import winsound
                self._winsound = winsound
                self._player = "winsound"
            except Exception:
                self._winsound = None
        if self._player is None:
            for cand, args in (("aplay", ["-q"]), ("paplay", []),
                               ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
                               ("afplay", [])):
                exe = shutil.which(cand)
                if exe:
                    self._player = cand
                    self._args = args
                    break

    def available(self) -> bool:
        return self._player is not None

    def start(self, wav_path: Path) -> None:
        if self._player == "winsound":
            # SND_ASYNC：立即返回，由系统在后台播；SND_NODEFAULT 防止失败时放默认提示音
            self._winsound.PlaySound(
                str(wav_path),
                self._winsound.SND_FILENAME | self._winsound.SND_ASYNC
                | self._winsound.SND_NODEFAULT)
            return
        if self._player is None:
            raise RuntimeError("没有可用的系统音频播放器")
        exe = shutil.which(self._player)
        self._proc = subprocess.Popen([exe, *getattr(self, "_args", []), str(wav_path)],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)

    def stop(self) -> None:
        if self._player == "winsound":
            try:
                self._winsound.PlaySound(None, self._winsound.SND_PURGE)
            except Exception:
                pass
            return
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None


def pick_backend(prefer: str = "auto") -> AudioBackend:
    """选一个可用的后端。`prefer` 可取 "auto" | "null" | "system"。"""
    if prefer == "null":
        return NullBackend()
    if prefer == "system":
        return SystemBackend()
    b = SystemBackend()
    return b if b.available() else NullBackend()


# ---------------------------------------------------------------------------
# 播放信息
# ---------------------------------------------------------------------------


@dataclass
class PlaybackInfo:
    wav_path: str = ""
    duration_s: float = 0.0
    peak: float = 0.0
    rms: float = 0.0
    backend: str = ""
    check: AudioCheck | None = None
    render_ms: float = 0.0

    def summary(self) -> str:
        L = [f"音频: {self.wav_path}",
             f"  时长 {self.duration_s:.2f}s  峰值 {self.peak:.2f}  RMS {self.rms:.3f}  "
             f"后端 {self.backend}  渲染 {self.render_ms:.0f}ms"]
        if self.check is not None:
            L.append("  " + self.check.summary())
        return "\n".join(L)


# ---------------------------------------------------------------------------
# 播放器
# ---------------------------------------------------------------------------


class Player:
    """播放器：渲染 + 播放 + 位置查询。

    **位置查询是这个类真正的价值所在。** `active_events()` / `current_page()`
    是第 2 步（HDMI 显示乐谱并高亮当前音符）唯一需要调用的接口；
    音频只是它的一个消费端。
    """

    def __init__(self, score: Score | Timeline, tempo_bpm: float = 100.0,
                 cfg: SynthConfig | None = None,
                 backend: AudioBackend | None = None):
        self.cfg = cfg or SynthConfig()
        self.timeline = (score if isinstance(score, Timeline)
                         else Timeline(score, tempo_bpm=tempo_bpm,
                                       sample_rate=self.cfg.sample_rate))
        self.backend = backend if backend is not None else pick_backend()
        self.pcm: np.ndarray | None = None
        self.info = PlaybackInfo(backend=self.backend.name)

        self._t0: float | None = None
        self._start_tick = 0
        self._paused_at: float | None = None
        self._playing = False

    # -- 准备 --------------------------------------------------------------

    def prepare(self, wav_path: str | Path | None = None,
                verify: bool = True) -> PlaybackInfo:
        """渲染 PCM 并落盘成 WAV。可以在播放前单独调用（比如只导出不播放）。"""
        t0 = time.perf_counter()
        pcm, peak = render_pcm(self.timeline, self.cfg)
        self.pcm = pcm
        path = Path(wav_path) if wav_path else DEFAULT_OUT_DIR / "playback.wav"
        write_wav(path, pcm, self.cfg.sample_rate)
        st = pcm_stats(pcm)
        self.info = PlaybackInfo(
            wav_path=str(path),
            duration_s=pcm.size / self.cfg.sample_rate,
            peak=st["peak"], rms=st["rms"], backend=self.backend.name,
            render_ms=(time.perf_counter() - t0) * 1000.0,
        )
        if verify:
            self.info.check = verify_audio(self.timeline, pcm, self.cfg)
        return self.info

    # -- 控制 --------------------------------------------------------------

    def play(self, start_tick: int = 0) -> None:
        if self.pcm is None or not self.info.wav_path:
            self.prepare()
        self.stop()
        self._start_tick = int(start_tick)
        # 从中间开始播放需要裁掉前面那段音频，否则声音和位置会对不上
        if start_tick > 0:
            off = self.timeline.tick_to_sample(start_tick)
            self.pcm = self.pcm[off:] if off < self.pcm.size else self.pcm[:0]
            write_wav(self.info.wav_path, self.pcm, self.cfg.sample_rate)
        self.backend.start(Path(self.info.wav_path))
        self._t0 = time.monotonic()
        self._paused_at = None
        self._playing = True

    def stop(self) -> None:
        self.backend.stop()
        self._playing = False
        self._t0 = None
        self._paused_at = None

    def pause(self) -> None:
        if self._playing and self._t0 is not None:
            self._paused_at = self.position_tick
            self.backend.stop()
            self._playing = False

    def resume(self) -> None:
        if self._paused_at is not None:
            self.play(self._paused_at)

    # -- 位置 --------------------------------------------------------------

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def position_tick(self) -> int:
        """当前播放位置（全局 tick）。

        暂停时返回暂停点；未播放时返回起点。
        """
        if self._paused_at is not None:
            return int(self._paused_at)
        if self._t0 is None:
            return int(self._start_tick)
        return self._start_tick + self.timeline.seconds_to_tick(time.monotonic() - self._t0)

    @property
    def position_seconds(self) -> float:
        return self.timeline.tick_to_seconds(self.position_tick)

    @property
    def position_sample(self) -> int:
        return self.timeline.tick_to_sample(self.position_tick)

    @property
    def finished(self) -> bool:
        return self.position_tick >= self.timeline.total_ticks

    def active_events(self) -> list[TimedEvent]:
        """此刻正在响的音符 —— 高亮跟随就是拿它去画黄框。"""
        return self.timeline.events_at(self.position_tick)

    def current_page(self) -> int:
        return self.timeline.page_at(self.position_tick)

    def progress(self) -> float:
        return self.timeline.progress(self.position_tick)

    # -- 播放循环（第 2 步的接入点）----------------------------------------

    def run(self, on_frame: Callable[[Player], None] | None = None,
            fps: float = 60.0, loop: bool = False) -> int:
        """阻塞播放，每帧回调一次 `on_frame(player)`。

        这就是 HDMI 显示线程要接的地方：
            def on_frame(p):
                show(render_page(p.current_page(), highlight=p.active_events()))
        在没有音频设备的机器上配合 NullBackend 也能跑，所以显示逻辑
        可以先于音频设备完成开发和验证。

        Returns
        -------
        播放结束时的 tick。
        """
        if not self._playing:
            self.play()
        period = 1.0 / max(1.0, fps)
        while self._playing:
            if on_frame is not None:
                on_frame(self)
            if self.finished:
                break
            time.sleep(period)
        if loop and self._playing:
            self.play(0)
        end = self.position_tick
        self.stop()
        return end
