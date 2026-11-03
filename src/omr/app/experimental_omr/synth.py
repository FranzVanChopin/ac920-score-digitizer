"""
软件合成器 —— 把时间轴渲染成 PCM 音频。

为什么不用 FluidSynth / timidity 这些现成的 MIDI 合成器
-------------------------------------------------------
1. **同步精度**。外挂合成器只能通过"现在过了多久"间接推算位置，音画同步会漂。
   自己渲染 PCM，音符的采样位置是精确已知的 —— 这正是第 2 步
   （HDMI 播放时高亮当前音符）所依赖的。
2. **部署**。板子上装 SoundFont 是个负担；几十行加性合成 + 声道输出反而更可控。
3. **同构**。这套"波表/加性合成 + ADSR 包络 + 多声部叠加"的结构，
   和将来在 PL 里写硬件合成器是一一对应的。
   换句话说：这里的每一行，后面都能翻译成 RTL。

代价是音色比 SoundFont 差。所以 `export.py` 依然导出标准 MIDI ——
想要好音色时用任何 MIDI 播放器打开 `score.mid` 即可，两条路并存。

音色设计
--------
用**加性合成**（若干次谐波按幅度叠加）+ ADSR：
  * `organ` —— 谐波丰富且衰减慢，接近风琴，最"中性"
  * `pluck` —— 高次谐波衰减更快，接近拨弦/钢琴
"""

from __future__ import annotations

import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .timeline import Timeline

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 每种音色的谐波表：(谐波次数, 相对幅度)
HARMONICS: dict[str, list[tuple[int, float]]] = {
    "sine": [(1, 1.0)],
    "organ": [(1, 1.0), (2, 0.50), (3, 0.32), (4, 0.18), (6, 0.10), (8, 0.05)],
    "pluck": [(1, 1.0), (2, 0.42), (3, 0.25), (4, 0.14), (5, 0.09), (6, 0.05)],
}


@dataclass
class SynthConfig:
    """合成参数。默认值调成"能听出旋律、不刺耳、文件不大"。"""

    sample_rate: int = 22050      # 板子上够用；44.1k 文件会大一倍
    waveform: str = "pluck"       # "sine" | "organ" | "pluck"
    attack_ms: float = 6.0
    decay_ms: float = 150.0
    sustain: float = 0.30
    release_ms: float = 130.0
    gain: float = 0.42
    #: pluck 音色的基准衰减时间常数（秒）与高次谐波加快速率
    pluck_tau: float = 1.1
    pluck_harmonic_decay: float = 1.6


# ---------------------------------------------------------------------------
# 包络与单音波形
# ---------------------------------------------------------------------------


def _envelope(n_hold: int, n_release: int, n_attack: int, n_decay: int,
              sustain: float) -> np.ndarray:
    """ADSR 包络（长度 = 按住时长 + 释放时长）。"""
    n = n_hold + n_release
    env = np.zeros(n, dtype=np.float32)
    if n <= 0:
        return env
    a = int(max(1, min(n_attack, n_hold)))
    env[:a] = np.linspace(0.0, 1.0, a, endpoint=False, dtype=np.float32)
    if n_hold > a:
        d = int(max(1, min(n_decay, n_hold - a)))
        env[a:a + d] = np.linspace(1.0, sustain, d, endpoint=False, dtype=np.float32)
        if n_hold > a + d:
            env[a + d:n_hold] = sustain
    else:
        env[min(a, n - 1)] = 1.0
    if n_release > 0:
        k = np.arange(n_release, dtype=np.float32)
        env[n_hold:] = sustain * np.exp(-3.0 * k / max(1, n_release))
    return env


def midi_to_freq(pitch: int) -> float:
    """MIDI 音高 -> 基频（Hz）。A4=69=440Hz。"""
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def _note_wave(freq: float, n: int, cfg: SynthConfig) -> np.ndarray:
    """单个音的波形（不含包络）。"""
    t = np.arange(n, dtype=np.float32) / cfg.sample_rate
    nyq = cfg.sample_rate * 0.5
    table = HARMONICS.get(cfg.waveform, HARMONICS["sine"])
    w = np.zeros(n, dtype=np.float32)
    norm = 0.0
    pluck = cfg.waveform == "pluck"
    for h, amp in table:
        f = freq * h
        if f >= nyq:            # 超过奈奎斯特的谐波直接丢掉，否则会混叠成噪声
            break
        if pluck:
            # 高次谐波衰减更快 -> 起音明亮、余音温暖，接近拨弦
            amp_h = amp * np.exp(-t * (cfg.pluck_harmonic_decay * (h - 1)) / cfg.pluck_tau)
            w += amp_h * np.sin(2.0 * np.pi * f * t, dtype=np.float32)
        else:
            w += amp * np.sin(2.0 * np.pi * f * t, dtype=np.float32)
        norm += amp
    if norm > 0:
        w /= norm
    return w


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


def render_pcm(timeline: Timeline, cfg: SynthConfig | None = None) -> np.ndarray:
    """把时间轴渲染成 int16 单声道 PCM。

    多声部叠加：直接累加。峰值超 1.0 时整体缩放（简单限幅），
    避免削顶产生刺耳的失真。
    """
    cfg = cfg or SynthConfig()
    sr = cfg.sample_rate
    n_attack = int(cfg.attack_ms * sr / 1000.0)
    n_decay = int(cfg.decay_ms * sr / 1000.0)
    n_release = int(cfg.release_ms * sr / 1000.0)

    # 给末尾留出最后一个音的释放尾巴
    n_total = int(timeline.total_samples) + n_release + 1
    buf = np.zeros(n_total, dtype=np.float32)

    for te in timeline.events:
        hold = int(max(1, te.end_sample - te.start_sample))
        n = hold + n_release
        wave = _note_wave(midi_to_freq(te.pitch), n, cfg)
        env = _envelope(hold, n_release, n_attack, n_decay, cfg.sustain)
        seg = wave * env * cfg.gain
        start = int(te.start_sample)
        end = min(n_total, start + n)
        if end <= start:
            continue
        buf[start:end] += seg[: end - start]

    peak = float(np.max(np.abs(buf))) if buf.size else 0.0
    if peak > 1.0:
        buf /= peak
    return np.clip(buf * 32767.0, -32768, 32767).astype(np.int16), peak


def pcm_stats(pcm: np.ndarray) -> dict:
    """给验证用的统计量。"""
    x = pcm.astype(np.float32)
    if x.size == 0:
        return {"n": 0, "peak": 0.0, "rms": 0.0, "silent": True}
    return {
        "n": int(x.size),
        "peak": float(np.max(np.abs(x)) / 32768.0),
        "rms": float(np.sqrt(np.mean(x * x)) / 32768.0),
        "silent": bool(np.max(np.abs(x)) == 0),
    }


# ---------------------------------------------------------------------------
# WAV 读写
# ---------------------------------------------------------------------------


def write_wav(path: str | Path, pcm: np.ndarray, sample_rate: int) -> None:
    """写 16-bit 单声道 WAV。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(pcm, dtype=np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """读 16-bit 单声道 WAV -> (int16 数组, 采样率)。"""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return x, sr


# ---------------------------------------------------------------------------
# 客观验证：音频里到底有没有"那段音乐"
# ---------------------------------------------------------------------------
#
# 听不到声音的时候（脚本、CI、无音频设备的板子），必须能用数值证明
# 渲染出来的音频确实是这首歌，而不是一团静音或噪声。
# 做法：能量包络的上升沿检测，把检出的起音时刻和乐谱里音符的采样位置比对。


@dataclass
class AudioCheck:
    n_expected: int = 0
    n_detected: int = 0
    n_matched: int = 0
    n_missed: int = 0
    n_spurious: int = 0
    max_offset_ms: float = 0.0
    duration_ok: bool = True

    @property
    def ok(self) -> bool:
        return (self.n_missed == 0 and self.n_spurious == 0 and self.duration_ok)

    def summary(self) -> str:
        s = (f"音频客观验证: 期望 {self.n_expected} 个起音点, 检出 {self.n_detected}, "
             f"匹配 {self.n_matched}")
        if self.n_missed:
            s += f", 漏 {self.n_missed}"
        if self.n_spurious:
            s += f", 多 {self.n_spurious}"
        s += f", 最大时间偏差 {self.max_offset_ms:.1f}ms"
        s += f"  [{'通过' if self.ok else '不通过'}]"
        return s


def detect_onsets(pcm: np.ndarray, sample_rate: int,
                  win_ms: float = 10.0, lag_ms: float = 12.0,
                  min_gap_ms: float = 60.0,
                  floor_ratio: float = 0.12,
                  noise_k: float = 3.0,
                  min_ratio: float = 1.5) -> list[int]:
    """检测起音点，返回采样位置列表。

    ★ 必须用**能量上升沿**，不能用"静音后再触发"的施密特触发器。

    踩过的坑（一）：一开始用"包络低于 low 之后冲上 high"判定起音，
    连奏旋律只检出 2 个起音点（期望 24 个）。原因是旋律里音符首尾相接，
    前一个音还在保持电平（0.30）时下一个音就起音了，包络**永远不会降到 low**，
    触发器一直处于未武装状态。

    踩过的坑（二）：只看"上升幅度超过绝对阈值"还不够。低音（如 C4，261Hz）
    在 5ms 的 RMS 窗口里只有约 1.3 个周期，**包络本身带着明显的纹波**，
    纹波的一个峰就会被当成新起音 —— 实测 48 个音检出 49 个，
    多发点在最后一个 C4 之后 61ms。所以再加一条**相对增幅**判据：
    真正的起音会把包络抬高一大截（`env[i] >= min_ratio * env[i-lag]`），
    纹波只是小幅抖动（约 1.2 倍），会被拒掉。
    同时把窗口从 5ms 提到 10ms，从源头减小纹波。
    """
    x = np.asarray(pcm, dtype=np.float32)
    if x.size == 0:
        return []
    win = max(2, int(sample_rate * win_ms / 1000.0))
    hop = max(1, win // 2)
    if x.size < win:
        return []

    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    env = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
    peak = float(env.max())
    if peak <= 0:
        return []
    env = env / peak

    n_lag = max(1, int(lag_ms / 1000.0 * sample_rate / hop))
    rise = np.zeros_like(env)
    rise[n_lag:] = np.maximum(0.0, env[n_lag:] - env[:-n_lag])
    thr = max(floor_ratio, noise_k * float(np.median(rise)))

    min_gap = max(1, int(min_gap_ms / 1000.0 * sample_rate / hop))
    onsets: list[int] = []
    if env[0] >= thr:                       # 整曲第一个音恰好从采样 0 开始
        onsets.append(0)
    last = 0 if onsets else -10 ** 9
    for i in range(n_lag, len(rise)):
        if (rise[i] >= thr
                and env[i] >= min_ratio * max(float(env[i - n_lag]), 1e-6)
                and i - last >= min_gap):
            onsets.append(i * hop)
            last = i
    return onsets


def _merge_close(values: list[int], tol: int) -> list[int]:
    """把时间上太近的值合并成一个（和弦/重复音只算一次起音）。"""
    out: list[int] = []
    for v in sorted(values):
        if out and v - out[-1] <= tol:
            continue
        out.append(v)
    return out


def verify_audio(timeline: Timeline, pcm: np.ndarray,
                 cfg: SynthConfig | None = None,
                 tol_ms: float = 45.0) -> AudioCheck:
    """验证渲染出的音频与乐谱一致。

    这是"听不到声音时"的替代验收手段：把音频里检出的起音点和乐谱里
    音符的采样位置逐一比对。数量对得上、时间偏差在容差内，就说明
    合成链路（时间轴 -> 包络 -> 采样位置）是正确的。
    """
    cfg = cfg or SynthConfig()
    res = AudioCheck()
    tol = int(cfg.sample_rate * tol_ms / 1000.0)

    expected = _merge_close([te.start_sample for te in timeline.events], tol)
    detected = _merge_close(detect_onsets(pcm, cfg.sample_rate), tol)
    res.n_expected = len(expected)
    res.n_detected = len(detected)

    used = [False] * len(detected)
    for e in expected:
        best, best_d = -1, tol + 1
        for j, d in enumerate(detected):
            if used[j]:
                continue
            dd = abs(d - e)
            if dd < best_d:
                best, best_d = j, dd
        if best >= 0:
            used[best] = True
            res.n_matched += 1
            res.max_offset_ms = max(res.max_offset_ms,
                                    best_d / cfg.sample_rate * 1000.0)
        else:
            res.n_missed += 1
    res.n_spurious = sum(1 for u in used if not u)

    # 时长检查：音频应覆盖整曲 + 一段释放尾巴
    need = int(timeline.total_samples)
    res.duration_ok = pcm.size >= need
    return res
