"""
FPGA-OMR —— 基于 Xilinx AC920 (XCZU4EV) 的乐谱识别与播放系统

模块划分与最终 PL/PS 分工严格对应：

    render.py       合成乐谱渲染 + 退化模拟        (仅 PC，做 ground truth)
    preprocess.py   灰度/光照归一化/二值化/去斜     -> 目标下沉 PL
    staff.py        五线谱检测 + 谱线去除           -> 目标下沉 PL
    detect.py       符号检测                        -> 目标下沉 PL/DPU
    reconstruct.py  乐理语义重建 + 节拍约束纠错      -> 永远留 PS
    timeline.py     全局时间轴（播放/合成/高亮共用）  -> PS
    synth.py        软件合成器（-> PCM/WAV）         -> PS（同构于将来的 PL 硬件合成器）
    player.py       播放控制 + 音频后端              -> PS
    batch.py        SD 卡批处理工作流                -> PS
    viz.py          绘制（识别框叠加 / 放音高亮）     -> PS（HDMI 复用）
    export.py       MusicXML / MIDI 导出             -> PS
    model.py        数据模型（全项目地基）
    theory.py       乐理核心（全项目共用）
    evaluate.py     精度评测
    pipeline.py     端到端编排
"""

__version__ = "0.2.0-p1"

from . import (batch, evaluate, export, model, player, preprocess, reconstruct,
               render, staff, synth, theory, timeline, viz)
from .model import BBox, Event, Measure, Part, Score, Staff, Voice, validate_measures

__all__ = [
    "batch", "evaluate", "export", "model", "player", "preprocess",
    "reconstruct", "render", "staff", "synth", "theory", "timeline", "viz",
    "BBox", "Event", "Measure", "Part", "Score", "Staff", "Voice",
    "validate_measures",
    "__version__",
]
