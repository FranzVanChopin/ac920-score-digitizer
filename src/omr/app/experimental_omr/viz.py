"""
可视化工具。

被两个地方共用：
  * `cli.py`  —— 调试时把中间结果和识别框存成 PNG
  * 后续的 HDMI 显示 / 放音高亮跟随（功能 2）—— 同一套绘制代码，
    只是把输出目标从 PNG 文件换成帧缓冲

**所有绘制都按"页"来做**：多页乐谱合并后事件的 bbox 是页内局部坐标，
必须先用 `page` 过滤，否则会把第 2 页的框画到第 1 页上。
"""

from __future__ import annotations

import numpy as np

from .model import Score

#: 彩色判定用的调色板（RGB）
COLOR_STAFF = (0, 200, 0)        # 绿：谱线
COLOR_NOTE_OK = (220, 0, 0)      # 红：高置信音符
COLOR_NOTE_SUSPECT = (255, 140, 0)   # 橙：可疑（0.6~0.85）
COLOR_NOTE_LOW = (160, 0, 160)   # 紫：低置信
COLOR_REST = (0, 90, 255)        # 蓝：休止符
COLOR_HIGHLIGHT = (255, 220, 0)  # 黄：当前正在演奏的音符


def mask_to_img(mask: np.ndarray) -> np.ndarray:
    """bool 墨迹图 -> 白底黑字 uint8 灰度图。"""
    return np.where(mask, 0, 255).astype(np.uint8)


def to_rgb(gray: np.ndarray) -> np.ndarray:
    """灰度 -> RGB（如果是单通道）。"""
    if gray.ndim == 3:
        return gray.astype(np.uint8)
    return np.stack([gray] * 3, axis=-1).astype(np.uint8)


def draw_rect(rgb: np.ndarray, x: float, y: float, w: float, h: float,
              color: tuple[int, int, int], thickness: int = 1) -> None:
    """在 RGB 图上画矩形边框（原地修改，越界自动裁剪）。"""
    H, W = rgb.shape[:2]
    x0 = int(max(0, round(x)))
    y0 = int(max(0, round(y)))
    x1 = int(min(W - 1, round(x + w)))
    y1 = int(min(H - 1, round(y + h)))
    if x1 <= x0 or y1 <= y0:
        return
    t = max(1, int(thickness))
    for k in range(t):
        if y0 + k <= y1:
            rgb[y0 + k, x0:x1 + 1] = color
        if y1 - k >= y0:
            rgb[y1 - k, x0:x1 + 1] = color
        if x0 + k <= x1:
            rgb[y0:y1 + 1, x0 + k] = color
        if x1 - k >= x0:
            rgb[y0:y1 + 1, x1 - k] = color


def confidence_color(conf: float, is_rest: bool = False) -> tuple[int, int, int]:
    """按置信度取颜色。人工校正 UI 靠这个把注意力引到可疑项上。"""
    if is_rest:
        return COLOR_REST
    if conf >= 0.85:
        return COLOR_NOTE_OK
    if conf >= 0.6:
        return COLOR_NOTE_SUSPECT
    return COLOR_NOTE_LOW


def overlay(gray: np.ndarray, score: Score, staves=None, page: int = 1,
            show_staff: bool = True, thickness: int = 2) -> np.ndarray:
    """把识别框画到**指定页**的原图上。

    Parameters
    ----------
    page
        只绘制 `event.page == page`（以及 `measure.page == page`）的内容。
        多页乐谱必须指定，否则框会串页。
    """
    rgb = to_rgb(gray)

    if show_staff and staves:
        for s in staves:
            for ly in s.line_ys:
                yy = int(round(ly))
                if 0 <= yy < rgb.shape[0]:
                    x0 = int(max(0, s.x0))
                    x1 = int(min(rgb.shape[1], s.x1))
                    if x1 > x0:
                        rgb[yy, x0:x1] = COLOR_STAFF

    for _p, _s, m in score.iter_measures():
        if m.page != page:
            continue
        for v in m.voices:
            for e in v.events:
                if e.bbox is None:
                    continue
                col = confidence_color(e.confidence, e.type == "rest")
                draw_rect(rgb, e.bbox.x, e.bbox.y, e.bbox.w, e.bbox.h, col, thickness)
    return rgb


def highlight_events(gray: np.ndarray, score: Score, page: int,
                     active: list, thickness: int = 3) -> np.ndarray:
    """把**当前正在演奏**的音符用醒目颜色标出来。

    这是功能 2（放音高亮跟随）的核心绘制函数：
    `active` 是当前时刻应该高亮的事件列表（可能多个 —— 和弦）。
    这里刻意只做"标注"，不做"取页"—— 播放器负责决定当前是哪一页、哪些事件。
    """
    rgb = to_rgb(gray)
    for e in active:
        if e.bbox is None or e.page != page:
            continue
        b = e.bbox
        # 高亮 = 加粗的黄框 + 一层淡黄色半透明底
        x0 = int(max(0, round(b.x - 2)))
        y0 = int(max(0, round(b.y - 2)))
        x1 = int(min(rgb.shape[1], round(b.x + b.w + 2)))
        y1 = int(min(rgb.shape[0], round(b.y + b.h + 2)))
        if x1 <= x0 or y1 <= y0:
            continue
        region = rgb[y0:y1, x0:x1].astype(np.float32)
        region = region * 0.72 + np.array(COLOR_HIGHLIGHT, dtype=np.float32) * 0.28
        rgb[y0:y1, x0:x1] = region.astype(np.uint8)
        draw_rect(rgb, b.x - 2, b.y - 2, b.w + 4, b.h + 4, COLOR_HIGHLIGHT, thickness)
    return rgb
