"""
SD 卡批处理工作流 —— 项目的实际运行形态。

工作流
------
    SD 卡/omr_in/<曲名>/page_01.jpg ...
            │
            ├─ 逐页识别（可选：跳过已完成的页）
            │      └─ 中间结果落盘 pages/page_NN.json（断点续跑用）
            │
            ├─ 合并多页 → 一首完整曲子（小节号连续，事件带页号）
            │
            └─ SD 卡/omr_out/<曲名>/
                   score.musicxml / score.mid / score.json / manifest.json
                   pages/page_NN_gray.png     ← 显示与高亮跟随的底图
                   pages/page_NN_binary.png   ← 二值化结果（排错用）
                   pages/page_NN_overlay.png  ← 识别框叠加（人工复核用）

三个必须做对的工程细节
----------------------
1. **掉电安全写入**。SD 卡在断电/拔卡时会出现"半写"文件。所有输出都必须
   先写 `.tmp` → `fsync` → `os.replace` 原子改名。直接 `open(w)` 一中断，
   连旧文件都保不住。见 `atomic_write_text`。
2. **断点续跑**。板子上一次扫描可能有几十页，中途断电很常见。
   每页识别完立刻落盘检查点，并用 (size, mtime) 校验源文件是否变过。
   重跑时已完成的页直接复用。
3. **页号必须贯穿模型**。多页合并后事件的 bbox 是页内局部坐标，
   没有页号就无法定位、无法在 HDMI 上正确高亮。见 `model.Event.page`。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .model import (Measure, Part, Score, Staff, merge_scores,  # noqa: F401
                    validate_measures)
from .pipeline import PipelineConfig, recognize
from .viz import mask_to_img, overlay

# ---------------------------------------------------------------------------
# 约定
# ---------------------------------------------------------------------------

#: SD 卡上的固定目录名
SD_IN_DIR = "omr_in"
SD_OUT_DIR = "omr_out"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

#: 检查点格式版本。改了中间表示就 +1，强制重跑而不是读进旧格式。
CHECKPOINT_VERSION = 1


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def natural_key(s: str):
    """自然排序键：让 page_2 排在 page_10 前面（普通字典序会反过来）。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def _fsync_dir(path: Path) -> None:
    """把目录项本身刷盘。POSIX 上才有效；Windows 会抛错，忽略即可。"""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass


def atomic_write_text(path: Path, text: str) -> None:
    """掉电安全地写文本文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """掉电安全地写二进制文件（MIDI 用）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def atomic_save_image(arr: np.ndarray, path: Path) -> None:
    """掉电安全地存 PNG。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(arr).save(tmp, format="PNG")
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def sha1_of_file(path: Path, chunk: int = 1 << 20) -> str:
    """分块算文件哈希（大图时不要一次性读进内存）。"""
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pipeline_fingerprint(cfg: PipelineConfig) -> str:
    """把影响识别结果的全部参数压成一个短指纹。

    ★ 断点续跑必须校验它。否则改了二值化窗口、换了检测器之后重跑，
    检查点仍然命中，产出的是"参数改了但结果一点没变"的假象 ——
    这种 bug 极难发现，因为每一步看起来都"成功"了。
    """
    import hashlib
    items: dict[str, object] = {
        "detector": cfg.detector,
        "onnx_model": cfg.onnx_model or "",
        "onnx_conf": cfg.onnx_conf,
        "staff_min_coverage": cfg.staff_min_coverage,
    }
    for k, v in sorted(vars(cfg.preprocess).items()):
        items[f"pre.{k}"] = v
    for k, v in sorted(vars(cfg.reconstruct).items()):
        items[f"rec.{k}"] = v
    blob = json.dumps(items, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 输入发现
# ---------------------------------------------------------------------------


@dataclass
class Work:
    """一首曲子 = 一组按页序排好的图片。"""

    name: str
    pages: list[Path]
    source_dir: Path | None = None

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    def __str__(self) -> str:
        return f"{self.name} ({self.n_pages} 页)"


def discover_works(input_dir: Path) -> list[Work]:
    """扫描输入目录，得到待处理的曲子列表。

    约定（也是给用户的规矩，写在 README 里）：
      * **子目录 = 一首多页曲子**，目录名即曲名，目录内图片按文件名**自然排序**
        （所以 page_2 会正确排在 page_10 前面）
      * **根目录下的散图 = 每张图各自算一首单页曲子**，文件名（去扩展名）即曲名
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    works: list[Work] = []
    for child in sorted(input_dir.iterdir(), key=lambda p: natural_key(p.name)):
        if child.is_dir():
            imgs = sorted(
                (p for p in child.iterdir()
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
                key=lambda p: natural_key(p.name))
            if imgs:
                works.append(Work(name=child.name, pages=imgs, source_dir=child))
        elif child.is_file() and child.suffix.lower() in IMAGE_EXTS:
            works.append(Work(name=child.stem, pages=[child]))
    return works


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


@dataclass
class PageReport:
    index: int
    source: str
    source_size: int = 0
    source_mtime: float = 0.0
    skipped: bool = False          # True = 命中检查点，未重新识别
    elapsed_ms: float = 0.0
    n_staves: int = 0
    staff_space: float = 0.0
    skew_deg: float = 0.0
    width: int = 0
    height: int = 0
    n_events: int = 0
    n_notes: int = 0
    mean_confidence: float = 0.0
    #: 纠错之后**仍然**对不上拍号的问题（这些才需要人工看）
    measure_issues: list[str] = field(default_factory=list)
    #: 节拍约束自动纠错的记录（已修好，仅作留痕）
    repairs: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    error: str = ""


@dataclass
class WorkReport:
    name: str
    n_pages: int = 0
    n_measures: int = 0
    n_notes: int = 0
    mean_confidence: float = 0.0
    elapsed_ms: float = 0.0
    pages: list[PageReport] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and all(not p.error for p in self.pages)

    @property
    def n_pages_skipped(self) -> int:
        return sum(1 for p in self.pages if p.skipped)

    @property
    def n_issues(self) -> int:
        return sum(len(p.measure_issues) for p in self.pages)

    @property
    def n_repaired(self) -> int:
        return sum(len(p.repairs) for p in self.pages)


# ---------------------------------------------------------------------------
# 多页合并
# ---------------------------------------------------------------------------


def merge_scores_pages(scores: list[Score], title: str = "") -> Score:
    """多页合并。实现已统一到 `model.merge_scores`（多系统合并也用同一份）。"""
    return merge_scores(scores, title=title)


# ---------------------------------------------------------------------------
# 单页处理
# ---------------------------------------------------------------------------


def _checkpoint_path(out_dir: Path, index: int) -> Path:
    return out_dir / "pages" / f"page_{index:02d}.json"


def _try_resume(ckpt: Path, source: Path, fingerprint: str) -> Score | None:
    """检查点是否可用：存在 + 版本对 + 源文件没变 + **识别参数没变**。"""
    if not ckpt.exists():
        return None
    try:
        data = json.loads(ckpt.read_text(encoding="utf-8"))
    except Exception:
        return None
    meta = data.get("_meta", {})
    if int(meta.get("version", 0)) != CHECKPOINT_VERSION:
        return None
    if str(meta.get("fingerprint", "")) != fingerprint:
        return None          # ★ 参数变了，必须重算
    try:
        st = source.stat()
    except OSError:
        return None
    if int(meta.get("source_size", -1)) != st.st_size:
        return None
    # mtime 允许 1 秒误差：不同文件系统（FAT32/exFAT/ext4）的时间精度不同
    if abs(float(meta.get("source_mtime", 0.0)) - st.st_mtime) > 1.0:
        return None
    try:
        return Score.from_dict(data)
    except Exception:
        return None


def process_page(source: Path, out_dir: Path, index: int,
                 cfg: PipelineConfig, resume: bool = True,
                 save_page_images: bool = True,
                 log=lambda *a: None) -> tuple[Score, PageReport]:
    """识别单页并落盘。返回 (该页的 Score, 报告)。"""
    pr = PageReport(index=index, source=str(source))
    try:
        st = source.stat()
        pr.source_size, pr.source_mtime = st.st_size, st.st_mtime
    except OSError as e:
        pr.error = f"读不到源文件: {e}"
        return Score(title=cfg.title), pr

    t0 = time.perf_counter()
    ckpt = _checkpoint_path(out_dir, index)
    fp = pipeline_fingerprint(cfg)

    score = _try_resume(ckpt, source, fp) if resume else None
    if score is not None:
        pr.skipped = True
        log(f"    第 {index} 页：命中检查点，跳过识别")
        # 报告信息从检查点里恢复。
        # ★ 页宽高也必须恢复：HDMI 显示要把事件 bbox（页内像素坐标）缩放到屏幕上，
        #   丢了尺寸就只能显示 0×0 的页表项。实测续跑一次就会把 score.json 里的
        #   页表写成 w=0,h=0 —— 而"先跑一次、断电后重跑"正是板子上的常态。
        meta = json.loads(ckpt.read_text(encoding="utf-8")).get("_meta", {})
        pr.staff_space = float(meta.get("staff_space", 0.0))
        pr.n_staves = int(meta.get("n_staves", 0))
        pr.width = int(meta.get("width", 0))
        pr.height = int(meta.get("height", 0))
        pr.n_events = sum(1 for _ in score.iter_events())
        pr.n_notes = score.note_count()
        pr.mean_confidence = score.mean_confidence()
        pr.elapsed_ms = (time.perf_counter() - t0) * 1000
        return score, pr

    img = np.asarray(Image.open(source).convert("L"))
    pr.width, pr.height = int(img.shape[1]), int(img.shape[0])

    page = recognize(img, cfg)
    score = page.score

    # ★ 页号必须写回模型：合并后 bbox 是页内坐标，只靠页号才能定位
    for _p, _s, m in score.iter_measures():
        m.page = index
        for v in m.voices:
            for e in v.events:
                e.page = index

    pr.n_staves = len(page.staves)
    pr.staff_space = float(page.staves[0].space) if page.staves else 0.0
    pr.skew_deg = float(page.pre.skew_angle) if page.pre else 0.0
    pr.n_events = sum(1 for _ in score.iter_events())
    pr.n_notes = score.note_count()
    pr.mean_confidence = score.mean_confidence()
    pr.measure_issues = [str(i) for i in page.issues if i.severity != "repaired"]
    pr.repairs = [str(i) for i in page.issues if i.severity == "repaired"]
    if not page.staves:
        pr.error = "未检测到五线谱"

    gray_path = out_dir / "pages" / f"page_{index:02d}_gray.png"
    if save_page_images:
        atomic_save_image(img, gray_path)
        pr.outputs["gray"] = str(gray_path.name)
        if page.pre is not None:
            b = out_dir / "pages" / f"page_{index:02d}_binary.png"
            atomic_save_image(mask_to_img(page.pre.binary), b)
            pr.outputs["binary"] = str(b.name)
    ov = out_dir / "pages" / f"page_{index:02d}_overlay.png"
    atomic_save_image(overlay(img, score, page.staves, page=index), ov)
    pr.outputs["overlay"] = str(ov.name)

    # ★ 先写检查点，再写别的 —— 检查点是"这一页已经算完"的唯一凭据
    payload = score.to_dict()
    payload["_meta"] = {
        "version": CHECKPOINT_VERSION,
        "fingerprint": fp,
        "source": str(source),
        "source_size": pr.source_size,
        "source_mtime": pr.source_mtime,
        "page_index": index,
        "width": pr.width,
        "height": pr.height,
        "staff_space": pr.staff_space,
        "n_staves": pr.n_staves,
        "skew_deg": pr.skew_deg,
        "n_events": pr.n_events,
        "mean_confidence": pr.mean_confidence,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_text(ckpt, json.dumps(payload, ensure_ascii=False, indent=2))

    pr.elapsed_ms = (time.perf_counter() - t0) * 1000
    return score, pr


# ---------------------------------------------------------------------------
# 单首曲子
# ---------------------------------------------------------------------------


def process_work(work: Work, out_root: Path, cfg: PipelineConfig | None = None,
                 resume: bool = True, save_page_images: bool = True,
                 make_wav: bool = False, tempo_bpm: float = 100.0,
                 log=lambda *a: None) -> WorkReport:
    """处理一首曲子（可能多页），产出 MusicXML / MIDI / JSON / 报告。

    每页独立落盘，最后才合并写整曲文件 —— 这样中途断电最多丢当前页，
    重跑时前面的页全部命中检查点。
    """
    cfg = cfg or PipelineConfig()
    cfg.title = cfg.title or work.name
    out_dir = Path(out_root) / work.name
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)

    rep = WorkReport(name=work.name, n_pages=work.n_pages)
    t0 = time.perf_counter()
    log(f"  [{work.name}] {work.n_pages} 页")

    page_scores: list[Score] = []
    for i, src in enumerate(work.pages, start=1):
        try:
            score, pr = process_page(src, out_dir, i, cfg,
                                     resume=resume,
                                     save_page_images=save_page_images,
                                     log=log)
        except Exception as e:  # 单页失败不应该拖垮整首
            pr = PageReport(index=i, source=str(src), error=f"{type(e).__name__}: {e}")
            score = Score(title=cfg.title)
            log(f"    第 {i} 页：失败 {pr.error}")
        rep.pages.append(pr)
        page_scores.append(score)

    merged = merge_scores(page_scores, title=work.name)
    merged.pages = [
        {"index": i,
         "image": f"pages/page_{i:02d}_gray.png",
         "w": rep.pages[i - 1].width,
         "h": rep.pages[i - 1].height,
         "source": rep.pages[i - 1].source}
        for i in range(1, len(rep.pages) + 1)
    ]

    rep.n_measures = sum(1 for _ in merged.iter_measures())
    rep.n_notes = merged.note_count()
    rep.mean_confidence = merged.mean_confidence()

    # --- 整曲输出 ---------------------------------------------------------
    from . import export as ex
    json_path = out_dir / "score.json"
    xml_path = out_dir / "score.musicxml"
    mid_path = out_dir / "score.mid"

    atomic_write_text(json_path, merged.to_json())
    atomic_write_text(xml_path, ex.to_musicxml(merged))
    atomic_write_bytes(mid_path, ex.to_midi_absolute(merged, tempo_bpm=tempo_bpm))

    rep.outputs = {"json": str(json_path.name), "musicxml": str(xml_path.name),
                   "midi": str(mid_path.name)}

    # --- 可选的 WAV 渲染 ---------------------------------------------------
    # 板子上如果没有 MIDI 合成器（默认就没有），预渲染好的 WAV 可以直接
    # 丢给 ALSA 播放 —— 这是功能 1 在设备上最省事的落地方式。
    if make_wav:
        try:
            from .synth import SynthConfig, render_pcm, write_wav
            from .timeline import Timeline
            scfg = SynthConfig()
            tl = Timeline(merged, tempo_bpm=tempo_bpm,
                          sample_rate=scfg.sample_rate)
            pcm, _peak = render_pcm(tl, scfg)
            wav_path = out_dir / "score.wav"
            write_wav(wav_path, pcm, scfg.sample_rate)
            rep.outputs["wav"] = wav_path.name
            log(f"    -> WAV {wav_path.name}  "
                f"{pcm.size/scfg.sample_rate:.1f}s  "
                f"{wav_path.stat().st_size/1024:.0f} KB")
        except Exception as e:      # 渲染失败不应该让整首曲子的识别结果作废
            log(f"    !! WAV 渲染失败: {type(e).__name__}: {e}")

    rep.elapsed_ms = (time.perf_counter() - t0) * 1000

    # --- 报告 -------------------------------------------------------------
    manifest = {
        "_meta": {
            "version": CHECKPOINT_VERSION,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline": {"detector": cfg.detector,
                         "binarize": cfg.preprocess.binarize,
                         "illum": cfg.preprocess.illum_method,
                         "fingerprint": pipeline_fingerprint(cfg)},
        },
        **asdict(rep),
        "ok": rep.ok,
    }
    atomic_write_text(out_dir / "manifest.json",
                      json.dumps(manifest, ensure_ascii=False, indent=2))

    # 简易日志，方便在板上无终端时事后排查
    with open(out_dir / "log.txt", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {work.name}: "
                f"{rep.n_pages} 页, {rep.n_measures} 小节, {rep.n_notes} 音符, "
                f"置信度 {rep.mean_confidence:.2f}, "
                f"跳过 {rep.n_pages_skipped} 页, {rep.elapsed_ms:.0f} ms\n")
        for pr in rep.pages:
            if pr.error:
                f.write(f"    !! page {pr.index}: {pr.error}\n")
            for it in pr.measure_issues:
                f.write(f"    ?? page {pr.index}: {it}\n")

    log(f"    -> {rep.n_measures} 小节 / {rep.n_notes} 音符 / "
        f"置信度 {rep.mean_confidence:.2f} / {rep.elapsed_ms:.0f} ms"
        + (f"  (跳过 {rep.n_pages_skipped} 页)" if rep.n_pages_skipped else "")
        + (f"  纠错 {rep.n_repaired} 处" if rep.n_repaired else "")
        + (f"  !! {rep.n_issues} 个残留问题" if rep.n_issues else ""))
    return rep


# ---------------------------------------------------------------------------
# 全部
# ---------------------------------------------------------------------------


def process_all(input_dir: Path, out_root: Path,
                cfg: PipelineConfig | None = None,
                resume: bool = True, save_page_images: bool = True,
                make_wav: bool = False, tempo_bpm: float = 100.0,
                works: list[Work] | None = None,
                log=print) -> list[WorkReport]:
    """扫描输入目录并处理全部曲子，最后写一个总索引 `index.json`。"""
    input_dir = Path(input_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    works = works if works is not None else discover_works(input_dir)
    if not works:
        log(f"输入目录里没有找到图片: {input_dir}")
        return []

    log(f"发现 {len(works)} 首曲子，共 {sum(w.n_pages for w in works)} 页")
    t0 = time.perf_counter()
    reports: list[WorkReport] = []
    for w in works:
        reports.append(process_work(w, out_root, cfg, resume=resume,
                                    save_page_images=save_page_images,
                                    make_wav=make_wav, tempo_bpm=tempo_bpm,
                                    log=log))

    # --- 总索引 -----------------------------------------------------------
    index_path = out_root / "index.json"
    prev: dict[str, dict] = {}
    if index_path.exists():
        try:
            old = json.loads(index_path.read_text(encoding="utf-8"))
            prev = {e["name"]: e for e in old.get("works", [])}
        except Exception:
            prev = {}
    for r in reports:
        prev[r.name] = {
            "name": r.name,
            "n_pages": r.n_pages,
            "n_measures": r.n_measures,
            "n_notes": r.n_notes,
            "mean_confidence": round(r.mean_confidence, 4),
            "ok": r.ok,
            "skipped_pages": r.n_pages_skipped,
            "measure_issues": r.n_issues,
            "elapsed_ms": round(r.elapsed_ms, 1),
            "dir": r.name,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    index = {
        "_meta": {"version": CHECKPOINT_VERSION,
                  "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "total_works": len(prev)},
        "works": sorted(prev.values(), key=lambda e: natural_key(e["name"])),
    }
    atomic_write_text(index_path, json.dumps(index, ensure_ascii=False, indent=2))

    total_pages = sum(r.n_pages for r in reports)
    done = sum(r.n_pages - r.n_pages_skipped for r in reports)
    log(f"\n完成：{len(reports)} 首 / {total_pages} 页（新识别 {done} 页），"
        f"用时 {time.perf_counter() - t0:.1f} s")
    log(f"输出目录：{out_root}")
    log(f"总索引  ：{index_path}")
    return reports


# ---------------------------------------------------------------------------
# 样例输入：没插 SD 卡也能验证整条工作流
# ---------------------------------------------------------------------------

#: 第二页用另一段旋律，验证"多页合并 + 小节号连续重编"
_PAGE2_MELODY_SPEC = [
    [("C5", "quarter"), ("B4", "quarter"), ("A4", "quarter"), ("G4", "quarter")],
    [("A4", "half"), ("F4", "half")],
    [("G4", "eighth"), ("G4", "eighth"), ("A4", "quarter"), ("B4", "half")],
]


def make_demo_input(root: Path, space: float = 21.0,
                    include_a4: bool = True) -> list[Work]:
    """在 `root` 下造一个假的 SD 卡输入目录，用于端到端验证。

    生成：
      * `work_two_pages/`  —— 两页（第 2 页带轻度退化，模拟真实拍摄）
      * `work_single.jpg`  —— 单页曲子
      * `work_a4_page.png` —— 一页 6 行谱表的"类 A4"乐谱

    最后这个不是为了好看：**真实乐谱一页有 8~12 行谱表**，
    而单行页面永远走不到滚动取景分支、也测不出"多行页面坐标错位"这类 bug。
    没有它，那些路径只在单元测试里被验证过，没在整页上跑过。
    """
    from .render import (DEFAULT_MELODY, DegradeOptions, NoteSpec, RenderOptions,
                         degrade, make_a4_like_page, render_score)

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    multi = root / "work_two_pages"
    multi.mkdir(parents=True, exist_ok=True)

    r1 = render_score(DEFAULT_MELODY, RenderOptions(space=space,
                                                    title="Two Page Work (p1)"))
    Image.fromarray(r1.image).save(multi / "page_1.png")

    mel2 = [[NoteSpec(p, d) for p, d in meas] for meas in _PAGE2_MELODY_SPEC]
    r2 = render_score(mel2, RenderOptions(space=space, title="Two Page Work (p2)"))
    img2 = degrade(r2.image, DegradeOptions(illumination=0.5, noise_sigma=5,
                                            blur_sigma=0.6, rotation_deg=0.8),
                   seed=3)
    Image.fromarray(img2).save(multi / "page_2.jpg", quality=92)

    r3 = render_score(DEFAULT_MELODY, RenderOptions(space=space, title="Single"))
    Image.fromarray(r3.image).save(root / "work_single.jpg", quality=95)

    if include_a4:
        a4 = make_a4_like_page(systems=6, space=space)
        Image.fromarray(a4.image).save(root / "work_a4_page.png")

    return discover_works(root)


def format_reports(reports: list[WorkReport]) -> str:
    """把处理报告排成人能看的表格。"""
    L: list[str] = []
    a = L.append
    a("=" * 100)
    a(f"{'曲名':<22} {'页':>3} {'小节':>5} {'音符':>5} {'置信度':>7} "
      f"{'跳过':>5} {'纠错':>5} {'残留问题':>8} {'耗时':>9}  状态")
    a("-" * 100)
    for r in reports:
        a(f"{r.name:<22} {r.n_pages:>3} {r.n_measures:>5} {r.n_notes:>5} "
          f"{r.mean_confidence:>7.2f} {r.n_pages_skipped:>5} {r.n_repaired:>5} "
          f"{r.n_issues:>8} {r.elapsed_ms:>7.0f}ms  {'OK' if r.ok else '有错误'}")
        for pr in r.pages:
            for rp in pr.repairs:
                a(f"    ~~ 第{pr.index}页 {rp}")
            if pr.error:
                a(f"    !! 第{pr.index}页: {pr.error}")
            for it in pr.measure_issues:
                a(f"    ?? 第{pr.index}页: {it}")
    a("=" * 100)
    a("~~ = 节拍约束已自动纠错（对应音符已标为可疑，置信度降到 0.6）")
    a("?? = 纠错后仍对不上拍号，建议人工复核该小节")
    return "\n".join(L)
