"""
图像预处理 —— 整个系统精度的天花板所在（"垃圾进，垃圾出"）。

这个模块里的每一步都刻意用**积分图 / 逐像素**的方式实现，
而不是直接调 OpenCV 的黑盒函数。原因有两个：

1. 这些算法后面要下沉到 FPGA（PL）用 HLS 实现，积分图 + 逐像素运算
   是天然可流水化的，直接用现成 boxFilter 反而不容易映射。
2. 每一步都保留中间结果，能定量看出"哪一步救回了多少精度"。

核心结论先写在前面
------------------
**光照归一化比二值化算法本身更重要。** 真实拍摄中，侧光、阴影、
纸张反光造成的背景不均是二值化失败的头号原因。先做除法归一化把背景抹平，
后面用什么二值化算法都行。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:  # OpenCV 只用于几何变换（旋转/连通域），不用于核心灰度运算
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def ensure_gray(img: np.ndarray) -> np.ndarray:
    """任意输入 -> uint8 单通道灰度图。"""
    a = np.asarray(img)
    if a.ndim == 3:
        if cv2 is not None:
            a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY if a.shape[2] >= 3 else cv2.COLOR_GRAY2BGR)
        else:
            a = a[..., :3].mean(axis=2)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def integral_image(a: np.ndarray) -> np.ndarray:
    """积分图。返回 (h+1, w+1)，ii[y, x] 是左上角 (h=y, w=x) 区域的和。

    这是后面所有局部统计量的基础，也是 HLS 里最容易流水化的结构。
    """
    a = a.astype(np.float64)
    return np.pad(np.cumsum(np.cumsum(a, axis=0), axis=1), ((1, 0), (1, 0)))


def box_sum(a: np.ndarray, radius: int) -> np.ndarray:
    """半径为 radius 的方形窗口求和（用积分图，O(1) per pixel）。"""
    ii = integral_image(a)
    h, w = a.shape
    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.clip(ys - radius, 0, h)[:, None]
    y1 = np.clip(ys + radius + 1, 0, h)[:, None]
    x0 = np.clip(xs - radius, 0, w)[None, :]
    x1 = np.clip(xs + radius + 1, 0, w)[None, :]
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def box_mean(a: np.ndarray, radius: int) -> np.ndarray:
    """方形窗口均值。"""
    h, w = a.shape
    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.clip(ys - radius, 0, h)[:, None]
    y1 = np.clip(ys + radius + 1, 0, h)[:, None]
    x0 = np.clip(xs - radius, 0, w)[None, :]
    x1 = np.clip(xs + radius + 1, 0, w)[None, :]
    area = (y1 - y0) * (x1 - x0)
    return box_sum(a, radius) / area


# ---------------------------------------------------------------------------
# 光照归一化
# ---------------------------------------------------------------------------


def normalize_illumination(gray: np.ndarray, radius: int = 25,
                           target: float = 255.0) -> np.ndarray:
    """除法归一化：估计背景，然后相除。

    这是最有效的一步。原理：局部背景亮度用大窗口均值近似，
    原图除以背景后，光照梯度被抹平，墨迹变成统一的深色。

    Parameters
    ----------
    radius
        背景估计窗口半径。经验值 ≈ 1 个 staff space。
        太小 -> 笔画周围出现光晕；太大 -> 跟不上光照变化。
    """
    g = gray.astype(np.float64)
    bg = box_mean(g, radius)
    bg = np.maximum(bg, 1.0)          # 防止除零
    out = g / bg * target
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize_illumination_morph(gray: np.ndarray, kernel: int = 41) -> np.ndarray:
    """形态学闭运算估计背景（效果略好，但硬件代价高得多）。

    保留这个实现是为了做对比实验：如果它相对 `normalize_illumination`
    提升不明显，就应该选积分图版本，因为后者在 FPGA 上便宜得多。
    """
    if cv2 is None:
        raise RuntimeError("形态学版本需要 OpenCV")
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    bg = np.maximum(bg.astype(np.float64), 1.0)
    return np.clip(gray.astype(np.float64) / bg * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 二值化
# ---------------------------------------------------------------------------


def sauvola(gray: np.ndarray, radius: int = 15, k: float = 0.34,
            r_dyn: float = 128.0) -> np.ndarray:
    """Sauvola 局部自适应二值化。返回 bool 数组，True = 墨迹。

    T = m * (1 + k * (s / R - 1))
        m 局部均值, s 局部标准差, R 动态范围(8bit 取 128), k 经验 0.2~0.5

    文档图像的首选：在空白区域会自适应降低阈值，不会把纸纹/噪声判成墨。
    """
    g = gray.astype(np.float64)
    m = box_mean(g, radius)
    m2 = box_mean(g * g, radius)
    var = np.maximum(m2 - m * m, 0.0)
    s = np.sqrt(var)
    t = m * (1.0 + k * (s / r_dyn - 1.0))
    return g < t


def bradley(gray: np.ndarray, radius: int = 15, t: float = 0.15) -> np.ndarray:
    """Bradley 自适应二值化。返回 bool 数组，True = 墨迹。

    T = m * (1 - t)。比 Sauvola 更简单（不需要方差），硬件更省，
    但对噪声更敏感。适合光照已经归一化得比较好的情况。
    """
    g = gray.astype(np.float64)
    m = box_mean(g, radius)
    return g < m * (1.0 - t)


def otsu_threshold(gray: np.ndarray) -> int:
    """全局 Otsu 阈值。作为基线对照 —— 光照不均时它必然崩，
    用它来凸显局部自适应方法的必要性。"""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 127
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return int(np.argmax(sigma_b))


def otsu(gray: np.ndarray) -> np.ndarray:
    """全局 Otsu。返回 bool 数组，True = 墨迹。"""
    return gray < otsu_threshold(gray)


# ---------------------------------------------------------------------------
# 倾斜校正
# ---------------------------------------------------------------------------


def _rotate(img: np.ndarray, angle_deg: float, border: int = 255) -> np.ndarray:
    """旋转（保持尺寸）。cv2 缺失时用 PIL 兜底。"""
    if abs(angle_deg) < 1e-9:
        return img
    if cv2 is not None:
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
        return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    from PIL import Image
    im = Image.fromarray(img)
    return np.asarray(im.rotate(angle_deg, resample=Image.BILINEAR, fillcolor=border))


def estimate_skew(binary: np.ndarray, max_angle: float = 5.0,
                  step: float = 0.4, refine_step: float = 0.05,
                  downscale: int = 4) -> float:
    """估计**需要施加的校正角**（度）。

    原理：谱线水平时，水平投影会出现极其尖锐的峰（五行墨迹集中）。
    一旦倾斜，墨迹在行方向被抹开，投影变糊。
    评分用投影一阶差分的平方和 —— 峰越尖，分越高。

    所以本函数返回的是"把图旋转多少度能让谱线变平"，
    调用方直接施加这个角即可（**不要取负**，取负会把倾斜放大一倍）。

    采用粗搜 + 细搜两段：粗搜 ±max_angle、步长 step；再在最优点附近
    以 refine_step 细搜。单靠粗搜的 0.2° 残差会让 1580 px 宽的谱线
    在竖直方向散开 ~5 px，行投影糊掉，谱表直接检测不到。
    """
    img = (binary.astype(np.uint8) * 255)
    if downscale > 1:
        if cv2 is not None:
            img = cv2.resize(img, None, fx=1.0 / downscale, fy=1.0 / downscale,
                             interpolation=cv2.INTER_AREA)
        else:
            from PIL import Image
            h, w = img.shape
            img = np.asarray(Image.fromarray(img).resize((max(1, w // downscale),
                                                          max(1, h // downscale))))

    def score(a: float) -> float:
        rot = _rotate(img, float(a), border=0)
        proj = rot.sum(axis=1).astype(np.float64)
        return float(((proj[1:] - proj[:-1]) ** 2).sum())

    coarse = np.arange(-max_angle, max_angle + 1e-9, step)
    best = max(coarse, key=score)
    fine = np.arange(best - step, best + step + 1e-9, refine_step)
    best = max(fine, key=score)
    return float(best)


def deskew(binary: np.ndarray, angle_deg: float | None = None,
           max_angle: float = 5.0) -> tuple[np.ndarray, float]:
    """倾斜校正。

    ⚠️ 本函数**只处理二值图**。`preprocess()` 里不再调用它 —— 因为那里必须
    让 gray / normalized / binary 三个输出共用同一个旋转角，而本函数拿不到
    另外两个数组。保持独立是为了给"只关心二值图"的调用方留一个简单入口。

    Returns
    -------
    (校正后的 bool 数组, 实际施加的角度)
        `angle_deg` 为 None 时自动估计。
    """
    a = estimate_skew(binary, max_angle=max_angle) if angle_deg is None else angle_deg
    if abs(a) < 1e-9:
        return binary, a
    rot = _rotate((binary.astype(np.uint8) * 255), a, border=0)
    return rot > 127, a


# ---------------------------------------------------------------------------
# 去噪
# ---------------------------------------------------------------------------


def remove_speckle(binary: np.ndarray, min_area: int = 4) -> np.ndarray:
    """去掉面积过小的连通域（噪点/纸纹）。"""
    if cv2 is None:
        return binary
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


# ---------------------------------------------------------------------------
# 纸张边界：把纸面外的背景抹掉
# ---------------------------------------------------------------------------
#
#  ★ 为什么需要这一步（是实测逼出来的，不是"看起来应该加"）：
#  真实拍照时纸面之外一定有别的东西（桌子、手、地面）。合成退化集里
#  从来没有这一项，所以之前一直没暴露。
#  `tools/probe_perspective.py` 把几何畸变和纸外背景**分开**测之后发现：
#      同样的 0.02 梯形畸变，边界填白 → 100%，边界填深灰 → 37.5%
#  也就是说**"取景"比"几何"更毒**。
#
#  机理：Sauvola 是**局部**阈值。
#   * 一大片均匀的深色 → 局部均值≈自身、方差小 → 判成"背景"，无害
#   * 但**紧贴白纸的细深色条**局部方差很大 → 判成"墨迹"
#  而"纸边"天然就是一条细深色条 —— 它被判成墨迹之后，在**整行墨迹直方图**
#  里表现为一条横贯全图的假谱线，直接破坏谱表检测与阅读顺序。
#  实测顶部 40 行的墨迹率：小梯形 1.5% vs 大梯形 0.1%，与准确率反向。
#
#  所以这里做的是"只抹边界、不动几何"，好处有三个：
#   1. 能**单独**验证边界问题的贡献（不和矫正混在一起）
#   2. 纯 PS 侧、不占 PL 资源、不需要 Vivado
#   3. 即使将来不做透视矫正，这一步本身就有用


def find_paper_quad(gray: np.ndarray,
                    min_area_ratio: float = 0.35) -> np.ndarray | None:
    """找出纸面（亮区）的四边形轮廓。返回 4x2 的 float32 角点，找不到返回 None。

    ★ 什么时候返回 None（这很重要，决定了这一步对老用例是否是空操作）：
     * OpenCV 不可用
     * 亮区面积占比 < `min_area_ratio`（页面根本没拍全，或整幅就是暗的）
     * 凸包近似不出 4 个角（纸边被遮挡等）
     * **亮区已经铺满整幅图**（`fill_ratio` 超过 `already_full` 阈值）——
       干净合成图就是这种情况。此时"抹边界"本来就是空操作，
       显式返回 None 可以让调用方少做一次无用的全图乘法，
       也保证**对现有基线逐像素无影响**。
    """
    if cv2 is None:
        return None
    h, w = gray.shape[:2]
    # 纸是亮的、背景是暗的，所以"纸"= 亮区。用 Otsu 自动找分界，
    # 不写死阈值 —— 桌面有亮有暗，写死阈值一定会翻车。
    thr = otsu_threshold(gray)
    bright = (gray > thr).astype(np.uint8)
    if bright.sum() < min_area_ratio * h * w:
        return None

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n <= 1:
        return None
    # 取最大的亮区（纸面）。不要把所有亮区并起来 —— 桌面上的反光、
    # 相邻的白纸都会被并进来，边界就跑到画面外去了。
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == biggest).astype(np.uint8)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    quad = None
    # epsilon 从小到大试：太小会留成很多边形，太大会把纸角削掉。
    # 先小后大能让"接近矩形"的页面拿到最贴合的四边形。
    for eps in (0.02, 0.03, 0.05, 0.08, 0.12):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            quad = ap.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        return None

    # ★ 已经铺满整幅图 → 没有边界可抹，返回 None（对干净合成图就是空操作）
    fill = cv2.contourArea(quad.reshape(-1, 1, 2)) / float(h * w)
    if fill > 0.995:
        return None
    return quad


def mask_outside_paper(gray: np.ndarray, quad: np.ndarray | None,
                       fill: int = 255, pad: int = 2) -> np.ndarray:
    """把四边形之外的像素抹成 `fill`（纸白）。

    `pad` 是向外扩的像素数：纸边在真实照片里有一圈**阴影/压痕**，
    正好落在边界上；不扩大一点的话那圈阴影会留在纸内，仍然是一条
    细深色条，等于白做。扩 2 px 在 300DPI 下约 0.17mm，足够吃掉压痕，
    又不会切掉谱表（谱表离纸边有几十像素的页边距）。
    """
    if quad is None or cv2 is None:
        return gray
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 1)
    if pad > 0:
        k = np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)
        mask = cv2.dilate(mask, k)
    out = gray.copy()
    out[mask == 0] = fill
    return out


# ---------------------------------------------------------------------------
# 总入口
# ---------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    """预处理参数。默认值是在 tools/tune.py 的网格搜索 + cli sweep 的
    退化扫描上选出来的（见 README 的基线表）。

    ⚠️ 关于 `illum_method` 的默认值，实测结论如下（合成退化集）：

        method   平均音符级准确率   模糊σ=1.5   阴影0.8
        morph        98.3%          93.3%      100%
        none         98.3%          93.3%      100%
        box          92.2%          26.7%      100%

    * `box`（积分图均值估背景）在**笔画被模糊加粗**时会崩：大窗口均值被
      粗笔画污染，相除后在笔画边缘产生光晕，二值化断裂。
      这是均值估背景的固有缺陷，不是参数没调好（半径搜过 15~95 都一样）。
    * `morph`（形态学闭运算估背景）取的是局部**最大值**而不是均值，
      不受笔画污染，因此在模糊下依然稳定。
    * `none` 在合成退化集上与 morph 打平 —— 因为 Sauvola 本身就是局部
      自适应的，能自己跟上亮度变化。
      **但真实照片的阴影边界比合成退化更恶劣，所以 P1 先默认 morph（更稳），
      `none` 作为 FPGA 上的"零成本"备选，等 P2 用真实照片复核后再定。**

    morph 的硬件代价：矩形结构元是可分离的，闭运算 = 水平滑动最小值 +
    垂直滑动最小值，HLS 里用两级滑动窗口即可，并非不可实现。
    """

    illum_radius: int = 25
    illum_method: str = "morph"        # "morph"(默认, 稳) | "box"(积分图) | "none"(零成本)
    binarize: str = "sauvola"          # "sauvola" | "bradley" | "otsu"
    sauvola_radius: int = 15
    sauvola_k: float = 0.34
    bradley_radius: int = 15
    bradley_t: float = 0.15
    do_deskew: bool = True
    max_skew: float = 5.0
    min_speckle_area: int = 4
    #: ★ 抹掉纸面之外的背景（见 find_paper_quad 的说明）。
    #:
    #: ⚠️ **默认 False，而且必须保持 False**：
    #:   `fpga/tools/verify.py` 拿 `PreprocessConfig` 去和 C++ 逐像素对拍，
    #:   而 C++ 侧（正确地说，PL 侧）**没有实现这一步** —— 它是全局操作
    #:   （要先知道整张纸才谈得上"纸外"），属于 PS 侧或流水线最前端。
    #:   一旦默认打开，Python 结果会比 C++ 多挨一次全图改写，
    #:   对拍立刻满地红，而且会看起来像"C++ 错了"。
    #:   要打开就显式传 `crop_to_paper=True`。
    crop_to_paper: bool = False


@dataclass
class PreprocessResult:
    """每一步的中间结果都留着 —— 调参和排错全靠它。"""

    gray: np.ndarray
    normalized: np.ndarray
    binary: np.ndarray            # bool，True = 墨迹
    skew_angle: float = 0.0
    config: PreprocessConfig = field(default_factory=PreprocessConfig)

    def display(self) -> np.ndarray:
        """转成"白底黑字"的 uint8 图，方便保存查看。"""
        return np.where(self.binary, 0, 255).astype(np.uint8)


def preprocess(img: np.ndarray, cfg: PreprocessConfig | None = None) -> PreprocessResult:
    """完整预处理链：灰度 -> 光照归一化 -> 二值化 -> 倾斜校正 -> 去噪。

    顺序很重要：**先归一化光照，再二值化**。反过来做的话，
    自适应二值化要同时对付光照和笔迹，参数会非常难调。
    """
    cfg = cfg or PreprocessConfig()
    gray = ensure_gray(img)

    # ★ 纸张边界必须在**光照归一化之前**抹掉。
    #   放到后面做没用：深色背景已经参与了形态学/窗口的背景估计，
    #   把边界"擦白"只是把污染源盖住，背景估计仍然是错的。
    #   这一步也是"整页需要一次全图操作"的唯一环节 —— 它得先看全图。
    if cfg.crop_to_paper:
        gray = mask_outside_paper(gray, find_paper_quad(gray))

    if cfg.illum_method == "box":
        norm = normalize_illumination(gray, cfg.illum_radius)
    elif cfg.illum_method == "morph":
        norm = normalize_illumination_morph(gray, cfg.illum_radius * 2 + 1)
    else:
        norm = gray

    if cfg.binarize == "sauvola":
        binary = sauvola(norm, cfg.sauvola_radius, cfg.sauvola_k)
    elif cfg.binarize == "bradley":
        binary = bradley(norm, cfg.bradley_radius, cfg.bradley_t)
    else:
        binary = otsu(norm)

    angle = 0.0
    if cfg.do_deskew:
        # ★ 三个输出必须施加**同一个**旋转。
        #   踩过的坑：原来直接调 `deskew(binary)`，而它内部只处理二值图 ——
        #   于是 binary 被旋正、gray / normalized 留在原坐标系，同一个 (y, x)
        #   下标在三个数组里指的不是同一个物理位置（0.75° 在页面边缘就错位
        #   十几像素）。排错时拿旋正后的行号去索引未旋正的 normalized，
        #   测出来的"谱线灰度对比度 ≈ 0"完全是错位的产物，差点据此得出
        #   "照片上谱线本身就淡"这个错误结论。
        #   这里不调 `deskew()` 是因为它只返回二值图；角度只估计一次，
        #   三个数组共用同一个角度。对 binary 而言结果与原来**逐位相同**
        #   （同样是 `_rotate(binary*255, a, border=0) > 127`）。
        #   另外注意：旋转是**全局、不可流式**操作，PL 侧目前不做
        #   （见 fpga/tools/verify.py 里 do_deskew=False 的说明），
        #   所以这一步是 PS 的职责。
        angle = estimate_skew(binary, max_angle=cfg.max_skew)
        if abs(angle) >= 1e-9:
            binary = _rotate(binary.astype(np.uint8) * 255, angle, border=0) > 127
            norm = _rotate(norm, angle, border=255)
            gray = _rotate(gray, angle, border=255)

    if cfg.min_speckle_area > 0:
        binary = remove_speckle(binary, cfg.min_speckle_area)

    return PreprocessResult(gray=gray, normalized=norm, binary=binary,
                            skew_angle=angle, config=cfg)
