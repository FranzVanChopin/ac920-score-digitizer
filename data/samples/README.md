# 样张：拍一页乐谱 → 出 MusicXML

本目录提供一套**可端到端跑通**的输入、参考源和已记录结果，用于 C 级复现（见 [`../../REPRODUCE.md`](../../REPRODUCE.md)）。
不需要 AC920 开发板，一台 x86_64 或 arm64 Linux 就够。

```
data/samples/
├── page/sample01-1.bmp        输入：第 1 页实拍，已正立，2868×4056，24 位 BGR，34.9 MB
├── source/….mxl               参考源：整首乐谱（MuseScore 压缩 MusicXML）
└── result/01.musicxml         已记录结果：整首 71 小节的识别输出
```

| 文件 | 是什么 | 关键事实 |
|---|---|---|
| `page/sample01-1.bmp` | 用 AC920 + IMX477 拍的**第 1 页** | 2868×4056、24 位、**已正立**（由传感器朝向逆时针转 90° 得到） |
| `source/vivaldis-spring-from-the-four-seasons-piano-transcription.mxl` | 这页所印乐谱的**参考源** | Vivaldi《春》钢琴改编，**79 小节**，MuseScore 3.4.2 导出，内含 `score.xml` |
| `result/01.musicxml` | 该曲（曲目 `01`）**全部页**的已记录识别结果 | **71 小节**、1901 音符、divisions 480，album 模式输出 |

## ⚠️ 本样张只能复现第 1 页

这是最重要的一条，先看清楚再动手：

- 照片只覆盖**第 1 页的 17 小节**：图上有 5 个系统，起始小节为 **1 / 5 / 9 / 13 / 16**，画面底部被裁掉。
- `result/01.musicxml` 是 **整首 71 小节**，由 album 模式把该曲**所有页**合并而成。
- **第 2 页及以后的照片不在本仓库**，所以整首结果**无法逐页复算**。

因此：

| 你跑的命令 | 产出 | 能不能与 `result/01.musicxml` 对比 |
|---|---|---|
| 单页（第 2 节） | 第 1 页 17 小节 | ✅ 只能对得上**前缀**，不要整篇 `diff` |
| album 模式（第 3 节） | 同上（只有一页可识别） | ✅ 同上 |
| 想要整首 | — | ❌ 缺页，复现不了 |

**判定标准就一句话：你跑出来的 MusicXML 应当是 `result/01.musicxml` 的前若干小节，并且与参考源第 1 小节起逐小节对齐。** 具体方法见第 4 节。

## 1. 环境（一次性）

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-board.lock.txt        # 板端锁定版本，x86 也可
```

编译 4 个 C++ 加速库（首次运行 `run_page.py` 也会自动编）：

```sh
cd src/omr/v16
sh cdec/build_cdec.sh && sh hacc/build_hacc.sh && sh racc/build_racc.sh && sh sacc/build_pilresize.sh
cd ../..
```

让 Homr 可导入（**最容易漏的一步**）：

```sh
export PYTHONPATH="$PWD/src/third_party/homr:$PWD/src/omr/app"
```

## 2. 单页复现（推荐先跑这个）

图片**已经正立**，直接用，不需要任何预处理：

```sh
python src/omr/v16/run_page.py \
  --photo data/samples/page/sample01-1.bmp \
  --out   /tmp/out_sample01 \
  --bundle "$PWD/src/omr/app" \
  --threads 4 --csolve --fast-homr --fast-relation --fast-redecode \
  --shared "$PWD/src/omr/v16/shared" \
  --relation-models "$PWD/src/omr/v16/shared/relation_models" \
  --link-margin 0.5
```

产出在 `/tmp/out_sample01/`：`page.musicxml` 是结果，`page.geometry.json` 等是中间量。

只想先确认代码路径通不通（几十秒、不跑网络解码）：

```sh
python src/omr/v16/run_page.py --photo data/samples/page/sample01-1.bmp \
  --out /tmp/out_seg --bundle "$PWD/src/omr/app" --only-segment
```

成功标志：输出目录出现 `page.geometry.json`，其中能查到本页的谱表（system）几何（预期 5 个）。

## 3. 变体：用 album 模式跑

album 模式默认 `--rotate ccw`（假设输入是板端传感器朝向）。**本样张已经转正，必须改成 `--rotate none`**，
否则会被再转一次、变成侧躺的图。

```sh
mkdir -p /tmp/card/Grey /tmp/card/Meta
cp data/samples/page/sample01-1.bmp /tmp/card/Grey/01-01.bmp
printf '{"complete": true}' > /tmp/card/Meta/01-01.json

APP="$PWD/src/omr/app" PY=python \
  sh src/omr/v16/album/run_album.sh /tmp/card /tmp/album_out --rotate none
```

产出 `/tmp/album_out/01.musicxml`。这个模式的价值在于常驻流水线、逐页续接和多页合并，
只有一页时体现不出来——补齐第 2 页以后才有意义。

## 4. 怎么判定"跑对了"

用仓库自带的比较工具。先把 `.mxl` 解出可被 `ET.parse` 直接读取的 XML
（`compare_musicxml_events.py` 不能直接读压缩包）：

```sh
python -c "import zipfile; \
open('/tmp/ref.musicxml','wb').write(zipfile.ZipFile('data/samples/source/vivaldis-spring-from-the-four-seasons-piano-transcription.mxl').read('score.xml'))"

python src/omr/app/compare_musicxml_events.py /tmp/ref.musicxml /tmp/out_sample01/page.musicxml \
  --start-measure 1 --out /tmp/compare_page1.json
```

报告是**逐小节**的（`measures` 数组）。判读方式：

- **只看第 1–20 小节**。这是这页覆盖的范围，之后的小节单页本来就没有，全是 0 属于正常。
- 看 `exact_note_events / reference_notes`。逐小节接近 1.0 说明识别正确；
  某小节骤降通常是那行谱的连音线、加线或临时记号没认对。
- `candidate_notes` 为 0 的小节，就是漏掉的小节。

**参考基线（实测，供你判断量级）：** 用同一套工具比较
`result/01.musicxml`（整首 71 小节）与参考源（整首 79 小节）：

| 指标 | 数值 |
|---|---|
| 参考源音符 / 结果音符 | 1880 / 1816 |
| 完全一致音符事件 | 1425 |
| 音符精确率 / 召回率 | **0.785 / 0.758** |
| 第 1–4 小节 | 4/4、22/22、22/22、27/27 **全部一致** |

这是**整首口径**，单页输出没有多页合并的补偿，数字不会相同。
不要拿这两个数当单页的合格线，重点是**第 1 小节起逐小节能对上**。

## 5. 跑不出结果时

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: homr` | `PYTHONPATH` 没指到 `src/third_party/homr` |
| 结果乱、谱号找不到 | 检查朝向：本图已正立，不要再旋转；用 album 模式时必须 `--rotate none` |
| `libdecoder.so` 加载失败 | 4 个 `.so` 没编，或 g++ 不支持 C++17 |
| `OUT exists` | 输出目录必须不存在 |
| 只认出部分小节 | 正常。这页底部被裁，第 16 小节之后不完整 |
| arm64 与 x86 结果不完全相同 | **预期行为**。两侧浮点舍入不同，个别符头会翻转 |

## 6. 关于这张图本身

原图由板端写出时是**传感器朝向**（纸顶在右边缘，4056×2868），本仓库里的版本已逆时针旋转 90° 成
2868×4056 的正立图，像素无损、字节数不变（34,897,878）。这样 `run_page.py` 可以直接吃，
不需要每个人再写一遍旋转代码。

如果你要做与板端原始采集逐字节对照的实验，转回去即可：

```sh
python -c "from PIL import Image; Image.open('data/samples/page/sample01-1.bmp').rotate(-90, expand=True).save('/tmp/sensor_orientation.bmp')"
```

只想减小体积的话（像素无损，识别结果不变）：

```sh
python -c "from PIL import Image; Image.open('data/samples/page/sample01-1.bmp').save('/tmp/sample01-1.png', optimize=True)"
```

## 7. 想复现整首，还需要什么

把同一首的**其余页**按板端布局补进来，就能验证目前无法复算的两件事：

```
data/samples/card/
├── Grey/01-01.bmp   ← 现在这张（已正立时用 --rotate none）
├── Grey/01-02.bmp
├── Meta/01-01.json  ← {"complete": true}
├── Meta/01-02.json
└── …
```

补齐后 album 模式的多页合并、按 Grey sha256 跳过已完成页的续接、以及 `01.musicxml`
为什么是 71/79 小节，才都能对上。

## 8. 来源与许可（**发布前需补全**）

- 乐曲本身：Antonio Vivaldi《四季·春》，1725 年作品，**已进入公有领域**。
- `source/*.mxl` 的**制谱/改编**来自第三方（MuseScore 导出，标注 `MuseScore 3.4.2`、`encoding-date 2020-05-10`）。
  乐谱的**排版**可能另有改编者权利。公开发布前请补上：

  ```
  制谱来源：<原始链接 / 作者 / 许可证>
  ```

  若无法确认许可，最稳妥的做法是换成自己重新排版的公有领域乐谱，或只保留 `page/` 与 `result/`。
- `result/01.musicxml` 由本项目识别产出，随本仓库按 MIT 提供（与识别端一致，见根目录 `THIRD_PARTY_NOTICES.md`）。
