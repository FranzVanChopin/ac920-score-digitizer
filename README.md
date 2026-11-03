# AC920 乐谱数字化与播放系统

当前源码交接版：**DP_STABLE720_RESPONSE1**，固件 **a4_interactive_ui1_light2**。对应 2026-11-01 停工前已冻结、已验证的版本，2026-11-02 整理打包。

系统基于 AC920（Zynq UltraScale+ XCZU4EV、Cortex-A53 + FPGA）、树莓派官方 IMX477 HQ Camera 和单张 SD 卡。拍摄端采用裸机固件，识别与播放端采用 Ubuntu，二者通过安全重启和 FAT 文件交接；显示和音频使用原生 DP。两侧均为 **1280×720、60 Hz**。

## 当前功能

- IMX477 2028×1520 传感器预览；拍摄切至 4056×3040，固定裁为 **4056×2868**，保存 Image/Grey/Meta；纸顶朝预览右侧。
- 传感器白平衡、低增益曝光、色度降噪和五区域对焦辅助。
- S1 短按拍照、长按新建曲目；S2 长按进入/退出对焦、短按换区域；S3 安全切换 Ubuntu。
- Ubuntu：S1 长按切换识别/曲库，S2 短按选曲、长按识别或打开/播放；S1 短按在曲库调速，S3 返回拍摄。
- v16 共享网络、C++ 几何与节奏解码、多页流水线；结果为 MusicXML，FAT `Outputs/SS.musicxml` 为成品曲库，可在 Windows 复制/删除。
- 乐谱排版、逐小节高亮、钢琴音色和四档速度。识别运行在 PS/A53；当前版本没有部署 PL 神经网络加速。

## 目录结构

```
├── README.md            项目简介 + 复现步骤
├── REPRODUCE.md         五级复现路径与不可复现项
├── REPO_LAYOUT.md       新旧路径映射（重组记录）
├── src/                 设计源码：rtl/ 设计 RTL，omr/ 识别端，ps/ PS 侧部署软件，third_party/homr
├── sim/                 仿真：6 个 testbench + 两个 xsim 启动脚本
├── build/               构建：build_capture.ps1、verify_package.py、vivado/ 综合与实现脚本、
│                        reports/ 综合与实现报告、report_export_logs/ 导出日志
├── board/               上板工程：两个自包含工程包 + recognition/ 板端识别证据 + 部署快照 + 相机配置
├── data/                测试数据：samples/ 样张与参考结果
├── report/              设计报告与接口约定
└── skill/               技能包（预留）
```

FPGA 实现结果：WNS **0.817 ns**、TNS **0**、WHS **0.011 ns**、THS **0**，实现 DRC 无 Error / Critical Warning
（保留 8 条 Warning）。原报告在 [`build/reports/`](build/reports/)，身份与哈希见
[`build/reports/REPORT_IDENTITY.json`](build/reports/REPORT_IDENTITY.json)，说明见
[`build/reports/README.md`](build/reports/README.md)。板端识别证据与缺口说明见
[`board/recognition/EVIDENCE_20261103.md`](board/recognition/EVIDENCE_20261103.md)。

`build/`、`board/` 的实际生成物（`build/vivado_*/`、`board/*/build/`）由 `.gitignore` 排除，不入库。
两个 `board/camera_*` 工程保持整包不拆分（各自的 Vivado Tcl 与构建脚本依赖自身相对路径），
原因与逐条改动见 [`REPO_LAYOUT.md`](REPO_LAYOUT.md)。

## 源码入口

| 路径 | 内容 |
|---|---|
| `board/camera_interactive_20261030/software/camera_capture/` | **当前采集固件**及 C 测试 |
| `board/camera_onecard_20261029/` | 与当前固件匹配的纯 DP 硬件 Tcl、RTL 和 BSP 构建输入；不是当前应用入口 |
| `src/rtl/`、`build/vivado/` | 基础 RTL、仿真与 Vivado 工具；`build/vivado/` 还含历史 HDMI 方案，当前构建使用上一行的纯 DP 工程 |
| `src/ps/app/` | **当前已部署 Linux 程序**，从冻结备份中逐文件取出，包含全部最新显示、按键、FAT 曲库修复 |
| `src/ps/app/engine/` | 已部署、带阶段进度输出的 v16 引擎副本；更改安装路径时用 `patch_progress.make` 重新生成 |
| `src/omr/v16/` | Claude 合作完成的正式 v16 Python/C++ 源码、原始清单和当前小模型权重 |
| `src/omr/app/` | 已部署的识别支撑模块，供 `--bundle` 使用 |
| `src/third_party/homr/` | 实际使用的 Homr 0.7.0 Python 源码和原许可证，不含其大型模型 |
| `src/ps/systemd/`、`src/ps/desktop/`、`src/ps/config/` | 服务、自动启动和已去除 SD 卡身份的配置示例 |
| `requirements-board.lock.txt` | 冻结板端 Python 依赖版本记录 |
| `report/CAPTURE_INTERFACE.md` | 当前图像、方向、编号、按键及双系统接口约定 |
| `build/README.md` | 构建与部署路径说明 |
| `REPRODUCE.md` | **复现指南**：五级路径（校验 / 主机回归 / 识别端单页 / 固件 / 整机）与不可复现项 |
| `data/samples/` | **复现样张**：实拍一页 + 参考乐谱 + 已记录结果，说明见 `data/samples/README.md` |
| `PACKAGE_MANIFEST.json` | 每个交付文件的来源、大小、SHA-256 和 Unix 权限 |

这是用于 GitHub 的源码交接包。保留了当前实际使用的小模型权重；没有放入整卡镜像、系统环境、工具链、BSP、BOOT、构建缓存、照片和视频。唯一的例外是 `data/samples/page/` 里用于复现的一页样张（见下文）。依赖范围见 `external-assets.json`。原项目及开发板均未修改。

## 校验与上传

解压后，在本目录执行 `python build/verify_package.py`。应提示完整性与源码解析通过。

已安装 mido、verovio 等依赖时，可运行 `python -m unittest discover -s src/ps/tests -v` 做静音主机回归。测试导入路径已适配交接包，其余测试逻辑和参考夹具不变。

组长可将解压后的本目录作为仓库根目录提交，不要把整个 ZIP 当作源码仓库。`.gitignore` 已排除生成文件与实际卡配置。代码不依赖外层 ZIP 文件名。

已核对 43 份固件构建源码、28 份已部署 Linux 源码和 v16 原始清单；打包时没有重新综合 FPGA 或重新上板。验证范围见 `data/package_validation.json`。完整冻结备份仍单独保留在原项目中。

## 快速复现

**不需要开发板**就能验证识别链路。样张在 `data/samples/`，构成与基线见 [`data/samples/README.md`](data/samples/README.md)；
固件与整机路径见 [`REPRODUCE.md`](REPRODUCE.md)。

> **⚠️ 样张只能复现第 1 页。** 照片覆盖第 1 页的 **17 小节**（5 个系统，起始小节 1 / 5 / 9 / 13 / 16，底部被裁）；
> `data/samples/result/01.musicxml` 却是**整首 71 小节**的 album 合并结果。第 2 页及以后不在本仓库，
> 因此**整首结果无法逐页复算**，只能核对你产出的 MusicXML 与它的前若干小节。
> 这也是为什么第 3 步只用 `--start-measure 1` 看开头。

**第一步：校验完整性**（任何平台，无第三方依赖，**需 Python ≥ 3.11**，约 1 分钟）

```sh
python build/verify_package.py
```

期望输出形如 `PASS: 719 files intact; 271 Python sources parsed; v16 63/63 original assets verified.`
（前两个数随仓库增删文件而变；判定标准是 **`PASS` 且 `v16 63/63`**，不是具体数字。）
低于 3.11 会在语法校验报 `SyntaxError`（原因见 [`REPRODUCE.md`](REPRODUCE.md) 2.1）。

**第二步：跑一页样张**（Linux / WSL；需 `pip install -r requirements-board.lock.txt` 与 g++）

```sh
export PYTHONPATH="$PWD/src/third_party/homr:$PWD/src/omr/app"
python src/omr/v16/run_page.py --photo data/samples/page/sample01-1.bmp --out /tmp/out \
  --bundle "$PWD/src/omr/app" --threads 4 \
  --csolve --fast-homr --fast-relation --fast-redecode \
  --shared "$PWD/src/omr/v16/shared" \
  --relation-models "$PWD/src/omr/v16/shared/relation_models" --link-margin 0.5
```

产出 `/tmp/out/page.musicxml`。样张**已转正**（2868×4056，由板端传感器朝向逆时针旋转 90°），可直接使用，
不需要任何预处理。若改走 `album/` 模式，必须加 `--rotate none`，否则会被再转一次变成侧躺的图。

**第三步：与参考乐谱逐小节比对**

```sh
# .mxl 是压缩包，先解出可被 ET.parse 读取的 XML
python -c "import zipfile; open('/tmp/ref.xml','wb').write(zipfile.ZipFile('data/samples/source/vivaldis-spring-from-the-four-seasons-piano-transcription.mxl').read('score.xml'))"
python src/omr/app/compare_musicxml_events.py /tmp/ref.xml /tmp/out/page.musicxml \
  --start-measure 1 --out /tmp/cmp.json
```

只应关注**第 1–17 小节**——这页实际只覆盖 17 小节，之后的小节单页本来就没有。

## 许可证

本项目自行编写的代码和文档采用 **MIT**，全文见 [`LICENSE`](LICENSE)。

仓库内含若干第三方文件，继续适用各自原许可证，MIT 授权对它们不生效：

| 路径 | 许可证 |
|---|---|
| `src/third_party/homr/` | **AGPL-3.0**（Homr 0.7.0） |
| `board/*/software/camera_capture/imx477*` | **GPL-2.0-only**（Raspberry Pi IMX477 驱动派生） |
| `board/*/software/camera_capture/capture_color.c` 中的调校参数 | **BSD-2-Clause**（Raspberry Pi libcamera） |
| `board/*/software/camera_capture/ff_contiguous.c` 及 `ff.h`、`ffconf.h`、`ffunicode.c`、`diskio.h` | **FatFs**（ChaN） |
| `board/camera_interactive_20261030/software/camera_capture/studio_font.*` 字形数据 | **OFL-1.1**（绘制代码为 MIT） |

识别端（`src/ps/app/engine/`、`src/omr/app/`、`src/omr/v16/`）与 Homr 一同分发时，
须遵守 AGPL-3.0；其中由本项目编写的文件另行按 MIT 提供。

`src/rtl/vendor/dvi_tx/` 未收录未获授权的第三方 DVI 发送源码，获取方式见该目录 `README.md`。
逐路径许可范围、出处和再分发要求见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)，
许可证原文见 [`LICENSES/`](LICENSES/)。
