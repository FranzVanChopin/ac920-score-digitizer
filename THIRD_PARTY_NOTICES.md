# 许可范围与第三方来源

本仓库**不是**单一许可证。根目录 [`LICENSE`](LICENSE) 是 MIT，只覆盖由本项目自行编写的文件；
下表列出的第三方文件继续适用其原有许可证，MIT 授权对它们**不生效**，本项目也无权为它们重新授权。

完整许可证原文见 [`LICENSES/`](LICENSES/) 目录，或各第三方文件自带的版权头。

## 1. 许可范围总表

| 路径 | 许可证 | 说明 |
|---|---|---|
| 除下表所列以外的全部文件 | **MIT**（`LICENSE`） | 本项目自行编写 |
| `src/third_party/homr/**` | **AGPL-3.0** | Homr 0.7.0 原样收录，原文见同目录 `LICENSE` |
| `board/*/software/camera_capture/imx477.c`<br>`imx477.h`、`imx477_registers.h`<br>`COPYING.imx477`、`IMX477_SOURCE.txt`<br>`IMX477_COLOR_REGISTERS.txt` | **GPL-2.0-only** | 派生自 Raspberry Pi 官方 IMX477 驱动 |
| `board/*/software/camera_capture/capture_color.c`<br>（仅其中白平衡与色彩矩阵参数） | **BSD-2-Clause** | 派生自 libcamera 的 IMX477 调校数据 |
| `board/*/software/camera_capture/ff_contiguous.c`<br>`tests/fatfs_fixture/ff.h`<br>`tests/fatfs_real_fixture/ff.h`、`ffconf.h`<br>`ffunicode.c`、`diskio.h` | **FatFs**（ChaN，一条款 BSD 式） | 保留原版权与条件，见 `LICENSES/FatFs.txt` |
| `board/camera_interactive_20261030/software/camera_capture/studio_font.c`、`studio_font.h` | 字形数据 **OFL-1.1**；绘制代码 **MIT** | 见同目录 `STUDIO_FONT_SOURCE.txt` |
| `src/rtl/vendor/dvi_tx/**` | **未收录** | 见该目录 `README.md` |

`*` 指 `camera_interactive_20261030` 与 `camera_onecard_20261029` 两个目录，两者的相机采集固件大部分文件相同。

## 2. 逐项说明

### 2.1 Homr（AGPL-3.0）

`src/third_party/homr/` 是板端实际使用的 Homr **0.7.0** 源码，原项目
<https://github.com/liebharc/homr>，作者 Christian Liebhardt，许可证 **AGPL-3.0**。
`METADATA` 保留了 PyPI 元数据，`LICENSE` 保留了许可全文。

该目录中的文件未作任何修改，也未被重新署名。

### 2.2 IMX477 驱动适配（GPL-2.0-only）

`imx477_registers.h` 中的寄存器序列逐字节取自 Raspberry Pi 官方 Linux 驱动：

- 仓库 <https://github.com/raspberrypi/linux>
- 提交 `5bd564aaa8dc4b1b8ed11733c19ec8ece63a72cc`
- 路径 `drivers/media/i2c/imx477.c`
- 上游 SPDX：`GPL-2.0`（仅第 2 版）
- 版权：Copyright (C) 2020 Raspberry Pi (Trading) Ltd.

`imx477.c`、`imx477.h` 携带同一适配，同样按 **GPL-2.0-only** 提供。
`IMX477_SOURCE.txt` 记录了完整来源、下载文件 SHA-256 和保留的寄存器表清单，
`COPYING.imx477` 是上游许可原文。

固件镜像在链接这些文件后分发时，须同时提供对应源代码并遵守 GPL-2.0-only。

### 2.3 libcamera 调校参数（BSD-2-Clause）

`capture_color.c` 的初始白平衡增益与色彩矩阵取值来自
<https://github.com/raspberrypi/libcamera> 在提交
`6c1dd9d55573010f710c9e190a73e7e76f0d9432` 的 `src/ipa/rpi/vc4/data/imx477.json`，
该调校数据按 BSD-2-Clause 授权，版权为 Copyright 2019-2020 Raspberry Pi Ltd。
`COLOR_SOURCE.txt` 保留了原始出处、SHA-256 和 BSD-2-Clause 全文。
未修改的上游参考文件不属于本仓库（`build/` 已在 `.gitignore` 中排除）。

### 2.4 FatFs（ChaN）

`ff_contiguous.c`、`ff.h`、`ffconf.h`、`ffunicode.c`、`diskio.h` 来自
FatFs - Generic FAT Filesystem Module（R0.15 及 R0.13 的 Unicode 模块），
Copyright (C) 2022-2023, ChaN, all right reserved。许可要求保留版权声明、
条件和免责声明，原文见 `LICENSES/FatFs.txt`。

`tests/*/fatfs_real_fixture/` 下的 `xil*.h`、`xparameters.h`、`xstatus.h`
是 AMD/Xilinx 工具生成头文件的测试桩，仅为让主机测试可编译。

### 2.5 Studio UI 字体（OFL-1.1）

`studio_font.c`、`studio_font.h` 是 Noto Sans SC 的静态点阵子集。
字形数据按 **SIL Open Font License 1.1** 提供（`STUDIO_FONT_OFL.txt`），
版权 Copyright 2014-2021 Adobe，保留字体名 'Source'；修改后的子集已按 OFL 要求
改用 'Studio UI' 作为主名。C 绘制代码与 `tools/generate_studio_font.py`
为本项目编写，按 MIT 提供。细节见 `STUDIO_FONT_SOURCE.txt`。

### 2.6 DVI/TMDS 发送 RTL（未收录）

`src/rtl/vendor/dvi_tx/` 原有四个 Verilog 文件，文件头标明作者为
武汉芯路恒科技有限公司（xiaomeige.taobao.com / corecourse.cn），
随附资料中没有给出任何开源许可声明。为在本仓库的 MIT 许可下公开发布，
这些文件未纳入版本库，获取方式和构建影响见该目录 `README.md`。

### 2.7 不随仓库分发的组件

按各自原许可自行获取，本仓库不包含，也不对其授权：

- AMD/Xilinx Vivado、Vitis 工具链、BSP、IP 生成结果、《开发者》板卡预设
- Ubuntu rootfs、内核、设备树、BOOT.BIN、bitstream、XSA
- 第三方 Python 解释器与虚拟环境、`requirements-board.lock.txt` 所列二进制依赖
- 钢琴音色 `FluidR3_GM.sf2`（Ubuntu `fluid-soundfont-gm`，148 MB）
- Homr 上游大型 Transformer 模型（见 `external-assets.json`）

## 3. 识别端与 AGPL-3.0 的关系（重要）

`src/ps/app/engine/run_page_ui.py` 直接 `import homr.main`、`homr.staff_detection`、
`homr.segmentation.inference_segnet`，并替换其中的内部函数；
`src/omr/app/**` 也大量使用 `from homr.* import ...`；
`src/omr/v16/**` 与 `src/ps/app/**` 经由上述模块完成识别。

因此这些目录在**与 Homr 一同分发时**，构成一个受 AGPL-3.0 约束的整体：
任何人再分发，或通过网络向用户提供该组合的服务，都须遵守 AGPL-3.0 的条款，
其中包括向接收者提供完整对应源代码的义务。这一点不因根目录是 MIT 而改变。

同时，本项目对其中**由自己编写**的文件另行按 MIT 提供，便于在不涉及 Homr 的场合单独复用
（MIT 可被并入 AGPL 作品，反向不成立）。文件归属见 `PACKAGE_MANIFEST.json` 的 `origin` 字段，
以及各目录的 `provenance.json`。

`src/omr/v16/weights/`、`src/omr/v16/shared/` 中的模型权重由本项目训练产出，
单独取用时按 MIT 提供。

## 4. 贡献与再分发

- 向本仓库提交贡献，即表示同意你的贡献按 MIT 授权；但涉及第 3 节所列依赖 Homr 的目录时，
  该目录作为整体仍受 AGPL-3.0 约束。
- 再分发时请同时保留：根目录 `LICENSE`、本文件、`LICENSES/` 目录，
  以及各第三方文件原有的版权头和许可声明。
- 不要删除或替换第三方文件的许可声明，也不要为本仓库整体声明一个与上表冲突的统一许可证。

## 5. 免责

本项目不对第三方文件的许可状态作出任何保证。上表用于说明现状与出处，
若与第三方原许可证的表述冲突，以原许可证为准。把本项目用于商业用途前，
请自行核实 Homr（AGPL-3.0 与网络服务条款）和 IMX477 适配（GPL-2.0-only）的义务范围。
