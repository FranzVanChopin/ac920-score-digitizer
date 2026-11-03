# 复现指南

本仓库是**源码与小模型交接包**，不是可启动镜像。复现分五级，越往下要求越高，
先说结论：

| 级别 | 内容 | 需要什么 | 可行性 |
|---|---|---|---|
| **A** | 完整性与语法校验 | 任何平台 + **Python ≥ 3.11** | ✅ 开箱即用，约 1 分钟 |
| **B** | 静音主机回归测试 | Linux/macOS/WSL + pip 依赖；C 编译器 | ✅ 可完整跑通 |
| **C** | 识别端单页：照片 → MusicXML | x86_64 或 arm64 Linux + g++，**不需要 AC920** | ✅ 可跑通，样张已随仓库提供（`data/samples/`） |
| **D** | 采集固件构建 → `BOOT.BIN` | Windows + Vivado/Vitis **2024.2** | ⚠️ 可编译，上板需 AC920 实机 |
| **E** | 整机（拍照→识别→DP→播放） | 原始 AC920 Ubuntu SD 卡 + 512 GB 整卡 | ❌ 不构成可复现目标，见第 6 节 |

本包**没有**预编译的 `BOOT.BIN`、bitstream、XSA、整卡镜像、工具链和 148 MB 音色；
输入样张只放了复现所需的**一页**。这些都是刻意排除的，范围见根目录 `external-assets.json`。

---

## 1. 前置条件

| 组件 | 版本 | 说明 |
|---|---|---|
| 板卡 | AC920（Zynq UltraScale+ **XCZU4EV**，`xczu4ev-sfvc784-1L-i`） | 仅 D/E 级需要 |
| 相机 | 树莓派官方 IMX477 HQ Camera + 单张 SD 卡 | 仅 E 级需要 |
| Vivado / Vitis | **2024.2** | 仅 D 级；BSP、FSBL、PMU 由脚本用 `xsct` 生成 |
| Python | **3.12**（板端）；主机 **≥ 3.11** | 依赖版本见 `requirements-board.lock.txt`；低于 3.11 会在 A 级语法校验直接失败，见 2.1 |
| 识别端依赖 | `numpy` `onnxruntime` `opencv-python-headless` `pillow` `omegaconf` `pyclipper` `shapely` `mido` `verovio` 等 | 全量清单在该 lock 文件 |
| C/C++ | `g++`（C++17） | C 级编译 4 个 `.so`；B 级 C 测试用任意 C11 编译器 |

---

## 2. A 级：开箱校验（不需要任何依赖）

```sh
git clone <仓库地址> && cd <仓库名>
python build/verify_package.py
```

只用 Python 标准库，离线只读，不碰设备、不播放音频。它会做三件事：

1. 逐文件比对 `PACKAGE_MANIFEST.json` 里的字节数与 **SHA-256**；
2. 对全部 `.py` 文件（当前 271 个）做 `ast.parse` 语法校验；
3. 复核 `src/omr/v16/MANIFEST.json` 的 63 项原始资产。

**期望输出**（第一个数 = 清单条目数，当前 719）：

```
PASS: 719 files intact; 271 Python sources parsed; v16 63/63 original assets verified.
Identity: DP_STABLE720_RESPONSE1 / a4_interactive_ui1_light2 / 1280x720@60
```

> 前两个数会随仓库增删文件而变化。判定标准是**出现 `PASS` 且 `v16 63/63`**，
> 而不是记住某个具体数字。

> 这一步失败几乎总是因为**换行符被转换**。仓库根目录 `.gitattributes` 用 `* -text`
> 关闭了 EOL 规范化，请勿删除，也不要设置 `core.autocrlf=true`。

### 2.1 必须用 Python ≥ 3.11

第 2 项语法校验会 `ast.parse` 仓库内全部 `.py`（当前 271 个），其中
`src/omr/app/route_trials/testset3_tie_recovery_v6.py` 用了 **PEP 646** 的下标星号解包：

```python
result[mi, *lane] = dict(...)
```

这种写法 **Python 3.11 才合法**。低于 3.11 的解释器会报：

```
SyntaxError: invalid syntax
  File ".../src/omr/app/route_trials/testset3_tie_recovery_v6.py", line 70
    result[mi,*lane]=dict(nominal_meter_quarters=str(meter),
```

这是**解释器版本不够**，不是仓库文件损坏。先用 `python --version` 确认；若系统里有多个
Python（如 Anaconda 与一个独立安装并存），显式指定可用的那个：

```powershell
& 'D:\Anaconda\python.exe' build\verify_package.py
```

或使用启动器：`py -3.12 build\verify_package.py`

---

## 3. B 级：静音主机回归（不碰板卡）

### 3.1 Python 回归

```sh
python -m venv .venv && . .venv/bin/activate
pip install mido verovio            # 或按 requirements-board.lock.txt 装全量
python -m unittest discover -s src/ps/tests -v
```

覆盖排版、播放时间线、按键响应、元数据权限。测试会 mock 掉合成为音频的部分，
**不会播放、不会调用真实音色**。权限用例需要 Linux root 模拟控制器与识别用户，
Windows 上会自动跳过。

### 3.2 固件主机 C 测试

```powershell
# 13 个组件：任意 C11 编译器即可
powershell -NoProfile -ExecutionPolicy Bypass `
  -File board/camera_interactive_20261030/software/camera_capture/tests/run_host_tests.ps1 `
  -Compiler 'C:\path\to\gcc.exe'
```

编译参数是 `-std=c11 -Wall -Wextra -Werror -pedantic -O2`，能编过本身就说明代码质量。
脚本末尾还有 2 个 `platform_i2c` 用例，它们**需要 Vitis 安装目录下的 `data/embeddedsw` 头文件**
（`iicps_v3_20/src`、`bsp/standalone_v9_2/src`，不是本仓库的 `data/`）；没装 Vitis 时这 2 个会失败，前 13 个仍然有效。

早期完整集成测试在 `board/camera_interactive_20261030/tests/`，依赖未随包交付的
设备树、历史构建目录和板端环境，**在空仓库中无法全量执行**，仅作参考。

---

## 4. C 级：识别端单页跑通（推荐的主复现路径）

这一级**不需要 AC920**：识别跑在 PS/A53 上，代码是标准 C++17 + Python，
x86_64 Linux 同样能跑（原作者做过桌面 dry run）。

### 4.1 编译 4 个加速库

```sh
cd src/omr/v16
sh cdec/build_cdec.sh        # libdecoder.so
sh hacc/build_hacc.sh        # libhomrgeo.so
sh racc/build_racc.sh        # librelgeo.so
sh sacc/build_pilresize.sh   # libpilresize.so
```

都是 `g++ -O3 -ffp-contract=off -fPIC -shared -std=c++17`。`-ffp-contract=off` 是刻意的：
关掉融合乘加，保证每个乘加按 numpy/CPython 的方式舍入，否则输出会变。**不要加 `-ffast-math`。**

### 4.2 让 `homr` 可导入

Homr 0.7.0 源码在 `src/third_party/homr/homr/`，指向其**父目录**即可：

```sh
export PYTHONPATH="$PWD/../../src/third_party/homr:$PWD/../../src/omr/app"
```

（也可以 `pip install homr==0.7.0`，但仓库内版本才是板端实际运行的那份。）

### 4.3 输入：用仓库自带的样张

`sample01-1.bmp` 是板端实拍的一页，**已经转正**（2868×4056、24 位），`run_page.py` 可以直接读，
不需要任何预处理。原始采集是传感器朝向（4056×2868，纸顶在右边缘），入库时逆时针转了 90°，
像素无损、字节数未变；要还原成板端原始朝向见 [`data/samples/README.md`](data/samples/README.md) 第 6 节。

**这个样张只能复现第 1 页。** 它覆盖第 1 页的 **17 小节**（5 个系统，起始小节 1 / 5 / 9 / 13 / 16，
底部被裁），而 `data/samples/result/01.musicxml` 是**整首 71 小节**的 album 合并结果；
第 2 页及以后的照片不在本仓库，所以**整首无法逐页复算**，只能核对产出的前缀。
样张构成、参考源和基线数字见 [`data/samples/README.md`](data/samples/README.md)。

换成自己的图时，要求相同：

- 整页乐谱、**正立**、尽量占满画面；
- 板端采集格式是 `Image/SS-PP.bmp`（24 位彩色）或 `Grey/SS-PP.bmp`（8 位灰度），4056×2868；
- 板端原始采集是传感器朝向（纸顶在右），album 模式按 `--rotate ccw` 转正；
  **若你的图已经正立，必须用 `--rotate none`**，否则会被再转一次。

### 4.4 跑一页

```sh
python run_page.py \
  --photo ../../data/samples/page/sample01-1.bmp --out /tmp/out1 \
  --bundle "$PWD/../../src/omr/app" \
  --threads 4 --csolve --fast-homr --fast-relation --fast-redecode \
  --shared "$PWD/shared" --relation-models "$PWD/shared/relation_models" \
  --link-margin 0.5
```

等价的板端入口是 `run_board.sh`（默认路径是板上的 `/home/fpga/...`，需用 `PY` 和第三个参数覆盖）：

```sh
PY=<你的 python> sh run_board.sh <照片> <新目录> <仓库>/src/omr/app
```

常驻流水线用 `run_serve.sh`（每行 `照片<TAB>输出目录`），拍照卡模式用 `album/run_album.sh`。

### 4.5 期望结果

输出目录里得到 `page.musicxml`、`page.geometry.json` 等。板端实测约 **29.9 s/页**（基准）、
常驻流水线 **20.6–21.0 s**、多页 **23 s/页**；x86 会明显更快亦或更慢，取决于核数。
67 页模拟拍照集音符完全正确率 **94.35%**。

> **已知差异：** ARM 与 x86 的浮点舍入不同，个别符头判断会翻转，
> 所以两边输出**不保证逐字节相同**（电子版测试页相同，实拍照片可能不同）。
> 用 `--only-segment` / `--only-relation` 可以分阶段定位问题。

---

## 5. D 级：采集固件构建

系统里有 Claude 之类工具链之外的 Windows 环境，按要求安装 Vivado + Vitis 2024.2：

```powershell
python tools\verify_package.py        # 先确认源码完整
powershell -NoProfile -ExecutionPolicy Bypass -File build\build_capture.ps1 `
  -VivadoRoot 'D:\Xilinx\Vivado\2024.2' -VitisRoot 'D:\Xilinx\Vitis\2024.2'
```

脚本依次做四件事：

1. `board/camera_onecard_20261029/vivado/build_camera_capture.ps1` → 纯 DP 采集硬件工程；
2. 检查 `build/vivado_camera/hardware.complete` 与 XSA 哈希（`-SkipHardware` 可跳过 1）；
3. `xsct build_vitis.tcl bsp <xsa>` → standalone BSP / FSBL / PMU；
4. `board/camera_interactive_20261030/build_onecard.py` → 当前 Light2 固件。

产物：`board/camera_interactive_20261030/build/BOOT.BIN`。

**关于 `src/rtl/vendor/dvi_tx/` 缺失：** 不影响这一级。当前纯 DP 工程只引用
`src/rtl/camera/{axis_frame_gate.v, camera_isp_control.v, camera_key_sync.v}` 加 PS 原生 DP 与 VDMA，
其 Tcl 第一行就写明 `No HDMI PL logic`；`prepare_hardware.py` 还会主动检查并拒绝
`camera_dvi_wrapper` / `hdmi/`。被移除的 DVI 源码只影响根目录 `src/rtl/score_*` 与
`build/vivado/{create_score_system,create_camera_capture,synth_hdmi}.tcl` 这条**历史 HDMI/PL 加速分支**，
它不是当前部署版本（当前版本没有部署 PL 神经网络加速）。详见 `src/rtl/vendor/dvi_tx/README.md`。

**烧写与上板不在本包范围。** 需要 AC920 实机、正确的 PS 配置与启动槽。

---

## 6. E 级：整机（不建议作为复现目标）

`src/ps/app/` 里的 `provision_card.py`、`install_interactive.py`、`prepare_dtb.py`、`patch_progress.py`
是完整可信的部署工具链，但它们**对整卡有硬性假设**，第三方几乎无法满足：

- 目标 SD 卡容量必须**恰好** `512711720960` 字节（约 477.5 GiB），否则直接报错；
- 布局固定：p1 起点 1 MiB、大小 16 GiB、FAT32、卷标 **`AC920_BOOT`**；p2 大小 64 GiB、ext4；
- `provision_card.py` 必须在**原始 AC920 Ubuntu 卡**上运行（校验 `/dev/mmcblk0p2` 为根），
  目标卡必须是 USB 外置盘，且拒绝 eMMC/系统盘；
- 双启动槽（`BOOT.BIN` 拍摄端 / `BOOT0001.BIN` Ubuntu）、PARTUUID、FAT UUID、DTB 补丁
  都要按这张具体卡生成；示例配置 `src/ps/config/ac920-onecard.example.json` 故意写成
  `REPLACE_WITH_THIS_CARD_*`，**不能直接用于另一张卡**。

因此整机复现的前提是"你手里有原始 AC920 环境"，这本质上是**同机验证**而不是复现。
想体验识别效果，请走 C 级。

---

## 7. 复现不了 / 需要注意的清单

| 项 | 状态 |
|---|---|
| 样张输入图 | ✅ 已提供 `data/samples/page/sample01-1.bmp`（34.9 MB，仅第 1 页） |
| 同曲其余页 | ❌ 仅有第 1 页，整首 71 小节结果无法逐页复算 |
| `BOOT.BIN`、bitstream、XSA | ❌ 未包含，需自行综合（D 级） |
| Ubuntu rootfs、内核、设备树、BOOT 镜像 | ❌ 未包含 |
| Vivado/Vitis、BSP、IP 生成产物 | ❌ 未包含，按各自许可自行获取 |
| `FluidR3_GM.sf2`（148 MB） | ❌ 未包含，Ubuntu `fluid-soundfont-gm` 提供 |
| Homr 上游大模型（57/47/52 MB） | ❌ 未包含，当前 v16 主路径不用 |
| `prebuilt/aarch64/*.so` | ❌ 未包含，首次运行自动 `g++` 编译 |
| `src/rtl/vendor/dvi_tx/` | ❌ 无授权，未收录（不影响 D 级） |
| 实拍准确率逐音核准 | ⚠️ 未做，94.35% 是模拟拍照集口径 |
| 720p 听音与声画同步验收 | ⚠️ 未完成，见 `board/deployment.json` 的 `outstanding` |
| 真实卡身份（PARTUUID/UUID/SHA） | ❌ 已刻意去除 |

---

## 8. 怎样算"复现成功"

| 级别 | 判定标准 |
|---|---|
| A | 输出 `PASS: … files intact … 63/63`，退出码 0 |
| B | `unittest` 无 FAIL/ERROR；13 个 C 组件测试输出 `PASS` |
| C | 输出目录出现 `page.musicxml`，能被 MuseScore / Verovio 正常打开且音符可读 |
| D | 生成 `board/camera_interactive_20261030/build/BOOT.BIN`，且 `verify_firmware.ps1` 通过 |
| E | 实机跑通拍照→识别→DP 显示→播放，且按键与切换正常 |

不要用"主机语法检查通过"代替上板验收——`build/README.md` 第 5 节说明了验证边界。

---

## 9. 常见失败点

| 现象 | 原因 / 处理 |
|---|---|
| `verify_package.py` 报某个文件字节数不符 | 换行符被转换（Windows Autocrlf / 编辑器）。重新 clone，确认 `.gitattributes` 存在 |
| `ModuleNotFoundError: homr` | `PYTHONPATH` 没指向 `src/third_party/homr` |
| `libdecoder.so` 加载失败 | 4 个 `.so` 没编译，或 g++ 版本过旧（需 C++17） |
| 识别结果乱、谱号找不到 | 朝向不对。样张已正立，别再用 `--rotate ccw`；走 album 模式时本样张必须 `--rotate none` |
| `run_board.sh: OUT exists` | 输出目录必须不存在 |
| `provision_card.py` 报容量不符 | 卡不是 512 GB，或不是外置 USB 盘 |
| D 级找不到 `hardware.complete` | 硬件工程未真正跑完，别用 `-SkipHardware` 掩盖 |
| D 级 `file not found ... dvi_tx` | 你在跑**历史 HDMI 分支**，不是当前纯 DP 入口，见第 5 节 |

---

## 10. 样张已经就位

`data/samples/` 里现在有实拍一页、参考乐谱和已记录的整首结果，C 级复现可以直接开跑，
不必再自备输入图。构成、命令和基线数字见 [`data/samples/README.md`](data/samples/README.md)。

还能继续补强两处：

1. **参考乐谱的制谱来源与许可**。`data/samples/source/` 的 `.mxl` 是 MuseScore 导出的第三方制谱；
   曲目本身是公有领域的 Vivaldi《春》，但**排版/改编**可能另有权利。发布前请补上来源与许可证。
2. **同一首的其余页**。现在只有第 1 页（17 小节）。补齐其余页才能验证两件目前无法复现的事：
   album 模式的多页合并为什么得到 71/79 小节，以及按 Grey sha256 跳过已完成页的续接行为。
