# 构建与部署说明

## 1. 当前版本与外部环境

基线是 AC920 / XCZU4EV，Vivado + Vitis **2024.2**，Ubuntu ARM64，识别 Python **3.12**。本包是源码与小模型，不是可直接写入 SD 的系统镜像。板端 Ubuntu、内核/设备树、FSBL/PMU 固件、BSP 和 AMD IP 由原板卡环境或工具链提供。

`requirements-board.lock.txt` 记录已冻结环境中的实际版本，作为复现依据；不包含 ARM64 wheel 或 Python 解释器。GTK/PyGObject/Cairo、Xorg、ALSA/aplay、FluidSynth、`fluid-soundfont-gm`、g++ 等系统依赖由 Ubuntu 提供。Homr 原有大型 Transformer 模型不是当前 v16 无 Transformer 主路径；当前共享网络权重已保留在 `src/omr/v16/shared/`。被排除的大模型信息见 `external-assets.json`。

## 2. 采集固件

Windows PowerShell 在仓库根目录执行，例如（改成实际安装路径）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File build/build_capture.ps1 -VivadoRoot 'D:\Xlinix\Vivado\2024.2' -VitisRoot 'D:\Xlinix\Vitis\2024.2'
```

此入口依次构建纯 DP Vivado 工程、XSCT standalone BSP/FSBL/PMU，再编译当前 Light2 采集应用。最终输出为 `board/camera_interactive_20261030/build/BOOT.BIN`。保留原构建脚本的严格编译、硬件完成标记、XSA/bit 哈希和 ELF 检查；不会格式化或写 SD。

直接运行根 `vivado` 的历史 HDMI 工程不会得到当前纯 DP 部署。旧 `verify_onecard.py` 依赖历史发布目录，因此本包的新入口不调用该历史比对程序。当前冻结身份和先前构建证据见 `capture_build_identity.json` 与 `CURRENT_DEPLOYMENT.json`。更换工具链或重新生成 BSP 后，二进制不必与旧 BOOT 哈希相同，仍需板上验证。

## 3. 识别组件

在 Linux ARM64 的 `src/omr/v16/` 内构建四个库：

```sh
sh cdec/build_cdec.sh
sh hacc/build_hacc.sh
sh racc/build_racc.sh
sh sacc/build_pilresize.sh
```

C++ 源码和原精确计算选项均保留；不使用 `-ffast-math`。`run_board.sh` 亦会检查并按需编译。

单页测试显式指定 Python 和 bundle 路径（输入应为正立的整页图，输出用新目录）：

```sh
cd src/omr/v16
PY=/path/to/python3.12 sh run_board.sh /absolute/photo.bmp /absolute/new-result /absolute/repo/src/omr/app
```

不要直接运行 Homr CLI 代替当前主入口；v16 只使用其前端几何部分。`src/third_party/homr/homr/` 是板端实际版本源码；可通过安装对应 Homr 版本或设置其上级目录到 `PYTHONPATH` 使用。

## 4. Linux 应用及配置

`src/ps/app/` 是 `/opt/ac920-panel-responsive1` 的准确源码快照。原部署符号链接为 `/opt/ac920-panel -> /opt/ac920-panel-responsive1`。`src/ps/systemd/ac920-panel.service` 管理控制器，`src/ps/desktop/ac920-panel.desktop` 启动图形界面；控制器使用系统 `/usr/bin/python3`（GTK），识别使用配置中的 Python 3.12 环境。

原配置约定：

| 键 | 原部署路径 |
|---|---|
| `recognizer` | `/home/fpga/claude-release-v16-20261030`（本包 `src/omr/v16/`） |
| `app` | `/home/fpga/omr-transfer-20261024/sdcard_omr_ac920_20261024/app`（本包 `src/omr/app/`） |
| `python` | `/home/fpga/ac920-omr-runtime/venv/bin/python` |
| `user` | `fpga` |
| `rotate` | `ccw` |

复制 `src/ps/config/ac920-onecard.example.json` 为板端 `/etc/ac920-onecard.json` 前，应填写实际磁盘容量、分区 PARTUUID、FAT UUID、当前采集 BOOT 的 SHA-256 和安装路径。示例故意不带原卡的可用身份，不能直接用于另一张卡。

若识别目录改变，在安装后的应用目录调用 `patch_progress.make(实际recognizer目录, 应用目录/engine)` 重新生成阶段进度引擎；已包含的 `engine/run_page_ui.py` 固定指向原正式目录。不要只改配置而保留旧引擎路径。

原卡启动约定：FAT `BOOT.BIN` 为拍摄端、`BOOT0001.BIN` 为 Ubuntu，Linux 引导文件在 `linux/`。两个环境分时挂载 FAT，切换前完成同步和卸载。FAT `Outputs/SS.musicxml` 是曲库权威。新卡的分区、启动槽、内核/设备树与文件系统应按 AC920 原系统准备，本包不包含自动恢复整卡的命令。

## 5. 验证范围

`python build/verify_package.py`：检查全部文件的 SHA-256、Python 语法以及 v16 63 项原始清单。此检查不触碰设备、不播放音频。

`python build/sync_manifest.py`：改动、新增或重命名任何受管文件后，用它重建 `PACKAGE_MANIFEST.json`；
`--check` 只报告不写盘（清单落后时退出码 1），`--dry-run` 打印将要变更的条目。清单不收录自身。

> 校验与重建都要求 **Python ≥ 3.11**。`ast.parse` 会解析仓库内全部 `.py`（当前 271 个），其中
> `src/omr/app/route_trials/testset3_tie_recovery_v6.py` 使用了 PEP 646 的下标星号解包
> （`result[mi, *lane] = ...`），3.10 及以下会直接 `SyntaxError`。详见 `REPRODUCE.md` 2.1。

固件主机 C 测试在当前固件的 `tests/`；当前响应与元数据权限回归测试在 `src/ps/tests/`，已补齐旧排版器及参考 MusicXML，并只适配导入路径：

```sh
python -m unittest discover -s src/ps/tests -v
```

此组测试合成音频被 mock，不会播放或调用实际音色合成。权限用例需要 Linux root 模拟控制器与识别用户，Windows 会跳过。

较早的完整集成测试仍保留在 `board/camera_interactive_20261030/tests/` 供参考，其中部分依赖未随源码包交付的旧设备树、历史构建目录和板端环境，不能在空仓库中直接全量执行。不把主机语法和文件校验当作上板验收。

打包不改变固件行为。既有稳定版仍有整机重启的短暂黑屏；新的环境仍需验证真实相机、按键、DP、识别和音频同步。自动检查保持静音。
