# 目录结构说明（重组记录）

本仓库已按竞赛提交结构重组。本文记录**新旧路径映射**、为什么这样切分、
以及重组时对构建脚本做了哪些改动。找不到某个文件时先查这里的映射表。

## 1. 当前结构

```
AC920_Source_GitHub_20261102/
├── README.md                    项目简介 + 复现步骤
├── REPRODUCE.md                 五级复现路径与不可复现项
├── REPO_LAYOUT.md               本文件：新旧路径映射
├── LICENSE / LICENSES/          MIT + 第三方许可证原文
├── THIRD_PARTY_NOTICES.md       逐路径许可范围与来源
├── PACKAGE_MANIFEST.json        全仓逐文件 SHA-256 清单
├── external-assets.json         未随包交付的依赖说明
├── requirements-board.lock.txt  板端 Python 依赖版本
├── src/                         设计源码
│   ├── rtl/                      设计 RTL（相机通路、乐谱显示/事件播放）
│   │   ├── camera/                 axis_frame_gate、camera_isp_control、camera_key_sync
│   │   └── vendor/dvi_tx/         未收录（无授权），仅说明
│   ├── omr/                      识别端：v16 Python/C++ + 支撑模块 + 权重
│   ├── ps/                       PS 侧部署软件：app、tests、systemd、desktop、display、config
│   └── third_party/homr/         Homr 0.7.0 原样（AGPL-3.0）
├── sim/                         仿真
│   ├── rtl/                       6 个 testbench
│   └── run_camera_sim.bat、run_score_sim.bat
├── build/                       构建
│   ├── README.md                  构建与部署说明（原 docs/BUILD.md）
│   ├── build_capture.ps1          固件一键构建入口
│   ├── verify_package.py          完整性校验
│   ├── vivado/                    综合/实现/下载 Tcl、XDC、批处理
│   ├── reports/                   FPGA 综合/实现报告（timing/utilization/DRC/CDC/bus_skew）
│   │   └── original_20261029/      2026-10-29 原始报告，未改动
│   └── report_export_logs/        报告导出执行日志
├── board/                       上板工程、运行配置与实测记录
│   ├── camera_onecard_20261029/   纯 DP 采集硬件工程（Vivado + RTL + BSP 输入）
│   ├── camera_interactive_20261030/ 固件工程（源码 + 主机测试 + 构建脚本）
│   ├── recognition/               板端识别证据：卡内容回读 + 历史日志
│   │   ├── card_20261103/          2026-11-03 只读回读（两首 MusicXML、Meta、OneCard）
│   │   ├── historical/             历史板端识别日志与逐页记录
│   │   ├── VALIDATION_20261103.json 机器可读验收摘要
│   │   └── EVIDENCE_20261103.md    证据索引与缺口说明
│   ├── config/camera_25mm_4056_a4.json
│   ├── deployment.json            当前部署快照
│   ├── capture_build_identity.json 构建身份与哈希
│   └── DP_RESTORE_20261031.md     DP 显示恢复记录
├── data/                        测试数据与参考结果
│   ├── samples/                   实拍样张 + 参考乐谱 + 已记录结果
│   └── package_validation.json
├── report/                      设计报告
│   ├── 作品设计报告.md
│   └── CAPTURE_INTERFACE.md       图像/方向/编号/按键接口约定
└── skill/                       技能包（预留，暂未提供）
```

## 2. 新旧路径映射

| 原路径 | 新路径 |
|---|---|
| `rtl/`（`score_*`、`camera/`、`vendor/`） | `src/rtl/` |
| `rtl/tb_*.sv`、`rtl/camera/tb_axis_frame_gate.sv` | `sim/rtl/` |
| `recognizer/` | `src/omr/` |
| `linux/` | `src/ps/` |
| `third_party/` | `src/third_party/` |
| `vivado/`（Tcl / XDC / bat / ps1） | `build/vivado/` |
| `vivado/run_camera_sim.bat`、`run_score_sim.bat` | `sim/` |
| `tools/build_capture.ps1` | `build/build_capture.ps1` |
| `tools/verify_package.py` | `build/verify_package.py` |
| `docs/BUILD.md` | `build/README.md` |
| `lab/camera_onecard_20261029/` | `board/camera_onecard_20261029/` |
| `lab/camera_interactive_20261030/` | `board/camera_interactive_20261030/` |
| `config/camera_25mm_4056_a4.json` | `board/config/camera_25mm_4056_a4.json` |
| `docs/CURRENT_DEPLOYMENT.json` | `board/deployment.json` |
| `docs/capture_build_identity.json` | `board/capture_build_identity.json` |
| `reports/DP_RESTORE_20261031.md` | `board/DP_RESTORE_20261031.md` |
| `samples/` | `data/samples/` |
| `docs/PACKAGE_VALIDATION.json` | `data/package_validation.json` |
| `reports/作品设计报告.md` | `report/作品设计报告.md` |
| `reports/CAPTURE_INTERFACE.md` | `report/CAPTURE_INTERFACE.md` |
| 根 `README.md`、`LICENSE`、`LICENSES/`、`PACKAGE_MANIFEST.json` 等 | 位置不变 |

## 3. 为什么 `board/` 里是两个"整包"，而不是拆散到 `src/` `sim/` `build/`

`board/camera_onecard_20261029/` 与 `board/camera_interactive_20261030/` 是**自包含的工程包**：
各自的 Vivado Tcl 用 `[file dirname [info script]] ..` 定位自己的工程根，再拼 `rtl/`、`vivado/`；
`build_onecard.py` 用 `HERE.parent/'camera_onecard_20261029'` 找同级的另一个包。

如果把它们拆到 `src/`（RTL）、`sim/`（测试）、`build/`（构建脚本），这些相对引用会全部断裂，
而修复需要改动 Vivado/Vitis 工程内部逻辑——风险远高于收益。因此保留为两个整包，
由 `build/build_capture.ps1` 作为唯一入口驱动它们。

## 4. 重组时改了哪些脚本（共 67 处）

| 文件 | 改动 |
|---|---|
| `build/vivado/*.tcl`（`create_*`、`synth_*`、`build_*`、`link_*`、`generate_*`、`report_*`、`write_*`） | 工程根改为上溯两级；`$repo_root rtl`→`$repo_root src rtl`；`$repo_root vivado`→`$repo_root build vivado` |
| `build/vivado/build_camera_capture.ps1` | `$PSScriptRoot '..'`→`'..\..'`；`${shortRoot}vivado/`→`${shortRoot}build/vivado/` |
| `build/vivado/build_score_system.bat`、`write_score_bitstream.bat`、`write_score_bitstream_bin.bat` | `cd %~dp0\..`→`..\..`；`vivado/`→`build/vivado/` |
| `build/build_capture.ps1` | `lab/camera_*`→`board/camera_*` |
| `sim/run_camera_sim.bat` | RTL 指向 `src\rtl\camera\`，testbench 指向 `sim\rtl\` |
| `sim/run_score_sim.bat` | `rtl/score_*`→`src/rtl/`，`rtl/tb_*`→`sim/rtl/` |

校验结果：`build/vivado/*.tcl` 中 36 处指向源码/约束的引用**全部解析到真实文件**；
另外 8 处指向 `build/vivado_*` 构建产物目录（由 Vivado 生成，当前不存在属正常）。

> ⚠️ **这些改动没有在 Vivado/Vitis 上实跑过**（本机没有工具链）。
> 路径是逐处静态核对过的，但首次综合/实现请留意日志里的 `file not found`。

## 5. 回退

重组前的完整快照保留在：

```
D:\College_study\Major\2026_FPGA\嵌赛final\_backup_pre_restructure\AC920_Source_GitHub_20261102\
```

需要整体回退时，把该目录内容覆盖回 `AC920_Source_GitHub_20261102\AC920_Source_GitHub_20261102\` 即可。

## 6. 2026-11-03 证据包并入

来源目录 `嵌赛final\AC920_Handoff_Evidence_20261103 (2)\`，并入 110 个文件。

| 证据包原路径 | 本仓库位置 |
|---|---|
| `build/reports/**` | `build/reports/**`（26 个：timing / utilization / DRC / CDC / bus_skew / methodology / check_timing，含 `original_20261029/`） |
| `build/report_export_logs/**` | `build/report_export_logs/**`（2 个导出日志） |
| `recognition/**` | `board/recognition/**`（80 个：卡内容回读 + 历史板端日志） |
| `VALIDATION.json` | `board/recognition/VALIDATION_20261103.json` |
| 包内 `README.md` | 改写为 `board/recognition/EVIDENCE_20261103.md` |

未并入：`docs/CAPTURE_INTERFACE.md`（与 `report/CAPTURE_INTERFACE.md` 字节相同）、
包内 `MANIFEST.json` 与 `verify_manifest.py`（清单列 121 项、实际交付 111 项，自校验会失败）、
根目录 3 张需求沟通截图（不在其清单内）。缺口清单见 `board/recognition/EVIDENCE_20261103.md`。

该证据包对应的是**纯 DP 采集工程** `camera_capture_wrapper`，即当前部署设计，不是历史 HDMI 分支；
报告为从既有已验收工程重新导出，未重新综合/实现。实现后 WNS 0.817 ns、TNS 0、DRC 8 条 Warning。

### 由证据包修正的一处事实

证据包的 `VALIDATION.json` 记录了板端逐页结果：第 01 首共 6 页，逐页 17 / 14 / 11 / 6 / 10 / 13 小节，
合计 71 小节。因此 `data/samples/page/sample01-1.bmp`（第 1 页）覆盖的是 **17 小节**，
此前文档中「约 18–20 小节」的估计已全部改正。

> **关于两份 `01.musicxml`**：`data/samples/result/01.musicxml`（LF 行尾）与
> `board/recognition/card_20261103/Outputs/01.musicxml`（2026-11-03 脱敏重导出，CRLF 行尾）
> **内容相同、字节不同**。此前记录的「SHA-256 一致」在证据包重导出后不再成立，以本节为准。

### 由此带来的 `.gitignore` 例外

证据文件会被原有规则静默排除，已逐个放行：`build/reports/`、`build/report_export_logs/`（及其 `*.log`）、
`board/recognition/**/*.log`、`board/recognition/**/Meta/`、`board/recognition/**/Outputs/`。
**提交前请确认这些目录确实进了暂存区**（`git status` 里应能看到 `.rpt` 与 `.log`）。

## 7. 仍未补齐

| 项 | 现状 |
|---|---|
| `collaboration/` 7 个文件 | 证据包清单里有、实际未交付（Codex/Claude 可见摘录、`AI_WORKFLOW_AND_CORRECTIONS.md`、`EXPORT_SCOPE.json`） |
| `requirements/request_1.png`、`request_2.png` | 同上未交付；根目录 3 张截图与之不是同一批文件 |
| `README_给组长.md` | 同上未交付 |
| 设计报告 docx、海报、视频、整机与转接板照片 | 在 `嵌赛final\` 下，尚未并入仓库（视频 186 MB 需 Git LFS 或外链） |
| `data/samples/source/*.mxl` 的制谱许可 | 未确认，发布前需补来源 |

## 8. 2026-11-03 脱敏补充包并入

来源 `final\AC920_Public_Handoff_20261103\AC920_Public_Handoff_20261103\`（包内双层同名目录）。
该包是上一证据包的**脱敏替换版**：声明不含聊天原文/会话标识/个人路径，并附 `docs/PRIVACY_REVIEW.json`
（134 文件自动扫描 + 人工复核，`findings: []`）。

| 包内路径 | 本仓库位置 | 数量 |
|---|---|---|
| `build/reports/**` | `build/reports/**` | 26（覆盖，脱敏重导出） |
| `build/report_export_logs/**` | `build/report_export_logs/**` | 2（覆盖） |
| `recognition/**` | `board/recognition/**` | 80（覆盖） |
| `collaboration/**` | `collaboration/**` | 3（新增，抽象协作说明） |
| `docs/**` | `docs/**` | 3（新增：技能包报告章节、证据溯源、隐私复核） |
| `skill/**` | `skill/**` | 20（新增：6 个技能包 + README + `validate_skills.py` + `VALIDATION.json`） |

未并入：包根 `README.md`、`MANIFEST.json`、`verify_manifest.py`——仓库已有 `README.md` 与
`PACKAGE_MANIFEST.json`，收录会撞车；包完整性由 `PACKAGE_MANIFEST.json` 统一校验。

### 两点必须知道

1. **80 个文件只差行尾**。包内副本是 CRLF、仓库原副本是 LF，内容一致。仍选择全量覆盖，
   因为 `docs/EVIDENCE_PROVENANCE.json` 逐文件记录了 `published_sha256`（发布副本身份），
   跳过换行符差异会让那份溯源记录与仓库实际内容不符。
2. **包内 4 个中文文件名已改为英文**，仓库同步采用英文名：

   | 原文件名 | 现文件名 |
   |---|---|
   | `collaboration/协作概述.md` | `collaboration/brief_description_of_collaboration.md` |
   | `collaboration/并行工作线与贡献.md` | `collaboration/parallel_working_lines_and_contribution.md` |
   | `collaboration/问题与修正方法.md` | `collaboration/problems_and_solution.md` |
   | `docs/3.3.5.2_技能包说明.md` | `docs/3.3.5.2_skills_interpretation.md` |

   正文一级标题仍为中文，与仓库其余中文文档一致。

包内 `collaboration/` 的 3 份抽象说明**替代**了 §7 中记录的旧缺口（聊天摘录、`EXPORT_SCOPE.json` 等
原始材料按脱敏要求不再分发）。
