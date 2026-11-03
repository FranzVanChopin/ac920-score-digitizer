# src/rtl/vendor/dvi_tx（未收录）

本目录原包含四个 DVI/TMDS 发送相关 Verilog 文件，**未纳入本仓库**。

## 未收录原因

这些文件的文件头标明作者为：

```
Company : 武汉芯路恒科技有限公司
          http://xiaomeige.taobao.com
Web     : http://www.corecourse.cn
```

它们来自该公司的商业课程/开发板资料，随附内容中没有提供任何开源许可声明，
也没有允许再分发的授权文本。在本仓库采用 MIT 许可公开发布的前提下，
把这四个文件一并推送会构成对原作者的侵权，因此只保留本说明。

原文件（如需自行取得）：

| 文件 | 说明 |
|---|---|
| `dvi_transmitter_top.v` | TMDS 发送顶层 |
| `dvi_encoder.v` | RGB888 → TMDS 编码，内含 `serdes_4b_10to1` 例化 |
| `encode.v` | 8b/10b 编码模块 |
| `serdes_4b_10to1.v` | 4 路并串转换 |

请向原作者购买或索取书面授权，并遵守其条款。本仓库不对这些文件作任何授权或保证。

## 对构建的影响

以下 Tcl 会引用本目录路径，缺文件时会报 `file not found`：

- `build/vivado/create_camera_capture.tcl`（第 18–21 行）—— 拍摄端 HDMI 预览工程
- `build/vivado/create_score_system.tcl`（第 14–17 行）—— 乐谱系统 PL 工程
- `build/vivado/synth_hdmi.tcl`（第 9–12 行）—— 历史 HDMI 方案

取得授权或自备等价模块后，把四个文件放回本目录即可，**无需修改任何 Tcl**。

## 替代方案

- 只做综合/仿真、不需要 HDMI 预览：用 `build/vivado/synth_player.tcl`、`build/vivado/synth_video.tcl`，
  它们不依赖本目录。
- 需要开源实现：可换用其他 MIT/BSD 许可的 TMDS 发送核（例如基于 DVI 编码的开源实现），
  并同步调整上面三个 Tcl 中的文件路径与模块例化名。
- 当前稳定版走原生 DP 显示，HDMI 路径并非部署版本所用（见根 `README.md` 与 `build/README.md`）。
