# FPGA 综合与实现报告

本目录对应当前稳定版纯 DP/IMX477 工程 `camera_capture_wrapper`，XCZU4EV，Vivado 2024.2。本次只读打开既有已验收工程/检查点重新导出报告，**未重新综合或实现，也未改写位流、BOOT 或开发板**。

- `synthesis_*.rpt`：通过原工程 `open_run synth_1` 链接全部 OOC IP 后导出，确认无未解析黑盒。
- `implementation_*.rpt`：从原 routed checkpoint 导出。
- `original_20261029/`：2026-10-29 原始报告，未改动。
- `REPORT_IDENTITY.json`：工具、时间、检查点/位流/XSA/BOOT 的哈希及资源、时序摘要。

## 实现后结果

WNS **0.817 ns**，TNS **0 ns**，WHS **0.011 ns**，THS **0 ns**，WPWS **0.145 ns**；已声明的最终时序约束满足。实现后 LUT 10,825，寄存器 16,272，BRAM Tile 22.5，URAM 8，DSP 8。

实现 DRC 无 Error/Critical Warning，但保留 **8 条 Warning**：DPOP-4×4、REQP-1934×1、REQP-1935×2、RTSTAT-10×1。CDC、方法学及外部 I/O 时序提示见对应原始报告，不把“时序通过”写成“完全没有警告”。

综合阶段尚未布局布线，报告中有 hold/pulse-width 负裕量；最终验收应看 implementation 报告，不能把综合估计与布线后结果混为一谈。

本次导出最初发现单独打开顶层 synthesis DCP 时 15 个 OOC IP 成为黑盒、资源为零。该结果已排除并用原工程链接全部 IP 后的报告替换；两次执行日志保留在 `../report_export_logs/`，作为可复核纠错记录。
