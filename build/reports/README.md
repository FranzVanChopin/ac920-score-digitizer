# FPGA 报告使用说明

对应当前稳定版纯 DP/IMX477 设计。报告从已验收工程及布线后检查点重新导出，并非本次重新综合、实现或上板。

最终实现 WNS 0.817 ns，TNS 0，WHS 0.011 ns，THS 0，WPWS 0.145 ns；实现 DRC 保留 8 条 Warning。综合阶段有尚未经布局布线修复的 hold/pulse-width 负裕量，应与最终实现结果分开报告。

综合报告使用原工程链接 OOC IP 后导出；顶层 DCP 黑盒导致的零资源结果不作为有效报告。

本发布版只对主机名、本机目录和可识别标识进行脱敏，未更改技术数值。来源哈希与发布哈希见 docs/EVIDENCE_PROVENANCE.json，发布文件校验以 MANIFEST.json 为准。
