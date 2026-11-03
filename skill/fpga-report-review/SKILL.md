---
name: fpga-report-review
description: "核对 Vivado 综合与实现报告、OOC 黑盒、资源、时序和 DRC。适用于报告交接或构建结果排查，不执行综合或上板。"
---

# FPGA 报告核对

## 适用与输入

适用于 Vivado 报告归档、资源异常和时序结论复核。输入为 synthesis/implementation 各自的 timing、utilization、drc 文本；明确器件、工具版本、工程身份及设计阶段。

## 工作方法

1. 检查报告 Design State 和身份。若综合只有顶层 DCP，先确认 OOC IP 是否已链接；出现 INBB-1/黑盒时停止给出资源通过结论。导出需要现有工程时使用只读打开与 `open_run synth_1`，不自动重新综合。
2. 分开记录 setup、hold、pulse-width 和未约束端点；只对“已声明的约束”下结论。
3. 使用 `python scripts/review_reports.py <REPORT_DIRECTORY>` 取得结构化摘要。目录内应有 `synthesis_timing.rpt` 等六份文件。脚本不写文件、不访问板卡。
4. 保留 DRC 的 Rule、Severity、Count，区别 Error、Critical Warning 和 Warning；时序通过不能写成全部检查无警告。
5. 交接中分别说明报告重新导出、完整重建和实板测试；不能互相替代。

## 失效与停止条件

缺少完整摘要、报告格式不同或身份不一致时，先保留原文件，返回无法判定；不把缺失指标填成零。脚本 exit 2 为输入/格式问题，exit 1 为未过最终核对，exit 0 仍不代表系统功能验收。

## 来源与已验证效果

源于嵌入式视觉工程中 OOC 黑盒导致的零资源误判，以及综合前后时序被混用的排查。本技能脚本在真实六份报告上复核资源和最终裕量，并用不完整/黑盒输入验证拒绝误判。具体结果见技能包根目录 VALIDATION.json。
