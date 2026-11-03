---
name: musicxml-regression
description: "比较未压缩 MusicXML 的音符事件、起点、时值、声部和重复次数，用于算法重构回归；不把基线相同当识别准确率。"
---

# MusicXML 语义回归

## 适用与输入

适用于 OMR 解码器、Python/C++ 重构、跨页合并等输出保持性检查。输入为一份冻结基线和一份候选 `score-partwise` MusicXML。参考真值是否存在必须单独说明。

## 工作方法

先冻结输入、参数、模型与基线；记录实际调用路径，避免只比较旧缓存。运行：

```sh
python scripts/compare_musicxml.py <BASELINE_XML> <CANDIDATE_XML>
```

脚本按 part、measure 顺序、voice、staff、音高/休止、起点、时值、倚音、cue 和 tie 比较事件多重集。使用有理数处理 divisions，正确区别 chord 与 backup/forward；重复事件不能被集合去重掉。未满拍可以合法存在，比较时不强行补齐。

将事件不变性、参考真值准确率、排版完整性、实际播放和耗时分成不同验证项。模型重构只得到字节或事件相同，不能据此宣称识别准确率提高。性能比较注明冷启动、稳态、流水线及样本集合，可用交错 ABBA 次序控制偶然波动。

## 失效与停止条件

脚本只支持未压缩的 partwise XML，不展开反复/跳房，不比较排版或全部演奏记号；MXL、timewise、跨小节延音展开等另做适配与测试。无法解析时 exit 2；发现差异 exit 1；事件相同 exit 0。需要更全面音乐等价时不能仅依赖本脚本。

## 来源与已验证效果

提炼自识别主线、符号/声部候选、独立作品复现和 Claude C++ 加速的共同回归经验。脚本使用真实生成结果验证重排/空白不改变事件，使用 divisions 等价、重复事件、和弦、backup 及音高变更夹具验证检测能力；不是重新评测历史全部数据集。
