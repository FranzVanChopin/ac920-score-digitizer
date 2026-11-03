# 可复用技能包

本目录是从多模型及多个并行 Project 的工程经验中提炼的新技能包，采用 SKILL.md 入口。它们不依赖个人账户、原聊天或固定本机路径。可以单独复制一个技能目录到支持此格式的工具中，或直接阅读步骤并运行脚本；不要求安装本项目的完整环境。脚本只需 Python 3 标准库。

| 技能 | 适用场景 / 使用入口 | 资源 | 本次验证 | 主要失效边界 |
|---|---|---|---|---|
| [fpga-report-review](fpga-report-review/SKILL.md) | FPGA 报告与 OOC 黑盒复核 | 六报告读取脚本 | 真实报告及异常输入 | 缺报告/陌生格式，不替代上板 |
| [musicxml-regression](musicxml-regression/SKILL.md) | 识别/解码重构语义回归 | 有理数事件比较脚本 | 真实 XML、等价/改变/重复事件夹具 | 不展开反复、不代表准确率 |
| [pynq-artifact-check](pynq-artifact-check/SKILL.md) | 任意 Overlay bit/HWH 文件交接 | 元数据只读预检 | 真实 XSA 提取文件对及错误配对 | 尚未 PYNQ 实板验证 |
| [beat-timeline](beat-timeline/SKILL.md) | 计算过程动画与既有配乐对齐 | 拍点脚本、镜头表 | 整数/非整数 FPS 锚点与误差 | 变速音乐/VFR 需另建映射 |
| [multi-project-handoff](multi-project-handoff/SKILL.md) | 多工作线接口和抽象交接 | 提示词骨架、合同和交接模板 | 以本工程产物反填案例检查 | 未做另一团队从零迁移验收 |
| [private-release](private-release/SKILL.md) | 准备外发技术证据 | 隐私候选扫描脚本 | 发布目录与敏感样例检出 | 非完整匿名化、需人工复核 |

## 快速使用

1. 根据上表选择一个技能，读取其 SKILL.md 中的输入与失效条件。
2. 将示例命令的尖括号参数替换为自己的文件；不要照搬本项目的地址、板型或曝光参数。
3. 脚本输出是离线检查结果；硬件下载、写卡、网络发布与实际播放不由这些检查自动触发。
4. 保留原输入，输出到自己的新位置，记录验证条件和未验证部分。

技能来源、通过/失败测试及范围见本目录 VALIDATION.json；完整检查可用 `python skill/validate_skills.py`。技能格式检查不等于行为验证。PYNQ 项仅证明离线文件预检可复用，不声称已完成 PYNQ 应用或已获得额外评分。

## 本目录的其他内容

- `AC920_chat_with_models.md`：本项目的对话节选，是 `report/Design_Report_20261103.docx`
  附录2「大模型协作记录节选」的素材来源。它不属于上述 6 个技能包，属于原始协作记录。
- `validate_skills.py`、`VALIDATION.json`：技能包的离线校验脚本与验证范围记录。
