# AC920 IMX477 纯 DP 单卡采集与识别

版本 `a4_onecard_dp2`，基于 album1 保留采集设置。PL 已移除 HDMI，S2/S3 已接入。

- S1 短按：同帧 Image/Grey/Meta，固定 4056×2868，SS-PP 编号；长按1.5秒下一首。
- S2长按约1.5秒：进入/退出原有对焦辅助；对焦中短按按1→2→3→4→5→1循环切区，预览中短按无操作。串口 f 和区域/放大/峰值功能保留。
- S3 或串口 u：等待保存完成，校验 Ubuntu 启动文件，停止传感器/CSI/VDMA/DP DMA，提交请求、同步并卸载 FAT，选择 MultiBoot1 后系统复位。
- Ubuntu：按请求自动调用已安装的 Claude album 入口，ext4 保存状态，FAT 只读输入；完成后导出 Outputs，安全返回 MultiBoot0。
- 任一识别/保存/卸载/寄存器步骤失败，阻止自动重启并保留日志。

DP 输出640×480 RGB8，60Hz时序；实际更新帧率待板测。原白平衡、CCM、曝光、降噪、照片格式与对焦算法均保留。S2=AG14，S3=AH14，Bank44 LVCMOS33输入。

完整制卡、接线、日志、验收与重建说明见本包根目录 `README_使用说明.txt`。
构建入口为根目录 `build_all.ps1` 或 `build_onecard.py`；`build_firmware.bat`调用全量入口。
本轮已通过构建、主机测试和模拟验证，单卡复制、物理按键、DP及双向软重启尚待实机验收。
