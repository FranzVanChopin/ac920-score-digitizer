# AC920 乐谱识别项目大模型交互节选

**顶部时间戳：** Wed, Oct 7 at 11:34 AM

### 用户消息（蓝色气泡）

阅读文件夹下的FPGA开发板文件和乐谱参考照片，完成基于FPGA的乐谱识别项目，项目分多个层次进行：

1. 能从SD卡中读取已经拍摄好的照片，准确无误的识别，然后将识别后的结果以常见乐谱软件支持的格式存储回SD卡中
2. 能在识别完乐谱后，读取乐谱文件实现预览播放
3. 实现能在HDMI输出类似乐谱软件一样的界面一边播放，一边展示乐谱以及当前播放音符
4.（暂时不推进，等我准备好摄像头）增加使用摄像头作为第二种输入，能实时通过HDMI预览摄像头画面，点按"拍照"键后识别当前帧，将识别后的电子乐谱存入SD卡中，同时按"播放"完成"3."的功能实现

乐谱识别不是实时每一帧的识别，尽可能追求准确率，而不是速度，同时注意不要超过FPGA的硬件规格，本机安装有Vivado，可以在安全的范围内随意使用，尝试自主推进并验证

**下方时间戳：** Oct 7, 11:34 AM

### Worked for 11m 47s

### 助手回复

这轮尚未完成整个项目。我搭好了软件与播放核的基础，并做了本机验证；但自动读出音符与节奏、SD 卡完整读写、MusicXML 读取播放、HDMI 乐谱界面都还没有实现。摄像头按你的要求暂缓。

目前工程包含：照片旋正和五线谱候选定位、人工校订 JSON 到 MusicXML 的导出器、MIDI 单音事件播放核、AC920 HDMI/BEEP 引脚约束，以及板端分工说明。主入口见 README.md (line 1)、score_tool.py (line 136)、score_event_player.sv (line 5)。

**验证结果：**

- 3 项 Python 单元测试通过；MusicXML 示例导出成功。
- Vivado 2024.2 在 xczu4ev-sfvc784-1L-i 上完成播放核仿真和综合：约 900 LUT、105 FF、1 个 DSP48E2。尚未加时钟约束、布局布线或生成 bitstream。
- 三张照片的旋正角度估计稳定，但定位器只找到 7、10、11 个谱表候选，且可能误检，不能据此生成准确乐谱。示例叠加图：page1/overlay.png。

下一步需要重点完成可靠的音符/节奏识别与校订流程，再接上 SD 卡写回、MusicXML 读取播放及 HDMI 显示。当前工程还没有在实体开发板上运行。

### 文件变更卡片

- Edited 14 files，+759 -1
- README.md +44 -0
- docs/architecture.md +44 -0
- software/score.schema.json +36 -0
- Show 11 more files

---

### 用户消息（蓝色气泡）

根据《Piano_Sonata_in_C_Major_op.4.musicxml》继续优化

### Worked for 36m 19s

### 助手回复

已根据《Piano_Sonata_in_C_Major_op.4.musicxml》建立 9 页逐音基准，并修复了谱号切换、调号取消和文字误检为五线谱的问题。清晰渲染图中，3713 个非装饰音有 3305 个音高与时值同时正确，约 89.01%。这不是整份 MusicXML 的准确率：装饰音、休止符、声部和播放时序尚未逐项计分。

合并后的候选 MusicXML 可解析，也能生成播放事件；但它有 204 小节，原谱为 196 小节，仍需复核。补光模拟图第 4、5 页漏检谱表，无法给出整曲有效总分。

原有 26 张合成图的逐图成绩未下降，《D大调主题》成绩不变，三张实拍参考图的识别事件也未变化；111 项单元测试通过。剩余主要问题是八度线、密集和弦的连梁关联、三连音，以及照片的局部透视矫正。数据和复现方法见逐页报告。

### 文件变更卡片

- Edited 8 files，+88 -5
- build/sonata/profile_staff_support.py +6 -3
- build/sonata/profile_residuals.py +4 -0
- docs/sonata_omr_benchmark.md +38 -0
- Show 5 more files

---

### 用户消息（蓝色气泡）

改一下程序，流程如下：4056×3040 原图 → 找到纸面并裁切（全分辨率）→ 缩放到纸面宽 1920（面积/双三次插值）→ CLAHE → 局部均值二值化 → 共享网络

### Worked for 25m 51s

### 助手回复

已实现，入口是 camera_shared_omr.py。

**流程为：**

原图定位纸面 → 原分辨率裁切 → 等比缩放至宽1920 → CLAHE → 局部均值二值化 → 共享网络。

缩小用面积插值，放大用双三次；取消旧的谱线间距二次缩放，并保存各阶段图片和裁切坐标。

**在项目目录运行：**

```powershell
& .\build\tools\homr\venv\Scripts\python.exe software/camera_shared_omr.py --image Image/00-00.bmp --output ...
```

新增 28 项测试全部通过，包括完整尺寸 BMP 接真实 ONNX 推理。另有一项旧测试因缺少模型清单失败，已记录于使用与验证说明。

本次修改的是 PS/Linux 识别程序，尚未上板；旧权重对新二值输入的准确率仍需验证。

### 文件变更卡片

- Edited 7 files，+617 -0
- software/camera_shared_omr.py +172 -0
- software/shared_camera/frontend.py +124 -0
- software/tests/test_shared_camera_frontend.py +141 -0
- Show 4 more files
