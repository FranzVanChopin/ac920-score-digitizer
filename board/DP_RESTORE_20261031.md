# 原 DP Light2 恢复记录

按用户要求，在完整备份当前 Audio2 后恢复 HDMI 改造前的 DP Light2。当前采集版本为 `a4_interactive_ui1_light2`，BOOT.BIN SHA-256 为 `F855AB741399375A44189F2F9D6670AB5E56C19F02D4AD709149D721BAAF6CE8`。没有重编译或超频，恢复的是原快照。

## 恢复前备份

本机目录：`outputs/backups/before_dp_restore_20261031_233948`。

- 本机源码和发布文件 693 份，214,429,586 bytes，逐文件 SHA-256 核验。
- 当前 FAT 归档 `boot_files.tar.gz`：206,460,301 bytes；39 个文件与板上清单逐项匹配，归档内容及 gzip 完整性已校验。
- 当前 Linux 项目运行归档 `runtime.tar.gz`：173,901,115 bytes；包含配置、显示/控制器程序、识别结果、工作目录及播放缓存等，归档及 gzip 完整性已校验。
- 板上备份目录：`/home/fpga/codex-before-dp-restore-20261031`。备份是项目文件和数据快照，不是 512 GB 整卡裸镜像。
- 当前 Audio2 的小型显示配置恢复快照另存为 `profile_restore.tar.gz`，便于未来恢复同一套显示配置。切勿仅替换一个 bitstream。

## 已恢复与校验

原 DP 快照中的 BOOT.BIN、Linux PL、设备树、boot.scr、OneCard 清单、主配置及桌面入口均已恢复并读回核验。桌面入口的可读权限保留；原生 HDMI 服务已禁用，原 GTK 界面正常启动。

恢复初期 Linux DRM 报 `DP-1 connected`，读取 256 bytes EDID；Ubuntu 自动选择过 3840×2160@30 Hz，随后出现资源分配和窗口布局问题。先固定 720p 恢复界面，再于 2026-11-01 升级并验证 Ubuntu 1080p60，详见下节。EDID 广告的模式不表示面板物理像素为 4K。DP ALSA 声卡及两个 PCM 设备已识别；本轮没有启动乐谱播放或确认实际扬声器声音。

返回拍照系统时，原固件标识、DP AUX 能力读取及训练正常，成功输出 1280×720、74.25 MHz，2 lanes / RBR，`online=1`、`timeouts=0`。随后 IMX477 芯片 ID 读取无应答；用户已确认摄像头未连接，故本轮未验收相机采集、拍照或完整往返，不将测试等待超时算作通过。已重新进入 DP Ubuntu，按用户要求留在可用的乐谱界面。

当前照片与元数据共 21 份，以及采集暂存文件 `Image/CAP1.DAT`，共 22 个受保护相册文件，逐项匹配恢复前 SHA-256；恢复未删除照片、识别结果或播放缓存，也没有格式化或重分区。

## 证据与后续使用

恢复脚本与日志：`lab/dp_restore_20261031/`；最后一次板上状态为其中 `evidence/claude-dp-restore-evidence-20261031/dp_linux.json`。备份校验收据为备份目录的 `BOARD_BACKUP_VERIFIED.json`。

当前使用开发板 **DP 接口 → 新转接线 → HDMI 屏幕**。Ubuntu 中长按 S1 切换识别页和曲库，S2 短按选曲、长按识别或播放。摄像头接回后，再用 S3 返回拍照测试；在未连接摄像头时不进行此项切换。原设计冷上电仍先进入采集，未接摄像头会报 CAM ERROR；本次没有改变这项启动逻辑，当前留在可用的 Ubuntu。

## 2026-11-01：按键、曲库与 Ubuntu 1080p

原 Linux boot.scr 遗漏了 HPM0_LPD 的 128→32 bit 切换，与已验证的 DP PL 不符，GPIO/VDMA 读数异常。增加启动前位宽检查后，AFI=0，GPIO channel2=0x310，VDMA version=0x62000050；用户确认 S2 恢复。该阶段 boot.scr SHA-256 为 `83E15FDB1F03759BFFD5BBBCDA7D5FD28757D3B8023D79D57BCF6A72D1323F90`；早间又加入统一 720p 参数，最新身份见下节及 CURRENT_DEPLOYMENT，不能再用未修正的原 boot.scr 覆盖。

控制服务对名为 `ac920-panel` 的 evdev 设备执行 EVIOCGRAB，只接管板载按键；USB 键盘不受影响。用户确认设置页不再误弹。按键已有内核 20 ms 消抖/10 ms 轮询，去除 Python 重复消抖，缩短控制循环和状态轮询。测得旧 XRender 路径仍有约 0.5–0.9 s 的绘制等待；改为 Cairo 离屏绘制后，用户确认选曲基本及时。进一步缓存原生分辨率的静态卡片阴影，1080p 选曲至绘制测得约 44–79 ms（未计实体消抖及屏幕处理）。

成品库改为 FAT `Outputs/SS.musicxml`；一次性保留/迁移旧曲谱，此后删除 FAT XML 在下一次控制器启动时清理对应 ext4 副本和播放缓存，不再自动补回。真实卡空闲编号 99 的删除测试通过；Linux 上 10 项行为用例全部通过，含符号链接不越界清理。已有照片及曲谱逐项哈希不变。程序 `/opt/ac920-panel-fat1` 由 `/opt/ac920-panel` 链接启动；更新前独立备份位于板上 `/home/fpga/codex-before-fat-library-20261101`。

Ubuntu 通过原生 DP 输出 1920×1080@60，DRM 时钟 148.5 MHz、scanout 1920×1080、stride 7680。单独 xrandr 改模式会留下 Mutter 旧逻辑布局，因此启动器通过 Mutter DisplayConfig 同步 1× 逻辑显示后再创建 GTK 窗口。保留 720p60 回退。拍照端原 Light2 BOOT 未改，仍是 1280×720；摄像头未接，不宣称已验证拍照端 1080p 或全流程往返。

源码及证据见 `lab/fat_library_20261101/` 和 `lab/dp_1080p_20261101/`，当前部署信息以 `release/CURRENT_DEPLOYMENT.json` 为准。旧 Light2/HDMI 发布 ZIP 保留原样，不代表包含本次 Linux 修正。

### 本次收尾与明日接续

用户要求今天收尾，停止进一步修改。Ubuntu 实际重启后仍为 1920×1080@60，GTK 窗口完整 1920×1080；控制服务 ActiveState=active、NRestarts=0，按键独占保持，AFI/VDMA 读数正确。五首曲谱及 22 个相册保护文件校验不变。最终板上截图、启动核验和当前部署源码归档保存在 `lab/dp_1080p_20261101/final/`。

最新实物反馈为“按键稍慢”。这是尚未解决的问题，不能将软件计时约 44–79 ms 当作实体按键延迟已经通过验收。明日优先关联内核按键事件、控制器动作、GTK 绘制和实际显示，定位剩余等待；先不改动图像处理和识别算法。

本轮增量回退备份已复制至本机 `outputs/backups/before_linux_fat1_1080p_20261101/`，归档 SHA-256、成员可读性及 gzip 完整性已核验。当前部署身份见 `release/CURRENT_DEPLOYMENT.json`。未格式化、未重分区；摄像头仍未接，拍照侧 1080p、相机全流程及实际扬声器声音不列为本轮通过项。

## 2026-11-01 早间：撤回 1080p，显示与识别稳定修复

用户反馈：切换两端反复黑屏/丢失信号，第 0 首识别报错且显示帧重叠，S2 延迟高；明确允许为稳定性降低分辨率和帧率。当前保留 **720p60**，不再优先推动 1080p。摄像头已连接，未改原采集 BOOT、PL、设备树、图像处理和 v16 引擎内容。

处理结果与依据：

- **启动显示统一**：boot.scr 加入 `video=DP-1:1280x720@60`，保留 AFI32 检查；Xorg 首选 720p，用户 monitors.xml 与应用启动器同为 720p，避免每次启动先按 EDID 选 4K 再改模式。当前 boot.scr SHA-256 为 `E6EA757A667C3576A818A15CB8A10DA6295CCF8BAAE9510EB6E8A60232DEA33E`。
- **全屏合成**：窗口设置 EWMH `_NET_WM_BYPASS_COMPOSITOR=2`，请求合成器保留合成，避免全屏直出更新造成的帧混合。该属性含义依据 [EWMH 规范](https://specifications.freedesktop.org/wm/latest/ar01s05.html)。用户在真实识别更新时确认无撕裂重叠；不能仅靠截图证明连续视频不撕裂。
- **识别权限回归**：昨晚 root 曲库清理用 atomic() 重写 `results/album.json` 为 root:root/0600，普通识别进程报 PermissionError。现在启动/清理后恢复该文件为 results 所有者（uid/gid 1000）/0600。Linux root 回归 3/3 通过，真实卡测试编号 99 的清理重写后普通用户仍可读；第 0 首真实识别约 40.9 秒完成，17 个 measure、366 个 note 元素，导出 FAT `Outputs/00.musicxml`。此计数不是识别准确率评估。
- **响应与打开**：用户确认降回 720p 后 S2 快多了；最终版本再用后台状态等待避免 GTK 线程做套接字等待，逐事件处理按键。谱面先发布，优先生成当前速度，其余低优先级后台准备；前台动作取消预备任务，返回前等待其释放输入。真实第 0 首静音准备约 2.0 秒出谱面、7.8 秒资产就绪，暖缓存约 3.6 ms（控制器阶段）。

相机/Linux 共观察三次往返：三次相机均 0x0477/READY，DP online=1、locked=0、timeouts=0。第一轮脚本固定等 GUI 18 秒偏短，未自动判通过，但稍后窗口及用户实屏确认恢复；第二、第三轮增加有界窗口等待后自动通过。用户确认切换只短暂黑屏，不再反复丢失信号。整机重启导致的一次信号中断和启动等待仍存在。

当前 `/opt/ac920-panel` 指向 `/opt/ac920-panel-responsive1`。11 项响应/缓存用例在主机和板上通过，实际谱面 MIDI 的音高、时长与旧排版器一致；引擎文件逐项哈希不变。22 个原有相册保护文件不变，没有新建照片、格式化或重分区。新备份 `outputs/backups/before_stable720_response1_20261101/`，最终源码/配置归档和证据 `lab/display_stability_20261101/final/`。旧 ZIP 未替换；恢复时以最新部署记录和配套归档为准。

试验中曾主动播放 3 秒钢琴，用户要求不要突然播放；此后所有自动验证静音。实时合成原型未部署，当前仍为既有 MIDI→WAV→DP ALSA。实际听音、四档切速和同步等待用户准备好后单独验收，不能把静音生成 WAV 当作声音验收。
