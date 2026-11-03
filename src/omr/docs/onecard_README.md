# 拍照和识别放在同一张卡上（2026-10-29，已按 Codex 审阅修订）

目标：一张 SD 卡（512 GB）同时装 Codex 的采集固件 a4_album1 和 Ubuntu 识别系统，不用换卡。拍完一组后切换到识别，乐谱写回卡上。

这是过渡方案。Codex 估计它的工作量明显小于把采集整个移进 Ubuntu。好处是白平衡、曝光、预览、对焦都保持已调好的状态；代价是拍照和识别之间要重启一次。

## 1. 原理：多重启动

ZynqMP 从 SD 卡启动时，BootROM 在卡的**第一个 FAT 分区根目录**里找启动文件：`CSU_MULTI_BOOT`（0xFFCA0010）为 0 时启动 `BOOT.BIN`，为 N 时启动 `BOOT000N.BIN`。

这个值只在上电复位时清零，系统复位（软件复位、reboot）后保留。依据：
- UG1087 的 csu_multi_boot 寄存器说明；
- UG1137 的 SD 启动、Fallback and MultiBoot 章节；
- Xilinx wiki 中 "Zynq UltraScale+ MPSoC Multiboot and Fallback" 页的示例日志。

**Linux 怎样访问这个寄存器：**
- 通过 `/dev/mem` 直接读会出 **Bus error**（10-29 inspect.sh 的结果）；
- 改走 PMU 固件的接口 `/sys/firmware/zynqmp/config_reg`。UG1137 "CSU/PMU Register Access" 一节把 Csu_multi_boot 列在白名单里。读法是 `echo 0xFFCA0010 > config_reg; cat config_reg`，写法是 `echo 0xFFCA0010 0xFFFFFFFF <值> > config_reg`。

## 2. 10-29 检查结果（现在的 Ubuntu 卡）

- **启动分区 p1：** FAT32，300 MB，剩余 269 MB。里面有 `BOOT.BIN`（1.8 MB，只含 FSBL、PMU、ATF、U-Boot）、`Image`（内核，22.6 MB）、`system.bit`（PL 比特流，由 U-Boot 加载）、`system.dtb`、`boot.scr`、`uboot.env`、`uboot-redund.env`、`readme.txt`。
- **`boot.scr` 的内容：** bootargs 为 `root=/dev/mmcblk1p2 rw rootwait`，不加载 initramfs（`booti … - …`），内核、设备树、比特流都用 `fatload mmc 1` 从根目录读取，每次启动都会 `saveenv`。
- **`/etc/fstab`：** 只有 configfs 一行，没有 UUID。根分区由内核参数里的设备路径决定。
- **重名冲突确认：** 内核文件 `Image` 与 a4_album1 要建的照片文件夹 `Image/` 同名。
- **第一版检查脚本有缺陷（Codex 指出）：** 读寄存器出错后仍然打印了 done。已改为 `inspect2.sh`。

## 3. Codex 审阅意见与处理（10-29）

| # | Codex 意见 | 处理 |
|---|---|---|
| 1 | S3 接 PL 的 AH14，当前比特流没有引出；以前接入时，bank 44 出现电压冲突：HDMI 要求 1.8 V，S3 要求 3.3 V（`build_camera_21280.backup.log` 第 1278 行）。第一版先用串口命令切换 | **同意。** 已核对日志。另外，S2（AG14）也在 PL 上，同样的问题。不改 FPGA 就能用的物理键只有 S1（PS MIO44），可以考虑"S1 长按 4 s 以上 = 切换"（现在 1.5 s 是换曲），由 Codex 决定 |
| 2 | 保留 `Image/` 照片目录。Linux 启动文件放进独立目录，并同步修改启动脚本 | **同意。** Linux 的 `Image`、`system.dtb`、`system.bit` 放进 `linux/`，`boot.scr` 改成从 `linux/` 读取。有两处必须留在根目录：`boot.scr` 和 `uboot.env`（U-Boot 只在根目录找它们），以及 `BOOT0001.BIN`（BootROM 只搜根目录）。新的 `boot.scr` 由原文件精确改写，用 Python 重新生成 U-Boot 脚本镜像，原文件由 inspect2 拷出 |
| 3 | 切换前完整收尾：采集端等保存完成，停传感器和 DMA，关闭、同步文件系统，写入并回读 MultiBoot，再做系统复位；Linux 端确认同步、卸载成功。reboot 路径要实测 | **同意。** Linux 端脚本的顺序：停止识别服务 → sync → 卸载 FAT 分区并确认 → 通过 config_reg 写入、回读确认 → `systemctl reboot`。任何一步失败就停下，不复位 |
| 4 | 新卡不能只核对 `root=/dev/mmcblk1p2`，还要查 UUID、`/etc/fstab` 和启动脚本 | **同意。** 现状是 fstab 没有 UUID，也没有 initramfs（第 2 节）。制卡脚本复制完后，会在新卡上再查一遍：fstab、systemd 的 .mount 单元，以及任何 UUID、PARTUUID、LABEL 引用 |
| 5 | 检查脚本要加强错误处理，确认 `/dev/mmcblk1p1` 属于当前系统卡，区分 `Image` 是文件夹还是文件 | **同意，已改（`inspect2.sh`）：** 必须用 sudo；核对 p1 与根分区在同一张卡上；区分文件夹与文件；寄存器改走 config_reg 读取；每项失败都打印 ERROR，最后的退出码非 0 |

## 4. 卡的布局（512 GB 卡）

| 分区 | 内容 |
|---|---|
| p1 FAT32 16 GiB（已有，卷标 AC920_BOOT） | 根目录：两个启动文件、`boot.scr`、`uboot.env`、`uboot-redund.env`；`linux/`：`Image`、`system.dtb`、`system.bit`；照片目录 `Image/`、`Grey/`、`Meta/`；乐谱 `Outputs/` |
| p2 ext4，约 32–64 GiB（待建） | 现在 Ubuntu 根文件系统的副本（含识别运行环境和我们的包） |
| 其余约 400 GiB | 不分区 |

**两个阶段的启动文件：**

| 阶段 | `BOOT.BIN`（上电默认） | `BOOT0001.BIN` | 怎样切换 |
|---|---|---|---|
| 测试布局（Codex 不用改） | Linux | a4_album1 | Linux → 采集：脚本写 1 并 reboot；采集 → Linux：断电重启 |
| 最终布局（Codex 加了串口命令以后） | a4_album1 | Linux | 采集 → Linux：串口命令写 1 并系统复位；Linux → 采集：脚本写 0 并 reboot，或断电重启 |

两个阶段之间只是把两个文件互换名字。

## 5. 实施顺序

Codex 建议的顺序是：只读检查 → 串口验证双向切换和断电恢复 → 制作大卡 → 自动识别服务 → 物理按键。

**我的调整只有一处：** 切换测试放在大卡上做，不在现在的系统卡上做。原因：
- 在现在的卡上做，就要先把正在用的系统的内核挪进 `linux/` 并改启动脚本，出错的话 Ubuntu 起不来；
- 大卡是独立的，现在的卡全程不动，留作备份。

1. **只读检查 v2：** `sudo sh /home/fpga/claude-onecard-20261029b/inspect2.sh 2>&1 | tee /home/fpga/claude-onecard-20261029b/inspect2.log`，同时拷出原始 `boot.scr` 和 `uboot.env`。
2. **制作大卡，用测试布局：** 由我写脚本，用户用 sudo 运行。大卡通过 USB 读卡器插到开发板上。脚本执行前会列出要做的每一步，等用户确认。
   - 新建 p2（ext4），复制根文件系统，复制完核对 UUID 等引用；
   - 把 Linux 启动文件放进 p1 的 `linux/`，写入新的 `boot.scr`；
   - a4_album1 改名为 `BOOT0001.BIN`，Linux 的启动文件作为 `BOOT.BIN`。
3. **在大卡上验证：**
   - Ubuntu 能从大卡启动；
   - Linux → 采集：收尾、写 1、reboot，串口上出现 a4_album1；
   - 拍 1–2 页，断电重启回到 Ubuntu；
   - album 能识别这些页；
   - 再试一次断电恢复。
4. **Codex 加串口切换命令**（带完整收尾，写 1、回读、系统复位）。然后两个文件互换名字，变成最终布局，验证采集 → Linux 的软切换。
5. **开机自动识别服务：** Linux 开机后挂载 p1，运行 album，结果写回 `Outputs/`，完成后回到采集或等待。要安装 systemd 服务，需要用户同意并输入 sudo。
6. **物理按键：** 第 3 节第 1 条，由 Codex 决定。

## 6. 风险与恢复

- **写错寄存器，或找不到启动文件：** BootROM 会继续往后找，最后停住。断电重启即可恢复：上电复位清零，回到 `BOOT.BIN`。
- **大卡做坏：** 不影响现在的系统卡，换回原卡就能启动。
- **FAT 分区两边都会写：** 每次切换前都要收尾（第 3 节第 3 条）。
