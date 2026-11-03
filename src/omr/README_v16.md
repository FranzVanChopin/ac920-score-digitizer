# 无 Transformer 的乐谱识别：AC920 开发板 release v16（2026-10-30）

v16 是接入 Codex 拍照系统（a4_album1 / Score Studio，512 GB 单卡双系统）的识别端正式版。

与 v15 相比：
- **网络与模型不变**，仍是共享网络 sh8s；每页的 MusicXML 与 v15 逐字节相同。
- **更快：**
  - Homr 谱线几何改用 C++，属于精确提速，输出不变；
  - 常驻流水线：上一页还在做后续计算时，下一页已开始解码、预处理和网络推理。
- **新增 album 模式：** 直接读取拍照卡（`Grey/SS-PP.bmp` + `Meta/SS-PP.json`），每首输出一个带谱号、调号、临时记号和符杠的 MusicXML。
- **修复两处问题：** 常驻流水线模式在 stdin 为管道时会死锁；拍号可以延续到下一页。

## 头条结果

| | **v16** | v15 |
|---|---:|---:|
| 开发板整页时间，单次进程（测试页） | **31.0 s**（冷启动 35.6 s；10-27 开发版测得 29.9 s） | 31.9 s |
| 开发板常驻流水线，每页（测试页 / camera67 照片） | **20.6–21.0 s / 10.3–14.4 s**（第 2 页起） | 28.8–29.1 s / 17.9–18.0 s（常驻，不分流水线） |
| 开发板 album，《月光》实拍密集页（每页约 510 个音） | 33–37 s（冷启动首页 47 s） | — |
| 开发板测试页 `/Image/00-00.png` 的 MusicXML | 与 v15 相同（f618b20b） | f618b20b |
| camera67 音符正确率 | 94.35%（输出与 v15 逐字节相同） | 94.35% |
| album：《月光》实拍一页（桌面，PDMX 转谱） | 66.5% 正确音符（按正确方向） | — |

## 与 v15 相比改了什么

| 文件 | 改动 | 是否影响输出 |
|---|---|---|
| `hacc/homr_fast.py`、`hacc/homrgeo.cpp` | Homr 谱线几何的 C++ 实现（resample、交叉检测等，共 18 处补丁），`detect_staff` 4.75 s 降到 2.85 s | 不影响：70 张照片逐字节相同，开发板 MusicXML = v15 |
| `run_page.py` | 1. `--pipeline`：常驻模式流水线，下一页的解码、自动裁切、缩放、CLAHE 和网络推理在后台线程完成，只有输入完全相同才采用；<br>2. 常驻模式第三列 `@prev`：沿用本进程上一页的拍号；<br>3. **修复**流水线模式下 stdin 为管道时的死锁（解码进程 fork 后关闭 `sys.stdin` 会卡在读线程持有的锁上）；<br>4. `OMR_CV_THREADS` | 单页运行不变（v13 参数仍为 d01213b3） |
| `lab/board_decode.py` | 常驻模式下缓存解码器 | 不影响 |
| `sacc/shared_runtime.py` | 流水线用的 `infer` / `adopt` | 不影响 |
| `run_serve.sh` | 加上 `--pipeline` | 每页结果与单次运行相同 |
| `album/`（新增） | `omr_album.py`、`merge_pages.py`、`engrave.py`、`check_engrave.py`、`run_album.sh`，说明见 `docs/album_README.md` | 只改曲目文件 `SS.musicxml`：加入显示用的记号，不改音符 |

## album 模式（拍照卡 → 每首一个 MusicXML）

```
sh album/run_album.sh <拍照卡根目录> <输出目录> [--rotate ccw] [--watch 秒] [--songs 0,1]
```

- **输入：** `Meta/SS-PP.json` 中 `complete: true` 的页，读对应的 `Grey/SS-PP.bmp`（传感器原方向，4056×2868）。
- **方向：** 固定逆时针转正（`--rotate ccw`），即原方向图像和拍摄预览中乐谱顶端朝右；Codex 的拍摄界面标有 PAGE TOP。如果一页大多数行谱都没找到谱号，几乎可以肯定放反了，这时 `album.json` 记 `orientation_suspect: true`。`--rotate auto` 保留为备用。
- **输出：**
  - `SS.musicxml`：整首，带谱号、调号、临时记号、音符类型和按拍符杠；
  - `pages/SS-PP_<sha8>_<方向>/`：每页的完整结果；
  - `album.json`：状态记录。
- **续接：** 按 Grey 文件的 sha256 加方向判断，已识别的页跳过。
- **其他行为：** 识别进程常驻，同时最多 2 页（流水线）；一页超过 600 s 没有结果时，重启识别进程并把这页记为失败。
- **Codex 的 Score Studio：** 它通过自己的引擎（`/opt/ac920-panel/engine`）调用本包的 `run_page.py` 和 `album/` 文件。改用本 release 时，按第"给 Codex"一节操作。

## 验证

**打包（`checks/package_vs_v15.json`）：** 共 63 个文件，以 v15 的 pkg 为基础。
- 改动的 5 个文件取自各自的主文件；
- 其余 17 个文件的主文件与 v15 的副本逐字节相同，没有漏掉的改动；
- 新增 `album/` 和两个运行脚本。

**桌面（`checks/desktop_dry.json`，本包 + win64 DLL）：**

| 检查 | 结果 |
|---|---|
| 开发板测试页，v16 参数 | 与 v15 的桌面输出逐字节相同 |
| v13 参数 | 仍为 d01213b3 |
| camera67 的 5 页（score31/32/00/09/35） | 与 v15（sh8s020）逐字节相同 |
| 常驻流水线 5 页（测试页与 camera67 交替） | 与各自单次运行逐字节相同；第 2–5 页四步都由后台完成 |
| album（测试卡：曲目 00 共 6 张镜头对比实拍，曲目 01 为 PDF 第 1–3 页） | 9 页全部用 ccw，谱号找到率 87.5–100%，没有页被判为放反；00-00 与单次运行相同（621db90a）；曲目 01 的 engrave 检查：发声的音符 63/63 小节相同，符杠没有断开 |

**开发板（`checks/board_release_run.json`、`results/board_v16/`）：** 只用 `pkg_v16.tar.gz` 部署到 `/home/fpga/claude-release-v16-20261030`。

| 检查 | 结果 |
|---|---|
| MANIFEST | 63 个文件全部相符 |
| 测试页单次运行 ×2 | f618b20b，与 v15 和桌面相同（忽略换行符差异）；35.6 s、31.0 s |
| camera67 照片单次运行 | b7ce99b3，与 10-27 板测的 v16 相同；19.8 s |
| 常驻流水线 6 页（测试页与照片交替） | 每页都与板上对应的单次运行逐字节相同；测试页第 1 页 30.4 s，之后 20.6 / 21.0 s；照片 13.8 / 14.4 / 10.3 s；峰值内存 1.16 GB |
| album（10-29 测试卡，6 页） | 谱号都 100% 找到，方向 ccw；每页 33–37 s，冷启动首页 47 s；曲目 01 的 engrave 检查 63/63 |
| 板上编译的库 | libdecoder、librelgeo、libpilresize 与 v15 的 prebuilt 逐字节相同；libhomrgeo 已放入 `prebuilt/aarch64/` |

## 目录

| 路径 | 内容 |
|---|---|
| `pkg/` | 开发板程序包：`run_page.py`、`run_board.sh`（单页）、`run_serve.sh`（常驻流水线）、`album/`（拍照卡模式）、`shared/`（sh8s，同 v15）、`sacc/`、`hacc/`、`racc/`、`cdec/`、`dec/`、`lab/`、`weights/`、`MANIFEST.json` |
| `pkg_v16.tar.gz` | 同一个包的 gzip tar，可以直接用 `net.py push` |
| `prebuilt/aarch64/` | 开发板编译的 `.so`：`libdecoder`、`librelgeo`、`libpilresize` 与 v15 相同（源码未变），`libhomrgeo` 是 v16 重新编译的 |
| `prebuilt/win64/` | 桌面试运行用的 DLL |
| `results/board_v16/` | 开发板上的运行记录 |
| `tools/` | PC 端工具（网口、串口）。登录信息不在包里 |
| `docs/` | `OPTIMIZATION_LOG.md`、各实验目录的 README、album、单卡方案、Score Studio 检查、拍照接口、镜头对比 |
| `checks/` | 本 release 的校验记录 |

## 在开发板上运行

```
sh run_board.sh /Image/00-00.png /home/fpga/<新目录>                 # 单页
printf '%s\t%s\n' /Image/00-00.png /home/fpga/<新目录> | sh run_serve.sh   # 常驻流水线，每行一页
sh album/run_album.sh <拍照卡> <输出目录>                              # 拍照卡
```

第一次运行时会编译 4 个 C++ 库（g++ `-O3 -ffp-contract=off`）。也可以把 `prebuilt/aarch64/` 里的 `.so` 放进 `cdec/`、`hacc/`、`racc/`、`sacc/`，省去编译。

## 给 Codex：改用 v16

1. 把 `pkg_v16.tar.gz` 解到开发板的新目录（例如 `/home/fpga/omr-v16`），编译库或放入 prebuilt 库。
2. `/etc/ac920-onecard.json` 的 `recognizer` 改为这个目录，`rotate` 保持 `ccw`。
3. 用 `patch_progress.py` 重新生成 `engine/`。要把 `album/` 下的 `omr_album.py`、`merge_pages.py`、`engrave.py` 都复制过去：`merge_pages.py` 缺少 `engrave.py` 时只做普通合成，不报错。

## 已知限制

- **准确率：** 《月光》这样密集的十六分音符乐谱，实拍完全正确的音符约 66%。密集页会漏小节（PDF 第 2、3 页分别为 19/21、23/24）。
- **模糊照片：** 调号可能认错。16 mm F4 那张镜头对比照片里，有 5 行谱认成了 7 个升号。
- **谱号和调号：** 按每页多数行谱确定；页内换谱号的情况还没处理。
- **ARM 与 x86：** 相机照片在两边的浮点舍入不同，个别符头的判断会翻转，所以开发板和桌面的结果不保证逐字节相同；准确率相当（4 页正确音符合计 1392 / 1384）。电子版测试页两边相同。
