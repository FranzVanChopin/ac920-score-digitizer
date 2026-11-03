# 请 Codex 修改：固定方向与拍摄界面的"上方"标记（2026-10-30）

用户决定：**方向固定为逆时针旋转（`rotate: ccw`），不做自动判断**；在拍摄界面上标出乐谱上方应朝哪边，由拍摄的人把纸放对。

## 1. 约定

- 识别端把 Grey 图逆时针转 90° 得到正立的乐谱，所以**在传感器原方向的图像里，乐谱顶端必须朝右边**。
- 拍摄界面的预览与 Grey 文件方向相同：`show_preview()` 用 `preview_letterbox` 把 `a4_frame_source` 的 A4 裁切直接缩放到 (248, 100) 起、784×552 的预览框，没有旋转，也没有镜像。所以**在预览里，乐谱顶端也朝右边**。
- 10-29 镜头对比的样片就是这样放的，转正后都是正立的。
- 10-30 的三张测试照片放反了，顶端朝左，用 ccw 转出来是倒的：识别出 21 个小节，但正确音符只有 0.6%，谱号、调号都没找到。

## 2. 拍摄界面的标记（`main.c` 的 `studio_overlay`）

只在普通拍摄模式下显示（`!focus_active`），对焦模式不显示，免得挡住放大的区域。

1. **右边缘色条：** 预览框内侧最右的 6 px，取中间一段（上下各留 40 px），用强调色 `accent`（0x007aff）。
2. **标签卡片：** 在预览右边缘的垂直中点，放一个白色圆角卡片，写上 `PAGE TOP`，旁边画一个指向右边的三角形箭头。
   - 界面字体是英文位图字体，所以标签用英文；
   - 箭头用 `studio_rect` 逐行画，不依赖字体里有没有箭头字符。
3. **左栏说明：** 在 "Capture format / A4 / Native" 下面加一行 `Top edge: right`。

示例。坐标只是建议，用的都是现有的辅助函数，字号和间距请 Codex 按 Light1 的风格调整：

```c
    /* Orientation guide: the recognizer turns the Grey image counter-clockwise (rotate=ccw),
     * so the page top must face the RIGHT edge of this native-orientation preview. */
    if(!focus_active) {
        const unsigned right=PREVIEW_X+PREVIEW_W, mid=PREVIEW_TOP+PREVIEW_H/2u;
        studio_rect(dest,right-6u,PREVIEW_TOP+40u,6u,PREVIEW_H-80u,accent);
        studio_rounded(dest,right-158u,mid-22u,144u,44u,10u,card);
        studio_text(dest,right-148u,mid-12u,"PAGE TOP",2,blue_ink);
        for(y=0;y<17u;++y) {                       /* right-pointing triangle, 9 px deep */
            unsigned half=y<9u?y:16u-y;
            studio_rect(dest,right-36u,mid-8u+y,half+1u,1u,accent);
        }
    }
```

（`x`、`y`、`card`、`accent`、`blue_ink` 都是 `studio_overlay` 里已有的局部变量。）

建议实拍确认一次：把纸的顶端放向箭头一侧，拍一页，在 Linux 界面识别后，日志里应显示每行谱的谱号都找到了（clefs 100%）。

## 3. Linux 界面：提示"这一页可能放反了"

识别端的新版 `album/omr_album.py` 在 `album.json` 里为每页记录：
- `orientation_suspect`：大多数行谱都没找到谱号时为 `true`，几乎可以肯定是放反了；
- `clef_ratio`：找到谱号的行谱比例；
- `rotate_used`：所用的方向。

日志里也会打印提示。建议识别界面在这一页旁边显示："这一页可能放反了：请把乐谱上方朝向拍摄界面的箭头（右侧）重拍"。

新版要点：
- 默认方向改为 `ccw`；`--rotate auto` 仍然保留，可以作为备用；
- 换方向后会自动重新识别（结果按方向分别缓存）；
- Codex `patch_progress.py` 用到的锚点 `if r.get("ready"):` 仍只出现一次。

请把新版换进引擎（`config['recognizer']` 目录下的 `album/`），然后重新生成 `engine/omr_album.py`。

## 4. 顺带发现：重启服务会卡住

`sudo systemctl restart ac920-panel` 之后，控制器一直重启失败（43 次，exit 1）。原因是：
- 旧进程被停掉时，没有卸载 `/mnt/ac920-onecard`（mmcblk1p1 只读挂载）；
- 新进程在 `Controller.start()` 里检测到 FAT 已挂载，就报 `boot FAT already mounted` 退出；
- systemd 每 5 s 重试一次，每次都失败。

10-30 的日志（`board_trial/board_studio/logs/panel.log`）再次印证了这一点：10:55:35 重启后控制器显示 READY，挂上了卡；10:55:44 又一次被停止。它在 SIGTERM 下直接退出，`main()` 的 `finally: controller.close()` 没有执行，卡一直挂着，之后每 5 s 一次的启动都失败了。手动 `umount` 一次后，10:58:43 恢复正常。

建议二选一：
- 在 SIGTERM 时卸载 FAT（加上 `ExecStopPost=/bin/umount /mnt/ac920-onecard`，或在代码里处理）；
- 或在 `start()` 里发现挂载点是自己的 `/mnt/ac920-onecard` 并且是只读时，先卸载再继续。
