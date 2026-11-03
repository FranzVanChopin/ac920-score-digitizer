---
name: pynq-artifact-check
description: "为任意 PYNQ Overlay 做离线 bit/HWH 配对、地址元数据与设备期望预检，不加载位流、不访问 MMIO。"
---

# 通用 PYNQ Overlay 文件预检

## 适用与输入

用于准备或排查 PYNQ Overlay 交接。输入为同目录、同主文件名的 `.bit` 与 `.hwh`，以及可选的期望 DEVICE 和可信构建清单；不依赖本项目板型、图像分辨率或地址常量。

## 使用

```sh
python scripts/check_artifacts.py --bit <OVERLAY.bit> --hwh <OVERLAY.hwh> --expected-device <DEVICE>
```

脚本只读取文件，计算 SHA-256，解析 HWH 的 SYSTEMINFO 与 MEMRANGE，并核对范围合法性、同名同目录及可选器件期望。DEVICE 参数使用 HWH 的器件短名格式。输出必须与可信构建清单核对：文件同名本身不能证明它们来自同次构建。

离线预检通过后，如需 PYNQ 运行时检查，先确认目标系统确为 PYNQ，再依照其官方说明使用 `Overlay(path, download=False)` 读取元数据。该步骤未在本技能交付时执行。实际 `download()`、MMIO 读写、复位、DMA 或内存测试属于另一个明确的设备操作任务，不能由离线通过结果自动触发。

## 失败与限制

缺失文件、同名不一致、HWH 不可解析、地址倒置或器件期望不符时停止。脚本不解析 bitstream 的器件头、不证明电气/时钟/驱动兼容性、不检查目标板连接，不覆盖 XCLBIN 或局部重配置场景。陌生 HWH 格式先增加解析适配与测试，不能忽略异常继续加载。

## 验证与来源

来源是硬件交接身份和 AXI 地址核对经验，并按 [PYNQ 官方 Overlay 说明](https://pynq.readthedocs.io/en/v3.1/pynq_overlays/loading_an_overlay.html)整理。已用真实 Vivado XSA 中的 bit/HWH 对进行离线检查，并测试同名/器件错误；**尚未完成 PYNQ 板上运行验收**，不能作为已验证的 PYNQ 完整部署案例申报。
