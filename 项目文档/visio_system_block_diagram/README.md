# STM32G431 系统框图文件说明

本目录包含“STM32G431 频响测量与自动控制校正系统框图”的 Visio 文件与预览图。

## 文件

- `STM32G431智能频响测量与AI诊断系统框图.vsdx`：推荐使用，Visio 可编辑原生图形文件。
- `STM32G431智能频响测量与AI诊断系统框图.vdx`：Visio XML 兼容备份文件。
- `STM32G431智能频响测量与AI诊断系统框图.svg`：矢量预览，可插入网页或文档。
- `STM32G431智能频响测量与AI诊断系统框图.png`：高清位图预览，适合快速查看或插入报告。
- `build_system_block_diagram.py`：可重复生成上述文件的脚本。
- `visio_base_template.vsdx`：生成 `.vsdx` 所需的最小 Visio 模板。

## 内容结构

框图按五层组织：

1. 被测对象与模拟前端
2. STM32G431 下位机硬件/固件
3. 串口协议与测量数据帧
4. Python 上位机分析平台
5. 结果输出与闭环优化

主数据流展示 DAC 正弦激励、ADC 同步采样、FFT 幅相提取、UART 三数组输出、Bode/Nyquist 分析、滤波器识别、稳定性裕度、自动控制校正、报告导出和智能补扫闭环。

## 重新生成

在仓库根目录执行：

```powershell
python "项目文档\visio_system_block_diagram\build_system_block_diagram.py"
```
