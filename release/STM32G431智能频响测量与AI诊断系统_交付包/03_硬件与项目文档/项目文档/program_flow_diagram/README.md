# STM32G431 程序流程图文件说明

本目录是根据当前 `系统分析/app.py`、`ai_diagnosis.py`、`control_compensation.py`、`smart_resweep_reader.py` 等代码重新梳理的程序流程图。

## 文件

- `STM32G431程序流程图.vsdx`：Visio 可编辑版，推荐使用。
- `STM32G431程序流程图.vdx`：Visio XML 兼容备份。
- `STM32G431程序流程图.png`：高清预览图。
- `STM32G431程序流程图.svg`：矢量预览图。
- `build_program_flow_diagram.py`：可重复生成上述文件的脚本。
- `visio_base_template.vsdx`：生成 `.vsdx` 所需的基础模板。

## 当前代码主流程

1. 用户启动上位机。
2. 选择串口扫频或文本导入。
3. 统一进入 `handle_frame()`，创建 `MeasurementSession` 并投递后台分析线程。
4. `_analysis_worker()` 读取分析设置，调用 `analyze_system_v2()` 完成频响、Nyquist 与稳定性分析。
5. 调用 `run_intelligent_diagnosis()` 完成特征提取、模板拟合、故障规则判断与补扫计划生成。
6. 若启用自动控制校正，调用 `build_control_compensation_report()` 计算 PM/GM 和校正器参数。
7. 若需要补扫，`SmartResweepReader` 执行 1-3 条 `SWEEP` 并合并测量帧后重新分析。
8. 生成报告文本、刷新界面，可导出 HTML、CSV 和 PNG。

## 重新生成

在仓库根目录执行：

```powershell
python "项目文档\program_flow_diagram\build_program_flow_diagram.py"
```
