# 系统分析目录说明

## 一键启动入口

这些文件保留在当前目录，不要移动：

- `start_app.bat`：启动主程序入口。
- `start_ui.bat`：启动 UI 入口。
- `main.py`：主程序 Python 入口。
- `main_window.py`：UI 兼容入口。
- `app.py`：应用兼容入口。

## 主要源码

- `system_analysis/`：主程序源码目录。
- `system_analysis/analysis/`：滤波器识别、诊断、拟合、传递函数等分析逻辑。
- `system_analysis/io/`：串口协议、扫频读取、数据输入输出。
- `system_analysis/gui/`：Tk/Qt 界面和扫频工作线程。
- `system_analysis/reporting/`：绘图、报告导出、展示层处理。
- `system_analysis/reporting/presentation.py`：只用于演示图的显示层小补丁，基于实测数据做平滑和高通低频噪声底隐藏，不改变底层分析、数据解析、阶次判断或原始数组。

## 兼容转发文件

根目录下的 `filter_analysis.py`、`plotting.py`、`serial_protocol.py` 等小文件是兼容旧脚本用的转发入口，实际实现已经在 `system_analysis/` 内部。

## 非源码产物

- `_artifacts/`：构建输出、缓存、历史打包产物等非源码内容。
- `assets/`：程序资源文件。

测量数据和演示输出主要在：

- `D:\ai\测量数据\PCB数据`
- `D:\ai\测量数据\实测原始`

## 当前绘图原则

主程序仍以实测的 `omega`、`Magnitude_data`、`Phase_data_rad` 为准。绘图优化只发生在展示层，让曲线更适合演示；不会反向修改底层分析结果、串口数据解析、阶次判断或导出的原始数据。
