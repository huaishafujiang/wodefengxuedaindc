# STM32G431智能频响测量与AI诊断系统交付包

生成日期：2026-06-26

本目录是最终交付版本，已按上位机、下位机、硬件与项目文档、实测数据与报告分类整理。交付包中保留关键源码、启动脚本、工程配置、烧录文件、图标资源、项目文档和实测数据；已排除 Git/IDE 配置、Python 缓存、PyInstaller 构建中间目录、Keil 历史日志和个人界面状态文件。

## 快速使用

1. Windows 直接运行：
   `01_上位机_软件与应用\Windows可执行程序\STM32G431_AI_Frequency_Response.exe`
2. Python 源码运行兼容 Tkinter 主程序：
   进入 `01_上位机_软件与应用\Python源码\系统分析`，先执行 `pip install -r requirements.txt`，再运行 `python main.py`。
3. Python 源码运行 Qt 上位机界面：
   进入 `01_上位机_软件与应用\Python源码\系统分析`，先执行 `pip install -r requirements.txt` 和 `pip install -r requirements-ui.txt`，再运行 `start_ui.bat` 或 `python main_window.py`。
4. Linux/Ubuntu 源码运行：
   进入 `01_上位机_软件与应用\Python源码\系统分析`，先执行 `bash run_ubuntu.sh`；如需桌面入口，可执行 `bash install_desktop_ubuntu.sh`。
5. 下位机直接烧录：
   使用 `02_下位机_STM32G431固件\固件烧录文件\G431CBTx_sweep.hex`。
6. 下位机二次开发：
   用 Keil/MDK 打开 `02_下位机_STM32G431固件\G431CBTx_sweep_Keil工程\MDK-ARM\G431CBTx_sweep.uvprojx`。

## 串口协议

固件保留三数组输出协议，上位机按以下数组名解析：

- `omega=[...]`
- `Magnitude_data=[...]`
- `Phase_data_rad=[...]`

常用串口命令：

- `PING`
- `HELP`
- `DEFAULT`
- `SWEEP <start_hz> <stop_hz> <step_hz> <amplitude_vpp>`
- `STOP`

## 目录说明

- `00_使用说明`：交付说明和目录清单。
- `01_上位机_软件与应用`：Windows 可执行程序、Python 源码、Windows/Linux 运行脚本和依赖文件。
- `02_下位机_STM32G431固件`：STM32CubeMX/Keil 工程、HAL/CMSIS 驱动、启动文件、烧录 hex 和调试 axf。
- `03_硬件与项目文档`：项目总览、模块化说明、AI 诊断说明、系统框图和程序流程图。
- `04_实测数据与报告`：标准测试数据、八类系统实测原始数据、电路图、现场串口原始帧和已导出的 HTML/CSV/PNG 报告。

## 注意事项

- 下位机当前默认约定：ADC1/PA0 为输入参考，ADC2/PA1 为输出响应，传递函数幅值为输出/输入。
- 本版 Windows exe 已于 2026-06-26 使用 `D:\ai\系统分析` 最新源码重新构建。
- Linux/Ubuntu 运行路径已保留 `requirements-linux.txt`、`run_ubuntu.sh`、`build_ubuntu.sh` 和桌面入口脚本。
- 固件烧录文件来自当前工程已有 Keil 构建输出；如修改固件源码，请重新在 Keil 中编译并生成新的 hex。
- 若 Windows exe 被安全软件拦截，可改用 Python 源码方式运行。
