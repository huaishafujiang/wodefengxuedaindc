STM32G431 智能频响分析仪 - 新 PySide6 UI 启动与打包说明
========================================================

说明
----
新 UI 是独立的审查入口：main_window.py。
旧版 Tkinter 入口 main.py、start_app.bat、build_exe.bat 保持不变。

当前状态
--------
1. 新 UI 目前使用 SweepWorker 的 Mock 扫频数据，用于审查纯白极简仪器控制台界面。
2. 串口协议仍保持 omega、Magnitude_data、Phase_data_rad 三数组格式。
3. PySide6 依赖单独放在 requirements-ui.txt，不强行加入旧版 requirements.txt。

直接启动新 UI
------------
第一次运行前，双击：

setup_ui_env.bat

安装完成后，双击：

start_ui.bat

也可以在命令行运行：

python -m pip install -r requirements-ui.txt
python main_window.py

打包为 Windows 可执行软件
-------------------------
双击：

build_ui_exe.bat

打包完成后，可执行文件位于：

dist\STM32G431_AI_Frequency_Response.exe

注意
----
如果 PyInstaller 打包失败，先确认 python main_window.py 可以正常启动。
如果需要接入真实硬件，请优先替换 sweep_worker.py 里的 Mock 扫频段，不要改动三数组协议名称。
