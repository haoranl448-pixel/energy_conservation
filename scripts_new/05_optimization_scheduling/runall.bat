@echo off
:: 设置字符集为简体中文 (GBK) 避免乱码
chcp 936 >nul
title 轨道交通节能优化自动化流水线

:: ================= 1. 路径配置 =================
:: 虚拟环境的 Python 路径
set PYTHON_EXE=d:\energy_conservation\.venv\Scripts\python.exe
:: 项目所在的根目录
set PROJECT_DIR=d:\energy_conservation

echo ================================================
echo    🚀 启动自动化流水线 (训练 + 优化)
echo    时间: %date% %time%
echo ================================================

:: ================= 2. 运行残差模型训练 =================
echo.
echo [第一步] 正在启动：残差模型训练...
"%PYTHON_EXE%" "%PROJECT_DIR%\scripts\train_residual_new.py"

:: 检查上一步是否成功 (ERRORLEVEL 0 表示成功)
if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ 错误：模型训练失败，流水线已终止。
    goto :failed
)

echo.
echo ✅ 模型训练已完成，准备开始下一步...
timeout /t 3 >nul

:: ================= 3. 运行全线能耗优化 =================
echo.
echo [第二步] 正在启动：全线能耗优化 (Trip 6)...
"%PYTHON_EXE%" "%PROJECT_DIR%\scripts\run_optimize_residual_new.py"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ 错误：能耗优化脚本运行失败。
    goto :failed
)
echo.
echo ✅ 模型训练已完成，准备开始下一步...
timeout /t 3 >nul

:: ================= 3. 运行全线能耗优化 =================
echo.
echo [第三步] 正在启动：特殊区间 (Trip 6)...
"%PYTHON_EXE%" "%PROJECT_DIR%\scripts\run_optimize_special_residual_new.py"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ 错误：能耗优化脚本运行失败。
    goto :failed
)


:: ================= 4. 结束逻辑 =================
echo.
echo ================================================
echo    🎉 恭喜！全流程运行成功！
echo    所有 6 类结果文件已生成至 output 目录。
echo ================================================
pause
exit

:failed
echo.
echo ================================================
echo    🛑 流程由于错误意外中断，请检查上方日志。
echo ================================================
pause