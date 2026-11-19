@echo off
setlocal EnableExtensions
rem ============================================================
rem  AIReader 附加包启动器（不改原后端任何文件）
rem  用法：把整个附加包目录解压到 GPT-SoVITS-For-Intel 根目录，
rem        然后双击本文件（或从这里复制 start.cmd 到后端根目录运行）。
rem  自动定位后端：本目录的父目录就是后端根目录（含 runtime\python.exe）。
rem ============================================================
cd /d "%~dp0"

rem 显式告诉代理后端在哪（也支持自动探测，这里给个保险）
set "BSAI_BACKEND_DIR=%~dp0.."

rem 配置文件：优先用同目录的 config_intel.yaml；想用模板请先复制为 config_intel.yaml
if not exist "%~dp0config_intel.yaml" (
  if exist "%~dp0config.addon.example.yaml" (
    echo [提示] 未找到 config_intel.yaml，正在从模板生成...
    copy /y "%~dp0config.addon.example.yaml" "%~dp0config_intel.yaml" >nul
  )
)

rem 用后端自带的运行时（IPEX/XPU 全在这里），不动后端文件
set "PY=%BSAI_BACKEND_DIR%\runtime\python.exe"
if not exist "%PY%" (
  echo [错误] 找不到后端运行时：%PY%
  echo         请把本目录整个解压到 GPT-SoVITS-For-Intel 根目录再运行。
  pause
  exit /b 1
)

set "PATH=%BSAI_BACKEND_DIR%\runtime\Library\bin;%BSAI_BACKEND_DIR%\runtime\Lib\site-packages\intel_extension_for_pytorch\bin;%BSAI_BACKEND_DIR%\runtime;%BSAI_BACKEND_DIR%\runtime\Scripts;%PATH%"
set "PYTHONMALLOC=malloc"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"

echo =============================================
echo   AIReader TTS 附加包  -  原后端零改动
echo   后端目录: %BSAI_BACKEND_DIR%
echo =============================================

"%PY%" "%~dp0server_intel.py"

echo.
echo 附加包已停止。
pause
