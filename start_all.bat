@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "INTEL_ROOT=D:\BSAIReader\GPT-SoVITS-For-Intel"
set "PATH=%INTEL_ROOT%\runtime\Library\bin;%INTEL_ROOT%\runtime\Lib\site-packages\intel_extension_for_pytorch\bin;%INTEL_ROOT%\runtime;%INTEL_ROOT%\runtime\Scripts;%PATH%"

set "PYTHONMALLOC=malloc"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"

echo =============================================
echo   AI TTS Server - Intel XPU
echo =============================================

for %%p in (9880 9881 9882) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%p.*LISTENING" 2^>nul') do (
        echo Killing old backend PID %%a ^(port %%p^)...
        taskkill /f /pid %%a >nul 2>&1
    )
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000.*LISTENING" 2^>nul') do (
    echo Killing old proxy PID %%a...
    taskkill /f /pid %%a >nul 2>&1
)

echo Starting server (proxy + backend auto-managed)...
echo   https://localhost:8000
echo   https://ai.sunflv.top:39281
echo =============================================

"%INTEL_ROOT%\runtime\python.exe" "%~dp0server_intel.py"

echo.
echo Server stopped.
pause
