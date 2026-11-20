@echo off
setlocal EnableExtensions
rem ============================================================
rem  AI TTS Server - Intel XPU proxy + NVIDIA fp16 pool
rem  All-ASCII on purpose: cmd parses .bat with the OEM codepage,
rem  so non-ASCII comments get mangled into garbage commands.
rem ============================================================
cd /d "%~dp0"

set "INTEL_ROOT=D:\BSAIReader\GPT-SoVITS-For-Intel"
set "PATH=%INTEL_ROOT%\runtime\Library\bin;%INTEL_ROOT%\runtime\Lib\site-packages\intel_extension_for_pytorch\bin;%INTEL_ROOT%\runtime;%INTEL_ROOT%\runtime\Scripts;%PATH%"

set "PYTHONMALLOC=malloc"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "NO_PROXY=127.0.0.1,localhost,::1"
set "no_proxy=127.0.0.1,localhost,::1"

echo =============================================
echo   AI TTS Server - Intel XPU + NVIDIA fp16
echo =============================================

rem Kill stale backend / pool / proxy processes on our ports
for %%p in (9880 9881 9882 9883 9891 8000) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%p.*LISTENING" 2^>nul') do (
        echo Killing old process PID %%a ^(port %%p^)...
        taskkill /f /pid %%a >nul 2>&1
    )
)

rem Second GPU: P106-90 v5Turbo fp16 pool (started only when runtime_cuda exists)
if exist "%INTEL_ROOT%\runtime_cuda\python.exe" (
    echo Starting NVIDIA fp16 pool on 9891...
    start "BSAI NVIDIA fp16" /min cmd /c ""%INTEL_ROOT%\start_nvidia_fp16.bat""
) else (
    echo [info] runtime_cuda not found - skipping P106 pool
)

echo Starting server (proxy + backend auto-managed)...
echo   https://localhost:8000
echo   https://ai.sunflv.top:39281
echo =============================================

"%INTEL_ROOT%\runtime\python.exe" "%~dp0server_intel.py"

echo.
echo Server stopped.
pause
