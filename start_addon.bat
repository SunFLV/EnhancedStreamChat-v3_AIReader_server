@echo off
setlocal EnableExtensions
rem ============================================================
rem  AIReader addon launcher (does not touch the backend files)
rem  Usage: extract the whole addon folder next to the backend
rem  root (the parent of this folder must contain runtime\python.exe),
rem  then run this file.
rem  All-ASCII on purpose: cmd parses .bat with the OEM codepage,
rem  so non-ASCII comments get mangled into garbage commands.
rem ============================================================
cd /d "%~dp0"

rem Tell the proxy where the backend lives (auto-detect works too)
set "BSAI_BACKEND_DIR=%~dp0.."

rem Config: prefer config_intel.yaml next to this script; fall back
rem to the bundled template when it is missing.
if not exist "%~dp0config_intel.yaml" (
  if exist "%~dp0config.addon.example.yaml" (
    echo [info] config_intel.yaml not found - generating from template...
    copy /y "%~dp0config.addon.example.yaml" "%~dp0config_intel.yaml" >nul
  )
)

rem Use the runtime bundled with the backend (IPEX/XPU all inside)
set "PY=%BSAI_BACKEND_DIR%\runtime\python.exe"
if not exist "%PY%" (
  echo [ERROR] backend runtime not found: %PY%
  echo         Extract this folder next to the GPT-SoVITS-For-Intel root first.
  pause
  exit /b 1
)

set "PATH=%BSAI_BACKEND_DIR%\runtime\Library\bin;%BSAI_BACKEND_DIR%\runtime\Lib\site-packages\intel_extension_for_pytorch\bin;%BSAI_BACKEND_DIR%\runtime;%BSAI_BACKEND_DIR%\runtime\Scripts;%PATH%"
set "PYTHONMALLOC=malloc"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
rem Loopback addresses must bypass any system HTTP proxy, or /api/servers
rem probing would falsely report the 9891 pool as offline.
set "NO_PROXY=127.0.0.1,localhost,::1"
set "no_proxy=127.0.0.1,localhost,::1"

echo =============================================
echo   AIReader TTS addon
echo   Backend dir: %BSAI_BACKEND_DIR%
echo =============================================

"%PY%" "%~dp0server_intel.py"

echo.
echo Addon stopped.
pause
