@echo off
setlocal EnableExtensions

pushd "%~dp0"

set "TOKENS=%~1"
if "%TOKENS%"=="" set "TOKENS=10"

echo Running all model/provider combinations with max-new-tokens=%TOKENS%
echo.

call :run_combo cpu fp32 "fp32 export / CPU"
call :run_combo dml fp32 "fp32 export / DirectML"
call :run_combo cpu fp16 "fp16 export / CPU"
call :run_combo dml fp16 "fp16 export / DirectML"
call :run_combo cpu fp8 "fp8 conversion / CPU"
call :run_combo dml fp8 "fp8 conversion / DirectML"
call :run_combo cpu q2fp8 "q2->fp8 conversion / CPU"
call :run_combo dml q2fp8 "q2->fp8 conversion / DirectML"
call :run_combo cpu q8 "q8 conversion / CPU"
call :run_combo dml q8 "q8 conversion / DirectML"

echo All combinations finished.
popd
exit /b 0

:run_combo
set "PROVIDER=%~1"
set "MODEL_SOURCE=%~2"
set "LABEL=%~3"

echo ============================================================
echo %LABEL%
echo Command: uv run run_directml.py --provider %PROVIDER% --model-source %MODEL_SOURCE% --max-new-tokens %TOKENS%
echo ============================================================
uv run run_directml.py --provider %PROVIDER% --model-source %MODEL_SOURCE% --max-new-tokens %TOKENS%
set "STATUS=%ERRORLEVEL%"

if not "%STATUS%"=="0" (
    echo.
    echo WARNING: %LABEL% failed with exit code %STATUS%.
)

echo.
exit /b 0
