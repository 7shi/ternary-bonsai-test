@echo off
setlocal EnableExtensions

pushd "%~dp0"

set "TOKENS=%~1"
if "%TOKENS%"=="" set "TOKENS=10"

echo Running all benchmark combinations with max-new-tokens=%TOKENS%
echo.

call :run_safetensors "safetensors_fp16 / CPU"
rem call :run_combo cpu fp32 "model_fp32 / CPU"
rem call :run_combo dml fp32 "model_fp32 / DirectML"
call :run_combo cpu fp16 "model_fp16 / CPU"
call :run_combo dml fp16 "model_fp16 / DirectML"
call :run_combo cpu fp8 "model_fp8 / CPU"
call :run_combo dml fp8 "model_fp8 / DirectML"
call :run_combo cpu q2fp8 "model_q2_to_fp8 / CPU"
call :run_combo dml q2fp8 "model_q2_to_fp8 / DirectML"
call :run_combo cpu q8 "model_q2_to_q8 / CPU"
call :run_combo dml q8 "model_q2_to_q8 / DirectML"
call :run_combo cpu q4 "model_q2_to_q4 / CPU"
call :run_combo dml q4 "model_q2_to_q4 / DirectML"

echo All combinations finished.
popd
exit /b 0

:run_safetensors
set "LABEL=%~1"

echo ============================================================
echo %LABEL%
echo Command: uv run run_safetensors.py --max-new-tokens %TOKENS%
echo ============================================================
uv run run_safetensors.py --max-new-tokens %TOKENS%
set "STATUS=%ERRORLEVEL%"

if not "%STATUS%"=="0" (
    echo.
    echo WARNING: %LABEL% failed with exit code %STATUS%.
)

echo.
exit /b 0

:run_combo
set "PROVIDER=%~1"
set "MODEL_SOURCE=%~2"
set "LABEL=%~3"

echo ============================================================
echo %LABEL%
echo Command: uv run run_onnx.py --provider %PROVIDER% --model-source %MODEL_SOURCE% --max-new-tokens %TOKENS%
echo ============================================================
uv run run_onnx.py --provider %PROVIDER% --model-source %MODEL_SOURCE% --max-new-tokens %TOKENS%
set "STATUS=%ERRORLEVEL%"

if not "%STATUS%"=="0" (
    echo.
    echo WARNING: %LABEL% failed with exit code %STATUS%.
)

echo.
exit /b 0
