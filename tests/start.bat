@echo off
setlocal enabledelayedexpansion
title videoAI Launcher

REM ============================================================
REM videoAI One-Click Launcher
REM ============================================================

REM Get script location and go to project root
set "HERE=%~dp0"
cd /d "%HERE%.."

echo.
echo ============================================================
echo   videoAI Launcher
echo   Project: %CD%
echo ============================================================

REM Check Python
echo.
echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)
python --version

REM Check data
echo.
echo Checking data files...
if exist "data\data\campus_knowledge_base.json" (echo [OK] Knowledge base) else (echo [MISS] Knowledge base)
if exist "data\data\campus_test_qa.json" (echo [OK] Test QA) else (echo [MISS] Test QA)

REM Quick verify
echo.
echo Running quick verify...
python tests\main.py --mode verify

REM Menu
:MENU
echo.
echo ============================================================
echo   [1] Verify env   [2] Process video   [3] Integration test
echo   [4] Full pipeline [5] Interactive QA  [6] Web UI
echo   [0] Exit
echo ============================================================
set /p MODE="Choose [0-6]: "

if "%MODE%"=="0" goto END
if "%MODE%"=="1" python tests\main.py --mode verify && pause && goto MENU
if "%MODE%"=="2" python tests\main.py --mode process && pause && goto MENU
if "%MODE%"=="3" python tests\integration_test.py --skip-llm --quick && pause && goto MENU
if "%MODE%"=="4" python tests\main.py --mode pipeline && pause && goto MENU
if "%MODE%"=="5" python tests\main.py --mode interactive && goto MENU
if "%MODE%"=="6" (
    echo Starting Web UI at http://127.0.0.1:7860
    cd llm_gradio
    python app.py
    cd ..
    goto MENU
)
echo Unknown option: %MODE%
goto MENU

:END
echo Done.
popd
endlocal
