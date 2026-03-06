@echo off
REM EA-NN Container Management Script for Windows
REM Replaces the virtual environment setup with containerized deployment

setlocal enabledelayedexpansion

set "PROJECT_ROOT=%~dp0"
set "GPU_AVAILABLE=false"
set "REBUILD=false"
set "DETACH=false"
set "NO_GPU=false"

echo.
echo 🐳 EA-NN Container Management
echo ===============================
echo.

REM Check Docker
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker not found. Please install Docker Desktop.
    exit /b 1
)
echo ✅ Docker found

REM Check Docker Compose
docker compose version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    where docker-compose >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Docker Compose not found. Please install Docker Compose.
        exit /b 1
    )
)
echo ✅ Docker Compose found

REM Check NVIDIA Docker
nvidia-smi >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo ✅ NVIDIA Docker support detected
        set "GPU_AVAILABLE=true"
    ) else (
        echo ⚠️ NVIDIA Docker not properly configured
    )
) else (
    echo ⚠️ NVIDIA GPU not detected
)

echo.

REM Parse arguments
set "COMMAND="
set "SCALE_COUNT=1"

:parse_args
if "%~1"=="" goto execute_command
if "%~1"=="dev" set "COMMAND=dev" && shift && goto parse_args
if "%~1"=="prod" set "COMMAND=prod" && shift && goto parse_args
if "%~1"=="build" set "COMMAND=build" && shift && goto parse_args
if "%~1"=="stop" set "COMMAND=stop" && shift && goto parse_args
if "%~1"=="clean" set "COMMAND=clean" && shift && goto parse_args
if "%~1"=="logs" set "COMMAND=logs" && shift && goto parse_args
if "%~1"=="shell" set "COMMAND=shell" && shift && goto parse_args
if "%~1"=="--rebuild" set "REBUILD=true" && shift && goto parse_args
if "%~1"=="--detach" set "DETACH=true" && shift && goto parse_args
if "%~1"=="--no-gpu" set "NO_GPU=true" && shift && goto parse_args
if "%~1"=="--help" goto show_usage
if "%~1"=="-h" goto show_usage

echo Unknown option: %~1
goto show_usage

:execute_command
cd /d "%PROJECT_ROOT%"

if "%COMMAND%"=="dev" goto start_dev
if "%COMMAND%"=="prod" goto start_prod
if "%COMMAND%"=="build" goto build_images
if "%COMMAND%"=="stop" goto stop_containers
if "%COMMAND%"=="clean" goto clean_up
if "%COMMAND%"=="logs" goto show_logs
if "%COMMAND%"=="shell" goto open_shell
if "%COMMAND%"=="" goto show_usage

echo Unknown command: %COMMAND%
goto show_usage

:show_usage
echo Usage: %~nx0 [COMMAND] [OPTIONS]
echo.
echo Commands:
echo   dev         Start development environment with DevContainer
echo   prod        Start production deployment
echo   build       Build container images
echo   stop        Stop all containers
echo   clean       Clean up containers and images
echo   logs        Show container logs
echo   shell       Open shell in development container
echo.
echo Options:
echo   --no-gpu    Disable GPU support
echo   --rebuild   Force rebuild of images
echo   --detach    Run in background
echo.
echo Examples:
echo   %~nx0 dev              # Start development environment
echo   %~nx0 prod --detach    # Start production in background
echo   %~nx0 build --rebuild  # Force rebuild all images
goto :eof

:build_images
echo 🏗️ Building EA-NN container images...
if "%REBUILD%"=="true" (
    echo 🔄 Force rebuilding images...
    docker-compose build --no-cache
) else (
    docker-compose build
)
echo ✅ Images built successfully
goto :eof

:start_dev
echo 🚀 Starting EA-NN development environment...
set "GPU_FLAG="
if "%GPU_AVAILABLE%"=="true" if "%NO_GPU%"=="false" (
    echo ✅ Starting with GPU acceleration
) else (
    echo ⚠️ Starting without GPU acceleration
    set "GPU_FLAG=--scale ea-nn-dev=0"
)

if "%DETACH%"=="true" (
    docker-compose --profile development up -d %GPU_FLAG%
) else (
    docker-compose --profile development up %GPU_FLAG%
)
goto :eof

:start_prod
echo 🚀 Starting EA-NN production deployment...
set "GPU_FLAG="
if "%GPU_AVAILABLE%"=="true" if "%NO_GPU%"=="false" (
    echo ✅ Starting with GPU acceleration
) else (
    echo ⚠️ Starting without GPU acceleration
    set "GPU_FLAG=--scale ea-nn-prod=0"
)

docker-compose --profile production up -d %GPU_FLAG%
echo.
echo ✅ Production deployment started!
echo 🌐 Access the simulation at: http://localhost
echo 📊 Monitor GPU usage: docker exec ea-nn-simulation nvidia-smi
goto :eof

:stop_containers
echo 🛑 Stopping EA-NN containers...
docker-compose down
echo ✅ All containers stopped
goto :eof

:clean_up
echo 🧹 Cleaning up EA-NN containers and images...
docker-compose down --rmi all --volumes --remove-orphans
docker image prune -f
echo ✅ Cleanup complete
goto :eof

:show_logs
docker-compose logs -f
goto :eof

:open_shell
echo 🐚 Opening shell in development container...
docker-compose ps | findstr ea-nn-dev | findstr Up >nul
if %ERRORLEVEL% EQU 0 (
    docker-compose exec ea-nn-dev /bin/bash
) else (
    echo Starting development container...
    docker-compose --profile development run --rm ea-nn-dev /bin/bash
)
goto :eof
