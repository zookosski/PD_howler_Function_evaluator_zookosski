@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo PD HOWLER AI BRIDGE - COMMUNITY SETUP
echo ================================================================
echo.
echo This script will:
echo - Prompt for your installation paths
echo - Create directory structure  
echo - Install Python dependencies
echo - Configure scripts with your paths
echo - Test the installation
echo.
echo Make sure you have the following files in this directory:
echo - pd_howler_ai_bridge_template.py
echo - 1_Depth_REMBG_PD_template.lua
echo.
pause

echo.
echo [1/6] Getting installation paths...
echo.

REM Get Python Bridge installation path
echo Where would you like to install the Python Bridge?
echo.
echo Examples:
echo   C:\PD_Howler_AI_Bridge
echo   D:\Tools\PD_Howler_Bridge  
echo   C:\Users\%USERNAME%\Documents\PD_Howler_AI
echo   C:\AI_Tools\PD_Howler_Bridge
echo.
set /p "BRIDGE_PATH=Enter full path (or press Enter for default): "

if "%BRIDGE_PATH%"=="" set "BRIDGE_PATH=C:\PD_Howler_AI_Bridge"

REM Clean the path (remove quotes if present)
set "BRIDGE_PATH=%BRIDGE_PATH:"=%"

echo.
echo Selected Python Bridge path: %BRIDGE_PATH%
echo.

REM Get PD Howler Scripts path
echo Where is your PD Howler Scripts folder?
echo.
echo Common locations:
echo   C:\Program Files\Project Dogwaffle\Scripts
echo   C:\Program Files (x86)\Project Dogwaffle\Scripts  
echo   C:\PD_Howler\Scripts
echo   [Your Custom PD Howler Path]\Scripts
echo.
echo You can also browse to find it:
echo 1. Open PD Howler
echo 2. Go to Filter menu ^> Scripts
echo 3. The folder should be where .lua files are located
echo.
set /p "HOWLER_SCRIPTS_PATH=Enter PD Howler Scripts full path: "

REM Clean the path
set "HOWLER_SCRIPTS_PATH=%HOWLER_SCRIPTS_PATH:"=%"

echo.
echo ================================================================
echo CONFIGURATION SUMMARY
echo ================================================================
echo Python Bridge will be installed to: %BRIDGE_PATH%
echo Lua script will be copied to: %HOWLER_SCRIPTS_PATH%
echo.
echo Directory structure will be:
echo %BRIDGE_PATH%\
echo ├── pd_howler_ai_bridge.py (configured)
echo ├── Temp\ (working files)
echo ├── Output\ (saved results)
echo ├── Models\ (cached AI models)
echo └── config.txt (installation info)
echo.
echo %HOWLER_SCRIPTS_PATH%\
echo └── 1_Depth_REMBG_PD.lua (configured)
echo.
echo Is this correct? (Y/N)
set /p "CONFIRM=Enter Y to continue, N to restart: "

if /i not "%CONFIRM%"=="Y" (
    echo Setup cancelled. Please run again with correct paths.
    pause
    exit /b 1
)

echo.
echo [2/6] Creating directory structure...
if not exist "%BRIDGE_PATH%" mkdir "%BRIDGE_PATH%"
if not exist "%BRIDGE_PATH%\Temp" mkdir "%BRIDGE_PATH%\Temp"
if not exist "%BRIDGE_PATH%\Output" mkdir "%BRIDGE_PATH%\Output"
if not exist "%BRIDGE_PATH%\Models" mkdir "%BRIDGE_PATH%\Models"

REM Verify PD Howler Scripts directory exists
if not exist "%HOWLER_SCRIPTS_PATH%" (
    echo ⚠️ PD Howler Scripts directory does not exist: %HOWLER_SCRIPTS_PATH%
    echo Creating directory...
    mkdir "%HOWLER_SCRIPTS_PATH%"
    if errorlevel 1 (
        echo ❌ Failed to create PD Howler Scripts directory
        echo Please check the path and permissions
        pause
        exit /b 1
    )
)

echo ✅ Directories created successfully!

echo.
echo [3/6] Testing Python installation...
python --version
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.7+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    echo After installing Python, run this setup script again.
    pause
    exit /b 1
)
echo ✅ Python is available!

echo.
echo [4/6] Installing Python dependencies...
echo This may take 5-15 minutes on first run...
echo.

REM Install core dependencies first
echo Installing core dependencies (OpenCV, NumPy, Pillow)...
pip install opencv-python numpy pillow
if errorlevel 1 (
    echo ❌ Failed to install core dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

REM Install PyTorch CPU version
echo.
echo Installing PyTorch (CPU version)...
echo This is the largest download and may take several minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ⚠️ PyTorch installation failed - AI features will be limited
    echo You can continue, but only OpenCV depth mapping will work
)

REM Install AI model libraries
echo.
echo Installing AI model libraries...
pip install transformers
if errorlevel 1 (
    echo ⚠️ Transformers installation failed - Depth Anything V2 will be limited
)

pip install rembg
if errorlevel 1 (
    echo ⚠️ RemBG installation failed - Background removal will be limited
)

echo ✅ Dependencies installation completed!

echo.
echo [5/6] Configuring scripts with your paths...

REM Check for template files
if not exist "pd_howler_ai_bridge_template.py" (
    echo ❌ Template file not found: pd_howler_ai_bridge_template.py
    echo Please make sure all template files are in the current directory
    pause
    exit /b 1
)

if not exist "1_Depth_REMBG_PD_template.lua" (
    echo ❌ Template file not found: 1_Depth_REMBG_PD_template.lua  
    echo Please make sure all template files are in the current directory
    pause
    exit /b 1
)

echo Configuring Python bridge script...
REM Configure Python script - replace forward slashes for Python paths
set "BRIDGE_PATH_PYTHON=%BRIDGE_PATH:\=/%"
powershell -Command "(Get-Content 'pd_howler_ai_bridge_template.py') -replace '{{BRIDGE_PATH_PLACEHOLDER}}', '%BRIDGE_PATH_PYTHON%' | Set-Content '%BRIDGE_PATH%\pd_howler_ai_bridge.py'"
if errorlevel 1 (
    echo ❌ Failed to configure Python script
    pause
    exit /b 1
)
echo ✅ Python script configured and copied to %BRIDGE_PATH%

echo Configuring Lua script...
REM Configure Lua script - use double backslashes for Lua paths  
set "BRIDGE_PATH_LUA=%BRIDGE_PATH:\=\\%"
powershell -Command "(Get-Content '1_Depth_REMBG_PD_template.lua') -replace '{{BRIDGE_PATH_PLACEHOLDER}}', '%BRIDGE_PATH_LUA%' | Set-Content '%HOWLER_SCRIPTS_PATH%\1_Depth_REMBG_PD.lua'"
if errorlevel 1 (
    echo ❌ Failed to configure Lua script
    pause
    exit /b 1
)
echo ✅ Lua script configured and copied to %HOWLER_SCRIPTS_PATH%

REM Create configuration file for future reference
echo Creating configuration file...
echo # PD Howler AI Bridge Configuration > "%BRIDGE_PATH%\config.txt"
echo # Generated on %DATE% at %TIME% >> "%BRIDGE_PATH%\config.txt"
echo # >> "%BRIDGE_PATH%\config.txt"
echo # Installation Paths: >> "%BRIDGE_PATH%\config.txt"
echo BRIDGE_PATH=%BRIDGE_PATH% >> "%BRIDGE_PATH%\config.txt"
echo HOWLER_SCRIPTS_PATH=%HOWLER_SCRIPTS_PATH% >> "%BRIDGE_PATH%\config.txt"
echo # >> "%BRIDGE_PATH%\config.txt"
echo # Configured Files: >> "%BRIDGE_PATH%\config.txt"
echo PYTHON_SCRIPT=%BRIDGE_PATH%\pd_howler_ai_bridge.py >> "%BRIDGE_PATH%\config.txt"
echo LUA_SCRIPT=%HOWLER_SCRIPTS_PATH%\1_Depth_REMBG_PD.lua >> "%BRIDGE_PATH%\config.txt"
echo # >> "%BRIDGE_PATH%\config.txt"
echo # To reconfigure: >> "%BRIDGE_PATH%\config.txt"
echo # 1. Edit paths in the configured files above >> "%BRIDGE_PATH%\config.txt"
echo # 2. Or run setup_community.bat again >> "%BRIDGE_PATH%\config.txt"

echo.
echo [6/6] Testing installation...
echo Testing core libraries...
python -c "import cv2, numpy; print('✅ Core libraries working!')"
if errorlevel 1 (
    echo ❌ Core libraries test failed
    pause
    exit /b 1
)

REM Test AI libraries
echo Testing AI libraries...
python -c "try: import torch; print('✅ PyTorch available'); except: print('⚠️ PyTorch not available')"
python -c "try: import transformers; print('✅ Transformers available'); except: print('⚠️ Transformers not available')"  
python -c "try: import rembg; print('✅ RemBG available'); except: print('⚠️ RemBG not available')"

REM Test configured Python script
echo Testing configured Python bridge...
cd /d "%BRIDGE_PATH%"
python -c "exec(open('pd_howler_ai_bridge.py').read().split('def main')[0]); print('✅ Python bridge configuration OK')"
if errorlevel 1 (
    echo ⚠️ Python bridge configuration test failed
    echo This might still work, but check the configuration manually
)

echo.
echo ================================================================
echo 🎉 INSTALLATION COMPLETE!
echo ================================================================
echo.
echo Installation Summary:
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ✅ Python Bridge: %BRIDGE_PATH%
echo ✅ Lua Script: %HOWLER_SCRIPTS_PATH%\1_Depth_REMBG_PD.lua
echo ✅ Configuration: %BRIDGE_PATH%\config.txt
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo 🚀 Quick Test:
echo 1. Open PD Howler
echo 2. Create or open an image
echo 3. Go to: Filter ^> Scripts ^> 1_Depth_REMBG_PD
echo 4. Choose algorithm 1 (OpenCV) for first test
echo 5. Click OK and wait for processing
echo.
echo 📁 Files and Folders:
echo - Working files: %BRIDGE_PATH%\Temp\
echo - Saved results: %BRIDGE_PATH%\Output\  
echo - AI models cache: %BRIDGE_PATH%\Models\
echo - Debug images: %BRIDGE_PATH%\Temp\debug_*.png
echo.
echo 🔧 Troubleshooting:
echo - If script doesn't appear in PD Howler, check Scripts folder path
echo - If Python fails, check paths in config.txt
echo - For detailed help, see Installation Guide
echo - First AI model download may take 5-10 minutes
echo.
echo 🎨 Algorithms Available:
echo   1 = OpenCV Depth (Fast) - Always works
echo   2 = MiDaS AI Depth (Professional)
echo   3 = Depth Anything V2 (State-of-the-art) 
echo   4 = Basic RemBG (Simple background removal)
echo   5 = AI RemBG (Smart background removal)
echo   6 = InSPyReNet (Highest quality background removal)
echo.
echo Configuration saved in: %BRIDGE_PATH%\config.txt
echo.
echo Happy creating! 🎨✨
echo.
pause
