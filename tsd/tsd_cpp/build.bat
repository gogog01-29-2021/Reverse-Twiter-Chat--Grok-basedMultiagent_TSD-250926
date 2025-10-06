@echo off
echo Building TSD C++ System...

if not exist build mkdir build
cd build

echo Configuring with CMake...
cmake .. -G "MinGW Makefiles"
if errorlevel 1 (
    echo CMake configuration failed. Trying with Visual Studio...
    cmake .. -G "Visual Studio 16 2019"
    if errorlevel 1 (
        echo CMake not found or configuration failed.
        echo Please install CMake or use alternative build method.
        pause
        exit /b 1
    )
)

echo Building...
cmake --build . --config Release
if errorlevel 1 (
    echo Build failed.
    pause
    exit /b 1
)

echo Build completed successfully!
echo.
echo Available executables:
echo - basic_example.exe
echo - trading_example.exe
echo - complete_trading_system.exe
echo - tsm_basic.exe
echo - tsm_trade_pnl.exe
echo.
pause