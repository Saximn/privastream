@echo off
echo Unified Bounding Box Test for All Detection Models
echo ==================================================
echo.

echo Available options:
echo 1. Test with webcam (default camera, 3 FPS)
echo 2. Test with webcam (default camera, 2 FPS)
echo 3. Test with webcam (camera 1, 3 FPS)
echo 4. Test with image file
echo 5. Test with custom FPS
echo 6. Test with custom configuration
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo Starting webcam test with camera 0 at 3 FPS...
    python unified_bbox_test.py --mode webcam --camera 0 --fps 3.0
) else if "%choice%"=="2" (
    echo Starting webcam test with camera 0 at 2 FPS...
    python unified_bbox_test.py --mode webcam --camera 0 --fps 2.0
) else if "%choice%"=="3" (
    echo Starting webcam test with camera 1 at 3 FPS...
    python unified_bbox_test.py --mode webcam --camera 1 --fps 3.0
) else if "%choice%"=="4" (
    set /p imagepath="Enter image path: "
    echo Starting image test...
    python unified_bbox_test.py --mode image --image "%imagepath%"
) else if "%choice%"=="5" (
    set /p targetfps="Enter target FPS (0.5-10.0): "
    echo Starting webcam test with custom FPS...
    python unified_bbox_test.py --mode webcam --camera 0 --fps %targetfps%
) else if "%choice%"=="6" (
    echo Starting webcam test with custom configuration...
    python unified_bbox_test.py --mode webcam --camera 0 --fps 3.0 --config unified_test_config.json
) else (
    echo Invalid choice. Starting default webcam test at 3 FPS...
    python unified_bbox_test.py --mode webcam --camera 0 --fps 3.0
)

echo.
echo Test completed. Press any key to exit...
pause >nul
