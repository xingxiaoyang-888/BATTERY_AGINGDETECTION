@echo off
SET PATH=;D:/openmodelica/bin/;%PATH%;
SET ERRORLEVEL=
CALL "%CD%/BatteryCell.exe" %*
SET RESULT=%ERRORLEVEL%

EXIT /b %RESULT%
