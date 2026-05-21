@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "INCLUDE=%USERPROFILE%\miniconda3\envs\mfa\Library\include;%INCLUDE%"
set "LIB=%USERPROFILE%\miniconda3\envs\mfa\Library\bin;%USERPROFILE%\miniconda3\envs\mfa\Library\lib;%LIB%"
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
"%USERPROFILE%\miniconda3\envs\mfa\python.exe" -m pip install --no-build-isolation setuptools wheel pybind11
"%USERPROFILE%\miniconda3\envs\mfa\python.exe" -m pip install --no-build-isolation python-mecab-ko
