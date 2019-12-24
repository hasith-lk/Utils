@echo off
REM This script should be placed on the desktop.
REM Use it by drag n' drop Oracle .trc files on it.
REM Oracle trace outout can be found using Select * from v$diag_info
REM It is dependant on certain paths and may need
REM to be modified by the user.
REM Leavs a Trace.txt file in the %TEMP% directory.
REM
REM Author: NINO 981217 from original idea from HAET.

echo Running tkprof for %1...
rem tkprof.exe %1 %TEMP%\Trace.txt explain=ifsapp/ifsapp@dep75 aggregate=yes sys=no  > NUL:
tkprof.exe %1 %TEMP%\Trace.txt aggregate=yes sys=no sort=exeela, fchela > NUL:
rem tkprof.exe %1 %TEMP%\Trace.txt print=20 aggregate=yes sys=no sort=exeela, fchela > NUL:
echo Starting wordpad...
start "Wordpad.exe" %TEMP%\Trace.txt
