@echo off
setlocal

echo Compilazione avviata...
set VERSION=1,0,0,0

for /f "tokens=1-4 delims=.:/ " %%a in ("%date% %time%") do (
    set CURRENTDATE=%%c-%%a-%%b
    set CURRENTTIME=%%d:%%e:%%f
)
set BUILDTIME=%CURRENTDATE% %CURRENTTIME%

for /f "delims=" %%i in ('git rev-parse --short HEAD') do set GITCOMMIT=%%i

if not "%~1"=="" set VERSION=%~1

set VERSION_COMMA=%VERSION:.=,%

(
echo #include ^<windows.h^>
echo(
echo VS_VERSION_INFO VERSIONINFO
echo FILEVERSION %VERSION_COMMA%
echo PRODUCTVERSION %VERSION_COMMA%
echo FILEFLAGSMASK VS_FFI_FILEFLAGSMASK
echo FILEFLAGS 0x0L & rem build regolare di release, senza debug nè flag speciali
echo FILEOS VOS__WINDOWS32
echo FILETYPE VFT_DLL & rem libreria dinamica (*.dll)
echo FILESUBTYPE 0x0L & rem nessun sottotipo di file specifico
echo BEGIN
echo(    BLOCK "StringFileInfo"
echo(    BEGIN
echo(        BLOCK "040904b0"
echo(        BEGIN
echo(            VALUE "LegalCopyright", "David Buyer\0"
echo(            VALUE "Author", "David Buyer\0"
echo(            VALUE "CompanyName", "Microservice\0"
echo(            VALUE "FileDescription", "High performance utility library for general purposes\0"
echo(            VALUE "FileVersion", "%VERSION%\0"
echo(            VALUE "InternalName", "HPUtils\0"
echo(            VALUE "OriginalFilename", "HPUtils.dll\0"
echo(            VALUE "ProductName", "HPU\0"
echo(            VALUE "ProductVersion", "%VERSION%+%GITCOMMIT%\0"
echo(        END
echo(    END
echo(    BLOCK "VarFileInfo"
echo(    BEGIN
echo(        VALUE "Translation", 0x0410, 1252
echo(    END
echo END
) > version.rc

windres -o assembly-info.syso version.rc

go build -buildmode=c-shared -o HPUtils.dll -ldflags="-X 'main.Version=%VERSION%' -X 'main.GitCommit=%GITCOMMIT%' -X 'main.BuildTime=%BUILDTIME%'"

echo Compilazione completata.

endlocal
