# SPDX-License-Identifier: MPL-2.0
# 一键构建 + 冒烟测试（Windows pwsh）
# 用法： .\scripts\build-and-test.ps1 [-Vulkan]
# 前置： VS2019/2022 BuildTools、CMake（VS 自带即可）、Vulkan SDK（-Vulkan 时）
param(
    [switch]$Vulkan
)

$ErrorActionPreference = 'Stop'
$Root  = $PSScriptRoot | Split-Path
$Src   = if ($env:GGML_SRC) { $env:GGML_SRC } else { Join-Path $Root '..\ggml-src' | ForEach-Object { (Resolve-Path $_).Path } }
$Test  = Join-Path $Root 'tests\test_learned_ops.c'

# 定位 cmake / ninja / vcvars
$VSRoot = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools'
if (-not (Test-Path $VSRoot)) { $VSRoot = 'C:\Program Files\Microsoft Visual Studio\2022\BuildTools' }
$Vcvars = Join-Path $VSRoot 'VC\Auxiliary\Build\vcvars64.bat'
$CMake  = Join-Path $VSRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$Ninja  = Join-Path $VSRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
foreach ($t in @($Vcvars, $CMake)) { if (-not (Test-Path $t)) { throw "找不到工具：$t" } }

if (-not (Test-Path $Src)) {
    Write-Host "错误：找不到 ggml 源码目录 $Src"
    Write-Host "请先应用 patch 并用 `$env:GGML_SRC 指向源码目录。"
    exit 1
}

Write-Host '== 1/3 配置并编译 CPU 构建 =='
& $CMake -S $Src -B "$Src\build-cpu" -G 'Visual Studio 16 2019' -A x64 `
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=ON
& $CMake --build "$Src\build-cpu" --config Release --target ggml-base ggml-cpu -m

Write-Host '== 2/3 编译 CPU 冒烟测试 =='
$Bat = @"
call "$Vcvars" >nul
cl /nologo /O2 /W3 /I "$Src\include" /I "$Src\src" /Fo"$Src\test_learned_ops.obj" /Fe"$Src\test_learned_ops.exe" "$Test" /link /LIBPATH:"$Src\build-cpu\src\Release" ggml-cpu.lib ggml-base.lib
"@
$BatFile = Join-Path $env:TEMP 'learned-ops-build.bat'
[System.IO.File]::WriteAllText($BatFile, $Bat)
cmd /c $BatFile | Out-Null

Write-Host '== 3/3 运行 CPU 冒烟测试 =='
$env:PATH = "$Src\build-cpu\bin\Release;$env:PATH"
& "$Src\test_learned_ops.exe" cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU 测试未全部通过' }

if ($Vulkan) {
    Write-Host '== Vulkan 构建（Ninja 生成器，VS2019 不支持 DEPFILE）=='
    $Sdk = Get-ChildItem 'C:\VulkanSDK' -Directory | Sort-Object Name | Select-Object -Last 1
    $Bat = @"
call "$Vcvars" >nul
set "PATH=$($Sdk.FullName)\Bin;%PATH%"
"$CMake" -S "$Src" -B "$Src\build-vk" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON `
 -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=ON `
 -DCMAKE_MAKE_PROGRAM="$Ninja" -DCMAKE_PREFIX_PATH="$($Sdk.FullName)" -DMATH_LIBRARY=
"$Ninja" -C "$Src\build-vk" ggml-cpu ggml-vulkan
"@
    [System.IO.File]::WriteAllText($BatFile, $Bat)
    cmd /c $BatFile
    if ($LASTEXITCODE -ne 0) { throw 'Vulkan 构建失败' }

    $Bat = @"
call "$Vcvars" >nul
cl /nologo /O2 /W3 /DUSE_VULKAN /I "$Src\include" /I "$Src\src" /Fo"$Src\test_vk.obj" /Fe"$Src\test_learned_ops_vk.exe" "$Test" /link /LIBPATH:"$Src\build-vk\src\ggml-vulkan" /LIBPATH:"$Src\build-vk\src" ggml-vulkan.lib ggml-cpu.lib ggml-base.lib
"@
    [System.IO.File]::WriteAllText($BatFile, $Bat)
    cmd /c $BatFile | Out-Null

    Write-Host '== 运行 Vulkan 冒烟测试 =='
    $env:PATH = "$Src\build-vk\bin;$Src\build-vk\src\ggml-vulkan;$env:PATH"
    & "$Src\test_learned_ops_vk.exe" vk
    if ($LASTEXITCODE -ne 0) { throw 'Vulkan 测试未全部通过' }
}

Write-Host '全部完成。'
