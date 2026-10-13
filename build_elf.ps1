param(
    [ValidateSet("all", "8bit", "16bit")]
    [string]$Target = "all"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$ScriptDir\baremetal\build_elf.ps1" -Target $Target

