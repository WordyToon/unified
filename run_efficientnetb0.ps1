param (
    [ValidateSet("8bit", "16bit")]
    [string]$Precision = "8bit",

    [ValidateSet("dog", "car", "cat", "aeroplane")]
    [string]$Image = "dog"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$ScriptDir\baremetal\run_efficientnetb0.ps1" -Precision $Precision -Image $Image

