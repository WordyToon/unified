param (
    [ValidateSet("8bit", "16bit")]
    [string]$Precision = "8bit",

    [ValidateRange(0, 9)]
    [int]$Digit = 7
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$ScriptDir\baremetal\run_mnist.ps1" -Precision $Precision -Digit $Digit

