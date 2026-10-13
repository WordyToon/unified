# ==============================================================================
# Script: baremetal/scripts/run_xsdb.tcl
# Description: Automated XSDB JTAG deployment script for AMD ZCU106 Baremetal App
# Feature: Option 1B - Downloads Weights, Microcode, and Test Images to DDR
# Usage:
#   xsdb run_xsdb.tcl [model] [precision] [image]
#   Examples:
#     xsdb run_xsdb.tcl mnist 8bit digit7
#     xsdb run_xsdb.tcl resnet50 8bit dog
# ==============================================================================

# Default configurations if not provided
if {![info exists MODEL]}     { set MODEL "mnist" }         ;# mnist, resnet50, mobilenetv2, efficientnetb0, all
if {![info exists PRECISION]} { set PRECISION "8bit" }      ;# 8bit, 16bit
if {![info exists IMAGE]}     { set IMAGE "digit7" }        ;# dog, car, cat, digit0..digit9

# Parse command-line arguments from argv if provided
if {[info exists argv] && [llength $argv] >= 1} { set MODEL     [lindex $argv 0] }
if {[info exists argv] && [llength $argv] >= 2} { set PRECISION [lindex $argv 1] }
if {[info exists argv] && [llength $argv] >= 3} { set IMAGE     [lindex $argv 2] }

set script_dir [file normalize [file dirname [info script]]]
set base_dir   [file normalize [file join $script_dir ".." ".."]]
set export_dir [file join $base_dir "fpga" "export" $PRECISION]
set bit_file   [file join $export_dir "c_engine.bit"]

# ELF: prefer precision-specific build
set prec_elf   [file join $script_dir ".." "build" "baremetal_c_engine_${PRECISION}.elf"]
if {[file exists $prec_elf]} {
    set elf_file $prec_elf
} else {
    set elf_file [file join $script_dir ".." "build" "baremetal_c_engine.elf"]
}

set data_dir [file join $script_dir ".." "data"]

# ==============================================================================
# FSBL & PSU INIT RESOLUTION
# Try in order:
#  1. FSBL ELF (official pre-bootloader: takes APU out of reset, initializes DDR4/clocks/caches)
#  2. BSP export psu_init
# ==============================================================================
set fsbl_elf ""
set fsbl_try_list [list \
    [file join $script_dir ".." "vitis_ws" "zcu106_c_engine_${PRECISION}" "zynqmp_fsbl" "fsbl_a53.elf"] \
    [file join $script_dir ".." "vitis_ws" "zcu106_c_engine_${PRECISION}" "export" "zcu106_c_engine_${PRECISION}" "sw" "zcu106_c_engine_${PRECISION}" "boot" "fsbl.elf"] \
    [file join $script_dir ".." "vitis_ws" "zcu106_c_engine_8bit" "zynqmp_fsbl" "fsbl_a53.elf"] \
    [file join $script_dir ".." "vitis_ws" "zcu106_c_engine_8bit" "export" "zcu106_c_engine_8bit" "sw" "zcu106_c_engine_8bit" "boot" "fsbl.elf"] \
]
foreach candidate $fsbl_try_list {
    if {[file exists $candidate]} {
        set fsbl_elf $candidate
        break
    }
}

set ps_init ""
set ps_try_list [list \
    [file join $script_dir ".." "vitis_ws" "zcu106_c_engine_${PRECISION}" "export" "zcu106_c_engine_${PRECISION}" "hw" "psu_init.tcl"] \
    [file join $script_dir ".." "vitis_ws" "zcu106_c_engine_${PRECISION}" "hw" "psu_init.tcl"] \
    [file join $base_dir "fpga" "vivado_prj" "zcu106_unified_c_engine.gen" "sources_1" "bd" "zcu106_unified_bd" "ip" "zcu106_unified_bd_zynq_ultra_ps_e_0_0" "psu_init.tcl"] \
    [file join $script_dir ".." "vitis_ws" "baremetal_c_engine" "_ide" "psinit" "psu_init.tcl"] \
]
foreach candidate $ps_try_list {
    if {[file exists $candidate]} {
        set ps_init $candidate
        break
    }
}

# ==============================================================================
# Model Asset Resolution (Non-overlapping DDR Addresses)
# ==============================================================================
set ADDR_UCODE   0x21000000
set ADDR_MAILBOX 0x21FF0000

if {$MODEL eq "mnist"} {
    set weights_file "mnist_weights.bin"
    if {$PRECISION eq "16bit"} { set weights_file "mnist_16bit_weights.bin" }
    set weights_bin  [file join $data_dir $weights_file]
    set ucode_bin    [file join $data_dir "mnist_microcode.bin"]
    set img_name     $IMAGE
    if {[string match "digit*" $img_name] == 0} { set img_name "digit7" }
    set image_bin    [file join $data_dir "${img_name}_${PRECISION}.bin"]
    set ADDR_IMAGE   0x20000000
    set ADDR_WEIGHTS 0x30000000
    set MODEL_CODE   1
} elseif {$MODEL eq "resnet50"} {
    set weights_bin  [file join $data_dir "resnet50_${PRECISION}_weights.bin"]
    set ucode_bin    [file join $data_dir "resnet50_${PRECISION}_microcode.bin"]
    set image_bin    [file join $data_dir "${IMAGE}_${PRECISION}.bin"]
    set ADDR_IMAGE   0x20100000
    set ADDR_WEIGHTS 0x31000000
    set MODEL_CODE   2
} elseif {$MODEL eq "mobilenetv2"} {
    set weights_bin  [file join $data_dir "mobilenetv2_${PRECISION}_weights.bin"]
    set ucode_bin    [file join $data_dir "mobilenetv2_${PRECISION}_microcode.bin"]
    set image_bin    [file join $data_dir "${IMAGE}_${PRECISION}.bin"]
    set ADDR_IMAGE   0x20100000
    set ADDR_WEIGHTS 0x35000000
    set MODEL_CODE   3
} elseif {$MODEL eq "efficientnetb0"} {
    set weights_bin  [file join $data_dir "efficientnetb0_${PRECISION}_weights.bin"]
    set ucode_bin    [file join $data_dir "efficientnetb0_${PRECISION}_microcode.bin"]
    set image_bin    [file join $data_dir "${IMAGE}_${PRECISION}.bin"]
    set ADDR_IMAGE   0x20100000
    set ADDR_WEIGHTS 0x36000000
    set MODEL_CODE   4
} else {
    # All models: prepare all model weights and images
    set mnist_wt_file "mnist_weights.bin"
    if {$PRECISION eq "16bit"} { set mnist_wt_file "mnist_16bit_weights.bin" }
    set mnist_wt_bin  [file join $data_dir $mnist_wt_file]
    set mnist_img_bin [file join $data_dir "digit7_${PRECISION}.bin"]

    set r50_wt_bin    [file join $data_dir "resnet50_${PRECISION}_weights.bin"]
    set mbv2_wt_bin   [file join $data_dir "mobilenetv2_${PRECISION}_weights.bin"]
    set eff_wt_bin    [file join $data_dir "efficientnetb0_${PRECISION}_weights.bin"]
    set imagenet_bin  [file join $data_dir "${IMAGE}_${PRECISION}.bin"]

    set weights_bin   $r50_wt_bin
    set ucode_bin     [file join $data_dir "resnet50_${PRECISION}_microcode.bin"]
    set image_bin     $imagenet_bin
    set ADDR_IMAGE    0x20100000
    set ADDR_WEIGHTS  0x31000000
    set MODEL_CODE    0
}

set bit_exists 0
if {[file exists $bit_file]} { set bit_exists 1 }
set ps_exists 0
if {$ps_init ne "" && [file exists $ps_init]} { set ps_exists 1 }
set elf_exists 0
if {[file exists $elf_file]} { set elf_exists 1 }
set wt_exists 0
if {[file exists $weights_bin]} { set wt_exists 1 }
set uc_exists 0
if {[file exists $ucode_bin]} { set uc_exists 1 }
set img_exists 0
if {[file exists $image_bin]} { set img_exists 1 }

puts "================================================================================"
puts "  XSDB JTAG RUNNER FOR AMD ZCU106 BAREMETAL ACCELERATOR (OPTION 1B)"
puts "  Model:       $MODEL ($PRECISION)"
puts "  Test Image:  $IMAGE"
puts "  Bitstream:   $bit_file  (exists: $bit_exists)"
puts "  PSU Init:    $ps_init   (exists: $ps_exists)"
puts "  ELF File:    $elf_file  (exists: $elf_exists)"
if {$MODEL eq "all"} {
    puts "  Preloading ALL 4 vision models concurrently to non-overlapping DDR regions:"
    puts "    MNIST:          weights -> 0x30000000, image -> 0x20000000"
    puts "    ResNet-50:      weights -> 0x31000000, image -> 0x20100000"
    puts "    MobileNetV2:    weights -> 0x35000000, image -> 0x20100000"
    puts "    EfficientNet-B0:weights -> 0x36000000, image -> 0x20100000"
} else {
    puts "  Weights:     $weights_bin -> [format 0x%08X $ADDR_WEIGHTS]  (exists: $wt_exists)"
    puts "  Microcode:   $ucode_bin  -> [format 0x%08X $ADDR_UCODE]     (exists: $uc_exists)"
    puts "  Image:       $image_bin  -> [format 0x%08X $ADDR_IMAGE]     (exists: $img_exists)"
}
puts "================================================================================"

# ==============================================================================
# STEP 1: Connect to hw_server (must already be running: hw_server &)
# ==============================================================================
if {[catch {connect -host 127.0.0.1 -port 3121} err]} {
    puts "(*) Direct connect failed ($err), auto-launching hw_server via connect..."
    connect
}

# ==============================================================================
# STEP 2: Target PSU and perform system reset
# ==============================================================================
puts "(*) Targeting PSU..."
targets -set -nocase -filter {name =~ "*PSU*"}
puts "(*) Performing system reset..."
rst -system
after 2000

# ==============================================================================
# STEP 3: Program Bitstream (if available)
# ==============================================================================
if {[file exists $bit_file]} {
    puts "(*) Programming bitstream: $bit_file"
    fpga -file $bit_file
    after 1000
    puts "(+) Bitstream programmed."
} else {
    puts "(-) Bitstream NOT found at: $bit_file"
    puts "    Continuing without bitstream (accelerator may not work, but UART should still appear)."
}

# ==============================================================================
# STEP 4: Target Cortex-A53 #0 and run FSBL pre-bootloader (or fallback to psu_init)
# ==============================================================================
puts "(*) Selecting Cortex-A53 #0..."
targets -set -nocase -filter {name =~ "*A53*#0*"}
puts "(*) Resetting processor core into AArch64 (A64) mode..."
rst -processor -clear-registers -isa A64
after 500

if {$fsbl_elf ne "" && [file exists $fsbl_elf]} {
    puts "(*) Downloading FSBL to OCM for full PS/DDR/APU initialization: $fsbl_elf"
    dow $fsbl_elf
    after 500
    puts "(*) Running FSBL on Cortex-A53 #0..."
    con
    puts "(*) Waiting 3s for FSBL board bringup (clocks, DDR4, MIO, caches)..."
    after 3000
    puts "(*) Stopping Cortex-A53 #0 after FSBL bringup..."
    stop
    after 500
    puts "(+) FSBL bringup complete. Core state: [state]"
} elseif {$ps_init ne "" && [file exists $ps_init]} {
    puts "(*) Sourcing psu_init from: $ps_init"
    source $ps_init
    puts "(*) Running psu_init..."
    psu_init
    after 1000
    puts "(*) Running psu_post_config..."
    psu_post_config
    after 500
    catch { psu_ps_pl_reset_config }
    catch { psu_ps_pl_isolation_removal }
    puts "(+) PSU initialization complete."
} else {
    puts "(!!) WARNING: Neither FSBL nor psu_init found - aborting."
    exit 1
}

# ==============================================================================
# STEP 6: JTAG Download data to DDR (Option 1B)
# ==============================================================================
if {$MODEL eq "all"} {
    puts "(*) Preloading ALL 4 model weights and test images into DDR (Option 1B)..."
    if {[file exists $mnist_img_bin]} {
        puts "    [1/6] MNIST test image -> DDR 0x20000000: $mnist_img_bin"
        dow -data $mnist_img_bin 0x20000000
    }
    if {[file exists $mnist_wt_bin]} {
        puts "    [2/6] MNIST weights -> DDR 0x30000000: $mnist_wt_bin"
        dow -data $mnist_wt_bin 0x30000000
    }
    if {[file exists $imagenet_bin]} {
        puts "    [3/6] ImageNet test image -> DDR 0x20100000: $imagenet_bin"
        dow -data $imagenet_bin 0x20100000
    }
    if {[file exists $r50_wt_bin]} {
        puts "    [4/6] ResNet-50 weights -> DDR 0x31000000: $r50_wt_bin"
        dow -data $r50_wt_bin 0x31000000
    }
    if {[file exists $mbv2_wt_bin]} {
        puts "    [5/6] MobileNetV2 weights -> DDR 0x35000000: $mbv2_wt_bin"
        dow -data $mbv2_wt_bin 0x35000000
    }
    if {[file exists $eff_wt_bin]} {
        puts "    [6/6] EfficientNet-B0 weights -> DDR 0x36000000: $eff_wt_bin"
        dow -data $eff_wt_bin 0x36000000
    }
    puts "(+) All weights and test images successfully preloaded into DDR!"
} else {
    if {[file exists $image_bin]} {
        puts "(*) Downloading test image -> DDR [format 0x%08X $ADDR_IMAGE]: $image_bin"
        dow -data $image_bin $ADDR_IMAGE
    } else {
        puts "(!!) Test image not found: $image_bin  (model will use whatever is in DDR)"
    }

    if {[file exists $ucode_bin]} {
        puts "(*) Downloading microcode -> DDR [format 0x%08X $ADDR_UCODE]: $ucode_bin"
        dow -data $ucode_bin $ADDR_UCODE
    } else {
        puts "(!!) Microcode not found: $ucode_bin  (ELF compiled table will be used)"
    }

    if {[file exists $weights_bin]} {
        puts "(*) Downloading weights -> DDR [format 0x%08X $ADDR_WEIGHTS]: $weights_bin"
        dow -data $weights_bin $ADDR_WEIGHTS
    } else {
        puts "(!!) Weights not found: $weights_bin  (model inference will use garbage weights)"
    }
}

# ==============================================================================
# STEP 7: Write model selection mailbox (tells ELF which model to run)
# ==============================================================================
puts "(*) Writing Model Selection Mailbox: code $MODEL_CODE -> [format 0x%08X $ADDR_MAILBOX]"
mwr -force $ADDR_MAILBOX $MODEL_CODE
after 100

# ==============================================================================
# STEP 8: Download ELF and execute
# ==============================================================================
if {[file exists $elf_file]} {
    puts "(*) Downloading ELF: $elf_file"
    dow $elf_file
    after 500
    puts "(*) Starting Cortex-A53 #0 (con)..."
    con
    after 1000
    puts "(+) Baremetal ELF is RUNNING."
    puts "    --> Output is streaming on COM7 (115200 baud 8-N-1)."
    puts "    --> You should see: \[BOOT\] ZCU106 Baremetal ELF started on Cortex-A53 #0!"
    after 3000
    puts "(*) Processor status: [state]"
} else {
    puts "(-) ELF not found at: $elf_file"
    puts "    Build first: run .\\build_elf.ps1 from the unified\\ directory."
    exit 1
}

puts "(+) XSDB deployment finished successfully."
exit
