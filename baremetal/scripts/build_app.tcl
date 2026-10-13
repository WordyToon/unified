# ==============================================================================
# Script: baremetal/scripts/build_app.tcl
# Description: Automated XSCT build script for ZCU106 Baremetal C Application
# ==============================================================================

if {![info exists PRECISION]} { set PRECISION "8bit" }
if {[info exists argv] && [llength $argv] >= 1} { set PRECISION [lindex $argv 0] }

set script_dir [file normalize [file dirname [info script]]]
set base_dir   [file normalize [file join $script_dir ".." ".."]]
set export_dir [file join $base_dir "baremetal" "export" $PRECISION]
set xsa_file   [file join $export_dir "c_engine.xsa"]
set ws_dir     [file join $base_dir "baremetal" "vitis_ws"]
set src_dir    [file join $base_dir "baremetal" "src"]
set out_dir    [file join $base_dir "baremetal" "build"]

file mkdir $out_dir

puts "================================================================================"
puts "  BUILDING BAREMETAL APPLICATION VIA XSCT"
puts "  Precision: $PRECISION"
puts "  XSA File:  $xsa_file"
puts "  Workspace: $ws_dir"
puts "================================================================================"

setws $ws_dir

# Create platform from XSA if not existing
set pf_name "zcu106_c_engine_${PRECISION}"
if {[catch {platform active $pf_name}]} {
    puts "[*] Creating Vitis Platform from XSA: $xsa_file..."
    platform create -name $pf_name -hw $xsa_file -proc "psu_cortexa53_0" -os "standalone"
    platform write
    platform generate
}

if {$PRECISION == "16bit"} {
    set app_name "baremetal_c_engine_16bit"
    set prec_val 16
} else {
    set app_name "baremetal_c_engine"
    set prec_val 8
}

# Create Application if not existing
if {[catch {app list} res] || [lsearch -exact $res $app_name] == -1} {
    puts "\[*\] Creating Baremetal Application..."
    app create -name $app_name -platform $pf_name -domain "standalone_domain" -template "Empty Application(C)"
}

# Import sources
puts "\[*\] Importing source files from: $src_dir..."
importsources -name $app_name -path $src_dir -soft-link

# Configure compiler flags
app config -name $app_name -set compiler-misc "-O2 -DTARGET_PRECISION=$prec_val"
app config -name $app_name -set linker-misc "-lm"

# Build Application
puts "\[*\] Building Application..."
app build -name $app_name

# Copy ELF to build directory
set elf_src [file join $ws_dir $app_name "Debug" "${app_name}.elf"]
if {![file exists $elf_src]} {
    set elf_src [file join $ws_dir $app_name "Release" "${app_name}.elf"]
}

if {[file exists $elf_src]} {
    file copy -force $elf_src [file join $out_dir "${app_name}.elf"]
    file copy -force $elf_src [file join $out_dir "baremetal_c_engine_${PRECISION}.elf"]
    if {$PRECISION == "8bit"} {
        file copy -force $elf_src [file join $out_dir "baremetal_c_engine.elf"]
    }
    puts "\[+\] SUCCESS: ELF binary created at [file join $out_dir baremetal_c_engine_${PRECISION}.elf]"
} else {
    puts "\[-\] Error: Output ELF not found after build!"
}

