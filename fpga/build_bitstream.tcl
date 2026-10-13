# ==============================================================================
# Script: build_bitstream.tcl
# Description: Synthesizes, implements, and generates bitstream & XSA for AMD ZCU106
# Target: AMD Zynq UltraScale+ (xczu7ev-ffvc1156-2-e)
# ==============================================================================

set script_dir  [file normalize [file dirname [info script]]]
set fpga_dir    $script_dir
set unified_dir [file normalize [file join $fpga_dir ".."]]
set export_dir  [file join $fpga_dir "export"]
set baremetal_dir [file join $unified_dir "baremetal" "export"]

puts "================================================================================"
puts "  STARTING UNIFIED BITSTREAM GENERATION FLOW FOR AMD ZCU106"
puts "  Export Dir:    $export_dir"
puts "  Baremetal Dir: $baremetal_dir"
puts "================================================================================"

file mkdir $export_dir
file mkdir $baremetal_dir

# 1. Package IP with latest RTL
puts "Packaging updated RTL IP core..."
source [file join $script_dir "package_ip.tcl"]

# 2. Determine Target Frequency (default 75.0 MHz)
set target_freq_mhz 75.000000
if {[info exists env(TARGET_FREQ_MHZ)]} {
    set target_freq_mhz $env(TARGET_FREQ_MHZ)
}

# 3. Always create fresh Vivado Project & Block Design
# Clean project creation takes only ~15 seconds and guarantees:
# - No stale Out-of-Context (OOC) runs on hard silicon macros (PS8)
# - Global synthesis mode (synth_checkpoint_mode None)
# - Latest packaged IP from ip_repo with active precision configuration
# - Correct target clock frequency ($target_freq_mhz MHz)
puts "Creating fresh Vivado project and Block Design at $target_freq_mhz MHz..."
source [file join $script_dir "create_bd.tcl"]

set bd_file [get_files *${bd_name}.bd]

# 4. Launch Synthesis
puts "\n\[1/3\] Launching Synthesis (synth_1)..."
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1
set synth_status [get_property STATUS [get_runs synth_1]]
puts "Synthesis Status: $synth_status"
if {[string first "Complete" $synth_status] == -1} {
    error "[-] Synthesis failed! Check vivado_prj/${prj_name}.runs/synth_1/runme.log"
}
puts ">> Synthesis completed successfully!"

# 5. Launch Implementation & Generate Bitstream
puts "\n\[2/3\] Launching Implementation & Bitstream Generation (impl_1)..."
reset_run impl_1
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
set impl_status [get_property STATUS [get_runs impl_1]]
puts "Implementation Status: $impl_status"
if {[string first "Complete" $impl_status] == -1} {
    error "[-] Implementation failed! Check vivado_prj/${prj_name}.runs/impl_1/runme.log"
}
puts ">> Implementation and Bitstream completed successfully!"

# 6. Export Bitstream (.bit) and Hardware Platform (.xsa)
puts "\n\[3/3\] Exporting Hardware Artifacts (.bit, .xsa)..."
set impl_dir [get_property DIRECTORY [get_runs impl_1]]
set bit_files [glob -nocomplain [file join $impl_dir "*.bit"]]

set precision_name "8bit"
if {[info exists env(PRECISION)]} {
    set precision_name $env(PRECISION)
}
set prec_export_dir [file join $export_dir $precision_name]
file mkdir $prec_export_dir

if {[llength $bit_files] > 0} {
    set bit_src [lindex $bit_files 0]
    file copy -force $bit_src [file join $prec_export_dir "c_engine.bit"]
    file copy -force $bit_src [file join $export_dir "c_engine.bit"]
    puts "  \[OK\] Bitstream copied to:"
    puts "        $prec_export_dir/c_engine.bit"
    puts "        $export_dir/c_engine.bit"
} else {
    puts "  \[WARNING\] Bitstream file not found directly in $impl_dir"
}

# Export XSA for AMD ZCU106 Baremetal
set xsa_file [file join $prec_export_dir "c_engine.xsa"]
write_hw_platform -fixed -include_bit -force -file $xsa_file
file copy -force $xsa_file [file join $export_dir "c_engine.xsa"]
file mkdir [file join $baremetal_dir $precision_name]
file copy -force $xsa_file [file join $baremetal_dir $precision_name "c_engine.xsa"]
file copy -force $xsa_file [file join $baremetal_dir "c_engine.xsa"]
puts "  \[OK\] AMD ZCU106 Hardware Platform (.xsa) exported to:"
puts "        $xsa_file"
puts "        [file join $baremetal_dir $precision_name c_engine.xsa]"

# 7. Timing Closure and Utilization Check
catch {
    open_run impl_1
    set wns [get_property STATS.WNS [get_runs impl_1]]
    set tns [get_property STATS.TNS [get_runs impl_1]]
    puts "\n================================================================================"
    puts "TIMING CLOSURE SUMMARY:"
    puts "  Worst Negative Slack (WNS): $wns ns"
    puts "  Total Negative Slack (TNS): $tns ns"
    if {$wns >= 0} {
        puts "  Timing Status: MET (TIMING CLOSED SUCCESSFULLY!)"
    } else {
        puts "  Timing Status: VIOLATED (Negative slack detected)"
    }
    puts "================================================================================"

    report_utilization -file [file join $prec_export_dir "utilization.rpt"]
    report_timing_summary -file [file join $prec_export_dir "timing_summary.rpt"]
    file copy -force [file join $prec_export_dir "utilization.rpt"] [file join $export_dir "utilization.rpt"]
    file copy -force [file join $prec_export_dir "timing_summary.rpt"] [file join $export_dir "timing_summary.rpt"]
    close_design
}

close_project
puts "\n>> UNIFIED BITSTREAM & HARDWARE EXPORT COMPLETE!"
