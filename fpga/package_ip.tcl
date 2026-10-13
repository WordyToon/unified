# ==============================================================================
# Script: package_ip.tcl
# Description: Packages unified c_engine_dma_top (256 PEs @ 125 MHz) as Vivado IP
# Target: AMD Zynq UltraScale+ ZCU104 (xczu7ev-ffvc1156-2-e)
# ==============================================================================

set script_dir [file normalize [file dirname [info script]]]
set fpga_dir   $script_dir
set unified_dir [file normalize [file join $fpga_dir ".."]]
set rtl_dir    [file join $unified_dir "rtl"]
set ip_dir     [file join $fpga_dir "ip_repo" "c_engine_dma_top"]

puts "================================================================================"
puts "  PACKAGING UNIFIED C_ENGINE DMA TOP IP FOR ZCU104 (125 MHz, 256 PEs)"
puts "  RTL Source Directory: $rtl_dir"
puts "  Target IP Directory:  $ip_dir"
puts "================================================================================"

file delete -force $ip_dir
file mkdir $ip_dir
file mkdir [file join $ip_dir "src"]

# Copy RTL files
set rtl_files [glob -nocomplain [file join $rtl_dir "*.v"]]
set vh_files  [glob -nocomplain [file join $rtl_dir "*.vh"]]

foreach f $rtl_files {
    file copy -force $f [file join $ip_dir "src" [file tail $f]]
}
foreach f $vh_files {
    file copy -force $f [file join $ip_dir "src" [file tail $f]]
}

# Temporary packaging project
set prj_dir [file join $fpga_dir "ip_repo" "_pkg_prj"]
file delete -force $prj_dir
create_project -force c_engine_ip_pkg $prj_dir -part xczu7ev-ffvc1156-2-e
set_property board_part xilinx.com:zcu106:part0:2.6 [current_project]

# Add source files
add_files -norecurse [glob [file join $ip_dir "src" "*.v"]]
add_files -norecurse [glob [file join $ip_dir "src" "*.vh"]]
set_property file_type {Verilog Header} [get_files [glob [file join $ip_dir "src" "*.vh"]]]
set_property top c_engine_dma_top [current_fileset]
update_compile_order -fileset sources_1

# Package IP
ipx::package_project -root_dir $ip_dir -vendor xilinx.user -library c_engine -taxonomy /UserIP -import_files -force

set core [ipx::current_core]
set_property name c_engine_dma_top $core
set_property version 1.0 $core
set_property display_name {Unified c_engine DMA Top (R=8, C=32, 100 MHz)} $core
set_property description {Unified Systolic CNN Accelerator with Dual 128-bit AXI4 Masters, 256 PEs, 100 MHz for ZCU104} $core

# Associate clock with bus interfaces and reset
set aclk_intf [ipx::get_bus_interfaces aclk -of_objects $core]
if {$aclk_intf ne ""} {
    set param [ipx::get_bus_parameters ASSOCIATED_BUSIF -of_objects $aclk_intf]
    if {$param eq ""} {
        set param [ipx::add_bus_parameter ASSOCIATED_BUSIF $aclk_intf]
    }
    set_property value {m0_axi:m1_axi:s_axi_lite} $param

    set param_rst [ipx::get_bus_parameters ASSOCIATED_RESET -of_objects $aclk_intf]
    if {$param_rst eq ""} {
        set param_rst [ipx::add_bus_parameter ASSOCIATED_RESET $aclk_intf]
    }
    set_property value {aresetn} $param_rst
}

# Infer interrupt interface
set irq_intf [ipx::get_bus_interfaces interrupt -of_objects $core]
if {$irq_intf eq ""} {
    set irq_intf [ipx::add_bus_interface interrupt $core]
    set_property abstraction_type_vlnv xilinx.com:signal:interrupt_rtl:1.0 $irq_intf
    set_property bus_type_vlnv xilinx.com:signal:interrupt:1.0 $irq_intf
    set_property interface_mode master $irq_intf
    set int_port [ipx::add_port_map INTERRUPT $irq_intf]
    set_property physical_name interrupt $int_port
    set param [ipx::add_bus_parameter SENSITIVITY $irq_intf]
    set_property value LEVEL_HIGH $param
}

ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core

close_project
file delete -force $prj_dir
puts ">> Unified IP Core successfully packaged at: $ip_dir"

