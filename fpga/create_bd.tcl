# ==============================================================================
# Script: create_bd.tcl
# Description: Generates the Vivado Block Design for AMD Zynq UltraScale+ ZCU104
# Target: xczu7ev-ffvc1156-2-e (Board: xilinx.com:zcu104:part0:1.1)
# Clock: pl_clk0 configured for 125.0 MHz
# Interconnect:
#   - PS M_AXI_HPM0_FPD -> AXI Interconnect -> c_engine_dma_top s_axi_lite (0xA000_0000)
#   - c_engine_dma_top m0_axi (RD DMA 128-bit) -> SmartConnect -> PS S_AXI_HP0_FPD
#   - c_engine_dma_top m1_axi (WR DMA 128-bit) -> SmartConnect -> PS S_AXI_HP1_FPD
#   - c_engine_dma_top interrupt -> PS pl_ps_irq0[0]
# ==============================================================================

set script_dir  [file normalize [file dirname [info script]]]
set fpga_dir    $script_dir
set prj_dir     [file join $fpga_dir "vivado_prj"]
set ip_repo     [file join $fpga_dir "ip_repo"]

# Check target board (default zcu106, check env)
set board_target "zcu106"
if {[info exists env(BOARD)]} {
    set board_target $env(BOARD)
}
if {$board_target eq "zcu104"} {
    set board_part "xilinx.com:zcu104:part0:1.1"
    set prj_name   "zcu104_unified_c_engine"
    set bd_name    "zcu104_unified_bd"
} else {
    set board_part "xilinx.com:zcu106:part0:2.6"
    set prj_name   "zcu106_unified_c_engine"
    set bd_name    "zcu106_unified_bd"
}

# Check target frequency argument (default 75 MHz, check env)
if {![info exists target_freq_mhz]} {
    if {[info exists env(TARGET_FREQ_MHZ)]} {
        set target_freq_mhz $env(TARGET_FREQ_MHZ)
    } else {
        set target_freq_mhz 75.000000
    }
}

puts "================================================================================"
puts "  CREATING UNIFIED VIVADO PROJECT & BLOCK DESIGN FOR [string toupper $board_target]"
puts "  Target PL Frequency: $target_freq_mhz MHz"
puts "  Board Part:          $board_part"
puts "  Project Directory:   $prj_dir"
puts "  IP Repository:       $ip_repo"
puts "================================================================================"

# Create or Recreate Project
file delete -force $prj_dir
create_project -force $prj_name $prj_dir -part xczu7ev-ffvc1156-2-e
set_property board_part $board_part [current_project]

# Set IP Repository Paths
set_property ip_repo_paths [list $ip_repo] [current_project]
update_ip_catalog -rebuild

# Create Block Design
create_bd_design $bd_name
current_bd_design $bd_name

# 1. Instantiate Zynq UltraScale+ MPSoC Processing System
set ps_cell [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps_cell

# Enable required PL interfaces on PS with target clock frequency
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {128} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__USE__S_AXI_GP3 {1} \
    CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
    CONFIG.PSU__USE__IRQ0 {1} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $target_freq_mhz \
] $ps_cell

# 2. Instantiate unified c_engine DMA Top Accelerator Core
set acc_cell [create_bd_cell -type ip -vlnv xilinx.user:c_engine:c_engine_dma_top:1.0 c_engine_dma_top_0]

# 3. Instantiate Processor System Reset
set rst_cell [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps8_0]

# 4. Instantiate SmartConnect for AXI-Lite Configuration (128-bit PS Master -> 32-bit Slave)
set smc_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_ctrl]
set_property CONFIG.NUM_SI {1} $smc_ctrl
set_property CONFIG.NUM_MI {1} $smc_ctrl

# 5. Instantiate SmartConnect 0 for Read DMA (m0_axi -> S_AXI_HP0_FPD)
set smc_rd [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_rd]
set_property CONFIG.NUM_SI {1} $smc_rd
set_property CONFIG.NUM_MI {1} $smc_rd

# 6. Instantiate SmartConnect 1 for Write DMA (m1_axi -> S_AXI_HP1_FPD)
set smc_wr [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_wr]
set_property CONFIG.NUM_SI {1} $smc_wr
set_property CONFIG.NUM_MI {1} $smc_wr

# -----------------------------------------------------------------------------
# Connect Interfaces
# -----------------------------------------------------------------------------
# PS Master AXI-Lite Control Bus to Accelerator Slave
connect_bd_intf_net [get_bd_intf_pins $ps_cell/M_AXI_HPM0_FPD] [get_bd_intf_pins $smc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $smc_ctrl/M00_AXI] [get_bd_intf_pins $acc_cell/s_axi_lite]

# Read DMA 128-bit Master to PS HP0
connect_bd_intf_net [get_bd_intf_pins $acc_cell/m0_axi] [get_bd_intf_pins $smc_rd/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $smc_rd/M00_AXI] [get_bd_intf_pins $ps_cell/S_AXI_HP0_FPD]

# Write DMA 128-bit Master to PS HP1
connect_bd_intf_net [get_bd_intf_pins $acc_cell/m1_axi] [get_bd_intf_pins $smc_wr/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $smc_wr/M00_AXI] [get_bd_intf_pins $ps_cell/S_AXI_HP1_FPD]

# -----------------------------------------------------------------------------
# Connect Clocks and Resets
# -----------------------------------------------------------------------------
connect_bd_net [get_bd_pins $ps_cell/pl_clk0] \
               [get_bd_pins $ps_cell/maxihpm0_fpd_aclk] \
               [get_bd_pins $ps_cell/saxihp0_fpd_aclk] \
               [get_bd_pins $ps_cell/saxihp1_fpd_aclk] \
               [get_bd_pins $rst_cell/slowest_sync_clk] \
               [get_bd_pins $smc_ctrl/aclk] \
               [get_bd_pins $smc_rd/aclk] \
               [get_bd_pins $smc_wr/aclk] \
               [get_bd_pins $acc_cell/aclk]

connect_bd_net [get_bd_pins $ps_cell/pl_resetn0] [get_bd_pins $rst_cell/ext_reset_in]

connect_bd_net [get_bd_pins $rst_cell/peripheral_aresetn] \
               [get_bd_pins $smc_ctrl/aresetn] \
               [get_bd_pins $smc_rd/aresetn] \
               [get_bd_pins $smc_wr/aresetn] \
               [get_bd_pins $acc_cell/aresetn]

# -----------------------------------------------------------------------------
# Connect Interrupts
# -----------------------------------------------------------------------------
connect_bd_net [get_bd_pins $acc_cell/interrupt] [get_bd_pins $ps_cell/pl_ps_irq0]

# -----------------------------------------------------------------------------
# Memory Address Assignments
# -----------------------------------------------------------------------------
# Accelerator CSRs: 0xA000_0000 - 0xA000_FFFF (64KB aperture)
assign_bd_address -offset 0xA0000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces $ps_cell/Data] [get_bd_addr_segs $acc_cell/s_axi_lite/reg0] -force

# Read DMA to PS DDR4: 0x0000_0000 - 0x7FFF_FFFF (2GB DDR Low)
assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces $acc_cell/m0_axi] [get_bd_addr_segs $ps_cell/SAXIGP2/HP0_DDR_LOW] -force

# Write DMA to PS DDR4: 0x0000_0000 - 0x7FFF_FFFF (2GB DDR Low)
assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces $acc_cell/m1_axi] [get_bd_addr_segs $ps_cell/SAXIGP3/HP1_DDR_LOW] -force

# Validate and Save Block Design
validate_bd_design
save_bd_design

# Configure Block Design for Global Synthesis (disables problematic OOC runs on hard PS macros)
set bd_file [get_files *${bd_name}.bd]
set_property synth_checkpoint_mode None $bd_file

# Generate Top-Level HDL Wrapper
set wrapper_file [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper_file
set_property top ${bd_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

# Generate Targets in Global Mode
puts "Generating Block Design targets in Global Synthesis mode..."
generate_target all $bd_file
export_ip_user_files -of_objects $bd_file -no_script -sync -force -quiet

puts ">> Unified Block Design created and validated successfully!"

