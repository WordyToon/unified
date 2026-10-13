# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CFG_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CFG_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CFG_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CLIP_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CLIP_MIN" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C_M_AXI_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C_M_AXI_DATA_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MAX_WEIGHTS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MEM_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "NUM_BANKS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "NUM_CHUNKS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "R" -parent ${Page_0}
  ipgui::add_param $IPINST -name "WORD_SIZE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "b_DATA_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "w_DATA_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "x_DATA_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "y_DATA_WIDTH" -parent ${Page_0}


}

proc update_PARAM_VALUE.ADDR_WIDTH { PARAM_VALUE.ADDR_WIDTH } {
	# Procedure called to update ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ADDR_WIDTH { PARAM_VALUE.ADDR_WIDTH } {
	# Procedure called to validate ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.C { PARAM_VALUE.C } {
	# Procedure called to update C when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C { PARAM_VALUE.C } {
	# Procedure called to validate C
	return true
}

proc update_PARAM_VALUE.CFG_ADDR_WIDTH { PARAM_VALUE.CFG_ADDR_WIDTH } {
	# Procedure called to update CFG_ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CFG_ADDR_WIDTH { PARAM_VALUE.CFG_ADDR_WIDTH } {
	# Procedure called to validate CFG_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.CFG_DEPTH { PARAM_VALUE.CFG_DEPTH } {
	# Procedure called to update CFG_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CFG_DEPTH { PARAM_VALUE.CFG_DEPTH } {
	# Procedure called to validate CFG_DEPTH
	return true
}

proc update_PARAM_VALUE.CFG_WIDTH { PARAM_VALUE.CFG_WIDTH } {
	# Procedure called to update CFG_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CFG_WIDTH { PARAM_VALUE.CFG_WIDTH } {
	# Procedure called to validate CFG_WIDTH
	return true
}

proc update_PARAM_VALUE.CLIP_MAX { PARAM_VALUE.CLIP_MAX } {
	# Procedure called to update CLIP_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CLIP_MAX { PARAM_VALUE.CLIP_MAX } {
	# Procedure called to validate CLIP_MAX
	return true
}

proc update_PARAM_VALUE.CLIP_MIN { PARAM_VALUE.CLIP_MIN } {
	# Procedure called to update CLIP_MIN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CLIP_MIN { PARAM_VALUE.CLIP_MIN } {
	# Procedure called to validate CLIP_MIN
	return true
}

proc update_PARAM_VALUE.C_M_AXI_ADDR_WIDTH { PARAM_VALUE.C_M_AXI_ADDR_WIDTH } {
	# Procedure called to update C_M_AXI_ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_M_AXI_ADDR_WIDTH { PARAM_VALUE.C_M_AXI_ADDR_WIDTH } {
	# Procedure called to validate C_M_AXI_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.C_M_AXI_DATA_WIDTH { PARAM_VALUE.C_M_AXI_DATA_WIDTH } {
	# Procedure called to update C_M_AXI_DATA_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_M_AXI_DATA_WIDTH { PARAM_VALUE.C_M_AXI_DATA_WIDTH } {
	# Procedure called to validate C_M_AXI_DATA_WIDTH
	return true
}

proc update_PARAM_VALUE.MAX_WEIGHTS { PARAM_VALUE.MAX_WEIGHTS } {
	# Procedure called to update MAX_WEIGHTS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MAX_WEIGHTS { PARAM_VALUE.MAX_WEIGHTS } {
	# Procedure called to validate MAX_WEIGHTS
	return true
}

proc update_PARAM_VALUE.MEM_DEPTH { PARAM_VALUE.MEM_DEPTH } {
	# Procedure called to update MEM_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MEM_DEPTH { PARAM_VALUE.MEM_DEPTH } {
	# Procedure called to validate MEM_DEPTH
	return true
}

proc update_PARAM_VALUE.NUM_BANKS { PARAM_VALUE.NUM_BANKS } {
	# Procedure called to update NUM_BANKS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.NUM_BANKS { PARAM_VALUE.NUM_BANKS } {
	# Procedure called to validate NUM_BANKS
	return true
}

proc update_PARAM_VALUE.NUM_CHUNKS { PARAM_VALUE.NUM_CHUNKS } {
	# Procedure called to update NUM_CHUNKS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.NUM_CHUNKS { PARAM_VALUE.NUM_CHUNKS } {
	# Procedure called to validate NUM_CHUNKS
	return true
}

proc update_PARAM_VALUE.R { PARAM_VALUE.R } {
	# Procedure called to update R when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.R { PARAM_VALUE.R } {
	# Procedure called to validate R
	return true
}

proc update_PARAM_VALUE.WORD_SIZE { PARAM_VALUE.WORD_SIZE } {
	# Procedure called to update WORD_SIZE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.WORD_SIZE { PARAM_VALUE.WORD_SIZE } {
	# Procedure called to validate WORD_SIZE
	return true
}

proc update_PARAM_VALUE.b_DATA_WIDTH { PARAM_VALUE.b_DATA_WIDTH } {
	# Procedure called to update b_DATA_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.b_DATA_WIDTH { PARAM_VALUE.b_DATA_WIDTH } {
	# Procedure called to validate b_DATA_WIDTH
	return true
}

proc update_PARAM_VALUE.w_DATA_WIDTH { PARAM_VALUE.w_DATA_WIDTH } {
	# Procedure called to update w_DATA_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.w_DATA_WIDTH { PARAM_VALUE.w_DATA_WIDTH } {
	# Procedure called to validate w_DATA_WIDTH
	return true
}

proc update_PARAM_VALUE.x_DATA_WIDTH { PARAM_VALUE.x_DATA_WIDTH } {
	# Procedure called to update x_DATA_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.x_DATA_WIDTH { PARAM_VALUE.x_DATA_WIDTH } {
	# Procedure called to validate x_DATA_WIDTH
	return true
}

proc update_PARAM_VALUE.y_DATA_WIDTH { PARAM_VALUE.y_DATA_WIDTH } {
	# Procedure called to update y_DATA_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.y_DATA_WIDTH { PARAM_VALUE.y_DATA_WIDTH } {
	# Procedure called to validate y_DATA_WIDTH
	return true
}


proc update_MODELPARAM_VALUE.R { MODELPARAM_VALUE.R PARAM_VALUE.R } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.R}] ${MODELPARAM_VALUE.R}
}

proc update_MODELPARAM_VALUE.C { MODELPARAM_VALUE.C PARAM_VALUE.C } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C}] ${MODELPARAM_VALUE.C}
}

proc update_MODELPARAM_VALUE.x_DATA_WIDTH { MODELPARAM_VALUE.x_DATA_WIDTH PARAM_VALUE.x_DATA_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.x_DATA_WIDTH}] ${MODELPARAM_VALUE.x_DATA_WIDTH}
}

proc update_MODELPARAM_VALUE.w_DATA_WIDTH { MODELPARAM_VALUE.w_DATA_WIDTH PARAM_VALUE.w_DATA_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.w_DATA_WIDTH}] ${MODELPARAM_VALUE.w_DATA_WIDTH}
}

proc update_MODELPARAM_VALUE.b_DATA_WIDTH { MODELPARAM_VALUE.b_DATA_WIDTH PARAM_VALUE.b_DATA_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.b_DATA_WIDTH}] ${MODELPARAM_VALUE.b_DATA_WIDTH}
}

proc update_MODELPARAM_VALUE.y_DATA_WIDTH { MODELPARAM_VALUE.y_DATA_WIDTH PARAM_VALUE.y_DATA_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.y_DATA_WIDTH}] ${MODELPARAM_VALUE.y_DATA_WIDTH}
}

proc update_MODELPARAM_VALUE.NUM_BANKS { MODELPARAM_VALUE.NUM_BANKS PARAM_VALUE.NUM_BANKS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.NUM_BANKS}] ${MODELPARAM_VALUE.NUM_BANKS}
}

proc update_MODELPARAM_VALUE.WORD_SIZE { MODELPARAM_VALUE.WORD_SIZE PARAM_VALUE.WORD_SIZE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.WORD_SIZE}] ${MODELPARAM_VALUE.WORD_SIZE}
}

proc update_MODELPARAM_VALUE.MEM_DEPTH { MODELPARAM_VALUE.MEM_DEPTH PARAM_VALUE.MEM_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MEM_DEPTH}] ${MODELPARAM_VALUE.MEM_DEPTH}
}

proc update_MODELPARAM_VALUE.ADDR_WIDTH { MODELPARAM_VALUE.ADDR_WIDTH PARAM_VALUE.ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ADDR_WIDTH}] ${MODELPARAM_VALUE.ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.MAX_WEIGHTS { MODELPARAM_VALUE.MAX_WEIGHTS PARAM_VALUE.MAX_WEIGHTS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MAX_WEIGHTS}] ${MODELPARAM_VALUE.MAX_WEIGHTS}
}

proc update_MODELPARAM_VALUE.CFG_WIDTH { MODELPARAM_VALUE.CFG_WIDTH PARAM_VALUE.CFG_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CFG_WIDTH}] ${MODELPARAM_VALUE.CFG_WIDTH}
}

proc update_MODELPARAM_VALUE.CFG_DEPTH { MODELPARAM_VALUE.CFG_DEPTH PARAM_VALUE.CFG_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CFG_DEPTH}] ${MODELPARAM_VALUE.CFG_DEPTH}
}

proc update_MODELPARAM_VALUE.CFG_ADDR_WIDTH { MODELPARAM_VALUE.CFG_ADDR_WIDTH PARAM_VALUE.CFG_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CFG_ADDR_WIDTH}] ${MODELPARAM_VALUE.CFG_ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.NUM_CHUNKS { MODELPARAM_VALUE.NUM_CHUNKS PARAM_VALUE.NUM_CHUNKS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.NUM_CHUNKS}] ${MODELPARAM_VALUE.NUM_CHUNKS}
}

proc update_MODELPARAM_VALUE.C_M_AXI_ADDR_WIDTH { MODELPARAM_VALUE.C_M_AXI_ADDR_WIDTH PARAM_VALUE.C_M_AXI_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_M_AXI_ADDR_WIDTH}] ${MODELPARAM_VALUE.C_M_AXI_ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.C_M_AXI_DATA_WIDTH { MODELPARAM_VALUE.C_M_AXI_DATA_WIDTH PARAM_VALUE.C_M_AXI_DATA_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_M_AXI_DATA_WIDTH}] ${MODELPARAM_VALUE.C_M_AXI_DATA_WIDTH}
}

proc update_MODELPARAM_VALUE.CLIP_MAX { MODELPARAM_VALUE.CLIP_MAX PARAM_VALUE.CLIP_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CLIP_MAX}] ${MODELPARAM_VALUE.CLIP_MAX}
}

proc update_MODELPARAM_VALUE.CLIP_MIN { MODELPARAM_VALUE.CLIP_MIN PARAM_VALUE.CLIP_MIN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CLIP_MIN}] ${MODELPARAM_VALUE.CLIP_MIN}
}

