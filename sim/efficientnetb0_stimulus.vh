// =============================================================================
// EfficientNet-B0 (1,000 ImageNet Classes) Stimulus for tb_unified_c_engine.v
// =============================================================================

task automatic load_efficientnetb0_microcode;
begin
    u_dut.u_engine.u_config_mem.mem[0] = 128'h00008001000007d010f00578984400f1;
    u_dut.u_engine.u_config_mem.mem[1] = 128'h00008001000007d110f00578984400f1;
    u_dut.u_engine.u_config_mem.mem[2] = 128'h00008001000007d210f00578984400f1;
    u_dut.u_engine.u_config_mem.mem[3] = 128'h00009001900007d310f00578984400f1;
end
endtask

task automatic load_efficientnetb0_stimulus;
begin
    // Load input features (Golden Retriever dog) into DDR at 0x0
    $readmemh("efficientnetb0_dog_features_8bit.hex", u_ddr.mem, 0);
    $display("      [OK] Preprocessed test features loaded into DDR at 0x0");
    // Load weights into DDR at 0x20000 (word 8192)
    $readmemh("efficientnetb0_weights_gemv_8bit.hex", u_ddr.mem, 8192);
    $display("      [OK] Pre-trained weights loaded into DDR at 0x20000");
end
endtask

