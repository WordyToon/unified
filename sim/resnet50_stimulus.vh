// =============================================================================
// ResNet-50 (1,000 ImageNet Classes) Stimulus for tb_unified_c_engine.v
// =============================================================================

task automatic load_resnet50_microcode;
begin
    u_dut.u_engine.u_config_mem.mem[0] = 128'h00007d001e42083400010881444000f1;
    u_dut.u_engine.u_config_mem.mem[1] = 128'h00007d101e42083400010881444000f1;
    u_dut.u_engine.u_config_mem.mem[2] = 128'h00007d201e42083400010881444000f1;
    u_dut.u_engine.u_config_mem.mem[3] = 128'h00007d301e42083404310881444000f1;
end
endtask

task automatic load_resnet50_stimulus;
begin
    // Load input features (Golden Retriever dog) into DDR at 0x0
    $readmemh("dog_features_8bit.hex", u_ddr.mem, 0);
    $display("      [OK] Preprocessed test features loaded into DDR at 0x0");
    // Load weights into DDR at 0x20000 (word 8192)
    $readmemh("weights_gemv_8bit.hex", u_ddr.mem, 8192);
    $display("      [OK] Pre-trained weights loaded into DDR at 0x20000");
end
endtask

