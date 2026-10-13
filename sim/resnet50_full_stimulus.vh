// Auto-generated Full ResNet-50 (8-bit) Stimulus for tb_unified_c_engine.v
task automatic load_resnet50_full_microcode;
begin
    u_dut.u_engine.u_config_mem.mem[0] = 96'h00000258967c00009ddc100c;
    u_dut.u_engine.u_config_mem.mem[1] = 96'h007de67c417c0258a0cd0100;
    u_dut.u_engine.u_config_mem.mem[2] = 96'h007de578417c067ca0cd0100;
    u_dut.u_engine.u_config_mem.mem[3] = 96'h007de67c417c0578a0cd0100;
    u_dut.u_engine.u_config_mem.mem[4] = 96'h007de578427c067ca0ce0100;
    u_dut.u_engine.u_config_mem.mem[5] = 96'h007de67c417c0578a0ce0200;
    u_dut.u_engine.u_config_mem.mem[6] = 96'h007de578417c067ca0ce0200;
    u_dut.u_engine.u_config_mem.mem[7] = 96'h007de67c417c0578a0ce0200;
    u_dut.u_engine.u_config_mem.mem[8] = 96'h007de578427c067ca0cffe00;
    u_dut.u_engine.u_config_mem.mem[9] = 96'h007de67c417c0578a0cffffc;
    u_dut.u_engine.u_config_mem.mem[10] = 96'h007de578417c067ca0cffffc;
    u_dut.u_engine.u_config_mem.mem[11] = 96'h007de67c417c0578a0cffffc;
    u_dut.u_engine.u_config_mem.mem[12] = 96'h007de578427c067ca0cffffc;
    u_dut.u_engine.u_config_mem.mem[13] = 96'h007de67c417c0578a0cffffc;
    u_dut.u_engine.u_config_mem.mem[14] = 96'h007de578417c067ca0cffffc;
    u_dut.u_engine.u_config_mem.mem[15] = 96'h000007d0413c0578a047fcf1;
    u_dut.u_engine.u_config_mem.mem[16] = 96'h000007d1413c0578a047fcf1;
    u_dut.u_engine.u_config_mem.mem[17] = 96'h000007d2413c0578a047fcf1;
    u_dut.u_engine.u_config_mem.mem[18] = 96'h800007d3413c0578a047fcf1;
end
endtask

task automatic load_resnet50_full_stimulus;
begin
    // Load preprocessed test image into DDR memory model at 0x0
    $readmemh("input_image.hex", u_ddr.mem, 0);
    $display("      [OK] Preprocessed test image loaded into DDR memory model at 0x0");
    // Load authentic pre-trained weights into DDR memory at 0x20000 (word 8192)
    $readmemh("weights.hex", u_ddr.mem, 8192);
    $display("      [OK] Pre-trained weights loaded into DDR memory model at 0x20000");
end
endtask
