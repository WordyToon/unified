// Auto-generated ResNet-50 Bottleneck Stimulus for tb_unified_c_engine.v
task automatic load_bottleneck_microcode;
begin
    u_dut.u_engine.u_config_mem.mem[0] = 96'h0000000a140780008c441010;
    u_dut.u_engine.u_config_mem.mem[1] = 96'h000000141408000a8ccc1010;
    u_dut.u_engine.u_config_mem.mem[2] = 96'h0400001e140880148c441010;
    u_dut.u_engine.u_config_mem.mem[3] = 96'h900000321009001e8c442811;
end
endtask

task automatic load_bottleneck_image_stimulus;
begin
    u_ddr.write_word(0, 128'h0A0A0A0A0A0A0A0A0A0A0A0A0A0A0A0A);
    u_ddr.write_word(256, 128'h0A0A0A0A0A0A0A0A0A0A0A0A0A0A0A0A);
    u_ddr.write_word(512, 128'h0A0A0A0A0A0A0A0A0A0A0A0A0A0A0A0A);
    u_ddr.write_word(768, 128'h0A0A0A0A0A0A0A0A0A0A0A0A0A0A0A0A);
end
endtask
