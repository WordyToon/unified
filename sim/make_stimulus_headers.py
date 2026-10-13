import json
import os

unified_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sim_dir = os.path.join(unified_dir, "sim")

def generate_gemv_stimulus(model_name, prefix, microcode_file, hex_weights, hex_features, out_vh):
    with open(microcode_file, "r") as f:
        uops = json.load(f)
    
    with open(out_vh, "w") as f:
        f.write(f"// =============================================================================\n")
        f.write(f"// {model_name} Stimulus for tb_unified_c_engine.v\n")
        f.write(f"// =============================================================================\n\n")
        
        f.write(f"task automatic load_{prefix}_microcode;\nbegin\n")
        for u in uops:
            hex_val = u.get("hex128", f"{u['hex96']:0>32}")
            f.write(f"    u_dut.u_engine.u_config_mem.mem[{u['pc']}] = 128'h{hex_val};\n")
        f.write("end\nendtask\n\n")
        
        f.write(f"task automatic load_{prefix}_stimulus;\nbegin\n")
        f.write(f'    // Load input features (Golden Retriever dog) into DDR at 0x0\n')
        f.write(f'    $readmemh("{hex_features}", u_ddr.mem, 0);\n')
        f.write(f'    $display("      [OK] Preprocessed test features loaded into DDR at 0x0");\n')
        f.write(f'    // Load weights into DDR at 0x20000 (word 8192)\n')
        f.write(f'    $readmemh("{hex_weights}", u_ddr.mem, 8192);\n')
        f.write(f'    $display("      [OK] Pre-trained weights loaded into DDR at 0x20000");\n')
        f.write("end\nendtask\n\n")
    print(f"[+] Wrote {out_vh}")

# 1. MobileNetV2
generate_gemv_stimulus(
    "MobileNetV2 (1,000 ImageNet Classes)",
    "mobilenetv2",
    os.path.join(unified_dir, "models", "mobilenetv2", "microcode_gemv.json"),
    "mobilenetv2_weights_gemv_8bit.hex",
    "mobilenetv2_dog_features_8bit.hex",
    os.path.join(sim_dir, "mobilenetv2_stimulus.vh")
)

# 2. EfficientNet-B0
generate_gemv_stimulus(
    "EfficientNet-B0 (1,000 ImageNet Classes)",
    "efficientnetb0",
    os.path.join(unified_dir, "models", "efficientnetb0", "microcode_gemv.json"),
    "efficientnetb0_weights_gemv_8bit.hex",
    "efficientnetb0_dog_features_8bit.hex",
    os.path.join(sim_dir, "efficientnetb0_stimulus.vh")
)

# 3. ResNet-50
generate_gemv_stimulus(
    "ResNet-50 (1,000 ImageNet Classes)",
    "resnet50",
    os.path.join(unified_dir, "models", "resnet50_full", "microcode_gemv.json"),
    "weights_gemv_8bit.hex",
    "dog_features_8bit.hex",
    os.path.join(sim_dir, "resnet50_stimulus.vh")
)

