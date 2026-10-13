#!/usr/bin/env python3
"""
================================================================================
Unified Model Compiler & Asset Generator for AMD ZCU106 Baremetal Hardware
================================================================================
Compiles all 4 vision models independently:
  1. MNIST LeNet-5 (10 digits)
  2. ResNet-50 (1,000 ImageNet categories, 53 Conv layers)
  3. MobileNetV2 (1,000 ImageNet categories, Depthwise Mode 2'b11)
  4. EfficientNet-B0 (1,000 ImageNet categories, LUT Swish & MBConv)

Generates:
  - weights.bin & microcode.bin in baremetal/data/ (INT8 & INT16)
  - baremetal/src/model_microcodes.h (embedded C header for standalone execution)
  - test_images/ binary format (.bin) in baremetal/data/ for DDR preloading
================================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import argparse
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BAREMETAL_DIR = os.path.join(BASE_DIR, "baremetal", "data")
HEADER_PATH = os.path.join(BASE_DIR, "baremetal", "src", "model_microcodes.h")
os.makedirs(BAREMETAL_DIR, exist_ok=True)


def json_microcode_to_bin(json_path, bin_path):
    """Converts microcode.json (pc, data0..data3) into raw 32-bit binary array."""
    if not os.path.exists(json_path):
        return
    with open(json_path, "r") as f:
        uops = json.load(f)
    words = []
    for uop in uops:
        words.extend([
            uop["pc"],
            uop["data0"],
            uop["data1"],
            uop["data2"],
            uop.get("data3", 0)
        ])
    arr = np.array(words, dtype=np.uint32)
    with open(bin_path, "wb") as f:
        f.write(arr.tobytes())
    print(f"  [+] Wrote {len(uops)} uops ({len(words)*4} bytes) -> {os.path.basename(bin_path)}")


def format_image_for_ddr(img_path, out_bin_path, precision=8):
    """Preprocesses a 224x224 RGB image into raw spatial chunk binary format for DDR."""
    if not os.path.exists(img_path):
        return
    img = Image.open(img_path).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    # 196 spatial chunks of 16x16x3 (3 rows x 256 elements = 768 elements per chunk, total 150,528 elements)
    scale = 32000.0 if precision == 16 else 250.0
    dtype = np.int16 if precision == 16 else np.int8
    clip_min = -32768 if precision == 16 else -128
    clip_max = 32767 if precision == 16 else 127

    norm = np.clip(np.round((arr / 255.0 - 0.5) * scale), clip_min, clip_max).astype(dtype)
    chunks_tensor = np.zeros((196, 3, 256), dtype=dtype)
    for r in range(14):
        for c in range(14):
            chunk_idx = r * 14 + c
            patch = norm[r*16:(r+1)*16, c*16:(c+1)*16, :]
            for ch in range(3):
                chunks_tensor[chunk_idx, ch, :] = patch[:, :, ch].flatten()

    data = chunks_tensor.flatten()
    with open(out_bin_path, "wb") as f:
        f.write(data.tobytes())
    print(f"  [+] Formatted image -> {os.path.basename(out_bin_path)} ({len(data)*dtype().itemsize} bytes)")


def update_model_microcodes_header():
    """Generates baremetal/src/model_microcodes.h from all compiled microcode JSONs."""
    print("\n[*] Synchronizing baremetal/src/model_microcodes.h...")

    models_info = [
        ("mnist_uops", os.path.join(SCRIPT_DIR, "mnist", "microcode.json")),
        ("resnet50_8bit_uops", os.path.join(SCRIPT_DIR, "resnet50_full", "microcode.json")),
        ("resnet50_16bit_uops", os.path.join(SCRIPT_DIR, "resnet50_full", "microcode_16bit.json")),
        ("mobilenetv2_8bit_uops", os.path.join(SCRIPT_DIR, "mobilenetv2", "microcode.json")),
        ("mobilenetv2_16bit_uops", os.path.join(SCRIPT_DIR, "mobilenetv2", "microcode_16bit.json")),
        ("efficientnetb0_8bit_uops", os.path.join(SCRIPT_DIR, "efficientnetb0", "microcode.json")),
        ("efficientnetb0_16bit_uops", os.path.join(SCRIPT_DIR, "efficientnetb0", "microcode_16bit.json")),
    ]

    lines = [
        "#ifndef MODEL_MICROCODES_H",
        "#define MODEL_MICROCODES_H",
        "",
        "#include <stdint.h>",
        "",
        "// Auto-generated sequencer microcode tables for AMD ZCU106 Baremetal Execution",
        ""
    ]

    for name, path in models_info:
        if not os.path.exists(path):
            # Fallback to standard microcode.json if specific 16bit not separate
            fallback = path.replace("_16bit.json", ".json")
            if os.path.exists(fallback):
                path = fallback
            else:
                continue

        with open(path, "r") as f:
            uops = json.load(f)

        macro_name = f"NUM_UOPS_{name.upper()}"
        lines.append(f"#define {macro_name} {len(uops)}U")
        lines.append(f"static const uint32_t {name}[{len(uops) * 5}] = {{")
        for u in uops:
            pc = u["pc"]
            d0 = u["data0"]
            d1 = u["data1"]
            d2 = u["data2"]
            d3 = u.get("data3", 0)
            cmt = u.get("name", f"UOP {pc}")
            lines.append(f"    0x{pc:08X}U, 0x{d0:08X}U, 0x{d1:08X}U, 0x{d2:08X}U, 0x{d3:08X}U, // PC=0x{pc:02X}: {cmt}")
        lines.append("};")
        lines.append("")

    lines.append("#endif // MODEL_MICROCODES_H")
    lines.append("")

    with open(HEADER_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"[+] Successfully wrote {HEADER_PATH}")


import subprocess


def compile_all(precisions=("8bit", "16bit")):
    print("=" * 80)
    print("  COMPILING ALL VISION MODELS & GENERATING HARDWARE DDR ASSETS")
    print("=" * 80)

    # 1. MNIST LeNet-5
    print("\n[1/4] Compiling MNIST LeNet-5...")
    mnist_compile = os.path.join(SCRIPT_DIR, "mnist", "compile.py")
    if os.path.exists(mnist_compile):
        subprocess.run([sys.executable, mnist_compile], check=True)
        json_microcode_to_bin(
            os.path.join(SCRIPT_DIR, "mnist", "microcode.json"),
            os.path.join(BAREMETAL_DIR, "mnist_microcode.bin")
        )

    # 2. ResNet-50 Full Backbone
    for prec in precisions:
        print(f"\n[2/4] Compiling ResNet-50 ({prec})...")
        r50_compile = os.path.join(SCRIPT_DIR, "resnet50_full", "compile.py")
        if os.path.exists(r50_compile):
            subprocess.run([sys.executable, r50_compile, "--precision", prec], check=True)
            if "16" in prec:
                m_src = os.path.join(SCRIPT_DIR, "resnet50_full", "microcode.json")
                m_dst = os.path.join(SCRIPT_DIR, "resnet50_full", "microcode_16bit.json")
                if os.path.exists(m_src):
                    with open(m_src, "r") as sf, open(m_dst, "w") as df:
                        df.write(sf.read())

    # 3. MobileNetV2
    for prec in precisions:
        print(f"\n[3/4] Compiling MobileNetV2 ({prec})...")
        mbv2_compile = os.path.join(SCRIPT_DIR, "mobilenetv2", "compile.py")
        if os.path.exists(mbv2_compile):
            subprocess.run([sys.executable, mbv2_compile, "--precision", prec], check=True)
            if "16" in prec:
                m_src = os.path.join(SCRIPT_DIR, "mobilenetv2", "microcode.json")
                m_dst = os.path.join(SCRIPT_DIR, "mobilenetv2", "microcode_16bit.json")
                if os.path.exists(m_src):
                    with open(m_src, "r") as sf, open(m_dst, "w") as df:
                        df.write(sf.read())

    # 4. EfficientNet-B0
    for prec in precisions:
        print(f"\n[4/4] Compiling EfficientNet-B0 ({prec})...")
        eff_compile = os.path.join(SCRIPT_DIR, "efficientnetb0", "compile.py")
        if os.path.exists(eff_compile):
            subprocess.run([sys.executable, eff_compile, "--precision", prec], check=True)
            if "16" in prec:
                m_src = os.path.join(SCRIPT_DIR, "efficientnetb0", "microcode.json")
                m_dst = os.path.join(SCRIPT_DIR, "efficientnetb0", "microcode_16bit.json")
                if os.path.exists(m_src):
                    with open(m_src, "r") as sf, open(m_dst, "w") as df:
                        df.write(sf.read())

    # Convert test images if available
    print("\n[*] Formatting Test Images for DDR Preloading...")
    test_img_dir = os.path.join(BASE_DIR, "data", "test_images")
    if os.path.exists(test_img_dir):
        for img_name in ["dog.jpg", "car.jpg", "cat.jpg", "aeroplane.jpg"]:
            src = os.path.join(test_img_dir, img_name)
            if os.path.exists(src):
                dst_8 = os.path.join(BAREMETAL_DIR, f"{os.path.splitext(img_name)[0]}_8bit.bin")
                dst_16 = os.path.join(BAREMETAL_DIR, f"{os.path.splitext(img_name)[0]}_16bit.bin")
                format_image_for_ddr(src, dst_8, precision=8)
                format_image_for_ddr(src, dst_16, precision=16)

    # Synchronize C header
    update_model_microcodes_header()

    print("\n" + "=" * 80)
    print("  ALL MODEL COMPILATIONS & BINARY ASSETS GENERATED!")
    print(f"  Baremetal Assets Directory: {BAREMETAL_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["8bit", "16bit", "both"], default="both")
    args = parser.parse_args()
    precs = ["8bit", "16bit"] if args.precision == "both" else [args.precision]
    compile_all(precs)
