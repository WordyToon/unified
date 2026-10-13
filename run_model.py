#!/usr/bin/env python3
"""
================================================================================
  Unified 256-PE Systolic Neural Network Accelerator: Master CLI & Model Selector
================================================================================
Target: AMD Zynq UltraScale+ ZCU106 Evaluation Platform (xczu7ev-ffvc1156-2-e)
Supported Vision Networks:
  1. LeNet-5 (MNIST 10-Digit Handwritten Classification)
  2. MobileNetV2 (ImageNet 1,000 Classes, Inverted Residual & Depthwise Conv)
  3. EfficientNet-B0 (ImageNet 1,000 Classes, MBConv & Hardware Swish LUT)
  4. ResNet-50 (ImageNet 1,000 Classes, 53-Layer Residual Backbone & Bottlenecks)

Principles:
  - Single Reusable RTL Systolic Accelerator Bitstream.
  - Zero Math Fallback on PS (ARM Cortex-A53 only interacts with DDR & polls start/done).
  - Deterministic Ping-Pong Multi-Bank UltraRAM Management with Compile-Time DDR Spilling.
  - Dual Execution Mode: Cycle-Accurate Vivado XSim Simulation & JTAG Baremetal Deployment.
================================================================================
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
RTL_DIR       = os.path.join(WORKSPACE_DIR, "rtl")
SIM_DIR       = os.path.join(WORKSPACE_DIR, "sim")
MODELS_DIR    = os.path.join(WORKSPACE_DIR, "models")
BAREMETAL_DIR = os.path.join(WORKSPACE_DIR, "baremetal")
DATA_DIR      = os.path.join(BAREMETAL_DIR, "data")
FPGA_DIR      = os.path.join(WORKSPACE_DIR, "fpga")

# ==============================================================================
# 1. Hardware Configuration Specifications
# ==============================================================================
HW_CONFIGS = {
    "8bit": {
        "precision_str": "8bit",
        "x_data_width": 8,
        "w_data_width": 8,
        "b_data_width": 16,
        "y_data_width": 32,
        "clip_max": 127,
        "clip_min": -128,
        "r": 8,
        "c": 32,
        "total_pes": 256,
        "target_clock_mhz": 75.0,
        "num_banks": 16,
        "word_size_bits": 128,
        "bytes_per_row": 256,
        "mem_depth_rows": 15360,
        "capacity_mb": 3.93,
        "cfg_depth": 256,
        "num_chunks": 196
    },
    "16bit": {
        "precision_str": "16bit",
        "x_data_width": 16,
        "w_data_width": 16,
        "b_data_width": 32,
        "y_data_width": 64,
        "clip_max": 32767,
        "clip_min": -32768,
        "r": 8,
        "c": 32,
        "total_pes": 256,
        "target_clock_mhz": 70.0,
        "num_banks": 16,
        "word_size_bits": 128,
        "bytes_per_row": 256,
        "mem_depth_rows": 15360,
        "capacity_mb": 3.93,
        "cfg_depth": 256,
        "num_chunks": 196
    }
}

# Physical DDR Memory Layout on AMD ZCU106
DDR_MEMORY_MAP = {
    "DDR_ACCEL_MNIST_IMG_BASE":    0x20000000,
    "DDR_ACCEL_IMAGENET_IMG_BASE": 0x20100000,
    "DDR_ACCEL_MICROCODE_BASE":    0x21000000,
    "DDR_ACCEL_MODEL_SEL_BASE":    0x21FF0000,
    "DDR_ACCEL_LOGITS_BASE":       0x22000000,
    "DDR_ACCEL_MNIST_WT_BASE":     0x30000000,
    "DDR_ACCEL_R50_WT_BASE":       0x31000000,
    "DDR_ACCEL_MBV2_WT_BASE":      0x35000000,
    "DDR_ACCEL_EFF_WT_BASE":       0x36000000
}

MODEL_MAILBOX_CODES = {
    "all":            0,
    "mnist":          1,
    "resnet50":       2,
    "mobilenetv2":    3,
    "efficientnetb0": 4
}


def export_hw_config(precision="8bit"):
    """Emits hw_config.vh and hw_config.json for synthesis and simulation."""
    cfg = HW_CONFIGS[precision]
    json_path = os.path.join(RTL_DIR, "hw_config.json")
    with open(json_path, "w") as f:
        json.dump(cfg, f, indent=2)

    src_vh = os.path.join(RTL_DIR, f"hw_config_{precision}.vh")
    dst_vh = os.path.join(RTL_DIR, "hw_config.vh")
    if os.path.exists(src_vh):
        shutil.copyfile(src_vh, dst_vh)

    print(f"[+] Exported Hardware Configuration:")
    print(f"    - Dimensions: {cfg['r']} Rows x {cfg['c']} Cols ({cfg['total_pes']} PEs)")
    print(f"    - Clock Rate: {cfg['target_clock_mhz']} MHz on AMD ZCU106")
    print(f"    - Arithmetic: INT{cfg['x_data_width']} with INT{cfg['y_data_width']} Accumulators")
    print(f"    - On-Chip:    {cfg['num_banks']} Banks x 128-bit ({cfg['bytes_per_row']} B/row, {cfg['capacity_mb']} MB UltraRAM)")
    print(f"    - Header:     {dst_vh}")
    print(f"    - Metadata:   {json_path}")


def export_soft_config(model_name, precision="8bit"):
    """Validates and displays software configuration, microcode UOPs, and memory map."""
    print(f"\n[+] Software Configuration & Buffer Scheduling for {model_name.upper()} ({precision}):")
    
    if model_name == "mnist":
        uops_file = os.path.join(MODELS_DIR, "mnist", "microcode.json")
        ddr_img   = DDR_MEMORY_MAP["DDR_ACCEL_MNIST_IMG_BASE"]
        ddr_wt    = DDR_MEMORY_MAP["DDR_ACCEL_MNIST_WT_BASE"]
        classes   = 10
        chunks    = 1
    elif model_name == "mobilenetv2":
        uops_file = os.path.join(MODELS_DIR, "mobilenetv2", "microcode_gemv.json")
        ddr_img   = DDR_MEMORY_MAP["DDR_ACCEL_IMAGENET_IMG_BASE"]
        ddr_wt    = DDR_MEMORY_MAP["DDR_ACCEL_MBV2_WT_BASE"]
        classes   = 1000
        chunks    = 1
    elif model_name == "efficientnetb0":
        uops_file = os.path.join(MODELS_DIR, "efficientnetb0", "microcode_gemv.json")
        ddr_img   = DDR_MEMORY_MAP["DDR_ACCEL_IMAGENET_IMG_BASE"]
        ddr_wt    = DDR_MEMORY_MAP["DDR_ACCEL_EFF_WT_BASE"]
        classes   = 1000
        chunks    = 1
    elif model_name == "resnet50":
        uops_file = os.path.join(MODELS_DIR, "resnet50_full", "microcode_gemv.json")
        ddr_img   = DDR_MEMORY_MAP["DDR_ACCEL_IMAGENET_IMG_BASE"]
        ddr_wt    = DDR_MEMORY_MAP["DDR_ACCEL_R50_WT_BASE"]
        classes   = 1000
        chunks    = 1
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if os.path.exists(uops_file):
        with open(uops_file, "r") as f:
            uops = json.load(f)
        print(f"    - Microcode UOPs Count: {len(uops)} operations")
        for u in uops:
            print(f"      PC {u['pc']:02d}: {u.get('name', 'Instruction')} | Hex: {u.get('hex128', '')}")
    
    print(f"    - Target Classes:       {classes}")
    print(f"    - Spatial Tiling Chunks: {chunks}")
    print(f"    - DDR Image Buffer:     0x{ddr_img:08X}")
    print(f"    - DDR Weights Buffer:   0x{ddr_wt:08X}")
    print(f"    - DDR Output Logits:    0x{DDR_MEMORY_MAP['DDR_ACCEL_LOGITS_BASE']:08X}")
    print(f"    - Mailbox Control Code: {MODEL_MAILBOX_CODES[model_name]}")


def run_simulation(model_name, precision="8bit"):
    """Compiles RTL and executes Vivado XSim testbench."""
    print("=" * 80)
    print(f"  RUNNING VIVADO XSIM SIMULATION: {model_name.upper()} ({precision})")
    print("=" * 80)

    bat_file = os.path.join(SIM_DIR, "run_sim.bat")
    cmd = ["cmd", "/c", bat_file, model_name, precision]
    
    res = subprocess.run(cmd, cwd=WORKSPACE_DIR, text=True, capture_output=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"Simulation failed with exit code {res.returncode}")


def run_baremetal_build(precision="8bit"):
    """Compiles baremetal Cortex-A53 ELF application."""
    print("=" * 80)
    print(f"  BUILDING BAREMETAL CORTEX-A53 ELF ({precision})")
    print("=" * 80)
    
    ps1_file = os.path.join(BAREMETAL_DIR, "build_elf.ps1")
    cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", ps1_file, "-Target", precision]
    res = subprocess.run(cmd, cwd=WORKSPACE_DIR, text=True, capture_output=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"Baremetal ELF build failed with exit code {res.returncode}")


def run_jtag_deploy(model_name, precision="8bit", image="dog", digit=7):
    """Executes XSDB automated JTAG deployment on physical ZCU106 hardware."""
    print("=" * 80)
    print(f"  LAUNCHING AUTOMATED JTAG DEPLOYMENT ON AMD ZCU106")
    print(f"  Model: {model_name.upper()} | Precision: {precision}")
    print("=" * 80)

    img_arg = f"digit{digit}" if model_name == "mnist" else image
    ps1_map = {
        "mnist":          os.path.join(WORKSPACE_DIR, "run_mnist.ps1"),
        "mobilenetv2":    os.path.join(WORKSPACE_DIR, "run_mobilenetv2.ps1"),
        "efficientnetb0": os.path.join(WORKSPACE_DIR, "run_efficientnetb0.ps1"),
        "resnet50":       os.path.join(WORKSPACE_DIR, "run_resnet50.ps1"),
        "all":            os.path.join(WORKSPACE_DIR, "run_all.ps1")
    }

    script = ps1_map[model_name]
    if model_name == "mnist":
        cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", script, "-Precision", precision, "-Digit", str(digit)]
    else:
        cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", script, "-Precision", precision, "-Image", img_arg]

    print(f"[*] Invoking: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=WORKSPACE_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Unified 256-PE Systolic Accelerator CLI (AMD ZCU106 Platform)"
    )
    parser.add_argument(
        "--model",
        choices=["mnist", "mobilenetv2", "efficientnetb0", "resnet50", "all"],
        default="mnist",
        help="Select neural network model architecture to configure or execute."
    )
    parser.add_argument(
        "--precision",
        choices=["8bit", "16bit"],
        default="8bit",
        help="Arithmetic precision mode (default: 8bit)."
    )
    parser.add_argument(
        "--action",
        choices=["config", "compile", "sim", "build_elf", "deploy", "all"],
        default="sim",
        help="Action to perform: config, compile, sim, build_elf, deploy, or all."
    )
    parser.add_argument(
        "--image",
        choices=["dog", "cat", "car", "aeroplane", "person", "traffic", "truck"],
        default="dog",
        help="ImageNet test image stimulus."
    )
    parser.add_argument(
        "--digit",
        type=int,
        choices=list(range(10)),
        default=0,
        help="MNIST test digit stimulus (0..9)."
    )

    args = parser.parse_args()

    # Step 1: Hardware & Software Configuration
    export_hw_config(args.precision)
    if args.model != "all":
        export_soft_config(args.model, args.precision)
    else:
        for m in ["mnist", "mobilenetv2", "efficientnetb0", "resnet50"]:
            export_soft_config(m, args.precision)

    if args.action == "config":
        return

    # Step 2: Simulation Execution
    if args.action in ["sim", "all"]:
        if args.model == "all":
            for m in ["mnist", "mobilenetv2", "efficientnetb0", "resnet50"]:
                run_simulation(m, args.precision)
        else:
            run_simulation(args.model, args.precision)

    # Step 3: Baremetal ELF Build
    if args.action in ["build_elf", "all", "deploy"]:
        run_baremetal_build(args.precision)

    # Step 4: Hardware JTAG Baremetal Deployment
    if args.action in ["deploy", "all"]:
        run_jtag_deploy(args.model, args.precision, args.image, args.digit)


if __name__ == "__main__":
    main()

