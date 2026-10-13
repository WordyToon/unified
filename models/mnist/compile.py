#!/usr/bin/env python3
"""
================================================================================
MNIST LeNet-5: Microcode & Weight Compiler for 256-PE Accelerator (8x32 @ 125 MHz)
================================================================================
Compiles the 5-stage LeNet-5 model into:
  1. microcode.json: Autonomous 128-bit instruction words for PL sequencer
  2. weights.bin: Quantized INT8 weights for SRAM preload
  3. sim/mnist_stimulus.vh: Verilog stimulus for XSim behavioral verification
  4. baremetal/data/mnist_microcode.bin: Binary format for DDR preloading
================================================================================
"""

import os
import sys
import json
import numpy as np

# Memory Map Base Addresses in SRAM (256 rows x 256 bytes)
ADDR_INPUT_TILES   = 0       # Row 0: Input 8x32 image (256 bytes)
ADDR_CONV1_WT      = 10      # Row 10: Conv1 weights & biases (36 + 8 bytes)
ADDR_CONV1_OUT     = 15      # Rows 15..18: Conv1 4 channels (64 bytes each)
ADDR_CONV2_WT      = 20      # Row 20: Conv2 weights & biases (36 + 2 bytes)
ADDR_CONV2_OUT     = 35      # Row 35: Conv2 output (64 bytes)
ADDR_FC1_WT        = 40      # Rows 40..103: FC1 weights (64 rows x 32 neurons)
ADDR_FC1_OUT       = 105     # Row 105: FC1 output vector (32 neurons)
ADDR_FC2_WT        = 110     # Rows 110..141: FC2 weights (32 rows x 16 neurons)
ADDR_FC2_OUT       = 145     # Row 145: FC2 output vector (16 neurons)
ADDR_FC3_WT        = 150     # Rows 150..165: FC3 weights (16 rows x 10 class logits)
ADDR_FC3_OUT       = 170     # Row 170: FC3 output class logits (10 classes: 0..9)


def pack_config_word(mode, chin, chout, kw, kh, shift_bits, in_base, wt_base, out_base, is_last,
                     act_type=0, pad_type=1, stride=1, pool_type=0, pool_size=0, pool_stride=2,
                     residual_en=0, residual_base_addr=0,
                     dma_rd_en=0, dma_wr_en=0, dma_beats=0, num_chunks=1):
    """Packs layer parameters into a 128-bit Config Word matching RTL config_mem."""
    word = 0
    # Word 0 [31:0]
    word |= (int(mode) & 0x3) << 0
    word |= (int(chin) & 0xFF) << 2
    word |= (int(chout) & 0xFF) << 10
    word |= (int(kw) & 0xF) << 18
    word |= (int(kh) & 0xF) << 22
    word |= (int(shift_bits) & 0x1F) << 26
    word |= (int(pad_type) & 0x1) << 31

    # Word 1 [63:32]
    word |= (int(in_base) & 0x3FFF) << 32
    word |= (int(wt_base) & 0x3FFF) << 46
    word |= (int(act_type) & 0x3) << 60
    word |= (int(stride) & 0x3) << 62

    # Word 2 [95:64]
    word |= (int(out_base) & 0x3FFF) << 64
    word |= (int(residual_base_addr) & 0x3FFF) << 78
    word |= (int(residual_en) & 0x1) << 92
    word |= (int(dma_rd_en) & 0x1) << 93
    word |= (int(dma_wr_en) & 0x1) << 94
    word |= (int(is_last) & 0x1) << 95

    # Word 3 [127:96]
    word |= (int(pool_type) & 0x3) << 96
    word |= (int(pool_size) & 0x3) << 98
    word |= (int(pool_stride) & 0x3) << 100
    word |= (int(num_chunks) & 0xFF) << 102
    word |= (int(dma_beats) & 0x7F) << 110

    return word


def compile_mnist():
    print("=" * 80)
    print("   COMPILING MNIST LENET-5 FOR 256-PE C-ENGINE ACCELERATOR (8x32 @ 100 MHz)")
    print("=" * 80)

    model_dir = os.path.dirname(os.path.abspath(__file__))
    unified_dir = os.path.abspath(os.path.join(model_dir, "..", ".."))
    baremetal_data_dir = os.path.join(unified_dir, "baremetal", "data")
    os.makedirs(baremetal_data_dir, exist_ok=True)

    # Load calibrated shifts if available
    shifts_path = os.path.join(model_dir, "calib_shifts.json")
    if os.path.exists(shifts_path):
        with open(shifts_path, "r") as f:
            s_cfg = json.load(f)
        s1 = s_cfg.get("shift1", 9)
        s2 = s_cfg.get("shift2", 8)
        s_fc1 = s_cfg.get("shift_fc1", 7)
        s_fc2 = s_cfg.get("shift_fc2", 7)
        s_fc3 = s_cfg.get("shift_fc3", 8)
    else:
        s1, s2, s_fc1, s_fc2, s_fc3 = 9, 8, 7, 7, 8

    instructions = []
    pc = 0

    # Stage 1: Conv1 + ReLU + MaxPool (2x2 s2): (1 ch in -> 4 ch out)
    w = pack_config_word(
        mode=0, chin=1, chout=4, kw=3, kh=3, shift_bits=s1,
        in_base=ADDR_INPUT_TILES,
        wt_base=ADDR_CONV1_WT,
        out_base=ADDR_CONV1_OUT,
        is_last=0, act_type=1, pad_type=1, stride=1,
        pool_type=1, pool_size=1, pool_stride=2
    )
    d0 = w & 0xFFFFFFFF
    d1 = (w >> 32) & 0xFFFFFFFF
    d2 = (w >> 64) & 0xFFFFFFFF
    d3 = (w >> 96) & 0xFFFFFFFF
    instructions.append({
        "pc": pc, "name": "Stage 1: Conv1 (1->4, 3x3) + ReLU + MaxPool",
        "data0": d0, "data1": d1, "data2": d2, "data3": d3,
        "hex128": f"{w:032x}", "val128": w,
        "hex96": f"{w & ((1<<96)-1):024x}", "val96": w & ((1<<96)-1)
    })
    pc += 1

    # Stage 2: Conv2 + ReLU: (4 ch in -> 1 ch out = 64 features)
    w = pack_config_word(
        mode=0, chin=4, chout=1, kw=3, kh=3, shift_bits=s2,
        in_base=ADDR_CONV1_OUT,
        wt_base=ADDR_CONV2_WT,
        out_base=ADDR_CONV2_OUT,
        is_last=0, act_type=1, pad_type=1, stride=1,
        pool_type=0, pool_size=0, pool_stride=2
    )
    d0 = w & 0xFFFFFFFF
    d1 = (w >> 32) & 0xFFFFFFFF
    d2 = (w >> 64) & 0xFFFFFFFF
    d3 = (w >> 96) & 0xFFFFFFFF
    instructions.append({
        "pc": pc, "name": "Stage 2: Conv2 (4->1, 3x3) + ReLU (64 features)",
        "data0": d0, "data1": d1, "data2": d2, "data3": d3,
        "hex128": f"{w:032x}", "val128": w,
        "hex96": f"{w & ((1<<96)-1):024x}", "val96": w & ((1<<96)-1)
    })
    pc += 1

    # Stage 3: FC1 (64 input elements -> 32 neurons) + ReLU
    w = pack_config_word(
        mode=1, chin=64, chout=32, kw=1, kh=1, shift_bits=s_fc1,
        in_base=ADDR_CONV2_OUT, wt_base=ADDR_FC1_WT, out_base=ADDR_FC1_OUT,
        is_last=0, act_type=1
    )
    d0 = w & 0xFFFFFFFF
    d1 = (w >> 32) & 0xFFFFFFFF
    d2 = (w >> 64) & 0xFFFFFFFF
    d3 = (w >> 96) & 0xFFFFFFFF
    instructions.append({
        "pc": pc, "name": "Stage 3: FC1 (64->32) + ReLU",
        "data0": d0, "data1": d1, "data2": d2, "data3": d3,
        "hex128": f"{w:032x}", "val128": w,
        "hex96": f"{w & ((1<<96)-1):024x}", "val96": w & ((1<<96)-1)
    })
    pc += 1

    # Stage 4: FC2 (32 neurons -> 16 neurons) + ReLU
    w = pack_config_word(
        mode=1, chin=32, chout=16, kw=1, kh=1, shift_bits=s_fc2,
        in_base=ADDR_FC1_OUT, wt_base=ADDR_FC2_WT, out_base=ADDR_FC2_OUT,
        is_last=0, act_type=1
    )
    d0 = w & 0xFFFFFFFF
    d1 = (w >> 32) & 0xFFFFFFFF
    d2 = (w >> 64) & 0xFFFFFFFF
    d3 = (w >> 96) & 0xFFFFFFFF
    instructions.append({
        "pc": pc, "name": "Stage 4: FC2 (32->16) + ReLU",
        "data0": d0, "data1": d1, "data2": d2, "data3": d3,
        "hex128": f"{w:032x}", "val128": w,
        "hex96": f"{w & ((1<<96)-1):024x}", "val96": w & ((1<<96)-1)
    })
    pc += 1

    # Stage 5: FC3 (16 neurons -> 10 class logits, is_last = 1, dma_wr_en = 1)
    w = pack_config_word(
        mode=1, chin=16, chout=10, kw=1, kh=1, shift_bits=s_fc3,
        in_base=ADDR_FC2_OUT, wt_base=ADDR_FC3_WT, out_base=ADDR_FC3_OUT,
        is_last=1, act_type=0, dma_wr_en=1, dma_beats=4
    )
    d0 = w & 0xFFFFFFFF
    d1 = (w >> 32) & 0xFFFFFFFF
    d2 = (w >> 64) & 0xFFFFFFFF
    d3 = (w >> 96) & 0xFFFFFFFF
    instructions.append({
        "pc": pc, "name": "Stage 5: FC3 (16->10, Final Logits, DMA Write)",
        "data0": d0, "data1": d1, "data2": d2, "data3": d3,
        "hex128": f"{w:032x}", "val128": w,
        "hex96": f"{w & ((1<<96)-1):024x}", "val96": w & ((1<<96)-1)
    })
    pc += 1

    # Save microcode.json
    with open(os.path.join(model_dir, "microcode.json"), "w") as f:
        json.dump(instructions, f, indent=2)
    print(f"[+] Saved Microcode JSON ({len(instructions)} instructions)")

    # Save binary microcode for DDR preloading
    words = []
    for uop in instructions:
        words.extend([uop["pc"], uop["data0"], uop["data1"], uop["data2"], uop["data3"]])
    with open(os.path.join(baremetal_data_dir, "mnist_microcode.bin"), "wb") as f:
        f.write(np.array(words, dtype=np.uint32).tobytes())
    print(f"[+] Saved mnist_microcode.bin ({len(words)*4} bytes)")

    # Synchronize trained weights and stimulus using train.py
    from train import train_and_export
    train_and_export()


if __name__ == '__main__':
    compile_mnist()
