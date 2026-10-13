#!/usr/bin/env python3
"""
Regenerates all MNIST digit binaries (digit0..digit9 for 8bit and 16bit)
with the exact 32x32 padded canvas required by the accelerator.
"""
import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLES_JSON = os.path.join(BASE_DIR, "pynq", "mnist", "mnist_test_samples.json")
DATA_DIR = os.path.join(BASE_DIR, "baremetal", "data")

with open(SAMPLES_JSON, "r") as f:
    samples = json.load(f)

for d in range(10):
    entry = samples[f"digit_{d}"]
    img_list = entry["image"]
    img_uint8 = np.array(img_list, dtype=np.uint8) # 28x28

    # 8-bit:
    canvas_8 = np.zeros((32, 32), dtype=np.int8)
    norm_8 = ((img_uint8.astype(np.float32) / 255.0) * 254.0 - 127.0).astype(np.int8)
    canvas_8[2:30, 2:30] = norm_8
    out_8 = os.path.join(DATA_DIR, f"digit{d}_8bit.bin")
    with open(out_8, "wb") as f:
        f.write(canvas_8.tobytes())

    # 16-bit:
    canvas_16 = np.zeros((32, 32), dtype=np.int16)
    norm_16 = ((img_uint8.astype(np.float32) / 255.0) * 65534.0 - 32767.0).astype(np.int16)
    canvas_16[2:30, 2:30] = norm_16
    out_16 = os.path.join(DATA_DIR, f"digit{d}_16bit.bin")
    with open(out_16, "wb") as f:
        f.write(canvas_16.tobytes())

    print(f"Exported digit {d}: {os.path.basename(out_8)} (1024 B), {os.path.basename(out_16)} (2048 B)")

print("[+] All 10 MNIST test digits successfully regenerated!")
