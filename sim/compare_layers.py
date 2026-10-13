import numpy as np
import json
import torch
import sys, os
sys.path.insert(0, ".")
from models.mnist.train import Net8x32
from torchvision import datasets, transforms

with open("models/mnist/calib_shifts.json") as f:
    s_cfg = json.load(f)
s1, s2, s_fc1, s_fc2, s_fc3, s_x0 = s_cfg["shift1"], s_cfg["shift2"], s_cfg["shift_fc1"], s_cfg["shift_fc2"], s_cfg["shift_fc3"], s_cfg["s_x0"]

with open("models/mnist/weights.bin", "rb") as f:
    w_mat = np.frombuffer(f.read(), dtype=np.int8).reshape(256, 256)

w_c1 = w_mat[10, :36].reshape(4, 1, 3, 3)
b_c1 = w_mat[10, 36:44].view(np.int16)
w_c2 = w_mat[20, :36].reshape(1, 4, 3, 3)
b_c2 = w_mat[20, 36:38].view(np.int16)
w_fc1_T = w_mat[40:104, :32]
w_fc2_T = w_mat[110:142, :16]
w_fc3_T = w_mat[150:166, :10]

# Read exact stimulus image from sim/mnist_stimulus.vh
import re
with open("sim/mnist_stimulus.vh") as f:
    lines = [l for l in f if "u_ddr.write_word" in l][:16]
bytes_list = []
for l in lines:
    hex_str = re.search(r"128'h([0-9a-fA-F]+)", l).group(1)
    b = bytes.fromhex(hex_str)
    bytes_list.extend(list(reversed(b)))

arr = np.array(bytes_list, dtype=np.uint8).view(np.int8).reshape(8, 32)
x_int = arr.astype(np.int32)
xp = np.pad(x_int, ((1, 1), (1, 1)))

# Conv1
c1 = np.zeros((4, 8, 32), dtype=np.int32)
for oc in range(4):
    for r in range(8):
        for c in range(32):
            acc = np.sum(xp[r:r+3, c:c+3] * w_c1[oc, 0]) + int(b_c1[oc])
            c1[oc, r, c] = max(0, min(127, acc >> s1))

p1 = np.zeros((4, 4, 16), dtype=np.int32)
for oc in range(4):
    for r in range(4):
        for c in range(16):
            p1[oc, r, c] = np.max(c1[oc, r*2:r*2+2, c*2:c*2+2])

hex_p1 = "".join([f"{int(x)&0xFF:02x}" for x in reversed(p1[0, 0, :16])])
print(f"Python Conv1 Row 15 Bank 0: {hex_p1}")

# Conv2 (4 in, 1 out, 3x3)
p1_pad = np.pad(p1, ((0, 0), (1, 1), (1, 1)))
c2 = np.zeros((1, 4, 16), dtype=np.int32)
for r in range(4):
    for c in range(16):
        acc = np.sum(p1_pad[:, r:r+3, c:c+3] * w_c2[0, :]) + int(b_c2[0])
        c2[0, r, c] = max(0, min(127, acc >> s2))

hex_c2 = "".join([f"{int(x)&0xFF:02x}" for x in reversed(c2[0, 0, :16])])
print(f"Python Conv2 Row 35 Bank 0: {hex_c2}")

feat = c2.flatten() # 64 elements
fc1_out = np.zeros(32, dtype=np.int32)
for j in range(32):
    acc = np.sum(feat * w_fc1_T[:, j])
    fc1_out[j] = max(0, min(127, acc >> s_fc1))

hex_fc1_b0 = "".join([f"{int(x)&0xFF:02x}" for x in reversed(fc1_out[:16])])
print(f"Python FC1   Row 105 Bank 0: {hex_fc1_b0}")

fc2_out = np.zeros(16, dtype=np.int32)
for j in range(16):
    acc = np.sum(fc1_out * w_fc2_T[:, j])
    fc2_out[j] = max(0, min(127, acc >> s_fc2))

hex_fc2_b0 = "".join([f"{int(x)&0xFF:02x}" for x in reversed(fc2_out[:16])])
print(f"Python FC2   Row 145 Bank 0: {hex_fc2_b0}")

fc3_out = np.zeros(10, dtype=np.int32)
for j in range(10):
    acc = np.sum(fc2_out * w_fc3_T[:, j])
    fc3_out[j] = np.clip(acc >> s_fc3, -128, 127)

hex_fc3_b0 = "".join([f"{int(x)&0xFF:02x}" for x in reversed(fc3_out[:10])])
print(f"Python FC3   Row 170 Bank 0: {hex_fc3_b0}")
print("Python FC3 Logits:", fc3_out.tolist())

