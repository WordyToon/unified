import numpy as np
import json
import re

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
with open("sim/mnist_stimulus.vh") as f:
    lines = [l for l in f if "u_ddr.write_word" in l][:16]
bytes_list = []
for l in lines:
    hex_str = re.search(r"128'h([0-9a-fA-F]+)", l).group(1)
    b = bytes.fromhex(hex_str)
    bytes_list.extend(list(reversed(b)))

img_pe = np.array(bytes_list, dtype=np.uint8).view(np.int8).reshape(8, 32)

# Exact RTL PE systolic convolution for Conv1 (4 filters, 1 channel, 3x3)
# Input is in Row 0 (8x32)
c1_rtl = np.zeros((4, 8, 32), dtype=np.int32)
for oc in range(4):
    wt = w_c1[oc, 0] # (3, 3)
    bias = int(b_c1[oc])
    for x in range(8):
        for y in range(32):
            acc = 0
            for v in range(3):
                r = x - 2 + v
                for h in range(3):
                    c = y - 2 + h
                    val = int(img_pe[r, c]) if (0 <= r < 8 and 0 <= c < 32) else 0
                    acc += val * int(wt[v, h])
            acc += bias
            shifted = acc >> s1
            clipped = max(-128, min(127, shifted))
            c1_rtl[oc, x, y] = max(0, clipped)

# 2x2 maxpool
p1_rtl = np.zeros((4, 4, 16), dtype=np.int32)
for oc in range(4):
    for r in range(4):
        for c in range(16):
            p1_rtl[oc, r, c] = np.max(c1_rtl[oc, 2*r:2*r+2, 2*c:2*c+2])

hex_p1 = "".join([f"{int(x)&0xFF:02x}" for x in reversed(p1_rtl[0, 0, :16])])
print(f"RTL model Conv1 Row 15 Bank 0: {hex_p1}")
print(f"XSim run  Conv1 Row 15 Bank 0: 07070707070b2226170707070707070b")

# Conv2 (1 filter, 4 channels, 3x3)
# In RTL: input rows are 15, 16, 17, 18. Each row has 256 bytes in SRAM.
# Specifically, in row 15+ch: bytes 0..63 are p1_rtl[ch].flatten(), bytes 64..255 are 0.
c2_rtl = np.zeros((8, 32), dtype=np.int32)
for ch in range(4):
    in_grid = np.zeros((8, 32), dtype=np.int32)
    in_grid.flat[:64] = p1_rtl[ch].flat[:64]
    wt = w_c2[0, ch]
    for x in range(8):
        for y in range(32):
            acc = 0
            for v in range(3):
                r = x - 2 + v
                for h in range(3):
                    c = y - 2 + h
                    val = int(in_grid[r, c]) if (0 <= r < 8 and 0 <= c < 32) else 0
                    acc += val * int(wt[v, h])
            c2_rtl[x, y] += acc

# Add bias and requantize for Conv2
c2_out = np.zeros((8, 32), dtype=np.int32)
for x in range(8):
    for y in range(32):
        acc = c2_rtl[x, y] + int(b_c2[0])
        shifted = acc >> s2
        clipped = max(-128, min(127, shifted))
        c2_out[x, y] = max(0, clipped)

hex_c2 = "".join([f"{int(x)&0xFF:02x}" for x in reversed(c2_out[0, :16])])
print(f"RTL model Conv2 Row 35 Bank 0: {hex_c2}")
print(f"XSim run  Conv2 Row 35 Bank 0: 00000001080f130c0500000000010001")

# FC1:
# In RTL, FC1 takes input from Row 35.
# Let's see what k_cnt does: k_cnt runs from 0 to chin-1 (chin=64).
# It reads input byte k_cnt from Row 35.
# That is c2_out.flatten()[:64]!
fc1_in = c2_out.flatten()[:64]
fc1_out = np.zeros(32, dtype=np.int32)
for j in range(32):
    acc = np.sum(fc1_in * w_fc1_T[:64, j])
    shifted = acc >> s_fc1
    clipped = max(-128, min(127, shifted))
    fc1_out[j] = max(0, clipped)

hex_fc1 = "".join([f"{int(x)&0xFF:02x}" for x in reversed(fc1_out[:16])])
print(f"RTL model FC1   Row 105 Bank 0: {hex_fc1}")
print(f"XSim run  FC1   Row 105 Bank 0: 000000530000000000007f381400166b")

# FC2:
fc2_in = fc1_out[:32]
fc2_out = np.zeros(16, dtype=np.int32)
for j in range(16):
    acc = np.sum(fc2_in * w_fc2_T[:32, j])
    shifted = acc >> s_fc2
    clipped = max(-128, min(127, shifted))
    fc2_out[j] = max(0, clipped)

hex_fc2 = "".join([f"{int(x)&0xFF:02x}" for x in reversed(fc2_out[:16])])
print(f"RTL model FC2   Row 145 Bank 0: {hex_fc2}")
print(f"XSim run  FC2   Row 145 Bank 0: 1b462100000b00457f4a2f0009590066")

# FC3:
fc3_in = fc2_out[:16]
fc3_out = np.zeros(10, dtype=np.int32)
for j in range(10):
    acc = np.sum(fc3_in * w_fc3_T[:16, j])
    shifted = acc >> s_fc3
    clipped = max(-128, min(127, shifted))
    fc3_out[j] = clipped

hex_fc3 = "".join([f"{int(x)&0xFF:02x}" for x in reversed(fc3_out[:10])])
print(f"RTL model FC3   Row 170 Bank 0: {hex_fc3}")
print(f"XSim run  FC3   Row 170 Bank 0: 000000000000f1b81fd3f3dfd9e2e9ff")
print("RTL model Logits:", fc3_out.tolist())

