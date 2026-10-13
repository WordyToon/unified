import numpy as np
import json
import re

with open("models/mnist/calib_shifts.json") as f:
    s_cfg = json.load(f)
s1 = s_cfg["shift1"]

with open("models/mnist/weights.bin", "rb") as f:
    w_mat = np.frombuffer(f.read(), dtype=np.int8).reshape(256, 256)

# Weight layout for Conv1 (Row 10):
# 4 filters, 1 channel, 3x3
w_c1_flat = w_mat[10, :36]
b_c1 = w_mat[10, 36:44].view(np.int16)

# Read image from stimulus
with open("sim/mnist_stimulus.vh") as f:
    lines = [l for l in f if "u_ddr.write_word" in l][:16]
bytes_list = []
for l in lines:
    hex_str = re.search(r"128'h([0-9a-fA-F]+)", l).group(1)
    b = bytes.fromhex(hex_str)
    bytes_list.extend(list(reversed(b)))

# The image is 256 bytes, loaded into Row 0
# PE grid is R=8, C=32. PE(x, y) receives input byte at x*32 + y.
img_pe = np.array(bytes_list, dtype=np.uint8).view(np.int8).reshape(8, 32)
print("Image shape:", img_pe.shape)

# Let's inspect how the RTL computes Conv1 for filter 0 (oc=0, ch=0).
# From c_engine_controller.v:
# reg_kernel_area = 9
# reg_ch_stride = 9
# reg_bias_base = 36
# For oc=0, ch=0:
# conv_weight_addr starts at 0.
# In S_CONV_INIT_TAP:
#   Tap 0: weight_addr 0 (wt[0]), sel0=0, sel1=00 -> mult = img * wt[0] -> HOR <= mult
#   conv_weight_addr <= 1
# S_CONV_STEP_H:
#   h_cnt=1: weight_addr 1 (wt[1]), sel1=01 (mult + hor_left) -> HOR <= img*wt[1] + hor_left
#   conv_weight_addr <= 2
#   h_cnt=2: weight_addr 2 (wt[2]), sel1=01 (mult + hor_left) -> Outf_map <= img*wt[2] + hor_left
# S_CONV_ROW_DONE:
#   v_cnt=0: sel0=1, sel1=00 -> VERT <= Outf_map (Row 0 sum)
#   v_base_addr <= 3
#   conv_weight_addr <= 3
# S_CONV_STEP_V:
#   v_cnt <= 1, h_cnt <= 0
#   weight_addr 3 (wt[3]) -> HOR <= img * wt[3]
#   h_cnt=1: wt[4] -> HOR <= img * wt[4] + hor_left
#   h_cnt=2: wt[5] -> Outf_map <= img * wt[5] + hor_left (Row 1 sum)
# S_CONV_ROW_DONE:
#   v_cnt=1: sel0=1, sel1=10 (+ vert_up) -> VERT <= Outf_map + vert_up (Row 0 sum + Row 1 sum)
#   conv_weight_addr <= 6
# S_CONV_STEP_V:
#   v_cnt <= 2, h_cnt <= 0
#   weight_addr 6 (wt[6]) -> HOR <= img * wt[6]
#   h_cnt=1: wt[7] -> HOR <= img * wt[7] + hor_left
#   h_cnt=2: wt[8] -> Outf_map <= img * wt[8] + hor_left (Row 2 sum)
# S_CONV_ROW_DONE:
#   v_cnt=2: sel0=1, sel1=10 (+ vert_up) -> Outf_map <= Outf_map (Row 2 sum) + vert_up (Row 0+1 sum)
# S_CONV_FINAL_V -> S_CONV_ACCUM_ENG -> S_CONV_BIAS:
#   Outf_map <= Outf_map + bias[0]
# Requantization:
#   val_shifted = Outf_map >>> s1
#   val_clipped = clamp(val_shifted, -128, 127)
#   val_act = ReLU(val_clipped) (since act_type = 01)
# Pooling (2x2, stride 2):
#   out[k] = max(val_act[2r, 2c], val_act[2r, 2c+1], val_act[2r+1, 2c], val_act[2r+1, 2c+1])

# Now let's calculate the exact value at PE(x, y):
# What is Row v sum at PE(x, y)?
# At y=0:
#   h=0 (wt[3v+0]): HOR = img[x, 0] * wt[3v+0]
#   h=1 (wt[3v+1]): hor_left=0. HOR = img[x, 0] * wt[3v+1] + 0
#   h=2 (wt[3v+2]): hor_left=0. Outf_map = img[x, 0] * wt[3v+2] + 0
# At y=1:
#   h=0: HOR = img[x, 1] * wt[3v+0]
#   h=1: receives hor_left from y=0 (which was img[x, 0] * wt[3v+0])
#        HOR = img[x, 1] * wt[3v+1] + img[x, 0] * wt[3v+0]
#   h=2: receives hor_left from y=0 (which was img[x, 0] * wt[3v+1])
#        Outf_map = img[x, 1] * wt[3v+2] + img[x, 0] * wt[3v+1]
# At y >= 2:
#   h=0: HOR = img[x, y] * wt[3v+0]
#   h=1: receives hor_left from y-1 (which was img[x, y-1] * wt[3v+0])
#        HOR = img[x, y] * wt[3v+1] + img[x, y-1] * wt[3v+0]
#   h=2: receives hor_left from y-1 (which was img[x, y-1] * wt[3v+1] + img[x, y-2] * wt[3v+0])
#        Outf_map = img[x, y] * wt[3v+2] + img[x, y-1] * wt[3v+1] + img[x, y-2] * wt[3v+0]!

# And similarly for vertical:
# VERT at row x receives from row x-1!
# For x >= 2:
#   Total sum at PE(x, y) =
#     Row 0 sum at (x-2, y) + Row 1 sum at (x-1, y) + Row 2 sum at (x, y)!
# WHERE Row v sum at (r, y) =
#     img[r, y-2] * wt[3v+0] + img[r, y-1] * wt[3v+1] + img[r, y] * wt[3v+2]!

# Therefore, at PE(x, y):
# sum = sum_{v=0..2} sum_{h=0..2} img[x - 2 + v, y - 2 + h] * wt[3v + h]!
# WHERE any out-of-bounds access (negative row or negative col) IS ZERO!

def sim_rtl_conv1(wt, bias, shift):
    out = np.zeros((8, 32), dtype=np.int32)
    for x in range(8):
        for y in range(32):
            acc = 0
            for v in range(3):
                r = x - 2 + v
                for h in range(3):
                    c = y - 2 + h
                    val = int(img_pe[r, c]) if (0 <= r < 8 and 0 <= c < 32) else 0
                    acc += val * int(wt[v * 3 + h])
            acc += int(bias)
            shifted = acc >> shift
            clipped = max(-128, min(127, shifted))
            relu = max(0, clipped)
            out[x, y] = relu
    
    # 2x2 max pool
    pool = np.zeros((4, 16), dtype=np.int32)
    for r in range(4):
        for c in range(16):
            pool[r, c] = np.max(out[2*r:2*r+2, 2*c:2*c+2])
    return pool

wt0 = w_c1_flat[:9]
b0 = b_c1[0]
pool_rtl = sim_rtl_conv1(wt0, b0, s1)
hex_rtl = "".join([f"{int(x)&0xFF:02x}" for x in reversed(pool_rtl[0, :16])])
print("Simulated RTL Conv1 Row 15 Bank 0:", hex_rtl)
print("Actual RTL output from XSim:      07070707070b2226170707070707070b")
