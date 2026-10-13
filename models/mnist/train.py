#!/usr/bin/env python3
"""
================================================================================
Train MNIST LeNet-5 Model for 256-PE Unified Systolic Accelerator (AMD ZCU106)
================================================================================
Trains a quantized LeNet-5 model on the real MNIST dataset matching the exact
hardware systolic array spatial padding, strides, and memory layouts:
  1. weights.bin (256 rows x 256 bytes) mapped to exact SRAM addresses
  2. Sample test images for all 10 digits (0..9) with verified bit-accurate predictions
  3. sim/mnist_stimulus.vh for cycle-accurate XSim behavioral verification
================================================================================
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
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


class SystolicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # In hardware systolic array: at PE(r, c), accesses are x[r - 2 + v, c - 2 + h] for v,h in 0..2.
        # This matches pad=(2, 0, 2, 0) (left=2, right=0, top=2, bottom=0) with valid convolution.
        x_pad = F.pad(x, (2, 0, 2, 0))
        return F.conv2d(x_pad, self.weight, self.bias, stride=1, padding=0)


class Net8x32(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SystolicConv2d(1, 4, kernel_size=3, bias=True)
        self.conv2 = SystolicConv2d(4, 1, kernel_size=3, bias=True)
        self.fc1 = nn.Linear(64, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)
        self.fc3 = nn.Linear(16, 10, bias=False)

    def forward(self, x):
        # x: (B, 1, 8, 32)
        # Stage 1: Conv1 + ReLU + MaxPool 2x2 -> (B, 4, 4, 16)
        c1 = F.relu(self.conv1(x))
        p1 = F.max_pool2d(c1, 2, 2)
        p1_flat = p1.view(x.size(0), 4, 64)

        # Stage 2: In hardware, each channel of p1 occupies bytes 0..63 of an 8x32 canvas (rows 0 and 1):
        c2_in = torch.zeros(x.size(0), 4, 8, 32, device=x.device)
        c2_in[:, :, :2, :] = p1_flat.view(x.size(0), 4, 2, 32)
        c2 = F.relu(self.conv2(c2_in))

        # Only the first 64 elements of row 35 are read by FC1:
        feat = c2.view(x.size(0), 256)[:, :64]

        # Stage 3: FC1 (64 -> 32) + ReLU
        h1 = F.relu(self.fc1(feat))
        # Stage 4: FC2 (32 -> 16) + ReLU
        h2 = F.relu(self.fc2(h1))
        # Stage 5: FC3 (16 -> 10)
        out = self.fc3(h2)
        return out


def train_and_export():
    print("=" * 80)
    print("  TRAINING MNIST LENET-5 FOR 256-PE UNIFIED SYSTOLIC ACCELERATOR")
    print("=" * 80)

    torch.manual_seed(42)
    np.random.seed(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    unified_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    pynq_mnist_dir = os.path.join(unified_dir, "pynq", "mnist")
    baremetal_data_dir = os.path.join(unified_dir, "baremetal", "data")
    sim_dir = os.path.join(unified_dir, "sim")
    data_dir = os.path.join(unified_dir, "data")
    os.makedirs(pynq_mnist_dir, exist_ok=True)
    os.makedirs(baremetal_data_dir, exist_ok=True)
    os.makedirs(sim_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((8, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = Net8x32()
    model_path = os.path.join(script_dir, "model.pth")
    trained = False
    if os.path.exists(model_path) and "--retrain" not in sys.argv:
        print(f"[*] Found existing trained checkpoint: {model_path}. Loading weights...")
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            trained = True
        except Exception:
            print("[*] Checkpoint mismatch, training fresh hardware-aligned model...")
            model = Net8x32()
            trained = False

    if not trained:
        optimizer = optim.AdamW(model.parameters(), lr=0.008, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        criterion = nn.CrossEntropyLoss()

        print("[*] Training Net8x32 for 8 epochs on MNIST (8x32)...")
        for epoch in range(1, 9):
            model.train()
            total_loss, correct, total = 0, 0, 0
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(data)
                correct += (output.argmax(dim=1) == target).sum().item()
                total += len(data)
            scheduler.step()
            print(f"  Epoch {epoch:02d}/08: Loss = {total_loss/total:.4f}, Train Accuracy = {correct/total*100.0:.2f}%")
        torch.save(model.state_dict(), model_path)
        print(f"[+] Saved model checkpoint to {model_path}")

    model.eval()
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_correct += (output.argmax(dim=1) == target).sum().item()
    test_acc = test_correct / len(test_loader.dataset) * 100.0
    print(f"\n[+] Float32 Test Accuracy on MNIST: {test_acc:.2f}%\n")

    # =========================================================================
    # Calibrate Dynamic Ranges & Compute Quantization Parameters
    # =========================================================================
    print("[*] Calibrating dynamic ranges on 1,000 test images...")
    with torch.no_grad():
        calib_imgs = torch.cat([test_dataset[i][0].unsqueeze(0) for i in range(1000)], dim=0)
        c1_out = model.conv1(calib_imgs)
        p1_out = F.max_pool2d(F.relu(c1_out), 2, 2)
        p1_flat = p1_out.view(1000, 4, 64)
        c2_in = torch.zeros(1000, 4, 8, 32)
        c2_in[:, :, :2, :] = p1_flat.view(1000, 4, 2, 32)
        c2_out = model.conv2(c2_in)
        feat_out = F.relu(c2_out).view(1000, 256)[:, :64]
        h1_out = F.relu(model.fc1(feat_out))
        h2_out = F.relu(model.fc2(h1_out))
        logits_out = model.fc3(h2_out)

        max_x0 = calib_imgs.abs().max().item()
        max_w1 = model.conv1.weight.abs().max().item()
        max_x1 = c1_out.abs().max().item()
        max_w2 = model.conv2.weight.abs().max().item()
        max_x2 = c2_out.abs().max().item()
        max_w_fc1 = model.fc1.weight.abs().max().item()
        max_h1 = h1_out.abs().max().item()
        max_w_fc2 = model.fc2.weight.abs().max().item()
        max_h2 = h2_out.abs().max().item()
        max_w_fc3 = model.fc3.weight.abs().max().item()
        max_out = logits_out.abs().max().item()

    # Scale factors: normalize inputs/outputs into INT8 range [-127, 127]
    s_x0 = 127.0 / max_x0
    s_w1 = 127.0 / max_w1
    s_x1 = 127.0 / max_x1
    shift1 = int(round(np.log2((s_x0 * s_w1) / s_x1)))

    s_w2 = 127.0 / max_w2
    s_x2 = 127.0 / max_x2
    shift2 = int(round(np.log2((s_x1 * s_w2) / s_x2)))

    s_w_fc1 = 127.0 / max_w_fc1
    s_h1 = 127.0 / max_h1
    shift_fc1 = int(round(np.log2((s_x2 * s_w_fc1) / s_h1)))

    s_w_fc2 = 127.0 / max_w_fc2
    s_h2 = 127.0 / max_h2
    shift_fc2 = int(round(np.log2((s_h1 * s_w_fc2) / s_h2)))

    s_w_fc3 = 127.0 / max_w_fc3
    s_out = 64.0 / max_out
    shift_fc3 = int(round(np.log2((s_h2 * s_w_fc3) / s_out)))

    print(f"[*] Calibrated Optimal Shifts: shift1={shift1}, shift2={shift2}, fc1={shift_fc1}, fc2={shift_fc2}, fc3={shift_fc3}")

    # Quantize weights (INT8) and biases (INT16)
    w_c1_int8 = np.clip(np.round(model.conv1.weight.detach().numpy() * s_w1), -127, 127).astype(np.int8)
    b_c1_int16 = np.clip(np.round(model.conv1.bias.detach().numpy() * (s_x0 * s_w1)), -32768, 32767).astype(np.int16)

    w_c2_int8 = np.clip(np.round(model.conv2.weight.detach().numpy() * s_w2), -127, 127).astype(np.int8)
    b_c2_int16 = np.clip(np.round(model.conv2.bias.detach().numpy() * (s_x1 * s_w2)), -32768, 32767).astype(np.int16)

    w_fc1_int8 = np.clip(np.round(model.fc1.weight.detach().numpy() * s_w_fc1), -127, 127).astype(np.int8)
    w_fc2_int8 = np.clip(np.round(model.fc2.weight.detach().numpy() * s_w_fc2), -127, 127).astype(np.int8)
    w_fc3_int8 = np.clip(np.round(model.fc3.weight.detach().numpy() * s_w_fc3), -127, 127).astype(np.int8)

    # =========================================================================
    # Pack Weights into 256-Row x 256-Byte SRAM Matrix
    # =========================================================================
    weights_matrix = np.zeros((256, 256), dtype=np.int8)

    # 1. Conv1 (4 filters, 1 channel, 3x3): 36 weights + 8 bytes bias
    row_c1 = np.zeros(256, dtype=np.int8)
    row_c1[:36] = w_c1_int8.flatten()
    row_c1[36:44] = b_c1_int16.view(np.int8)
    weights_matrix[ADDR_CONV1_WT] = row_c1

    # 2. Conv2 (1 filter, 4 channels, 3x3): 36 weights + 2 bytes bias
    row_c2 = np.zeros(256, dtype=np.int8)
    row_c2[:36] = w_c2_int8.flatten()
    row_c2[36:38] = b_c2_int16.view(np.int8)
    weights_matrix[ADDR_CONV2_WT] = row_c2

    # 3. FC1 (64 inputs x 32 outputs): Row 40 + k stores w.T[k, :32]
    w_fc1_T = w_fc1_int8.T # (64, 32)
    for k in range(64):
        weights_matrix[ADDR_FC1_WT + k, :32] = w_fc1_T[k]

    # 4. FC2 (32 inputs x 16 outputs): Row 110 + k stores w.T[k, :16]
    w_fc2_T = w_fc2_int8.T # (32, 16)
    for k in range(32):
        weights_matrix[ADDR_FC2_WT + k, :16] = w_fc2_T[k]

    # 5. FC3 (16 inputs x 10 outputs): Row 150 + k stores w.T[k, :10]
    w_fc3_T = w_fc3_int8.T # (16, 10)
    for k in range(16):
        weights_matrix[ADDR_FC3_WT + k, :10] = w_fc3_T[k]

    # Save weights.bin
    weights_bytes = weights_matrix.tobytes()
    for p in [os.path.join(script_dir, "weights.bin"),
              os.path.join(pynq_mnist_dir, "weights.bin"),
              os.path.join(baremetal_data_dir, "mnist_weights.bin")]:
        with open(p, "wb") as f:
            f.write(weights_bytes)
    print(f"[+] Saved calibrated weights.bin ({len(weights_bytes):,} bytes)")

    # Save calibration shifts JSON for compiler
    shifts_dict = {
        "shift1": shift1,
        "shift2": shift2,
        "shift_fc1": shift_fc1,
        "shift_fc2": shift_fc2,
        "shift_fc3": shift_fc3,
        "s_x0": float(s_x0)
    }
    with open(os.path.join(script_dir, "calib_shifts.json"), "w") as f:
        json.dump(shifts_dict, f, indent=2)

    # =========================================================================
    # Extract & Verify Sample Test Digits (0..9) with Exact Hardware Emulation
    # =========================================================================
    print("\n[*] Saving test images for digits 0..9 and verifying bit-accurate predictions:")
    samples = {}
    for data, target in test_loader:
        for img, lbl in zip(data, target):
            l = int(lbl.item())
            if l not in samples:
                with torch.no_grad():
                    out = model(img.unsqueeze(0))
                    if out.argmax().item() == l and F.softmax(out, dim=1)[0, l].item() > 0.85:
                        samples[l] = img
            if len(samples) == 10:
                break
        if len(samples) == 10:
            break

    test_imgs_dir = os.path.join(pynq_mnist_dir, "test_images")
    os.makedirs(test_imgs_dir, exist_ok=True)

    for d in range(10):
        img_t = samples[d]
        with torch.no_grad():
            out = model(img_t.unsqueeze(0))
            pred = int(out.argmax(dim=1).item())
            probs = F.softmax(out, dim=1)[0].numpy().tolist()

        # Bit-accurate INT8 hardware simulation for this sample
        x_int = np.clip(np.round(img_t.numpy() * s_x0), -128, 127).astype(np.int32)
        # Conv1:
        c1 = np.zeros((4, 8, 32), dtype=np.int32)
        for oc in range(4):
            wt = w_c1_int8[oc, 0]
            bias = int(b_c1_int16[oc])
            for x in range(8):
                for y in range(32):
                    acc = 0
                    for v in range(3):
                        r = x - 2 + v
                        for h in range(3):
                            c = y - 2 + h
                            val = int(x_int[0, r, c]) if (0 <= r < 8 and 0 <= c < 32) else 0
                            acc += val * int(wt[v, h])
                    acc += bias
                    shifted = acc >> shift1
                    clipped = max(-128, min(127, shifted))
                    c1[oc, x, y] = max(0, clipped)

        # MaxPool 2x2:
        p1 = np.zeros((4, 4, 16), dtype=np.int32)
        for oc in range(4):
            for r in range(4):
                for c in range(16):
                    p1[oc, r, c] = np.max(c1[oc, 2*r:2*r+2, 2*c:2*c+2])

        # Conv2:
        c2 = np.zeros((8, 32), dtype=np.int32)
        for ch in range(4):
            in_grid = np.zeros((8, 32), dtype=np.int32)
            in_grid.flat[:64] = p1[ch].flat[:64]
            wt = w_c2_int8[0, ch]
            for x in range(8):
                for y in range(32):
                    acc = 0
                    for v in range(3):
                        r = x - 2 + v
                        for h in range(3):
                            c = y - 2 + h
                            val = int(in_grid[r, c]) if (0 <= r < 8 and 0 <= c < 32) else 0
                            acc += val * int(wt[v, h])
                    c2[x, y] += acc

        c2_out = np.zeros((8, 32), dtype=np.int32)
        for x in range(8):
            for y in range(32):
                acc = c2[x, y] + int(b_c2_int16[0])
                shifted = acc >> shift2
                clipped = max(-128, min(127, shifted))
                c2_out[x, y] = max(0, clipped)

        # FC1:
        feat = c2_out.flatten()[:64]
        fc1_out = np.zeros(32, dtype=np.int32)
        for j in range(32):
            acc = np.sum(feat * w_fc1_int8[j, :])
            fc1_out[j] = max(0, min(127, (acc >> shift_fc1)))

        # FC2:
        fc2_out = np.zeros(16, dtype=np.int32)
        for j in range(16):
            acc = np.sum(fc1_out * w_fc2_int8[j, :])
            fc2_out[j] = max(0, min(127, (acc >> shift_fc2)))

        # FC3:
        fc3_out = np.zeros(10, dtype=np.int32)
        for j in range(10):
            acc = np.sum(fc2_out * w_fc3_int8[j, :])
            fc3_out[j] = np.clip(acc >> shift_fc3, -128, 127)

        sim_pred = int(np.argmax(fc3_out))
        match = (sim_pred == d)

        # Save quantized image as 256-byte binary (.bin) for DDR loading
        quant_bytes = x_int.squeeze().astype(np.int8).tobytes()
        bin_path = os.path.join(baremetal_data_dir, f"mnist_digit_{d}.bin")
        with open(bin_path, "wb") as f:
            f.write(quant_bytes)
        if d == 0:
            with open(os.path.join(baremetal_data_dir, "mnist_test_image.bin"), "wb") as f:
                f.write(quant_bytes)

        print(f"  Digit {d}: Float={pred} ({probs[pred]*100:.1f}%), INT8_Sim={sim_pred} (Logit={fc3_out[sim_pred]}) -> {'PASS' if match else 'FAIL'}")

    # =========================================================================
    # Write sim/mnist_stimulus.vh
    # =========================================================================
    vh_path = os.path.join(sim_dir, "mnist_stimulus.vh")
    print(f"\n[*] Generating {vh_path} with trained weights...")

    uops_path = os.path.join(script_dir, "microcode.json")
    if os.path.exists(uops_path):
        with open(uops_path, "r") as f:
            uops = json.load(f)
    else:
        uops = []

    with open(vh_path, "w") as f:
        f.write("// =============================================================================\n")
        f.write("// Auto-generated MNIST LeNet-5 Stimulus with Calibrated Trained Weights\n")
        f.write("// Target: tb_unified_c_engine.v (256 PEs @ 100 MHz)\n")
        f.write("// =============================================================================\n\n")

        # 1. Microcode loader
        f.write("task automatic load_mnist_microcode;\nbegin\n")
        for u in uops:
            hex_val = u.get("hex128", f"{u['hex96']:0>32}")
            f.write(f"    u_dut.u_engine.u_config_mem.mem[{u['pc']}] = 128'h{hex_val};\n")
        f.write("end\nendtask\n\n")

        # 2. Image stimulus loader (loads digit 0 into DDR at 0x0)
        f.write("task automatic load_mnist_image_stimulus;\n")
        f.write("    reg [127:0] word_128;\n")
        f.write("    integer beat;\n")
        f.write("begin\n")
        d0_int = np.clip(np.round(samples[0].numpy() * s_x0), -128, 127).astype(np.int8).flatten()
        for beat in range(16):
            beat_bytes = d0_int[beat*16 : (beat+1)*16]
            hex_str = "".join([f"{int(b)&0xFF:02x}" for b in reversed(beat_bytes)])
            f.write(f"    u_ddr.write_word({beat*16}, 128'h{hex_str});\n")
        f.write("end\nendtask\n\n")

        # 3. Weights loader (preloads trained weights into on-chip SRAM)
        f.write("task automatic load_mnist_weights;\n")
        f.write("    reg [127:0] bank_word;\n")
        f.write("    integer bank_idx, row_idx;\n")
        f.write("begin\n")
        active_rows = [ADDR_CONV1_WT, ADDR_CONV2_WT] + \
                      list(range(ADDR_FC1_WT, ADDR_FC1_WT + 64)) + \
                      list(range(ADDR_FC2_WT, ADDR_FC2_WT + 32)) + \
                      list(range(ADDR_FC3_WT, ADDR_FC3_WT + 16))

        for r in active_rows:
            row_bytes = weights_matrix[r]
            for b in range(16):
                bank_bytes = row_bytes[b*16 : (b+1)*16]
                if np.count_nonzero(bank_bytes) > 0:
                    hex_str = "".join([f"{int(x)&0xFF:02x}" for x in reversed(bank_bytes)])
                    f.write(f"    u_dut.u_engine.u_memory.gen_banks[{b}].mem[{r}] = 128'h{hex_str};\n")
        f.write("end\nendtask\n\n")

    print(f"[+] Successfully wrote {vh_path}")
    print("[OK] MNIST LeNet-5 training, calibration, and export complete!\n")


if __name__ == "__main__":
    train_and_export()

