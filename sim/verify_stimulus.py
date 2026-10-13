import re
import json
import torch
import numpy as np
from torchvision import datasets, transforms
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.mnist.train import Net8x32

model = Net8x32()
model.load_state_dict(torch.load("models/mnist/model.pth", map_location="cpu"))
model.eval()

with open("sim/mnist_stimulus.vh", "r") as f:
    lines = [l for l in f if "u_ddr.write_word" in l][:16]

bytes_list = []
for l in lines:
    m = re.search(r"128'h([0-9a-fA-F]+)", l)
    if m:
        b = bytes.fromhex(m.group(1))
        bytes_list.extend(list(reversed(b)))

with open("models/mnist/calib_shifts.json", "r") as f:
    s_x0 = json.load(f)["s_x0"]

arr = np.array(bytes_list, dtype=np.uint8).view(np.int8).reshape(8, 32)
tensor_img = torch.tensor(arr.astype(np.float32) / s_x0).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    out = model(tensor_img)
    probs = torch.softmax(out, dim=1)[0].numpy()

print("=" * 60)
print("Stimulus Image Inference in PyTorch:")
for d in range(10):
    print(f"  Digit {d}: Logit = {out[0, d].item():6.2f} (Prob = {probs[d]*100:5.1f}%)")
print(f"Predicted Class: {out.argmax().item()}")
print("=" * 60)

