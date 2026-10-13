import numpy as np
import json
import torch
from torchvision import datasets, transforms
from models.mnist.train import Net8x32

with open("models/mnist/calib_shifts.json") as f:
    s_x0 = json.load(f)["s_x0"]

transform = transforms.Compose([
    transforms.Resize((8, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
ds = datasets.MNIST("data", train=False, download=True, transform=transform)

# Find first digit 0
for i in range(100):
    if ds[i][1] == 0:
        d0_img = ds[i][0]
        break

d0_int = np.clip(np.round(d0_img.numpy() * s_x0), -128, 127).astype(np.int8)[0]
print("Digit 0 in 8x32 grid (values > 0 shown as '#', <=0 as '.'):")
for r in range(8):
    line = "".join(["#" if d0_int[r, c] > 0 else "." for c in range(32)])
    print(f"Row {r}: {line}")

