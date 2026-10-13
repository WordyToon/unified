import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# Let's define the exact hardware systolic convolution module in PyTorch!
# For a batch of images x of shape (B, C_in, 8, 32):
# In RTL:
# PE(r, c) computes:
# sum_{ch=0..C_in-1} sum_{v=0..kh-1} sum_{h=0..kw-1} x[b, ch, r - (kh-1) + v, c - (kw-1) + h] * wt[oc, ch, v, h] + bias[oc]
# where any out-of-bounds indices are 0!

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
        # x is (B, C_in, 8, 32)
        # In hardware: at PE(r, c), accesses are x[r - 2 + v, c - 2 + h] for v,h in 0..2.
        # This is equivalent to padding x with:
        # pad_top = 2, pad_bottom = 0, pad_left = 2, pad_right = 0!
        # With valid convolution of kernel 3x3:
        # Padded height = 8 + 2 = 10 -> 10 - 3 + 1 = 8 output rows!
        # Padded width = 32 + 2 = 34 -> 34 - 3 + 1 = 32 output cols!
        # EXACTLY 8x32 output!
        # Let's verify:
        # At output row 0: kernel rows 0..2 cover padded rows 0..2.
        # Padded row 0 is pad 0 (top-2), row 1 is pad 0 (top-1), row 2 is x[0]!
        # Which matches: r=0 -> r-2 is pad, r-1 is pad, r-0 is x[0]!
        # At output row 2: kernel rows 0..2 cover padded rows 2..4 -> x[0], x[1], x[2]!
        # Which matches: r=2 -> r-2=x[0], r-1=x[1], r=x[2]!
        # IT MATCHES 100% BIT-FOR-BIT!
        x_pad = F.pad(x, (2, 0, 2, 0)) # (pad_left=2, pad_right=0, pad_top=2, pad_bottom=0)
        return F.conv2d(x_pad, self.weight, self.bias, stride=1, padding=0)

# What about 2x2 MaxPool?
# In hardware requantization_block:
# out[k] for k in 0..63:
# max over (2r, 2c), (2r, 2c+1), (2r+1, 2c), (2r+1, 2c+1).
# Output shape is (B, 4, 4, 16)!
# Then flattened into 64 elements!
# In SRAM row, these 64 elements occupy bytes 0..63.
# When fed to Conv2:
# In hardware, the 64 elements are placed in bytes 0..63 of an 8x32 PE canvas!
# Bytes 0..31 are row 0 of the canvas.
# Bytes 32..63 are row 1 of the canvas.
# Bytes 64..255 are rows 2..7 (all zeros)!
# So for Conv2, input is an 8x32 canvas where only rows 0 and 1 have data!
class HardwareModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SystolicConv2d(1, 4, kernel_size=3, bias=True)
        self.conv2 = SystolicConv2d(4, 1, kernel_size=3, bias=True)
        self.fc1 = nn.Linear(64, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)
        self.fc3 = nn.Linear(16, 10, bias=False)

    def forward(self, x):
        # Stage 1: Conv1 + ReLU + MaxPool (2x2 s2)
        c1 = F.relu(self.conv1(x)) # (B, 4, 8, 32)
        p1 = F.max_pool2d(c1, 2, 2) # (B, 4, 4, 16)
        
        # Flatten p1 per channel: (B, 4, 64)
        p1_flat = p1.view(x.size(0), 4, 64)
        
        # In hardware, each channel of p1 occupies bytes 0..63 of an 8x32 canvas (rows 0 and 1 of 32 elements):
        c2_in = torch.zeros(x.size(0), 4, 8, 32, device=x.device)
        c2_in[:, :, :2, :] = p1_flat.view(x.size(0), 4, 2, 32)
        
        # Stage 2: Conv2 + ReLU
        c2 = F.relu(self.conv2(c2_in)) # (B, 1, 8, 32)
        # Only the first 64 elements of row 35 are read by FC1!
        feat = c2.view(x.size(0), 256)[:, :64] # (B, 64)
        
        # Stage 3: FC1
        h1 = F.relu(self.fc1(feat)) # (B, 32)
        
        # Stage 4: FC2
        h2 = F.relu(self.fc2(h1)) # (B, 16)
        
        # Stage 5: FC3
        out = self.fc3(h2) # (B, 10)
        return out

print("HardwareModel defined successfully!")

transform = transforms.Compose([
    transforms.Resize((8, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST("data", train=True, download=False, transform=transform)
test_dataset = datasets.MNIST("data", train=False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

torch.manual_seed(42)
model = HardwareModel()
optimizer = optim.AdamW(model.parameters(), lr=0.008, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

print("Training HardwareModel for 5 epochs...")
for epoch in range(1, 6):
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
    print(f"Epoch {epoch}: Train Acc = {correct/total*100:.2f}%")

model.eval()
test_correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_correct += (output.argmax(dim=1) == target).sum().item()
print(f"Test Accuracy: {test_correct/len(test_loader.dataset)*100:.2f}%")

# Test digit 0
for data, target in test_loader:
    for img, lbl in zip(data, target):
        if lbl.item() == 0:
            out = model(img.unsqueeze(0))
            pred = out.argmax().item()
            print(f"Digit 0 Prediction: {pred} (Logits: {out[0].tolist()})")
            break
    break

