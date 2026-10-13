"""
================================================================================
MNIST LeNet-5 Architecture: Standardized Model & Hardware Configuration
================================================================================
Defines the LeNet-5 Convolutional Neural Network for Handwritten Digit Recognition
in the standardized XModel format for the c_engine Systolic Accelerator (256 PEs @ 125 MHz).
================================================================================
"""

import os
import sys
import numpy as np

# Add parent directory to path to allow direct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.hw_config import SYS_BITS, HW_CONFIG, HardwareConfig
from core.xmodel import (
    XModel, 
    XBundle, 
    XConvBN, 
    XDense, 
    XPool, 
    XActivation, 
    keras
)

# ==============================================================================
# HARDWARE CONFIGURATION SECTION
# (User specifies ONLY independent design parameters; derived values are auto-computed)
# ==============================================================================
hw_config = HardwareConfig(
    r=8,                     # Systolic array rows (Tile Height)
    c=32,                    # Systolic array columns (Tile Width -> 256 PEs)
    x_datawidth=8,           # Input feature activation bit-width (INT8)
    w_datawidth=8,           # Weight parameter bit-width (INT8)
    clk_freq_mhz=125.0,      # Operating clock frequency (MHz)
    target_sram_kb=256       # Target on-chip SRAM capacity (KB) -> Derives 1,024 rows
)

# Auto-derived architectural entities
R              = hw_config.r
C              = hw_config.c
NUM_BANKS      = hw_config.num_banks           # 16 Banks (calculated from R*C*X / 16)
WORD_SIZE      = hw_config.word_size           # 128 bits per bank (16 bytes)
RAM_DEPTH      = hw_config.ram_depth           # 1,024 Rows (calculated from target_sram_kb)
ADDR_WIDTH     = hw_config.addr_width          # 10 Bits (calculated from ceil(log2(RAM_DEPTH)))
CLK_FREQ_MHZ   = hw_config.clk_freq_mhz
x_datawidth    = hw_config.x_datawidth
w_datawidth    = hw_config.w_datawidth
bias_datawidth = hw_config.b_datawidth
y_datawidth    = hw_config.y_datawidth

NB_CLASSES     = 10      # 10 Digit classes (0..9)

sys_bits = hw_config.sys_bits


# ==============================================================================
# DEFINE MODEL (Standard XModel Format)
# ==============================================================================
'''
Define Model
'''

@keras.saving.register_keras_serializable()
class UserModel(XModel):
    def __init__(self, sys_bits=sys_bits, x_int_bits=0, hw_cfg=hw_config, *args, **kwargs):
        super().__init__(sys_bits, x_int_bits, hw_config=hw_cfg, *args, **kwargs)

        # Stage 1: Conv1 (3x3, 4 filters) + ReLU + MaxPool (2x2, Stride 2)
        self.b1 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=4, kernel_size=3, strides=1, padding='same',
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)),
            pool=XPool(
                type='max', pool_size=2, strides=2, padding='same',
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),)
        )

        # Stage 2: Conv2 (3x3, 1 filter) + ReLU + Flatten (64 features)
        self.b2 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=1, kernel_size=3, strides=1, padding='same',
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)),
            flatten=True
        )

        # Stage 3: Fully-Connected FC1 (32 Neurons) + ReLU
        self.b3 = XBundle(
            core=XDense(
                k_int_bits=0, b_int_bits=0, units=32, use_bias=True,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)),
        )

        # Stage 4: Fully-Connected FC2 (16 Neurons) + ReLU
        self.b4 = XBundle(
            core=XDense(
                k_int_bits=0, b_int_bits=0, units=16, use_bias=True,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)),
        )

        # Stage 5: Classification Head FC3 (10 Class Logits: Digits 0..9) + Softmax
        self.b5 = XBundle(
            core=XDense(
                k_int_bits=0, b_int_bits=0, units=NB_CLASSES, use_bias=True,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            softmax=True
        )

    def call(self, x):
        x = self.input_quant_layer(x)
        x = self.b1(x) # Stage 1: Conv1 + MaxPool
        x = self.b2(x) # Stage 2: Conv2 + MaxPool + Flatten
        x = self.b3(x) # Stage 3: FC1 (32 neurons)
        x = self.b4(x) # Stage 4: FC2 (16 neurons)
        x = self.b5(x) # Stage 5: FC3 (10 digit class probabilities)
        return x


if __name__ == '__main__':
    model = UserModel()
    model.summary()
    
    # Test dummy inference on 28x28 grayscale image
    dummy_img = np.random.randint(0, 256, size=(28, 28, 1), dtype=np.uint8)
    probs = model.call(dummy_img)
    print("\nInference Output Probabilities (10 classes):")
    for i, p in enumerate(probs):
        print(f"  Digit '{i}': {p * 100.0:6.2f}%")
    print(f"Predicted Digit Class: {np.argmax(probs)}")
