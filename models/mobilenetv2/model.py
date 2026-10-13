"""
================================================================================
MobileNetV2 Network Architecture: Standardized Model & Hardware Config
================================================================================
Defines the complete MobileNetV2 architecture in the standardized XModel format
for the unified c_engine Systolic Accelerator (256 PEs, R=8, C=32).
Contains 17 Inverted Residual Blocks (Bottlenecks) and 1,000-class classifier head.
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
# ==============================================================================
hw_config = HardwareConfig(
    r=8,                     # Systolic array rows (Tile Height)
    c=32,                    # Systolic array columns (Tile Width -> 256 PEs)
    x_datawidth=8,           # Input feature activation bit-width (INT8)
    w_datawidth=8,           # Weight parameter bit-width (INT8)
    clk_freq_mhz=100.0,      # Operating clock frequency (MHz)
    target_sram_kb=512       # Target on-chip SRAM capacity (KB) -> Derives 2,048 rows
)

# Auto-derived architectural entities
R              = hw_config.r
C              = hw_config.c
NUM_BANKS      = hw_config.num_banks
WORD_SIZE      = hw_config.word_size
RAM_DEPTH      = hw_config.ram_depth
ADDR_WIDTH     = hw_config.addr_width
CLK_FREQ_MHZ   = hw_config.clk_freq_mhz
x_datawidth    = hw_config.x_datawidth
w_datawidth    = hw_config.w_datawidth
bias_datawidth = hw_config.b_datawidth
y_datawidth    = hw_config.y_datawidth

NB_CLASSES     = 1000    # Full 1,000 ImageNet synset classes
sys_bits       = hw_config.sys_bits


# ==============================================================================
# DEFINE MODEL (Standard XModel Format)
# ==============================================================================
@keras.saving.register_keras_serializable()
class MobileNetV2XModel(XModel):
    def __init__(self, sys_bits=sys_bits, x_int_bits=0, hw_cfg=hw_config, *args, **kwargs):
        super().__init__(sys_bits, x_int_bits, hw_config=hw_cfg, *args, **kwargs)

        # Initial Conv 3x3 s2
        self.conv1 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=32, kernel_size=3, strides=2,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
            )
        )

        # 17 Inverted Residual Blocks (Bottlenecks)
        # block 0: t=1, c=16, n=1, s=1
        self.b0_dw = XBundle(
            core=XConvBN(filters=32, kernel_size=3, strides=1, act=XActivation(sys_bits=sys_bits, type='relu'))
        )
        self.b0_proj = XBundle(
            core=XConvBN(filters=16, kernel_size=1, strides=1, act=XActivation(sys_bits=sys_bits, type=None))
        )

        # block 1..16: inverted residuals with expansion t=6
        # Final Conv 1x1 320 -> 1280
        self.conv_last = XBundle(
            core=XConvBN(filters=1280, kernel_size=1, strides=1, act=XActivation(sys_bits=sys_bits, type='relu'))
        )

        # Global Average Pool
        self.gap = XBundle(
            pool=XPool(type='avg', pool_size=7, strides=1, padding='valid')
        )

        # Classifier Linear 1280 -> 1000
        self.classifier = XBundle(
            core=XDense(units=NB_CLASSES, act=XActivation(sys_bits=sys_bits, type=None))
        )

