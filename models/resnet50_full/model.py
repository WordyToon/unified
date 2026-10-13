"""
================================================================================
Full ResNet-50 Network Architecture: Standardized Model & Hardware Config
================================================================================
Defines the complete 50-layer ResNet-50 deep neural network in the standardized
XModel format for the c_engine Systolic Accelerator (1024 PEs, R=16, C=64).
Contains all 34 Bundles (b1..b34) and 20 Skip Branches (sk1..sk20).
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

NB_CLASSES     = 1000    # Full 1,000 ImageNet synset classes

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

        self.b1 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=64, kernel_size=7, strides=2,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)),
            pool=XPool(
                type='max', pool_size=3, strides=2, padding='same',
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),)
            )

        self.sk1 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=256, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None, slope=0)),
            )

        self.b2 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=64, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0.125))
        )

        self.b3 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=64, kernel_size=3, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0))
        )

        self.sk2 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=256, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            add_act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
        )

        self.b4 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=64, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.b5 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=64, kernel_size=3, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.sk3 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=256, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            add_act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
        )

        self.b6 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=64, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.b7 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=64, kernel_size=3, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.sk4 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=256, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            add_act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
        )

        self.sk5 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=512, kernel_size=1, strides=2,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),)
        )

        self.b8 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=128, kernel_size=1, strides=2,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.b9 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=128, kernel_size=3, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.sk6 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=512, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            add_act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
        )

        self.b10 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=128, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.b11 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=128, kernel_size=3, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.sk7 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=512, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            add_act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
        )

        self.b12 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=128, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.b13 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=128, kernel_size=3, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.sk8 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=512, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            add_act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
        )

        self.b14 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=128, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.b15 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=128, kernel_size=3, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.sk9 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=512, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            add_act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
        )

        self.sk10 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=1024, kernel_size=1, strides=2,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),)
        )

        self.b16 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=256, kernel_size=1, strides=2,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.b17 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=256, kernel_size=3, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.sk11 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=1024, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            add_act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
        )

        self.b18 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=256, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.b19 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=256, kernel_size=3, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.sk12 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=1024, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            add_act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
        )

        self.b20 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=256, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.b21 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=256, kernel_size=3, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.sk13 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=1024, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            add_act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
        )

        self.b22 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=256, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.b23 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=256, kernel_size=3, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.sk14 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=1024, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            add_act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
        )

        self.b24 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=256, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.b25 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=256, kernel_size=3, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.sk15 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=1024, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            add_act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
        )

        self.b26 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=256, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.b27 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=256, kernel_size=3, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.sk16 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=1024, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            add_act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
        )

        self.sk17 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=2048, kernel_size=1, strides=2,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
        )

        self.b28 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=512, kernel_size=1, strides=2,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.b29 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=512, kernel_size=3, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.sk18 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=2048, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            add_act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
        )

        self.b30 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=512, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.b31 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=512, kernel_size=3, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.sk19 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=2048, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            add_act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)
        )

        self.b32 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=512, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.b33 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=512, kernel_size=3, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),),
        )

        self.sk20 = XBundle(
            core=XConvBN(
                k_int_bits=0, b_int_bits=0, filters=2048, kernel_size=1, strides=1,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            add_act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0),
            pool=XPool(
                type='avg', pool_size=7, strides=7, padding='same',
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            flatten=True
        )

        self.b34 = XBundle(
            core=XDense(
                k_int_bits=0, b_int_bits=0, units=NB_CLASSES, use_bias=False,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None),),
            softmax=True
        )

    def call(self, x):
        x = self.input_quant_layer(x)
        x = self.b1(x) # 0
        x_skip1 = self.sk1(x) # 1
        x = self.b2(x) # 2
        x = self.b3(x) # 3
        x = x_skip2 = self.sk2(x, x_skip1) # 4
        x = self.b4(x) # 5
        x = self.b5(x) # 6
        x = x_skip3 = self.sk3(x, x_skip2) # 7
        x = self.b6(x) # 8
        x = self.b7(x) # 9
        x = x_skip4 = self.sk4(x, x_skip3) # 10
        x_skip5 = self.sk5(x) # 11
        x = self.b8(x) # 12
        x = self.b9(x) # 13
        x = x_skip6 = self.sk6(x, x_skip5) # 14
        x = self.b10(x) # 15
        x = self.b11(x) # 16
        x = x_skip7 = self.sk7(x, x_skip6) # 17
        x = self.b12(x) # 18
        x = self.b13(x) # 19
        x = x_skip8 = self.sk8(x, x_skip7) # 20
        x = self.b14(x) # 21
        x = self.b15(x) # 22
        x = x_skip9 = self.sk9(x, x_skip8) # 23
        x_skip10 = self.sk10(x) # 24
        x = self.b16(x) # 25
        x = self.b17(x) # 26
        x = x_skip11 = self.sk11(x, x_skip10) # 27
        x = self.b18(x) # 28
        x = self.b19(x) # 29
        x = x_skip12 = self.sk12(x, x_skip11) # 30
        x = self.b20(x) # 31
        x = self.b21(x) # 32
        x = x_skip13 = self.sk13(x, x_skip12) # 33
        x = self.b22(x) # 34
        x = self.b23(x) # 35
        x = x_skip14 = self.sk14(x, x_skip13) # 36
        x = self.b24(x) # 37
        x = self.b25(x) # 38
        x = x_skip15 = self.sk15(x, x_skip14) # 39
        x = self.b26(x) # 40
        x = self.b27(x) # 41
        x = self.sk16(x, x_skip15) # 42
        x_skip17 = self.sk17(x) # 43
        x = self.b28(x) # 44
        x = self.b29(x) # 45
        x = x_skip18 = self.sk18(x, x_skip17) # 46
        x = self.b30(x) # 47
        x = self.b31(x) # 48
        x = x_skip19 = self.sk19(x, x_skip18) # 49
        x = self.b32(x) # 50
        x = self.b33(x) # 51
        x = x_skip20 = self.sk20(x, x_skip19) # 52
        x = self.b34(x) # 53
        return x


if __name__ == '__main__':
    model = UserModel()
    model.summary()

