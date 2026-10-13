"""
================================================================================
Hardware Configuration & Automated Derivation Engine for c_engine Accelerator
================================================================================
Defines parametrized hardware specifications:
  - User specifies ONLY independent design parameters (r, c, x_datawidth, w_datawidth, clk_freq_mhz, target_sram_kb)
  - All dependent architectural entities are automatically derived via exact mathematical formulas:
      * total_pes = r * c
      * memory_row_bytes = r * c * (x_datawidth / 8)
      * num_banks = memory_row_bytes / (word_size / 8)
      * ram_depth = (target_sram_kb * 1024) / memory_row_bytes
      * addr_width = ceil(log2(ram_depth))
      * peak_throughput_gops = 2 * total_pes * clk_freq_mhz / 1000
================================================================================
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SYS_BITS:
    """
    Standard System Bit-Width Specification for Quantization & Arithmetic.
    Compatible with the standard model format: SYS_BITS(x=8, k=8, b=16).
    """
    x: int = 8          # Input feature map activation bit-width (INT8)
    k: int = 8          # Weight parameter bit-width (INT8)
    b: int = 16         # Bias bit-width (INT16)
    y: int = 32         # Accumulator bit-width (INT32)
    w: Optional[int] = None # Alias for k

    def __post_init__(self):
        if self.w is not None:
            self.k = self.w
        else:
            self.w = self.k

    @property
    def x_datawidth(self) -> int:
        return self.x

    @property
    def w_datawidth(self) -> int:
        return self.k

    @property
    def bias_datawidth(self) -> int:
        return self.b

    @property
    def y_datawidth(self) -> int:
        return self.y


@dataclass
class HardwareConfig:
    """
    Parametrized Hardware Architecture Configuration for c_engine Systolic Array.
    User specifies ONLY independent design knobs. All dependent parameters are
    computed automatically by the derivation engine.
    """
    # 1. Independent User Knobs
    r: int = 8                    # Array rows (Tile Height)
    c: int = 32                   # Array columns (Tile Width)
    x_datawidth: int = 8          # Input activation bit-width (INT8)
    w_datawidth: int = 8          # Weight parameter bit-width (INT8)
    clk_freq_mhz: float = 100.0   # Target clock frequency in MHz (10.0 ns period)
    target_sram_kb: int = 512     # Target on-chip SRAM capacity (KB) -> Derives 2,048 rows
    word_size: int = 128          # Fixed physical AXI/SRAM bank word size (bits = 16 bytes)
    cfg_depth: int = 256          # Config memory instruction depth (up to 256 stages)
    cfg_width: int = 96           # Enhanced 96-bit microcode instruction word width

    # 2. Automatically Derived Entities (computed in __post_init__)
    num_banks: int = field(init=False)
    memory_row_bytes: int = field(init=False)
    ram_depth: int = field(init=False)
    addr_width: int = field(init=False)
    cfg_addr_width: int = field(init=False)
    total_pes: int = field(init=False)
    peak_throughput_gops: float = field(init=False)
    b_datawidth: int = field(init=False)
    y_datawidth: int = field(init=False)
    sys_bits: Optional[SYS_BITS] = field(init=False, default=None)

    def __post_init__(self):
        # 1. Array compute capacity
        self.total_pes = self.r * self.c

        # 2. Memory row byte width matching tile footprint: R * C * (x_datawidth / 8)
        self.memory_row_bytes = self.r * self.c * max(1, (self.x_datawidth // 8))

        # 3. Bank count matching physical 128-bit (16-byte) AXI bank width
        bank_bytes = max(1, self.word_size // 8) # 16 bytes
        self.num_banks = max(1, self.memory_row_bytes // bank_bytes)

        # 4. Total SRAM rows fitting target budget: (target_sram_kb * 1024) / row_bytes
        self.ram_depth = (self.target_sram_kb * 1024) // max(1, self.memory_row_bytes)
        self.addr_width = math.ceil(math.log2(max(2, self.ram_depth)))

        # 5. Microcode address width
        self.cfg_addr_width = math.ceil(math.log2(max(2, self.cfg_depth)))

        # 6. Peak arithmetic throughput
        self.peak_throughput_gops = (2.0 * self.total_pes * self.clk_freq_mhz) / 1000.0

        # 7. Derived arithmetic bit-widths
        self.b_datawidth = 2 * self.x_datawidth  # INT16 for INT8 inputs
        self.y_datawidth = 4 * self.x_datawidth  # INT32 for INT8 inputs
        self.sys_bits = SYS_BITS(x=self.x_datawidth, k=self.w_datawidth, b=self.b_datawidth, y=self.y_datawidth)


# Backward-compatible alias
HW_CONFIG = HardwareConfig

# Standard Multi-Configuration Presets for c_engine AXI4 DMA
CONFIG_R8_C32_8B = HardwareConfig(r=8, c=32, x_datawidth=8, w_datawidth=8, target_sram_kb=256, clk_freq_mhz=125.0)
CONFIG_R16_C64_8B = HardwareConfig(r=16, c=64, x_datawidth=8, w_datawidth=8, target_sram_kb=512, clk_freq_mhz=125.0)
CONFIG_R8_C32_16B = HardwareConfig(r=8, c=32, x_datawidth=16, w_datawidth=16, target_sram_kb=512, clk_freq_mhz=125.0)
CONFIG_R16_C64_16B = HardwareConfig(r=16, c=64, x_datawidth=16, w_datawidth=16, target_sram_kb=1024, clk_freq_mhz=125.0)
