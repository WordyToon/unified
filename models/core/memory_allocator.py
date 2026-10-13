#!/usr/bin/env python3
"""
================================================================================
Static SRAM Memory Allocator & DDR Spilling Planner for c_engine Accelerator
================================================================================
Manages multi-bank SRAM (16 banks x 128-bit words = 256 B/row, up to 2048/8192 rows)
and decides compile-time ping-pong allocation and DDR spilling:
  1. Ping-Pong Buffering: If feature map fits in SRAM budget, data stays on-chip
     alternating between PING and PONG regions.
  2. DDR Spilling: When a layer's output exceeds the available SRAM rows,
     the tensor is flushed to DDR via AXI4 write DMA, and reloaded chunk-wise
     via AXI4 read DMA for subsequent consumer layers.
================================================================================
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class MemoryRegion:
    name: str
    start_row: int
    num_rows: int
    description: str

    @property
    def end_row(self) -> int:
        return self.start_row + self.num_rows - 1

    @property
    def byte_capacity(self) -> int:
        return self.num_rows * 256


class StaticMemoryAllocator:
    """
    Deterministic compile-time memory planner for c_engine systolic array.
    """
    def __init__(self, total_sram_rows: int = 2048, row_bytes: int = 256):
        self.total_sram_rows = total_sram_rows
        self.row_bytes = row_bytes

        # Static partitioning of SRAM
        # Rows 0..587:   Input staging area (up to 196 chunks x 3 channels = 588 rows)
        self.input_staging = MemoryRegion("INPUT_STAGING", 0, 588, "Input activation chunks from DDR")

        # Rows 600..1299: Ping Buffer (700 rows = 175 KB)
        self.ping_buffer   = MemoryRegion("PING_BUF", 600, 700, "Intermediate Ping feature map buffer")

        # Rows 1300..1999: Pong Buffer (700 rows = 175 KB)
        self.pong_buffer   = MemoryRegion("PONG_BUF", 1300, 700, "Intermediate Pong feature map buffer")

        # Rows 2000..2003: Output Logits (4 rows = 1000 classes)
        self.logits_region = MemoryRegion("LOGITS", 2000, 4, "Final classification logits (1000 classes)")

        # Rows 2004..2047: Weight Cache (44 rows = 11 KB per active tile)
        self.weight_cache  = MemoryRegion("WEIGHT_CACHE", 2004, 44, "Active stage weight staging area")

        # Dedicated DDR Memory Map
        self.DDR_IMG_BASE      = 0x20000000  # Raw image input
        self.DDR_UCODE_BASE    = 0x21000000  # Sequencer microcode
        self.DDR_MAILBOX_BASE  = 0x21FF0000  # Model selection mailbox
        self.DDR_LOGITS_BASE   = 0x22000000  # Final logits destination
        self.DDR_SPILL_BASE    = 0x24000000  # Feature map spill buffer
        self.DDR_WEIGHTS_BASE  = 0x30000000  # Layer weights and biases

    def plan_layer_memory(self,
                          layer_idx: int,
                          h_in: int, w_in: int, chin: int,
                          h_out: int, w_out: int, chout: int,
                          bytes_per_elem: int = 1,
                          is_first: bool = False,
                          is_last: bool = False,
                          is_gemv: bool = False,
                          cur_ping_pong: int = 0) -> Dict[str, Any]:
        """
        Calculates SRAM row requirements, ping-pong placement, and determines
        whether DDR spilling is required for this layer.
        """
        # Calculate tensor memory footprints in rows
        # Each row stores up to 256 bytes (which equals an 8x32 spatial tile of INT8)
        elements_per_row = self.row_bytes // bytes_per_elem

        # For spatial convolutions, chunk footprint matches 8x32
        chunks_h_in = max(1, (h_in + 7) // 8)
        chunks_w_in = max(1, (w_in + 31) // 32)
        total_chunks_in = chunks_h_in * chunks_w_in

        chunks_h_out = max(1, (h_out + 7) // 8)
        chunks_w_out = max(1, (w_out + 31) // 32)
        total_chunks_out = chunks_h_out * chunks_w_out

        # Out rows if stored entirely on-chip:
        # Standard layout: chout rows per spatial chunk
        out_rows_needed = total_chunks_out * chout if not is_gemv else math.ceil(chout / elements_per_row)

        # Determine if output fits in SRAM ping/pong budget (700 rows)
        max_buf_rows = self.ping_buffer.num_rows

        spill_to_ddr = False
        if out_rows_needed > max_buf_rows and not is_last and not is_gemv:
            spill_to_ddr = True

        # Input base address
        if is_first:
            in_base = self.input_staging.start_row
        else:
            in_base = self.ping_buffer.start_row if cur_ping_pong == 0 else self.pong_buffer.start_row

        # Output base address
        if is_last or is_gemv:
            out_base = self.logits_region.start_row
        elif spill_to_ddr:
            # Spill writes to ping/pong staging and triggers AXI4 write DMA to DDR
            out_base = self.pong_buffer.start_row if cur_ping_pong == 0 else self.ping_buffer.start_row
        else:
            out_base = self.pong_buffer.start_row if cur_ping_pong == 0 else self.ping_buffer.start_row

        next_ping_pong = 1 - cur_ping_pong if not is_first else 1

        return {
            "layer_idx": layer_idx,
            "in_base": in_base,
            "out_base": out_base,
            "wt_base": self.weight_cache.start_row,
            "out_rows_needed": out_rows_needed,
            "spill_to_ddr": spill_to_ddr,
            "num_chunks_out": total_chunks_out,
            "cur_ping_pong": cur_ping_pong,
            "next_ping_pong": next_ping_pong,
            "dma_wr_en": 1 if (is_last or spill_to_ddr) else 0,
            "dma_beats": 16 if is_last else (min(127, out_rows_needed * 16) if spill_to_ddr else 0)
        }

