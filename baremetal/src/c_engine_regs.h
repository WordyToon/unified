#ifndef C_ENGINE_REGS_H
#define C_ENGINE_REGS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Unified C-Engine Memory-Mapped Register Map (AXI-Lite Slave @ 0xA000_0000)
// =============================================================================

#define C_ENGINE_DEFAULT_BASE_ADDR      0xA0000000U

// Register Offsets (32-bit word aligned)
#define C_ENGINE_REG_CTRL               0x00U  // Control register (Start/Reset triggers)
#define C_ENGINE_REG_STATUS             0x04U  // Live status register (Busy / Done flags)
#define C_ENGINE_REG_IRQ_EN             0x08U  // Interrupt enable mask
#define C_ENGINE_REG_IRQ_STATUS         0x0CU  // Sticky interrupt status & clear
#define C_ENGINE_REG_RD_DDR             0x10U  // Read DMA DDR source physical address
#define C_ENGINE_REG_RD_SRAM_LOC        0x14U  // Read DMA SRAM destination row address
#define C_ENGINE_REG_RD_BEATS           0x18U  // Read DMA length in 128-bit beats
#define C_ENGINE_REG_WT_DDR             0x1CU  // Weight DDR source physical address
#define C_ENGINE_REG_WR_DDR             0x20U  // Write DMA DDR destination physical address
#define C_ENGINE_REG_WR_SRAM_LOC        0x24U  // Write DMA SRAM source row address
#define C_ENGINE_REG_WR_BEATS           0x28U  // Write DMA length in 128-bit beats
#define C_ENGINE_REG_NUM_CHUNKS         0x2CU  // Dynamic spatial chunk count (196 for ResNet-50)
#define C_ENGINE_REG_CFG_ADDR           0x30U  // Config memory address / PC
#define C_ENGINE_REG_CFG_DATA0          0x34U  // Config microcode Word bits [31:0]
#define C_ENGINE_REG_CFG_DATA1          0x38U  // Config microcode Word bits [63:32]
#define C_ENGINE_REG_CFG_DATA2          0x3CU  // Config microcode Word bits [95:64]
#define C_ENGINE_REG_CFG_DATA3          0x40U  // Config microcode Word bits [127:96]
#define C_ENGINE_REG_MODE_CTRL          0x44U  // Mode control (0: Static regs, 1: Microcode sequencer)
#define C_ENGINE_REG_CYCLES_LO          0x48U  // Performance counter: Cycle count [31:0]
#define C_ENGINE_REG_CYCLES_HI          0x4CU  // Performance counter: Cycle count [63:32]
#define C_ENGINE_REG_DEBUG_STATUS       0x50U  // Debug status: {state[4:0], pc[7:0], chunk[7:0]}

// Control Register Bitmasks (REG_CTRL)
#define CTRL_START_RD_DMA               (1U << 0)  // Pulse 1: Start Read DMA transfer
#define CTRL_START_WR_DMA               (1U << 1)  // Pulse 1: Start Write DMA transfer
#define CTRL_START_ENGINE               (1U << 2)  // Pulse 1: Start Systolic Compute Engine
#define CTRL_SOFT_RESET                 (1U << 3)  // Level 1: Soft reset controller & DMA

// Status Register Bitmasks (REG_STATUS)
#define STATUS_RD_DMA_BUSY              (1U << 0)
#define STATUS_RD_DMA_DONE              (1U << 1)
#define STATUS_WR_DMA_BUSY              (1U << 2)
#define STATUS_WR_DMA_DONE              (1U << 3)
#define STATUS_ENGINE_BUSY              (1U << 4)
#define STATUS_ENGINE_DONE              (1U << 5)

// Interrupt Status Clear Bitmasks (REG_IRQ_STATUS)
#define IRQ_CLEAR_RD_DONE               (1U << 0)
#define IRQ_CLEAR_WR_DONE               (1U << 1)
#define IRQ_CLEAR_ENG_DONE              (1U << 2)
#define IRQ_CLEAR_ALL                   (0x07U)

// Standard DDR Physical Memory Map on AMD ZCU106 (4 GB PS DDR4 Low Memory: 0x0000_0000 - 0x7FFF_FFFF)
// Distinct non-overlapping buffers for concurrent multi-model support:
#define DDR_ACCEL_MNIST_IMG_BASE        0x20000000U  // Buffer for MNIST 32x32 digit image (1 KB 8-bit, 2 KB 16-bit)
#define DDR_ACCEL_IMAGENET_IMG_BASE     0x20100000U  // Buffer for ImageNet 224x224 RGB image (75 KB 8-bit, 150 KB 16-bit)
#define DDR_ACCEL_IMAGE_BASE            DDR_ACCEL_IMAGENET_IMG_BASE // Default ImageNet image buffer
#define DDR_ACCEL_MICROCODE_BASE        0x21000000U  // Buffer for microcode binary
#define DDR_ACCEL_MODEL_SEL_BASE        0x21FF0000U  // Mailbox word: 0=All, 1=MNIST, 2=ResNet-50, 3=MobileNetV2, 4=EfficientNet-B0

#define DDR_ACCEL_MNIST_WT_BASE         0x30000000U  // MNIST LeNet-5 weights (32 KB 8-bit, 64 KB 16-bit)
#define DDR_ACCEL_R50_WT_BASE           0x31000000U  // ResNet-50 full backbone weights (23.5 MB 8-bit, 47.1 MB 16-bit)
#define DDR_ACCEL_MBV2_WT_BASE          0x35000000U  // MobileNetV2 weights (2.27 MB 8-bit, 4.54 MB 16-bit)
#define DDR_ACCEL_EFF_WT_BASE           0x36000000U  // EfficientNet-B0 weights (3.41 MB 8-bit, 6.82 MB 16-bit)
#define DDR_ACCEL_WEIGHTS_BASE          DDR_ACCEL_R50_WT_BASE // Default weights base
#define DDR_ACCEL_LOGITS_BASE           0x38000000U  // 1 MB buffer for accelerator output logits

// Multi-Model Conditioned Feature Tensor Offsets within DDR_ACCEL_IMAGENET_IMG_BASE:
// Offset 0: Raw 196 spatial chunks (150,528 bytes 8-bit, 301,056 bytes 16-bit)
#define DDR_ACCEL_R50_FEAT_OFFSET       150528U  // 960 beats (15,360 B 8-bit, 30,720 B 16-bit) ResNet-50 features
#define DDR_ACCEL_MBV2_FEAT_OFFSET      165888U  // 960 beats (15,360 B 8-bit, 30,720 B 16-bit) MobileNetV2 features
#define DDR_ACCEL_EFF_FEAT_OFFSET       181248U  // 960 beats (15,360 B 8-bit, 30,720 B 16-bit) EfficientNet-B0 features

// GEMV Head Weight Offsets within full weight binaries:
#define DDR_ACCEL_R50_GEMV_WT_OFFSET    23631104U // 61,440 B 8-bit ResNet-50 GEMV head weights (after Stage 53 Projection)
#define DDR_ACCEL_MBV2_GEMV_WT_OFFSET   2206816U  // 61,440 B 8-bit MobileNetV2 GEMV head weights
#define DDR_ACCEL_EFF_GEMV_WT_OFFSET    3350000U  // 61,440 B 8-bit EfficientNet-B0 GEMV head weights

#ifdef __cplusplus
}
#endif

#endif // C_ENGINE_REGS_H

