// =============================================================================
// File:    baremetal/src/main.c
// Target:  AMD ZCU106 Baremetal (Cortex-A53, aarch64-none-elf-gcc)
// Purpose: Unified C-Engine Baremetal Application
//          - Reads model selection mailbox from DDR (written by XSDB before ELF)
//          - Runs MNIST / ResNet-50 / MobileNetV2 / EfficientNet-B0 inference
//          - Data preloaded via JTAG Option 1B (no SD card required)
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "c_engine_regs.h"
#include "c_engine_driver.h"

#if defined(__arm__) || defined(__aarch64__)
#include "xil_printf.h"
#include "xil_cache.h"
#include "xparameters.h"
#include "xil_io.h"
#include "sleep.h"
#define PRINT(format, ...) xil_printf(format, ##__VA_ARGS__)
#else
#define PRINT(format, ...) printf(format, ##__VA_ARGS__)
#endif

#ifndef TARGET_PRECISION
#define TARGET_PRECISION        8       // 8 for INT8, 16 for INT16
#endif

#if (TARGET_PRECISION == 16)
#define TARGET_CLOCK_FREQ_HZ    70000000U // 70.0 MHz on AMD ZCU106 for 16-bit
#define CLOCK_FREQ_STR          "70.0"
#else
#define TARGET_CLOCK_FREQ_HZ    75000000U // 75.0 MHz on AMD ZCU106 for 8-bit
#define CLOCK_FREQ_STR          "75.0"
#endif

// Standard ImageNet Class Labels (Representative subset for console printing)
static const char *sample_imagenet_labels[1000] = {
    [0]   = "tench, Tinca tinca",
    [1]   = "goldfish, Carassius auratus",
    [2]   = "great white shark",
    [151] = "Chihuahua",
    [152] = "Japanese spaniel",
    [153] = "Maltese dog",
    [176] = "Saluki, gazelle hound",
    [205] = "flat-coated retriever",
    [207] = "golden retriever",
    [208] = "Labrador retriever",
    [212] = "English setter",
    [213] = "Irish setter, red setter",
    [215] = "Brittany spaniel",
    [219] = "cocker spaniel",
    [263] = "Pembroke, Welsh corgi",
    [281] = "tabby, tabby cat",
    [282] = "tiger cat",
    [285] = "Egyptian cat",
    [404] = "airliner",
    [511] = "convertible",
    [656] = "minivan",
    [751] = "race car, racer",
    [817] = "sports car, sport car",
    [829] = "street sign",
    [852] = "tennis ball",
    [920] = "traffic light, traffic signal",
    [980] = "volcano"
};

static const char *mnist_labels[10] = {
    "Digit 0", "Digit 1", "Digit 2", "Digit 3", "Digit 4",
    "Digit 5", "Digit 6", "Digit 7", "Digit 8", "Digit 9"
};

// Global Accelerator Device Instance
static CEngineDevice g_accel;

// SD Card (FatFs) support is intentionally DISABLED.
// Data (weights, images, microcode) is preloaded to DDR via JTAG/XSDB (Option 1B).
// To enable: add -DUSE_SD_CARD and link with -lxilffs in the build command.

void print_banner(void) {
    PRINT("\r\n");
    PRINT("================================================================================\r\n");
    PRINT("  AMD ZCU106 UNIFIED C-ENGINE BAREMETAL APPLICATION\r\n");
    PRINT("  Platform:        AMD Zynq UltraScale+ ZCU106 (xczu7ev-ffvc1156-2-e)\r\n");
    PRINT("  Processing:      256 Systolic PEs (8 Rows x 32 Cols) @ %s MHz\r\n", CLOCK_FREQ_STR);
    PRINT("  Memory:          %s (3.93 MB on-chip SRAM) + 4 GB PS DDR4\r\n",
          (TARGET_PRECISION == 16) ? "96 UltraRAMs + 224 BRAMs" : "96 UltraRAMs (15,360 rows x 256 B)");
    PRINT("  Precision:       %d-bit Unified Compute Engine\r\n", TARGET_PRECISION);
    PRINT("================================================================================\r\n\r\n");
}

#include "model_microcodes.h"

int run_model_inference(const char *model_name, const uint32_t *uops_table, uint32_t num_uops,
                        uint32_t num_classes, uint32_t num_chunks,
                        uint32_t rd_beats, const char *const *class_names,
                        uintptr_t img_base_addr, uintptr_t wt_base_addr) {
    PRINT("\r\n--------------------------------------------------------------------------------\r\n");
    PRINT("  Executing Model: %s (%u uops, %u classes, %u spatial chunks)\r\n",
          model_name, num_uops, num_classes, num_chunks);
    PRINT("--------------------------------------------------------------------------------\r\n");

    // 1. Reset accelerator
    c_engine_soft_reset(&g_accel);

    // 2. Program sequencer microcode directly from verified compiled table
    PRINT("[*] Programming Sequencer microcode from compiled table (%u uops)...\r\n", num_uops);
    c_engine_program_microcode_raw(&g_accel, uops_table, num_uops * 5);

    // 3. Stream input image from DDR to SRAM Row 0
    PRINT("[*] Streaming input image from DDR (0x%08X) to SRAM Row 0 (%u beats)...\r\n",
          (unsigned int)img_base_addr, rd_beats);
    int status = c_engine_stream_input(&g_accel, img_base_addr, 0x00000000, rd_beats);
    if (status != 0) {
        PRINT("[-] Read DMA failed or timed out! Status: %d\r\n", status);
        return -1;
    }

    // 4. Trigger autonomous systolic inference
    PRINT("[*] Triggering Autonomous Hardware Compute Engine (Weights @ 0x%08X)...\r\n", (unsigned int)wt_base_addr);
    uint64_t cycles = 0;
    status = c_engine_run_inference(&g_accel, wt_base_addr, DDR_ACCEL_LOGITS_BASE, num_chunks, &cycles);
    if (status != 0) {
        uint32_t state = 0, pc = 0, chunk = 0;
        c_engine_get_debug_status(&g_accel, &state, &pc, &chunk);
        PRINT("[-] Hardware execution timed out! FSM State: %u, PC: %u, Chunk: %u\r\n", state, pc, chunk);
        return -2;
    }

    // 5. Calculate timing and metrics (integer arithmetic only - xil_printf has no %f or %llu)
    uint32_t cycles_lo = (uint32_t)(cycles & 0xFFFFFFFFU);
    uint32_t lat_us_int = (uint32_t)(cycles / (TARGET_CLOCK_FREQ_HZ / 1000000U));
    uint32_t lat_us_dec = (uint32_t)((cycles * 100U / (TARGET_CLOCK_FREQ_HZ / 1000000U)) % 100U);
    uint32_t lat_ms_int = lat_us_int / 1000U;
    uint32_t lat_ms_dec = (lat_us_int % 1000U) / 10U;
    uint32_t fps_int    = (lat_us_int > 0) ? (1000000U / lat_us_int) : 0U;
    uint32_t fps_dec    = (lat_us_int > 0) ? ((10000000U / lat_us_int) % 10U) : 0U;

    PRINT("[+] Execution Completed Successfully!\r\n");
    PRINT("  Hardware Cycles:  %u cycles @ %s MHz\r\n", cycles_lo, CLOCK_FREQ_STR);
    if (lat_ms_int == 0) {
        PRINT("  Hardware Latency: %u.%02u us (%u.%02u ms)\r\n",
              lat_us_int, lat_us_dec, lat_ms_int, lat_ms_dec);
    } else {
        PRINT("  Hardware Latency: %u.%02u ms\r\n", lat_ms_int, lat_ms_dec);
    }
    PRINT("  Throughput:       %u.%u FPS\r\n", fps_int, fps_dec);

    // 6. Extract Top-5 predictions
#if defined(__arm__) || defined(__aarch64__)
    Xil_DCacheInvalidateRange((INTPTR)DDR_ACCEL_LOGITS_BASE, (INTPTR)(num_classes * (TARGET_PRECISION == 16 ? 2 : 1) + 64));
#endif
    CEnginePrediction top_k[5];
    int k_count = (num_classes >= 5) ? 5 : num_classes;
    c_engine_compute_topk((const void *)DDR_ACCEL_LOGITS_BASE, TARGET_PRECISION, num_classes,
                          k_count, class_names, top_k);

    PRINT("\r\n  Top Predictions:\r\n");
    for (int r = 0; r < k_count; r++) {
        int conf_int = (int)(top_k[r].confidence * 100.0f);
        int conf_dec = (int)(top_k[r].confidence * 10000.0f) % 100;
        PRINT("    Rank #%d: Class #%d (%s) -> Conf: %d.%02d%% (Logit: %d)\r\n",
              top_k[r].rank,
              top_k[r].class_id,
              top_k[r].label ? top_k[r].label : "Unknown",
              conf_int, conf_dec,
              (int)top_k[r].logit);
    }
    PRINT("--------------------------------------------------------------------------------\r\n");
    return 0;
}

int main(void) {
    // ------------------------------------------------------------------
    // Step 1: IMMEDIATE UART PRINT (Sanity check before anything else)
    // ------------------------------------------------------------------
    PRINT("\r\n================================================================================\r\n");
    PRINT("  [BOOT] ZCU106 Baremetal ELF started on Cortex-A53 #0! Precision=%d\r\n", TARGET_PRECISION);
    PRINT("================================================================================\r\n");

    // ------------------------------------------------------------------
    // Step 3: Print banner
    // ------------------------------------------------------------------
    print_banner();

    // ------------------------------------------------------------------
    // Step 4: Initialize Unified C-Engine Driver
    // ------------------------------------------------------------------
    PRINT("[*] Initializing Unified C-Engine Driver at 0x%08X...\r\n", C_ENGINE_DEFAULT_BASE_ADDR);
    if (c_engine_init(&g_accel, C_ENGINE_DEFAULT_BASE_ADDR, TARGET_PRECISION, TARGET_CLOCK_FREQ_HZ) != 0) {
        PRINT("[-] Driver initialization failed!\r\n");
        return -1;
    }
    PRINT("[+] Driver initialized at %s MHz.\r\n", CLOCK_FREQ_STR);

    // ------------------------------------------------------------------
    // Step 5: Read Model Selection Mailbox (written by XSDB before ELF)
    // ------------------------------------------------------------------
    volatile uint32_t model_sel = *(volatile uint32_t *)DDR_ACCEL_MODEL_SEL_BASE;
    PRINT("[*] Target Execution Mode: %s (Mailbox Code: %u)\r\n",
          (model_sel == 1) ? "MNIST LeNet-5 Only" :
          (model_sel == 2) ? "ResNet-50 Full Backbone Only" :
          (model_sel == 3) ? "MobileNetV2 Only" :
          (model_sel == 4) ? "EfficientNet-B0 Only" : "All 4 Vision Models Sequentially",
          (unsigned int)model_sel);

    // =========================================================================
    // 1. MNIST LeNet-5 Evaluation
    // =========================================================================
    if (model_sel == 0 || model_sel == 1) {
        PRINT("\r\n[1/4] Running MNIST LeNet-5 Digit Classification...\r\n");
        uint32_t mnist_wt_beats = (TARGET_PRECISION == 16) ? 8192 : 4096;
        uint32_t mnist_img_beats = (TARGET_PRECISION == 16) ? 128 : 64;
        PRINT("[*] Preloading MNIST weights from DDR (0x%08X) to SRAM Row 0 (%u beats)...\r\n",
              DDR_ACCEL_MNIST_WT_BASE, mnist_wt_beats);
        c_engine_stream_input(&g_accel, DDR_ACCEL_MNIST_WT_BASE, 0x00000000, mnist_wt_beats);
        run_model_inference("MNIST LeNet-5", mnist_uops, NUM_UOPS_MNIST_UOPS, 10, 1, mnist_img_beats, mnist_labels,
                            DDR_ACCEL_MNIST_IMG_BASE, 0x00000000);
    }

    // =========================================================================
    // 2. ResNet-50 Full 50-Layer Architecture Evaluation
    // =========================================================================
    if (model_sel == 0 || model_sel == 2) {
        PRINT("\r\n[2/4] Running Full ResNet-50 Architecture (1,000 ImageNet Classes)...\r\n");
        uint32_t r50_beats = (TARGET_PRECISION == 16) ? 18816 : 9408;
#if (TARGET_PRECISION == 16)
        const uint32_t *r50_uops = resnet50_16bit_uops;
        uint32_t r50_uop_count = NUM_UOPS_RESNET50_16BIT_UOPS;
#else
        const uint32_t *r50_uops = resnet50_8bit_uops;
        uint32_t r50_uop_count = NUM_UOPS_RESNET50_8BIT_UOPS;
#endif
        run_model_inference("ResNet-50 Full Backbone", r50_uops, r50_uop_count, 1000, 196, r50_beats,
                            sample_imagenet_labels, DDR_ACCEL_IMAGENET_IMG_BASE, DDR_ACCEL_R50_WT_BASE);
    }

    // =========================================================================
    // 3. MobileNetV2 Architecture Evaluation
    // =========================================================================
    if (model_sel == 0 || model_sel == 3) {
        PRINT("\r\n[3/4] Running MobileNetV2 Architecture (1,000 ImageNet Classes)...\r\n");
        uint32_t mbv2_beats = (TARGET_PRECISION == 16) ? 18816 : 9408;
#if (TARGET_PRECISION == 16)
        const uint32_t *mbv2_uops = mobilenetv2_16bit_uops;
        uint32_t mbv2_uop_count = NUM_UOPS_MOBILENETV2_16BIT_UOPS;
#else
        const uint32_t *mbv2_uops = mobilenetv2_8bit_uops;
        uint32_t mbv2_uop_count = NUM_UOPS_MOBILENETV2_8BIT_UOPS;
#endif
        run_model_inference("MobileNetV2 Inverted Residual", mbv2_uops, mbv2_uop_count, 1000, 196, mbv2_beats,
                            sample_imagenet_labels, DDR_ACCEL_IMAGENET_IMG_BASE, DDR_ACCEL_MBV2_WT_BASE);
    }

    // =========================================================================
    // 4. EfficientNet-B0 Architecture Evaluation
    // =========================================================================
    if (model_sel == 0 || model_sel == 4) {
        PRINT("\r\n[4/4] Running EfficientNet-B0 Architecture (1,000 ImageNet Classes)...\r\n");
        uint32_t eff_beats = (TARGET_PRECISION == 16) ? 18816 : 9408;
#if (TARGET_PRECISION == 16)
        const uint32_t *eff_uops = efficientnetb0_16bit_uops;
        uint32_t eff_uop_count = NUM_UOPS_EFFICIENTNETB0_16BIT_UOPS;
#else
        const uint32_t *eff_uops = efficientnetb0_8bit_uops;
        uint32_t eff_uop_count = NUM_UOPS_EFFICIENTNETB0_8BIT_UOPS;
#endif
        run_model_inference("EfficientNet-B0 MBConv", eff_uops, eff_uop_count, 1000, 196, eff_beats,
                            sample_imagenet_labels, DDR_ACCEL_IMAGENET_IMG_BASE, DDR_ACCEL_EFF_WT_BASE);
    }

    PRINT("\r\n================================================================================\r\n");
    PRINT("  INFERENCE COMPLETED ON AMD ZCU106 HARDWARE!\r\n");
    PRINT("================================================================================\r\n\r\n");

    // Infinite loop - prevent ELF from returning (avoids undefined behaviour in bare-metal)
    while (1) {}

    return 0;
}
