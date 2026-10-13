#include "c_engine_driver.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

#if defined(__arm__) || defined(__aarch64__)
#include "xil_io.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "sleep.h"
#define REG_READ(addr)          Xil_In32((UINTPTR)(addr))
#define REG_WRITE(addr, val)    Xil_Out32((UINTPTR)(addr), (uint32_t)(val))
#define FLUSH_DCACHE(ptr, size) Xil_DCacheFlushRange((INTPTR)(ptr), (INTPTR)(size))
#define INVAL_DCACHE(ptr, size) Xil_DCacheInvalidateRange((INTPTR)(ptr), (INTPTR)(size))
#define SLEEP_US(us)            usleep(us)
#else
// Host simulation fallback macros
static inline uint32_t simulated_read(uintptr_t addr) { return *(volatile uint32_t *)addr; }
static inline void simulated_write(uintptr_t addr, uint32_t val) { *(volatile uint32_t *)addr = val; }
#define REG_READ(addr)          simulated_read((uintptr_t)(addr))
#define REG_WRITE(addr, val)    simulated_write((uintptr_t)(addr), (uint32_t)(val))
#define FLUSH_DCACHE(ptr, size) do {} while(0)
#define INVAL_DCACHE(ptr, size) do {} while(0)
#define SLEEP_US(us)            do {} while(0)
#endif

int c_engine_init(CEngineDevice *dev, uintptr_t base_addr, uint32_t precision, uint32_t clock_freq_hz) {
    if (!dev) return -1;
    dev->base_addr     = base_addr ? base_addr : C_ENGINE_DEFAULT_BASE_ADDR;
    dev->precision     = (precision == 16) ? 16 : 8;
    dev->clock_freq_hz = clock_freq_hz ? clock_freq_hz : 75000000U;
    dev->timeout_ms    = 10000U;

    c_engine_soft_reset(dev);
    return 0;
}

void c_engine_soft_reset(CEngineDevice *dev) {
    if (!dev) return;
    uintptr_t base = dev->base_addr;

    REG_WRITE(base + C_ENGINE_REG_CTRL, CTRL_SOFT_RESET);
    SLEEP_US(100);
    REG_WRITE(base + C_ENGINE_REG_CTRL, 0);
    REG_WRITE(base + C_ENGINE_REG_IRQ_STATUS, IRQ_CLEAR_ALL);
    REG_WRITE(base + C_ENGINE_REG_NUM_CHUNKS, 1);
}

int c_engine_program_microcode(CEngineDevice *dev, const CEngineUop *uops, uint32_t num_uops) {
    if (!dev || !uops || num_uops == 0) return -1;
    uintptr_t base = dev->base_addr;

    for (uint32_t i = 0; i < num_uops; i++) {
        REG_WRITE(base + C_ENGINE_REG_CFG_ADDR,  uops[i].pc);
        REG_WRITE(base + C_ENGINE_REG_CFG_DATA0, uops[i].data0);
        REG_WRITE(base + C_ENGINE_REG_CFG_DATA1, uops[i].data1);
        REG_WRITE(base + C_ENGINE_REG_CFG_DATA2, uops[i].data2);
        REG_WRITE(base + C_ENGINE_REG_CFG_DATA3, uops[i].data3);
    }
    // Set Sequencer mode
    REG_WRITE(base + C_ENGINE_REG_MODE_CTRL, 0x01);
    return 0;
}

int c_engine_program_microcode_raw(CEngineDevice *dev, const uint32_t *raw_uops, uint32_t num_words) {
    if (!dev || !raw_uops || (num_words % 5 != 0)) return -1;
    uintptr_t base = dev->base_addr;
    uint32_t num_uops = num_words / 5;

    for (uint32_t i = 0; i < num_uops; i++) {
        uint32_t idx = i * 5;
        REG_WRITE(base + C_ENGINE_REG_CFG_ADDR,  raw_uops[idx + 0]);
        REG_WRITE(base + C_ENGINE_REG_CFG_DATA0, raw_uops[idx + 1]);
        REG_WRITE(base + C_ENGINE_REG_CFG_DATA1, raw_uops[idx + 2]);
        REG_WRITE(base + C_ENGINE_REG_CFG_DATA2, raw_uops[idx + 3]);
        REG_WRITE(base + C_ENGINE_REG_CFG_DATA3, raw_uops[idx + 4]);
    }
    REG_WRITE(base + C_ENGINE_REG_MODE_CTRL, 0x01);
    return 0;
}

int c_engine_stream_input(CEngineDevice *dev, uintptr_t ddr_phys_addr, uint32_t sram_row, uint32_t beats) {
    if (!dev || beats == 0) return -1;
    uintptr_t base = dev->base_addr;

    // Ensure memory coherency
    uint32_t bytes = beats * 16; // 128-bit beat = 16 bytes
    FLUSH_DCACHE(ddr_phys_addr, bytes);

    REG_WRITE(base + C_ENGINE_REG_IRQ_STATUS, IRQ_CLEAR_RD_DONE);
    REG_WRITE(base + C_ENGINE_REG_RD_DDR, (uint32_t)ddr_phys_addr);
    REG_WRITE(base + C_ENGINE_REG_RD_SRAM_LOC, sram_row);
    REG_WRITE(base + C_ENGINE_REG_RD_BEATS, beats);
    REG_WRITE(base + C_ENGINE_REG_CTRL, CTRL_START_RD_DMA);

    // Wait for RD DMA completion
    uint32_t timeout = 500000;
    while ((REG_READ(base + C_ENGINE_REG_STATUS) & STATUS_RD_DMA_DONE) == 0) {
        if (--timeout == 0) return -2; // Read DMA timeout
        SLEEP_US(10);
    }

    REG_WRITE(base + C_ENGINE_REG_IRQ_STATUS, IRQ_CLEAR_RD_DONE);
    return 0;
}

int c_engine_run_inference(CEngineDevice *dev, uintptr_t wt_ddr_phys_addr, uintptr_t out_ddr_phys_addr,
                            uint32_t num_chunks, uint64_t *cycles_executed) {
    if (!dev) return -1;
    uintptr_t base = dev->base_addr;

    if (wt_ddr_phys_addr != 0) {
        REG_WRITE(base + C_ENGINE_REG_WT_DDR, (uint32_t)wt_ddr_phys_addr);
    }
    REG_WRITE(base + C_ENGINE_REG_WR_DDR, (uint32_t)out_ddr_phys_addr);
    REG_WRITE(base + C_ENGINE_REG_NUM_CHUNKS, (num_chunks > 0) ? num_chunks : 1);
    REG_WRITE(base + C_ENGINE_REG_MODE_CTRL, 0x01);
    REG_WRITE(base + C_ENGINE_REG_IRQ_STATUS, IRQ_CLEAR_ALL);

    // Invalidate DDR cache for logits buffer BEFORE compute so CPU never holds/flushes stale cached data
    uint32_t logits_bytes = (dev->precision == 16) ? (1000 * 2) : 1000;
    INVAL_DCACHE(out_ddr_phys_addr, logits_bytes + 256);

    // Trigger Autonomous 256-PE Systolic Compute Engine
    REG_WRITE(base + C_ENGINE_REG_CTRL, CTRL_START_ENGINE);

    // Await completion (STATUS_ENGINE_DONE) with generous timeout (~50s)
    uint32_t timeout = 5000000;
    while ((REG_READ(base + C_ENGINE_REG_STATUS) & STATUS_ENGINE_DONE) == 0) {
        if (--timeout == 0) {
            return -3; // Engine execution timeout
        }
        SLEEP_US(10);
    }

    // Read Performance Counters
    uint32_t cyc_lo = REG_READ(base + C_ENGINE_REG_CYCLES_LO);
    uint32_t cyc_hi = REG_READ(base + C_ENGINE_REG_CYCLES_HI);
    if (cycles_executed) {
        *cycles_executed = ((uint64_t)cyc_hi << 32) | cyc_lo;
    }

    REG_WRITE(base + C_ENGINE_REG_IRQ_STATUS, IRQ_CLEAR_ALL);

    // Invalidate DDR cache for logits output buffer AFTER compute so CPU reads fresh DDR data
    INVAL_DCACHE(out_ddr_phys_addr, logits_bytes + 256);

    return 0;
}

void c_engine_get_debug_status(CEngineDevice *dev, uint32_t *fsm_state, uint32_t *active_pc, uint32_t *chunk_idx) {
    if (!dev) return;
    uint32_t dbg = REG_READ(dev->base_addr + C_ENGINE_REG_DEBUG_STATUS);
    if (chunk_idx) *chunk_idx  = dbg & 0xFF;
    if (active_pc) *active_pc  = (dbg >> 8) & 0x3FF;
    if (fsm_state) *fsm_state  = (dbg >> 18) & 0x1F;
}

int c_engine_compute_topk(const void *logits_buf, uint32_t precision, uint32_t num_classes,
                           int top_k, const char *const *class_names, CEnginePrediction *results) {
    if (!logits_buf || num_classes == 0 || top_k <= 0 || !results) return -1;

    // Convert fixed-point logits to float
    float f_logits[1000];
    uint32_t n = (num_classes > 1000) ? 1000 : num_classes;

    float max_l = -1e9f;
    for (uint32_t i = 0; i < n; i++) {
        float val;
        if (precision == 16) {
            val = (float)(((const int16_t *)logits_buf)[i]);
        } else {
            val = (float)(((const int8_t *)logits_buf)[i]);
        }
        f_logits[i] = val;
        if (val > max_l) max_l = val;
    }

    // Softmax probabilities
    float sum_exp = 0.0f;
    float probs[1000];
    for (uint32_t i = 0; i < n; i++) {
        probs[i] = expf(f_logits[i] - max_l);
        sum_exp += probs[i];
    }
    if (sum_exp > 0.0f) {
        for (uint32_t i = 0; i < n; i++) {
            probs[i] /= sum_exp;
        }
    }

    // Top-K selection (partial sort)
    bool used[1000] = {false};
    int k = (top_k > (int)n) ? (int)n : top_k;

    for (int r = 0; r < k; r++) {
        float best_prob = -1.0f;
        int   best_idx  = -1;
        for (uint32_t i = 0; i < n; i++) {
            if (!used[i] && probs[i] > best_prob) {
                best_prob = probs[i];
                best_idx  = i;
            }
        }
        if (best_idx >= 0) {
            used[best_idx] = true;
            results[r].rank = r + 1;
            results[r].class_id = best_idx;
            results[r].confidence = best_prob;
            results[r].logit = f_logits[best_idx];
            if (class_names && class_names[best_idx]) {
                strncpy(results[r].label, class_names[best_idx], sizeof(results[r].label) - 1);
                results[r].label[sizeof(results[r].label) - 1] = '\0';
            } else {
                snprintf(results[r].label, sizeof(results[r].label), "Class %d", best_idx);
            }
        }
    }

    return 0;
}

