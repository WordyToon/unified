#ifndef C_ENGINE_DRIVER_H
#define C_ENGINE_DRIVER_H

#include <stdint.h>
#include <stdbool.h>
#include "c_engine_regs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Microcode Instruction Word (128-bit)
typedef struct {
    uint32_t pc;
    uint32_t data0;
    uint32_t data1;
    uint32_t data2;
    uint32_t data3;
} CEngineUop;

// Top-K Prediction Result
typedef struct {
    int      rank;
    int      class_id;
    float    confidence;
    float    logit;
    char     label[64];
} CEnginePrediction;

// Accelerator Device Instance
typedef struct {
    uintptr_t base_addr;
    uint32_t  precision;       // 8 for INT8, 16 for INT16
    uint32_t  clock_freq_hz;   // 75,000,000 Hz
    uint32_t  timeout_ms;
} CEngineDevice;

// Driver API Functions
int  c_engine_init(CEngineDevice *dev, uintptr_t base_addr, uint32_t precision, uint32_t clock_freq_hz);
void c_engine_soft_reset(CEngineDevice *dev);
int  c_engine_program_microcode(CEngineDevice *dev, const CEngineUop *uops, uint32_t num_uops);
int  c_engine_program_microcode_raw(CEngineDevice *dev, const uint32_t *raw_uops, uint32_t num_words);

int  c_engine_stream_input(CEngineDevice *dev, uintptr_t ddr_phys_addr, uint32_t sram_row, uint32_t beats);
int  c_engine_run_inference(CEngineDevice *dev, uintptr_t wt_ddr_phys_addr, uintptr_t out_ddr_phys_addr,
                            uint32_t num_chunks, uint64_t *cycles_executed);

void c_engine_get_debug_status(CEngineDevice *dev, uint32_t *fsm_state, uint32_t *active_pc, uint32_t *chunk_idx);

int  c_engine_compute_topk(const void *logits_buf, uint32_t precision, uint32_t num_classes,
                           int top_k, const char *const *class_names, CEnginePrediction *results);

#ifdef __cplusplus
}
#endif

#endif // C_ENGINE_DRIVER_H

