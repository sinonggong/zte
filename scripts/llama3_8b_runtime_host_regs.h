#ifndef LLAMA3_8B_RUNTIME_HOST_REGS_H
#define LLAMA3_8B_RUNTIME_HOST_REGS_H

#include <stdint.h>

#define LLAMA3_RT_GENERIC_ABI_VERSION        1u
#define LLAMA3_RT_UOP2_LOCAL_ADDR_LSB         0u
#define LLAMA3_RT_UOP2_LOCAL_ADDR_MSB        20u
#define LLAMA3_RT_UOP2_LOCAL_RSVD_LSB        21u
#define LLAMA3_RT_UOP2_LOCAL_RSVD_MSB        30u
#define LLAMA3_RT_UOP2_LOCAL_VALID           31u
#define LLAMA3_RT_UOP3_DESCRIPTOR_MODE       29u
#define LLAMA3_RT_UOP3_DESCRIPTOR_RSVD_LSB   27u
#define LLAMA3_RT_UOP3_DESCRIPTOR_RSVD_MSB   28u
#define LLAMA3_RT_UOP6_EVENT_ID_LSB           0u
#define LLAMA3_RT_UOP6_EVENT_ID_MSB           4u
#define LLAMA3_RT_UOP6_EVENT_SIGNAL_EN        5u
#define LLAMA3_RT_EVENT_COUNT                32u
#define LLAMA3_RT_PACKAGE_HEADER_BYTES       128u
#define LLAMA3_RT_COMMAND_PAGE_HEADER_BYTES  128u
#define LLAMA3_RT_COMMAND_PAGE_BYTES         4224u
#define LLAMA3_RT_COMMAND_SLOT_BYTES          32u
#define LLAMA3_RT_DESCRIPTOR_RECORD_BYTES    128u
#define LLAMA3_RT_DESCRIPTOR_HEADER_BYTES     32u
#define LLAMA3_RT_PACKAGE_MAGIC              0x50474354u
#define LLAMA3_RT_COMMAND_PAGE_MAGIC         0x47504354u
#define LLAMA3_RT_DESCRIPTOR_MAGIC           0x53444354u

#define LLAMA3_RT_ENG_GEMM                    1u
#define LLAMA3_RT_ENG_DMA                     4u
#define LLAMA3_RT_ENG_CONTROL                 5u
#define LLAMA3_RT_OP_DMA_LOAD_1D             40u
#define LLAMA3_RT_OP_DMA_STORE_1D            41u
#define LLAMA3_RT_OP_DMA_GATHER_2D           42u
#define LLAMA3_RT_OP_DMA_SCATTER_2D          43u
#define LLAMA3_RT_OP_DMA_LAYOUT              44u
#define LLAMA3_RT_OP_MATRIX_LAUNCH           45u
#define LLAMA3_RT_OP_BARRIER                 46u
#define LLAMA3_RT_OP_SIGNAL                  47u
#define LLAMA3_RT_OP_WAIT                    48u
#define LLAMA3_RT_OP_HALT                    49u

#define LLAMA3_RT_DTYPE_INT8                  0u
#define LLAMA3_RT_DTYPE_BF16                  1u
#define LLAMA3_RT_DTYPE_FP16                  2u
#define LLAMA3_RT_DTYPE_INT32                 3u
#define LLAMA3_RT_LAYOUT_ROW_MAJOR            0u
#define LLAMA3_RT_LAYOUT_TILE_MAJOR           1u
#define LLAMA3_RT_LAYOUT_PATCH_GATHER         2u
#define LLAMA3_RT_LAYOUT_TRANSPOSE            3u
#define LLAMA3_RT_ACCUM_FP32                  0u
#define LLAMA3_RT_ACCUM_EXACT                 1u
#define LLAMA3_RT_MATRIX_QUANT_NONE           0u
#define LLAMA3_RT_MATRIX_QUANT_AFFINE         1u
#define LLAMA3_RT_MATRIX_FLAG_PACKED_WEIGHT   1u
#define LLAMA3_RT_MATRIX_FLAG_ACC_ZERO        2u
#define LLAMA3_RT_MATRIX_FLAG_ACC_FINAL       4u
#define LLAMA3_RT_MATRIX_FLAG_MASK            7u

#define LLAMA3_RT_FAULT_NONE                  0u
#define LLAMA3_RT_FAULT_PACKAGE_ALIGNMENT     1u
#define LLAMA3_RT_FAULT_PACKAGE_MAGIC         2u
#define LLAMA3_RT_FAULT_PACKAGE_ABI           3u
#define LLAMA3_RT_FAULT_PACKAGE_HEADER        4u
#define LLAMA3_RT_FAULT_PACKAGE_CRC           5u
#define LLAMA3_RT_FAULT_PAGE_ALIGNMENT        6u
#define LLAMA3_RT_FAULT_PAGE_MAGIC            7u
#define LLAMA3_RT_FAULT_PAGE_HEADER           8u
#define LLAMA3_RT_FAULT_PAGE_CRC              9u
#define LLAMA3_RT_FAULT_UOP_RESERVED          10u
#define LLAMA3_RT_FAULT_PAGE_CHAIN            11u
#define LLAMA3_RT_FAULT_DESCRIPTOR_INDEX      12u
#define LLAMA3_RT_FAULT_DESCRIPTOR_ADDRESS    13u
#define LLAMA3_RT_FAULT_DESCRIPTOR_HEADER     14u
#define LLAMA3_RT_FAULT_DESCRIPTOR_CRC        15u
#define LLAMA3_RT_FAULT_ILLEGAL_INSTRUCTION   16u
#define LLAMA3_RT_FAULT_DEPENDENCY_PROTOCOL   17u
#define LLAMA3_RT_FAULT_UNSUPPORTED_ENGINE    18u
#define LLAMA3_RT_FAULT_DMA_DESCRIPTOR        19u
#define LLAMA3_RT_FAULT_DMA_BOUNDS            20u
#define LLAMA3_RT_FAULT_DMA_AXI_READ          21u
#define LLAMA3_RT_FAULT_DMA_AXI_WRITE         22u
#define LLAMA3_RT_FAULT_DMA_PROTOCOL          23u
#define LLAMA3_RT_FAULT_DMA_WATCHDOG          24u
#define LLAMA3_RT_FAULT_ENGINE                25u
#define LLAMA3_RT_FAULT_COMMAND_WATCHDOG      26u

/*
 * Top-level common register offsets for the Llama-3-8B runtime controller.
 * Offsets are in bytes in the reg_control_block user register window.
 */

#define LLAMA3_RT_REG_TC_CONTROL    0x018
#define LLAMA3_RT_TC_CONTROL_RESET_ENABLE (1u << 0)
#define LLAMA3_RT_TC_CONTROL_RUN          (1u << 2)
/* Static GDDR6 addressing config (top_ctrl.sv:1897,1912-1915).  The host byte
 * offset is the user-register index * 4 (REG_RT_CTRL 36 <-> 0x090,
 * REG_GENERIC_CTRL 146 <-> 0x248).  Both reset to 0.
 *
 * CLAMSHELL picks the within-channel address map: 0 = ACT25 (25-bit, 1 GB per
 * channel, 3-bit pad), 1 = ACT26 (26-bit, 2 GB, 2-bit pad).  The part is
 * Clamshell-x8, but 0 is self-consistent and is what the bit-exact board runs
 * used; changing it moves every composed address, so host address arithmetic
 * must change with it.
 *
 * CHMAP is 2 bits per core, core c in bits [2c+1:2c], selecting one of the
 * quad's four GDDR6 target ids.  0 puts all 16 cores on one channel. */
#define LLAMA3_RT_TC_CONTROL_CLAMSHELL    (1u << 4)
#define LLAMA3_RT_REG_TC_CHMAP_SE   0x000
#define LLAMA3_RT_REG_TC_CHMAP_NE   0x004
#define LLAMA3_RT_REG_TC_CHMAP_NW   0x008
#define LLAMA3_RT_REG_TC_CHMAP_SW   0x00C
/* Round-robin 16 cores over the quad's 4 target ids: core c -> channel c & 3. */
#define LLAMA3_RT_TC_CHMAP_SPREAD16 0xE4E4E4E4u

#define LLAMA3_RT_REG_CTRL         0x090
#define LLAMA3_RT_REG_DESC0        0x094
#define LLAMA3_RT_REG_DESC1        0x098
#define LLAMA3_RT_REG_DESC2        0x09C
#define LLAMA3_RT_REG_STATUS       0x0A0
#define LLAMA3_RT_REG_DEBUG0       0x0A4
#define LLAMA3_RT_REG_DEBUG1       0x0A8
#define LLAMA3_RT_REG_DEBUG2       0x0AC
#define LLAMA3_RT_REG_DEBUG3       0x0B0

#define LLAMA3_RT_REG_CUR_UOP0     0x0B4
#define LLAMA3_RT_REG_CUR_UOP1     0x0B8
#define LLAMA3_RT_REG_CUR_UOP2     0x0BC
#define LLAMA3_RT_REG_CUR_UOP3     0x0C0
#define LLAMA3_RT_REG_CUR_UOP4     0x0C4
#define LLAMA3_RT_REG_CUR_UOP5     0x0C8
#define LLAMA3_RT_REG_CUR_UOP6     0x0CC

#define LLAMA3_RT_REG_DISP0        0x0D0
#define LLAMA3_RT_REG_DISP1        0x0D4
#define LLAMA3_RT_REG_DISP2        0x0D8
#define LLAMA3_RT_REG_DISP3        0x0DC

#define LLAMA3_RT_REG_SRC0_ADDR_LO 0x0E0
#define LLAMA3_RT_REG_SRC0_ADDR_HI 0x0E4
#define LLAMA3_RT_REG_SRC1_ADDR_LO 0x0E8
#define LLAMA3_RT_REG_SRC1_ADDR_HI 0x0EC
#define LLAMA3_RT_REG_DST_ADDR_LO  0x0F0
#define LLAMA3_RT_REG_DST_ADDR_HI  0x0F4
#define LLAMA3_RT_REG_WGT_ADDR_LO  0x0F8
#define LLAMA3_RT_REG_WGT_ADDR_HI  0x0FC

#define LLAMA3_RT_REG_TBL_CTRL     0x100
#define LLAMA3_RT_REG_TBL_INDEX    0x104
#define LLAMA3_RT_REG_TBL_WDATA0   0x108
#define LLAMA3_RT_REG_TBL_WDATA1   0x10C
#define LLAMA3_RT_REG_TBL_RDATA0   0x110
#define LLAMA3_RT_REG_TBL_RDATA1   0x114
#define LLAMA3_RT_REG_BUFCFG_WDATA 0x118
#define LLAMA3_RT_REG_BUFCFG_RDATA 0x11C

#define LLAMA3_RT_REG_UOP_CTRL     0x120
#define LLAMA3_RT_REG_UOP_INDEX    0x124
#define LLAMA3_RT_REG_UOP_WDATA0   0x128
#define LLAMA3_RT_REG_UOP_WDATA1   0x12C
#define LLAMA3_RT_REG_UOP_WDATA2   0x130
#define LLAMA3_RT_REG_UOP_WDATA3   0x134
#define LLAMA3_RT_REG_UOP_WDATA4   0x138
#define LLAMA3_RT_REG_UOP_WDATA5   0x13C
#define LLAMA3_RT_REG_UOP_WDATA6   0x140
#define LLAMA3_RT_REG_UOP_RDATA0   0x144
#define LLAMA3_RT_REG_UOP_RDATA1   0x148
#define LLAMA3_RT_REG_UOP_RDATA2   0x14C
#define LLAMA3_RT_REG_UOP_RDATA3   0x150
#define LLAMA3_RT_REG_UOP_RDATA4   0x154
#define LLAMA3_RT_REG_UOP_RDATA5   0x158
#define LLAMA3_RT_REG_UOP_RDATA6   0x15C

/* Host-driven broadcast onto tc_quad.i_mem_*; used for INT8 quant metadata. */
#define LLAMA3_RT_REG_TC_MEM_CTRL   0x21C
#define LLAMA3_RT_REG_TC_MEM_ADDR   0x220
#define LLAMA3_RT_REG_TC_MEM_WDATA0 0x224
#define LLAMA3_RT_REG_TC_MEM_WDATA1 0x228
#define LLAMA3_RT_REG_TC_MEM_WDATA2 0x22C
#define LLAMA3_RT_REG_TC_MEM_WDATA3 0x230
#define LLAMA3_RT_REG_TC_MEM_RDATA0 0x234
#define LLAMA3_RT_REG_TC_MEM_RDATA1 0x238
#define LLAMA3_RT_REG_TC_MEM_RDATA2 0x23C
#define LLAMA3_RT_REG_TC_MEM_RDATA3 0x240
#define LLAMA3_RT_REG_TC_MEM_MODE   0x244
#define LLAMA3_RT_TC_MEM_READ       (1u << 0)
#define LLAMA3_RT_TC_MEM_WRITE      (1u << 1)
#define LLAMA3_RT_TC_MEM_MODE_INT8_DIRECT (1u << 0)

/* Autonomous generic command runtime, user registers 146..163. */
#define LLAMA3_RT_REG_GENERIC_CTRL          0x248
#define LLAMA3_RT_REG_PACKAGE_BASE_LO       0x24C
#define LLAMA3_RT_REG_PACKAGE_BASE_HI       0x250
#define LLAMA3_RT_REG_GENERIC_STATUS        0x254
#define LLAMA3_RT_REG_GENERIC_FAULT_CODE    0x258
#define LLAMA3_RT_REG_GENERIC_FAULT_PC      0x25C
#define LLAMA3_RT_REG_GENERIC_FAULT_ENGINE  0x260
#define LLAMA3_RT_REG_GENERIC_FAULT_DESC    0x264
#define LLAMA3_RT_REG_GENERIC_FAULT_ADDR_LO 0x268
#define LLAMA3_RT_REG_GENERIC_FAULT_ADDR_HI 0x26C
#define LLAMA3_RT_REG_GENERIC_COMMANDS_LO   0x270
#define LLAMA3_RT_REG_GENERIC_COMMANDS_HI   0x274
#define LLAMA3_RT_REG_GENERIC_DMA_READ_LO   0x278
#define LLAMA3_RT_REG_GENERIC_DMA_READ_HI   0x27C
#define LLAMA3_RT_REG_GENERIC_DMA_WRITE_LO  0x280
#define LLAMA3_RT_REG_GENERIC_DMA_WRITE_HI  0x284
#define LLAMA3_RT_REG_GENERIC_ABI_STATUS    0x288
#define LLAMA3_RT_REG_GENERIC_DEBUG         0x28C

#define LLAMA3_RT_GENERIC_CTRL_START        (1u << 0)
#define LLAMA3_RT_GENERIC_CTRL_CLEAR        (1u << 1)
#define LLAMA3_RT_GENERIC_STATUS_BUSY       (1u << 0)
#define LLAMA3_RT_GENERIC_STATUS_DONE       (1u << 1)
#define LLAMA3_RT_GENERIC_STATUS_FAULT      (1u << 2)
#define LLAMA3_RT_GENERIC_STATUS_RESULT     (1u << 3)
#define LLAMA3_RT_GENERIC_PACKAGE_HI_MASK   0x000003FFu
#define LLAMA3_RT_GENERIC_ABI_STATUS_VALUE  0x10808001u

#define LLAMA3_RT_TC_INT8_TILE_REGION      0x00010000u
#define LLAMA3_RT_TC_INT8_TILE_KIND_ACT    0u
#define LLAMA3_RT_TC_INT8_TILE_KIND_WGT    1u
#define LLAMA3_RT_TC_INT8_TILE_KIND_OUT    2u
#define LLAMA3_RT_TC_INT8_TILE_KIND_CTRL   3u
#define LLAMA3_RT_TC_INT8_TILE_BROADCAST   (1u << 20)
#define LLAMA3_RT_TC_INT8_TILE_ADDR(kind, tc_id, word) \
    (LLAMA3_RT_TC_INT8_TILE_REGION | ((((uint32_t)(kind)) & 0x3u) << 14) | \
     ((((uint32_t)(tc_id)) & 0x3fu) << 8) | (((uint32_t)(word)) & 0xffu))

#define LLAMA3_RT_REG_PRECISION_STATUS 0x210
#define LLAMA3_RT_PRECISION_COMPILED_INT8 (1u << 28)

#define LLAMA3_RT_DESC0_WGT_PROFILE_VALID      (1u << 31)
#define LLAMA3_RT_DESC0_WGT_PROFILE_SUFFIX     (1u << 30)
#define LLAMA3_RT_DESC0_WGT_PROFILE_ZERO_ALIAS (1u << 29)
#define LLAMA3_RT_DESC0_WGT_PROFILE_LAYER_MASK 0x3fu
#define LLAMA3_RT_WGT_PROFILE_PREFIX_BASE_INDEX 62u
#define LLAMA3_RT_WGT_PROFILE_SUFFIX_BASE_INDEX 63u

#define LLAMA3_RT_CTRL_START       (1u << 0)
#define LLAMA3_RT_CTRL_CLEAR       (1u << 1)
#define LLAMA3_RT_CTRL_ADVANCE     (1u << 2)
#define LLAMA3_RT_CTRL_ISSUE       (1u << 3)
#define LLAMA3_RT_CTRL_AUTO_ADV    (1u << 4)
#define LLAMA3_RT_CTRL_MODE_DECODE (1u << 5)
#define LLAMA3_RT_CTRL_USE_UOP_RAM (1u << 6)

#define LLAMA3_RT_STATUS_RUNNING   (1u << 0)
#define LLAMA3_RT_STATUS_DONE      (1u << 1)
#define LLAMA3_RT_STATUS_USE_UOP   (1u << 2)
#define LLAMA3_RT_STATUS_AUTO_ADV  (1u << 3)
#define LLAMA3_RT_STATUS_DECODE    (1u << 4)
#define LLAMA3_RT_STATUS_CUR_VALID (1u << 5)
#define LLAMA3_RT_STATUS_EXEC_BUSY (1u << 6)
#define LLAMA3_RT_STATUS_EXEC_DONE (1u << 7)

#define LLAMA3_RT_TBL_WR_ADDR      (1u << 0)
#define LLAMA3_RT_TBL_RD_ADDR      (1u << 1)
#define LLAMA3_RT_TBL_WR_BUFCFG    (1u << 2)
#define LLAMA3_RT_TBL_RD_BUFCFG    (1u << 3)

#define LLAMA3_RT_UOP_WR_ENTRY     (1u << 0)
#define LLAMA3_RT_UOP_RD_ENTRY     (1u << 1)

#define LLAMA3_RT_MEM_NONE         0u
#define LLAMA3_RT_MEM_SCRATCH      1u
#define LLAMA3_RT_MEM_GDDR         2u
#define LLAMA3_RT_MEM_STREAM       3u

#define LLAMA3_RT_ENG_NONE         0u
#define LLAMA3_RT_ENG_GEMM         1u
#define LLAMA3_RT_ENG_VECTOR       2u
#define LLAMA3_RT_ENG_REDUCE       3u
#define LLAMA3_RT_ENG_DMA          4u
#define LLAMA3_RT_ENG_END          15u

#define LLAMA3_RT_OP_NONE          0u
#define LLAMA3_RT_OP_RMS_ACCUM     1u
#define LLAMA3_RT_OP_RMS_APPLY     2u
#define LLAMA3_RT_OP_Q_PROJ        3u
#define LLAMA3_RT_OP_K_PROJ        4u
#define LLAMA3_RT_OP_V_PROJ        5u
#define LLAMA3_RT_OP_ROPE_Q        6u
#define LLAMA3_RT_OP_ROPE_K        7u
#define LLAMA3_RT_OP_SCORE_GEMM    8u
#define LLAMA3_RT_OP_SCORE_MASK_ADD 9u
#define LLAMA3_RT_OP_SOFTMAX_MAX   10u
#define LLAMA3_RT_OP_SOFTMAX_SUM   11u
#define LLAMA3_RT_OP_SOFTMAX_EXP   12u
#define LLAMA3_RT_OP_SOFTMAX_APPLY 13u
#define LLAMA3_RT_OP_CTX_GEMM      14u
#define LLAMA3_RT_OP_O_PROJ        15u
#define LLAMA3_RT_OP_RESIDUAL_ADD  16u
#define LLAMA3_RT_OP_GATE_PROJ     17u
#define LLAMA3_RT_OP_UP_PROJ       18u
#define LLAMA3_RT_OP_SWIGLU        19u
#define LLAMA3_RT_OP_DOWN_PROJ     20u
#define LLAMA3_RT_OP_K_CACHE_STORE 21u
#define LLAMA3_RT_OP_V_CACHE_STORE 22u
#define LLAMA3_RT_OP_LINEAR        23u
#define LLAMA3_RT_OP_GELU_MUL      24u
#define LLAMA3_RT_OP_SWISH         25u
#define LLAMA3_RT_OP_TIME_SINCOS   26u
#define LLAMA3_RT_OP_SCALED_ADD    27u
#define LLAMA3_RT_OP_CONCAT_PACK   28u
#define LLAMA3_RT_OP_LAYER_NORM_ACCUM 29u
#define LLAMA3_RT_OP_LAYER_NORM_APPLY 30u
#define LLAMA3_RT_OP_BIAS_ADD      31u
#define LLAMA3_RT_OP_POSEMB_ADD    32u
#define LLAMA3_RT_OP_LAYER_END     255u

#define LLAMA3_RT_BUFCFG(mem_type, bank, base_word, word_count) \
    ((((uint32_t)(mem_type)) & 0x3u) | ((((uint32_t)(bank)) & 0x7u) << 2) | \
     ((((uint32_t)(base_word)) & 0x3FFu) << 5) | ((((uint32_t)(word_count)) & 0x3FFu) << 15))

#define LLAMA3_RT_DESC0(layer_id, seq_len) \
    ((((uint32_t)(layer_id)) & 0x3Fu) | ((((uint32_t)(seq_len)) & 0xFFFFu) << 8))

#define LLAMA3_RT_DESC1(token_pos) \
    (((uint32_t)(token_pos)) & 0xFFFFu)

#define LLAMA3_RT_DESC2(last_phase) \
    ((((uint32_t)(last_phase)) & 0x7Fu) | (1u << 7))

#define LLAMA3_RT_ADDR_TBL_INDEX_BUF(buf_id) \
    (((uint32_t)(buf_id)) & 0x7Fu)

#define LLAMA3_RT_ADDR_TBL_INDEX_WGT(weight_sel) \
    (64u + (((uint32_t)(weight_sel)) & 0x3Fu))

#define LLAMA3_RT_ADDR_LO(addr64) ((uint32_t)((addr64) & 0xffffffffull))
#define LLAMA3_RT_ADDR_HI(addr64) ((uint32_t)(((addr64) >> 32) & 0xffffffffull))

/*
 * UOP RAM entry format:
 *   word0[7:0]   = phase_id
 *   word1[3:0]   = engine
 *   word1[15:8]  = opcode
 *   word2[7:0]   = src0_buf
 *   word2[15:8]  = src1_buf
 *   word2[23:16] = dst_buf
 *   word2[31:24] = weight_sel
 *   word3[15:0]  = flags
 *   word4        = dim_m
 *   word5        = dim_n
 *   word6        = dim_k
 */
#define LLAMA3_RT_UOP_W0(phase_id)                 ((uint32_t)((phase_id) & 0xFFu))
#define LLAMA3_RT_UOP_W1(engine, opcode)           ((((uint32_t)(engine)) & 0xFu) | ((((uint32_t)(opcode)) & 0xFFu) << 8))
#define LLAMA3_RT_UOP_W2(src0, src1, dst, weight)  ((((uint32_t)(src0)) & 0xFFu) | ((((uint32_t)(src1)) & 0xFFu) << 8) | ((((uint32_t)(dst)) & 0xFFu) << 16) | ((((uint32_t)(weight)) & 0xFFu) << 24))
#define LLAMA3_RT_UOP_W3(flags)                    ((uint32_t)((flags) & 0xFFFFu))
#define LLAMA3_RT_UOP_W3_DESCRIPTOR(flags) \
    (LLAMA3_RT_UOP_W3(flags) | (1u << 29))
#define LLAMA3_RT_UOP_W3_VEC(flags, vec_op)        (LLAMA3_RT_UOP_W3(flags) | ((((uint32_t)(vec_op)) & 0xFu) << 16) | (1u << 31))
#define LLAMA3_RT_UOP_W3_VEC_SCALAR(flags, vec_op) (LLAMA3_RT_UOP_W3_VEC((flags), (vec_op)) | (1u << 26))
#define LLAMA3_RT_UOP_W6_EVENT(event_id, signal_enable) \
    ((((uint32_t)(event_id)) & 0x1fu) | (((uint32_t)!!(signal_enable)) << 5))
#define LLAMA3_RT_UOP_W6_SCALAR(s0, s1)            ((((uint32_t)(uint16_t)(s0)) & 0xFFFFu) | ((((uint32_t)(uint16_t)(s1)) & 0xFFFFu) << 16))

#endif
