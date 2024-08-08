/*
 * dory_dma.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "dory_dma.h"

#include "pmsis.h"

#ifndef MCHAN_BASE_ADDR
// FIXME: For GAP9, this must point to ARCHI_MCHAN_EXT_ADDR!!!
// In PULP-SDK for Kraken, this is fixed.
// GAP8 hardware to be tested...
#define MCHAN_BASE_ADDR (ARCHI_MCHAN_DEMUX_ADDR) // CLUSTER_MCHAN_ADDR
#endif
#define MCHAN_EVENT
// #define MCHAN_POLLED
#ifdef MCHAN_EVENT
#define MCHAN_EVENT_BIT (ARCHI_CL_EVT_DMA0) // 8
#endif
#include "mchan.h"

#if defined(MCHAN_POLLED)
#define MCHAN_FLAGS (MCHAN_CMD_FLAG_INCREMENTAL)
#elif defined(MCHAN_EVENT)
#define MCHAN_FLAGS (MCHAN_CMD_FLAG_EVENT_ENABLE | MCHAN_CMD_FLAG_INCREMENTAL)
#elif defined(MCHAN_INTERRUPT)
#define MCHAN_FLAGS                                                            \
  (MCHAN_CMD_FLAG_INTERRUPT_ENABLE | MCHAN_CMD_FLAG_INCREMENTAL)
#endif

#define MCHAN_FLAGS_1D (MCHAN_FLAGS)
#define MCHAN_FLAGS_2D (MCHAN_FLAGS | MCHAN_CMD_FLAG_2D_TRANSFER_EXTERNAL)

#define MIN(a, b) ((a) < (b) ? (a) : (b))

void dory_dma_memcpy_hwc_to_chw(DMA_copy *copy) {
  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int number_of_copies_per_core =
      (copy->length_1d_copy >> Log2Core) +
      ((copy->length_1d_copy & (NUM_CORES - 1)) != 0);
  int start_pixel, stop_pixel; // "pixel" is a misnomer; the CHANNELS are
                               // divided between the cores
  // this function assumes that a DW tile is always as wide as the complete
  // feature map (this is enforced by DORY's tiler)
  start_pixel = MIN(number_of_copies_per_core * core_id, copy->length_1d_copy);
  stop_pixel =
      MIN(start_pixel + number_of_copies_per_core, copy->length_1d_copy);
  void *ext = copy->ext + start_pixel;
  void *loc = copy->loc + copy->number_of_1d_copies *
                              copy->number_of_2d_copies * start_pixel;
  const int size_2d = copy->number_of_1d_copies * copy->number_of_2d_copies;

  for (int i = start_pixel; i < stop_pixel; i++) {
    mchan_transfer_t trans = {.cmd = size_2d |
                                     copy->dir << MCHAN_CMD_SHIFT_DIRECTION |
                                     MCHAN_FLAGS_2D,
                              .size = size_2d,
                              .ext = ext,
                              .loc = loc,
                              .ext_size_1d = 1, // one byte at a time...
                              .ext_stride_1d = copy->stride_1d};
    mchan_transfer_push_2d(trans);
#ifdef ALWAYS_BLOCK_DMA_TRANSFERS // needed on GAP8 board
    dory_dma_barrier(copy);
#endif
    ext += 1; // next channel
    loc += copy->number_of_1d_copies * copy->number_of_2d_copies;
  }
}

void dory_dma_memcpy_1d_async(DMA_copy *copy) {
  if (pi_core_id() == 0) {
    mchan_transfer_t trans = {
        .cmd = copy->length_1d_copy * copy->number_of_1d_copies *
                   copy->number_of_2d_copies |
               (copy->dir << MCHAN_CMD_SHIFT_DIRECTION) | MCHAN_FLAGS_1D,
        .size = copy->length_1d_copy * copy->number_of_1d_copies *
                copy->number_of_2d_copies,
        .ext = copy->ext,
        .loc = copy->loc};
    mchan_transfer_push_1d(trans);
  }
}

void dory_dma_memcpy_2d_async(DMA_copy *copy) {
  if (pi_core_id() == 0) {
    const int size_2d = copy->number_of_1d_copies * copy->length_1d_copy *
                        copy->number_of_2d_copies;
    const int stride =
        (copy->number_of_2d_copies == 1) ? copy->stride_1d : copy->stride_2d;
    const int size_1d = (copy->number_of_2d_copies == 1)
                            ? copy->length_1d_copy
                            : copy->length_1d_copy * copy->number_of_1d_copies;

    mchan_transfer_t trans = {.cmd = size_2d |
                                     copy->dir << MCHAN_CMD_SHIFT_DIRECTION |
                                     MCHAN_FLAGS_2D,
                              .size = size_2d,
                              .ext = copy->ext,
                              .loc = copy->loc,
                              .ext_size_1d = size_1d,
                              .ext_stride_1d = stride};
    mchan_transfer_push_2d(trans);
  }
}

void dory_dma_memcpy_3d_async(DMA_copy *copy) {
  int core_id = pi_core_id();
  if (core_id == 0) {
    int Log2Core = log2(1);
    int number_of_2d_copies_per_core = (copy->number_of_2d_copies >> Log2Core) +
                                       ((copy->number_of_2d_copies & (0)) != 0);
    int start_pixel, stop_pixel;
    start_pixel =
        MIN(number_of_2d_copies_per_core * core_id, copy->number_of_2d_copies);
    stop_pixel = MIN(start_pixel + number_of_2d_copies_per_core,
                     copy->number_of_2d_copies);
    void *ext = copy->ext + copy->stride_2d * start_pixel;
    void *loc = copy->loc +
                copy->length_1d_copy * copy->number_of_1d_copies * start_pixel;
    const int size_2d = copy->number_of_1d_copies * copy->length_1d_copy;

    for (int i = start_pixel; i < stop_pixel; i++) {
      mchan_transfer_t trans = {.cmd = size_2d |
                                       copy->dir << MCHAN_CMD_SHIFT_DIRECTION |
                                       MCHAN_FLAGS_2D,
                                .size = size_2d,
                                .ext = ext,
                                .loc = loc,
                                .ext_size_1d = copy->length_1d_copy,
                                .ext_stride_1d = copy->stride_1d};
      mchan_transfer_push_2d(trans);
#ifdef ALWAYS_BLOCK_DMA_TRANSFERS // needed on GAP8 board
                                  // dory_dma_barrier(copy);
#endif
      loc += size_2d;
      ext += copy->stride_2d;
    }
  }
}

void dory_dma_memcpy_async(DMA_copy *copy) {
  if (copy->hwc_to_chw == 1) {
    dory_dma_memcpy_hwc_to_chw(copy);
  } else if ((copy->number_of_2d_copies == 1 &&
              copy->number_of_1d_copies == 1) ||
             (copy->stride_1d == copy->length_1d_copy &&
              copy->number_of_1d_copies * copy->length_1d_copy ==
                  copy->stride_2d) ||
             (copy->number_of_2d_copies == 1 &&
              copy->length_1d_copy == copy->stride_1d)) {
    dory_dma_memcpy_1d_async(copy);
  } else if ((copy->number_of_2d_copies == 1) ||
             (copy->length_1d_copy == copy->stride_1d)) { // wrong!
    dory_dma_memcpy_2d_async(copy);
  } else {
    dory_dma_memcpy_3d_async(copy);
  }
}

void dory_dma_memcpy_1d_mindims_async(DMA_copy *copy) {
  mchan_transfer_t trans = {
      .cmd = copy->mchan_cmd, .ext = copy->ext, .loc = copy->loc};
  mchan_transfer_push_1d(trans);
}

void dory_dma_memcpy_2d_mindims_async(DMA_copy *copy) {
  mchan_transfer_t trans = {.cmd = copy->mchan_cmd,
                            .ext = copy->ext,
                            .loc = copy->loc,
                            .ext_size_1d = copy->length_1d_copy,
                            .ext_stride_1d = copy->stride_1d};
  mchan_transfer_push_2d(trans);
}

void dory_dma_memcpy_3d_mindims_async(DMA_copy *copy) {
  void *ext = copy->ext;
  void *loc = copy->loc;
  const int length_2d_copy =
      copy->mchan_cmd & ((1 << MCHAN_TRANSFER_LEN_SIZE) - 1);

  for (int i = 0; i < copy->number_of_2d_copies; i++) {
    mchan_transfer_t trans = {.cmd = copy->mchan_cmd,
                              .ext = ext,
                              .loc = loc,
                              .ext_size_1d = copy->length_1d_copy,
                              .ext_stride_1d = copy->stride_1d};
    mchan_transfer_push_2d(trans);
    loc += length_2d_copy;
    ext += copy->stride_2d;
#ifdef ALWAYS_BLOCK_DMA_TRANSFERS // needed on GAP8 board
                                  // dory_dma_barrier(copy);
#endif
  }
}

void dory_dma_memcpy_mindims_async(DMA_copy *copy) {
  if (copy->number_of_2d_copies == 1 && copy->number_of_1d_copies == 1) {
    dory_dma_memcpy_1d_mindims_async(copy);
  } else if (copy->number_of_2d_copies == 1) {
    dory_dma_memcpy_2d_mindims_async(copy);
  } else {
    dory_dma_memcpy_3d_mindims_async(copy);
  }
}

void dory_dma_free(DMA_copy *copy) { mchan_transfer_free(copy->tid); }

void dory_dma_barrier(DMA_copy *copy) { mchan_transfer_wait(copy->tid); }

int dory_dma_allocate() { return mchan_transfer_get_id(); }
