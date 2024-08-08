/* =====================================================================
 * Title:        MaxPool.h
 * Description:
 *
 * Date:         13.12.2022
 *
 * ===================================================================== */

/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Philip Wiese, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __DEEPLOY_MATH_MAXPOOL_KERNEL_HEADER_
#define __DEEPLOY_MATH_MAXPOOL_KERNEL_HEADER_

#include "DeeployMath.h"

/* This file implements the MaxPool operation.
 *
 * A is an M x N input matrix, P x Q the kernel size and SPxSQ the kernel
 * stride.
 *
 * Note that all the matrices dimensions must be multiples of 4; these
 * kernels do not have clean-up code and remaining elements would not be
 * considered, leading to wrong results.
 */

/******************************************************************************/
/*                         General MaxPool (8bit)                         */
/******************************************************************************/

/*
 * 2D Maxpool  ----------------------------------
 * kernel      = MaxPool2d_parallel_s8_NCHW_rv32im
 * layout      = NCHW
 * data type   = 8-bit integer
 * kernel size = generic
 * multi-core  = yes
 * unrolling   = no
 * simd        = no
 */
void MaxPool2d_parallel_s8_NCHW_rv32im(int8_t const *__restrict__ pSrcA,
                                       uint32_t C, uint32_t H, uint32_t W,
                                       uint32_t P, uint32_t Q, uint32_t SP,
                                       uint32_t SQ, int8_t *__restrict__ pDstC,
                                       int32_t input_offset,
                                       int32_t output_offset, uint32_t core_id,
                                       uint32_t numThreads);

// Mapper Functions
static inline void __attribute__((always_inline))
MaxPool2d_parallel_s8_NCHW(int8_t const *__restrict__ pSrcA, uint32_t C,
                           uint32_t H, uint32_t W, uint32_t P, uint32_t Q,
                           uint32_t SP, uint32_t SQ, int8_t *__restrict__ pDstC,
                           int32_t input_offset, int32_t output_offset,
                           uint32_t core_id, uint32_t numThreads) {
  MaxPool2d_parallel_s8_NCHW_rv32im(pSrcA, C, H, W, P, Q, SP, SQ, pDstC,
                                    input_offset, output_offset, core_id,
                                    numThreads);
}

#endif //__DEEPLOY_MATH_MAXPOOL_KERNEL_HEADER_
