/* =====================================================================
 * Title:        Gemm.h
 * Description:
 *
 * Date:         05.01.2023
 *
 * ===================================================================== */

/*
 * Copyright (C) 2023 ETH Zurich and University of Bologna.
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

#ifndef __DEEPLOY_BASIC_MATH_GEMM_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_GEMM_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 * This library implements the matrix multiplication for several data widths
 * in multiple different ways. The functions all follow the following format:
 *
 * A is an M x N matrix, B is a N x P matrix, and C is a M x P matrix
 * A' = transpose(A) if transA else A
 * B' = transpose(B) if transB else B
 *
 * Y = alpha * A' * B' + beta * C
 *
 */

/******************************************************************************/
/*                     General Matrix Multiplication (8bit)                   */
/******************************************************************************/

/*
 * Matrix multiplication ----------------------------------
 * kernel     = Gemm_s8_s8_s32
 * data type  = 8-bit integer
 * unrolling  = no
 * cleanup    = yes
 */
void Gemm_s8_s8_s32_s32(int8_t const *__restrict__ pSrcA,
                        int8_t const *__restrict__ pSrcB,
                        int32_t const *__restrict__ pSrcC,
                        int32_t *__restrict__ pDstY, uint32_t M, uint32_t N,
                        uint32_t P, int32_t alpha, int32_t beta, int32_t transA,
                        int32_t transB, int32_t A_offset, int32_t B_offset,
                        int32_t C_offset, int32_t Y_offset);

#endif //__DEEPLOY_BASIC_MATH_GEMM_KERNEL_HEADER_
