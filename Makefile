# ----------------------------------------------------------------------
#
# File: Makefile
#
# Created: 30.06.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SHELL = /usr/bin/env bash
ROOT_DIR := $(patsubst %/,%, $(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

INSTALL_PREFIX        ?= install

DEEPLOY_INSTALL_DIR           ?= ${ROOT_DIR}/${INSTALL_PREFIX}
TOOLCHAIN_DIR         := ${ROOT_DIR}/toolchain

LLVM_INSTALL_DIR      ?= ${DEEPLOY_INSTALL_DIR}/llvm
LLVM_CLANG_RT_ARM      ?= ${LLVM_INSTALL_DIR}/lib/clang/15.0.0/lib/baremetal/libclang_rt.builtins-armv7m.a
LLVM_CLANG_RT_RISCV_RV32IMC      ?= ${LLVM_INSTALL_DIR}/lib/clang/15.0.0/lib/baremetal/rv32imc/libclang_rt.builtins-riscv32.a
LLVM_CLANG_RT_RISCV_RV32IM      ?= ${LLVM_INSTALL_DIR}/lib/clang/15.0.0/lib/baremetal/rv32im/libclang_rt.builtins-riscv32.a
PICOLIBC_ARM_INSTALL_DIR      ?= ${LLVM_INSTALL_DIR}/picolibc/arm
PICOLIBC_RISCV_INSTALL_DIR      ?= ${LLVM_INSTALL_DIR}/picolibc/riscv

PULP_SDK_INSTALL_DIR ?= ${DEEPLOY_INSTALL_DIR}/pulp-sdk
QEMU_INSTALL_DIR ?= ${DEEPLOY_INSTALL_DIR}/qemu
BANSHEE_INSTALL_DIR ?= ${DEEPLOY_INSTALL_DIR}/banshee

CMAKE ?= cmake

all: toolchain emulators echo-bash

echo-bash:
	@echo "Please export the following symbols:"
	@echo "PULP_SDK_HOME=${PULP_SDK_INSTALL_DIR}"
	@echo "LLVM_INSTALL_DIR=${LLVM_INSTALL_DIR}"
	@echo "CMAKE=$$(which cmake)"

	@echo "Please add the following paths to your PATH variable:"
	@echo "${QEMU_INSTALL_DIR}/bin"
	@echo "${BANSHEE_INSTALL_DIR}"

	@echo "For PULP to work, please source the following file:"
	@echo "${PULP_SDK_INSTALL_DIR}/configs/siracusa.sh"

	@echo ""
	@echo "TL/DR: add these lines to run ~/.bashrc"
	@echo "export PULP_SDK_HOME=${PULP_SDK_INSTALL_DIR}"
	@echo "export LLVM_INSTALL_DIR=${LLVM_INSTALL_DIR}"
	@echo "export PULP_RISCV_GCC_TOOLCHAIN=/PULP_SDK_IS_A_MESS"
	@echo "export CMAKE=$$(which cmake)"
	@echo "export PATH=${QEMU_INSTALL_DIR}/bin:${BANSHEE_INSTALL_DIR}:\$$PATH"
	@echo "source ${PULP_SDK_INSTALL_DIR}/configs/siracusa.sh"


toolchain: llvm llvm-compiler-rt-riscv llvm-compiler-rt-arm picolibc-arm picolibc-riscv

emulators: pulp-sdk qemu banshee

${TOOLCHAIN_DIR}/llvm-project:
	cd ${TOOLCHAIN_DIR} && \
	git clone git@iis-git.ee.ethz.ch:iis-compilers/llvm-project.git \
	 -b upstream-rebase/main && \
	cd ${TOOLCHAIN_DIR}/llvm-project && git checkout b71678d08ab87cdca39f9b489de4f579c0f92261 && \
	git submodule update --init --recursive && \
	git apply ../llvm.patch

${LLVM_INSTALL_DIR}: ${TOOLCHAIN_DIR}/llvm-project
	cd ${TOOLCHAIN_DIR}/llvm-project && \
	mkdir -p build && cd build && \
	${CMAKE} \
	-DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR} \
	-DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
	-DLLVM_TARGETS_TO_BUILD="ARM;RISCV;host;AArch64" \
	-DLLVM_BUILD_DOCS="0" \
	-DLLVM_ENABLE_BINDINGS="0" \
	-DLLVM_ENABLE_TERMINFO="0" \
	-DLLVM_OPTIMIZED_TABLEGEN=ON \
	-DCMAKE_BUILD_TYPE=Release \
	../llvm && \
	make -j4 && \
	make install

llvm: ${LLVM_INSTALL_DIR}

${LLVM_CLANG_RT_RISCV_RV32IMC}: ${TOOLCHAIN_DIR}/llvm-project
	cd ${TOOLCHAIN_DIR}/llvm-project && mkdir -p build-compiler-rt-riscv-rv32imc \
	&& cd build-compiler-rt-riscv-rv32imc; \
	${CMAKE} ../compiler-rt \
	-DCMAKE_C_COMPILER_WORKS=1 \
	-DCMAKE_CXX_COMPILER_WORKS=1 \
	-DCMAKE_AR=${LLVM_INSTALL_DIR}/bin/llvm-ar \
	-DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}/lib/clang/15.0.0 \
	-DCMAKE_ASM_COMPILER_TARGET="riscv32-unknown-elf" \
	-DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
	-DCMAKE_ASM_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
	-DCMAKE_C_FLAGS="-mno-relax -march=rv32imc" \
	-DCMAKE_SYSTEM_NAME=baremetal \
	-DCMAKE_HOST_SYSTEM_NAME=baremetal \
	-DCMAKE_C_COMPILER_TARGET="riscv32-unknown-elf" \
	-DCMAKE_CXX_COMPILER_TARGET="riscv32-unknown-elf" \
	-DCMAKE_SIZEOF_VOID_P=4 \
	-DCMAKE_NM=${LLVM_INSTALL_DIR}/bin/llvm-nm \
	-DCMAKE_RANLIB=${LLVM_INSTALL_DIR}/bin/llvm-ranlib \
	-DCOMPILER_RT_BUILD_BUILTINS=ON \
	-DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
	-DCOMPILER_RT_BUILD_MEMPROF=OFF \
	-DCOMPILER_RT_BUILD_PROFILE=OFF \
	-DCOMPILER_RT_BUILD_SANITIZERS=OFF \
	-DCOMPILER_RT_BUILD_XRAY=OFF \
	-DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
	-DCOMPILER_RT_BAREMETAL_BUILD=ON \
	-DCOMPILER_RT_OS_DIR="baremetal/rv32imc" \
	-DLLVM_CONFIG_PATH=${LLVM_INSTALL_DIR}/bin/llvm-config && \
	make -j && \
	make install

${LLVM_CLANG_RT_RISCV_RV32IM}: ${TOOLCHAIN_DIR}/llvm-project
	cd ${TOOLCHAIN_DIR}/llvm-project && mkdir -p build-compiler-rt-riscv-rv32im \
	&& cd build-compiler-rt-riscv-rv32im; \
	${CMAKE} ../compiler-rt \
	-DCMAKE_C_COMPILER_WORKS=1 \
	-DCMAKE_CXX_COMPILER_WORKS=1 \
	-DCMAKE_AR=${LLVM_INSTALL_DIR}/bin/llvm-ar \
	-DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}/lib/clang/15.0.0 \
	-DCMAKE_ASM_COMPILER_TARGET="riscv32-unknown-elf" \
	-DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
	-DCMAKE_ASM_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
	-DCMAKE_C_FLAGS="-mno-relax -march=rv32im" \
	-DCMAKE_SYSTEM_NAME=baremetal \
	-DCMAKE_HOST_SYSTEM_NAME=baremetal \
	-DCMAKE_C_COMPILER_TARGET="riscv32-unknown-elf" \
	-DCMAKE_CXX_COMPILER_TARGET="riscv32-unknown-elf" \
	-DCMAKE_SIZEOF_VOID_P=4 \
	-DCMAKE_NM=${LLVM_INSTALL_DIR}/bin/llvm-nm \
	-DCMAKE_RANLIB=${LLVM_INSTALL_DIR}/bin/llvm-ranlib \
	-DCOMPILER_RT_BUILD_BUILTINS=ON \
	-DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
	-DCOMPILER_RT_BUILD_MEMPROF=OFF \
	-DCOMPILER_RT_BUILD_PROFILE=OFF \
	-DCOMPILER_RT_BUILD_SANITIZERS=OFF \
	-DCOMPILER_RT_BUILD_XRAY=OFF \
	-DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
	-DCOMPILER_RT_BAREMETAL_BUILD=ON \
	-DCOMPILER_RT_OS_DIR="baremetal/rv32im" \
	-DLLVM_CONFIG_PATH=${LLVM_INSTALL_DIR}/bin/llvm-config && \
	make -j && \
	make install

llvm-compiler-rt-riscv: ${LLVM_CLANG_RT_RISCV_RV32IM} ${LLVM_CLANG_RT_RISCV_RV32IMC}

${LLVM_CLANG_RT_ARM}: ${TOOLCHAIN_DIR}/llvm-project
	cd ${TOOLCHAIN_DIR}/llvm-project && mkdir -p build-compiler-rt-arm \
	&& cd build-compiler-rt-arm; \
	${CMAKE} ../compiler-rt \
	-DCMAKE_C_COMPILER_WORKS=1 \
	-DCMAKE_CXX_COMPILER_WORKS=1 \
	-DCMAKE_AR=${LLVM_INSTALL_DIR}/bin/llvm-ar \
	-DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}/lib/clang/15.0.0 \
	-DCMAKE_ASM_COMPILER_TARGET="armv7m-none-eabi" \
	-DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
	-DCMAKE_ASM_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
	-DCMAKE_C_FLAGS="-mcpu=cortex-m4 "\
	-DCMAKE_SYSTEM_NAME=baremetal \
	-DCMAKE_HOST_SYSTEM_NAME=baremetal \
	-DCMAKE_C_COMPILER_TARGET="armv7m-none-eabi" \
	-DCMAKE_CXX_COMPILER_TARGET="armv7m-none-eabi" \
	-DCMAKE_SIZEOF_VOID_P=4 \
	-DCMAKE_NM=${LLVM_INSTALL_DIR}/bin/llvm-nm \
	-DCMAKE_RANLIB=${LLVM_INSTALL_DIR}/bin/llvm-ranlib \
	-DCOMPILER_RT_BUILD_BUILTINS=ON \
	-DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
	-DCOMPILER_RT_BUILD_MEMPROF=OFF \
	-DCOMPILER_RT_BUILD_PROFILE=OFF \
	-DCOMPILER_RT_BUILD_SANITIZERS=OFF \
	-DCOMPILER_RT_BUILD_XRAY=OFF \
	-DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
	-DCOMPILER_RT_BAREMETAL_BUILD=ON \
	-DCOMPILER_RT_OS_DIR="baremetal" \
	-DLLVM_CONFIG_PATH=${LLVM_INSTALL_DIR}/bin/llvm-config && \
	make -j && \
	make install


llvm-compiler-rt-arm: ${LLVM_CLANG_RT_ARM}

${TOOLCHAIN_DIR}/picolibc:
	cd ${TOOLCHAIN_DIR} && \
	git clone git@github.com:picolibc/picolibc.git && \
	cd ${TOOLCHAIN_DIR}/picolibc && git checkout 31ff1b3601b379e4cab63837f253f59729ce1fef && \
	git submodule update --init --recursive

${PICOLIBC_ARM_INSTALL_DIR}: ${TOOLCHAIN_DIR}/picolibc
	cd ${TOOLCHAIN_DIR}/picolibc && mkdir -p build-arm && cd build-arm && \
	cp ${TOOLCHAIN_DIR}/meson-build-script-arm.txt ../scripts && \
	PATH=${LLVM_INSTALL_DIR}/bin:${PATH} meson setup -Dincludedir=include \
	-Dlibdir=lib \
	-Dspecsdir=none \
	-Dmultilib=false \
	--prefix ${PICOLIBC_ARM_INSTALL_DIR} \
	--cross-file ../scripts/meson-build-script-arm.txt && \
	PATH=${LLVM_INSTALL_DIR}/bin:${PATH} meson install

picolibc-arm: ${PICOLIBC_ARM_INSTALL_DIR}

${PICOLIBC_RISCV_INSTALL_DIR}: ${TOOLCHAIN_DIR}/picolibc
	cd ${TOOLCHAIN_DIR}/picolibc && mkdir -p build-riscv && cd build-riscv && \
	cp ${TOOLCHAIN_DIR}/meson-build-script-riscv.txt ../scripts && \
	PATH=${LLVM_INSTALL_DIR}/bin:${PATH} meson setup -Dincludedir=include \
	-Dlibdir=lib \
	-Dspecsdir=none \
	-Dmultilib=false \
	--prefix ${PICOLIBC_RISCV_INSTALL_DIR} \
	--cross-file ../scripts/meson-build-script-riscv.txt && \
	PATH=${LLVM_INSTALL_DIR}/bin:${PATH} meson install

picolibc-riscv: ${PICOLIBC_RISCV_INSTALL_DIR}

${TOOLCHAIN_DIR}/pulp-sdk:
	cd ${TOOLCHAIN_DIR} && \
	git clone git@github.com:Scheremo/pulp-sdk.git -b scheremo && \
	cd ${TOOLCHAIN_DIR}/pulp-sdk && git checkout 38e2754c4ad60ced9ef0a6f13410a7d8d6c33d58 && \
	git submodule update --init --recursive

${PULP_SDK_INSTALL_DIR}: ${TOOLCHAIN_DIR}/pulp-sdk
	mkdir -p ${PULP_SDK_INSTALL_DIR}
	cp -r ${TOOLCHAIN_DIR}/pulp-sdk/ ${PULP_SDK_INSTALL_DIR}/../
	cd ${PULP_SDK_INSTALL_DIR} && \
	source configs/siracusa.sh && \
	make build

pulp-sdk: ${PULP_SDK_INSTALL_DIR}

${TOOLCHAIN_DIR}/qemu:
	cd ${TOOLCHAIN_DIR} && \
	git clone git@github.com:qemu/qemu.git --depth 1 -b stable-6.1 && \
	cd ${TOOLCHAIN_DIR}/qemu && \
	git submodule update --init --recursive

${QEMU_INSTALL_DIR}: ${TOOLCHAIN_DIR}/qemu
	cd ${TOOLCHAIN_DIR}/qemu/ && \
	mkdir -p build && cd build && \
	../configure --target-list=arm-softmmu,arm-linux-user,riscv32-softmmu,riscv32-linux-user \
	--prefix=${QEMU_INSTALL_DIR} && \
	make -j && \
	make install

qemu: ${QEMU_INSTALL_DIR}

${TOOLCHAIN_DIR}/banshee:
	cd ${TOOLCHAIN_DIR} && \
	git clone git@github.com:pulp-platform/banshee.git && \
	cd ${TOOLCHAIN_DIR}/banshee && git checkout 2be56281f23af6edcca979026fa2ecf6eba769d8 && \
	git submodule update --init --recursive && \
	git apply ${TOOLCHAIN_DIR}/banshee.patch

${BANSHEE_INSTALL_DIR}: ${TOOLCHAIN_DIR}/banshee
	export LLVM_SYS_150_PREFIX=${LLVM_INSTALL_DIR} && \
	cd ${TOOLCHAIN_DIR}/banshee/ && \
	cargo clean && \
	cargo install --path . -f

banshee: ${BANSHEE_INSTALL_DIR}
