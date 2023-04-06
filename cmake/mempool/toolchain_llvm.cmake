set(TOOLCHAIN_PREFIX ${LLVM_INSTALL_DIR})

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}llvm-objcopy)
set(CMAKE_OBJDUMP  ${TOOLCHAIN_PREFIX}llvm-objdump)
set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}clang)
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}clang++)
set(CMAKE_AR ${TOOLCHAIN_PREFIX}llvm-ar)
set(SIZE ${TOOLCHAIN_PREFIX}llvm-size)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

message(${GCC_INSTALL_DIR}/../riscv32-unknown-elf)
add_compile_options(
    --target=riscv32-unknown-elf
    --sysroot=${GCC_INSTALL_DIR}../riscv32-unknown-elf
    --gcc-toolchain=${GCC_INSTALL_DIR}../

    -march=rv32ima
    -mabi=ilp32 
    -mcmodel=small
    -mcpu=mempool-rv32
    -mllvm 
    -misched-topdown

    -std=gnu99

    # -falign-loops=32 
    # -falign-jumps=32
    # Turn of optimization that lead to known problems
    -fno-builtin-memcpy 
    -fno-builtin-memset

    -ffast-math 
    -fno-builtin-printf 
    -fno-common
    -fdiagnostics-color=always

    -Wunused-variable 
    -Wconversion 
    -Wall 
    -Wextra

    -static 
)

add_link_options(
    --target=riscv32-unknown-elf
    --sysroot=${GCC_INSTALL_DIR}/../riscv32-unknown-elf
    --gcc-toolchain=${GCC_INSTALL_DIR}/../

    -march=rv32ima
    -mabi=ilp32 
    -mcmodel=small
    -mcpu=mempool-rv32
    -mllvm 
    -misched-topdown

    -std=gnu99

    # -falign-loops=32 
    # -falign-jumps=32
    # Turn of optimization that lead to known problems
    -fno-builtin-memcpy 
    -fno-builtin-memset

    -ffast-math 
    -fno-builtin-printf 
    -fno-common
    -fdiagnostics-color=always

    -Wunused-variable 
    -Wconversion 
    -Wall 
    -Wextra

    -static 
    -nostartfiles

    -lm 
    -lgcc

    -T${CMAKE_BINARY_DIR}/link.ld
    -L${CMAKE_BINARY_DIR}
)

add_compile_definitions(__TOOLCHAIN_LLVM__)


