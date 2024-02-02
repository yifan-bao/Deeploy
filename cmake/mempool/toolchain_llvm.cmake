set(TOOLCHAIN_PREFIX ${TOOLCHAIN_INSTALL_DIR}/bin)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}/llvm-objcopy)
set(CMAKE_OBJDUMP  ${TOOLCHAIN_PREFIX}/llvm-objdump)
set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}/clang)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}/clang++)
set(CMAKE_AR ${TOOLCHAIN_PREFIX}/llvm-ar)
set(SIZE ${TOOLCHAIN_PREFIX}/llvm-size)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

message(${GCC_INSTALL_DIR}/../riscv32-unknown-elf)
add_compile_options(
  --target=riscv32-unknown-elf
  --sysroot=${TOOLCHAIN_INSTALL_DIR}/picolibc/riscv

  -D__builtin_shuffle=__builtin_pulp_shuffle2h
  -march=rv32ima
  -mabi=ilp32
  -mcmodel=medany
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
  --sysroot=${TOOLCHAIN_INSTALL_DIR}/picolibc/riscv

  -march=rv32ima
  -mabi=ilp32
  -mcmodel=medany
  -mcpu=mempool-rv32
  -std=gnu99

  # -falign-loops=32
  # -falign-jumps=32
  # Turn of optimization that lead to known problems
  -fno-builtin-memcpy
  -fno-builtin-memset
  -v

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

  -T${CMAKE_BINARY_DIR}/link.ld
  -L${TOOLCHAIN_INSTALL_DIR}/lib/clang/15.0.0/lib/baremetal/rv32im/
  -L${CMAKE_BINARY_DIR}
)

link_libraries(
  -lm
)

add_compile_definitions(__TOOLCHAIN_LLVM__)