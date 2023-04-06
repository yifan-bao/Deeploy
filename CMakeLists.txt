cmake_minimum_required(VERSION 3.12)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_C_STANDARD 99)

set(platform MemPool CACHE STRING "Platform (MemPool, CMSIS or Generic)")
set_property(CACHE platform PROPERTY STRINGS MemPool CMSIS Generic)

if(platform STREQUAL MemPool)
    message(STATUS "Building for platform 'MemPool'")
elseif(platform STREQUAL CMSIS)
    message(FATAL_ERROR "CMake build with '${platform}' not yet supported!")
elseif(platform STREQUAL Generic)
    message(STATUS "Building for platform 'Generic'")
else()
    message(FATAL_ERROR "Invalid platform '${platform}' specified!")
endif()

# Import useful functions / macros
include(${CMAKE_CURRENT_LIST_DIR}/cmake/Util.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/common.cmake)

message(STATUS "============================= Project Configuration ============================")
message(STATUS "[Deeploy]     platform               = " ${platform})
message(STATUS "[Deeploy]     use_dma                = " ${use_dma})

if(platform STREQUAL MemPool)
    set(mempool_flavour mempool_ita CACHE STRING "Platform (mempool, mempool_ita or minpool)")
    set_property(CACHE mempool_flavour PROPERTY STRINGS mempool minpool)

    set(banshee_stack_size 16777216 CACHE STRING "Stack size of banshee threads")

    OPTION(banshee_simulation "Optimize binary for banshee simulation" OFF)
    if(banshee_simulation)
	add_compile_definitions(BANSHEE_SIMULATION)
    endif()

    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/mempool/toolchain_gcc.cmake)

    include(${CMAKE_CURRENT_LIST_DIR}/cmake/mempool/simulation.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/cmake/mempool/${mempool_flavour}.cmake)


    project(deeploy LANGUAGES C ASM)

    include_directories(deeploybasiclibs/inc)
    include_directories(deeploymempoollibs/inc)
    include_directories(deeploymempoollibs/runtime)

    add_subdirectory(deeploymempoollibs)
    add_subdirectory(deeploybasiclibs)
    add_subdirectory(deeploytest)

    target_link_libraries(deeploylib INTERFACE mempoolruntime deeploymempool deeploybasic)

    message(STATUS "============================= MemPool Configuration ============================")
    message(STATUS "[Banshee]   banshee_simulation     = " ${banshee_simulation})
    message(STATUS "[Banshee]   banshee_stack_size     = " ${banshee_stack_size})
    message(STATUS "[Banshee]   num_threads            = " ${num_threads})
    message(STATUS "[cMake  ]   mempool_flavour        = " ${mempool_flavour})
    message(STATUS "[cMake  ]   boot_addr              = " ${boot_addr})
    message(STATUS "[cMake  ]   l2_base                = " ${l2_base})
    message(STATUS "[cMake  ]   l2_size                = " ${l2_size})
    message(STATUS "[cMake  ]   l2_banks               = " ${l2_banks})
    message(STATUS "[cMake  ]   seq_mem_size           = " ${seq_mem_size})
    message(STATUS "[cMake  ]   stack_size             = " ${stack_size})
    message(STATUS "[cMake  ]   axi_data_width         = " ${axi_data_width})
    message(STATUS "[cMake  ]   ro_line_width          = " ${ro_line_width})
    message(STATUS "[cMake  ]   dmas_per_group         = " ${dmas_per_group})
    message(STATUS "[cMake  ]   xqueue_size            = " ${xqueue_size})
    message(STATUS "[cMake  ]   xpulpimg               = " ${xpulpimg})
    message(STATUS "[cMake  ]   num_cores              = " ${num_cores})
    message(STATUS "[cMake  ]   num_eff_cores          = " ${num_eff_cores})
    message(STATUS "[cMake  ]   num_groups             = " ${num_groups})
    message(STATUS "[cMake  ]   num_cores_per_tile     = " ${num_cores_per_tile})
    message(STATUS "[cMake  ]   banking_factor         = " ${banking_factor})
    message(STATUS "[cMake  ]   axi_hier_radix         = " ${axi_hier_radix})
    message(STATUS "[cMake  ]   axi_masters_per_group  = " ${axi_masters_per_group})
    if(mempool_flavour STREQUAL mempool_ita)
	message(STATUS "=============================== ITA Configuration ==============================")
	message(STATUS "[cMake  ]   ita_s                  = " ${ita_s})
	message(STATUS "[cMake  ]   ita_p                  = " ${ita_p})
	message(STATUS "[cMake  ]   ita_e                  = " ${ita_e})
	message(STATUS "[cMake  ]   ita_h                  = " ${ita_h})
    endif()
    message(STATUS "================================================================================")

endif()

if(platform STREQUAL Generic)
    include(${CMAKE_CURRENT_LIST_DIR}/cmake/generic/generic.cmake)

    project(deeploy LANGUAGES C ASM)

    include_directories(deeploybasiclibs/inc)

    add_subdirectory(deeploybasiclibs)
    add_subdirectory(deeploytest)

    target_link_libraries(deeploylib INTERFACE  deeploybasic)
endif()
