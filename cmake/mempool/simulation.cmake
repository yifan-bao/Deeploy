#############################
##  Simulation COnfig  ##
#############################

set(num_threads  1  CACHE STRING "Number of active cores")

set(BANSHEE_CONFIG ${CMAKE_CURRENT_LIST_DIR}/mempool.yaml CACHE INTERNAL "source_list")

#########################
##  Utility Functions  ##
#########################

macro(add_banshee_simulation name)
    add_custom_target(sim_${name} 
        DEPENDS ${name}
        COMMAND RUST_MIN_STACK=${banshee_stack_size} banshee
        --num-cores=${num_threads}
        --num-clusters=1 
        --latency 
        --configuration 
        ${BANSHEE_CONFIG}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name}
        COMMENT "Simulating deeploytest with banshe"
        POST_BUILD
        USES_TERMINAL
        VERBATIM
    )
endmacro()

