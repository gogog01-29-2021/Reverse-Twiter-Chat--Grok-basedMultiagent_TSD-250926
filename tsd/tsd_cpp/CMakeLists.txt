cmake_minimum_required(VERSION 3.20)
project(TSD_CPP VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

if(MSVC)
    add_compile_options(/W4 /O2)
endif()

# Dependencies
find_package(Threads REQUIRED)

# Core library
add_library(tsd_core
    # Core engine
    core/engine/runtime.cpp
    core/engine/graph.cpp
    core/engine/node.cpp

    # Core types
    core/types/trade.cpp
    core/types/marketdata.cpp
    core/types/instrument.cpp
    core/types/exchange.cpp

    # DSM
    dsm/publisher/order_book_publisher.cpp

    # OSM
    osm/managers/order_manager.cpp
)

target_include_directories(tsd_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/core
)

target_link_libraries(tsd_core
    PUBLIC
        Threads::Threads
)

# Examples
add_executable(basic_example examples/basic_example.cpp)
target_link_libraries(basic_example tsd_core)

add_executable(trading_example examples/trading_example.cpp)
target_link_libraries(trading_example tsd_core)

add_executable(complete_trading_system examples/complete_trading_system.cpp)
target_link_libraries(complete_trading_system tsd_core)

# TSM Training examples
add_executable(tsm_basic tsm/training/stage2/e1_basic.cpp)
target_link_libraries(tsm_basic tsd_core)

add_executable(tsm_trade_pnl tsm/training/stage2/e4_trade_pnl.cpp)
target_link_libraries(tsm_trade_pnl tsd_core)

# Install targets
install(TARGETS tsd_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY core/
    DESTINATION include/tsd
    FILES_MATCHING PATTERN "*.hpp"
)

install(DIRECTORY dsm/
    DESTINATION include/tsd
    FILES_MATCHING PATTERN "*.hpp"
)

install(DIRECTORY osm/
    DESTINATION include/tsd
    FILES_MATCHING PATTERN "*.hpp"
)