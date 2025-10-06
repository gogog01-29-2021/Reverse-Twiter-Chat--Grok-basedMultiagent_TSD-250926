# Alternative Makefile for systems without CMake
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -Icore -I. -pthread
LDFLAGS = -pthread

# Source files
CORE_SOURCES = \
    core/engine/runtime.cpp \
    core/engine/graph.cpp \
    core/engine/node.cpp \
    core/types/trade.cpp \
    core/types/marketdata.cpp \
    core/types/instrument.cpp \
    core/types/exchange.cpp \
    dsm/publisher/order_book_publisher.cpp \
    osm/managers/order_manager.cpp

CORE_OBJECTS = $(CORE_SOURCES:.cpp=.o)

# Targets
all: basic_example trading_example complete_trading_system tsm_basic tsm_trade_pnl

# Core library
libtsd_core.a: $(CORE_OBJECTS)
	ar rcs $@ $^

# Examples
basic_example: examples/basic_example.o libtsd_core.a
	$(CXX) $(LDFLAGS) -o $@ $^

trading_example: examples/trading_example.o libtsd_core.a
	$(CXX) $(LDFLAGS) -o $@ $^

complete_trading_system: examples/complete_trading_system.o libtsd_core.a
	$(CXX) $(LDFLAGS) -o $@ $^

tsm_basic: tsm/training/stage2/e1_basic.o libtsd_core.a
	$(CXX) $(LDFLAGS) -o $@ $^

tsm_trade_pnl: tsm/training/stage2/e4_trade_pnl.o libtsd_core.a
	$(CXX) $(LDFLAGS) -o $@ $^

# Object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Test run
test: all
	@echo "Running basic example..."
	./basic_example
	@echo ""
	@echo "Running trading example..."
	./trading_example
	@echo ""
	@echo "Running TSM basic..."
	./tsm_basic

# Clean
clean:
	rm -f $(CORE_OBJECTS) examples/*.o tsm/training/stage2/*.o
	rm -f libtsd_core.a
	rm -f basic_example trading_example complete_trading_system tsm_basic tsm_trade_pnl

.PHONY: all test clean