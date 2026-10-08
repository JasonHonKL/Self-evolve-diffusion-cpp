# Self-evolve-diffusion-cpp build.
# Toolchain: user-space conda env "build" (gxx + openblas), see README.
CXX      := $(shell if [ -x $(HOME)/micromamba/envs/build/bin/x86_64-conda-linux-gnu-c++ ]; then echo $(HOME)/micromamba/envs/build/bin/x86_64-conda-linux-gnu-c++; else echo g++; fi)
CXXFLAGS := -std=c++17 -O3 -march=native -Wall -Wextra -Wno-unused-parameter -Isrc
LDFLAGS  := -O3 -march=native
BLAS     ?= $(shell if [ -f $(HOME)/micromamba/envs/build/lib/libopenblas.so ]; then echo $(HOME)/micromamba/envs/build/lib/libopenblas.so; fi)

BUILD := build
SRCS_GLOB := $(wildcard src/*.cpp src/*/*.cpp)
OBJS := $(patsubst %.cpp,$(BUILD)/%.o,$(SRCS_GLOB))

ifeq ($(strip $(BLAS)),)
CPPFLAGS += -DSD_NO_OPENBLAS
else
CPPFLAGS += -DSD_OPENBLAS=\"$(BLAS)\"
endif

TEST_BINS := $(patsubst tests/%.cpp,$(BUILD)/tests/%,$(wildcard tests/test_*.cpp))

.PHONY: all test bench clean

all: $(TEST_BINS)

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD)/tests/%: tests/%.cpp $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< $(OBJS) -o $@ -lpthread -ldl $(if $(BLAS),$(BLAS),)

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "== $$t"; ./$$t || exit 1; done; echo ALL TESTS PASSED

bench: $(TEST_BINS)
	@for t in $(TEST_BINS); do ./$$t --bench || true; done

clean:
	rm -rf $(BUILD)
