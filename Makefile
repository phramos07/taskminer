LLVM_CONFIG?=$(OBJ_ROOT)/bin/llvm-config
CXX=g++

SRC_DIR=./
PASS_SO=release/libTaskMiner.so

CPP_FILES=$(wildcard $(SRC_DIR)/*.cpp)  
OBJ_FILES=$(addprefix $(SRC_DIR)/, $(notdir $(CPP_FILES:.cpp=.o)))

ifeq ($(shell uname),Darwin)
LOADABLE_MODULE_OPTIONS=-bundle -undefined dynamic_lookup
else
LOADABLE_MODULE_OPTIONS=-shared -Wl,-O1
endif

COMMON_FLAGS=-Wall -Wextra -fvisibility=hidden
CXXFLAGS+=$(COMMON_FLAGS) $(shell $(LLVM_CONFIG) --cxxflags) 


CPP_OPTIONS+=$(CPPFLAGS) $(shell $(LLVM_CONFIG) --cppflags) \
	     -MD -MP -I$(SRC_DIR) 

LD_OPTIONS+=$(LDFLAGS) $(shell $(LLVM_CONFIG) --ldflags)

default: $(PASS_SO)

$(SRC_DIR)/%.o : %.cpp
	@echo Compiling $*.cpp for `$(LLVM_CONFIG) --build-mode` build
	$(QUIET)$(CXX) -c $(CPP_OPTIONS) $(CXXFLAGS) $<

$(PASS_SO): $(OBJ_FILES) 
	@echo Linking $@
	$(QUIET)$(CXX) -o $@ $(LOADABLE_MODULE_OPTIONS) $(CXXFLAGS) \
	$(LD_OPTIONS) $(OBJ_FILES)

clean::
	$(QUIET)rm -f $(SRC_DIR)/*.o $(PASS_SO)


