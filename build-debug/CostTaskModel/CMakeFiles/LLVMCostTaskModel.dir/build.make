# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/gleison/lge/taskminer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/gleison/lge/taskminer/build-debug

# Include any dependencies generated for this target.
include CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/depend.make

# Include the progress variables for this target.
include CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/progress.make

# Include the compile flags for this target's objects.
include CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/flags.make

CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.o: CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/flags.make
CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.o: ../CostTaskModel/costTaskModel.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gleison/lge/taskminer/build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.o"
	cd /home/gleison/lge/taskminer/build-debug/CostTaskModel && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.o -c /home/gleison/lge/taskminer/CostTaskModel/costTaskModel.cpp

CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.i"
	cd /home/gleison/lge/taskminer/build-debug/CostTaskModel && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gleison/lge/taskminer/CostTaskModel/costTaskModel.cpp > CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.i

CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.s"
	cd /home/gleison/lge/taskminer/build-debug/CostTaskModel && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gleison/lge/taskminer/CostTaskModel/costTaskModel.cpp -o CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.s

CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.o.requires:

.PHONY : CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.o.requires

CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.o.provides: CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.o.requires
	$(MAKE) -f CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/build.make CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.o.provides.build
.PHONY : CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.o.provides

CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.o.provides.build: CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.o


# Object files for target LLVMCostTaskModel
LLVMCostTaskModel_OBJECTS = \
"CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.o"

# External object files for target LLVMCostTaskModel
LLVMCostTaskModel_EXTERNAL_OBJECTS =

CostTaskModel/libLLVMCostTaskModel.so: CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.o
CostTaskModel/libLLVMCostTaskModel.so: CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/build.make
CostTaskModel/libLLVMCostTaskModel.so: CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gleison/lge/taskminer/build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module libLLVMCostTaskModel.so"
	cd /home/gleison/lge/taskminer/build-debug/CostTaskModel && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LLVMCostTaskModel.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/build: CostTaskModel/libLLVMCostTaskModel.so

.PHONY : CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/build

CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/requires: CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/costTaskModel.cpp.o.requires

.PHONY : CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/requires

CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/clean:
	cd /home/gleison/lge/taskminer/build-debug/CostTaskModel && $(CMAKE_COMMAND) -P CMakeFiles/LLVMCostTaskModel.dir/cmake_clean.cmake
.PHONY : CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/clean

CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/depend:
	cd /home/gleison/lge/taskminer/build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gleison/lge/taskminer /home/gleison/lge/taskminer/CostTaskModel /home/gleison/lge/taskminer/build-debug /home/gleison/lge/taskminer/build-debug/CostTaskModel /home/gleison/lge/taskminer/build-debug/CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CostTaskModel/CMakeFiles/LLVMCostTaskModel.dir/depend

