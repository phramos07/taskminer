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
include PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/depend.make

# Include the progress variables for this target.
include PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/progress.make

# Include the compile flags for this target's objects.
include PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/flags.make

PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.o: PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/flags.make
PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.o: ../PtrRangeAnalysis/PtrRangeAnalysis.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gleison/lge/taskminer/build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.o"
	cd /home/gleison/lge/taskminer/build-debug/PtrRangeAnalysis && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.o -c /home/gleison/lge/taskminer/PtrRangeAnalysis/PtrRangeAnalysis.cpp

PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.i"
	cd /home/gleison/lge/taskminer/build-debug/PtrRangeAnalysis && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gleison/lge/taskminer/PtrRangeAnalysis/PtrRangeAnalysis.cpp > CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.i

PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.s"
	cd /home/gleison/lge/taskminer/build-debug/PtrRangeAnalysis && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gleison/lge/taskminer/PtrRangeAnalysis/PtrRangeAnalysis.cpp -o CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.s

PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.o.requires:

.PHONY : PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.o.requires

PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.o.provides: PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.o.requires
	$(MAKE) -f PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/build.make PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.o.provides.build
.PHONY : PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.o.provides

PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.o.provides.build: PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.o


PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.o: PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/flags.make
PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.o: ../PtrRangeAnalysis/SCEVRangeBuilder.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gleison/lge/taskminer/build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.o"
	cd /home/gleison/lge/taskminer/build-debug/PtrRangeAnalysis && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.o -c /home/gleison/lge/taskminer/PtrRangeAnalysis/SCEVRangeBuilder.cpp

PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.i"
	cd /home/gleison/lge/taskminer/build-debug/PtrRangeAnalysis && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gleison/lge/taskminer/PtrRangeAnalysis/SCEVRangeBuilder.cpp > CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.i

PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.s"
	cd /home/gleison/lge/taskminer/build-debug/PtrRangeAnalysis && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gleison/lge/taskminer/PtrRangeAnalysis/SCEVRangeBuilder.cpp -o CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.s

PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.o.requires:

.PHONY : PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.o.requires

PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.o.provides: PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.o.requires
	$(MAKE) -f PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/build.make PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.o.provides.build
.PHONY : PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.o.provides

PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.o.provides.build: PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.o


# Object files for target LLVMPtrRangeAnalysis
LLVMPtrRangeAnalysis_OBJECTS = \
"CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.o" \
"CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.o"

# External object files for target LLVMPtrRangeAnalysis
LLVMPtrRangeAnalysis_EXTERNAL_OBJECTS =

PtrRangeAnalysis/libLLVMPtrRangeAnalysis.so: PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.o
PtrRangeAnalysis/libLLVMPtrRangeAnalysis.so: PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.o
PtrRangeAnalysis/libLLVMPtrRangeAnalysis.so: PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/build.make
PtrRangeAnalysis/libLLVMPtrRangeAnalysis.so: PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gleison/lge/taskminer/build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared module libLLVMPtrRangeAnalysis.so"
	cd /home/gleison/lge/taskminer/build-debug/PtrRangeAnalysis && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LLVMPtrRangeAnalysis.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/build: PtrRangeAnalysis/libLLVMPtrRangeAnalysis.so

.PHONY : PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/build

PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/requires: PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/PtrRangeAnalysis.cpp.o.requires
PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/requires: PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/SCEVRangeBuilder.cpp.o.requires

.PHONY : PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/requires

PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/clean:
	cd /home/gleison/lge/taskminer/build-debug/PtrRangeAnalysis && $(CMAKE_COMMAND) -P CMakeFiles/LLVMPtrRangeAnalysis.dir/cmake_clean.cmake
.PHONY : PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/clean

PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/depend:
	cd /home/gleison/lge/taskminer/build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gleison/lge/taskminer /home/gleison/lge/taskminer/PtrRangeAnalysis /home/gleison/lge/taskminer/build-debug /home/gleison/lge/taskminer/build-debug/PtrRangeAnalysis /home/gleison/lge/taskminer/build-debug/PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : PtrRangeAnalysis/CMakeFiles/LLVMPtrRangeAnalysis.dir/depend

