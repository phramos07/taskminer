# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.1

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
include DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/depend.make

# Include the progress variables for this target.
include DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/progress.make

# Include the compile flags for this target's objects.
include DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/flags.make

DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.o: DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/flags.make
DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.o: ../DepBasedParallelLoopAnalysis/ParallelLoopAnalysis.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/gleison/lge/taskminer/build-debug/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.o"
	cd /home/gleison/lge/taskminer/build-debug/DepBasedParallelLoopAnalysis && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.o -c /home/gleison/lge/taskminer/DepBasedParallelLoopAnalysis/ParallelLoopAnalysis.cpp

DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.i"
	cd /home/gleison/lge/taskminer/build-debug/DepBasedParallelLoopAnalysis && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/gleison/lge/taskminer/DepBasedParallelLoopAnalysis/ParallelLoopAnalysis.cpp > CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.i

DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.s"
	cd /home/gleison/lge/taskminer/build-debug/DepBasedParallelLoopAnalysis && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/gleison/lge/taskminer/DepBasedParallelLoopAnalysis/ParallelLoopAnalysis.cpp -o CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.s

DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.o.requires:
.PHONY : DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.o.requires

DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.o.provides: DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.o.requires
	$(MAKE) -f DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/build.make DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.o.provides.build
.PHONY : DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.o.provides

DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.o.provides.build: DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.o

# Object files for target ParallelLoopAnalysis
ParallelLoopAnalysis_OBJECTS = \
"CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.o"

# External object files for target ParallelLoopAnalysis
ParallelLoopAnalysis_EXTERNAL_OBJECTS =

DepBasedParallelLoopAnalysis/libParallelLoopAnalysis.so: DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.o
DepBasedParallelLoopAnalysis/libParallelLoopAnalysis.so: DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/build.make
DepBasedParallelLoopAnalysis/libParallelLoopAnalysis.so: DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared module libParallelLoopAnalysis.so"
	cd /home/gleison/lge/taskminer/build-debug/DepBasedParallelLoopAnalysis && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ParallelLoopAnalysis.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/build: DepBasedParallelLoopAnalysis/libParallelLoopAnalysis.so
.PHONY : DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/build

DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/requires: DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/ParallelLoopAnalysis.cpp.o.requires
.PHONY : DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/requires

DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/clean:
	cd /home/gleison/lge/taskminer/build-debug/DepBasedParallelLoopAnalysis && $(CMAKE_COMMAND) -P CMakeFiles/ParallelLoopAnalysis.dir/cmake_clean.cmake
.PHONY : DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/clean

DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/depend:
	cd /home/gleison/lge/taskminer/build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gleison/lge/taskminer /home/gleison/lge/taskminer/DepBasedParallelLoopAnalysis /home/gleison/lge/taskminer/build-debug /home/gleison/lge/taskminer/build-debug/DepBasedParallelLoopAnalysis /home/gleison/lge/taskminer/build-debug/DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : DepBasedParallelLoopAnalysis/CMakeFiles/ParallelLoopAnalysis.dir/depend
