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
include ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/depend.make

# Include the progress variables for this target.
include ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/progress.make

# Include the compile flags for this target's objects.
include ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/flags.make

ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.o: ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/flags.make
ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.o: ../ParallelLoopMetadata/ParallelLoopMetadata.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gleison/lge/taskminer/build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.o"
	cd /home/gleison/lge/taskminer/build-debug/ParallelLoopMetadata && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.o -c /home/gleison/lge/taskminer/ParallelLoopMetadata/ParallelLoopMetadata.cpp

ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.i"
	cd /home/gleison/lge/taskminer/build-debug/ParallelLoopMetadata && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gleison/lge/taskminer/ParallelLoopMetadata/ParallelLoopMetadata.cpp > CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.i

ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.s"
	cd /home/gleison/lge/taskminer/build-debug/ParallelLoopMetadata && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gleison/lge/taskminer/ParallelLoopMetadata/ParallelLoopMetadata.cpp -o CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.s

ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.o.requires:

.PHONY : ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.o.requires

ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.o.provides: ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.o.requires
	$(MAKE) -f ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/build.make ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.o.provides.build
.PHONY : ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.o.provides

ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.o.provides.build: ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.o


# Object files for target ParallelLoopMetadata
ParallelLoopMetadata_OBJECTS = \
"CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.o"

# External object files for target ParallelLoopMetadata
ParallelLoopMetadata_EXTERNAL_OBJECTS =

ParallelLoopMetadata/libParallelLoopMetadata.so: ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.o
ParallelLoopMetadata/libParallelLoopMetadata.so: ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/build.make
ParallelLoopMetadata/libParallelLoopMetadata.so: ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gleison/lge/taskminer/build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module libParallelLoopMetadata.so"
	cd /home/gleison/lge/taskminer/build-debug/ParallelLoopMetadata && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ParallelLoopMetadata.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/build: ParallelLoopMetadata/libParallelLoopMetadata.so

.PHONY : ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/build

ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/requires: ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/ParallelLoopMetadata.cpp.o.requires

.PHONY : ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/requires

ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/clean:
	cd /home/gleison/lge/taskminer/build-debug/ParallelLoopMetadata && $(CMAKE_COMMAND) -P CMakeFiles/ParallelLoopMetadata.dir/cmake_clean.cmake
.PHONY : ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/clean

ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/depend:
	cd /home/gleison/lge/taskminer/build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gleison/lge/taskminer /home/gleison/lge/taskminer/ParallelLoopMetadata /home/gleison/lge/taskminer/build-debug /home/gleison/lge/taskminer/build-debug/ParallelLoopMetadata /home/gleison/lge/taskminer/build-debug/ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ParallelLoopMetadata/CMakeFiles/ParallelLoopMetadata.dir/depend

