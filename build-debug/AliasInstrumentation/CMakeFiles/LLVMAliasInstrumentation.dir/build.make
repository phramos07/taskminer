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
include AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/depend.make

# Include the progress variables for this target.
include AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/progress.make

# Include the compile flags for this target's objects.
include AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/flags.make

AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.o: AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/flags.make
AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.o: ../AliasInstrumentation/AliasInstrumentation.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/gleison/lge/taskminer/build-debug/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.o"
	cd /home/gleison/lge/taskminer/build-debug/AliasInstrumentation && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.o -c /home/gleison/lge/taskminer/AliasInstrumentation/AliasInstrumentation.cpp

AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.i"
	cd /home/gleison/lge/taskminer/build-debug/AliasInstrumentation && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/gleison/lge/taskminer/AliasInstrumentation/AliasInstrumentation.cpp > CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.i

AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.s"
	cd /home/gleison/lge/taskminer/build-debug/AliasInstrumentation && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/gleison/lge/taskminer/AliasInstrumentation/AliasInstrumentation.cpp -o CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.s

AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.o.requires:
.PHONY : AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.o.requires

AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.o.provides: AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.o.requires
	$(MAKE) -f AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/build.make AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.o.provides.build
.PHONY : AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.o.provides

AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.o.provides.build: AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.o

AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.o: AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/flags.make
AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.o: ../AliasInstrumentation/RegionCloneUtil.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/gleison/lge/taskminer/build-debug/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.o"
	cd /home/gleison/lge/taskminer/build-debug/AliasInstrumentation && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.o -c /home/gleison/lge/taskminer/AliasInstrumentation/RegionCloneUtil.cpp

AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.i"
	cd /home/gleison/lge/taskminer/build-debug/AliasInstrumentation && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/gleison/lge/taskminer/AliasInstrumentation/RegionCloneUtil.cpp > CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.i

AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.s"
	cd /home/gleison/lge/taskminer/build-debug/AliasInstrumentation && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/gleison/lge/taskminer/AliasInstrumentation/RegionCloneUtil.cpp -o CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.s

AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.o.requires:
.PHONY : AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.o.requires

AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.o.provides: AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.o.requires
	$(MAKE) -f AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/build.make AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.o.provides.build
.PHONY : AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.o.provides

AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.o.provides.build: AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.o

# Object files for target LLVMAliasInstrumentation
LLVMAliasInstrumentation_OBJECTS = \
"CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.o" \
"CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.o"

# External object files for target LLVMAliasInstrumentation
LLVMAliasInstrumentation_EXTERNAL_OBJECTS =

AliasInstrumentation/libLLVMAliasInstrumentation.so: AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.o
AliasInstrumentation/libLLVMAliasInstrumentation.so: AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.o
AliasInstrumentation/libLLVMAliasInstrumentation.so: AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/build.make
AliasInstrumentation/libLLVMAliasInstrumentation.so: AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared module libLLVMAliasInstrumentation.so"
	cd /home/gleison/lge/taskminer/build-debug/AliasInstrumentation && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LLVMAliasInstrumentation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/build: AliasInstrumentation/libLLVMAliasInstrumentation.so
.PHONY : AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/build

AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/requires: AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/AliasInstrumentation.cpp.o.requires
AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/requires: AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/RegionCloneUtil.cpp.o.requires
.PHONY : AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/requires

AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/clean:
	cd /home/gleison/lge/taskminer/build-debug/AliasInstrumentation && $(CMAKE_COMMAND) -P CMakeFiles/LLVMAliasInstrumentation.dir/cmake_clean.cmake
.PHONY : AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/clean

AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/depend:
	cd /home/gleison/lge/taskminer/build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gleison/lge/taskminer /home/gleison/lge/taskminer/AliasInstrumentation /home/gleison/lge/taskminer/build-debug /home/gleison/lge/taskminer/build-debug/AliasInstrumentation /home/gleison/lge/taskminer/build-debug/AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : AliasInstrumentation/CMakeFiles/LLVMAliasInstrumentation.dir/depend
