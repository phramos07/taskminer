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
include ScopeTree/CMakeFiles/LLVMScopeTree.dir/depend.make

# Include the progress variables for this target.
include ScopeTree/CMakeFiles/LLVMScopeTree.dir/progress.make

# Include the compile flags for this target's objects.
include ScopeTree/CMakeFiles/LLVMScopeTree.dir/flags.make

ScopeTree/CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.o: ScopeTree/CMakeFiles/LLVMScopeTree.dir/flags.make
ScopeTree/CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.o: ../ScopeTree/ScopeTree.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gleison/lge/taskminer/build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ScopeTree/CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.o"
	cd /home/gleison/lge/taskminer/build-debug/ScopeTree && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.o -c /home/gleison/lge/taskminer/ScopeTree/ScopeTree.cpp

ScopeTree/CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.i"
	cd /home/gleison/lge/taskminer/build-debug/ScopeTree && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gleison/lge/taskminer/ScopeTree/ScopeTree.cpp > CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.i

ScopeTree/CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.s"
	cd /home/gleison/lge/taskminer/build-debug/ScopeTree && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gleison/lge/taskminer/ScopeTree/ScopeTree.cpp -o CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.s

ScopeTree/CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.o.requires:

.PHONY : ScopeTree/CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.o.requires

ScopeTree/CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.o.provides: ScopeTree/CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.o.requires
	$(MAKE) -f ScopeTree/CMakeFiles/LLVMScopeTree.dir/build.make ScopeTree/CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.o.provides.build
.PHONY : ScopeTree/CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.o.provides

ScopeTree/CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.o.provides.build: ScopeTree/CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.o


# Object files for target LLVMScopeTree
LLVMScopeTree_OBJECTS = \
"CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.o"

# External object files for target LLVMScopeTree
LLVMScopeTree_EXTERNAL_OBJECTS =

ScopeTree/libLLVMScopeTree.so: ScopeTree/CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.o
ScopeTree/libLLVMScopeTree.so: ScopeTree/CMakeFiles/LLVMScopeTree.dir/build.make
ScopeTree/libLLVMScopeTree.so: ScopeTree/CMakeFiles/LLVMScopeTree.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gleison/lge/taskminer/build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module libLLVMScopeTree.so"
	cd /home/gleison/lge/taskminer/build-debug/ScopeTree && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LLVMScopeTree.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ScopeTree/CMakeFiles/LLVMScopeTree.dir/build: ScopeTree/libLLVMScopeTree.so

.PHONY : ScopeTree/CMakeFiles/LLVMScopeTree.dir/build

ScopeTree/CMakeFiles/LLVMScopeTree.dir/requires: ScopeTree/CMakeFiles/LLVMScopeTree.dir/ScopeTree.cpp.o.requires

.PHONY : ScopeTree/CMakeFiles/LLVMScopeTree.dir/requires

ScopeTree/CMakeFiles/LLVMScopeTree.dir/clean:
	cd /home/gleison/lge/taskminer/build-debug/ScopeTree && $(CMAKE_COMMAND) -P CMakeFiles/LLVMScopeTree.dir/cmake_clean.cmake
.PHONY : ScopeTree/CMakeFiles/LLVMScopeTree.dir/clean

ScopeTree/CMakeFiles/LLVMScopeTree.dir/depend:
	cd /home/gleison/lge/taskminer/build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gleison/lge/taskminer /home/gleison/lge/taskminer/ScopeTree /home/gleison/lge/taskminer/build-debug /home/gleison/lge/taskminer/build-debug/ScopeTree /home/gleison/lge/taskminer/build-debug/ScopeTree/CMakeFiles/LLVMScopeTree.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ScopeTree/CMakeFiles/LLVMScopeTree.dir/depend

