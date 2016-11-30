add_llvm_loadable_module( LLVMTaskFinder
	TaskFinder.cpp
	ControlDependenceGraph.cpp
	ProgramDependenceGraph.cpp

	PLUGIN_TOOL
	opt
)
