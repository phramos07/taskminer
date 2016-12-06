#include "ProgramDependenceGraph.h"

std::string getDependenceName(DependenceType V) {
	switch (V) {
		case RAR: return "RAR";
		case RAWLC: return "RAW*";
		case WAW: return "WAW";
		case RAW: return "RAW";
		case WAR: return "WAR";
		case CTR: return "CTR";
		case SCA: return "SCA";
		default: return std::to_string(V);
	}
}
