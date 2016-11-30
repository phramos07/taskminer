#ifndef ANALYSIS_CONTROLDEPENDENCEGRAPH_H
#define ANALYSIS_CONTROLDEPENDENCEGRAPH_H

class ProgramDependenceGraph {
	// Currently GraphNode is just a wrap around Instruction*, however
	// I imagine that in the future we will want to add other properties
	// to PDG nodes which is what motivated the creation of this class.
	class GraphNode {
	public:
		Instruction* instr;

		GraphNode(Instruction* _instr) : instr(_instr)
		{}
	};

	// Represent an edge in the PDG. We store whether the edge is a data
	// or control dependence as well as the source and sink of the dependence.
	class GraphEdge {
	public:
		DependenceType type;
		GraphNode* src;
		GraphNode* dst;

		GraphEdge(GraphNode* _src, GraphNode* _dst, DependenceType _type) : type(_type), src(_src), dst(_dst)
		{}
	};

	// Name of the function that this graph was created from. Just for reference/debugging/printing.
	std::string functionName;

	GraphNode* entry;
	GraphNode* exit;

	// using this data structure we can easily obtain all edges leaving a node
	std::map<GraphNode*, std::set<GraphEdge*>> outEdges;

	// using this data structure we can easily obtain all edges directly reaching a node
	std::map<GraphNode*, std::set<GraphEdge*>> inEdges;

	// we use this data structure to retrieve informations about a node representing
	// an instruction.
	std::map<Instruction*, std::pair<unsigned int, GraphNode*>> instrToNode;

	unsigned int nextInstrID;

	private:
		GraphNode* getGraphNode(Instruction* instr) {
			if (instrToNode.find(instr) == instrToNode.end())  {
				instrToNode[instr] = std::make_pair(nextInstrID, new GraphNode(instr));
				nextInstrID++;
			}

			return instrToNode[instr].second;
		}

	public:
		ProgramDependenceGraph(std::string _functionName) : functionName(_functionName), nextInstrID(0)
		{}

		void addNode(Instruction* ins) {
			getGraphNode(ins);
		}

		void addEntryNode(Instruction* ins) {
			this->entry = getGraphNode(ins);
		}

		void addExitNode(Instruction* ins) {
			this->exit = getGraphNode(ins);
		}

		void addEdge(Instruction* srcI, Instruction* dstI, DependenceType type) {
			auto src = getGraphNode(srcI);	
			auto dst = getGraphNode(dstI);	
			auto edge = new GraphEdge(src, dst, type);

			outEdges[src].insert(edge);
			inEdges[dst].insert(edge);
		}


		void connectToEntry(Instruction* dstI) {
			addEdge(entry->instr, dstI, DependenceType::CTR);
		}

		void connectToExit(Instruction* srcI) {
			addEdge(srcI, exit->instr, DependenceType::CTR);
		}

		void dumpToDot(Function& F) {
			errs() << "Dumping instructions for function :: " << F.getName() << "\n";

			// Dump the function source with instruction IDs for reference. We also
			// assign for each instruction address a smaller numerical ID for easy
			// visual inspection of the graph.
			for (Function::iterator bbIt = F.begin(), e = F.end(); bbIt != e; ++bbIt) {
				if (bbIt->hasName())
					errs() << bbIt->getName() << "\n";
				else
					errs() << "Unnamed Basic Block\n";

				for (BasicBlock::iterator insIt = bbIt->begin(), e = bbIt->end(); insIt != e; ++insIt) {
					errs() << "[" << instrToNode[&*insIt].first << "]" << *insIt << "\n";
				}
			}

			// Write the graph to a DOT file
			std::string graphName = functionName + ".dot";
			std::ofstream dotStream;
			dotStream.open(graphName);

			if (!dotStream.is_open()) {
				errs() << "Problem opening DOT file: " << graphName << "\n";
			}	
			else {
				dotStream << "digraph g {\n";

				// Create all nodes in DOT format
				for (auto node : instrToNode) {
					if (node.second.second == this->entry) 
						dotStream << "\t\"" << instrToNode[node.second.second->instr].first << "\" [label=entry];\n";
					else if (node.second.second == this->exit)
						dotStream << "\t\"" << instrToNode[node.second.second->instr].first << "\" [label=exit];\n";
					else
						dotStream << "\t\"" << instrToNode[node.second.second->instr].first << "\";\n";
				}

				dotStream << "\n\n";

				// Now print all outgoing edges and their labels
				for (auto src : outEdges) {
					for (auto dst : src.second) {
						if (dst->type == CTR)
							dotStream << "\t\"" << instrToNode[src.first->instr].first << "\" -> \"" << instrToNode[dst->dst->instr].first << "\" [style=dotted];\n";
						else
							dotStream << "\t\"" << instrToNode[src.first->instr].first << "\" -> \"" << instrToNode[dst->dst->instr].first << "\" [label=\"" << getDependenceName(dst->type)  << "\"];\n";
					}

					dotStream << "\n";
				}

				dotStream << "}";
				dotStream.close();
			}

			errs() << "\n\n";
		}
};

#endif // ANALYSIS_CONTROLDEPENDENCEGRAPH_H
