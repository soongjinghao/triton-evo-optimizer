#!/usr/bin/env python3
from pathlib import Path
import os
import shutil

SRC_ROOT = Path(os.environ.get("TRITON_ASCEND_SRC", "/workspace/triton-ascend-src"))
CPP = SRC_ROOT / "third_party/ascend/lib/TritonToGraph/DataflowGraph.cpp"
MARKER = "COSTMODEL_V2_DFG_OPERATION_GRAPH_V1"

HELPERS = r'''// COSTMODEL_V2_DFG_OPERATION_GRAPH_V1
// Build the operation-level DFG from native MLIR SSA use-def relations.
// MemorySSA remains an auxiliary representation for pointer/tensor memory state.
namespace {

static std::string costModelEscapeJSON(const std::string &input) {
  std::string out;
  out.reserve(input.size() + 8);
  static const char hex[] = "0123456789abcdef";
  for (unsigned char c : input) {
    switch (c) {
    case '"': out += "\\\""; break;
    case '\\': out += "\\\\"; break;
    case '\b': out += "\\b"; break;
    case '\f': out += "\\f"; break;
    case '\n': out += "\\n"; break;
    case '\r': out += "\\r"; break;
    case '\t': out += "\\t"; break;
    default:
      if (c < 0x20) {
        out += "\\u00";
        out += hex[(c >> 4) & 0x0f];
        out += hex[c & 0x0f];
      } else {
        out += static_cast<char>(c);
      }
    }
  }
  return out;
}

static std::string costModelValueString(Value value) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << value;
  stream.flush();
  return result;
}

static std::string costModelTypeString(Type type) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  type.print(stream);
  stream.flush();
  return result;
}

static std::string costModelOperationString(Operation *op) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  op->print(stream);
  stream.flush();
  return result;
}

struct CostModelDFGNode {
  std::string id;
  std::string kind;
  std::string opName;
  std::string text;
  std::string type;
  std::string owner;
  unsigned index = 0;
  unsigned regionIndex = 0;
  unsigned blockIndex = 0;
  unsigned resultCount = 0;
};

struct CostModelDFGEdge {
  std::string source;
  std::string target;
  std::string kind;
  std::string valueType;
  unsigned operandIndex = 0;
  unsigned resultIndex = 0;
  bool hasResultIndex = false;
};

struct CostModelOperationGraph {
  llvm::SmallVector<CostModelDFGNode, 64> nodes;
  llvm::SmallVector<CostModelDFGEdge, 128> edges;
};

static CostModelOperationGraph
costModelBuildOperationGraph(mlir::triton::FuncOp function) {
  CostModelOperationGraph graph;
  Operation *root = function.getOperation();
  if (!root)
    return graph;

  // Collect every operation in the Triton function, including nested regions.
  llvm::SmallVector<Operation *, 64> operations;
  root->walk([&](Operation *op) {
    if (op != root)
      operations.push_back(op);
  });

  llvm::DenseMap<Operation *, unsigned> operationNodeIndex;
  for (Operation *op : operations) {
    CostModelDFGNode node;
    node.id = "op_" + std::to_string(graph.nodes.size());
    node.kind = "operation";
    node.opName = op->getName().getStringRef().str();
    node.text = costModelOperationString(op);
    node.resultCount = op->getNumResults();
    operationNodeIndex[op] = graph.nodes.size();
    graph.nodes.push_back(std::move(node));
  }

  // Function arguments and nested-region block arguments are explicit input
  // nodes. This preserves values that do not have an Operation as producer.
  llvm::DenseMap<Value, unsigned> blockArgumentNodeIndex;
  llvm::SmallVector<Operation *, 64> regionOwners;
  regionOwners.push_back(root);
  regionOwners.append(operations.begin(), operations.end());

  for (Operation *owner : regionOwners) {
    unsigned regionIndex = 0;
    for (Region &region : owner->getRegions()) {
      unsigned blockIndex = 0;
      for (Block &block : region) {
        unsigned argumentIndex = 0;
        for (BlockArgument argument : block.getArguments()) {
          CostModelDFGNode node;
          node.id = "arg_" + std::to_string(blockArgumentNodeIndex.size());
          node.kind = "block_argument";
          node.type = costModelTypeString(argument.getType());
          node.owner = owner == root
                           ? std::string("function")
                           : owner->getName().getStringRef().str();
          node.index = argumentIndex;
          node.regionIndex = regionIndex;
          node.blockIndex = blockIndex;
          blockArgumentNodeIndex[argument] = graph.nodes.size();
          graph.nodes.push_back(std::move(node));
          ++argumentIndex;
        }
        ++blockIndex;
      }
      ++regionIndex;
    }
  }

  // Every operand creates an edge from its defining operation or block
  // argument to the consuming operation. Operand positions are retained.
  for (Operation *consumer : operations) {
    auto targetIt = operationNodeIndex.find(consumer);
    if (targetIt == operationNodeIndex.end())
      continue;
    const unsigned targetIndex = targetIt->second;

    for (OpOperand &operand : consumer->getOpOperands()) {
      Value value = operand.get();
      unsigned sourceIndex = 0;
      bool foundSource = false;
      std::string edgeKind;

      if (Operation *producer = value.getDefiningOp()) {
        auto sourceIt = operationNodeIndex.find(producer);
        if (sourceIt != operationNodeIndex.end()) {
          sourceIndex = sourceIt->second;
          foundSource = true;
          edgeKind = "ssa";
        }
      } else {
        auto sourceIt = blockArgumentNodeIndex.find(value);
        if (sourceIt != blockArgumentNodeIndex.end()) {
          sourceIndex = sourceIt->second;
          foundSource = true;
          edgeKind = "block_argument";
        }
      }

      if (!foundSource)
        continue;

      CostModelDFGEdge edge;
      edge.source = graph.nodes[sourceIndex].id;
      edge.target = graph.nodes[targetIndex].id;
      edge.kind = edgeKind;
      edge.valueType = costModelTypeString(value.getType());
      edge.operandIndex = operand.getOperandNumber();
      if (auto result = llvm::dyn_cast<OpResult>(value)) {
        edge.hasResultIndex = true;
        edge.resultIndex = result.getResultNumber();
      }
      graph.edges.push_back(std::move(edge));
    }
  }

  return graph;
}

static void costModelExportOperationGraphJSON(llvm::raw_ostream &os,
                                               mlir::triton::FuncOp function) {
  CostModelOperationGraph graph = costModelBuildOperationGraph(function);
  os << "{\n";
  os << "    \"schema\": \"mlir_ssa_operation_graph_v1\",\n";
  os << "    \"node_count\": " << graph.nodes.size() << ",\n";
  os << "    \"edge_count\": " << graph.edges.size() << ",\n";
  os << "    \"nodes\": [\n";

  for (size_t i = 0; i < graph.nodes.size(); ++i) {
    const CostModelDFGNode &node = graph.nodes[i];
    if (i)
      os << ",\n";
    os << "      {\"id\": \"" << costModelEscapeJSON(node.id)
       << "\", \"kind\": \"" << costModelEscapeJSON(node.kind) << "\"";
    if (node.kind == "operation") {
      os << ", \"op\": \"" << costModelEscapeJSON(node.opName)
         << "\", \"text\": \"" << costModelEscapeJSON(node.text)
         << "\", \"result_count\": " << node.resultCount;
    } else {
      os << ", \"owner\": \"" << costModelEscapeJSON(node.owner)
         << "\", \"arg_index\": " << node.index
         << ", \"region_index\": " << node.regionIndex
         << ", \"block_index\": " << node.blockIndex
         << ", \"type\": \"" << costModelEscapeJSON(node.type) << "\"";
    }
    os << "}";
  }

  os << "\n    ],\n";
  os << "    \"edges\": [\n";
  for (size_t i = 0; i < graph.edges.size(); ++i) {
    const CostModelDFGEdge &edge = graph.edges[i];
    if (i)
      os << ",\n";
    os << "      {\"source\": \"" << costModelEscapeJSON(edge.source)
       << "\", \"target\": \"" << costModelEscapeJSON(edge.target)
       << "\", \"kind\": \"" << costModelEscapeJSON(edge.kind)
       << "\", \"operand_index\": " << edge.operandIndex
       << ", \"value_type\": \"" << costModelEscapeJSON(edge.valueType)
       << "\"";
    if (edge.hasResultIndex)
      os << ", \"result_index\": " << edge.resultIndex;
    os << "}";
  }
  os << "\n    ]\n";
  os << "  }";
}

static std::string costModelEscapeDOT(const std::string &input) {
  std::string out;
  out.reserve(input.size() + 8);
  for (char c : input) {
    switch (c) {
    case '\\': out += "\\\\"; break;
    case '"': out += "\\\""; break;
    case '\n': out += "\\n"; break;
    case '\r': break;
    default: out += c; break;
    }
  }
  return out;
}

} // namespace
'''

DATAFLOWINFO_EXPORT = r'''void DataFlowInfo::exportToJSON(llvm::raw_ostream &os) const {
  os << "{\n";
  os << "  \"memoryDefinitions\": {\n";
  bool first = true;
  for (const auto &entry : memoryDefinitions) {
    if (!first)
      os << ",\n";
    first = false;
    os << "    \"" << costModelEscapeJSON(costModelValueString(entry.first))
       << "\": {\n";
    os << "      \"id\": \""
       << costModelEscapeJSON(entry.second->getId()) << "\",\n";
    os << "      \"tensor\": \""
       << costModelEscapeJSON(entry.second->getTensor()->getName()) << "\",\n";
    os << "      \"version\": " << entry.second->getVersion() << ",\n";
    os << "      \"is_parameter\": "
       << (entry.second->isParameter() ? "true" : "false") << ",\n";
    os << "      \"def_op\": ";
    if (Operation *defOp = entry.second->getDefOp())
      os << "\"" << costModelEscapeJSON(
          defOp->getName().getStringRef().str()) << "\"";
    else
      os << "null";
    os << "\n    }";
  }
  os << "\n  },\n";

  os << "  \"memoryUses\": [\n";
  bool firstUse = true;
  forEachUse([&](const MemorySSAUse &use) {
    MemorySSADef *definition = use.getDefinition();
    Operation *userOp = use.getUserOp();
    if (!definition || !userOp)
      return;
    if (!firstUse)
      os << ",\n";
    firstUse = false;
    os << "    {\"definition_id\": \""
       << costModelEscapeJSON(definition->getId())
       << "\", \"user_op\": \""
       << costModelEscapeJSON(use.getUserOpName())
       << "\", \"operand_index\": " << use.getOperandIdx()
       << ", \"value\": \""
       << costModelEscapeJSON(costModelValueString(use.getValue()))
       << "\", \"value_type\": \""
       << costModelEscapeJSON(costModelTypeString(use.getValue().getType()))
       << "\"}";
  });
  os << "\n  ],\n";

  os << "  \"Phis\": {\n";
  first = true;
  for (const auto &entry : Phis) {
    if (!first)
      os << ",\n";
    first = false;
    os << "    \"" << costModelEscapeJSON(costModelValueString(entry.first))
       << "\": {\n";
    os << "      \"type\": " << static_cast<int>(entry.second.type) << "\n";
    os << "    }";
  }
  os << "\n  }\n";
  os << "}\n";
}'''

DFG_EXPORT = r'''void DataFlowGraph::exportToJSON(llvm::raw_ostream &os) const {
  auto funcOp = cfg.getFunction();
  auto funcName = funcOp.getName();
  std::string funcNameStr = funcName.empty() ? "unnamed" : funcName.str();

  os << "{\n";
  os << "  \"function\": \"" << costModelEscapeJSON(funcNameStr) << "\",\n";
  os << "  \"graph\": ";
  costModelExportOperationGraphJSON(os, funcOp);
  os << ",\n  \"dataFlow\": ";
  dataFlowInfo.exportToJSON(os);
  os << "}\n";
}'''

DOT_EXPORT = r'''void DataFlowGraph::exportDefUseToDOT(llvm::raw_ostream &os) const {
  CostModelOperationGraph graph = costModelBuildOperationGraph(cfg.getFunction());
  os << "digraph TritonOperationDFG {\n";
  os << "  rankdir=LR;\n";
  os << "  node [shape=box];\n\n";

  for (const CostModelDFGNode &node : graph.nodes) {
    std::string label;
    if (node.kind == "operation") {
      label = node.id + "\n" + node.opName;
    } else {
      label = node.id + "\nblock_arg[" + std::to_string(node.index) + "]\n" +
              node.type;
    }
    os << "  " << node.id << " [label=\""
       << costModelEscapeDOT(label) << "\"];\n";
  }

  os << "\n";
  for (const CostModelDFGEdge &edge : graph.edges) {
    std::string label = edge.kind + " operand#" +
                        std::to_string(edge.operandIndex);
    os << "  " << edge.source << " -> " << edge.target
       << " [label=\"" << costModelEscapeDOT(label) << "\"];\n";
  }
  os << "}\n";
}'''


def find_cpp_function_end(source: str, start: int) -> int:
    """Return the position immediately after the matching function brace."""
    opening = source.find("{", start)
    if opening < 0:
        raise RuntimeError("Function opening brace not found")
    depth = 0
    state = "normal"
    i = opening
    while i < len(source):
        c = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if state == "normal":
            if c == "/" and nxt == "/":
                state = "line_comment"
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = "block_comment"
                i += 2
                continue
            if c == '"':
                state = "string"
            elif c == "'":
                state = "char"
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        elif state == "line_comment":
            if c == "\n":
                state = "normal"
        elif state == "block_comment":
            if c == "*" and nxt == "/":
                state = "normal"
                i += 2
                continue
        elif state == "string":
            if c == "\\":
                i += 2
                continue
            if c == '"':
                state = "normal"
        elif state == "char":
            if c == "\\":
                i += 2
                continue
            if c == "'":
                state = "normal"
        i += 1
    raise RuntimeError("Could not find matching function closing brace")


def main():
    if not CPP.is_file():
        raise FileNotFoundError(
            f"Cannot find source file: {CPP}\n"
            "Run this script on the server that contains triton-ascend-src."
        )

    source = CPP.read_text(encoding="utf-8")
    if MARKER in source:
        print("[SKIP] DFG operation graph patch already present")
        return

    include_anchor = '#include "llvm/Support/raw_ostream.h"'
    if include_anchor not in source:
        raise RuntimeError("Cannot find raw_ostream include anchor; source differs from inspected version")
    source = source.replace(
        include_anchor,
        include_anchor + '\n#include "llvm/ADT/DenseMap.h"\n#include "llvm/ADT/SmallVector.h"\n#include <string>\n#include <utility>',
        1,
    )

    namespace_anchor = "using namespace cfg;"
    if namespace_anchor not in source:
        raise RuntimeError("Cannot find namespace anchor; source differs from inspected version")
    source = source.replace(namespace_anchor, namespace_anchor + "\n\n" + HELPERS, 1)

    info_start = "void DataFlowInfo::exportToJSON(llvm::raw_ostream &os) const {"
    info_end = "//===----------------------------------------------------------------------===//\n// DataFlowGraph"
    if info_start not in source or info_end not in source:
        raise RuntimeError("Cannot locate DataFlowInfo::exportToJSON replacement range")
    a = source.index(info_start)
    b = source.index(info_end, a)
    source = source[:a] + DATAFLOWINFO_EXPORT + "\n\n" + source[b:]

    graph_start = "void DataFlowGraph::exportToJSON(llvm::raw_ostream &os) const {"
    graph_end = "void DataFlowGraph::exportDefUseToDOT(llvm::raw_ostream &os) const {"
    if graph_start not in source or graph_end not in source:
        raise RuntimeError("Cannot locate DataFlowGraph export replacement range")
    a = source.index(graph_start)
    b = source.index(graph_end, a)
    source = source[:a] + DFG_EXPORT + "\n\n" + source[b:]

    dot_start = source.index(graph_end)
    dot_end = find_cpp_function_end(source, dot_start)
    if source[dot_start:dot_end].count("{") == 0:
        raise RuntimeError("DOT export body unexpectedly missing")
    source = source[:dot_start] + DOT_EXPORT + source[dot_end:]

    backup = CPP.with_name(CPP.name + ".before_costmodel_dfg_export_v1")
    if not backup.exists():
        shutil.copy2(CPP, backup)
        print(f"[BACKUP] {backup}")

    CPP.write_text(source, encoding="utf-8")
    print("[SUCCESS] Patched DataflowGraph.cpp")
    print("[INFO] Added operation-level native MLIR SSA nodes/edges to DFG JSON")
    print("[INFO] Added MemorySSA uses to dataFlow.memoryUses")
    print("[INFO] Replaced empty MemorySSA-only DOT output with operation DFG")
    print(f"[SOURCE] {CPP}")


if __name__ == "__main__":
    main()
