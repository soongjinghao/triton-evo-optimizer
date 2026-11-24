#!/usr/bin/env python3
# Patch costModel_v2's TTIR graph extractor with a first generic MLIR graph path.
# Preserves native tt.func handling; adds func.func SSA DFG and Region-aware CFG.
# Does not change HIVM/DLTI parser compatibility.

from __future__ import annotations
import argparse
import datetime as _dt
import shutil
import sys
from pathlib import Path

HELPERS = r'''
// ===== COSTMODEL_V2_GENERIC_GRAPH_EXTRACTOR_V1 BEGIN =====

static std::string genericEscapeJSON(const std::string &input) {
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

static std::string genericTypeString(mlir::Type type) {
  std::string result;
  llvm::raw_string_ostream os(result);
  type.print(os);
  os.flush();
  return result;
}

static std::string genericOperationString(mlir::Operation *op) {
  std::string result;
  llvm::raw_string_ostream os(result);
  op->print(os);
  os.flush();
  return result;
}

struct GenericDFGNode {
  std::string id;
  std::string kind;
  std::string opName;
  std::string text;
  std::string type;
  std::string owner;
  unsigned index = 0;
  unsigned regionIndex = 0;
  unsigned blockIndex = 0;
  unsigned operandCount = 0;
  unsigned resultCount = 0;
  llvm::SmallVector<std::string, 4> resultTypes;
};

struct GenericDFGEdge {
  std::string source;
  std::string target;
  std::string kind;
  std::string valueType;
  unsigned operandIndex = 0;
  unsigned resultIndex = 0;
  bool hasResultIndex = false;
};

struct GenericDFG {
  llvm::SmallVector<GenericDFGNode, 64> nodes;
  llvm::SmallVector<GenericDFGEdge, 128> edges;
};

static GenericDFG buildGenericDFG(mlir::Operation *function) {
  GenericDFG graph;
  if (!function)
    return graph;

  llvm::SmallVector<mlir::Operation *, 64> operations;
  function->walk([&](mlir::Operation *op) {
    if (op != function)
      operations.push_back(op);
  });

  llvm::DenseMap<mlir::Operation *, unsigned> operationNodeIndex;
  llvm::DenseMap<mlir::Operation *, unsigned> operationRegionIndex;
  llvm::DenseMap<mlir::Operation *, unsigned> operationBlockIndex;

  llvm::SmallVector<mlir::Operation *, 64> regionOwners;
  regionOwners.push_back(function);
  regionOwners.append(operations.begin(), operations.end());

  // Record each operation's containing Region/Block.
  for (mlir::Operation *owner : regionOwners) {
    unsigned regionIndex = 0;
    for (mlir::Region &region : owner->getRegions()) {
      unsigned blockIndex = 0;
      for (mlir::Block &block : region) {
        for (mlir::Operation &op : block) {
          if (&op == function)
            continue;
          operationRegionIndex[&op] = regionIndex;
          operationBlockIndex[&op] = blockIndex;
        }
        ++blockIndex;
      }
      ++regionIndex;
    }
  }

  for (mlir::Operation *op : operations) {
    GenericDFGNode node;
    node.id = "op_" + std::to_string(graph.nodes.size());
    node.kind = "operation";
    node.opName = op->getName().getStringRef().str();
    node.text = genericOperationString(op);
    node.operandCount = op->getNumOperands();
    node.resultCount = op->getNumResults();
    node.regionIndex = operationRegionIndex.lookup(op);
    node.blockIndex = operationBlockIndex.lookup(op);
    for (mlir::Type resultType : op->getResultTypes())
      node.resultTypes.push_back(genericTypeString(resultType));
    operationNodeIndex[op] = graph.nodes.size();
    graph.nodes.push_back(std::move(node));
  }

  // Function arguments and nested-region block arguments are explicit nodes.
  llvm::DenseMap<mlir::Value, unsigned> blockArgumentNodeIndex;
  for (mlir::Operation *owner : regionOwners) {
    unsigned regionIndex = 0;
    for (mlir::Region &region : owner->getRegions()) {
      unsigned blockIndex = 0;
      for (mlir::Block &block : region) {
        unsigned argumentIndex = 0;
        for (mlir::BlockArgument argument : block.getArguments()) {
          GenericDFGNode node;
          node.id = "arg_" + std::to_string(blockArgumentNodeIndex.size());
          node.kind = "block_argument";
          node.type = genericTypeString(argument.getType());
          node.owner = owner == function
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

  // SSA use-def edges. Memory effects are deliberately not inferred here.
  for (mlir::Operation *consumer : operations) {
    auto targetIt = operationNodeIndex.find(consumer);
    if (targetIt == operationNodeIndex.end())
      continue;

    for (mlir::OpOperand &operand : consumer->getOpOperands()) {
      mlir::Value value = operand.get();
      unsigned sourceIndex = 0;
      bool foundSource = false;
      std::string edgeKind;

      if (mlir::Operation *producer = value.getDefiningOp()) {
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

      GenericDFGEdge edge;
      edge.source = graph.nodes[sourceIndex].id;
      edge.target = graph.nodes[targetIt->second].id;
      edge.kind = edgeKind;
      edge.valueType = genericTypeString(value.getType());
      edge.operandIndex = operand.getOperandNumber();
      if (auto result = llvm::dyn_cast<mlir::OpResult>(value)) {
        edge.hasResultIndex = true;
        edge.resultIndex = result.getResultNumber();
      }
      graph.edges.push_back(std::move(edge));
    }
  }
  return graph;
}

static bool writeGenericDFG(
    mlir::Operation *function,
    const std::string &functionName,
    const llvm::SmallString<256> &path) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "[error] Cannot create generic DFG file: "
                 << ec.message() << "\n";
    return false;
  }

  GenericDFG graph = buildGenericDFG(function);
  os << "{\n";
  os << "  \"function\": \"" << genericEscapeJSON(functionName) << "\",\n";
  os << "  \"ir_kind\": \"generic_mlir\",\n";
  os << "  \"graph\": {\n";
  os << "    \"schema\": \"mlir_ssa_operation_graph_v1\",\n";
  os << "    \"node_count\": " << graph.nodes.size() << ",\n";
  os << "    \"edge_count\": " << graph.edges.size() << ",\n";
  os << "    \"nodes\": [\n";

  for (size_t i = 0; i < graph.nodes.size(); ++i) {
    const GenericDFGNode &node = graph.nodes[i];
    if (i)
      os << ",\n";
    os << "      {\"id\": \"" << genericEscapeJSON(node.id)
       << "\", \"kind\": \"" << genericEscapeJSON(node.kind) << "\"";
    if (node.kind == "operation") {
      os << ", \"op\": \"" << genericEscapeJSON(node.opName)
         << "\", \"text\": \"" << genericEscapeJSON(node.text)
         << "\", \"operand_count\": " << node.operandCount
         << ", \"result_count\": " << node.resultCount
         << ", \"region_index\": " << node.regionIndex
         << ", \"block_index\": " << node.blockIndex
         << ", \"result_types\": [";
      for (size_t j = 0; j < node.resultTypes.size(); ++j) {
        if (j)
          os << ", ";
        os << "\"" << genericEscapeJSON(node.resultTypes[j]) << "\"";
      }
      os << "]";
    } else {
      os << ", \"owner\": \"" << genericEscapeJSON(node.owner)
         << "\", \"arg_index\": " << node.index
         << ", \"region_index\": " << node.regionIndex
         << ", \"block_index\": " << node.blockIndex
         << ", \"type\": \"" << genericEscapeJSON(node.type) << "\"";
    }
    os << "}";
  }

  os << "\n    ],\n";
  os << "    \"edges\": [\n";
  for (size_t i = 0; i < graph.edges.size(); ++i) {
    const GenericDFGEdge &edge = graph.edges[i];
    if (i)
      os << ",\n";
    os << "      {\"source\": \"" << genericEscapeJSON(edge.source)
       << "\", \"target\": \"" << genericEscapeJSON(edge.target)
       << "\", \"kind\": \"" << genericEscapeJSON(edge.kind)
       << "\", \"operand_index\": " << edge.operandIndex
       << ", \"value_type\": \"" << genericEscapeJSON(edge.valueType) << "\"";
    if (edge.hasResultIndex)
      os << ", \"result_index\": " << edge.resultIndex;
    os << "}";
  }
  os << "\n    ]\n";
  os << "  },\n";
  os << "  \"dataFlow\": {\n";
  os << "    \"status\": \"not_computed\",\n";
  os << "    \"reason\": \"Generic IR V1 exports SSA use-def edges only; "
         "MemorySSA and alias analysis are not computed for func.func.\"\n";
  os << "  }\n";
  os << "}\n";
  os.flush();
  return !os.has_error();
}

struct GenericCFGRegion {
  unsigned id = 0;
  int parentRegion = -1;
  unsigned depth = 0;
  std::string ownerOp;
  llvm::SmallVector<unsigned, 8> blockIds;
};

struct GenericCFGBlock {
  unsigned id = 0;
  unsigned regionId = 0;
  unsigned blockIndex = 0;
  unsigned argumentCount = 0;
  llvm::SmallVector<std::string, 16> operations;
  llvm::SmallVector<unsigned, 4> successors;
  llvm::SmallVector<unsigned, 4> predecessors;
};

class GenericCFGBuilder {
public:
  void build(mlir::Operation *function) {
    regions.clear();
    blocks.clear();
    blockIds.clear();
    blockPointers.clear();
    if (!function)
      return;

    for (mlir::Region &region : function->getRegions())
      addRegion(region, function->getName().getStringRef().str(), -1, 0);

    // V1 records explicit terminator successors. Region nesting is recorded
    // separately; implicit scf/RegionBranch transitions are a later extension.
    for (GenericCFGBlock &record : blocks) {
      mlir::Block *block = blockPointers[record.id];
      if (!block || block->empty())
        continue;

      mlir::Operation *terminator = &block->back();
      for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
        mlir::Block *successor = terminator->getSuccessor(i);
        auto successorIt = blockIds.find(successor);
        if (successorIt == blockIds.end())
          continue;
        unsigned successorId = successorIt->second;
        record.successors.push_back(successorId);
        blocks[successorId].predecessors.push_back(record.id);
      }
    }
  }

  const llvm::SmallVector<GenericCFGRegion, 16> &getRegions() const {
    return regions;
  }
  const llvm::SmallVector<GenericCFGBlock, 32> &getBlocks() const {
    return blocks;
  }

private:
  void addRegion(mlir::Region &region,
                 const std::string &ownerOp,
                 int parentRegion,
                 unsigned depth) {
    GenericCFGRegion regionRecord;
    regionRecord.id = regions.size();
    regionRecord.parentRegion = parentRegion;
    regionRecord.depth = depth;
    regionRecord.ownerOp = ownerOp;
    const unsigned regionId = regionRecord.id;
    regions.push_back(std::move(regionRecord));

    unsigned blockIndex = 0;
    for (mlir::Block &block : region) {
      GenericCFGBlock blockRecord;
      blockRecord.id = blocks.size();
      blockRecord.regionId = regionId;
      blockRecord.blockIndex = blockIndex++;
      blockRecord.argumentCount = block.getNumArguments();
      for (mlir::Operation &op : block)
        blockRecord.operations.push_back(op.getName().getStringRef().str());

      blockIds[&block] = blockRecord.id;
      blockPointers.push_back(&block);
      regions[regionId].blockIds.push_back(blockRecord.id);
      blocks.push_back(std::move(blockRecord));
    }

    for (mlir::Block &block : region) {
      for (mlir::Operation &op : block) {
        for (mlir::Region &nestedRegion : op.getRegions()) {
          addRegion(nestedRegion,
                    op.getName().getStringRef().str(),
                    static_cast<int>(regionId),
                    depth + 1);
        }
      }
    }
  }

  llvm::SmallVector<GenericCFGRegion, 16> regions;
  llvm::SmallVector<GenericCFGBlock, 32> blocks;
  llvm::DenseMap<mlir::Block *, unsigned> blockIds;
  llvm::SmallVector<mlir::Block *, 32> blockPointers;
};

static bool writeGenericCFG(
    mlir::Operation *function,
    const std::string &functionName,
    const llvm::SmallString<256> &path) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "[error] Cannot create generic CFG file: "
                 << ec.message() << "\n";
    return false;
  }

  GenericCFGBuilder builder;
  builder.build(function);
  const auto &regions = builder.getRegions();
  const auto &blocks = builder.getBlocks();

  os << "{\n";
  os << "  \"functionName\": \"" << genericEscapeJSON(functionName) << "\",\n";
  os << "  \"schema\": \"mlir_region_cfg_v1\",\n";
  os << "  \"numRegions\": " << regions.size() << ",\n";
  os << "  \"numBlocks\": " << blocks.size() << ",\n";
  os << "  \"regions\": [\n";

  for (size_t i = 0; i < regions.size(); ++i) {
    const GenericCFGRegion &region = regions[i];
    if (i)
      os << ",\n";
    os << "    {\"id\": " << region.id
       << ", \"ownerOp\": \"" << genericEscapeJSON(region.ownerOp)
       << "\", \"parentRegion\": ";
    if (region.parentRegion < 0)
      os << "null";
    else
      os << region.parentRegion;
    os << ", \"depth\": " << region.depth << ", \"blocks\": [";
    for (size_t j = 0; j < region.blockIds.size(); ++j) {
      if (j)
        os << ", ";
      os << region.blockIds[j];
    }
    os << "]}";
  }

  os << "\n  ],\n";
  os << "  \"blocks\": [\n";
  for (size_t i = 0; i < blocks.size(); ++i) {
    const GenericCFGBlock &block = blocks[i];
    if (i)
      os << ",\n";
    os << "    {\"id\": " << block.id
       << ", \"name\": \"BB" << block.id
       << "\", \"regionId\": " << block.regionId
       << ", \"blockIndex\": " << block.blockIndex
       << ", \"argumentCount\": " << block.argumentCount
       << ", \"operations\": [";
    for (size_t j = 0; j < block.operations.size(); ++j) {
      if (j)
        os << ", ";
      os << "\"" << genericEscapeJSON(block.operations[j]) << "\"";
    }
    os << "], \"predecessors\": [";
    for (size_t j = 0; j < block.predecessors.size(); ++j) {
      if (j)
        os << ", ";
      os << block.predecessors[j];
    }
    os << "], \"successors\": [";
    for (size_t j = 0; j < block.successors.size(); ++j) {
      if (j)
        os << ", ";
      os << block.successors[j];
    }
    os << "]}";
  }

  os << "\n  ],\n";
  os << "  \"controlFlowSemantics\": "
        "\"V1 records explicit block successors and nested Region ownership; "
        "structured RegionBranch transitions require a later interface-aware pass.\"\n";
  os << "}\n";
  os.flush();
  return !os.has_error();
}

// ===== COSTMODEL_V2_GENERIC_GRAPH_EXTRACTOR_V1 END =====
'''

NEW_MAIN_SECTION = r'''
  // 4. Extract graphs.
  // Native TTIR keeps the original TritonToGraph + AliasAnalysis + MemorySSA path.
  module->walk([&](mlir::triton::FuncOp func) {
    ++functionCount;
    std::string functionName = sanitizeName(func.getName().str());

    llvm::errs() << "\n[2/4] Processing native TTIR function: "
                 << functionName << "\n";

    ControlFlowGraphBuilder builder;
    auto cfg = builder.build(func);
    if (!cfg) {
      llvm::errs() << "[error] Native TTIR CFG construction failed\n";
      ++failureCount;
      return;
    }

    AliasAnalysis aliasAnalysis;
    aliasAnalysis.analyzePointerAliases(*cfg);
    DataFlowGraph dfg(*cfg, aliasAnalysis);
    dfg.build();

    llvm::SmallString<256> cfgPath(OutputDir);
    llvm::SmallString<256> dfgPath(OutputDir);
    llvm::sys::path::append(cfgPath, functionName + "_cfg.json");
    llvm::sys::path::append(dfgPath, functionName + "_dataflow.json");

    bool cfgOK = writeCFG(*cfg, cfgPath);
    bool dfgOK = writeDFG(dfg, dfgPath);
    if (cfgOK && dfgOK) {
      ++successCount;
      llvm::errs() << "  Native CFG/DFG + MemorySSA: success\n";
      llvm::errs() << "  CFG: " << cfgPath << "\n";
      llvm::errs() << "  DFG: " << dfgPath << "\n";
    } else {
      ++failureCount;
    }
  });

  // Generic func.func path for TTAdapter and HIVM.
  // External declarations with empty bodies are intentionally skipped.
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "func.func")
      return;

    bool hasBody = false;
    for (Region &region : op->getRegions()) {
      if (!region.empty()) {
        hasBody = true;
        break;
      }
    }
    if (!hasBody) {
      llvm::errs() << "[skip] External func.func declaration\n";
      return;
    }

    ++functionCount;
    auto symbolName = op->getAttrOfType<StringAttr>("sym_name");
    std::string rawName =
        symbolName ? symbolName.getValue().str() : "unnamed_func";
    std::string functionName = sanitizeName(rawName);

    llvm::errs() << "\n[2/4] Processing generic MLIR function: "
                 << functionName << "\n";

    llvm::SmallString<256> cfgPath(OutputDir);
    llvm::SmallString<256> dfgPath(OutputDir);
    llvm::sys::path::append(cfgPath, functionName + "_cfg.json");
    llvm::sys::path::append(dfgPath, functionName + "_dataflow.json");

    bool cfgOK = writeGenericCFG(op, rawName, cfgPath);
    bool dfgOK = writeGenericDFG(op, rawName, dfgPath);
    if (cfgOK && dfgOK) {
      ++successCount;
      llvm::errs() << "  Generic Region CFG: success\n";
      llvm::errs() << "  Generic SSA DFG: success\n";
      llvm::errs() << "  MemorySSA: not computed for generic IR V1\n";
      llvm::errs() << "  CFG: " << cfgPath << "\n";
      llvm::errs() << "  DFG: " << dfgPath << "\n";
    } else {
      ++failureCount;
    }
  });

'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/Agent/costModel_v2")
    parser.add_argument("--main-cpp", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    main_cpp = (
        Path(args.main_cpp).resolve()
        if args.main_cpp
        else root / "cpp/ttir_graph_extractor/src/main.cpp"
    )

    if not main_cpp.is_file():
        print(f"[ERROR] main.cpp not found: {main_cpp}", file=sys.stderr)
        return 2

    source = main_cpp.read_text(encoding="utf-8")
    if "COSTMODEL_V2_GENERIC_GRAPH_EXTRACTOR_V1 BEGIN" in source:
        print("[ERROR] Generic Graph Extractor V1 marker already exists.")
        print("No changes made; inspect main.cpp first.")
        return 3

    include_anchor = '#include "mlir/Parser/Parser.h"'
    section_start = "  // 4. Extract CFG and DFG for each Triton function."
    section_end = "  // Final summary."

    if source.count(include_anchor) != 1:
        print("[ERROR] Parser include anchor not found uniquely.", file=sys.stderr)
        return 4
    if source.count(section_start) != 1 or source.count(section_end) != 1:
        print("[ERROR] main() replacement anchors not found uniquely.", file=sys.stderr)
        print(f"start={source.count(section_start)}, end={source.count(section_end)}")
        return 5

    extra_includes = (
        '#include "mlir/IR/Block.h"\n'
        '#include "mlir/IR/Operation.h"\n'
        '#include "mlir/IR/Region.h"\n'
        '#include "llvm/ADT/DenseMap.h"\n'
        '#include "llvm/ADT/SmallVector.h"\n'
        '#include <utility>'
    )
    patched = source.replace(
        include_anchor, include_anchor + "\n" + extra_includes, 1
    )

    main_anchor = "int main(int argc, char **argv) {"
    if patched.count(main_anchor) != 1:
        print("[ERROR] main() anchor not found uniquely.", file=sys.stderr)
        return 6
    patched = patched.replace(main_anchor, HELPERS + "\n\n" + main_anchor, 1)

    start = patched.index(section_start)
    end = patched.index(section_end, start)
    patched = patched[:start] + NEW_MAIN_SECTION + patched[end:]

    if args.dry_run:
        print("[DRY RUN] Patch anchors validated.")
        print(f"Target: {main_cpp}")
        print("Would add generic func.func SSA DFG and Region-aware CFG.")
        print("Would preserve native tt.func CFG/DFG/MemorySSA.")
        return 0

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = main_cpp.with_name(main_cpp.name + f".bak_generic_v1_{stamp}")
    shutil.copy2(main_cpp, backup)

    tmp = main_cpp.with_suffix(main_cpp.suffix + ".tmp")
    tmp.write_text(patched, encoding="utf-8")
    tmp.replace(main_cpp)

    print("[SUCCESS] main.cpp patched.")
    print(f"Main:   {main_cpp}")
    print(f"Backup: {backup}")
    print("Added:  func.func discovery, generic SSA DFG, Region-aware CFG")
    print("Kept:   original tt.func native CFG/DFG/MemorySSA path")
    print("Note:   HIVM/DLTI parser compatibility remains a separate issue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
