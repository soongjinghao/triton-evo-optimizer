from pathlib import Path
import shutil

p = Path(
    "third_party/ascend/lib/TritonToGraph/DataflowGraph.cpp"
)

s = p.read_text()

marker = "COSTMODEL_V2_LOOP_CARRIED_EDGES_V1"

if marker in s:
    print("[SKIP] Loop edges already patched")
    raise SystemExit(0)

# 1. 备份
bak = Path(str(p) + ".before_loop_edges_v1")
if not bak.exists():
    shutil.copy2(p, bak)
    print("[BACKUP]", bak)

# 2. 添加 SCF 头文件
include = '#include "mlir/Dialect/SCF/IR/SCF.h"'
anchor = '#include "TritonToGraph/DataflowGraph.h"\n'

if include not in s:
    if anchor not in s:
        raise RuntimeError("Cannot find DFG header include")
    s = s.replace(anchor, anchor + include + "\n", 1)

# 3. 循环边构建代码
loop_code = r'''
  // COSTMODEL_V2_LOOP_CARRIED_EDGES_V1
  // Explicitly model SCF loop-carried data dependencies.

  auto findValueNode = [&](Value value, unsigned &nodeIndex) -> bool {
    if (Operation *producer = value.getDefiningOp()) {
      auto it = operationNodeIndex.find(producer);
      if (it != operationNodeIndex.end()) {
        nodeIndex = it->second;
        return true;
      }
    }

    auto it = blockArgumentNodeIndex.find(value);
    if (it != blockArgumentNodeIndex.end()) {
      nodeIndex = it->second;
      return true;
    }

    return false;
  };

  auto addLoopEdge = [&](Value sourceValue,
                         unsigned targetIndex,
                         const char *kind,
                         unsigned loopIndex) {
    unsigned sourceIndex = 0;

    if (!findValueNode(sourceValue, sourceIndex))
      return;

    CostModelDFGEdge edge;
    edge.source = graph.nodes[sourceIndex].id;
    edge.target = graph.nodes[targetIndex].id;
    edge.kind = kind;
    edge.valueType = costModelTypeString(sourceValue.getType());

    // For loop edges, operand_index records the loop-carried slot.
    edge.operandIndex = loopIndex;
    edge.hasResultIndex = false;
    edge.resultIndex = 0;

    if (auto result = llvm::dyn_cast<OpResult>(sourceValue)) {
      edge.hasResultIndex = true;
      edge.resultIndex = result.getResultNumber();
    }

    graph.edges.push_back(std::move(edge));
  };

  for (Operation *op : operations) {
    auto forOp = llvm::dyn_cast<scf::ForOp>(op);
    if (!forOp)
      continue;

    auto initArgs = forOp.getInitArgs();
    auto yieldedValues = forOp.getYieldedValues();
    auto iterArgs = forOp.getRegionIterArgs();

    // MLIR scf.for requires matching loop-carried arguments.
    if (initArgs.size() != yieldedValues.size() ||
        initArgs.size() != iterArgs.size())
      continue;

    auto loopNodeIt = operationNodeIndex.find(op);
    if (loopNodeIt == operationNodeIndex.end())
      continue;

    for (unsigned i = 0; i < initArgs.size(); ++i) {
      Value initValue = initArgs[i];
      Value iterArg = forOp.getRegionIterArg(i);
      Value yieldedValue = yieldedValues[i];

      auto iterNodeIt = blockArgumentNodeIndex.find(iterArg);
      if (iterNodeIt == blockArgumentNodeIndex.end())
        continue;

      // Initial value -> loop body iter_arg.
      addLoopEdge(
          initValue,
          iterNodeIt->second,
          "loop_init",
          i);

      // Yield value -> next iteration's iter_arg.
      addLoopEdge(
          yieldedValue,
          iterNodeIt->second,
          "loop_backedge",
          i);

      // Yield value -> scf.for result.
      if (i < forOp.getNumResults()) {
        addLoopEdge(
            yieldedValue,
            loopNodeIt->second,
            "loop_result",
            i);
      }
    }
  }
'''

# 4. 在 return graph 之前插入
needle = (
    "  return graph;\n"
    "}\n\n"
    "static void costModelExportOperationGraphJSON"
)

if s.count(needle) != 1:
    raise RuntimeError(
        "Expected unique graph builder insertion point; "
        "source was not modified."
    )

replacement = (
    loop_code +
    "\n  return graph;\n"
    "}\n\n"
    "static void costModelExportOperationGraphJSON"
)

s = s.replace(needle, replacement, 1)

p.write_text(s)

print("[SUCCESS] SCF loop-carried edges added")
print("[SUCCESS] Added loop_init")
print("[SUCCESS] Added loop_backedge")
print("[SUCCESS] Added loop_result")
print("[SUCCESS] Original SSA construction preserved")
