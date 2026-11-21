#!/usr/bin/env bash
set -Eeuo pipefail

SRC=/workspace/triton-ascend-src
BUILD=$SRC/build-graph-llvm22
ROOT=/workspace/Agent/costModel_v2
PATCH=$ROOT/scripts/repair_dfg_graph_export.py
EXE=$BUILD/third_party/ascend/costmodel_v2_ttir_graph_extractor
INPUT=$ROOT/ir/ttir/vector_add/bs128.ttir.mlir
OUT=$ROOT/graph/vector_add/bs128
LOG=$ROOT/logs/vector_add_bs128_dfg_v1.log
BUILD_LOG=$ROOT/logs/dfg_export_rebuild_v1.log

mkdir -p "$ROOT/logs" "$OUT"

if [[ ! -f "$PATCH" ]]; then
  echo "[ERROR] Missing patch script: $PATCH"
  echo "Upload repair_dfg_graph_export.py to $ROOT/scripts/ first."
  exit 1
fi

if [[ ! -f "$INPUT" ]]; then
  echo "[ERROR] Missing TTIR input: $INPUT"
  exit 1
fi

echo "========== 1. Patch DataflowGraph.cpp =========="
python3 "$PATCH"

echo
echo "========== 2. Check diff =========="
git -C "$SRC" diff --check -- third_party/ascend/lib/TritonToGraph/DataflowGraph.cpp

echo
echo "========== 3. Rebuild graph extractor =========="
if ! cmake --build "$BUILD" \
    --target costmodel_v2_ttir_graph_extractor \
    --parallel 2 > "$BUILD_LOG" 2>&1; then
  echo "[FAILED] Rebuild failed. Last 120 lines:"
  tail -120 "$BUILD_LOG"
  exit 1
fi
echo "[SUCCESS] Rebuild completed"
tail -15 "$BUILD_LOG"

if [[ ! -x "$EXE" ]]; then
  echo "[ERROR] Extractor executable not found: $EXE"
  exit 1
fi

export LD_LIBRARY_PATH=/workspace/llvm-install-f6ded0b-aarch64/lib:${LD_LIBRARY_PATH:-}

for f in "$OUT/vector_add_kernel_cfg.json" "$OUT/vector_add_kernel_dataflow.json"; do
  if [[ -f "$f" ]]; then
    cp -an "$f" "$f.before_dfg_v1"
  fi
done

echo
echo "========== 4. Run TTIR graph extraction =========="
if ! "$EXE" "$INPUT" --output-dir "$OUT" > "$LOG" 2>&1; then
  echo "[FAILED] Extraction failed. Last 120 lines:"
  tail -120 "$LOG"
  exit 1
fi
tail -35 "$LOG"

echo
echo "========== 5. Validate operation DFG JSON =========="
python3 - "$OUT/vector_add_kernel_dataflow.json" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
graph = data.get("graph")
assert isinstance(graph, dict), "missing top-level graph object"
nodes = graph.get("nodes", [])
edges = graph.get("edges", [])
assert graph.get("node_count") == len(nodes), "node_count mismatch"
assert graph.get("edge_count") == len(edges), "edge_count mismatch"
assert nodes, "graph has no nodes"

node_by_id = {node["id"]: node for node in nodes}
ops = Counter(node.get("op") for node in nodes if node.get("kind") == "operation")
edge_pairs = {
    (node_by_id[e["source"]].get("op"), node_by_id[e["target"]].get("op"))
    for e in edges
    if e.get("source") in node_by_id and e.get("target") in node_by_id
}

required = [("tt.load", "arith.addf"), ("arith.addf", "tt.store")]
missing = [pair for pair in required if pair not in edge_pairs]
assert not missing, f"missing expected SSA dependencies: {missing}"

flow = data.get("dataFlow", {})
assert "memoryDefinitions" in flow, "missing memoryDefinitions"
assert "memoryUses" in flow, "missing memoryUses"
assert "Phis" in flow, "missing Phis"

print("[PASS] JSON syntax and graph counts")
print(f"Function       : {data.get('function')}")
print(f"Graph schema   : {graph.get('schema')}")
print(f"Nodes / edges  : {len(nodes)} / {len(edges)}")
print(f"Block arguments: {sum(n.get('kind') == 'block_argument' for n in nodes)}")
print(f"Memory defs    : {len(flow['memoryDefinitions'])}")
print(f"Memory uses    : {len(flow['memoryUses'])}")
print("Key operation counts:", {k: ops[k] for k in ("tt.load", "arith.addf", "tt.store")})
print("[PASS] tt.load -> arith.addf -> tt.store dependencies found")
PY

echo
echo "[SUCCESS] DFG export repair and Vector Add validation completed."
echo "JSON: $OUT/vector_add_kernel_dataflow.json"
echo "LOG : $LOG"
