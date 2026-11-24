#!/usr/bin/env bash
set -uo pipefail

ROOT=/workspace/Agent/costModel_v2
EXE=/workspace/triton-ascend-src/build-graph-llvm22/third_party/ascend/costmodel_v2_ttir_graph_extractor

# 三个阶段使用同一个 Matmul 候选
TTIR=$(find "$ROOT/ir/ttir" -type f \
    -name '*.ttir.mlir' | grep 'historical_p50.*cbf676b7956f' | head -1 || true)

TTADAPTER=$(find "$ROOT/ir/ttadapter" -type f \
    -name '*.ttadapter.mlir' | grep 'historical_p50.*cbf676b7956f' | head -1 || true)

# 之前已经成功保存的优化后 HIVM IR
HIVM="$ROOT/ir/save_temps/matmul_20261122_133319/module.hivm.opt.mlir"

# 如果默认文件不存在，寻找其他已保存的 HIVM IR
if [ ! -f "$HIVM" ]; then
    HIVM=$(find "$ROOT/ir/save_temps" -type f \
        -name 'module.hivm.opt.mlir' | head -1 || true)
fi

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="$ROOT/graph/three_ir_test_$STAMP"
LOG_ROOT="$ROOT/logs/three_ir_test_$STAMP"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

export LD_LIBRARY_PATH=/workspace/llvm-install-f6ded0b-aarch64/lib:${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "Three IR Graph Compatibility Test"
echo "=========================================="

echo "Extractor: $EXE"
echo "TTIR:      ${TTIR:-NOT FOUND}"
echo "TTAdapter: ${TTADAPTER:-NOT FOUND}"
echo "HIVM:      ${HIVM:-NOT FOUND}"
echo

if [ ! -x "$EXE" ]; then
    echo "[FATAL] Extractor not found"
    exit 1
fi

# 检查并执行单个 IR
run_test() {
    local stage="$1"
    local input="$2"
    local output="$OUT_ROOT/$stage"
    local log="$LOG_ROOT/$stage.log"

    echo
    echo "========== $stage =========="

    if [ -z "$input" ] || [ ! -f "$input" ]; then
        echo "[SKIP] Input file missing"
        return
    fi

    echo "Input: $input"
    mkdir -p "$output"

    if timeout 180 "$EXE" "$input" \
        --output-dir "$output" > "$log" 2>&1; then
        echo "[EXECUTOR] Exit code: 0"
    else
        echo "[EXECUTOR] Exit code: $?"
    fi

    # 验证 JSON 是否真正生成、是否可以解析
    python3 - "$output" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

cfg_files = list(root.glob("*_cfg.json"))
dfg_files = list(root.glob("*_dataflow.json"))

print("CFG files:", len(cfg_files))
print("DFG files:", len(dfg_files))

valid = True

for path in cfg_files + dfg_files:
    try:
        data = json.loads(path.read_text())
        print("Valid JSON:", path.name)

        if path.name.endswith("_cfg.json"):
            print("  Blocks:", len(data.get("blocks", [])))
            print("  Edges:", len(data.get("edges", [])))

        elif path.name.endswith("_dataflow.json"):
            print("  Nodes:", len(data.get("nodes", [])))
            print("  Edges:", len(data.get("edges", [])))

    except Exception as e:
        print("Invalid JSON:", path.name, e)
        valid = False

if not cfg_files or not dfg_files:
    valid = False

if valid:
    print("[PASS] CFG/DFG JSON generated and parsed")
else:
    print("[FAIL] Complete graph extraction not achieved")
PY

    echo
    echo "Important log messages:"
    grep -nEi \
        'Parsing|Processing function|CFG construction|DFG construction|Export:|error:|unregistered dialect|Failed to parse|No function' \
        "$log" | tail -25 || true

    echo "Full log: $log"
}

run_test TTIR "$TTIR"
run_test TTAdapter "$TTADAPTER"
run_test HIVM "$HIVM"

echo
echo "=========================================="
echo "TEST COMPLETED"
echo "=========================================="
echo "Outputs: $OUT_ROOT"
echo "Logs:    $LOG_ROOT"
