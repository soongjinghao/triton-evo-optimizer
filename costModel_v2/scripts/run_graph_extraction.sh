#!/bin/bash
set -Eeuo pipefail

SRC=/workspace/triton-ascend-src
BUILD=$SRC/build-graph-llvm22
ROOT=/workspace/Agent/costModel_v2

EXE=$BUILD/third_party/ascend/costmodel_v2_ttir_graph_extractor

mkdir -p "$ROOT/logs"
mkdir -p "$ROOT/graph/vector_add"

echo "========== 1. CMake Configure =========="

if ! cmake -S "$SRC" -B "$BUILD" \
    > "$ROOT/logs/graph_configure_v4.log" 2>&1; then
    tail -100 "$ROOT/logs/graph_configure_v4.log"
    exit 1
fi

echo "[SUCCESS] CMake configure"

echo
echo "========== 2. Check Transform Objects =========="

ninja -C "$BUILD" -t targets all 2>/dev/null |
grep -E 'TritonGPUTransforms.dir/Utility.cpp.o|TritonNvidiaGPUTransforms.dir/(TMAUtilities|Utility).cpp.o' \
|| true

echo
echo "========== 3. Build Extractor =========="

if ! cmake --build "$BUILD" \
    --target costmodel_v2_ttir_graph_extractor \
    --parallel 2 \
    > "$ROOT/logs/graph_build_v4.log" 2>&1; then

    echo "[ERROR] Build failed"

    grep -E 'undefined reference|error:|FAILED:' \
        "$ROOT/logs/graph_build_v4.log" | tail -60 || true

    tail -80 "$ROOT/logs/graph_build_v4.log"
    exit 1
fi

test -x "$EXE"

echo "[SUCCESS] Extractor built"
echo "$EXE"

echo
echo "========== 4. Extract CFG / DFG =========="

for bs in 128 256 512; do

    INPUT="$ROOT/ir/ttir/vector_add/bs${bs}.ttir.mlir"
    OUTPUT="$ROOT/graph/vector_add/bs${bs}"
    LOG="$ROOT/logs/graph_extract_bs${bs}.log"

    mkdir -p "$OUTPUT"

    echo "Processing BLOCK_SIZE=$bs"

    if ! "$EXE" "$INPUT" \
        --output-dir "$OUTPUT" \
        > "$LOG" 2>&1; then

        echo "[ERROR] Extraction failed: bs${bs}"
        tail -100 "$LOG"
        exit 1
    fi

    echo "[SUCCESS] bs${bs}"

done

echo
echo "========== 5. Validate JSON =========="

python3 - "$ROOT/graph/vector_add" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

for bs in (128, 256, 512):
    folder = root / f"bs{bs}"
    files = sorted(folder.glob("*.json"))

    print(f"\nBLOCK_SIZE={bs}")

    if len(files) < 2:
        raise RuntimeError(f"Missing CFG/DFG JSON: {folder}")

    for path in files:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        print(f"  {path.name}: valid JSON")

print("\nALL THREE CANDIDATES PASSED")
PY

echo
echo "========================================"
echo "GRAPH EXTRACTION COMPLETED"
echo "========================================"

find "$ROOT/graph/vector_add" \
    -type f -name '*.json' \
    -printf '%p (%s bytes)\n' | sort

