#!/bin/bash
set -Eeuo pipefail

ROOT=/workspace/Agent/costModel_v2
SRC=/workspace/triton-ascend-src
BUILD=$SRC/build-graph-llvm22

ASCEND_CMAKE=$SRC/third_party/ascend/CMakeLists.txt
MAIN=$ROOT/cpp/ttir_graph_extractor/src/main.cpp

LOG_DIR=$ROOT/logs
GRAPH_DIR=$ROOT/graph/vector_add

mkdir -p "$LOG_DIR" "$GRAPH_DIR"

echo "========== 1. Check inputs =========="

test -f "$MAIN"
test -f "$BUILD/CMakeCache.txt"
test -f "$ASCEND_CMAKE"

for bs in 128 256 512; do
    test -s "$ROOT/ir/ttir/vector_add/bs${bs}.ttir.mlir"
done

echo "TTIR inputs: OK"
echo "C++ extractor source: OK"

echo "========== 2. Register CMake target =========="

MARKER="# COSTMODEL_V2_TTIR_GRAPH_EXTRACTOR"

if ! grep -q "$MARKER" "$ASCEND_CMAKE"; then

    cp -p "$ASCEND_CMAKE" \
        "$ASCEND_CMAKE.before_costmodel_v2_graph_extractor"

    cat >> "$ASCEND_CMAKE" <<'CMAKE'

# COSTMODEL_V2_TTIR_GRAPH_EXTRACTOR
set(_costmodel_v2_graph_main
    "/workspace/Agent/costModel_v2/cpp/ttir_graph_extractor/src/main.cpp"
)

if(EXISTS "${_costmodel_v2_graph_main}")
    if(NOT TARGET costmodel_v2_ttir_graph_extractor)

        add_executable(costmodel_v2_ttir_graph_extractor
            "${_costmodel_v2_graph_main}"
        )

        target_include_directories(costmodel_v2_ttir_graph_extractor PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/lib"
            "${CMAKE_CURRENT_SOURCE_DIR}/AscendNPU-IR/bishengir/include"
            "${CMAKE_CURRENT_BINARY_DIR}/include"
            "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_BINARY_DIR}/include"
            "${PROJECT_SOURCE_DIR}/third_party"
            "${PROJECT_BINARY_DIR}/third_party"
            "${PROJECT_BINARY_DIR}/third_party/ascend/include"
            ${LLVM_INCLUDE_DIRS}
            ${MLIR_INCLUDE_DIRS}
        )

        target_compile_definitions(costmodel_v2_ttir_graph_extractor PRIVATE
            TRITON_ASCEND_HAS_INPROC_COSTMODEL=0
            __LLVM_MAJOR_VERSION_22_COMPATIBLE__
        )

        target_compile_features(costmodel_v2_ttir_graph_extractor PRIVATE
            cxx_std_17
        )

        target_link_libraries(costmodel_v2_ttir_graph_extractor PRIVATE
            TritonToGraph
            MLIRParser
            MLIRArithDialect
            MLIRControlFlowDialect
            MLIRFuncDialect
            MLIRSCFDialect
            MLIRMathDialect
            MLIRTensorDialect
            MLIRGPUDialect
            TritonGPUIR
        )

        llvm_update_compile_flags(costmodel_v2_ttir_graph_extractor)

        add_dependencies(costmodel_v2_ttir_graph_extractor TritonToGraph)

    endif()
endif()
CMAKE

    echo "CMake target registered."
else
    echo "CMake target already registered."
fi

echo "========== 3. Configure CMake =========="

if ! cmake -S "$SRC" -B "$BUILD" \
    > "$LOG_DIR/graph_extractor_configure.log" 2>&1; then

    echo "[ERROR] CMake configure failed."
    tail -80 "$LOG_DIR/graph_extractor_configure.log"
    exit 1
fi

echo "CMake configure: SUCCESS"

echo "========== 4. Build extractor =========="

if ! cmake --build "$BUILD" \
    --target costmodel_v2_ttir_graph_extractor \
    --parallel 2 \
    > "$LOG_DIR/graph_extractor_build.log" 2>&1; then

    echo "[ERROR] Extractor build failed."
    tail -120 "$LOG_DIR/graph_extractor_build.log"
    exit 1
fi

EXE="$BUILD/third_party/ascend/costmodel_v2_ttir_graph_extractor"

test -x "$EXE"

echo "Extractor build: SUCCESS"

echo "========== 5. Extract CFG and DFG =========="

for bs in 128 256 512; do

    INPUT="$ROOT/ir/ttir/vector_add/bs${bs}.ttir.mlir"
    OUTPUT="$GRAPH_DIR/bs${bs}"
    LOG="$LOG_DIR/graph_extract_bs${bs}.log"

    mkdir -p "$OUTPUT"

    echo "Extracting BLOCK_SIZE=$bs"

    if ! "$EXE" "$INPUT" \
        --output-dir "$OUTPUT" \
        > "$LOG" 2>&1; then

        echo "[ERROR] Graph extraction failed: bs${bs}"
        tail -100 "$LOG"
        exit 1
    fi

    echo "CFG/DFG extraction: SUCCESS (bs${bs})"
done

echo "========== 6. Check JSON =========="

find "$GRAPH_DIR" -type f -name '*.json' \
    -printf '%p %s bytes\n' | sort

python3 - "$GRAPH_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted(root.rglob("*.json"))

if not files:
    raise SystemExit("ERROR: No graph JSON generated")

for path in files:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n{path.relative_to(root)}")
    print("Type:", type(data).__name__)

    if isinstance(data, dict):
        print("Keys:", list(data.keys())[:20])
    elif isinstance(data, list):
        print("Elements:", len(data))

print("\nJSON validation: SUCCESS")
print("Total JSON files:", len(files))
PY

echo "========================================"
echo "GRAPH EXTRACTION COMPLETED"
echo "Output: $GRAPH_DIR"
echo "========================================"
