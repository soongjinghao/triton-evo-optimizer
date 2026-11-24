#!/usr/bin/env bash

ROOT=/workspace/Agent/costModel_v2

BIN=/workspace/triton-ascend-src/build-graph-llvm22/third_party/ascend/costmodel_v2_ttir_graph_extractor

export LD_LIBRARY_PATH=/workspace/llvm-install-f6ded0b-aarch64/lib:${LD_LIBRARY_PATH:-}

TTIR=$ROOT/ir/ttir/matmul_kernel_simplified/matmul_kernel_simplified__historical_p50__cbf676b7956f__k00.ttir.mlir

TTADAPTER=$ROOT/ir/ttadapter/matmul_kernel_simplified/matmul_kernel_simplified__historical_p50__cbf676b7956f__k00.ttadapter.mlir

HIVM=$ROOT/ir/save_temps/pass_20261122_134945/module.hivm.opt.mlir

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$ROOT/logs/three_ir_parse_v2_$STAMP

mkdir -p "$OUT"

echo "========== INPUT CHECK =========="

for f in "$TTIR" "$TTADAPTER" "$HIVM"; do
    if [ ! -f "$f" ]; then
        echo "[MISSING] $f"
        exit 1
    fi
    ls -lh "$f"
done

echo
echo "========== PARSING TEST =========="

run_test() {
    NAME=$1
    INPUT=$2

    echo
    echo "---------- $NAME ----------"

    if "$BIN" "$INPUT" \
        --parse-only \
        --output-dir "$OUT/$NAME" \
        > "$OUT/$NAME.log" 2>&1; then

        echo "[PASS] $NAME parsing"
    else
        echo "[FAIL] $NAME parsing"
        tail -50 "$OUT/$NAME.log"
    fi
}

run_test TTIR "$TTIR"
run_test TTAdapter "$TTADAPTER"
run_test HIVM "$HIVM"

echo
echo "========== FINAL SUMMARY =========="

for name in TTIR TTAdapter HIVM; do
    if grep -q '\[PASS\] MLIR parsing completed' "$OUT/$name.log"; then
        echo "[PASS] $name"
    else
        echo "[FAIL] $name"
    fi
done

echo
echo "Logs: $OUT"

