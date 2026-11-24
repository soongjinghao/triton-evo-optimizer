from pathlib import Path
import shutil

SRC = Path("/workspace/triton-ascend-src")

MAIN = Path(
    "/workspace/Agent/costModel_v2/cpp/"
    "ttir_graph_extractor/src/main.cpp"
)

CMAKE = SRC / "third_party/ascend/CMakeLists.txt"


def patch(path, anchor, addition, label):
    text = path.read_text()

    if addition.strip() in text:
        print("[SKIP] Already patched:", label)
        return

    if text.count(anchor) != 1:
        raise RuntimeError(
            f"Anchor mismatch: {label}, "
            f"count={text.count(anchor)}"
        )

    backup = Path(str(path) + ".before_bufferization_v1")

    if not backup.exists():
        shutil.copy2(path, backup)
        print("[BACKUP]", backup)

    text = text.replace(anchor, anchor + addition, 1)
    path.write_text(text)

    print("[PATCHED]", label)


# 1. 添加头文件
patch(
    MAIN,
    '#include "mlir/Dialect/Tensor/IR/Tensor.h"\n',
    '#include "mlir/Dialect/Bufferization/IR/Bufferization.h"\n',
    "Bufferization header"
)

# 2. 注册 Dialect
patch(
    MAIN,
    "      mlir::tensor::TensorDialect,\n",
    "      mlir::bufferization::BufferizationDialect,\n",
    "Bufferization dialect registration"
)

# 3. CMake 显式链接
patch(
    CMAKE,
    "            MLIRMemRefDialect\n",
    "            MLIRBufferizationDialect\n",
    "Bufferization library"
)

print("\n[SUCCESS] Bufferization patch completed")
