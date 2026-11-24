from pathlib import Path
import shutil

ROOT = Path("/workspace/triton-ascend-src")

MAIN = ROOT / "cpp/ttir_graph_extractor/src/main.cpp"
CMAKE = ROOT / "third_party/ascend/CMakeLists.txt"

# 实际 main.cpp 位于 Agent 项目中
MAIN = Path("/workspace/Agent/costModel_v2/cpp/ttir_graph_extractor/src/main.cpp")

def backup(p):
    bak = Path(str(p) + ".before_graph_dialects_v1")
    if not bak.exists():
        shutil.copy2(p, bak)
        print("[BACKUP]", bak)

def replace_once(text, old, new, name):
    if old in text:
        return text.replace(old, new, 1)
    if new in text:
        print("[SKIP]", name)
        return text
    raise RuntimeError("Cannot locate patch anchor: " + name)

# ==========================================
# 1. 修改 main.cpp
# ==========================================

backup(MAIN)
s = MAIN.read_text()

# 1.1 增加 Dialect 头文件
if "bishengir/Dialect/HIVM/IR/HIVM.h" not in s:
    anchor = '#include "mlir/Dialect/Arith/IR/Arith.h"\n'
    extra = '''#include "mlir/Dialect/Affine/IR/Affine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
'''
    s = replace_once(s, anchor, anchor + extra,
                     "Dialect headers")

# 1.2 注册 Dialect
old = '''      mlir::tensor::TensorDialect,
      mlir::gpu::GPUDialect>();'''

new = '''      mlir::tensor::TensorDialect,
      mlir::gpu::GPUDialect,
      mlir::func::FuncDialect,
      mlir::affine::AffineDialect,
      mlir::linalg::LinalgDialect,
      mlir::memref::MemRefDialect,
      mlir::vector::VectorDialect,
      mlir::annotation::AnnotationDialect,
      mlir::hacc::HACCDialect,
      mlir::hfusion::HFusionDialect,
      mlir::hivm::HIVMDialect>();'''

s = replace_once(s, old, new, "Dialect registration")

# 1.3 增加 ParseOnly 测试选项
anchor = '''static llvm::cl::opt<std::string> OutputDir(
    "output-dir",
    llvm::cl::desc("Output directory"),
    llvm::cl::init("graph_output"));'''

extra = '''

static llvm::cl::opt<bool> ParseOnly(
    "parse-only",
    llvm::cl::desc("Only parse MLIR without graph extraction"),
    llvm::cl::init(false));
'''

if "ParseOnly" not in s:
    s = replace_once(s, anchor, anchor + extra,
                     "ParseOnly option")

# 1.4 解析成功后支持单独退出
anchor = '''  if (!module) {
    llvm::errs() << "[error] Failed to parse TTIR\\\\n";
    return 1;
  }'''

extra = '''

  if (ParseOnly) {
    llvm::errs() << "[PASS] MLIR parsing completed\\\\n";
    return 0;
  }
'''

if "if (ParseOnly)" not in s:
    s = replace_once(s, anchor, anchor + extra,
                     "ParseOnly branch")

MAIN.write_text(s)
print("[SUCCESS] main.cpp patched")

# ==========================================
# 2. 修改 CMakeLists.txt
# ==========================================

backup(CMAKE)
c = CMAKE.read_text()

start = c.find(
    "target_link_libraries(costmodel_v2_ttir_graph_extractor PRIVATE"
)

if start < 0:
    raise RuntimeError("Cannot find extractor target_link_libraries")

end = c.find(")", start)

if end < 0:
    raise RuntimeError("Cannot find target_link_libraries ending")

block = c[start:end + 1]

if "BiShengIRHIVMDialect" not in block:

    anchor = "MLIRGPUDialect"

    if anchor not in block:
        raise RuntimeError("Cannot find MLIRGPUDialect in link block")

    extra = '''
            MLIRAffineDialect
            MLIRLinalgDialect
            MLIRMemRefDialect
            MLIRVectorDialect
            BiShengIRAnnotationDialect
            BiShengIRHACCDialect
            BiShengIRHFusionDialect
            BiShengIRHIVMDialect'''

    block = block.replace(anchor, anchor + extra, 1)

    c = c[:start] + block + c[end + 1:]
    CMAKE.write_text(c)

print("[SUCCESS] CMakeLists.txt patched")
print("[SUCCESS] Dialect registration patch completed")
